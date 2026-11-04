#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
"""架构签名探针：msprof 原始 CSV → 结构化的架构签名。

这是论文中「NCU pass」在昇腾上的对应物。论文 §3.2 说一次 NCU pass 给出
launch structure、occupancy、pipeline utilization 和 stall decomposition，
arch agent 据此推断 kernel 数、角色分工、流水深度和限制资源。

msprof 能给出这四类里的三类（occupancy 没有直接对应，见下），但它是以
七份扁平 CSV 的形式给的，几百个列。直接把 CSV 丢给 arch agent 有两个问题：

  读不动   几百列里只有十几列和架构决策相关，其余是噪声
  读错了   最危险的一类错误是「缺失的数据」和「真的是 0」长得一样。
           ResourceConflictRatio 没采到时，停顿占比全是空；若按 0 处理，
           读出来就是「完全没有停顿」—— 一个本该引发追查的信号被反转成
           了「这里很健康」。论文把 agent 自评的探针单列为一类 bug 来源，
           这就是其中一种。

所以本探针做三件事：

  一  把七份 CSV 归并成一份结构化签名，每个值带上它的来源通道
      （milestone 的 trigger.channel 可以直接引用这个字符串）
  二  先查测量可信度。降频、ICache 颠簸、指标缺失，任一成立就**拒绝**
      输出架构假设 —— 不可信的数据推出的架构是错的，而且错得很有说服力
  三  把「实测」「推导」「假设」三层明确分开，不让推断冒充观测

输出里还有一项 msprof 没有直接给、但可以从 Task Duration 与各核执行时间
推导出来的量：**暴露时间**。它是论文「timeline probe」在昇腾上最接近的
替代 —— 见 exposure 段的说明。

用法:
    # msprof 原始目录（含 PROF_<Metric>/ 子目录）
    python3 signature_probe.py --prof-dir ./prof_out --op-name AddLayerNorm

    # msprof_perf_summary.py 归档后的目录（含 op_summary_<Metric>.csv）
    python3 signature_probe.py --archive-dir ./round_1 --op-name AddLayerNorm

    # 给出理论最小搬运量后，额外算出数据复用倍数
    python3 signature_probe.py --archive-dir ./round_1 --op-name AddLayerNorm \\
        --min-gm-bytes 8388608

退出码: 0 = 签名可用; 1 = 测量不可信或数据不足; 2 = 用法或 IO 错误
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 与 ops-profiling 对齐的常量
# ---------------------------------------------------------------------------

#: 与 ops/ops-profiling/scripts/msprof_perf_summary.py 的 METRICS 保持一致。
#: 改这里之前先确认那边也改了，否则归档目录里会找不到文件。
METRICS = [
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "L2Cache",
    "ResourceConflictRatio",
]

#: 判定阈值全部引自 ops/ops-profiling/references/csv_fields_reference.md。
#: 不要在这里自行发明阈值 —— 那会让门禁建立在没有出处的数字上。
THRESHOLDS = {
    "vec_bound_ratio": 50.0,        # aiv_vec_ratio > 50% = VEC Bound
    "cube_normal_ratio": 40.0,      # aic_cube_ratio > 40% 正常（MatMul）
    "scalar_bound_ratio": 30.0,     # scalar_ratio > 30% = SCALAR Bound
    "mte2_bound_ratio": 50.0,       # mte2_ratio > 50% = MTE2 Bound
    "fixpipe_misalign_ratio": 15.0,  # fixpipe_ratio > 15% = 地址未对齐
    "icache_miss_warn": 15.0,       # icache_miss_rate > 15% 需关注
    "l2_hit_good": 80.0,            # total_hit_rate > 80% 良好
    "l2_hit_poor": 50.0,            # < 50% 需优化
    "bw_usage_good": 60.0,          # 带宽利用率 > 60% 良好
    "conflict_total_good": 5.0,     # vec_total_cflt_ratio < 5% 良好
    "conflict_total_severe": 15.0,  # > 15% 严重
    "conflict_sub_good": 3.0,       # bank / bankgroup / mte 冲突 < 3%
}

#: 暴露时间超过这个比例就判为串行延迟受限，对应论文中需要额外上 timeline
#: probe 的那类 kernel。这个阈值没有文档出处，是本工具的工程判断，
#: 因此输出里标注为 derived 而非 measured。
EXPOSURE_SERIAL_PCT = 20.0

MISSING = None  # 显式区分「没采到」与「采到了且为 0」


class ProbeError(Exception):
    """探针无法产出可用签名。"""


# ---------------------------------------------------------------------------
# CSV 读取
# ---------------------------------------------------------------------------

#: 单元格解析不出数时的第三种情形。
#:
#: safe_float 原来只有两态：有数，或 None。而 None 被当成「没采到」—— 这对
#: "N/A"（msprof 明确表示该指标不适用）是对的，对 `"12.3abc"`、
#: `"<overflow>"`、半截写坏的行就不对了。
#:
#: 两者该做的事完全不同：
#:   没采到   正常情形。换个采集组、或这个算子本来就没有这个单元
#:   解析不动 **异常情形**。那一列的量纲或格式和我们以为的不一样，
#:            而同一份 CSV 里别的列照样被读走了 —— 于是签名里混着
#:            一个我们读错了的字段，而它看起来只是「没采到」
#:
#: 把后者说成前者，等于把一次「工具与数据不匹配」静默降级成「这次没采」。
#: 而探针的全部职责就是不让这种降级发生（SKILL.md 那条贯穿原则）。
UNPARSEABLE = "unparseable"


def safe_float(value: Any) -> Optional[float]:
    """把 CSV 单元格转成 float。两态版本，给不关心解析失败的调用点。

    转不出来返回 None 而不是 0.0。这一条是本探针最重要的设计：
    msprof 用 "N/A" 表示该指标不适用（比如非 Mix 算子的 Mix Block Dim），
    若折成 0 会和真实的 0 混在一起，后面所有判定都跟着错。

    要区分「没采到」和「解析不动」时用 parse_cell。
    """
    value_or_none, _ = parse_cell(value)
    return value_or_none


def parse_cell(value: Any) -> Tuple[Optional[float], Optional[str]]:
    """三态解析：(数值, 问题)。

      (3.14, None)          正常
      (None, None)          没采到 —— 空、"N/A"、"-" 这类明确的不适用标记
      (None, UNPARSEABLE)   有内容但解析不动 —— 这是异常，调用方要响亮报出来
    """
    if value is None:
        return None, None
    text = str(value).strip().rstrip("\t ")
    if text in ("", "N/A", "NA", "-", "null", "None"):
        return None, None
    try:
        return float(text), None
    except (ValueError, TypeError):
        return None, UNPARSEABLE


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return list(csv.DictReader(handle))


def row_duration(row: Dict[str, str]) -> float:
    for key in ("Task Duration(us)", "task_duration", "Task Duration"):
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return 0.0


def _row_task_id(row: Dict[str, str]) -> Optional[int]:
    for key in ("Task ID", "task_id", "Task Id"):
        value = safe_float(row.get(key))
        if value is not None:
            return int(value)
    return None


def _is_self_contradictory(row: Dict[str, str]) -> bool:
    """这一行的核执行时间超过了任务总时长 —— 不可能，说明它不是稳态读数。

    一次采集里算子的头几次执行是冷启动：指令缓存冷、权重未就位、
    Task Duration 的计时窗口和各核计数器的窗口也对不齐，于是会出现
    aiv_time > Task Duration 这种自相矛盾的行。

    这类行必须排除，不只是因为它偏大。它偏大的幅度很可观（实测冷启动
    408us vs 稳态 233us，高 75%），而这个数会成为整个 campaign 的参考基线：
    后面每一轮候选都在和一个虚高 75% 的数做比，于是任何平庸实现都显得在
    接近参考，优化会提前停下。
    """
    duration = row_duration(row)
    if duration <= 0:
        return False
    for key in ("aiv_time(us)", "aiv_time", "aic_time(us)", "aic_time"):
        value = safe_float(row.get(key))
        if value is not None and value > duration:
            return True
    return False


def select_row(rows: List[Dict[str, str]], op_name: Optional[str],
               steady_from: Optional[int] = None) -> Tuple[Optional[Dict[str, str]],
                                                           Dict[str, Any]]:
    """挑出目标算子的**稳态**那一行，并说明是怎么挑的。

    一次采集里同一个算子通常有十来行（每次执行一行）。早先的做法是取第一行，
    而第一行恰恰是冷启动 —— 最不该用的那一行。

    现在的做法分三步：
      1. 按算子名筛出候选行（精确匹配优先，退化到子串以便传简称）
      2. 丢掉自相矛盾的行（核执行时间 > 任务时长），即冷启动
      3. 剩下的取**中位数附近**那一行

    第 3 步取中位数而不是最小值：最小值是运气最好的一次，拿它当参考基线
    会让判据偏严；中位数代表这个参考稳定能跑出来的水平，这才是要对标的对象。

    挑行过程一并返回，写进签名的 row_selection 字段 —— 「用了哪一行」直接
    决定全部后续结论，不该是个看不见的内部决定。
    """
    info: Dict[str, Any] = {
        "candidates": 0, "dropped_cold_start": 0, "dropped_by_steady_from": 0,
        "picked_task_id": None, "picked_duration_us": None,
        "duration_min_us": None, "duration_median_us": None,
        "duration_max_us": None, "steady_from": steady_from,
        # 能不能按耗时排序。排不出来时「丢冷启动 + 取中位数」两道防线都失效，
        # 这件事必须说出来，否则可信度检查会报一个并不存在的稳态选择。
        "duration_ranked": False,
        # 供跨 metric 一致性检查用
        "picked_op_name": None, "picked_block_dim": None,
        "note": "",
    }
    if not rows:
        info["note"] = "CSV 里没有数据行"
        return None, info

    if op_name:
        pool = [r for r in rows
                if (r.get("Op Name") or r.get("op_name") or "").strip() == op_name]
        if not pool:
            pool = [r for r in rows
                    if op_name.lower() in (r.get("Op Name")
                                           or r.get("op_name") or "").strip().lower()]
            if pool:
                info["note"] = f"按子串匹配到 {op_name}"
        if not pool:
            info["note"] = f"没有匹配 {op_name} 的行"
            return None, info
    else:
        # 不给算子名时先认出被测算子：一次采集里可能混入框架算子，取耗时
        # 最长的那个算子。认出来之后再在**它自己**的多次执行里挑稳态行 ——
        # 跨算子取中位数没有意义，那是把 TinyOp 和 BigOp 混在一起算。
        longest = max(rows, key=row_duration)
        target = (longest.get("Op Name") or longest.get("op_name") or "").strip()
        if target:
            pool = [r for r in rows
                    if (r.get("Op Name") or r.get("op_name") or "").strip() == target]
            info["note"] = f"未指定算子名，按最长耗时认定为 {target}"
        else:
            pool = list(rows)
            info["note"] = "未指定算子名，且 CSV 里没有算子名列"

    info["candidates"] = len(pool)

    if steady_from is not None:
        def _at_or_after(row: Dict[str, str]) -> bool:
            task_id = _row_task_id(row)
            return task_id is None or task_id >= steady_from

        kept = [r for r in pool if _at_or_after(r)]
        info["dropped_by_steady_from"] = len(pool) - len(kept)
        if kept:
            pool = kept
        else:
            # 全被滤掉时退回不滤，但要在 row_selection 里说出来 ——
            # 否则 --steady-from 写错一个数字会悄悄不起作用
            info["note"] = (f"--steady-from {steady_from} 滤掉了所有行，"
                            "本次忽略该选项")

    consistent = [r for r in pool if not _is_self_contradictory(r)]
    info["dropped_cold_start"] = len(pool) - len(consistent)
    if consistent:
        pool = consistent
    elif info["dropped_cold_start"]:
        # 全部行都自相矛盾：别偷偷挑一行，让可信度检查看到这件事
        info["note"] = ("所有候选行都出现核执行时间 > 任务时长，"
                        "整次采集都没有稳态数据")

    durations = sorted(row_duration(r) for r in pool)
    if durations and durations[-1] > 0:
        median = statistics.median(durations)
        info["duration_min_us"] = durations[0]
        info["duration_median_us"] = median
        info["duration_max_us"] = durations[-1]
        info["duration_ranked"] = True
        picked = min(pool, key=lambda r: abs(row_duration(r) - median))
    else:
        picked = pool[0]
        info["note"] = (info["note"] + "；" if info["note"] else "") + (
            "没有可解析的 Task Duration 列，无法区分冷启动与稳态，取了第一行")

    info["picked_task_id"] = _row_task_id(picked)
    info["picked_duration_us"] = row_duration(picked)
    info["picked_op_name"] = (picked.get("Op Name")
                              or picked.get("op_name") or "").strip() or None
    block_dim = safe_float(picked.get("Block Dim") or picked.get("block_dim"))
    info["picked_block_dim"] = block_dim
    return picked, info


def _group_dir_of(hit: Path, metric: str) -> Path:
    """一份 CSV 属于哪一次采集。

    msprof 实际落盘比文档里写的多一层：
        <prof_dir>/PROF_GROUP_<时间戳>/PROF_<Metric>/PROF_000001_*/...
    所以「这次采集」指的是 PROF_<Metric> 的上一级目录。
    """
    marker = f"PROF_{metric}"
    for parent in hit.parents:
        if parent.name == marker:
            return parent.parent
    return hit.parent


def locate_metric_csvs(prof_dir: Optional[Path],
                       archive_dir: Optional[Path],
                       group: Optional[str] = None) -> Tuple[Dict[str, Path], List[str]]:
    """定位七份 metric CSV，返回 (已找到, 缺失的 metric 名)。

    支持两种布局，因为上游 ops-profiling 的采集态和归档态路径不同：
      采集态  <prof_dir>/[PROF_GROUP_*/]PROF_<Metric>/**/.../op_summary_*.csv
      归档态  <archive_dir>/op_summary_<Metric>.csv

    采集态用 `**` 开头来容忍中间的 PROF_GROUP_* 层 —— 文档里记的是不带这一层
    的树，而实机落盘带。写死一层就会「明明采到了却说没找到」。

    多次采集并存时**不自动挑一次**。早先是把命中结果排序取第一个，于是
    时间最早、可能只采到两个 metric 的那一次会被静默选中：缺指标的签名里
    停顿数据是缺失的，读起来像「完全没有停顿」，而下游未必检查这件事。
    宁可报错要求 --group 指定。
    """
    found: Dict[str, Path] = {}
    missing: List[str] = []

    if archive_dir is not None:
        for metric in METRICS:
            candidate = archive_dir / f"op_summary_{metric}.csv"
            if candidate.is_file():
                found[metric] = candidate
            else:
                missing.append(metric)
        if found or prof_dir is None:
            return found, missing
        missing = []

    if prof_dir is None:
        return found, list(METRICS)

    # 先按「采集批次」归拢，再决定用哪一批
    groups: Dict[str, Dict[str, List[Path]]] = {}
    for metric in METRICS:
        pattern = os.path.join(
            str(prof_dir), "**", f"PROF_{metric}", "**",
            "mindstudio_profiler_output", "op_summary_*.csv",
        )
        for raw in sorted(glob.glob(pattern, recursive=True)):
            hit = Path(raw)
            key = str(_group_dir_of(hit, metric))
            groups.setdefault(key, {}).setdefault(metric, []).append(hit)

    if not groups:
        return {}, list(METRICS)

    if group:
        matched = [k for k in groups if Path(k).name == group or k.endswith(group)]
        if not matched:
            raise ProbeError(
                f"没有名为 {group!r} 的采集批次。本目录下有:\n  "
                + "\n  ".join(f"{Path(k).name}（{len(v)}/{len(METRICS)} 个 metric）"
                              for k, v in sorted(groups.items()))
            )
        if len(matched) > 1:
            raise ProbeError(
                f"{group!r} 同时匹配到多个采集批次: "
                + ", ".join(Path(k).name for k in sorted(matched))
            )
        chosen = matched[0]
    elif len(groups) == 1:
        chosen = next(iter(groups))
    else:
        raise ProbeError(
            f"{prof_dir} 下有 {len(groups)} 次采集并存，不能替你挑一次:\n  "
            + "\n  ".join(
                f"{Path(k).name}（{len(v)}/{len(METRICS)} 个 metric："
                f"{', '.join(sorted(v))}）"
                for k, v in sorted(groups.items()))
            + "\n\n用 --group <目录名> 指定用哪一次。\n"
              "请挑 metric 齐全的那一次：缺 metric 的签名里停顿数据是缺失的，"
              "读起来像「完全没有停顿」，而那是最容易把人带错方向的误读。"
        )

    for metric in METRICS:
        hits = groups[chosen].get(metric) or []
        if not hits:
            missing.append(metric)
        elif len(hits) > 1:
            raise ProbeError(
                f"采集批次 {Path(chosen).name} 里 {metric} 有 {len(hits)} 份 CSV，"
                "无法确定用哪一份:\n  "
                + "\n  ".join(str(h) for h in hits)
                + "\n\n同一个 metric 采了多次。请清理到只剩一份，"
                  "或改用 ops-profiling 的归档流程后传 --archive-dir。"
            )
        else:
            found[metric] = hits[0]
    return found, missing


def merge_metric_rows(found: Dict[str, Path],
                      op_name: Optional[str],
                      steady_from: Optional[int] = None
                      ) -> Tuple[Dict[str, str], Dict[str, str],
                                 List[str], Dict[str, Any]]:
    """把各 metric 里目标算子的那一行合并成一个大 dict。

    返回 (合并后的列, 列名 -> 来源 metric, 行缺失的 metric, 挑行过程)。
    记录来源是为了让输出里每个值都能写出它的 channel。
    """
    merged: Dict[str, str] = {}
    origin: Dict[str, str] = {}
    row_missing: List[str] = []
    selection: Dict[str, Any] = {}

    for metric, path in found.items():
        try:
            rows = read_csv_rows(path)
        except OSError as exc:
            raise ProbeError(f"无法读取 {path}: {exc}") from exc
        row, info = select_row(rows, op_name, steady_from)
        selection[metric] = info
        if row is None:
            row_missing.append(metric)
            continue
        for key, value in row.items():
            if key is None:
                continue
            column = key.strip()
            if column and column not in merged:
                merged[column] = value
                origin[column] = metric
    return merged, origin, row_missing, selection


# ---------------------------------------------------------------------------
# 取值助手
# ---------------------------------------------------------------------------

#: 判断「这份 CSV 的比例列是 0-1 小数还是百分数」时用的参照列。
#:
#: 用流水段占比来判，是因为它们一定有实质数值：一个 kernel 总得有某个流水段
#: 是忙的。换成冲突率或等待率就判不了 —— 那些值在一个干净的 kernel 上本来
#: 就可以全都小于 1%，和「0-1 小数」长得一模一样。
SCALE_PROBE_COLUMNS = (
    "aiv_vec_ratio", "aic_cube_ratio", "aiv_mte2_ratio", "aic_mte2_ratio",
    "aiv_mte3_ratio", "aic_mte3_ratio", "aiv_scalar_ratio", "aic_scalar_ratio",
    "aic_mte1_ratio", "aic_fixpipe_ratio", "vec_ratio", "mte2_ratio",
)


def detect_ratio_scale(merged: Dict[str, str]) -> Tuple[float, str]:
    """这份 CSV 的比例列要不要乘 100，才能和百分数阈值比。

    为什么必须判：msprof 的同一类数据有两种写法，而列名会告诉你是哪一种 ——
    带 `(%)` 后缀的列（如 `GM_to_UB_bw_usage_rate(%)`）已经是百分数，
    不带的（如 `aiv_vec_ratio`、`aiv_mte2_wait_ratio`）是 0-1 小数。
    本仓 `ops/ops-profiling/scripts/perf_summary.py` 正是这么处理的：
    对后一类乘 100 再显示，对前一类直接显示。

    本探针的阈值全是百分数（VEC Bound 取 50、ICache 告警取 15、冲突严重取
    15）。把 0-1 的原始值直接拿去比，这些闸门一个都不可能触发 ——
    最大可能值 1.0 连最低的阈值 3 都到不了。于是「不通过」从来不出现，
    而输出看着完全正常：限制资源会显示 0.874 而不是 87.4%。
    可信度检查里的 ICache 那一条也就此失效。

    判法：看流水段占比的最大值。超过 1 说明这份文件用的是百分数写法，
    不用动；否则按 0-1 处理，乘 100。判完对**所有**小数族列统一应用同一个
    系数 —— 逐列各判会把一个真实的 0.5% 冲突率错当成小数放大成 50%。
    """
    seen = []
    for column in SCALE_PROBE_COLUMNS:
        if column in merged:
            value = safe_float(merged[column])
            if value is not None:
                seen.append(value)
    if not seen:
        return 1.0, "无流水段占比可参照，按百分数原样处理（未经确认）"
    peak = max(seen)
    if peak > 1.0:
        return 1.0, f"流水段占比峰值 {peak}，判为百分数写法"
    return 100.0, f"流水段占比峰值 {peak} 不超过 1，判为 0-1 小数写法，已乘 100"


class Reader:
    """按列名取值，并记录它来自哪份 CSV。

    msprof 的列名在不同版本/不同核类型下有 aiv_ / aic_ / ai*_ 前缀差异，
    所以每个逻辑量都给一组候选列名，取第一个取到的。
    """

    def __init__(self, merged: Dict[str, str], origin: Dict[str, str]) -> None:
        self.merged = merged
        self.origin = origin
        self.ratio_scale, self.ratio_scale_basis = detect_ratio_scale(merged)
        #: 有内容但解析不动的列。由 number() 累积，组装签名时进 trust 段。
        self.unparseable: List[Dict[str, str]] = []

    def ratio(self, *columns: str) -> Tuple[Optional[float], Optional[str]]:
        """取一个比例列，统一换算成百分数。

        用于列名**不带** `(%)` 的那一族（占比、等待率、冲突率、ICache 缺失率）。
        带 `(%)` 的列本来就是百分数，用 number() 取。
        """
        value, channel = self.number(*columns)
        if value is None:
            return value, channel
        return value * self.ratio_scale, channel

    def number(self, *columns: str) -> Tuple[Optional[float], Optional[str]]:
        """取一个数值字段。解析不动的列记进 self.unparseable。

        为什么要记而不是就地报错：一列读不动不代表整份签名不可用，但它
        **必须被说出来** —— 不然这个字段在签名里和「没采到」长得一样，
        而两者该做的事完全不同（见 UNPARSEABLE 的注释）。
        调用方在组装签名时把这份清单塞进 trust 段，由那里决定扣不扣留。
        """
        for column in columns:
            if column in self.merged:
                value, problem = parse_cell(self.merged[column])
                if problem is not None:
                    metric = self.origin.get(column, "?")
                    self.unparseable.append({
                        "column": column,
                        "channel": f"msprof:{metric}.csv:{column}",
                        "raw": str(self.merged[column])[:80],
                    })
                    continue
                if value is not None:
                    metric = self.origin.get(column, "?")
                    return value, f"msprof:{metric}.csv:{column}"
        return MISSING, None

    def text(self, *columns: str) -> Tuple[Optional[str], Optional[str]]:
        for column in columns:
            if column in self.merged:
                raw = str(self.merged[column]).strip()
                if raw:
                    metric = self.origin.get(column, "?")
                    return raw, f"msprof:{metric}.csv:{column}"
        return MISSING, None


def field(value: Optional[float], channel: Optional[str],
          note: str = "") -> Dict[str, Any]:
    """统一的字段形状：值 + 通道 + 可选说明。

    value 为 None 表示没采到。消费方必须区分它和 0 —— 这正是本探针
    存在的主要理由之一。
    """
    out: Dict[str, Any] = {"value": value, "channel": channel,
                           "present": value is not None}
    if note:
        out["note"] = note
    return out


# ---------------------------------------------------------------------------
# 一、测量可信度
# ---------------------------------------------------------------------------

def check_trust(reader: Reader, missing_metrics: List[str],
                selection: Optional[Dict[str, Any]] = None,
                stalls: Optional[Dict[str, Any]] = None,
                assume_no_dvfs: bool = False) -> Dict[str, Any]:
    """架构推断之前必过的一关。

    这一步不是形式检查。降频会把计算段和搬运段按不同比例拉长，于是一个
    计算受限的 kernel 可以看起来像搬运受限 —— 据此反推出的架构是错的，
    而且因为数据看着很正常，错误极难被发现。
    """
    checks: List[Dict[str, Any]] = []
    blocking = False

    # 解析不动的列：阻塞。
    #
    # 这一条要阻塞而不是告警，理由在「这个字段在签名里长什么样」：它和
    # 「没采到」一模一样（value=None、present=false）。于是下游读到的是
    # 「这次没采到这个指标」，而真相是**那一列的量纲或格式和我们以为的
    # 不一样**，同一份 CSV 里别的列照样被读走了。
    #
    # 也就是说，签名里混着一个我们读错了的字段，而它伪装成一个缺失字段。
    # 这正是本探针反复要堵的那种失效：把「工具与数据不匹配」静默降级成
    # 「这次没采」。
    if reader.unparseable:
        blocking = True
        columns = [item["column"] for item in reader.unparseable]
        samples = "; ".join(f"{i['column']}={i['raw']!r}"
                            for i in reader.unparseable[:3])
        checks.append({
            "name": "cell_parse", "verdict": "fail",
            "detail": (f"{len(columns)} 列有内容但解析不出数值（{columns}）。\n"
                       f"  样本: {samples}\n"
                       "  这不是「没采到」—— 空值与 N/A 会被正常识别成缺失。"
                       "有内容而读不动，说明那一列的\n"
                       "  量纲或格式和本工具以为的不一样，而同一份 CSV 里"
                       "别的列照样被读走了。\n"
                       "  于是签名里混着一个读错了的字段，它伪装成一个缺失"
                       "字段 —— 所以这里扣留全部架构假设，\n"
                       "  先确认 msprof 版本与列定义（见 ops/ops-profiling 的 "
                       "csv_fields_reference.md）。"),
            "channel": reader.unparseable[0]["channel"],
            "unparseable": reader.unparseable,
        })

    current, cur_ch = reader.number("Current Freq", "current_freq")
    rated, rated_ch = reader.number("Rated Freq", "rated_freq")
    if current is None or rated is None:
        # 取不到频率**必须**阻塞，不能判 unknown 放行。
        #
        # 这条检查是扣留架构假设的第一道闸：降频会把各流水段按不同比例拉长，
        # 于是一个计算受限的 kernel 可以看起来像搬运受限。放行等于让这道闸
        # 在「数据缺失」这种最常见的情形下消失，而输出看着完全正常。
        #
        # 而且本机就是会缺：ops-profiling 的 msprof-guide 写明
        # `op_summary.csv` 不含 Current Freq / Rated Freq 两列，频率要从
        # PROF_Sample 下的 aicore.db 反推。所以这不是罕见情形，是**默认**情形 ——
        # 必须显式补采或显式豁免，不能靠「判不了就算过」糊过去。
        if assume_no_dvfs:
            checks.append({
                "name": "frequency", "verdict": "warn",
                "detail": ("未采到频率字段，但已用 --assume-no-dvfs 声明本机锁频。"
                           "这是一句人工声明，不是测量 —— 若实际发生了降频，"
                           "本次签名推出的架构结论是错的"),
                "channel": None,
            })
        else:
            blocking = True
            checks.append({
                "name": "frequency", "verdict": "fail",
                "detail": ("未采到频率字段（Current Freq / Rated Freq），"
                           "无法判断是否发生 DVFS 降频。"
                           "op_summary.csv 本来就不含这两列（见 ops-profiling 的 "
                           "msprof-guide）。"
                           "正解是导出 msprof_<时间戳>.db 并读其中的 AICORE_FREQ 表"
                           "（timestampNs + freq，无需额外开关，导出 db 时默认生成），"
                           "它给的是频率变化点的真实序列。"
                           "**不要**用 aicore_time/max_cycles 反推主频："
                           "aicore_time 的定义预设了「所有 Block 同时调度且执行时长"
                           "相等」，官方也写明 DVFS 下该字段不准确、不建议参考 —— "
                           "拿它反推频率再去度量逐核不等时长，口径自相矛盾。"
                           "降频会把各流水段按不同比例拉长，据此反推的架构是错的，"
                           "所以这里不放行 —— 确认本机锁频后可用 --assume-no-dvfs 豁免"),
                "channel": cur_ch or rated_ch,
            })
    elif rated > 0 and current < rated * 0.99:
        blocking = True
        checks.append({
            "name": "frequency", "verdict": "fail",
            "detail": (f"发生 DVFS 降频：{current} < 额定 {rated}。"
                       "各流水段被按不同比例拉长，占比分布已失真，"
                       "据此推断架构会得出错误结论"),
            "channel": cur_ch,
        })
    else:
        checks.append({
            "name": "frequency", "verdict": "pass",
            "detail": f"满频运行：{current} / {rated}", "channel": cur_ch,
        })

    icache, icache_ch = reader.ratio(
        "aiv_icache_miss_rate", "aic_icache_miss_rate", "icache_miss_rate")
    if icache is None:
        # 同理：取不到不等于没问题。取指开销会混进各段占比。
        blocking = True
        checks.append({
            "name": "icache", "verdict": "fail",
            "detail": ("未采到 ICache 缺失率（aiv_icache_miss_rate）。"
                       "取指开销会混进各流水段占比，取不到就无法排除这个干扰 —— "
                       "缺失不等于通过。请确认采集了 PipeUtilization 这一组"),
            "channel": None,
        })
    elif icache > THRESHOLDS["icache_miss_warn"]:
        blocking = True
        checks.append({
            "name": "icache", "verdict": "fail",
            "detail": (f"ICache 缺失率 {icache}% 超过 "
                       f"{THRESHOLDS['icache_miss_warn']}%，取指开销混入各段占比"),
            "channel": icache_ch,
        })
    else:
        checks.append({
            "name": "icache", "verdict": "pass",
            "detail": f"ICache 缺失率 {icache}%", "channel": icache_ch,
        })

    # 取的是稳态行还是冷启动行。放进可信度检查而不是只记在日志里：
    # 冷启动行实测比稳态高 75%，而它看起来是一个完全正常的耗时数字，
    # 后面每一轮候选都会和这个虚高的基线做比。
    if selection:
        cold_dropped = sum(i.get("dropped_cold_start", 0) for i in selection.values())
        no_steady = [m for m, i in selection.items()
                     if i.get("dropped_cold_start") and i.get("candidates")
                     and "所有候选行" in (i.get("note") or "")]
        picked = sorted({i.get("picked_task_id") for i in selection.values()
                         if i.get("picked_task_id") is not None})
        unranked = sorted(m for m, i in selection.items()
                          if i.get("candidates") and not i.get("duration_ranked"))
        if no_steady:
            blocking = True
            checks.append({
                "name": "steady_state", "verdict": "fail",
                "detail": (f"{', '.join(sorted(no_steady))} 的所有行都出现"
                           "核执行时间 > 任务时长，整次采集没有稳态数据。"
                           "多跑几次让算子预热后重新采集"),
                "channel": None,
            })
        elif unranked:
            checks.append({
                "name": "steady_state", "verdict": "unknown",
                "detail": (f"{', '.join(unranked)} 没有可解析的 Task Duration 列，"
                           "无法区分冷启动与稳态，取的是第一行。"
                           "冷启动行实测比稳态高 75%，据此做的基线会偏高"),
                "channel": None,
            })
        else:
            checks.append({
                "name": "steady_state", "verdict": "pass",
                "detail": (
                    "取稳态行（Task ID "
                    + (", ".join(str(t) for t in picked) if picked else "未标注")
                    + f"）；{len(selection)} 份 metric 合计丢弃 "
                      f"{cold_dropped} 行冷启动"),
                "channel": None,
            })

    # 比例列的口径没判出来时，所有百分数阈值都可能是 100 倍的错 ——
    # 而错的方向是「闸门永不触发」：输出里一条告警都没有，看着像一切正常。
    # 这件事必须进可信度检查，不能只留在 ratio_scale.basis 里等人去看。
    if "未经确认" in (getattr(reader, "ratio_scale_basis", "") or ""):
        blocking = True
        checks.append({
            "name": "ratio_scale", "verdict": "fail",
            "detail": ("无法判断比例列是 0-1 小数还是百分数（缺少流水段占比列"
                       "作参照）。两种口径差 100 倍，判错的后果是占比、等待率、"
                       "冲突率的全部阈值都不可能触发 —— 输出里一条告警都没有，"
                       "看着像一切正常。请补采 PipeUtilization 后重做"),
            "channel": None,
        })

    # 停顿数据：此前只检查「文件在不在」和「目标行在不在」，**不检查列**。
    # 于是列名在这版 CANN 上改了、或采到一份全 N/A 的占位文件，
    # missing_metrics 是空的 → metric_coverage 通过 → 架构假设照出，
    # 而输出里 most_blocked 显示「未采到停顿数据，无法判断谁在等谁」。
    # 停顿是架构判断的一半依据，这一半静默消失是最坏的情形。
    if stalls is not None and "ResourceConflictRatio" not in missing_metrics:
        # 只在那份 CSV 确实存在时才查列。整份文件缺失由 metric_coverage 负责，
        # 在这里重复报一遍、还说成「文件在但列不在」，反而把人指错方向。
        waits = stalls.get("wait") or {}
        conflicts = stalls.get("conflict") or {}
        wait_present = sum(1 for e in waits.values() if e.get("present"))
        cflt_present = sum(1 for e in conflicts.values() if e.get("present"))
        if wait_present == 0 and cflt_present == 0:
            blocking = True
            checks.append({
                "name": "stall_data", "verdict": "fail",
                "detail": ("ResourceConflictRatio 的文件和目标行都在，但十个停顿列"
                           "（5 个 wait + 5 个 cflt）一个都没取到值。"
                           "多半是列名在这版 CANN 上变了，或采到的是一份全 N/A 的"
                           "占位文件。停顿是架构判断的一半依据 —— "
                           "「文件在」不等于「数据在」"),
                "channel": None,
            })
        elif wait_present == 0 or cflt_present == 0:
            checks.append({
                "name": "stall_data", "verdict": "warn",
                "detail": (f"停顿数据只取到一半（wait {wait_present}/5，"
                           f"冲突 {cflt_present}/5）。缺的那半在输出里标 "
                           "present=false，不要当成 0"),
                "channel": None,
            })
        else:
            checks.append({
                "name": "stall_data", "verdict": "pass",
                "detail": f"停顿列齐（wait {wait_present}/5，冲突 {cflt_present}/5）",
                "channel": None,
            })

    # 七份 metric 来自**七次独立的程序运行**（msprof 每次只支持一个
    # --aic-metrics 组）。合并成一份签名时，若各份挑中的行其实是不同算子、
    # 或不同的 dispatch 路径，那么签名里的每个数单独都对、合起来是拼的 ——
    # 而架构假设恰恰建立在跨份的乘法上（如「搬运占比高但带宽利用率低」
    # 把 PipeUtilization 的占比和 Memory 的带宽放在一起比）。
    # Op Name 和 Block Dim 是最便宜的两个交叉校验。
    if selection:
        names = {i.get("picked_op_name") for i in selection.values()
                 if i.get("picked_op_name")}
        dims = {i.get("picked_block_dim") for i in selection.values()
                if i.get("picked_block_dim") is not None}
        if len(names) > 1:
            blocking = True
            checks.append({
                "name": "cross_metric", "verdict": "fail",
                "detail": (f"各份 metric 挑中的算子不是同一个：{sorted(names)}。"
                           "七份 CSV 来自七次独立运行，合并出的签名是拼的 —— "
                           "请给 --op-name 一个精确的算子名"),
                "channel": None,
            })
        elif len(dims) > 1:
            blocking = True
            checks.append({
                "name": "cross_metric", "verdict": "fail",
                "detail": (f"各份 metric 挑中的行 Block Dim 不一致：{sorted(dims)}。"
                           "说明合并的是两条不同的 dispatch 路径，"
                           "跨份相乘得出的结论不成立"),
                "channel": None,
            })
        else:
            checks.append({
                "name": "cross_metric", "verdict": "pass",
                "detail": (f"各份 metric 挑中的是同一个算子"
                           + (f"（Block Dim {dims.pop()}）" if dims else "")),
                "channel": None,
            })

    if missing_metrics:
        severity = "fail" if "ResourceConflictRatio" in missing_metrics else "warn"
        if severity == "fail":
            blocking = True
        checks.append({
            "name": "metric_coverage", "verdict": severity,
            "detail": (f"缺少 metric: {', '.join(missing_metrics)}。"
                       "缺失的指标在输出里标为 present=false，"
                       "**不要**把它当作 0 —— 停顿数据缺失时读起来会像"
                       "「完全没有停顿」，那是本探针最想避免的误读"),
            "channel": None,
        })
    else:
        checks.append({
            "name": "metric_coverage", "verdict": "pass",
            "detail": f"七份 metric 齐全", "channel": None,
        })

    return {
        "trustworthy": not blocking,
        "checks": checks,
        "blocking_reasons": [c["detail"] for c in checks if c["verdict"] == "fail"],
    }


# ---------------------------------------------------------------------------
# 二、各段提取
# ---------------------------------------------------------------------------

def extract_launch(reader: Reader) -> Dict[str, Any]:
    """launch structure：核数与角色分工。

    Mix Block Dim 是昇腾上最接近论文「warp 角色专门化」的直接观测：
    文档写明非 Mix 算子显示 N/A，所以它有值就说明 Cube 与 Vector 有分工。
    """
    block_dim, bd_ch = reader.number("Block Dim", "block_dim")
    mix_raw, mix_ch = reader.text("Mix Block Dim", "mix_block_dim")
    duration, dur_ch = reader.number("Task Duration(us)", "Task Duration", "task_duration")
    aic_time, aic_ch = reader.number("aic_time(us)", "aic_time")
    aiv_time, aiv_ch = reader.number("aiv_time(us)", "aiv_time")

    mix_value = safe_float(mix_raw) if mix_raw is not None else None
    if mix_raw is None:
        role = "unknown"
        role_note = "未采到 Mix Block Dim，无法判断是否 MIX 算子"
    elif mix_value is None:
        role = "single_role"
        role_note = "Mix Block Dim 为 N/A：纯 Vector 或纯 Cube 算子，无核间分工"
    else:
        role = "mix"
        role_note = f"Mix Block Dim = {mix_value}：MIX 算子，Cube 与 Vector 分工协作"

    return {
        "block_dim": field(block_dim, bd_ch),
        "mix_block_dim": field(mix_value, mix_ch,
                               "N/A 表示非 MIX 算子，不是 0"),
        "task_duration_us": field(duration, dur_ch),
        "aic_time_us": field(aic_time, aic_ch),
        "aiv_time_us": field(aiv_time, aiv_ch),
        "role_split": {"value": role, "note": role_note, "channel": mix_ch},
    }


def extract_pipeline(reader: Reader) -> Dict[str, Any]:
    """pipeline utilization：各流水段 cycle 占比，并判出限制资源。"""
    specs = [
        ("vec", ("aiv_vec_ratio", "vec_ratio"), THRESHOLDS["vec_bound_ratio"], "VEC Bound"),
        ("cube", ("aic_cube_ratio", "cube_ratio"), THRESHOLDS["cube_normal_ratio"], "Cube 活跃"),
        ("scalar", ("aiv_scalar_ratio", "aic_scalar_ratio", "scalar_ratio"),
         THRESHOLDS["scalar_bound_ratio"], "SCALAR Bound"),
        ("mte1", ("aic_mte1_ratio", "mte1_ratio"), None, None),
        ("mte2", ("aiv_mte2_ratio", "aic_mte2_ratio", "mte2_ratio"),
         THRESHOLDS["mte2_bound_ratio"], "MTE2 Bound"),
        ("mte3", ("aiv_mte3_ratio", "aic_mte3_ratio", "mte3_ratio"), None, None),
        ("fixpipe", ("aic_fixpipe_ratio", "fixpipe_ratio"),
         THRESHOLDS["fixpipe_misalign_ratio"], "地址未对齐"),
    ]

    out: Dict[str, Any] = {}
    flags: List[Dict[str, Any]] = []
    best_name, best_value = None, -1.0

    for name, columns, threshold, label in specs:
        value, channel = reader.ratio(*columns)
        out[name] = field(value, channel)
        if value is None:
            continue
        if threshold is not None and value > threshold:
            flags.append({
                "flag": label, "segment": name, "value": value,
                "threshold": threshold, "channel": channel,
            })
        # fixpipe 反映的是对齐问题而非算力占用，不参与限制资源评选
        if name != "fixpipe" and value > best_value:
            best_name, best_value = name, value

    out["flags"] = flags
    out["limiting"] = {
        "value": best_name,
        "ratio": best_value if best_name else None,
        "note": ("占比最高的流水段。这是候选的限制资源，"
                 "但要结合停顿数据确认 —— 占比高也可能是因为它在等别人"),
        "channel": out.get(best_name, {}).get("channel") if best_name else None,
    }
    return out


def extract_stalls(reader: Reader) -> Dict[str, Any]:
    """stall decomposition：各单元被阻塞的占比与冲突细分。

    这一段是论文 NCU pass 里 stall decomposition 的直接对应物。
    ResourceConflictRatio 是上板独有数据，仿真拿不到 —— 所以它缺失时
    check_trust 会判 fail，而不是降级继续。
    """
    waits = {
        "cube_wait": ("aic_cube_wait_ratio",),
        "vec_wait": ("aiv_vec_wait_ratio",),
        "mte1_wait": ("aic_mte1_wait_ratio", "mte1_wait_ratio"),
        "mte2_wait": ("aiv_mte2_wait_ratio", "aic_mte2_wait_ratio", "mte2_wait_ratio"),
        "mte3_wait": ("aiv_mte3_wait_ratio", "aic_mte3_wait_ratio", "mte3_wait_ratio"),
    }
    conflicts = {
        "total": ("aiv_vec_total_cflt_ratio",),
        "bankgroup": ("aiv_vec_bankgroup_cflt_ratio",),
        "bank": ("aiv_vec_bank_cflt_ratio",),
        "resource": ("aiv_vec_resc_cflt_ratio",),
        "mte": ("aiv_vec_mte_cflt_ratio",),
    }

    out: Dict[str, Any] = {"wait": {}, "conflict": {}, "flags": []}

    for name, columns in waits.items():
        value, channel = reader.ratio(*columns)
        out["wait"][name] = field(value, channel)

    for name, columns in conflicts.items():
        value, channel = reader.ratio(*columns)
        out["conflict"][name] = field(value, channel)

    total = out["conflict"]["total"]["value"]
    if total is not None:
        if total > THRESHOLDS["conflict_total_severe"]:
            out["flags"].append({
                "flag": "冲突严重", "value": total,
                "threshold": THRESHOLDS["conflict_total_severe"],
                "channel": out["conflict"]["total"]["channel"],
            })
        elif total > THRESHOLDS["conflict_total_good"]:
            out["flags"].append({
                "flag": "冲突偏高", "value": total,
                "threshold": THRESHOLDS["conflict_total_good"],
                "channel": out["conflict"]["total"]["channel"],
            })

    for name in ("bankgroup", "bank", "mte"):
        value = out["conflict"][name]["value"]
        if value is not None and value > THRESHOLDS["conflict_sub_good"]:
            out["flags"].append({
                "flag": f"{name} 冲突超标", "value": value,
                "threshold": THRESHOLDS["conflict_sub_good"],
                "channel": out["conflict"][name]["channel"],
            })

    # 等待占比最高的单元往往不是瓶颈本身，而是瓶颈的受害者
    present = {k: v["value"] for k, v in out["wait"].items() if v["value"] is not None}
    if present:
        worst = max(present, key=lambda k: present[k])
        out["most_blocked"] = {
            "value": worst, "ratio": present[worst],
            "note": ("等待占比最高的单元通常是受害者而非瓶颈。"
                     "它在等谁，要结合 pipeline 占比一起看"),
            "channel": out["wait"][worst]["channel"],
        }
    else:
        out["most_blocked"] = {
            "value": None, "ratio": None,
            "note": "未采到停顿数据，无法判断谁在等谁", "channel": None,
        }
    return out


def extract_memory(reader: Reader, min_gm_bytes: Optional[int]) -> Dict[str, Any]:
    """数据搬运量与带宽利用率。

    这一段比占比更有价值：占比只说「搬运占了多少时间」，搬运量说
    「到底搬了多少字节」。有了字节数和理论下界，就能算出数据复用倍数 ——
    那是对参考实现 tiling 策略最直接的一个量化约束。
    """
    volumes = {
        "read_main_memory_kb": ("read_main_memory_datas(KB)",),
        "write_main_memory_kb": ("write_main_memory_datas(KB)",),
        "gm_to_l1_kb": ("GM_to_L1_datas(KB)",),
        "l1_to_gm_kb": ("L1_to_GM_datas(KB)(estimate)", "L1_to_GM_datas(KB)"),
        "l0c_to_l1_kb": ("L0C_to_L1_datas(KB)",),
        "l0c_to_gm_kb": ("L0C_to_GM_datas(KB)",),
        "gm_to_ub_kb": ("GM_to_UB_datas(KB)",),
        "ub_to_gm_kb": ("UB_to_GM_datas(KB)",),
    }
    usages = {
        "gm_to_l1": ("GM_to_L1_bw_usage_rate(%)",),
        "l1_to_gm": ("L1_to_GM_bw_usage_rate(%)(estimate)", "L1_to_GM_bw_usage_rate(%)"),
        "gm_to_ub": ("GM_to_UB_bw_usage_rate(%)",),
        "ub_to_gm": ("UB_to_GM_bw_usage_rate(%)",),
    }

    out: Dict[str, Any] = {"volume": {}, "bw_usage": {}, "flags": []}
    for name, columns in volumes.items():
        value, channel = reader.number(*columns)
        out["volume"][name] = field(value, channel)
    for name, columns in usages.items():
        value, channel = reader.number(*columns)
        out["bw_usage"][name] = field(value, channel)

    # 带宽利用率低 + 搬运占比高 = 搬运方式有问题（战术层可改）
    # 带宽利用率高 + 搬运占比高 = 已达硬件极限（只能减少搬运量，架构层）
    for name, entry in out["bw_usage"].items():
        value = entry["value"]
        if value is None:
            continue
        if value < THRESHOLDS["bw_usage_good"]:
            out["flags"].append({
                "flag": f"{name} 带宽利用率偏低",
                "value": value, "threshold": THRESHOLDS["bw_usage_good"],
                "note": ("搬运效率问题，属战术层（对齐、碎片化）；"
                         "与「搬运量过大」是两回事，改法不同"),
                "channel": entry["channel"],
            })

    gm_to_ub = out["volume"]["gm_to_ub_kb"]["value"]
    if min_gm_bytes is not None and min_gm_bytes > 0 and gm_to_ub is not None:
        actual_bytes = gm_to_ub * 1024.0
        reuse = actual_bytes / float(min_gm_bytes)
        out["reuse_factor"] = {
            "value": reuse,
            "note": (f"实际 GM→UB 搬运 {actual_bytes:.0f} 字节 ÷ 理论下界 "
                     f"{min_gm_bytes} 字节。接近 1 表示几乎无冗余读；"
                     "明显大于 1 表示同一份数据被反复搬入，"
                     "这是提高驻留层级或加大复用粒度的直接依据"),
            "channel": out["volume"]["gm_to_ub_kb"]["channel"],
            "derived": True,
        }
    else:
        out["reuse_factor"] = {
            "value": None,
            "note": ("未给 --min-gm-bytes，无法计算复用倍数。"
                     "理论下界 = 必须读的输入字节数，由算子语义和 shape 决定"),
            "channel": None, "derived": True,
        }
    return out


def extract_cache(reader: Reader) -> Dict[str, Any]:
    hit, channel = reader.number(
        "aiv_total_hit_rate(%)", "aic_total_hit_rate(%)", "total_hit_rate(%)")
    out: Dict[str, Any] = {"total_hit_rate": field(hit, channel), "flags": []}
    if hit is not None:
        if hit < THRESHOLDS["l2_hit_poor"]:
            out["flags"].append({
                "flag": "L2 命中率需优化", "value": hit,
                "threshold": THRESHOLDS["l2_hit_poor"],
                "note": "局部性差，可能是分块顺序或访问步长的架构决策所致",
                "channel": channel,
            })
        elif hit < THRESHOLDS["l2_hit_good"]:
            out["flags"].append({
                "flag": "L2 命中率一般", "value": hit,
                "threshold": THRESHOLDS["l2_hit_good"], "channel": channel,
            })
    return out


# ---------------------------------------------------------------------------
# 三、暴露时间（timeline probe 的昇腾替代）
# ---------------------------------------------------------------------------

#: 逐核耗时差到多少倍算「有掉队核」。
#:
#: 上游 timeline 探针的判据是「同角色的执行跨度差 >2× 即告警」。
#: 这里是同一条判据的无时间轴版本。
UNEVEN_SPAN_FACTOR = 2.0

#: 逐核归档文件名。由 ops-profiling 的 msprof_perf_summary.py 产出，
#: 和七份 op_summary CSV 并排放在归档目录里。
PER_CORE_FILENAME = "per_core_cycles.csv"


def load_per_core_cycles(archive_dir: Optional[Path]) -> List[Tuple[int, int]]:
    """读逐核 cycle。空列表表示这一轮没采到。"""
    if archive_dir is None:
        return []
    path = archive_dir / PER_CORE_FILENAME
    if not path.is_file():
        return []
    out: List[Tuple[int, int]] = []
    try:
        for row in read_csv_rows(path):
            core = safe_float(row.get("coreid"))
            cycles = safe_float(row.get("task_cycles"))
            if core is not None and cycles is not None and cycles > 0:
                out.append((int(core), int(cycles)))
    except OSError:
        return []
    return out


def compute_core_balance(archive_dir: Optional[Path]) -> Dict[str, Any]:
    """逐核负载是否均衡 —— 「尾部收敛」这一个分量的直接测量。

    为什么值得单独做：`exposure` 把启动开销、核间等待、尾部收敛三件事混在
    一个标量里，输出里明说「不告诉你暴露在哪一段」。而尾部收敛这一段是
    **可以直接测的**：Task Duration 必然 ≥ 最慢那个核的时长，所以逐核分布
    的离散度就是它的下界。

    数据来源是 ops-profiling 已经归档的 `per_core_cycles.csv`
    （`coreid, task_cycles` 两列），和七份 op_summary 并排放着。

    **比例直接在 cycle 上算，不做频率换算** —— 这一点要紧：
    `aicore_time(us)` 的官方定义预设了「所有 Block 同时调度且执行时长相等」，
    拿它反推主频再去度量逐核**不等**时长，口径自相矛盾，而且官方写明
    DVFS 下该字段不准确。而 (max−min)/max 是个比值，分子分母同单位，
    根本不需要知道频率。
    """
    rows = load_per_core_cycles(archive_dir)
    if not rows:
        return {
            "available": False,
            "reason": (f"归档目录里没有 {PER_CORE_FILENAME}。"
                       "它由 ops-profiling 的 msprof_perf_summary.py 从 "
                       "PROF_Sample 下的 aicore.db 产出 —— "
                       "采集时要带 --aic-mode=sample-based"),
        }
    cycles = [c for _, c in rows]
    lo, hi = min(cycles), max(cycles)
    spread_pct = (hi - lo) / hi * 100.0 if hi > 0 else 0.0
    ratio = (hi / lo) if lo > 0 else None
    ordered = sorted(rows, key=lambda item: -item[1])
    out: Dict[str, Any] = {
        "available": True,
        "cores": len(rows),
        "min_cycles": lo,
        "max_cycles": hi,
        "spread_pct": round(spread_pct, 2),
        "max_over_min": round(ratio, 2) if ratio else None,
        "slowest": [{"core": c, "cycles": v} for c, v in ordered[:3]],
        "fastest": [{"core": c, "cycles": v} for c, v in ordered[-3:][::-1]],
        "unit": "cycle（不做频率换算：比值不需要知道频率，"
                "而 aicore_time 反推主频的口径在 DVFS 下不成立）",
        "flags": [],
    }
    if ratio and ratio >= UNEVEN_SPAN_FACTOR:
        out["flags"].append({
            "flag": "有掉队核",
            "value": out["max_over_min"],
            "threshold": UNEVEN_SPAN_FACTOR,
            "detail": (f"最慢核 {ordered[0][0]} 是最快核的 "
                       f"{out['max_over_min']} 倍。Task Duration 由最慢那个核"
                       "定义，所以这部分暴露时间是尾部收敛而不是核间等待 —— "
                       "该改的是划分方式（block 映射 / 尾轮均衡），"
                       "不是流水重叠"),
        })
    return out


def compute_role_exposure(launch: Dict[str, Any],
                          pipeline: Dict[str, Any],
                          exposure: Dict[str, Any]) -> Dict[str, Any]:
    """按角色（AIC / AIV）把暴露时间分摊开。

    为什么要这一段：`exposure` 只说「有多少时间没被计算覆盖」，`most_blocked`
    只说「谁等得最久」的名字 —— 而架构层要决定**先攻哪条决策**时，需要的是
    「每个单元各吃掉多少 µs」。论文实现的角色账给的就是这个数，它的读法规则
    是硬的：**暴露时间最多的那个角色拥有残差，诊断的 locus 必须落在该角色
    治下的决策上**。

    论文的算法是 `exposed = duration × share × (1 − tensor_active)`。
    昇腾这边三个因子都是现成的，而且有一处比上游干净：

      duration       aic_time_us / aiv_time_us 分别给（上游只有一个标量）
      tensor_active  aic_cube_ratio / aiv_vec_ratio **按角色分开**
                     —— AIC 与 AIV 是物理分离的核、各有独立 PMU 计数器，
                     所以每个角色的「闲置比例」能单独算，角色之间不互相污染。
                     上游的 tensor_active 是一个标量，所有角色共用它。

    映射的边界要说清（会写进输出的 caveat）：CUDA 的 producer/consumer
    warp group 是同一个 SM 内的**软件**角色，数量由 kernel 自己定；A2 的
    AIC/AIV 是两类**物理**核，配比由 SoC 固定。所以这里的「角色」只有两个，
    不是 N 个；纯 Vector 或纯 Cube 算子退化成单角色，此时这一段不产生信息，
    如实标成 not_applicable 而不是算出一个 100% 的平凡结论。
    """
    roles = (("aic", "aic_time_us", "cube"), ("aiv", "aiv_time_us", "vec"))
    present = [name for name, time_key, _ in roles
               if isinstance(launch.get(time_key, {}).get("value"), (int, float))
               and launch[time_key]["value"] > 0]
    if len(present) < 2:
        return {
            "available": False,
            "reason": ("只有一类核在工作（"
                       + (present[0] if present else "两类都没采到")
                       + "）。角色账在单角色下不产生信息：残差必然 100% 落在"
                         "那一个角色上，而那不是一条新结论"),
            "roles": {},
        }

    out: Dict[str, Any] = {"available": True, "roles": {}}
    for name, time_key, ratio_key in roles:
        duration = launch[time_key]["value"]
        entry = pipeline.get(ratio_key) or {}
        active_pct = entry.get("value") if entry.get("present") else None
        if not isinstance(active_pct, (int, float)):
            out["roles"][name] = {
                "exposed_us": None, "present": False,
                "reason": f"没采到 {ratio_key}，无法算这个角色的闲置比例",
            }
            continue
        idle_frac = max(0.0, 1.0 - active_pct / 100.0)
        out["roles"][name] = {
            "present": True,
            "duration_us": duration,
            "active_pct": active_pct,
            "idle_frac": round(idle_frac, 4),
            "exposed_us": round(duration * idle_frac, 3),
            "channel": entry.get("channel"),
        }

    scored = [(name, info["exposed_us"]) for name, info in out["roles"].items()
              if info.get("present") and info.get("exposed_us") is not None]
    if scored:
        scored.sort(key=lambda item: -item[1])
        out["owner"] = scored[0][0]
        out["owner_exposed_us"] = scored[0][1]
        out["note"] = (
            f"暴露时间最多的是 {scored[0][0]}（{scored[0][1]} us）。"
            "按论文的读法，残差归它所有 —— 课程表的下一条决策应当落在"
            "这个角色治下，而不是另一个角色上")
    out["caveat"] = (
        "这是估计不是测量：一个角色在另一类核忙的时候等待，是被隐藏而不是"
        "被暴露。另外昇腾的角色是两类物理核（配比由 SoC 固定），"
        "不是 CUDA 那种数量可由 kernel 自定的软件角色"
    )
    out["derived_from"] = ("launch.aic_time_us / aiv_time_us × "
                           "(1 − pipeline.cube/vec 占比)")
    return out


def compute_exposure(launch: Dict[str, Any]) -> Dict[str, Any]:
    """推导暴露时间：Task Duration 中没有被任何核的执行时间覆盖的部分。

    论文对串行延迟受限的 kernel 额外上 timeline probe，看它的时间花在
    哪段没有重叠上。msprof 在算子级没有给等价的时间线视图，但它给了
    Task Duration 和 aic_time / aiv_time，三者之差就是**没有被计算覆盖
    的时间** —— 启动开销、核间等待、尾部收敛都落在这里。

    这个量不是 timeline probe 的完整替代：它只告诉你有多少时间暴露着，
    不告诉你暴露在哪一段。所以它的作用是**判断要不要继续深挖**，
    而不是直接给出结论。这一点在输出的 note 里写明，避免被当成定论。

    aic 与 aiv 取较大者而非相加：两者在 MIX 算子里是并行的，相加会
    高估覆盖时间，从而低估暴露时间 —— 那会让一个真有问题的 kernel
    看起来很健康。
    """
    duration = launch["task_duration_us"]["value"]
    aic = launch["aic_time_us"]["value"]
    aiv = launch["aiv_time_us"]["value"]

    if duration is None or (aic is None and aiv is None):
        return {
            "available": False,
            "reason": ("缺少 Task Duration 或 aic/aiv 执行时间，无法推导暴露时间"),
            "serial_latency_bound": None,
        }

    covered = max(v for v in (aic, aiv) if v is not None)
    exposed = duration - covered
    exposed_pct = (exposed / duration * 100.0) if duration > 0 else 0.0

    if exposed < 0:
        # 两个量来自不同 metric 的采集，偶有不一致；如实报告而不是静默裁剪
        return {
            "available": False,
            "reason": (f"覆盖时间 {covered}us 大于 Task Duration {duration}us，"
                       "两者来自不同次采集时可能不一致，本次不做暴露时间判断"),
            "serial_latency_bound": None,
            "covered_us": covered, "task_duration_us": duration,
        }

    return {
        "available": True,
        "task_duration_us": duration,
        "covered_us": covered,
        "covered_from": "max(aic_time, aiv_time)",
        "exposed_us": exposed,
        "exposed_pct": exposed_pct,
        "threshold_pct": EXPOSURE_SERIAL_PCT,
        "serial_latency_bound": exposed_pct > EXPOSURE_SERIAL_PCT,
        "derived": True,
        "note": ("暴露时间 = Task Duration 减去被计算覆盖的时间，包含启动开销、"
                 "核间等待与尾部收敛。它告诉你有多少时间没被利用，"
                 "**不告诉你暴露在哪一段** —— 判为串行延迟受限时，"
                 "下一步应当做消融（lever_probe.py）把它定位到具体环节。"
                 f"阈值 {EXPOSURE_SERIAL_PCT}% 是本工具的工程判断，无文档出处"),
    }


# ---------------------------------------------------------------------------
# 四、架构假设
# ---------------------------------------------------------------------------

def derive_hypotheses(launch, pipeline, stalls, memory, cache,
                      exposure, role_exposure=None) -> List[Dict[str, Any]]:
    """把观测翻译成**带证据的架构假设**。

    这里产出的每一条都是「假设」而不是「结论」。它们的用途是喂给
    milestone 的 trigger —— 而 trigger 的作用正是说明「观测到什么才
    需要这个决策」。所以每条都带上 channel，让 milestone 可以直接引用。
    """
    out: List[Dict[str, Any]] = []

    def add(claim: str, evidence: Dict[str, Any], channel: Optional[str],
            strength: str, caveat: str = "") -> None:
        item = {"claim": claim, "evidence": evidence,
                "channel": channel, "strength": strength}
        if caveat:
            item["caveat"] = caveat
        out.append(item)

    role = launch["role_split"]["value"]
    if role == "mix":
        add("参考采用 Cube/Vector 核间分工（MIX 算子）",
            {"mix_block_dim": launch["mix_block_dim"]["value"]},
            launch["mix_block_dim"]["channel"], "supported",
            "分工存在是确定的；谁等谁、几级流水不可由此推断")
    elif role == "single_role":
        add("参考是纯 Vector 或纯 Cube 算子，无核间分工",
            {"mix_block_dim": "N/A"}, launch["mix_block_dim"]["channel"],
            "supported")

    mte2 = pipeline["mte2"]["value"]
    gm_usage = memory["bw_usage"]["gm_to_ub"]["value"]
    if mte2 is not None and mte2 > THRESHOLDS["mte2_bound_ratio"]:
        if gm_usage is not None and gm_usage >= THRESHOLDS["bw_usage_good"]:
            add("搬运已接近带宽上限，继续优化搬运方式无收益；"
                "要更快只能减少搬运总量（架构层决策）",
                {"mte2_ratio": mte2, "gm_to_ub_bw_usage": gm_usage},
                pipeline["mte2"]["channel"], "supported")
        elif gm_usage is not None:
            add("搬运占比高但带宽利用率低，问题在搬运方式而非搬运量；"
                "属战术层（对齐、碎片化、步长）",
                {"mte2_ratio": mte2, "gm_to_ub_bw_usage": gm_usage},
                memory["bw_usage"]["gm_to_ub"]["channel"], "supported")
        else:
            add("搬运受限（MTE2 Bound），但缺带宽利用率，"
                "无法区分「搬得太多」与「搬得太慢」",
                {"mte2_ratio": mte2}, pipeline["mte2"]["channel"], "weak",
                "补采 Memory metric 后可区分")

    reuse = memory["reuse_factor"]["value"]
    if reuse is not None and reuse > 1.5:
        add(f"存在约 {reuse:.1f} 倍的冗余 GM 读取，"
            "参考的驻留层级或复用粒度有提升空间",
            {"reuse_factor": reuse},
            memory["reuse_factor"]["channel"], "supported",
            "复用倍数给出冗余总量，但不唯一确定 tile 形状")

    bank = stalls["conflict"]["bank"]["value"]
    bankgroup = stalls["conflict"]["bankgroup"]["value"]
    if bank is not None and bank > THRESHOLDS["conflict_sub_good"]:
        add("UB 地址分配导致 bank 冲突，参考的 buffer 布局是可观测的约束",
            {"bank_cflt_ratio": bank}, stalls["conflict"]["bank"]["channel"],
            "supported", "冲突存在可测；具体布局仍不可直接读出")
    if bankgroup is not None and bankgroup > THRESHOLDS["conflict_sub_good"]:
        add("block stride 设置导致 bankgroup 冲突",
            {"bankgroup_cflt_ratio": bankgroup},
            stalls["conflict"]["bankgroup"]["channel"], "supported")

    fixpipe = pipeline["fixpipe"]["value"]
    if fixpipe is not None and fixpipe > THRESHOLDS["fixpipe_misalign_ratio"]:
        add("输出地址未对齐，这通常由输出张量布局的架构决策导致",
            {"fixpipe_ratio": fixpipe}, pipeline["fixpipe"]["channel"],
            "supported")

    hit = cache["total_hit_rate"]["value"]
    if hit is not None and hit < THRESHOLDS["l2_hit_poor"]:
        add("L2 命中率偏低，分块顺序或访问步长的局部性有问题",
            {"l2_total_hit_rate": hit}, cache["total_hit_rate"]["channel"],
            "supported")

    if exposure.get("serial_latency_bound"):
        add(f"约 {exposure['exposed_pct']:.1f}% 的时间未被计算覆盖，"
            "疑似串行延迟受限",
            {"exposed_pct": exposure["exposed_pct"],
             "exposed_us": exposure["exposed_us"]},
            "derived:task_duration - max(aic_time, aiv_time)", "weak",
            "暴露时间说明有多少时间闲置，不说明闲置在哪一段；"
            "需用 lever_probe.py 做消融定位")

    # 角色归属。这是论文里一条硬的读法规则：暴露时间最多的那个角色拥有
    # 残差，课程表的下一条决策必须落在它治下 —— 而不是落在另一个角色上。
    # 有了这条，arch agent 在「vec_wait 和 mte2_wait 同时偏高」时就有了
    # 量化依据，不必凭 most_blocked 那一个名字猜。
    if role_exposure and role_exposure.get("available") and \
            role_exposure.get("owner"):
        owner = role_exposure["owner"]
        others = {n: i.get("exposed_us") for n, i in
                  (role_exposure.get("roles") or {}).items() if n != owner}
        add(f"残差归 {owner} 所有：它的暴露时间 "
            f"{role_exposure['owner_exposed_us']} us 高于另一角色（{others}）。"
            f"下一条决策应落在 {owner} 治下",
            role_exposure.get("derived_from", "derived:role exposure"),
            "weak", role_exposure.get("caveat"))

    return out


def collect_open_questions(stalls, memory, exposure,
                           core_balance=None) -> List[Dict[str, Any]]:
    """列出本次采集回答不了的问题。

    论文 §3.2：仪器分辨不了的，留作显式的开放问题。诚实标注的开放问题
    可以被后续实验回答，伪装成结论的猜测不能。
    """
    out: List[Dict[str, Any]] = []

    def add(question: str, why: str, probe: str) -> None:
        out.append({"question": question, "why_unresolvable": why,
                    "proposed_probe": probe})

    if memory["reuse_factor"]["value"] is None:
        add("参考的实际数据复用倍数是多少？",
            "未提供理论最小搬运量（--min-gm-bytes），无法与实测搬运量对比",
            "按算子语义与 shape 算出必须读的字节数后重跑本探针")

    # tile 形状在两种情况下都是未知的：没算复用倍数时固然不知道，
    # 算出来了也只约束了冗余总量，不唯一确定划分方式。
    add("参考采用的具体 tile 形状是什么？",
        "搬运量约束了冗余总量，但多种 tile 划分可产生相同的搬运量",
        "实现若干候选 tile 形状，比各自的搬运量与占比分布和参考的接近度")

    if not exposure.get("available"):
        add("参考有多少时间没有被计算覆盖？",
            exposure.get("reason", "缺少推导所需字段"),
            "补采 PipeUtilization 与 OpBasicInfo 后重跑")
    elif exposure.get("serial_latency_bound"):
        # 这个问题现在只剩一半答不了。
        #
        # 暴露时间混着三件事：启动开销、核间等待、尾部收敛。其中**尾部收敛
        # 是可测的** —— 逐核 cycle 分布已经归档，而 Task Duration 必然
        # ≥ 最慢那个核的时长。所以逐核测过之后，要留作开放问题的只有
        # 前两项，不该把整个问题都说成「答不了」（上游的规矩：合法的 waiver
        # 要带理由和已试通道清单，不能把能答的一起 waive 掉）。
        balanced = core_balance or {}
        if balanced.get("available"):
            uneven = bool(balanced.get("flags"))
            add("暴露的时间里，启动开销与核间等待各占多少？",
                ("逐核分布已测（"
                 + ("发现掉队核，尾部收敛是其中一个分量"
                    if uneven else
                    f"逐核离散只有 {balanced.get('spread_pct')}%，"
                    "尾部收敛不是主要分量")
                 + "），但启动开销与核间等待这两项 msprof 算子级无法分离 —— "
                   "PM 采样最细 10ms，而 kernel 是百微秒级，差约 50 倍"),
                "用 lever_probe.py 逐项消融：去掉某个环节看暴露时间如何变化")
        else:
            add("暴露的时间具体花在哪一段？",
                ("msprof 算子级数据没有时间线视图，只能给出暴露总量。"
                 "逐核分布本可切出「尾部收敛」这一个分量，但本次没采到 "
                 f"{PER_CORE_FILENAME}"),
                "采集时带 --aic-mode=sample-based 产出逐核 cycle；"
                "剩余部分用 lever_probe.py 逐项消融")

    if stalls["most_blocked"]["value"] is not None:
        add(f"{stalls['most_blocked']['value']} 在等谁？",
            "等待占比说明某单元被阻塞，但不指出阻塞源",
            "对照 pipeline 各段占比；必要时做消融实验")

    add("参考的流水级数（double buffer 深度）是多少？",
        "重叠的效果可观测（各段饱和度），深度本身不可观测",
        "把「存在重叠」作为架构决策，深度交给战术层扫描")

    add("参考在哪里做了重算、哪里做了存储？",
        "只能从搬运量变化间接感知，而搬运量变化有多种成因",
        "做一版偏重算、一版偏存储的实现，比搬运量与参考的接近度")

    return out


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------

def build_signature(prof_dir: Optional[Path], archive_dir: Optional[Path],
                    op_name: Optional[str],
                    min_gm_bytes: Optional[int],
                    group: Optional[str] = None,
                    steady_from: Optional[int] = None,
                    assume_no_dvfs: bool = False) -> Dict[str, Any]:
    found, missing = locate_metric_csvs(prof_dir, archive_dir, group)
    if not found:
        searched = []
        if archive_dir is not None:
            searched.append(f"{archive_dir}/op_summary_<Metric>.csv")
        if prof_dir is not None:
            searched.append(
                f"{prof_dir}/**/PROF_<Metric>/**/mindstudio_profiler_output/"
                "op_summary_*.csv")
        raise ProbeError(
            "未找到任何 msprof metric CSV。已查找:\n  "
            + "\n  ".join(searched)
            + f"\n期望的 metric: {', '.join(METRICS)}\n"
            "采集方式见 ops/ops-profiling 的 msprof_profile_run.sh"
        )

    merged, origin, row_missing, selection = merge_metric_rows(
        found, op_name, steady_from)
    if not merged:
        raise ProbeError(
            f"在已找到的 CSV 里没有匹配到算子 "
            f"{op_name!r}。可用的 CSV: {[str(p) for p in found.values()]}"
        )

    reader = Reader(merged, origin)
    all_missing = sorted(set(missing) | set(row_missing))

    launch = extract_launch(reader)
    pipeline = extract_pipeline(reader)
    stalls = extract_stalls(reader)
    memory = extract_memory(reader, min_gm_bytes)
    cache = extract_cache(reader)
    exposure = compute_exposure(launch)
    role_exposure = compute_role_exposure(launch, pipeline, exposure)
    core_balance = compute_core_balance(archive_dir)

    # 可信度检查排在**全部**取数之后。
    #
    # 两个理由，后一个是这次挪位置的原因：
    #   它要看停顿列取到没有          → 必须在 extract_stalls 之后
    #   它要看哪些列解析不动          → 必须在**最后一次** reader.number()
    #                                   之后，否则 memory / cache 两段里
    #                                   读出的解析失败它看不见，而那正是
    #                                   搬运量与命中率所在的两段
    trust = check_trust(reader, all_missing, selection, stalls, assume_no_dvfs)

    resolved_name, name_channel = reader.text("Op Name", "op_name")

    signature: Dict[str, Any] = {
        "schema_version": "1.0",
        "op_name": resolved_name or op_name,
        "op_name_channel": name_channel,
        "source": {
            "metrics_found": sorted(found),
            "metrics_missing": all_missing,
            "files": {m: str(p) for m, p in sorted(found.items())},
        },
        # 用了哪一行直接决定全部后续结论，所以写进签名而不是埋在内部
        "row_selection": selection,
        # 比例列的口径同理：判错了所有阈值判定都跟着失效，而输出看着正常
        "ratio_scale": {
            "factor": reader.ratio_scale,
            "basis": reader.ratio_scale_basis,
            "applies_to": "列名不带 (%) 的占比/等待率/冲突率/ICache 缺失率",
        },
        "trust": trust,
        "launch": launch,
        "pipeline": pipeline,
        "stalls": stalls,
        "memory": memory,
        "cache": cache,
        "exposure": exposure,
        # 按角色分摊的暴露时间。决定「先攻哪条决策」要的是这个数，
        # 而 exposure 只给总量、most_blocked 只给名字。
        "role_exposure": role_exposure,
        # 逐核分布。exposure 把启动开销、核间等待、尾部收敛混在一个标量里，
        # 这一段把「尾部收敛」那个分量单独测出来。
        "core_balance": core_balance,
    }

    if trust["trustworthy"]:
        signature["architectural_hypotheses"] = derive_hypotheses(
            launch, pipeline, stalls, memory, cache, exposure, role_exposure)
    else:
        signature["architectural_hypotheses"] = []
        signature["hypotheses_withheld"] = (
            "测量不可信，已扣留架构假设。不可信数据推出的架构是错的，"
            "而且因为数据看着正常，错误极难被发现。先解决 trust 里的"
            "阻塞项再重新采集"
        )

    signature["open_questions"] = collect_open_questions(
        stalls, memory, exposure, core_balance)
    return signature


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def render_text(sig: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append

    add(f"算子      {sig.get('op_name')}")
    add(f"metric    齐 {len(sig['source']['metrics_found'])}/{len(METRICS)}"
        + (f"，缺 {', '.join(sig['source']['metrics_missing'])}"
           if sig['source']['metrics_missing'] else ""))
    add("")

    add("— 测量可信度 —")
    for check in sig["trust"]["checks"]:
        mark = {"pass": "通过", "fail": "不通过", "warn": "告警",
                "unknown": "未知"}[check["verdict"]]
        add(f"  [{mark}] {check['name']}: {check['detail']}")
    add("")

    launch = sig["launch"]
    add("— launch structure —")
    add(f"  Block Dim        {launch['block_dim']['value']}")
    add(f"  角色分工         {launch['role_split']['note']}")
    add(f"  Task Duration    {launch['task_duration_us']['value']} us")
    for metric, info in sorted(sig.get("row_selection", {}).items()):
        if info.get("picked_duration_us") is None:
            continue
        add(f"  取样({metric})  Task ID={info.get('picked_task_id')}"
            f" {info['picked_duration_us']}us"
            f"（候选 {info.get('candidates')} 行，"
            f"丢冷启动 {info.get('dropped_cold_start')} 行，"
            f"稳态区间 {info.get('duration_min_us')}~{info.get('duration_max_us')}us）")
    add("")

    add("— pipeline 占比 —")
    for name in ("vec", "cube", "scalar", "mte1", "mte2", "mte3", "fixpipe"):
        entry = sig["pipeline"][name]
        shown = f"{entry['value']}" if entry["present"] else "未采到"
        add(f"  {name:<9} {shown}")
    limiting = sig["pipeline"]["limiting"]
    add(f"  限制资源         {limiting['value']}（{limiting['ratio']}）")
    for flag in sig["pipeline"]["flags"]:
        add(f"  ! {flag['flag']}: {flag['segment']}={flag['value']} > {flag['threshold']}")
    add("")

    add("— stall decomposition —")
    for name, entry in sig["stalls"]["wait"].items():
        shown = f"{entry['value']}" if entry["present"] else "未采到"
        add(f"  {name:<11} {shown}")
    for flag in sig["stalls"]["flags"]:
        add(f"  ! {flag['flag']}: {flag['value']} > {flag['threshold']}")
    blocked = sig["stalls"]["most_blocked"]
    if blocked["value"]:
        add(f"  阻塞最重        {blocked['value']}（{blocked['ratio']}）")
    add("")

    exposure = sig["exposure"]
    add("— 暴露时间（timeline probe 替代）—")
    if exposure.get("available"):
        add(f"  覆盖 {exposure['covered_us']} us / 总 {exposure['task_duration_us']} us")
        add(f"  暴露 {exposure['exposed_us']:.2f} us（{exposure['exposed_pct']:.1f}%）")
        add(f"  串行延迟受限    {'是' if exposure['serial_latency_bound'] else '否'}")
    else:
        add(f"  不可用：{exposure.get('reason')}")
    add("")

    add("— 架构假设 —")
    if sig["architectural_hypotheses"]:
        for item in sig["architectural_hypotheses"]:
            mark = "有支撑" if item["strength"] == "supported" else "较弱"
            add(f"  [{mark}] {item['claim']}")
            add(f"           通道: {item['channel']}")
            if item.get("caveat"):
                add(f"           注意: {item['caveat']}")
    else:
        add(f"  {sig.get('hypotheses_withheld', '无')}")
    add("")

    add("— 开放问题 —")
    for q in sig["open_questions"]:
        add(f"  · {q['question']}")
        add(f"    为何答不了: {q['why_unresolvable']}")
        add(f"    建议探法:   {q['proposed_probe']}")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行。独立出来是为了让文档命令门禁能拿到它 ——
    门禁拿真实 parser 去解析文档里的示例，parser 藏在 main 里就够不着。
    """
    parser = argparse.ArgumentParser(
        description="从 msprof CSV 提取结构化架构签名（论文 NCU pass 的昇腾对应物）",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prof-dir", type=Path,
                        help="msprof 采集目录（含 PROF_<Metric>/ 子目录）")
    source.add_argument("--archive-dir", type=Path,
                        help="归档目录（含 op_summary_<Metric>.csv）")
    parser.add_argument("--op-name", default=None,
                        help="目标算子名；不给则在全部行里挑")
    parser.add_argument("--group", default=None, metavar="DIR_NAME",
                        help="采集态下指定用哪一次采集（PROF_GROUP_* 的目录名）。"
                             "多次采集并存且未指定时会报错，不会替你挑一次")
    parser.add_argument("--assume-no-dvfs", action="store_true",
                        help="声明本机已锁频，容忍频率字段缺失。"
                             "这是一句人工声明不是测量 —— 实际降频时"
                             "本次签名推出的架构结论是错的")
    parser.add_argument("--steady-from", type=int, default=None, metavar="TASK_ID",
                        help="只用 Task ID 不小于该值的行。"
                             "默认已自动丢弃冷启动行（核执行时间 > 任务时长），"
                             "这个选项用于手工指定更严的稳态起点")
    parser.add_argument("--min-gm-bytes", type=int, default=None,
                        help="理论最小 GM 读取字节数，给了才能算数据复用倍数")
    parser.add_argument("--output", type=Path, default=None,
                        help="把签名 JSON 写到文件")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="标准输出用 JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        sig = build_signature(args.prof_dir, args.archive_dir,
                              args.op_name, args.min_gm_bytes,
                              args.group, args.steady_from,
                              args.assume_no_dvfs)
    except ProbeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.output:
        try:
            args.output.write_text(
                json.dumps(sig, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        except OSError as exc:
            print(f"错误: 无法写入 {args.output}: {exc}", file=sys.stderr)
            return 2

    if args.as_json:
        print(json.dumps(sig, ensure_ascii=False, indent=2))
    else:
        print(render_text(sig))

    if not sig["trust"]["trustworthy"]:
        print("\n测量不可信，架构假设已扣留。先处理上面标为「不通过」的项。",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
