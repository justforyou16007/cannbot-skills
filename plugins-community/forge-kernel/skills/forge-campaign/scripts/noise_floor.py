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
"""噪声底测定。

整个 ForgeKernel 的接受判据是一句话：候选的同 job 增益必须超过噪声底。
所以噪声底定错了，后面全是错的 ——

  定太低  把测量抖动当成真实加速，收下一堆其实没用的改动，
          而且这些改动会污染知识库，被后续 campaign 当成有效经验复用。
  定太高  真实的小幅改进被当噪声丢掉，优化提前停在远低于硬件上限的地方。

论文报告它们观测到的噪声底是 1%~5%，那是它们环境的实测值，不是常数。
机器、芯片、负载、同机其他进程、是否降频都会改变它。所以本脚本的立场是：
**量出来，不要抄**。

两种测法，优先用前一种
-----------------------
cv_pct（首选）
    配对评测对每个 shape 跑 num_trials 次事件计时，取中位数作结果，
    同时算出变异系数。这是**同一次 job 内部**的离散度，正对应论文说的
    same-job noise floor。直接读，不需要额外跑任务。

比值离散（回退）
    结果里没有 cv_pct 时，对同一配置重复跑若干次配对评测，取这些增益
    比值之间的离散度。它们本应全等于同一个真值，离散即噪声。
    缺点是混进了 job 之间的环境差异，估计偏保守。

回退测法给出三个数：
  median      中位数，论文用它作为最终 claim
  cv_pct      变异系数，刻画普遍抖动
  floor_pct   取变异系数与半极差的较大者 —— 两者分别对「普遍抖动」和
              「偶发离群」敏感，取大者是保守选择

用法:
    # 首选：直接读评测结果里的 cv_pct
    python3 noise_floor.py --from-results evaluation_results.json

    # 回退：从多次评测结果的比值离散估计
    python3 noise_floor.py --from-results run1.json run2.json run3.json --method spread

    # 直接给比值
    python3 noise_floor.py --ratios 1.01 0.99 1.02 1.00

退出码: 0 = 正常; 1 = 样本不足或数据异常; 2 = 用法或 IO 错误
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

#: 少于这个样本数算不出可信的离散度。
MIN_SAMPLES = 3

#: 论文观测到的区间，仅用于在结果明显偏离时提醒，不用于裁剪结果。
PAPER_RANGE_PCT = (1.0, 5.0)

#: 噪声底的硬下限。低于它一律按它算。
#:
#: 这不是「论文的经验值」，而是论文实现里一行代码：它的自我提升脚本把
#: 每代的噪声底取成 `max(0.015, spread)`，注释称之为「循环的平台期地板」。
#: 道理是：重复测量算出来的离散度本身也是个估计，样本少时它可以偶然很小。
#: 用一个偶然很小的数当判据，等于把「这一轮运气好」当成「环境干净」，
#: 之后任何微小正增益都会被收下。
#:
#: 我们这边尤其需要它：评测总是先跑 baseline 再跑 evolved，位置偏置实测
#: 约 2.7%，方向恒定、多测不收敛。一个低于 1.5% 的噪声底在这个偏置面前
#: 没有任何意义。
NOISE_FLOOR_MIN_PCT = 1.5

#: 评测自己判定「没跑成」的 gating 取值。
GATING_FAILED = "failed"


class NoiseFloorError(Exception):
    """数据不足以得出噪声底。"""


def _positive_times(doc: Dict[str, Any]) -> List[float]:
    """把这份结果里所有正耗时收集起来，用来判断评测到底有没有跑出数来。"""
    times: List[float] = []
    for side in ("baseline", "evolved"):
        entry = doc.get(side)
        if isinstance(entry, dict):
            value = entry.get("time_us")
            if isinstance(value, (int, float)) and value > 0:
                times.append(float(value))
    shape_results = doc.get("shape_results")
    if isinstance(shape_results, dict):
        for group in shape_results.values():
            if not isinstance(group, list):
                continue
            for item in group:
                if not isinstance(item, dict):
                    continue
                for key in ("baseline_time_us", "evolved_time_us"):
                    value = item.get(key)
                    if isinstance(value, (int, float)) and value > 0:
                        times.append(float(value))
    return times


def _time_fields_present(doc: Dict[str, Any]) -> bool:
    """这份结果里到底有没有耗时字段（不管值是多少）。"""
    for side in ("baseline", "evolved"):
        entry = doc.get(side)
        if isinstance(entry, dict) and "time_us" in entry:
            return True
    shape_results = doc.get("shape_results")
    if isinstance(shape_results, dict):
        for group in shape_results.values():
            if not isinstance(group, list):
                continue
            for item in group:
                if isinstance(item, dict) and (
                        "baseline_time_us" in item or "evolved_time_us" in item):
                    return True
    return False


def assert_usable(doc: Dict[str, Any], label: str) -> None:
    """崩掉的评测不许用来量噪声底。

    这是本脚本最要紧的一道门。原因在于崩溃时写进结果文件的是
    `time_us=-1` 和 `cv_pct=0.0` —— 那个 0.0 在形态上和「环境极其干净」
    完全一样，于是脚本会一声不响地输出 0.00%，还提示把它传给 campaign init。

    噪声底 0 的含义是「任何正增益都算真实改进」。后果不是少测准一点：
    测量抖动会被当成加速收下，写进知识库和产物库，后续 campaign 还会把它
    当成有效经验复用。错得很深，而且因为每一步看起来都正常，很难回溯。
    """
    gating = doc.get("gating")
    if gating == GATING_FAILED:
        raise NoiseFloorError(
            f"{label} 的 gating=failed，这次评测没跑成（编译失败/精度不过/运行报错），"
            "不能用来量噪声底。\n"
            "崩溃的结果里 cv_pct 是填进去的 0.0，拿它算出来的 0.00% 会让"
            "后续任何正增益都被判为真实改进。\n"
            "先把评测跑通（看结果文件里的 error 字段和评测日志），再重新量。"
        )
    # 只在「字段在、但值全是非正数」时拒绝。这正是崩溃的指纹（写成 -1）。
    # 压根没有耗时字段的文件是另一回事（手写的、或只留了比值的旧文件），
    # 那种情况判不出跑没跑成，不该在这里一刀切。
    if _time_fields_present(doc) and not _positive_times(doc):
        raise NoiseFloorError(
            f"{label} 里的 time_us 全是非正数，评测没有产出有效读数"
            "（崩溃时会写成 -1）。\n"
            "通常是算子没加载成功或子进程报错。先看结果文件里的 error 字段。"
        )


def detect_forced_self_baseline(doc: Dict[str, Any]) -> List[str]:
    """认出「自评场景下 speedup 被强制改成 1.0」的结果。

    评测脚本在 baseline 路径与 evolved 路径指向同一处时，会把每个 shape 的
    speedup 直接改成 1.0（自评理应持平）。但它**不改** baseline_time_us 和
    evolved_time_us，所以「speedup 恰好是 1.0，而两侧耗时并不相等」就是这件事
    发生过的指纹。

    为什么必须认出来：比值离散法的全部信息来自 speedup 的波动。speedup 被
    钉成常数 1.0 之后，离散度必然是 0.00% —— 这个 0 不是「环境干净」，
    是「这个测法在自评场景下什么也没测」。
    """
    hits: List[str] = []
    shape_results = doc.get("shape_results")
    if not isinstance(shape_results, dict):
        return hits
    for group, items in shape_results.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            speedup = item.get("speedup")
            t_ref = item.get("baseline_time_us")
            t_cand = item.get("evolved_time_us")
            if (speedup == 1.0
                    and isinstance(t_ref, (int, float)) and t_ref > 0
                    and isinstance(t_cand, (int, float)) and t_cand > 0
                    and abs(t_ref - t_cand) / t_ref > 1e-6):
                hits.append(f"{group}/{item.get('name', '?')}: "
                            f"speedup=1.0 但 {t_ref:.2f}us vs {t_cand:.2f}us")
    return hits


def extract_cv_values(doc: Dict[str, Any]) -> List[float]:
    """从一份 evaluation_results.json 里收集所有 cv_pct。

    配对评测对每个 shape 跑 num_trials 次事件计时，取中位数作结果，同时
    算出变异系数。这个数是**同一次 job 内部**的离散度，比「多次 job 之间
    的比值离散」更贴近论文说的 same-job noise floor —— 后者还混进了
    job 之间的环境差异。所以有 cv_pct 时优先用它。

    cv_pct 可能出现在多处（顶层、baseline/evolved、各 shape 结果），
    全部收集，由调用方取最保守的那个。
    """
    found: List[float] = []

    def _collect(value: Any) -> None:
        if isinstance(value, (int, float)) and value >= 0:
            found.append(float(value))

    _collect(doc.get("cv_pct"))
    for side in ("baseline", "evolved"):
        entry = doc.get(side)
        if isinstance(entry, dict):
            _collect(entry.get("cv_pct"))

    shape_results = doc.get("shape_results")
    if isinstance(shape_results, dict):
        for group in shape_results.values():
            if not isinstance(group, list):
                continue
            for item in group:
                if isinstance(item, dict):
                    _collect(item.get("cv_pct"))

    return found


def compute_from_cv(cv_values: List[float]) -> Dict[str, Any]:
    """直接用实测变异系数作噪声底。

    取最大值而非平均：噪声底是「多大的增益才算真的」这条线，取最宽松的
    读数会让线划得太低，把抖动收进来。
    """
    clean = [float(v) for v in cv_values if isinstance(v, (int, float)) and v >= 0]
    if not clean:
        raise NoiseFloorError("评测结果中没有 cv_pct")

    measured = max(clean)
    floor_pct = max(measured, NOISE_FLOOR_MIN_PCT)
    if measured == 0.0:
        # 恰好 0 不可能是真实的事件计时结果：多次计时完全一致意味着根本没有
        # 重复测量。真实来源是「崩溃时把 cv_pct 填成了 0.0」或「子进程没上报
        # 这个字段」。放过去就等于把噪声底定成 0。
        raise NoiseFloorError(
            "收集到的 cv_pct 全是 0。事件计时重复多次不可能完全一致，"
            "所以这不是「环境干净」，而是这个字段没被真正填上 —— "
            "评测崩溃时会填 0.0，子进程没上报时也是 0.0。\n"
            "请确认评测跑通（gating 不是 failed、time_us > 0、"
            "结果里每个 shape 的 cv_pct > 0）后重新量。"
        )

    warnings: List[str] = []
    if floor_pct > PAPER_RANGE_PCT[1]:
        warnings.append(
            f"噪声底 {floor_pct:.2f}% 高于论文观测区间上限 {PAPER_RANGE_PCT[1]}%。"
            "测量环境可能不稳定：检查是否有其他进程占用同一张卡、"
            "是否发生 DVFS 降频、num_trials 是否够大。"
            "在这个噪声水平下，小于它的真实改进无法被识别"
        )
    clamped = measured < NOISE_FLOOR_MIN_PCT
    if clamped:
        warnings.append(
            f"实测变异系数只有 {measured:.2f}%，低于硬下限 "
            f"{NOISE_FLOOR_MIN_PCT}%，已按下限取值。"
            "cv 量的是同进程内重复计时的抖动，不含跨进程的位置偏置"
            "（先跑 baseline 后跑 evolved，实测约 2.7%、方向恒定）——"
            "所以一个很小的 cv 不代表判据可以定得很低"
        )

    return {
        "method": "cv_pct",
        "samples": len(clean),
        "cv_values": clean,
        "measured_pct": measured,
        "floor_pct": floor_pct,
        "floor_clamped": clamped,
        "warnings": warnings,
    }


def compute(ratios: List[float]) -> Dict[str, Any]:
    """从一组本应相等的增益比值算出噪声底。

    参数 ratios 是重复测量同一配置得到的 t_ref / t_candidate。理想情况下
    它们应当完全相同；实际的离散就是噪声。
    """
    clean = [float(r) for r in ratios if isinstance(r, (int, float)) and r > 0]
    if len(clean) < MIN_SAMPLES:
        raise NoiseFloorError(
            f"样本不足：有效读数 {len(clean)} 个，至少需要 {MIN_SAMPLES} 个。"
            "样本太少时算出的离散度本身就是噪声，不足以作为判据"
        )

    median = statistics.median(clean)
    if median <= 0:
        raise NoiseFloorError("中位数非正，输入数据异常")

    # 变异系数：标准差相对均值的比例，刻画普遍抖动
    mean = statistics.fmean(clean)
    stdev = statistics.stdev(clean) if len(clean) > 1 else 0.0
    cv_pct = (stdev / mean * 100.0) if mean > 0 else 0.0

    # 全极差相对最小值：对偶发离群更敏感。
    #
    # 口径按论文实现对齐（`spread = (max - min) / min`）。早先写的是
    # 「半极差比中位数」—— 比值都在 1.0 附近时 median ≈ min，那个 /2.0
    # 让同一批数据算出的噪声底只有论文的一半，判据系统性偏松一倍。
    spread_pct = (max(clean) - min(clean)) / min(clean) * 100.0

    # 「实测离散为 0」这件事必须在钳位**之前**判。
    #
    # 钳位会把 0 抬到 1.5%，于是「读数完全相同」这个信号就被盖住了 ——
    # 而那恰恰不是「环境干净」，是这些数不是独立测出来的（最典型的是自评
    # 场景下 speedup 被钉成常数 1.0）。拿一个被钳位救回来的 1.5% 继续跑，
    # 等于用一个合法外观掩盖了一份无效输入。
    measured_pct = max(cv_pct, spread_pct)
    if measured_pct == 0.0:
        raise NoiseFloorError(
            "这些比值完全相同，离散度为 0。重复测量不可能给出一模一样的比值，"
            "所以这不是「环境干净」，而是这些数并非独立测出来的。\n"
            "最常见的原因是自评场景（baseline 与 evolved 指向同一处）：评测会"
            "把每个 shape 的 speedup 直接钉成 1.0。改用 --method cv。"
        )

    # 取两个统计量的大者，保守；再按硬下限抬一次
    floor_pct = max(measured_pct, NOISE_FLOOR_MIN_PCT)
    clamped = measured_pct < NOISE_FLOOR_MIN_PCT

    warnings: List[str] = []
    if floor_pct > PAPER_RANGE_PCT[1]:
        warnings.append(
            f"噪声底 {floor_pct:.2f}% 高于论文观测区间上限 {PAPER_RANGE_PCT[1]}%。"
            "测量环境可能不稳定：检查是否有其他进程占用同一张卡、"
            "是否开启了频率调节、重复次数是否够。"
            "在这个噪声水平下，小于它的真实改进无法被识别"
        )
    if clamped:
        warnings.append(
            f"实测离散只有 {measured_pct:.2f}%，低于硬下限 "
            f"{NOISE_FLOOR_MIN_PCT}%，已按下限取值。"
            "样本少时离散度本身就是个不稳的估计；而且评测的位置偏置"
            "（先跑 baseline 后跑 evolved）实测约 2.7%，方向恒定，"
            "低于下限的噪声底在它面前没有意义"
        )

    return {
        "method": "ratio_spread",
        "samples": len(clean),
        "ratios": clean,
        "median": median,
        "mean": mean,
        "stdev": stdev,
        "cv_pct": cv_pct,
        "spread_pct": spread_pct,
        "measured_pct": measured_pct,
        "floor_pct": floor_pct,
        "floor_clamped": clamped,
        "warnings": warnings,
    }


def ratio_from_result(doc: Dict[str, Any]) -> Optional[float]:
    """从一份 evaluation_results.json 里取出增益比值。

    优先用脚本已经算好的 comparison.speedup；没有就用两边时间现算。
    两种来源都没有时返回 None，由调用方决定是跳过还是报错。
    """
    comparison = doc.get("comparison")
    if isinstance(comparison, dict):
        speedup = comparison.get("speedup")
        if isinstance(speedup, (int, float)) and speedup > 0:
            return float(speedup)

    baseline = doc.get("baseline") or {}
    evolved = doc.get("evolved") or {}
    if isinstance(baseline, dict) and isinstance(evolved, dict):
        t_ref = baseline.get("time_us")
        t_cand = evolved.get("time_us")
        if (isinstance(t_ref, (int, float)) and isinstance(t_cand, (int, float))
                and t_ref > 0 and t_cand > 0):
            return float(t_ref) / float(t_cand)
    return None


def _load_docs(paths: List[Path]) -> List[Dict[str, Any]]:
    """读入结果文件，并在这里一次性挡掉没跑成的评测。

    两种测法（cv 与比值离散）都从这里取数，所以门放这一处就够，不会出现
    「一条路径有检查、另一条没有」的不对称。
    """
    docs: List[Dict[str, Any]] = []
    for path in paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise NoiseFloorError(f"无法读取 {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise NoiseFloorError(f"{path} 不是合法 JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise NoiseFloorError(f"{path} 的顶层不是对象，不是评测结果文件")
        assert_usable(doc, str(path))
        docs.append(doc)
    return docs


def _ratios_from_docs(docs: List[Dict[str, Any]],
                      labels: List[str]) -> List[float]:
    ratios: List[float] = []
    skipped = 0
    for doc, label in zip(docs, labels):
        forced = detect_forced_self_baseline(doc)
        if forced:
            raise NoiseFloorError(
                f"{label} 是自评结果（baseline 与 evolved 指向同一处），"
                "比值离散法在这种结果上量不出噪声底。\n"
                f"证据: {forced[0]}\n"
                "评测脚本在自评时把每个 shape 的 speedup 直接钉成 1.0，"
                "于是比值之间的离散必然是 0.00% —— 那个 0 不代表环境干净，"
                "代表这个测法在这里什么也没测到。\n"
                "自评场景请改用 --method cv：它读的是同一次 job 内多次事件计时"
                "的变异系数，不经过 speedup，钉死 speedup 影响不到它。"
            )
        ratio = ratio_from_result(doc)
        if ratio is None:
            skipped += 1
            continue
        ratios.append(ratio)
    if skipped:
        print(
            f"提示: {skipped} 份结果缺少 comparison.speedup 且无法从 time_us 现算，已跳过",
            file=sys.stderr,
        )
    return ratios


def build_parser() -> argparse.ArgumentParser:
    """构造命令行。独立出来是为了让文档命令门禁能拿到它 ——
    门禁拿真实 parser 去解析文档里的示例，parser 藏在 main 里就够不着。
    """
    parser = argparse.ArgumentParser(
        description="量出噪声底，而不是沿用论文里的经验值",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-results", nargs="+", type=Path, metavar="JSON",
                        help="若干份 evaluation_results.json")
    source.add_argument("--ratios", nargs="+", type=float, metavar="R",
                        help="直接给出重复测得的增益比值")
    parser.add_argument("--method", choices=["auto", "cv", "spread"], default="auto",
                        help="auto（默认）优先用结果里的 cv_pct，取不到才退回比值离散")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="以 JSON 输出，供脚本消费")
    parser.add_argument("--output", type=Path, default=None,
                        help="同时写入指定文件")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.ratios:
            if args.method == "cv":
                print("错误: --ratios 模式下没有 cv_pct 可读", file=sys.stderr)
                return 2
            result = compute(list(args.ratios))
        else:
            paths = list(args.from_results)
            labels = [str(p) for p in paths]
            docs = _load_docs(paths)
            cv_values = [v for doc in docs for v in extract_cv_values(doc)]

            if args.method == "cv":
                result = compute_from_cv(cv_values)
            elif args.method == "spread":
                result = compute(_ratios_from_docs(docs, labels))
            elif cv_values:
                result = compute_from_cv(cv_values)
            else:
                print(
                    "提示: 结果中没有 cv_pct，退回用多份结果之间的比值离散估计。"
                    "注意这个估计混入了 job 之间的环境差异，比同 job 的 cv_pct 保守",
                    file=sys.stderr,
                )
                result = compute(_ratios_from_docs(docs, labels))
    except NoiseFloorError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if args.output:
        try:
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"错误: 无法写入 {args.output}: {exc}", file=sys.stderr)
            return 2

    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        method_label = {
            "cv_pct": "同 job 内变异系数（首选）",
            "ratio_spread": "多次结果之间的比值离散（回退）",
        }[result["method"]]
        print(f"测法        {method_label}")
        print(f"样本数      {result['samples']}")
        if result["method"] == "ratio_spread":
            print(f"中位数      {result['median']:.4f}")
            print(f"变异系数    {result['cv_pct']:.2f}%")
            print(f"全极差/最小 {result['spread_pct']:.2f}%")
        print(f"噪声底      {result['floor_pct']:.2f}%"
              + ("（已按硬下限抬高）" if result.get("floor_clamped") else ""))
        print()
        print(f"含义: 增益小于 {result['floor_pct']:.2f}% 的候选无法与测量抖动区分，")
        print("      按论文的接受判据应当拒绝并记为 no-go。")
        print()
        print("把这个数传给 campaign.py init --noise-floor。")
        for warning in result["warnings"]:
            print(f"\n警告: {warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
