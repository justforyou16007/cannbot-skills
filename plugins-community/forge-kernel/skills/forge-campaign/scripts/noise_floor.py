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


class NoiseFloorError(Exception):
    """数据不足以得出噪声底。"""


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

    floor_pct = max(clean)
    warnings: List[str] = []
    if floor_pct > PAPER_RANGE_PCT[1]:
        warnings.append(
            f"噪声底 {floor_pct:.2f}% 高于论文观测区间上限 {PAPER_RANGE_PCT[1]}%。"
            "测量环境可能不稳定：检查是否有其他进程占用同一张卡、"
            "是否发生 DVFS 降频（对比 msprof 的 Current Freq 与 Rated Freq）、"
            "num_trials 是否够大。在这个噪声水平下，小于它的真实改进无法被识别"
        )

    return {
        "method": "cv_pct",
        "samples": len(clean),
        "cv_values": clean,
        "floor_pct": floor_pct,
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

    # 半极差相对中位数：对偶发离群更敏感
    half_range_pct = (max(clean) - min(clean)) / 2.0 / median * 100.0

    # 取大者，保守
    floor_pct = max(cv_pct, half_range_pct)

    warnings: List[str] = []
    if floor_pct > PAPER_RANGE_PCT[1]:
        warnings.append(
            f"噪声底 {floor_pct:.2f}% 高于论文观测区间上限 {PAPER_RANGE_PCT[1]}%。"
            "测量环境可能不稳定：检查是否有其他进程占用同一张卡、"
            "是否开启了频率调节、重复次数是否够。"
            "在这个噪声水平下，小于它的真实改进无法被识别"
        )
    if floor_pct < PAPER_RANGE_PCT[0]:
        warnings.append(
            f"噪声底 {floor_pct:.2f}% 低于论文观测区间下限 {PAPER_RANGE_PCT[0]}%。"
            "环境很干净，但请确认重复测量之间确实重新执行了 kernel，"
            "而不是复用了缓存结果 —— 那会让噪声假性偏低"
        )

    return {
        "method": "ratio_spread",
        "samples": len(clean),
        "ratios": clean,
        "median": median,
        "mean": mean,
        "stdev": stdev,
        "cv_pct": cv_pct,
        "half_range_pct": half_range_pct,
        "floor_pct": floor_pct,
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
    docs: List[Dict[str, Any]] = []
    for path in paths:
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except OSError as exc:
            raise NoiseFloorError(f"无法读取 {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise NoiseFloorError(f"{path} 不是合法 JSON: {exc}") from exc
    return docs


def _ratios_from_docs(docs: List[Dict[str, Any]]) -> List[float]:
    ratios: List[float] = []
    skipped = 0
    for doc in docs:
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


def main() -> int:
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
    args = parser.parse_args()

    try:
        if args.ratios:
            if args.method == "cv":
                print("错误: --ratios 模式下没有 cv_pct 可读", file=sys.stderr)
                return 2
            result = compute(list(args.ratios))
        else:
            docs = _load_docs(list(args.from_results))
            cv_values = [v for doc in docs for v in extract_cv_values(doc)]

            if args.method == "cv":
                result = compute_from_cv(cv_values)
            elif args.method == "spread":
                result = compute(_ratios_from_docs(docs))
            elif cv_values:
                result = compute_from_cv(cv_values)
            else:
                print(
                    "提示: 结果中没有 cv_pct，退回用多份结果之间的比值离散估计。"
                    "注意这个估计混入了 job 之间的环境差异，比同 job 的 cv_pct 保守",
                    file=sys.stderr,
                )
                result = compute(_ratios_from_docs(docs))
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
            print(f"半极差      {result['half_range_pct']:.2f}%")
        print(f"噪声底      {result['floor_pct']:.2f}%")
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
