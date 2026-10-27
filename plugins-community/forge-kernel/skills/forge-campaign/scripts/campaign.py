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
"""campaign 状态机：嵌套双循环与自我提升。

ForgeKernel 跑的是两层循环，这个脚本是两层的调度中枢。

内层（coding agent）
    一次实现一行决策，单次评测判定接受或回滚。接受的条件只有一个：
    同 job 增益超过实测噪声底。没有测量支持的改动一律回滚并记为死路 ——
    「看起来更合理」不是接受的理由。

外层（arch agent）
    这是单回合 agent 没有的东西。当内层连续若干轮毫无进展、决策图走穷、
    或门禁陷入僵局时，把控制权交回架构层：重新分析、给各个手段定价、
    改课程表。关键在于它可以**重开 M0 的架构决策**，而不是在一个错的
    架构下继续硬调。论文说这正是通用 agent loop 卡住的地方 —— 架构在
    头几次尝试就定死，之后的时间再多也改不了它。

自我提升（§3.5）
    campaign 产出的 kernel 被封存成下一代的参考，重新分析、重新锚定目标，
    每阶段目标增益 +5%。源码不跨代边界，知识库跨。链条停在确认的阶段增益
    跌破噪声底的那一刻 —— 那个停点本身就是对硬件上限的一次估计。

状态持久化在 campaign_state.json，支持断点重入：上下文丢了也能从文件
恢复进度。

用法:
    python3 campaign.py init --operator add_layer_norm --arch DAV_2201 \\
        --reference-kind builtin_aclnn --reference-id aclnnAddLayerNorm \\
        --noise-floor 2.3 --target-gain 1.05 --state campaign_state.json
    python3 campaign.py status --state campaign_state.json
    python3 campaign.py record --state campaign_state.json \\
        --result evaluation_results.json --decision D0.1
    python3 campaign.py promote --state campaign_state.json --artifact ./best_kernel
    python3 campaign.py ledger --state campaign_state.json

退出码: 0 = 正常; 1 = 状态不允许该操作; 2 = 用法或 IO 错误
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"

# 状态机的五个状态
STATE_ANALYZE = "analyze"      # 架构层在分析参考 / 修订课程
STATE_IMPLEMENT = "implement"  # 编码层在实现当前决策行
STATE_STALLED = "stalled"      # 内层无进展，等待架构层重新进入
STATE_SHIPPED = "shipped"      # 本代达标，可提升为下一代参考
STATE_STOPPED = "stopped"      # 自我提升收敛，链条结束

#: 内层连续多少轮无改进就把控制权交回架构层（论文 §3.3：3 次迭代）。
DEFAULT_STAGNATION_THRESHOLD = 3

#: 自我提升每阶段的目标增益（论文 §3.5：+5%）。
DEFAULT_PHASE_TARGET_GAIN = 1.05

VERDICT_ACCEPT = "accept"
VERDICT_REVERT = "revert"
VERDICT_REGRESSION = "regression"


class CampaignError(Exception):
    """状态不允许当前操作。"""


# ---------------------------------------------------------------------------
# 判定规则（纯函数，便于测试）
# ---------------------------------------------------------------------------

def decide_candidate(gain: float, noise_floor_pct: float) -> str:
    """接受判据。

    论文 §3.3：候选被接受当且仅当它的同 job 增益超过噪声底；没有测量
    支持的改动回滚并记为死路。

    「超过」是严格的：恰好等于噪声底不算超过，因为那个增益与测量抖动
    无法区分。这里要用容差比较而不是直接 `>`，因为浮点运算会让本该相等
    的两个数差出一点点 —— 例如 (1.02-1.0)*100 实际得到 2.0000000000000018，
    直接比较就会把一个正好落在噪声底上的候选当成超过而收下。判不准的时候
    宁可拒：收错一个噪声会污染知识库，拒错一个真改进只是损失一次机会。

    返回三种判定：
      accept      增益确实超过噪声底
      revert      增益在噪声范围内，无法与抖动区分
      regression  明确变慢，超出噪声范围
    """
    if noise_floor_pct < 0:
        raise ValueError("噪声底不能为负")
    delta_pct = (gain - 1.0) * 100.0

    # 落在噪声底边界上（含浮点误差）一律按噪声处理
    if math.isclose(delta_pct, noise_floor_pct, rel_tol=1e-9, abs_tol=1e-9):
        return VERDICT_REVERT
    if math.isclose(delta_pct, -noise_floor_pct, rel_tol=1e-9, abs_tol=1e-9):
        return VERDICT_REVERT

    if delta_pct > noise_floor_pct:
        return VERDICT_ACCEPT
    if delta_pct < -noise_floor_pct:
        return VERDICT_REGRESSION
    return VERDICT_REVERT


def should_stop_promotion(gain: float, noise_floor_pct: float) -> bool:
    """自我提升的终止判据。

    论文 §3.5：链条停在确认的阶段增益跌破同 job 噪声底时。到了那一步，
    说明在当前方法下已经找不到可测量的改进 —— 停点本身是对上限的估计，
    但不构成证明：也可能只是这个循环找不到了。
    """
    return decide_candidate(gain, noise_floor_pct) != VERDICT_ACCEPT


def next_action(state: Dict[str, Any]) -> Dict[str, str]:
    """根据当前状态给出下一步动作和理由。

    理由字段很重要：它会被写进 ledger，让「为什么当时这么走」可追溯。
    """
    phase = state.get("state")

    if phase == STATE_STOPPED:
        return {"action": "stop", "reason": "自我提升已收敛，增益跌破噪声底"}
    if phase == STATE_SHIPPED:
        return {"action": "promote",
                "reason": "本代已达目标增益，可封存为下一代参考"}
    if phase == STATE_ANALYZE:
        return {"action": "analyze",
                "reason": "等待架构层产出或修订 milestone 课程"}
    if phase == STATE_STALLED:
        stag = state.get("stagnation", {})
        return {
            "action": "re-analyze",
            "reason": (
                f"内层连续 {stag.get('since_improvement', '?')} 轮无改进，"
                "交回架构层：做消融、给手段定价、必要时重开 M0 的架构决策"
            ),
        }
    return {"action": "implement",
            "reason": "继续实现当前决策行"}


# ---------------------------------------------------------------------------
# 状态读写
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state(path: Path) -> Dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CampaignError(f"无法读取状态文件 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CampaignError(f"状态文件 {path} 已损坏，不是合法 JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise CampaignError(f"状态文件 {path} 顶层必须是对象")
    return doc


def save_state(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = _now()
    # 先写临时文件再替换，避免中途失败留下半截状态
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        raise CampaignError(f"无法写入状态文件 {path}: {exc}") from exc


def new_state(operator: str, arch: str, reference: Dict[str, Any],
              noise_floor_pct: float, target_gain: float,
              stagnation_threshold: int) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "operator": operator,
        "arch": arch,
        "generation": 0,
        "reference": reference,
        "noise_floor_pct": noise_floor_pct,
        "target_gain": target_gain,
        "state": STATE_ANALYZE,
        "cursor": {"milestone": None, "decision": None, "iteration": 0},
        "stagnation": {
            "since_improvement": 0,
            "threshold": stagnation_threshold,
        },
        "best": {"gain": 1.0, "time_us": None, "artifact": None},
        "ledger": [],
        "created_at": _now(),
        "updated_at": _now(),
    }


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.state)
    if path.exists() and not args.force:
        print(f"错误: {path} 已存在；要重开 campaign 请加 --force（会丢弃现有进度）",
              file=sys.stderr)
        return 2
    if args.noise_floor < 0:
        print("错误: 噪声底不能为负", file=sys.stderr)
        return 2
    if args.noise_floor == 0:
        print(
            "警告: 噪声底为 0 意味着任何正增益都会被接受，"
            "测量抖动会被当成真实加速。请先用 noise_floor.py 量一次。",
            file=sys.stderr,
        )

    state = new_state(
        operator=args.operator,
        arch=args.arch,
        reference={
            "kind": args.reference_kind,
            "id": args.reference_id,
            "path": args.reference_path,
            "sealed": True,
            "generation": 0,
        },
        noise_floor_pct=args.noise_floor,
        target_gain=args.target_gain,
        stagnation_threshold=args.stagnation_threshold,
    )
    save_state(path, state)
    print(f"已创建 campaign: {args.operator} @ {args.arch}")
    print(f"  参考      {args.reference_kind}:{args.reference_id}（封闭，只测不读）")
    print(f"  噪声底    {args.noise_floor}%")
    print(f"  目标增益  {args.target_gain}")
    print(f"  状态文件  {path}")
    print(f"\n下一步: {next_action(state)['action']} — {next_action(state)['reason']}")
    return 0


def _extract_gain(doc: Dict[str, Any]) -> Optional[float]:
    """从评测结果里取增益。与 noise_floor.py 用同一套取数逻辑。"""
    comparison = doc.get("comparison")
    if isinstance(comparison, dict):
        speedup = comparison.get("speedup")
        if isinstance(speedup, (int, float)) and speedup > 0:
            return float(speedup)
    baseline = doc.get("baseline") or {}
    evolved = doc.get("evolved") or {}
    if isinstance(baseline, dict) and isinstance(evolved, dict):
        t_ref, t_cand = baseline.get("time_us"), evolved.get("time_us")
        if (isinstance(t_ref, (int, float)) and isinstance(t_cand, (int, float))
                and t_ref > 0 and t_cand > 0):
            return float(t_ref) / float(t_cand)
    return None


def _precision_passed(doc: Dict[str, Any]) -> Optional[bool]:
    comparison = doc.get("comparison")
    if isinstance(comparison, dict) and "precision_passed" in comparison:
        return bool(comparison["precision_passed"])
    evolved = doc.get("evolved")
    if isinstance(evolved, dict) and "precision_passed" in evolved:
        return bool(evolved["precision_passed"])
    return None


def apply_result(state: Dict[str, Any], doc: Dict[str, Any],
                 decision: str) -> Dict[str, Any]:
    """把一次评测结果并入状态，返回这次的裁决记录。

    正确性优先于性能：精度不过的候选直接回滚，连增益都不看 —— 一个
    算错的 kernel 再快也没有意义。
    """
    noise = float(state.get("noise_floor_pct", 0.0))
    entry: Dict[str, Any] = {
        "at": _now(),
        "generation": state.get("generation", 0),
        "decision": decision,
        "gain": None,
        "verdict": None,
        "reason": "",
    }

    precision = _precision_passed(doc)
    if precision is False:
        entry["verdict"] = VERDICT_REVERT
        entry["reason"] = "精度不达标，回滚；算错的 kernel 再快也无意义"
    else:
        gain = _extract_gain(doc)
        if gain is None:
            entry["verdict"] = VERDICT_REVERT
            entry["reason"] = (
                "评测结果里既无 comparison.speedup 也无法从 time_us 现算；"
                "无测量支持的改动一律回滚"
            )
        else:
            entry["gain"] = gain
            verdict = decide_candidate(gain, noise)
            entry["verdict"] = verdict
            delta = (gain - 1.0) * 100.0
            if verdict == VERDICT_ACCEPT:
                entry["reason"] = f"增益 {delta:+.2f}% 超过噪声底 {noise:.2f}%"
            elif verdict == VERDICT_REGRESSION:
                entry["reason"] = f"退化 {delta:+.2f}%，超出噪声范围，回滚"
            else:
                entry["reason"] = (
                    f"增益 {delta:+.2f}% 落在噪声底 {noise:.2f}% 以内，"
                    "无法与测量抖动区分，回滚并记为 no-go"
                )

    # 更新游标与停滞计数
    cursor = state.setdefault("cursor", {})
    cursor["decision"] = decision
    cursor["iteration"] = int(cursor.get("iteration", 0)) + 1

    stag = state.setdefault("stagnation", {
        "since_improvement": 0, "threshold": DEFAULT_STAGNATION_THRESHOLD})

    if entry["verdict"] == VERDICT_ACCEPT:
        stag["since_improvement"] = 0
        best = state.setdefault("best", {})
        if entry["gain"] is not None and entry["gain"] > float(best.get("gain", 1.0)):
            best["gain"] = entry["gain"]
            evolved = doc.get("evolved") or {}
            if isinstance(evolved, dict):
                best["time_us"] = evolved.get("time_us")
        # 达到目标增益就可以收工，论文：campaign 在达到目标增益时停止，
        # 而不是一直跑到跑不动为止
        if float(state["best"]["gain"]) >= float(state.get("target_gain", 1.0)):
            state["state"] = STATE_SHIPPED
        else:
            state["state"] = STATE_IMPLEMENT
    else:
        stag["since_improvement"] = int(stag.get("since_improvement", 0)) + 1
        threshold = int(stag.get("threshold", DEFAULT_STAGNATION_THRESHOLD))
        if stag["since_improvement"] >= threshold:
            state["state"] = STATE_STALLED
        else:
            state["state"] = STATE_IMPLEMENT

    state.setdefault("ledger", []).append(entry)
    return entry


def cmd_record(args: argparse.Namespace) -> int:
    path = Path(args.state)
    try:
        state = load_state(path)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if state.get("state") == STATE_STOPPED:
        print("错误: campaign 已收敛结束，不接受新结果", file=sys.stderr)
        return 1

    try:
        doc = json.loads(Path(args.result).read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"错误: 无法读取评测结果 {args.result}: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"错误: {args.result} 不是合法 JSON: {exc}", file=sys.stderr)
        return 2

    entry = apply_result(state, doc, args.decision)
    save_state(path, state)

    print(f"决策 {args.decision}: {entry['verdict']}")
    print(f"  {entry['reason']}")
    action = next_action(state)
    print(f"\n下一步: {action['action']} — {action['reason']}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """把本代产出封存为下一代的参考。

    论文 §3.5：参考只被当作可测量的二进制使用，所以 ForgeKernel 自己
    产出的 kernel 完全可以充当这个角色。提升抬高了性能锚点，知识库则
    降低了够到它的成本 —— 两条反馈边不重复。
    """
    path = Path(args.state)
    try:
        state = load_state(path)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if state.get("state") not in (STATE_SHIPPED, STATE_STALLED, STATE_IMPLEMENT):
        print(f"错误: 当前状态 '{state.get('state')}' 不允许提升", file=sys.stderr)
        return 1

    gain = float(state.get("best", {}).get("gain", 1.0))
    noise = float(state.get("noise_floor_pct", 0.0))

    if should_stop_promotion(gain, noise) and not args.force:
        state["state"] = STATE_STOPPED
        state.setdefault("ledger", []).append({
            "at": _now(),
            "generation": state.get("generation", 0),
            "decision": None,
            "gain": gain,
            "verdict": "stop",
            "reason": (
                f"本代增益 {(gain - 1) * 100:+.2f}% 未超过噪声底 {noise:.2f}%，"
                "自我提升链条终止。该停点是对硬件上限的一次估计，"
                "但不构成证明 —— 也可能只是这个循环找不到了"
            ),
        })
        save_state(path, state)
        print("自我提升收敛：本代增益未超过噪声底，链条终止。")
        print("这个停点是对上限的估计，不是证明。如确需继续请加 --force。")
        return 0

    generation = int(state.get("generation", 0)) + 1
    state["generation"] = generation
    state["reference"] = {
        "kind": "sealed_kernel",
        "id": f"{state.get('operator')}_gen{generation - 1}",
        "path": args.artifact,
        "sealed": True,
        "generation": generation - 1,
        "promoted_gain": gain,
    }
    # 新一代重新开始：课程表要对新参考重新分析，目标按阶段增益重设
    state["state"] = STATE_ANALYZE
    state["cursor"] = {"milestone": None, "decision": None, "iteration": 0}
    state["stagnation"]["since_improvement"] = 0
    state["target_gain"] = args.phase_target_gain
    state["best"] = {"gain": 1.0, "time_us": None, "artifact": None}
    state.setdefault("ledger", []).append({
        "at": _now(),
        "generation": generation,
        "decision": None,
        "gain": gain,
        "verdict": "promote",
        "reason": (
            f"第 {generation - 1} 代产出封存为第 {generation} 代的参考；"
            f"源码不跨代边界，知识库跨。新阶段目标增益 {args.phase_target_gain}"
        ),
    })
    save_state(path, state)

    print(f"已提升到第 {generation} 代")
    print(f"  新参考    {state['reference']['id']}（封闭）")
    print(f"  上代增益  {(gain - 1) * 100:+.2f}%")
    print(f"  新目标    {args.phase_target_gain}")
    print("\n下一步: analyze — 对新参考重读签名、重锚目标与机制行")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    path = Path(args.state)
    try:
        state = load_state(path)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    action = next_action(state)
    if args.as_json:
        print(json.dumps({"state": state, "next": action},
                         ensure_ascii=False, indent=2))
        return 0

    best = state.get("best", {})
    stag = state.get("stagnation", {})
    ref = state.get("reference", {})
    print(f"算子      {state.get('operator')} @ {state.get('arch')}")
    print(f"代次      第 {state.get('generation')} 代")
    print(f"参考      {ref.get('kind')}:{ref.get('id')}")
    print(f"状态      {state.get('state')}")
    print(f"噪声底    {state.get('noise_floor_pct')}%")
    print(f"目标增益  {state.get('target_gain')}")
    print(f"当前最佳  {float(best.get('gain', 1.0)):.4f}"
          f"（{(float(best.get('gain', 1.0)) - 1) * 100:+.2f}%）")
    print(f"停滞计数  {stag.get('since_improvement')}/{stag.get('threshold')}")
    print(f"台账条目  {len(state.get('ledger', []))}")
    print(f"\n下一步: {action['action']} — {action['reason']}")
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    """台账：每次裁决和它的理由。

    论文要求 campaign ledger 同时记录本次发明的和外来的经验，让知识库
    的贡献可审计而不是口说无凭。
    """
    path = Path(args.state)
    try:
        state = load_state(path)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    entries: List[Dict[str, Any]] = state.get("ledger", [])
    if args.as_json:
        print(json.dumps(entries, ensure_ascii=False, indent=2))
        return 0
    if not entries:
        print("台账为空")
        return 0
    for entry in entries:
        gain = entry.get("gain")
        gain_text = f"{(gain - 1) * 100:+.2f}%" if isinstance(gain, (int, float)) else "—"
        decision = entry.get("decision") or "—"
        print(f"[gen{entry.get('generation')}] {decision:<8} "
              f"{str(entry.get('verdict')):<11} {gain_text:>9}  {entry.get('reason')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ForgeKernel campaign 状态机：嵌套双循环与自我提升",
    )
    parser.add_argument("--state", default="campaign_state.json",
                        help="状态文件路径（默认 campaign_state.json）")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="以 JSON 输出，供脚本消费")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="创建 campaign")
    p_init.add_argument("--operator", required=True)
    p_init.add_argument("--arch", required=True)
    p_init.add_argument("--reference-kind", required=True,
                        choices=["builtin_aclnn", "opp_install", "sealed_kernel"])
    p_init.add_argument("--reference-id", required=True)
    p_init.add_argument("--reference-path", default=None)
    p_init.add_argument("--noise-floor", type=float, required=True,
                        help="实测噪声底百分比，用 noise_floor.py 量")
    p_init.add_argument("--target-gain", type=float, default=DEFAULT_PHASE_TARGET_GAIN)
    p_init.add_argument("--stagnation-threshold", type=int,
                        default=DEFAULT_STAGNATION_THRESHOLD,
                        help="内层连续多少轮无改进交回架构层")
    p_init.add_argument("--force", action="store_true", help="覆盖已有状态文件")
    p_init.set_defaults(func=cmd_init)

    p_status = sub.add_parser("status", help="查看进度与下一步")
    p_status.set_defaults(func=cmd_status)

    p_record = sub.add_parser("record", help="并入一次评测结果并裁决")
    p_record.add_argument("--result", required=True,
                          help="evaluation_results.json 路径")
    p_record.add_argument("--decision", required=True,
                          help="本次实现的决策行 id，如 D0.1")
    p_record.set_defaults(func=cmd_record)

    p_promote = sub.add_parser("promote", help="封存本代产出为下一代参考")
    p_promote.add_argument("--artifact", required=True, help="本代产出 kernel 路径")
    p_promote.add_argument("--phase-target-gain", type=float,
                           default=DEFAULT_PHASE_TARGET_GAIN)
    p_promote.add_argument("--force", action="store_true",
                           help="即使增益未超噪声底也强行提升")
    p_promote.set_defaults(func=cmd_promote)

    p_ledger = sub.add_parser("ledger", help="查看裁决台账")
    p_ledger.set_defaults(func=cmd_ledger)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
