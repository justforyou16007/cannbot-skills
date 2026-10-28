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
"""campaign 状态机：嵌套双循环、机制裁决与自我提升。

ForgeKernel 跑的是两层循环，这个脚本是两层的调度中枢。

内层（coding agent）
    一次实现一行决策。一次 GPU job 的裁决由三部分组成：独立的正确性
    检查、与参考的配对计时、**以及 profiling**。三者缺一不可 ——
    profiling 掉出裁决回路，巧合加速就会被当成真机制收编。

外层（arch agent）
    单回合 agent 没有的东西。三个重入口：连续若干轮没有新下界、
    **决策图走穷**、**门禁僵持**。它能重开 M0 的架构决策，而不是在一个
    已经错了的架构下继续硬调 —— 论文测出来，通用 agent loop 的天花板
    正卡在这里。

自我提升（§3.5）
    被接受的 kernel 封存成下一代参考。源码不跨代边界，知识库跨。
    链条停在确认的阶段增益跌破噪声底时。

三件事值得单独说明
------------------

**机制行进裁决。** 增益超过噪声底只说明「变快了」，不说明「我设计的
机制落地了」。两者在总时间上长得一模一样，但后者无法复用、无法解释、
换个算子就不灵。所以 record 接收 signature_probe 的输出与 milestone，
按机制行的停顿锚点核验；核验不过的降为 accept_unverified —— **保留但
不计入 best，也不得作为知识库入库依据**。论文 §3.4 专门讲这件事：
一次 campaign 达成目标，不等于它当中提出的每个解释都成立。

**停滞只数「测出来了但不够」。** 精度不过、评测缺失是实现问题不是架构
问题。把它们计进停滞，连续三个编译 bug 就会白白消耗一次架构重入机会。
两者分开计数、分开提示。

**foundation 补丁按依赖图重启。** 重开 M0 的某个决策时，所有（直接或
间接）依赖它的下游实现都建立在旧假设上，必须作废。靠 agent 自己记必然
会漏，而漏掉一条，新 M0 上就还挂着旧假设的实现 —— 归因直接错掉，
且恰恰错在整个流程最关键的那一步。reopen 子命令按 depends_on 算反向
闭包，自动作废。

用法:
    campaign.py init --state campaign_state.json \\
        --operator add_layer_norm --input-shape "B=8,S=4096,H=1024,fp16" \\
        --arch DAV_2201 --reference-kind builtin_aclnn \\
        --reference-id aclnnAddLayerNorm \\
        --noise-floor 2.3 --target-gain 1.05 --budget 40

    campaign.py record --state campaign_state.json \\
        --result evaluation_results.json --decision D0.1 \\
        --signature architecture_signature.json --milestone milestone.json \\
        --recipe R-2201-001 --symptom "MTE2 占比过高"

    campaign.py reopen --state campaign_state.json \\
        --milestone milestone.json --decision D0.1 --reason "拓扑决策有缺陷"

    campaign.py set-noise-floor --state campaign_state.json --value 2.7
    campaign.py promote --state campaign_state.json --artifact ./best
    campaign.py ledger --state campaign_state.json --recipes

`--state` 与 `--json` 写在子命令前后都可以，两种形态都能跑。

退出码: 0 = 正常; 1 = 状态不允许该操作; 2 = 用法或 IO 错误
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SCHEMA_VERSION = "2.0"

# 状态机
STATE_ANALYZE = "analyze"      # 架构层在分析参考 / 修订课程
STATE_IMPLEMENT = "implement"  # 编码层在实现当前决策行
STATE_STALLED = "stalled"      # 需要架构层重新进入
STATE_SHIPPED = "shipped"      # 本代达标，可提升为下一代参考
STATE_STOPPED = "stopped"      # 自我提升收敛或预算耗尽

#: 外层重入的口子（论文 §3.3 给了前三个；第四个是本实现补的，见下）。
STALL_NO_IMPROVEMENT = "no_improvement"
STALL_GRAPH_EXHAUSTED = "graph_exhausted"
STALL_GATE_IMPASSE = "gate_impasse"

#: 第四个口子：机制始终没被核验过。
#:
#: 论文没有这一条，因为在它的实现里计时与 profiling 在同一个 job 里完成，
#: 不存在「忘了采 profiling」这种状态。昇腾上两者必须分两次跑
#: （acl.prof.init 每进程只能调一次），于是多出一种失效方式：
#: 一直不传 --signature，每轮都判 accept_unverified —— 它不计入 best
#: 所以永远到不了 shipped，又会清零停滞计数所以永远不转 stalled，
#: 循环就这么一直跑下去，而 status 每轮都说「继续实现」。
#:
#: 每轮确实打印了提示，但连跑的 agent 不会因为一行提示改变行为 ——
#: 它看到的是「接受了，继续」。所以要让这件事走和另外三个口子同一条通路，
#: 架构层才看得见。
STALL_MECHANISM_UNVERIFIED = "mechanism_never_verified"

#: 内层连续多少轮「测出来了但增益不够」才交回架构层。
DEFAULT_STAGNATION_THRESHOLD = 3

#: 实现类失败（精度不过、评测缺失）连续多少次提示人工介入。
#: 它**不**触发架构重入 —— 那是实现问题，不是架构问题。
DEFAULT_IMPL_FAILURE_THRESHOLD = 3

#: 连续多少次 accept_unverified 判定为「profiling 没接进回路」。
DEFAULT_UNVERIFIED_THRESHOLD = 3

#: 自我提升每阶段的目标增益（论文 §3.5：+5%）。
DEFAULT_PHASE_TARGET_GAIN = 1.05

VERDICT_ACCEPT = "accept"
VERDICT_ACCEPT_UNVERIFIED = "accept_unverified"
VERDICT_REVERT = "revert"
VERDICT_REGRESSION = "regression"

#: 对候选的裁决。台账里还有 reopen / promote / stop / noise_floor 这类
#: **管理条目**，它们不代表「某条决策被试过」，统计时必须排除。
CANDIDATE_VERDICTS = frozenset({
    VERDICT_ACCEPT, VERDICT_ACCEPT_UNVERIFIED,
    VERDICT_REVERT, VERDICT_REGRESSION,
})

#: 噪声底状态。参考换了之后必须重测 —— 新参考是另一个二进制，
#: 运行时特征不同，而噪声底论文定义的是「同 job 噪声底」。
NOISE_MEASURED = "measured"
NOISE_PENDING = "pending_remeasure"


class CampaignError(Exception):
    """状态不允许当前操作。"""


# ---------------------------------------------------------------------------
# 判定规则（纯函数，便于测试）
# ---------------------------------------------------------------------------

def decide_candidate(gain: float, noise_floor_pct: float) -> str:
    """增益判据：这次改动在性能上算不算数。

    论文 §3.3：候选被接受当且仅当它的同 job 增益超过噪声底。

    「超过」是严格的：恰好等于噪声底不算超过，因为那个增益与测量抖动
    无法区分。这里要用容差比较而不是直接 `>`，因为浮点运算会让本该相等
    的两个数差出一点点 —— 例如 (1.02-1.0)*100 实际得到 2.0000000000000018，
    直接比较就会把一个正好落在噪声底上的候选当成超过而收下。判不准的时候
    宁可拒：收错一个噪声会污染知识库，拒错一个真改进只是损失一次机会。

    注意这只是**增益**判据。最终裁决还要过机制核验，见 apply_result。
    """
    if noise_floor_pct < 0:
        raise ValueError("噪声底不能为负")
    delta_pct = (gain - 1.0) * 100.0

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
    """自我提升的终止判据（论文 §3.5）。

    链条停在确认的阶段增益跌破同 job 噪声底时。那个停点是对硬件上限的
    一次估计，**不是证明** —— 也可能只是这个循环找不到了。
    """
    return decide_candidate(gain, noise_floor_pct) != VERDICT_ACCEPT


# ---------------------------------------------------------------------------
# 机制核验：把 profiling 接进裁决回路
# ---------------------------------------------------------------------------

#: 机制行锚点表达式的形状：<通道> <比较符> <数值>
RE_MECH_EXPR = re.compile(
    r"(?P<channel>[\w:.()%/\-]+?)\s*(?P<op><=|>=|==|!=|<|>)\s*(?P<value>-?\d+(?:\.\d+)?)"
)

_COMPARATORS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9),
    "!=": lambda a, b: not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9),
}


def lookup_signature_value(signature: Dict[str, Any],
                           needle: str) -> Tuple[Optional[float], Optional[str]]:
    """在架构签名里按通道名找值。

    signature_probe 给每个字段都带上了它的来源通道，所以这里递归找
    「channel 里含有该字段名且 present 为真」的那一项即可，不需要在两边
    维护一份字段名映射表 —— 那种表一定会漂移。

    返回 (值, 实际命中的通道)。找不到返回 (None, None)；
    **找不到与值为 0 必须分开**，理由同 signature_probe 的设计。
    """
    found: List[Tuple[float, str]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            channel = node.get("channel")
            value = node.get("value")
            if (isinstance(channel, str) and needle in channel
                    and node.get("present") is True
                    and isinstance(value, (int, float))):
                found.append((float(value), channel))
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(signature)
    if found:
        return found[0]

    # exposure 段是推导量，不走 field() 的形状，单独找一次
    exposure = signature.get("exposure")
    if isinstance(exposure, dict) and exposure.get("available"):
        value = exposure.get(needle)
        if isinstance(value, (int, float)):
            return float(value), f"derived:exposure.{needle}"
    return None, None


def verify_mechanism_line(line: Dict[str, Any],
                          signature: Dict[str, Any]) -> Dict[str, Any]:
    """核验一条机制行的锚点在本次 signature 上是否成立。

    三种结果，**不要混淆后两种**：
      True   表达式求值为真，机制确实落地
      False  求值为假，机制没有落地 —— 那么这次的增益来自别处
      None   无法求值（字段没采到、表达式写法不认识）。这不等于失败，
             但也不构成验证，所以同样不足以支撑 accept
    """
    line_id = line.get("id", "<无 id>")
    label = line.get("label")
    anchor = line.get("anchor") or {}
    expr = anchor.get("expr") if isinstance(anchor, dict) else None

    out: Dict[str, Any] = {"id": line_id, "label": label, "expr": expr}

    if not isinstance(expr, str) or not expr.strip():
        out.update(verified=None, reason="机制行没有可求值的锚点表达式")
        return out

    match = RE_MECH_EXPR.search(expr)
    if not match:
        out.update(verified=None,
                   reason=f"锚点表达式 {expr!r} 不是「通道 比较符 数值」的形状，无法求值")
        return out

    channel = match.group("channel")
    field = channel.rsplit(":", 1)[-1].strip()
    threshold = float(match.group("value"))
    op = match.group("op")

    actual, hit_channel = lookup_signature_value(signature, field)
    if actual is None:
        out.update(verified=None,
                   reason=(f"签名里没有 {field} 的有效读数。"
                           "缺失不等于不成立 —— 但也不构成验证"))
        return out

    ok = _COMPARATORS[op](actual, threshold)
    out.update(
        verified=bool(ok), actual=actual, threshold=threshold,
        operator=op, channel=hit_channel,
        reason=(f"{field} 实测 {actual} {op} {threshold} "
                f"{'成立' if ok else '不成立'}"),
    )
    return out


def collect_mechanism_lines(milestone_doc: Dict[str, Any],
                            decision_id: str) -> List[Dict[str, Any]]:
    """取出该决策所属 milestone 的机制行。

    机制行挂在 milestone 的门禁上而不是单条决策上，所以先按决策 id
    定位到它所在的 milestone。
    """
    for milestone in milestone_doc.get("milestones", []) or []:
        if not isinstance(milestone, dict):
            continue
        ids = {d.get("id") for d in milestone.get("decisions", []) or []
               if isinstance(d, dict)}
        if decision_id in ids:
            gate = milestone.get("gate") or {}
            lines = gate.get("mechanism_lines") if isinstance(gate, dict) else None
            return [ln for ln in (lines or []) if isinstance(ln, dict)]
    return []


def judge_mechanism(milestone_doc: Optional[Dict[str, Any]],
                    signature: Optional[Dict[str, Any]],
                    decision_id: str) -> Dict[str, Any]:
    """机制裁决：本次改动有没有把它声称的机制做出来。

    没给 milestone 或 signature 时返回 unknown —— 工具不替调用方假设
    「没查就是通过」。
    """
    if milestone_doc is None or signature is None:
        return {
            "status": "unknown",
            "lines": [],
            "reason": ("未提供 --milestone 与 --signature，机制未经核验。"
                       "增益超过噪声底只说明变快了，不说明设计的机制落地了"),
        }

    if signature.get("trust", {}).get("trustworthy") is False:
        return {
            "status": "unknown", "lines": [],
            "reason": ("架构签名自判测量不可信（降频 / ICache 颠簸 / "
                       "停顿数据缺失），据此核验机制没有意义"),
        }

    lines = collect_mechanism_lines(milestone_doc, decision_id)
    if not lines:
        return {
            "status": "unknown", "lines": [],
            "reason": f"milestone 里找不到决策 {decision_id} 所属的机制行",
        }

    results = [verify_mechanism_line(line, signature) for line in lines]
    if all(r["verified"] is True for r in results):
        status = "verified"
        reason = f"{len(results)} 条机制行全部成立"
    elif any(r["verified"] is False for r in results):
        status = "refuted"
        failed = [r["id"] for r in results if r["verified"] is False]
        reason = (f"机制行 {failed} 不成立 —— 这次的增益来自别处，"
                  "不能把它附带的解释当成已验证的机制")
    else:
        status = "unknown"
        pending = [r["id"] for r in results if r["verified"] is None]
        reason = f"机制行 {pending} 无法求值，未构成验证"

    return {"status": status, "lines": results, "reason": reason}


# ---------------------------------------------------------------------------
# 决策图：重启范围、走穷与僵持
# ---------------------------------------------------------------------------

def load_milestone(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CampaignError(f"无法读取 milestone {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CampaignError(f"milestone {path} 不是合法 JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise CampaignError(f"milestone {path} 顶层必须是对象")
    return doc


def all_decisions(milestone_doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for milestone in milestone_doc.get("milestones", []) or []:
        if not isinstance(milestone, dict):
            continue
        for decision in milestone.get("decisions", []) or []:
            if isinstance(decision, dict) and decision.get("id"):
                out[str(decision["id"])] = decision
    return out


def dependents_closure(milestone_doc: Dict[str, Any], root: str) -> Set[str]:
    """算出（直接或间接）依赖 root 的所有决策。

    论文 §3.3：追加型补丁原地恢复，**基础型补丁从最早的依赖决策重启**。
    重开一个决策时，建立在它之上的下游实现都基于旧假设，必须一并作废 ——
    漏掉一条，新架构上就还挂着旧假设的实现，归因会错，而且恰恰错在整个
    流程最关键的那一步。

    V3 规则已经保证依赖图无环，所以这里不必防环；但仍用 visited 兜底，
    免得一份未经校验的 milestone 把这里转死。
    """
    decisions = all_decisions(milestone_doc)
    if root not in decisions:
        raise CampaignError(
            f"milestone 里没有决策 {root}；已有: {sorted(decisions)}")

    reverse: Dict[str, List[str]] = {did: [] for did in decisions}
    for did, decision in decisions.items():
        for dep in decision.get("depends_on") or []:
            if isinstance(dep, str) and dep in reverse:
                reverse[dep].append(did)

    out: Set[str] = set()
    stack = [root]
    visited = {root}
    while stack:
        node = stack.pop()
        for child in reverse.get(node, []):
            if child not in visited:
                visited.add(child)
                out.add(child)
                stack.append(child)
    return out


def graph_exhausted(state: Dict[str, Any],
                    milestone_doc: Optional[Dict[str, Any]]) -> bool:
    """决策图是否走穷：每条决策都试过，但还没达标。

    这是论文给的第二个外层重入口。少了它，决策图走完而没达标时状态会
    停在 implement，没有人提示该回架构层。

    「试过」只算对候选的裁决。台账里还有 reopen / promote / stop /
    noise_floor 这类管理条目，其中 reopen 带着被重开那条决策的 id ——
    若按 id 一概算作试过，刚重开一个叶子决策就会被立刻误判为走穷，
    于是架构层刚做完重开，状态机又让它回去重新分析，空转一轮。
    """
    if milestone_doc is None:
        return False
    decisions = set(all_decisions(milestone_doc))
    if not decisions:
        return False
    tried = {
        entry.get("decision") for entry in state.get("ledger", [])
        if entry.get("decision")
        and not entry.get("invalidated")
        and entry.get("verdict") in CANDIDATE_VERDICTS
    }
    return decisions.issubset(tried)


def gate_impasse(milestone_doc: Optional[Dict[str, Any]]) -> Optional[str]:
    """门禁是否僵持：有 OPEN 行，或 DEFERRED 行指向的下游手段不存在。

    这是第三个外层重入口。DEFERRED 把未解的测试交给一个具名的下游手段，
    若那个手段在决策图里根本不存在，这个测试实际是被丢弃了 —— 门禁就此
    悬空，没人会再去碰它。
    """
    if milestone_doc is None:
        return None
    decisions = set(all_decisions(milestone_doc))

    for milestone in milestone_doc.get("milestones", []) or []:
        if not isinstance(milestone, dict):
            continue
        mid = milestone.get("id", "?")
        gate = milestone.get("gate") or {}
        if not isinstance(gate, dict):
            continue
        for category in ("correctness_lines", "performance_lines", "mechanism_lines"):
            for line in gate.get(category, []) or []:
                if not isinstance(line, dict):
                    continue
                status = line.get("status")
                lid = line.get("id", "?")
                if status == "OPEN":
                    return (f"{mid}.{category}.{lid} 仍为 OPEN，"
                            "门禁无法关闭且无人接手")
                if status == "DEFERRED":
                    lever = line.get("downstream_lever")
                    if lever and lever not in decisions:
                        return (f"{mid}.{category}.{lid} 推迟给了 {lever}，"
                                "但决策图里没有这个手段 —— 该测试实际被丢弃了")
    return None


# ---------------------------------------------------------------------------
# 下一步动作
# ---------------------------------------------------------------------------

def next_action(state: Dict[str, Any]) -> Dict[str, str]:
    """根据当前状态给出下一步动作和理由。

    理由会被写进 ledger，让「为什么当时这么走」可追溯。
    """
    phase = state.get("state")

    if phase == STATE_STOPPED:
        return {"action": "stop",
                "reason": state.get("stop_reason", "campaign 已结束")}
    if phase == STATE_SHIPPED:
        return {"action": "promote",
                "reason": "本代已达目标增益，可封存为下一代参考"}
    if phase == STATE_ANALYZE:
        if state.get("noise_floor_status") == NOISE_PENDING:
            return {"action": "measure-noise-floor",
                    "reason": ("参考已更换，噪声底必须对新参考重测。"
                               "新参考是另一个二进制，运行时特征不同")}
        return {"action": "analyze",
                "reason": "等待架构层产出或修订 milestone 课程"}
    if phase == STATE_STALLED:
        reasons = state.get("stall_reasons") or [{
            "kind": state.get("stall_reason", STALL_NO_IMPROVEMENT),
            "detail": state.get("stall_detail", ""),
        }]
        titles = {
            STALL_NO_IMPROVEMENT: "内层无进展",
            STALL_GRAPH_EXHAUSTED: "决策图走穷",
            STALL_GATE_IMPASSE: "门禁僵持",
            STALL_MECHANISM_UNVERIFIED: "机制从未核实",
        }
        listed = "；".join(
            f"{titles.get(r['kind'], r['kind'])}（{r['detail']}）"
            for r in reasons)
        return {
            "action": "re-analyze",
            "reason": (f"{listed}。交回架构层：做消融定价、必要时用 reopen "
                       "重开 M0 的架构决策，而不是在当前架构下继续硬调"),
        }
    return {"action": "implement", "reason": "继续实现当前决策行"}


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        raise CampaignError(f"无法写入状态文件 {path}: {exc}") from exc


def new_state(operator: str, input_shape: str, arch: str,
              reference: Dict[str, Any], noise_floor_pct: float,
              target_gain: float, stagnation_threshold: int,
              budget: Optional[int] = None) -> Dict[str, Any]:
    """建立 campaign 状态。

    论文 §3.1：一次 campaign 取一个算子、**一个输入形状**、一个封存的
    参考和一个目标增益。input_shape 不是可选项 —— 噪声底、架构签名、
    门禁锚点全都是 shape 相关的，同一个 tiling 决策在不同 shape 下的
    trigger 完全不同。一份 state 跨 shape 复用会把不同 shape 的结果混进
    同一个 best，promote 时封存的参考与锚定的目标就对不上了。
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "operator": operator,
        "input_shape": input_shape,
        "arch": arch,
        "generation": 0,
        "reference": reference,
        "noise_floor_pct": noise_floor_pct,
        "noise_floor_status": NOISE_MEASURED,
        "target_gain": target_gain,
        "budget": {"max_records": budget, "used": 0},
        "state": STATE_ANALYZE,
        "cursor": {"milestone": None, "decision": None, "iteration": 0},
        "stagnation": {
            "since_improvement": 0,
            "threshold": stagnation_threshold,
        },
        "implementation_failures": {
            "consecutive": 0,
            "threshold": DEFAULT_IMPL_FAILURE_THRESHOLD,
        },
        "mechanism_unverified": {
            "consecutive": 0,
            "threshold": DEFAULT_UNVERIFIED_THRESHOLD,
        },
        "best": {"gain": 1.0, "time_us": None, "artifact": None},
        "ledger": [],
        "created_at": _now(),
        "updated_at": _now(),
    }


# ---------------------------------------------------------------------------
# 取数
# ---------------------------------------------------------------------------

def _extract_gain(doc: Dict[str, Any]) -> Optional[float]:
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


# ---------------------------------------------------------------------------
# 裁决
# ---------------------------------------------------------------------------

def apply_result(state: Dict[str, Any], doc: Dict[str, Any], decision: str,
                 milestone_doc: Optional[Dict[str, Any]] = None,
                 signature: Optional[Dict[str, Any]] = None,
                 recipes_used: Optional[List[str]] = None,
                 recipes_invented: Optional[List[str]] = None,
                 symptom: str = "") -> Dict[str, Any]:
    """把一次评测结果并入状态，返回裁决记录。

    裁决顺序刻意如此：

      1. 正确性优先于一切 —— 算错的 kernel 再快也没有意义
      2. 没有测量支持的改动一律回滚 —— agent 声称有效不算数
      3. 增益判据 —— 超过噪声底才算变快
      4. **机制核验** —— 变快了还要确认是设计的机制起的作用

    第 4 步是这份实现与「只看增益」的关键差别。论文 §3.4 说认证把真正
    起作用的机制和只是伴随发生的分开；一次 campaign 达成目标，不等于它
    当中提出的每个解释都成立。机制没核实的改动仍然保留（它确实更快），
    但**不计入 best、不得作为入库依据** —— 否则一个巧合会被写进知识库，
    再被其他算子照着做决策。
    """
    noise = float(state.get("noise_floor_pct", 0.0))
    evolved = doc.get("evolved") or {}
    evolved_time = (evolved.get("time_us")
                    if isinstance(evolved, dict) else None)
    entry: Dict[str, Any] = {
        "at": _now(),
        "generation": state.get("generation", 0),
        "input_shape": state.get("input_shape"),
        "decision": decision,
        "gain": None,
        # 台账要能自给自足地重算 best —— reopen 作废若干条目之后，
        # best 必须从剩下的条目里恢复，包括 time_us。
        "time_us": evolved_time if isinstance(evolved_time, (int, float)) else None,
        "verdict": None,
        "reason": "",
        "mechanism": None,
        "recipes_used": sorted(set(recipes_used or [])),
        "recipes_invented": sorted(set(recipes_invented or [])),
        "counts_toward_best": False,
        "kb_admissible": False,
        "symptom": symptom,
    }

    precision = _precision_passed(doc)
    measured = False

    if precision is False:
        entry["verdict"] = VERDICT_REVERT
        entry["reason"] = "精度不达标，回滚；算错的 kernel 再快也无意义"
        entry["failure_kind"] = "precision"
    else:
        gain = _extract_gain(doc)
        if gain is None:
            entry["verdict"] = VERDICT_REVERT
            entry["reason"] = (
                "评测结果里既无 comparison.speedup 也无法从 time_us 现算；"
                "无测量支持的改动一律回滚")
            entry["failure_kind"] = "no_measurement"
        else:
            measured = True
            entry["gain"] = gain
            gain_verdict = decide_candidate(gain, noise)
            delta = (gain - 1.0) * 100.0

            if gain_verdict == VERDICT_ACCEPT:
                mechanism = judge_mechanism(milestone_doc, signature, decision)
                entry["mechanism"] = mechanism
                if mechanism["status"] == "verified":
                    entry["verdict"] = VERDICT_ACCEPT
                    entry["counts_toward_best"] = True
                    entry["kb_admissible"] = True
                    entry["reason"] = (
                        f"增益 {delta:+.2f}% 超过噪声底 {noise:.2f}%，"
                        f"且机制已核实（{mechanism['reason']}）")
                else:
                    entry["verdict"] = VERDICT_ACCEPT_UNVERIFIED
                    entry["reason"] = (
                        f"增益 {delta:+.2f}% 超过噪声底 {noise:.2f}%，"
                        f"但机制未核实：{mechanism['reason']}。"
                        "保留该改动，但不计入 best，也不得作为入库依据 —— "
                        "变快了不等于设计的机制落地了")
            elif gain_verdict == VERDICT_REGRESSION:
                entry["verdict"] = VERDICT_REGRESSION
                entry["reason"] = f"退化 {delta:+.2f}%，超出噪声范围，回滚"
            else:
                entry["verdict"] = VERDICT_REVERT
                entry["reason"] = (
                    f"增益 {delta:+.2f}% 落在噪声底 {noise:.2f}% 以内，"
                    "无法与测量抖动区分，回滚并记为 no-go")

    # 游标
    cursor = state.setdefault("cursor", {})
    cursor["decision"] = decision
    cursor["iteration"] = int(cursor.get("iteration", 0)) + 1

    # 预算
    budget = state.setdefault("budget", {"max_records": None, "used": 0})
    budget["used"] = int(budget.get("used", 0)) + 1

    stag = state.setdefault("stagnation", {
        "since_improvement": 0, "threshold": DEFAULT_STAGNATION_THRESHOLD})
    impl = state.setdefault("implementation_failures", {
        "consecutive": 0, "threshold": DEFAULT_IMPL_FAILURE_THRESHOLD})
    unver = state.setdefault("mechanism_unverified", {
        "consecutive": 0, "threshold": DEFAULT_UNVERIFIED_THRESHOLD})

    if entry["verdict"] in (VERDICT_ACCEPT, VERDICT_ACCEPT_UNVERIFIED):
        stag["since_improvement"] = 0
        impl["consecutive"] = 0
        if entry["verdict"] == VERDICT_ACCEPT:
            unver["consecutive"] = 0
        else:
            # 连续若干次「变快了但说不清为什么」= profiling 没接进回路
            unver["consecutive"] = int(unver.get("consecutive", 0)) + 1
        # 只有机制核实过的改动才抬高 best。未核实的保留但不计入 ——
        # 否则 promote 时会把一个解释不清的结果封成下一代的参考。
        if entry["counts_toward_best"] and entry["gain"] is not None:
            best = state.setdefault("best", {})
            if entry["gain"] > float(best.get("gain", 1.0)):
                best["gain"] = entry["gain"]
                best["time_us"] = entry.get("time_us")
    elif measured:
        # 测出来了但不够 —— 这才是架构层该关心的停滞
        stag["since_improvement"] = int(stag.get("since_improvement", 0)) + 1
        impl["consecutive"] = 0
    else:
        # 精度不过 / 评测缺失是实现问题，不消耗架构重入机会
        impl["consecutive"] = int(impl.get("consecutive", 0)) + 1

    state.setdefault("ledger", []).append(entry)
    _advance_state(state, milestone_doc)
    return entry


def _advance_state(state: Dict[str, Any],
                   milestone_doc: Optional[Dict[str, Any]]) -> None:
    """按本轮结果决定状态机往哪走。

    达标与预算先判，因为它们是终止条件：论文说 campaign 在达到目标增益时
    停止，而不是一直跑到跑不动为止。

    三个外层重入口**可能同时成立**，所以全部记录下来，再按「具体程度」选
    主因：僵持点名到具体门禁行，走穷点名到整张图，无改进只是一个计数。
    只报一个会让架构层丢掉另外两条线索 —— 而它重新进入时正需要这些线索
    来决定先动哪里。
    """
    best_gain = float(state.get("best", {}).get("gain", 1.0))
    if best_gain >= float(state.get("target_gain", 1.0)):
        state["state"] = STATE_SHIPPED
        state.pop("stall_reason", None)
        state.pop("stall_reasons", None)
        state.pop("stall_detail", None)
        return

    budget = state.get("budget", {})
    cap = budget.get("max_records")
    if isinstance(cap, int) and cap > 0 and int(budget.get("used", 0)) >= cap:
        state["state"] = STATE_STOPPED
        state["stop_reason"] = (
            f"预算耗尽：已用 {budget['used']}/{cap} 次评测。"
            "继续请用 set-budget 提高上限")
        return

    reasons: List[Dict[str, str]] = []

    # 这一条排最前：机制一直没核验说明 profiling 根本没接进回路，
    # 此时 best 卡在 1.0、另外几个信号也都是降级的，先修这个。
    unver = state.get("mechanism_unverified", {})
    if int(unver.get("consecutive", 0)) >= int(
            unver.get("threshold", DEFAULT_UNVERIFIED_THRESHOLD)):
        reasons.append({
            "kind": STALL_MECHANISM_UNVERIFIED,
            "detail": (f"连续 {unver.get('consecutive')} 次判为 "
                       "accept_unverified —— 改动都更快了，但一次机制也没核实。"
                       "通常是 record 没带 --signature / --milestone；"
                       "best 会一直卡在 1.0，campaign 永远到不了 shipped"),
        })

    impasse = gate_impasse(milestone_doc)
    if impasse:
        reasons.append({"kind": STALL_GATE_IMPASSE, "detail": impasse})

    if graph_exhausted(state, milestone_doc):
        reasons.append({"kind": STALL_GRAPH_EXHAUSTED,
                        "detail": "决策图中每条决策都已试过，但仍未达标"})

    stag = state.get("stagnation", {})
    if int(stag.get("since_improvement", 0)) >= int(
            stag.get("threshold", DEFAULT_STAGNATION_THRESHOLD)):
        reasons.append({
            "kind": STALL_NO_IMPROVEMENT,
            "detail": (f"连续 {stag.get('since_improvement')} 轮有有效测量"
                       "但增益不足"),
        })

    if reasons:
        state["state"] = STATE_STALLED
        state["stall_reasons"] = reasons
        state["stall_reason"] = reasons[0]["kind"]
        state["stall_detail"] = reasons[0]["detail"]
        return

    state["state"] = STATE_IMPLEMENT
    state.pop("stall_reason", None)
    state.pop("stall_reasons", None)
    state.pop("stall_detail", None)


# ---------------------------------------------------------------------------
# 死路表
# ---------------------------------------------------------------------------

def nogo_command(state: Dict[str, Any], entry: Dict[str, Any]) -> str:
    """生成一条可直接粘贴执行的死路登记命令。

    论文 §3.3 把回滚与记死路写成一件事。靠 agent 自觉登记是不牢的 ——
    连续失败的时候最容易跳过的就是这种记账动作，而死路表的全部价值在于
    跨代存活：漏记的那条，下一代会原封不动再试一遍。
    """
    symptom = entry.get("symptom") or "<填入本次的 profiler 症状>"
    gain = entry.get("gain")
    evidence = (f"[MEASURED] 增益 {(gain - 1) * 100:+.2f}%，未超噪声底 "
                f"{state.get('noise_floor_pct')}%"
                if isinstance(gain, (int, float))
                else f"[MEASURED] {entry.get('reason', '')}")
    return (
        "python3 <forge-hwkb>/scripts/hwkb.py nogo add \\\n"
        f"    --symptom {json.dumps(symptom, ensure_ascii=False)} \\\n"
        f"    --tried {json.dumps(entry.get('decision', ''), ensure_ascii=False)} \\\n"
        f"    --evidence {json.dumps(evidence, ensure_ascii=False)} \\\n"
        f"    --arch {state.get('arch')} "
        f"--operator {state.get('operator')}"
    )


def write_nogo(nogo_path: Path, state: Dict[str, Any],
               entry: Dict[str, Any]) -> int:
    """直接写入 hwkb 的死路表。

    格式由 forge-hwkb 的 hwkb.py 定义（`{"entries": [...]}`）；
    这里只追加，不做任何解释。改格式时两边要一起改。
    """
    entries: List[Dict[str, Any]] = []
    if nogo_path.exists():
        try:
            doc = json.loads(nogo_path.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                entries = doc.get("entries", []) or []
        except (OSError, json.JSONDecodeError):
            entries = []

    gain = entry.get("gain")
    evidence = (f"[MEASURED] 增益 {(gain - 1) * 100:+.2f}%，未超噪声底 "
                f"{state.get('noise_floor_pct')}%"
                if isinstance(gain, (int, float))
                else f"[MEASURED] {entry.get('reason', '')}")

    entries.append({
        "symptom": entry.get("symptom", ""),
        "tried": entry.get("decision", ""),
        "evidence": evidence,
        "arch": state.get("arch"),
        "operator": state.get("operator"),
        "input_shape": state.get("input_shape"),
    })
    nogo_path.parent.mkdir(parents=True, exist_ok=True)
    nogo_path.write_text(
        json.dumps({"entries": entries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return len(entries)


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
        print("警告: 噪声底为 0 意味着任何正增益都会被接受，"
              "测量抖动会被当成真实加速。请先用 noise_floor.py 量一次。",
              file=sys.stderr)

    state = new_state(
        operator=args.operator, input_shape=args.input_shape, arch=args.arch,
        reference={
            "kind": args.reference_kind, "id": args.reference_id,
            "path": args.reference_path, "sealed": True, "generation": 0,
        },
        noise_floor_pct=args.noise_floor, target_gain=args.target_gain,
        stagnation_threshold=args.stagnation_threshold, budget=args.budget,
    )
    save_state(path, state)
    print(f"已创建 campaign: {args.operator} @ {args.arch}")
    print(f"  输入形状  {args.input_shape}")
    print(f"  参考      {args.reference_kind}:{args.reference_id}（封闭，只测不读）")
    print(f"  噪声底    {args.noise_floor}%")
    print(f"  目标增益  {args.target_gain}")
    if args.budget:
        print(f"  预算      {args.budget} 次评测")
    action = next_action(state)
    print(f"\n下一步: {action['action']} — {action['reason']}")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    path = Path(args.state)
    try:
        state = load_state(path)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if state.get("state") == STATE_STOPPED:
        print(f"错误: campaign 已结束（{state.get('stop_reason', '')}），不接受新结果",
              file=sys.stderr)
        return 1

    if state.get("noise_floor_status") == NOISE_PENDING:
        print("错误: 参考已更换，噪声底尚未对新参考重测。\n"
              "  新参考是另一个二进制，运行时特征不同，沿用上一代的噪声底会让\n"
              "  接受判据失准。先跑 noise_floor.py，再用：\n"
              f"    campaign.py set-noise-floor --state {path} --value <实测值>\n"
              "  （<实测值> 填 noise_floor.py 输出的「噪声底」那一行）",
              file=sys.stderr)
        return 1

    try:
        doc = json.loads(Path(args.result).read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"错误: 无法读取评测结果 {args.result}: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"错误: {args.result} 不是合法 JSON: {exc}", file=sys.stderr)
        return 2

    signature = None
    if args.signature:
        try:
            signature = json.loads(Path(args.signature).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"错误: 无法读取签名 {args.signature}: {exc}", file=sys.stderr)
            return 2
        except json.JSONDecodeError as exc:
            print(f"错误: {args.signature} 不是合法 JSON: {exc}", file=sys.stderr)
            return 2

    try:
        milestone_doc = load_milestone(
            Path(args.milestone) if args.milestone else None)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    entry = apply_result(
        state, doc, args.decision, milestone_doc=milestone_doc,
        signature=signature, recipes_used=args.recipe,
        recipes_invented=args.invented, symptom=args.symptom,
    )
    save_state(path, state)

    print(f"决策 {args.decision}: {entry['verdict']}")
    print(f"  {entry['reason']}")

    mechanism = entry.get("mechanism")
    if isinstance(mechanism, dict) and mechanism.get("lines"):
        print("\n  机制核验:")
        for line in mechanism["lines"]:
            mark = {True: "成立", False: "不成立", None: "无法求值"}[line["verified"]]
            print(f"    [{mark}] {line['id']} ({line.get('label')}): {line['reason']}")

    if entry["verdict"] == VERDICT_ACCEPT_UNVERIFIED:
        print("\n  注意: 该改动保留但不计入 best，也不得作为知识库入库依据。")
        print("        要让它算数，补上 --signature 与 --milestone 重新核验机制。")

    if entry["verdict"] in (VERDICT_REVERT, VERDICT_REGRESSION):
        if args.nogo_store:
            count = write_nogo(Path(args.nogo_store), state, entry)
            print(f"\n  已登记死路（当前 {count} 条）: {args.nogo_store}")
        else:
            print("\n  这个方向应登记进死路表，否则下一代会原封不动再试一遍：")
            print("  " + nogo_command(state, entry).replace("\n", "\n  "))

    action = next_action(state)
    print(f"\n下一步: {action['action']} — {action['reason']}")
    return 0


def cmd_reopen(args: argparse.Namespace) -> int:
    """重开一个架构决策，按依赖图作废所有下游实现。"""
    path = Path(args.state)
    try:
        state = load_state(path)
        milestone_doc = load_milestone(Path(args.milestone))
        if milestone_doc is None:
            raise CampaignError("reopen 必须提供 --milestone")
        affected = dependents_closure(milestone_doc, args.decision)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    invalidated = {args.decision} | affected
    count = 0
    for entry in state.get("ledger", []):
        if entry.get("decision") in invalidated and not entry.get("invalidated"):
            entry["invalidated"] = True
            entry["invalidated_by"] = args.decision
            entry["counts_toward_best"] = False
            entry["kb_admissible"] = False
            count += 1

    # best 要按剩余有效条目重算，否则会留着一个来自已作废实现的数字。
    # gain 与 time_us 必须一起取，不能只算 gain 而把 time_us 置空 ——
    # 那样「重算」就只做了一半。
    best_gain = 1.0
    best_time = None
    for entry in state.get("ledger", []):
        if entry.get("invalidated") or not entry.get("counts_toward_best"):
            continue
        gain = entry.get("gain")
        if isinstance(gain, (int, float)) and gain > best_gain:
            best_gain = gain
            best_time = entry.get("time_us")
    state["best"] = {"gain": best_gain, "time_us": best_time,
                     "artifact": state.get("best", {}).get("artifact")}

    state["state"] = STATE_ANALYZE
    state["cursor"] = {"milestone": None, "decision": None, "iteration": 0}
    # 三个计数器一起归零，与 promote 的处理对齐
    state.setdefault("stagnation", {})["since_improvement"] = 0
    state.setdefault("implementation_failures", {})["consecutive"] = 0
    state.setdefault("mechanism_unverified", {})["consecutive"] = 0
    state.pop("stall_reason", None)
    state.pop("stall_reasons", None)
    state.pop("stall_detail", None)
    state.setdefault("ledger", []).append({
        "at": _now(),
        "generation": state.get("generation", 0),
        "input_shape": state.get("input_shape"),
        "decision": args.decision,
        "gain": None,
        "verdict": "reopen",
        "reason": (f"重开决策 {args.decision}；按 depends_on 反向闭包作废 "
                   f"{len(affected)} 条下游决策的 {count} 条台账记录。"
                   f"理由：{args.reason or '未填'}"),
        "invalidated_decisions": sorted(invalidated),
        "counts_toward_best": False,
        "kb_admissible": False,
    })
    save_state(path, state)

    print(f"已重开 {args.decision}")
    if affected:
        print(f"  受影响的下游决策: {', '.join(sorted(affected))}")
    else:
        print("  没有下游决策依赖它")
    print(f"  作废台账记录: {count} 条")
    print(f"  best 重算为: {best_gain:.4f}")
    print("\n下一步: analyze — 这些决策的实现都建立在旧假设上，须重做")
    return 0


def cmd_set_noise_floor(args: argparse.Namespace) -> int:
    path = Path(args.state)
    try:
        state = load_state(path)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    if args.value < 0:
        print("错误: 噪声底不能为负", file=sys.stderr)
        return 2

    old = state.get("noise_floor_pct")
    state["noise_floor_pct"] = args.value
    state["noise_floor_status"] = NOISE_MEASURED
    state.setdefault("ledger", []).append({
        "at": _now(), "generation": state.get("generation", 0),
        "input_shape": state.get("input_shape"), "decision": None,
        "gain": None, "verdict": "noise_floor",
        "reason": f"噪声底由 {old}% 更新为 {args.value}%",
        "counts_toward_best": False, "kb_admissible": False,
    })
    save_state(path, state)
    print(f"噪声底已更新: {old}% → {args.value}%")
    action = next_action(state)
    print(f"下一步: {action['action']} — {action['reason']}")
    return 0


def cmd_set_budget(args: argparse.Namespace) -> int:
    path = Path(args.state)
    try:
        state = load_state(path)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    budget = state.setdefault("budget", {"max_records": None, "used": 0})
    budget["max_records"] = args.max_records
    if state.get("state") == STATE_STOPPED and "预算耗尽" in str(
            state.get("stop_reason", "")):
        state["state"] = STATE_IMPLEMENT
        state.pop("stop_reason", None)
    save_state(path, state)
    print(f"预算已设为 {args.max_records} 次评测（已用 {budget.get('used', 0)}）")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """把本代产出封存为下一代的参考（论文 §3.5）。"""
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
        state["stop_reason"] = (
            f"自我提升收敛：本代增益 {(gain - 1) * 100:+.2f}% 未超过噪声底 "
            f"{noise:.2f}%")
        state.setdefault("ledger", []).append({
            "at": _now(), "generation": state.get("generation", 0),
            "input_shape": state.get("input_shape"), "decision": None,
            "gain": gain, "verdict": "stop",
            "reason": (f"{state['stop_reason']}。该停点是对硬件上限的一次估计，"
                       "但不构成证明 —— 也可能只是这个循环找不到了"),
            "counts_toward_best": False, "kb_admissible": False,
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
        "path": args.artifact, "sealed": True,
        "generation": generation - 1, "promoted_gain": gain,
    }
    state["state"] = STATE_ANALYZE
    state["cursor"] = {"milestone": None, "decision": None, "iteration": 0}
    state.setdefault("stagnation", {})["since_improvement"] = 0
    state.setdefault("implementation_failures", {})["consecutive"] = 0
    state.setdefault("mechanism_unverified", {})["consecutive"] = 0
    state["target_gain"] = args.phase_target_gain
    state["best"] = {"gain": 1.0, "time_us": None, "artifact": None}
    # 参考换了，噪声底必须重测
    state["noise_floor_status"] = NOISE_PENDING
    state.setdefault("ledger", []).append({
        "at": _now(), "generation": generation,
        "input_shape": state.get("input_shape"), "decision": None,
        "gain": gain, "verdict": "promote",
        "reason": (f"第 {generation - 1} 代产出封存为第 {generation} 代的参考；"
                   f"源码不跨代边界，知识库跨。新阶段目标增益 "
                   f"{args.phase_target_gain}。噪声底待对新参考重测"),
        "counts_toward_best": False, "kb_admissible": False,
    })
    save_state(path, state)

    print(f"已提升到第 {generation} 代")
    print(f"  新参考    {state['reference']['id']}（封闭）")
    print(f"  上代增益  {(gain - 1) * 100:+.2f}%")
    print(f"  新目标    {args.phase_target_gain}")
    print("\n下一步: measure-noise-floor — 新参考是另一个二进制，"
          "运行时特征不同，噪声底必须重测")
    print(f"  量完后: campaign.py set-noise-floor --state {path} --value <实测值>")
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
    impl = state.get("implementation_failures", {})
    ref = state.get("reference", {})
    budget = state.get("budget", {})

    print(f"算子      {state.get('operator')} @ {state.get('arch')}")
    print(f"输入形状  {state.get('input_shape')}")
    print(f"代次      第 {state.get('generation')} 代")
    print(f"参考      {ref.get('kind')}:{ref.get('id')}")
    print(f"状态      {state.get('state')}"
          + (f"（{state.get('stall_reason')}）" if state.get("stall_reason") else ""))
    noise_mark = ("待重测" if state.get("noise_floor_status") == NOISE_PENDING
                  else "已实测")
    print(f"噪声底    {state.get('noise_floor_pct')}%（{noise_mark}）")
    print(f"目标增益  {state.get('target_gain')}")
    print(f"当前最佳  {float(best.get('gain', 1.0)):.4f}"
          f"（{(float(best.get('gain', 1.0)) - 1) * 100:+.2f}%，仅计机制已核实的）")
    print(f"停滞计数  {stag.get('since_improvement')}/{stag.get('threshold')}"
          "（只数有有效测量但增益不足）")
    print(f"实现失败  {impl.get('consecutive')}/{impl.get('threshold')}"
          "（精度/评测问题，不触发架构重入）")
    unver = state.get("mechanism_unverified", {})
    print(f"机制未核实 {unver.get('consecutive')}/{unver.get('threshold')}"
          "（连续多次即判 profiling 没接进回路）")
    cap = budget.get("max_records")
    print(f"预算      {budget.get('used', 0)}/{cap if cap else '不限'}")
    print(f"台账条目  {len(state.get('ledger', []))}")
    print(f"\n下一步: {action['action']} — {action['reason']}")
    return 0


def summarise_recipes(state: Dict[str, Any]) -> Dict[str, Any]:
    """汇总知识库的使用与产出。

    论文 §3.4：处方是返回路径 —— campaign 台账同时记录本轮发明的和外来的
    卡片，这让知识库的贡献可审计而不是口说无凭。

    没有这份汇总就无法回答「知识库到底有没有在帮忙」：跨算子复用率多少、
    哪些卡片真被用上、新一代里多少增益来自外来经验。
    """
    used: Dict[str, int] = {}
    invented: Dict[str, int] = {}
    accepted_with_foreign = 0
    accepted_total = 0

    for entry in state.get("ledger", []):
        if entry.get("invalidated"):
            continue
        for recipe in entry.get("recipes_used", []) or []:
            used[recipe] = used.get(recipe, 0) + 1
        for recipe in entry.get("recipes_invented", []) or []:
            invented[recipe] = invented.get(recipe, 0) + 1
        if entry.get("verdict") in (VERDICT_ACCEPT, VERDICT_ACCEPT_UNVERIFIED):
            accepted_total += 1
            if entry.get("recipes_used"):
                accepted_with_foreign += 1

    return {
        "foreign_recipes": dict(sorted(used.items())),
        "invented_recipes": dict(sorted(invented.items())),
        "accepted_total": accepted_total,
        "accepted_with_foreign_recipe": accepted_with_foreign,
        "foreign_assist_rate": (accepted_with_foreign / accepted_total
                                if accepted_total else 0.0),
    }


def cmd_ledger(args: argparse.Namespace) -> int:
    path = Path(args.state)
    try:
        state = load_state(path)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    entries: List[Dict[str, Any]] = state.get("ledger", [])

    if args.recipes:
        summary = summarise_recipes(state)
        if args.as_json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        print("— 外来经验（本轮引用了别处总结的卡片）—")
        for name, count in summary["foreign_recipes"].items() or []:
            print(f"  {name:<20} 被引用 {count} 次")
        if not summary["foreign_recipes"]:
            print("  （无）")
        print("\n— 自产经验（本轮新总结的卡片）—")
        for name, count in summary["invented_recipes"].items() or []:
            print(f"  {name:<20} 登记 {count} 次")
        if not summary["invented_recipes"]:
            print("  （无）")
        print(f"\n接受的改动 {summary['accepted_total']} 条，"
              f"其中 {summary['accepted_with_foreign_recipe']} 条用到了外来经验"
              f"（{summary['foreign_assist_rate'] * 100:.0f}%）")
        print("\n这个比例是判断知识库值不值得继续投入的依据。")
        return 0

    if args.as_json:
        print(json.dumps(entries, ensure_ascii=False, indent=2))
        return 0
    if not entries:
        print("台账为空")
        return 0
    for entry in entries:
        gain = entry.get("gain")
        gain_text = (f"{(gain - 1) * 100:+.2f}%"
                     if isinstance(gain, (int, float)) else "—")
        decision = entry.get("decision") or "—"
        marks = []
        if entry.get("invalidated"):
            marks.append("已作废")
        if entry.get("verdict") == VERDICT_ACCEPT_UNVERIFIED:
            marks.append("机制未核实")
        suffix = f"  [{', '.join(marks)}]" if marks else ""
        print(f"[gen{entry.get('generation')}] {decision:<8} "
              f"{str(entry.get('verdict')):<18} {gain_text:>9}{suffix}")
        print(f"      {entry.get('reason')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造命令行。

    `--state` 与 `--json` 同时挂在顶层和每个子命令上，两种写法都能跑：

        campaign.py --state s.json record ...
        campaign.py record --state s.json ...

    只挂顶层是 argparse 的默认做法，但它要求全局参数出现在子命令**之前**，
    而人和 agent 的直觉都是写在后面。写错时 argparse 只报
    「unrecognized arguments」，不说该放哪里 —— agent 拿到这个报错往往不会
    去调参数位置，而是转而自己判断，可这套设计的要害恰恰是不让 agent
    自己裁决。两种写法都收下，这个口子就不存在了。

    子命令那份用 SUPPRESS 作默认值：不传时不写入 namespace，顶层的取值
    得以保留；传了才覆盖。若用普通默认值，`--state X record` 会被子命令的
    默认值悄悄覆盖回去。
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state", default=argparse.SUPPRESS,
                        help="状态文件路径（默认 campaign_state.json）")
    common.add_argument("--json", action="store_true", dest="as_json",
                        default=argparse.SUPPRESS,
                        help="以 JSON 输出，供脚本消费")

    parser = argparse.ArgumentParser(
        description="ForgeKernel campaign 状态机：嵌套双循环、机制裁决与自我提升",
    )
    parser.add_argument("--state", default="campaign_state.json",
                        help="状态文件路径（也可写在子命令后面）")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="以 JSON 输出（也可写在子命令后面）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="创建 campaign", parents=[common])
    p_init.add_argument("--operator", required=True)
    p_init.add_argument("--input-shape", required=True,
                        help="本次 campaign 的输入形状。噪声底、签名、门禁锚点"
                             "都与 shape 绑定，一份 state 不得跨 shape 复用")
    p_init.add_argument("--arch", required=True)
    p_init.add_argument("--reference-kind", required=True,
                        choices=["builtin_aclnn", "opp_install", "sealed_kernel"])
    p_init.add_argument("--reference-id", required=True)
    p_init.add_argument("--reference-path", default=None)
    p_init.add_argument("--noise-floor", type=float, required=True,
                        help="实测噪声底百分比，用 noise_floor.py 量")
    p_init.add_argument("--target-gain", type=float,
                        default=DEFAULT_PHASE_TARGET_GAIN)
    p_init.add_argument("--stagnation-threshold", type=int,
                        default=DEFAULT_STAGNATION_THRESHOLD)
    p_init.add_argument("--budget", type=int, default=None,
                        help="评测次数上限；不给则不限")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_status = sub.add_parser("status", help="查看进度与下一步", parents=[common])
    p_status.set_defaults(func=cmd_status)

    p_record = sub.add_parser("record", help="并入一次评测结果并裁决",
                              parents=[common])
    p_record.add_argument("--result", required=True,
                          help="evaluation_results.json 路径")
    p_record.add_argument("--decision", required=True, help="本次实现的决策行 id")
    p_record.add_argument("--signature", default=None,
                          help="signature_probe.py 的输出；机制核验需要它")
    p_record.add_argument("--milestone", default=None,
                          help="milestone.json；机制核验与走穷/僵持判定需要它")
    p_record.add_argument("--recipe", action="append", default=[],
                          help="本次用到的外来经验卡片 id，可重复")
    p_record.add_argument("--invented", action="append", default=[],
                          help="本次新总结的经验卡片 id，可重复")
    p_record.add_argument("--symptom", default="",
                          help="本次的 profiler 症状，回滚时写进死路表")
    p_record.add_argument("--nogo-store", default=None,
                          help="死路表路径；给了就在回滚时直接写入")
    p_record.set_defaults(func=cmd_record)

    p_reopen = sub.add_parser("reopen", help="重开架构决策并作废下游实现",
                              parents=[common])
    p_reopen.add_argument("--milestone", required=True)
    p_reopen.add_argument("--decision", required=True, help="要重开的决策 id")
    p_reopen.add_argument("--reason", default="")
    p_reopen.set_defaults(func=cmd_reopen)

    p_nf = sub.add_parser("set-noise-floor", help="更新实测噪声底",
                          parents=[common])
    p_nf.add_argument("--value", type=float, required=True)
    p_nf.set_defaults(func=cmd_set_noise_floor)

    p_bg = sub.add_parser("set-budget", help="设置或提高评测次数上限",
                          parents=[common])
    p_bg.add_argument("--max-records", type=int, required=True)
    p_bg.set_defaults(func=cmd_set_budget)

    p_promote = sub.add_parser("promote", help="封存本代产出为下一代参考",
                               parents=[common])
    p_promote.add_argument("--artifact", required=True)
    p_promote.add_argument("--phase-target-gain", type=float,
                           default=DEFAULT_PHASE_TARGET_GAIN)
    p_promote.add_argument("--force", action="store_true",
                           help="即使增益未超噪声底也强行提升")
    p_promote.set_defaults(func=cmd_promote)

    p_ledger = sub.add_parser("ledger", help="查看裁决台账", parents=[common])
    p_ledger.add_argument("--recipes", action="store_true",
                          help="汇总知识库的使用与产出")
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
