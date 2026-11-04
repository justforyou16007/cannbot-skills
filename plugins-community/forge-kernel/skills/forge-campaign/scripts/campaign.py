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
        --noise-floor 2.3 --target-gain 1.05 --budget 40 \\
        --device-id 0

    campaign.py record --state campaign_state.json \\
        --result evaluation_results.json --decision D0.1 \\
        --signature architecture_signature.json --milestone milestone.json \\
        --recipe R-2201-001 --symptom "MTE2 占比过高"

    campaign.py reopen --state campaign_state.json \\
        --milestone milestone.json --decision D0.1 --reason "拓扑决策有缺陷"

    campaign.py set-noise-floor --state campaign_state.json --value 2.7
    campaign.py promote --state campaign_state.json --artifact ./best \\
        --library ~/.forge-kernel/lib --signature architecture_signature.json
    campaign.py ledger --state campaign_state.json --recipes

`--state` 与 `--json` 写在子命令前后都可以，两种形态都能跑。

退出码: 0 = 正常; 1 = 状态不允许该操作; 2 = 用法或 IO 错误
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

SCHEMA_VERSION = "2.0"

#: 评测次数的默认上限。
#:
#: 早先默认不限。不限的代价是循环没有任何出口：停滞计数只要被「超过噪声底」
#: 清零就永远不转 stalled，而 status 每轮都说「继续实现」。论文实现同时卡
#: 三个维度（墙钟 4 小时、60 轮、teacher 3 轮），我们至少要有轮数这一个。
DEFAULT_BUDGET = 60

#: 本轮实测抖动达到存量噪声底的多少倍时，判定「环境变了，旧判据不再可信」。
#:
#: 不用「超过就拒」：存量噪声底本身是从某一次 cv 量出来的，之后每轮在它
#: 上下浮动是正常的。那样的门禁会天天响，然后被反射性豁免。
STALE_FLOOR_FACTOR = 1.5

#: 认得的架构串。
#:
#: 真正的来源是 forge-hwkb 的 ARCH_DIRS —— 架构成熟卡片按这些串分目录存放。
#: 这里填一个对不上的串（比如 "ascend910b"），campaign 能一路跑完，但每张
#: 卡片入库时都会被 hwkb 判 A-ARCH 拒掉：几十轮的经验一条也存不下来，而这
#: 件事要跑很久才会暴露。所以在 init 就挡住。
#:
#: 两边一致性由 tests/test_campaign.py 的用例盯着，不靠人记得同步改。
KNOWN_ARCHS = ("DAV_2201", "DAV_3510")

#: 参考定义文件名与「内置实现」这种来源的名字，来源是同目录的
#: reference_adapter.py。这里只用到两个常量，为此在模块顶层 import 整个脚本
#: 不值得；一致性同样由 tests/test_campaign.py 的用例盯着。
SPEC_FILENAME = "reference_spec.json"
SEALED_PATHS_FILENAME = "sealed_paths.txt"
KIND_BUILTIN = "builtin_aclnn"

# 状态机
STATE_ANALYZE = "analyze"      # 架构层在分析参考 / 修订课程
STATE_IMPLEMENT = "implement"  # 编码层在实现当前决策行
STATE_STALLED = "stalled"      # 需要架构层重新进入
STATE_SHIPPED = "shipped"      # 本代达标，可提升为下一代参考
STATE_STOPPED = "stopped"      # 自我提升收敛或预算耗尽
STATE_BLOCKED = "blocked"      # 连续实现失败，需要人介入（不消耗架构重入）

#: 外层重入的口子（论文 §3.3 给了前三个；第四个是本实现补的，见下）。
STALL_NO_IMPROVEMENT = "no_improvement"
STALL_GRAPH_EXHAUSTED = "graph_exhausted"
STALL_GATE_IMPASSE = "gate_impasse"

#: 第四、第五个口子：连续多次「变快了但机制不算数」。
#:
#: 论文没有这两条，因为在它的实现里计时与 profiling 在同一个 job 里完成，
#: 不存在「忘了采 profiling」这种状态。昇腾上两者必须分两次跑
#: （acl.prof.init 每进程只能调一次），于是多出一种失效方式：
#: 一直判 accept_unverified —— 它不计入 best 所以永远到不了 shipped，
#: 又会清零停滞计数所以永远不转 stalled，循环就这么一直跑下去，
#: 而 status 每轮都说「继续实现」。
#:
#: 两条分开，是因为**该做的事完全不同**：
#:
#:   UNVERIFIED  机制没被核验过（多半是 record 没带 --signature）。
#:               这是流程缺口，补参数重跑即可。
#:
#:   REFUTED     机制被核验了，而且求值为假。这**不是**流程问题，
#:               而是一个很强的架构信号：对参考行为的解释错了，但实现
#:               碰对了别的东西。该做的是改课程表里的机制假设，不是补参数。
#:               论文说认证把真正起作用的机制和只是伴随发生的分开 ——
#:               要抓的正是这种情形。
#:
#: 混成一条会让架构层去补参数重跑，而真正该改的课程表没人碰。
STALL_MECHANISM_UNVERIFIED = "mechanism_never_verified"
STALL_MECHANISM_REFUTED = "mechanism_refuted"

#: 连续实现失败。**不是**外层重入口 —— 实现 bug 不该花掉一次架构重入，
#: 但它必须改变动作，否则同一个错会重复几十次把预算吃光。
STALL_IMPL_FAILURE = "implementation_blocked"

#: 内层连续多少轮「测出来了但增益不够」才交回架构层。
DEFAULT_STAGNATION_THRESHOLD = 3

#: 实现类失败（精度不过、评测缺失）连续多少次提示人工介入。
#: 它**不**触发架构重入 —— 那是实现问题，不是架构问题。
DEFAULT_IMPL_FAILURE_THRESHOLD = 3

#: 连续多少次 accept_unverified 判定为「profiling 没接进回路」。
DEFAULT_UNVERIFIED_THRESHOLD = 3

#: 自我提升每阶段的目标增益（论文 §3.5：+5%）。
DEFAULT_PHASE_TARGET_GAIN = 1.05

#: 机制行的锚点字段至少要移动这么多（相对上一轮的百分比）才算「动过」。
#:
#: 没有这条检查，机制行只是一个阈值比较，而阈值来自参考实测，所以
#: 「低于 4.2」实际含义是「我比参考等得少」。它拦不住这种情形：
#: 某个 kernel 的 MTE2 停顿本来就只有 0.9，改动做的是别的事，
#: 机制行照样成立 —— 门禁看起来过了，实际什么都没验证。
#:
#: 这个 5% 是工程判断，**无文档出处**。停顿类指标的测量噪声没有像时延
#: 那样被实测过（噪声底量的是时延），所以这里取一个保守的相对阈值，
#: 并在输出里如实标注它是推导判据而非实测判据。
MECHANISM_MOVE_MIN_PCT = 5.0

#: 比较极小值时的分母保护。停顿占比可以真的是 0。
_MOVE_EPS = 1e-6

VERDICT_ACCEPT = "accept"
VERDICT_ACCEPT_UNVERIFIED = "accept_unverified"
VERDICT_REVERT = "revert"
VERDICT_REGRESSION = "regression"
#: 确认复测。不是一次候选裁决 —— 它回答「上次那个增益再测一次还在不在」。
VERDICT_CONFIRM = "confirm"

#: 中性记录：这一轮改了代码，但**没有测**。
#:
#: 为什么需要这个裁决：有些改动不产生可测的时间变化 —— 重命名、拆函数、
#: 补注释、把一个常量提成 tiling 参数而取值不变。它们仍然改变了**代码身份**
#: （artifact_sha），而机制做差的全部前提是「被保留下来的最后一条台账记录
#: 就是当前代码的样子」。
#:
#: 没有这个裁决，这类改动只有两条路：
#:   跑一次完整评测   为一个必然无变化的改动花掉一次机时与一次预算
#:   不记             台账里的 sha 从此对不上工作树 —— 而下一轮的做差基线
#:                    就是据此取的，于是那一轮算出来的「因本次改动而移动」
#:                    其实混着这次没记的改动
#:
#: 第二条是会实际发生的那条（因为它免费）。所以要有一条「记代码身份、
#: 不记性能」的路。
VERDICT_NEUTRAL = "neutral"

#: 对候选的裁决。台账里还有 reopen / promote / stop / noise_floor 这类
#: **管理条目**，它们不代表「某条决策被试过」，统计时必须排除。
CANDIDATE_VERDICTS = frozenset({
    VERDICT_ACCEPT, VERDICT_ACCEPT_UNVERIFIED,
    VERDICT_REVERT, VERDICT_REGRESSION,
    # neutral 算候选条目：它确实是「某条决策被动过一次」，只是没测性能。
    # 不算的话，`graph_exhausted` 会以为这条决策还没试过。
    VERDICT_NEUTRAL,
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

#: 布尔连接词。`and`/`or` 要词边界（否则 "operand" 里的 "and" 会命中），
#: 而 `&&`/`||` 不能加 \b —— 符号两侧没有「单词字符」，\b 在那里不成立，
#: 加了等于这两个分支永远匹配不上。两类分开写。
RE_BOOL_CONNECTIVE = re.compile(r"\b(?:and|or)\b|&&|\|\|", re.IGNORECASE)

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


def snapshot_signature(signature: Dict[str, Any]) -> Dict[str, float]:
    """把一份架构签名压成 {通道: 数值}，存进台账供下一轮做差。

    存快照而不是存文件路径：路径会失效（目录被清、工作区换位置），
    而顺序证据一旦丢了就补不回来 —— 那次测量已经过去了。快照是几十个
    数字，代价可以忽略。

    只收 present 为真的字段。缺失的不进快照，下一轮做差时就会如实报告
    「上一轮没有这个读数」，而不是拿 0 去减。
    """
    out: Dict[str, float] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            channel = node.get("channel")
            value = node.get("value")
            if (isinstance(channel, str) and node.get("present") is True
                    and isinstance(value, (int, float))):
                out.setdefault(channel, float(value))
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(signature)

    exposure = signature.get("exposure")
    if isinstance(exposure, dict) and exposure.get("available"):
        for key in ("exposed_pct", "exposed_us", "covered_us"):
            value = exposure.get(key)
            if isinstance(value, (int, float)):
                out.setdefault(f"derived:exposure.{key}", float(value))
    return out


def lookup_snapshot_value(snapshot: Optional[Dict[str, float]],
                          needle: str) -> Optional[float]:
    """在快照里按字段名找值。快照的键是完整通道串，这里按子串匹配。"""
    if not snapshot:
        return None
    for channel, value in snapshot.items():
        if needle in channel:
            return value
    return None


def lookup_signature_matches(signature: Dict[str, Any],
                             needle: str) -> List[Tuple[float, str]]:
    """在架构签名里找出所有通道名含该字段的读数。

    返回全部匹配而不是第一个，因为**匹配到多个本身就是个问题**：
    msprof 的同名指标在 Cube 与 Vector 上各有一份（aic_ / aiv_ 前缀），
    锚点若只写 `mte2_wait_ratio`，两边都能匹配上。随便挑一个意味着可能
    拿 Cube 的读数去回答一个关于 Vector 的问题 —— 而两者毫不相干。
    """
    found: List[Tuple[float, str]] = []
    seen: Set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            channel = node.get("channel")
            value = node.get("value")
            if (isinstance(channel, str) and needle in channel
                    and node.get("present") is True
                    and isinstance(value, (int, float))
                    and channel not in seen):
                seen.add(channel)
                found.append((float(value), channel))
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(signature)

    # exposure 段是推导量，不走 field() 的形状，单独找一次
    exposure = signature.get("exposure")
    if isinstance(exposure, dict) and exposure.get("available"):
        value = exposure.get(needle)
        if isinstance(value, (int, float)):
            channel = f"derived:exposure.{needle}"
            if channel not in seen:
                found.append((float(value), channel))
    return found


def lookup_signature_value(signature: Dict[str, Any],
                           needle: str) -> Tuple[Optional[float], Optional[str]]:
    """取第一个匹配。歧义检测见 lookup_signature_matches。

    返回 (值, 实际命中的通道)。找不到返回 (None, None)；
    **找不到与值为 0 必须分开**，理由同 signature_probe 的设计。
    """
    matches = lookup_signature_matches(signature, needle)
    return matches[0] if matches else (None, None)


def _expected_direction(operator: str) -> Optional[int]:
    """从比较符推出机制声称的变化方向。

    `field < 阈值` 说明声称的是「把它降下来」，`field > 阈值` 是「升上去」。
    比较符本身已经编码了哪边算好，不必再维护一张「每个标签该增该减」的
    对照表 —— 那张表会和词汇表漂移。
    """
    if operator in ("<", "<="):
        return -1
    if operator in (">", ">="):
        return +1
    return None  # == / != 不蕴含方向


def verify_mechanism_line(line: Dict[str, Any],
                          signature: Dict[str, Any],
                          previous_snapshot: Optional[Dict[str, float]] = None,
                          move_min_pct: float = MECHANISM_MOVE_MIN_PCT,
                          reference_snapshot: Optional[Dict[str, float]] = None,
                          ) -> Dict[str, Any]:
    """核验一条机制行：阈值成立，**而且**那个字段确实因本次改动而移动。

    两步缺一不可。只查阈值的话，一个本来停顿就很低的 kernel 会让机制行
    平凡成立 —— 门禁看起来过了，实际什么都没验证。论文要的顺序证据正是
    「改动前后两次读数」这个序对。

    四种结果，**不要混**：
      True     阈值成立，且字段朝声称的方向移动了 —— 机制确实落地
      False    阈值不成立 —— 机制没落地，这次的增益来自别处
      vacuous  阈值成立但字段没动（或动反了）—— 不是本次改动造成的，
               和 False 一样说明增益另有来源，但成因不同，要分开说
      None     无法求值（字段没采到、缺上一轮读数、表达式不认识）。
               不等于失败，但也不构成验证
    """
    line_id = line.get("id", "<无 id>")
    label = line.get("label")
    anchor = line.get("anchor") or {}
    expr = anchor.get("expr") if isinstance(anchor, dict) else None

    out: Dict[str, Any] = {"id": line_id, "label": label, "expr": expr,
                           "vacuous": False}

    if not isinstance(expr, str) or not expr.strip():
        out.update(verified=None, reason="机制行没有可求值的锚点表达式")
        return out

    match = RE_MECH_EXPR.search(expr)
    if not match:
        out.update(verified=None,
                   reason=f"锚点表达式 {expr!r} 不是「通道 比较符 数值」的形状，无法求值")
        return out

    # 复合条件不能按「只看第一个子句」处理。
    #
    # search() 只取第一处匹配，后面的部分被丢掉。于是
    # `a < 4.2 and b < 0.3` 这种写法只有 a 被核验，而输出照样报「机制成立」——
    # 一条只验证了一半的机制行会被当成完整的顺序证据，封进产物库和知识库。
    # 要么两个子句都验，要么明说验不了；悄悄验一半是最坏的那种。
    #
    # 这里选「明说验不了」：第二步「字段是否因本次改动而移动」是按单个通道
    # 做的，多子句需要每个通道各有基线、各判方向，那是另一件事。拆成多条
    # 机制行本来也更清楚 —— 一条机制对一个停顿字段。
    leftover = (expr[:match.start()] + expr[match.end():]).strip()
    if RE_MECH_EXPR.search(leftover) or RE_BOOL_CONNECTIVE.search(expr):
        out.update(
            verified=None,
            reason=(f"锚点表达式 {expr!r} 含多个条件，本核验只认单个"
                    "「通道 比较符 数值」。\n"
                    f"  只验第一个子句（{match.group(0).strip()}）再报「成立」"
                    "会让一条半验证的机制行被当成完整证据收下，"
                    "所以这里直接判为无法核验。\n"
                    "  请拆成多条机制行，每条锚定一个停顿字段 —— "
                    "一条机制对应一个可观测量，本来也更清楚"))
        return out

    channel = match.group("channel")
    field = channel.rsplit(":", 1)[-1].strip()
    threshold = float(match.group("value"))
    op = match.group("op")

    # 阈值本身要受参考实测值约束。
    #
    # 不约束的后果是实测出来的：一条锚在参考值 1.6 的停顿行被写成 ≤6.0，
    # 于是实现以**两倍于参考**的等待量舒舒服服坐在带内，轮轮过门禁。
    # 带宽宽出参考几倍的验收线只剩仪式价值。
    #
    # 默认带宽取参考实测的 1.25 倍（比噪声底高一档）。要更宽必须在机制行里
    # 写 band_rationale 说明理由 —— 那样至少是一个有记录的决定，而不是
    # 一个随手写大的数字。
    band = line.get("band_rationale")
    out["band_rationale"] = band

    matches = lookup_signature_matches(signature, field)
    if len(matches) > 1:
        # 同名指标在 Cube 与 Vector 上各有一份。挑哪个都是蒙 ——
        # 拿 Cube 的读数回答关于 Vector 的问题，两者毫不相干，
        # 而算出来的百分比会显得很有说服力。
        names = sorted(c.rsplit(":", 1)[-1] for _v, c in matches)
        out.update(
            verified=None,
            reason=(f"字段 {field} 在签名里匹配到 {len(matches)} 个通道"
                    f"（{names}）—— 同名指标在不同硬件单元上各有一份，"
                    "无法判断这条机制行说的是哪一个。"
                    "请把锚点写成带单元前缀的全名（如 aiv_ / aic_）"))
        return out

    if not matches:
        out.update(verified=None,
                   reason=(f"签名里没有 {field} 的有效读数。"
                           "缺失不等于不成立 —— 但也不构成验证"))
        return out

    actual, hit_channel = matches[0]
    out.update(actual=actual, threshold=threshold, operator=op,
               channel=hit_channel)

    # 带宽纪律：只对「越小越好」的上界型阈值判（< / <=）。
    # 下界型（> / >=）的语义相反，参考值在它那一侧不构成同样的约束。
    # 注意取的是**参考**快照，不是上一轮快照。
    #
    # 这条纪律问的是「这个验收线和参考实际做到的水平差多远」。拿上一轮的
    # 读数当锚会让带宽随着实现一起漂：上一轮坐在 5.5（参考 1.6），下一轮
    # 的带宽就变成 6.9 —— 参考从此退出判据。
    reference_value = (reference_snapshot or {}).get(hit_channel)
    if (op in ("<", "<=") and isinstance(reference_value, (int, float))
            and reference_value > _MOVE_EPS):
        cap = reference_value * MECHANISM_BAND_FACTOR
        out["reference_value"] = reference_value
        out["band_cap"] = cap
        if threshold > cap and not band:
            out.update(
                verified=False,
                reason=(f"机制行的阈值 {threshold} 宽出参考实测值 "
                        f"{reference_value} 的 {threshold / reference_value:.1f} 倍"
                        f"（默认带宽上限 {cap:.3f} = 参考 × "
                        f"{MECHANISM_BAND_FACTOR}）。\n"
                        "  这样的验收线只剩仪式价值：实现可以带着远高于参考的"
                        "等待量坐在带内，轮轮过门禁。\n"
                        "  要么把阈值收到带内，要么在机制行里写 "
                        "band_rationale 说明为什么这个参考通道上更宽是合理的"))
            return out

    threshold_ok = _COMPARATORS[op](actual, threshold)
    if not threshold_ok:
        out.update(verified=False,
                   reason=f"{field} 实测 {actual} {op} {threshold} 不成立")
        return out

    # 阈值成立，再看它是不是因本次改动而移动的。
    # 基线按**实际命中的那个通道**精确取，不再按字段名模糊匹配 ——
    # 模糊匹配会把 Cube 的基线配给 Vector 的实测，算出一个看着很漂亮
    # 但毫无意义的百分比。
    previous = (previous_snapshot or {}).get(hit_channel)
    if previous is None:
        out.update(
            verified=None,
            reason=(f"{field} 实测 {actual} {op} {threshold} 成立，"
                    f"但基线里没有同一通道（{hit_channel}）的读数，"
                    "无法判断它是否因本次改动而变化。"
                    "阈值成立也可能只是它本来就在那个位置 —— "
                    "这不构成顺序证据"))
        return out

    out["previous"] = previous
    delta = actual - previous
    denominator = max(abs(previous), _MOVE_EPS)
    moved_pct = delta / denominator * 100.0
    out["moved_pct"] = moved_pct

    expected = _expected_direction(op)
    if expected is None:
        # == / != 不蕴含方向，只要动过即可
        enough = abs(moved_pct) >= move_min_pct
        direction_ok = True
    else:
        enough = abs(moved_pct) >= move_min_pct
        direction_ok = (moved_pct < 0) if expected < 0 else (moved_pct > 0)

    if enough and direction_ok:
        arrow = "降至" if delta < 0 else "升至"
        out.update(verified=True,
                   reason=(f"{field} 由 {previous} {arrow} {actual}"
                           f"（{moved_pct:+.1f}%），且 {op} {threshold} 成立"))
        return out

    if not enough:
        out.update(
            verified=False, vacuous=True,
            reason=(f"{field} 本来就是 {previous}，本轮 {actual}，"
                    f"仅变动 {moved_pct:+.1f}%（低于 {move_min_pct}%）。"
                    f"阈值 {op} {threshold} 成立，但不是本次改动造成的 —— "
                    "这条机制行是平凡通过的"))
        return out

    out.update(
        verified=False, vacuous=True,
        reason=(f"{field} 由 {previous} 变为 {actual}（{moved_pct:+.1f}%），"
                f"方向与 {op} {threshold} 声称的相反。"
                "阈值仍成立，但机制没有朝声称的方向起作用"))
    return out


def _iter_all_gate_lines(doc: Dict[str, Any]):
    """遍历 milestone 里所有门禁行，产出 (milestone_id, 类别, 行)。"""
    for milestone in doc.get("milestones", []) or []:
        if not isinstance(milestone, dict):
            continue
        mid = milestone.get("id", "<无 id>")
        gate = milestone.get("gate") or {}
        if not isinstance(gate, dict):
            continue
        for category in ("correctness_lines", "performance_lines",
                         "mechanism_lines"):
            for line in gate.get(category, []) or []:
                if isinstance(line, dict):
                    yield mid, category, line


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


def milestone_of_decision(milestone_doc: Dict[str, Any],
                          decision_id: str) -> Optional[Dict[str, Any]]:
    """这条决策属于哪个 milestone。"""
    for milestone in milestone_doc.get("milestones", []) or []:
        if not isinstance(milestone, dict):
            continue
        ids = {d.get("id") for d in milestone.get("decisions", []) or []
               if isinstance(d, dict)}
        if decision_id in ids:
            return milestone
    return None


def check_gate_coupling(milestone_doc: Optional[Dict[str, Any]],
                        decision_id: str,
                        mechanism: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """课程表里标 PASS 的机制行，和本轮实测的机制结论对不对得上。

    此前这两份文件**从不对账**：课程表自己写 `status: PASS`，台账自己记
    `mechanism.status`，两边各说各话。于是会出现一个很具体的情形 ——

      课程表：M1 机制行 status=PASS，closed_by 指着 job 7
      台账：  那一轮的 mechanism.status=refuted

    读课程表的人（下一轮的编码层、做 harvest 的人、看 status 的人）得到的是
    「这条机制已经核实」，而唯一做过核验的那次说的是相反的话。课程表是三个
    产物里**唯一会被下一轮当作前提**的那个，所以这个不一致不是记错一笔账，
    是让后续所有决策建立在一个假前提上。

    这里只报告，不改文件 —— `record` 不该去写课程表（那是架构层的产物，
    编码层与编排层都不许改，`AGENTS.md` 的禁止项里写着）。报告进台账，
    `status` 据此提示该去修哪一侧。

    返回 None 表示没有不一致（或无从对账）。
    """
    if milestone_doc is None:
        return None
    status = mechanism.get("status")
    if status in (None, "verified"):
        # verified 与课程表的 PASS 一致；unknown 不构成「相反的话」。
        return None
    if status == "unknown":
        return None

    milestone = milestone_of_decision(milestone_doc, decision_id)
    if milestone is None:
        return None
    gate = milestone.get("gate") or {}
    if not isinstance(gate, dict):
        return None

    claimed_pass = [
        str(line.get("id"))
        for line in gate.get("mechanism_lines", []) or []
        if isinstance(line, dict) and str(line.get("status", "")).upper() == "PASS"
    ]
    if not claimed_pass:
        return None

    return {
        "milestone": milestone.get("id", "<无 id>"),
        "claimed_pass": claimed_pass,
        "measured_status": status,
        "detail": (
            f"课程表的 {milestone.get('id')} 里机制行 {claimed_pass} 标着 "
            f"PASS，而本轮实测把机制判为 {status}。\n"
            "    两份文件在说相反的话，而课程表是唯一会被下一轮当作**前提**"
            "的那个产物 ——\n"
            "    读它的人会得到「这条机制已经核实」，于是后续决策建立在一个"
            "假前提上。\n"
            "    该修哪一侧取决于归因："
            f"{mechanism.get('attribution') or '未分流'}。\n"
            "      implementation → 修实现，课程表原样不动（那条 PASS 是"
            "对参考的正确读法）\n"
            "      curriculum     → reopen 那条决策，PASS 跟着一起作废"),
    }


def judge_structure(structure_report: Optional[Dict[str, Any]]
                    ) -> Dict[str, Any]:
    """结构核验的裁决：声称的结构在不在。

    这一关排在停顿核验**之前**，因为它能在任何性能数出来之前给出结论，
    而且它的结论优先：**未落地的结构优先于改验收措辞** —— 把阈值改宽
    不会让一个「不在」变成「在」。

    三态，各自对应完全不同的下一步：
      ok        结构都在，继续看停顿
      absent    结构不在 → **实现退化**。该修实现、或给实现补一张经验卡片，
                不该改课程表（归因错了会把改对的东西改坏）
      unchecked 没查或查不了 —— 不构成任何结论
    """
    if not isinstance(structure_report, dict):
        return {"status": "unchecked",
                "reason": "没有结构核验报告（record 未传 --structure-report）"}
    assertions = structure_report.get("assertions")
    if not isinstance(assertions, list) or not assertions:
        return {"status": "unchecked", "reason": "结构报告里没有任何断言"}

    failed = [a for a in assertions if isinstance(a, dict)
              and a.get("verdict") == "fail"]
    unknown = [a for a in assertions if isinstance(a, dict)
               and a.get("verdict") == "unknown"]
    if failed:
        names = [str(a.get("assertion")) for a in failed]
        return {
            "status": "absent",
            "failed": names,
            "reason": (f"{len(failed)} 条结构断言不成立（{names}）。"
                       "声称的结构没有被构建出来 —— 这是实现问题，"
                       "不是课程表对参考行为的解释错了"),
        }
    if unknown:
        names = [str(a.get("assertion")) for a in unknown]
        return {
            "status": "unchecked",
            "reason": (f"{len(unknown)} 条结构断言查不了（{names}）—— "
                       "「查不了」不等于「通过」"),
        }
    return {"status": "ok",
            "reason": f"{len(assertions)} 条结构断言全部成立"}


def judge_fence(fence_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """转发围栏的裁决：这一轮到底有没有被查过「是不是转发给参考」。

    围栏脚本本来就有（`fence_check.py`），文档也写明它排在任何计时之前 ——
    但做裁决的 `record` 此前**完全不看它**。于是这道纪律是纯自愿的：
    跑不跑、结果如何，对裁决没有任何影响。

    这正是本轮一直在修的那种形态：**一道看起来在、但关不上的门**。而这道
    门守的东西很具体 —— 候选直接调厂商实现时，加速比、精度、所有门禁全过，
    一行 kernel 都没写。自我提升场景更隐蔽：第 n 代转发给第 n−1 代，链条
    可以这样空转很多轮，每轮都「没有退化」。

    三态：
      clean      围栏跑过且干净
      violated   围栏命中 —— 这一轮的数不能用，无论它多好看
      unchecked  没传报告 —— 不构成结论，也不得抬高 best
    """
    if not isinstance(fence_report, dict):
        return {"status": "unchecked",
                "reason": "没有围栏报告（record 未传 --fence-report）"}
    findings = fence_report.get("findings")
    if findings is None:
        # 工具的输出里一定有 findings 这个键（空表表示干净）。没有它说明
        # 这不是一份 fence_check 的报告 —— 读成「干净」就是凭一份来源不明
        # 的 JSON 放行。
        return {"status": "unchecked",
                "reason": ("这份报告里没有 findings 键，不像是 fence_check.py "
                           "的输出。\n"
                           "    拿一份认不出的 JSON 当围栏证据，"
                           "和没有证据是同一件事")}
    if not isinstance(findings, list):
        return {"status": "unchecked",
                "reason": f"报告里的 findings 不是数组（{type(findings).__name__}）"}
    if findings:
        rules = sorted({str(f.get("rule")) for f in findings
                        if isinstance(f, dict)})
        return {
            "status": "violated",
            "rules": rules,
            "count": len(findings),
            "reason": (f"围栏命中 {len(findings)} 处（规则 {rules}）。"
                       "候选转发给了参考实现或厂商实现 —— "
                       "这一轮的加速比不是它自己挣来的"),
        }

    # 「阻断项为空」和「围栏干净」不是一回事。
    #
    # 通道项（dlopen / ctypes / 子进程 / 动态 import / 任意 aclnn 接口）不自动
    # 判不通过 —— 正当实现里也会出现。但它们是**按算子名写的那五条规则看不见
    # 的那条路**：`ctypes.CDLL("libopapi.so").aclnnXxx` 里那个名字是运行时拼
    # 出来的字符串，五条规则全绿而候选一行 kernel 都没写。
    #
    # 所以这里给第三态而不是直接 clean：裁决仍然放行（不压 best），但台账里
    # 留下「这一轮有未说明的通道」。把它读成 clean 等于让那条路永久免检。
    channels = fence_report.get("channels")
    if isinstance(channels, list) and channels:
        rules = sorted({str(f.get("rule")) for f in channels
                        if isinstance(f, dict)})
        return {
            "status": "clean_with_channels",
            "rules": rules,
            "count": len(channels),
            "reason": (f"围栏的阻断项为空，但有 {len(channels)} 处"
                       f"**能通向**厂商实现的路（{rules}）。\n"
                       "    这些构造在正当实现里也会出现，所以不判不通过；"
                       "但按算子名写的规则看不见它们 ——\n"
                       "    转发不需要写出名字。请在 milestone 或经验卡片里"
                       "说明每条通道通向哪里"),
        }
    return {"status": "clean",
            "reason": f"围栏跑过且干净（扫了 "
                      f"{fence_report.get('files_scanned', '?')} 个源文件）"}


#: 设备执行至少要占到墙上时间的多少。
#:
#: 低于这个比例时，`time_us` 量的主要不是 kernel ——  是 host 侧的下发、
#: 排队与同步。而机制核验读的是 msprof 的**设备侧**通道。两者量的东西
#: 几乎不相交时，「增益超噪声底 + 机制已核实」就是两个互不相关的事实被
#: 钉在了一起：那个百分比来自 host，那个机制结论来自 device。
#:
#: 20% 是本工具的工程判断，没有文档出处 —— 它只要能把「host 主导」这种
#: 定性情形拦下来就够了，不需要精确。
DEVICE_SHARE_MIN = 0.20


def reconcile_device_share(doc: Dict[str, Any],
                           signature: Optional[Dict[str, Any]]
                           ) -> Optional[Dict[str, Any]]:
    """把 msprof 的设备执行时长和评测的墙上时间对一次账。

    两个数都早就在手：`signature.launch.task_duration_us` 来自 msprof，
    `time_us` 来自评测的 NPU Event 计时。此前没有任何地方把它们放在一起看。

    返回 None 表示这笔账对不了（缺数），否则返回一份对账结果。
    对不了**不算通过** —— 调用方据此报告，不据此放行。
    """
    if not isinstance(signature, dict):
        return None
    launch = signature.get("launch")
    if not isinstance(launch, dict):
        return None
    field = launch.get("task_duration_us")
    device_us = field.get("value") if isinstance(field, dict) else field
    if not isinstance(device_us, (int, float)) or device_us <= 0:
        return None

    wall_us = _evolved_time_us(doc)
    if not isinstance(wall_us, (int, float)) or wall_us <= 0:
        return None

    share = float(device_us) / float(wall_us)
    return {
        "device_us": float(device_us),
        "wall_us": float(wall_us),
        "device_share": round(share, 4),
        "threshold": DEVICE_SHARE_MIN,
        "ok": share >= DEVICE_SHARE_MIN,
        "reason": (
            f"设备执行 {device_us:.1f}us / 墙上 {wall_us:.1f}us = "
            f"{share * 100:.1f}%"
            + ("" if share >= DEVICE_SHARE_MIN else
               f"，低于 {DEVICE_SHARE_MIN * 100:.0f}%。\n"
               "    这说明 time_us 量的主要不是 kernel，而是 host 侧的下发、"
               "排队与同步；\n"
               "    而机制核验读的是 msprof 的设备侧通道。两者量的东西几乎"
               "不相交，\n"
               "    「增益超噪声底 + 机制已核实」会变成两个互不相关的事实被"
               "钉在一起。\n"
               "    先把 host 开销摊掉（加大 num_trials、或一对 Event 套 N 次"
               "连发）再测")),
    }


def _claimed_position_matches_reference(op: str, threshold: float,
                                        reference_value: float) -> bool:
    """这条机制行声称的位置，和参考**实际**坐的位置对得上吗。

    这里刻意**不**用「参考是否满足这条行的阈值」来判。那个判法在最常见的
    写法上就错：课程表写 `< 4.2`、参考读数正好 4.2（要求实现追平参考），
    `4.2 < 4.2` 为假，于是一条完全正当的行会被判成「读错了参考」。

    真正要问的是条件原文那句话 ——「参考侧该字段**本来就不在**声称的位置」。
    也就是说，这是一个「差多远」的问题，不是一个「满不满足」的问题。
    所以复用已有的带宽常数 MECHANISM_BAND_FACTOR（1.25，比噪声底高一档）：

      上界型（< / <=）  阈值 ≥ 参考 / 1.25
                        → 声称的位置就在参考附近（或更松），读法没问题
                        → 阈值 < 参考 / 1.25 则是在说「参考坐得比它实际
                          低得多」，那是读错了参考，而且会让实现去追一个
                          参考自己都达不到的数

      下界型（> / >=）  方向相反：阈值 ≤ 参考 × 1.25

    `==` / `!=` 不蕴含方向，按「数量级对得上」处理。
    """
    if reference_value <= _MOVE_EPS:
        # 参考侧本来就接近 0。这种通道上「声称的位置」无从比较 ——
        # 任何上界都在它之上，任何下界都在它之下。
        return True
    if op in ("<", "<="):
        return threshold >= reference_value / MECHANISM_BAND_FACTOR
    if op in (">", ">="):
        return threshold <= reference_value * MECHANISM_BAND_FACTOR
    return (reference_value / MECHANISM_BAND_FACTOR <= threshold
            <= reference_value * MECHANISM_BAND_FACTOR)


def attribute_mechanism_failure(
        results: List[Dict[str, Any]],
        reference_snapshot: Optional[Dict[str, float]]) -> Dict[str, Any]:
    """机制没落地时，该修的是实现还是课程表。

    上游的分流规则写得很具体：

      > 声称无冲突布局而冲突仍高 ⇒ **学生没做到（退化回退）⇒ 加知识库卡片，
      > 不要改课程表**；只有「参考侧该字段本来就不在声称的位置」才是课程表缺陷

    判据因此是**参考快照**，而不是本轮读数：

      声称的位置与参考实际位置对得上（差在 1.25 倍带内）
        → 课程表对参考的读法是对的；没做到的是实现
        → `implementation`（补一张经验卡片，别动课程表）

      对不上
        → 课程表声称参考坐在一个它实际不在的位置上。这不只是记错一个数 ——
          实现会被要求去追一个参考自己都达不到的目标
        → `curriculum`（该重开那条决策）

      参考侧没这个读数，或这条行本身解析不成一个可核验的断言
        → 判不了。这里取 `implementation`，理由见下。

    为什么判不了时偏向 implementation：两种误判的代价不对称。

      判成 curriculum 而其实是实现  架构层被连续叫回来重写课程表，而下一轮
                                    实现还是同一个退化回退 —— 改对的东西被
                                    改坏、改坏的东西没人碰。这是实测观察到的
                                    那条路。
      判成 implementation 而其实是课程表  实现失败计数爬到阈值 → 状态转
                                    `blocked` → **停下来交给人**。

    后者是一个会响的失败，前者是一个不会响的失败。所以默认偏向前者。
    """
    offenders = [r for r in results
                 if r.get("verified") is False or r.get("vacuous")]
    if not offenders:
        return {"attribution": None, "reason": "没有不成立的机制行"}

    misread: List[str] = []
    not_delivered: List[str] = []
    untriageable: List[str] = []

    for item in offenders:
        channel = item.get("channel")
        threshold = item.get("threshold")
        op = item.get("operator")
        line_id = str(item.get("id"))
        if (not isinstance(channel, str) or op not in _COMPARATORS
                or not isinstance(threshold, (int, float))):
            # 连「这条行想断言什么」都没解析出来（表达式形状不对、多子句、
            # 同名通道歧义）。分流不了 —— 归进 untriageable，不要当成
            # 「读错了参考」：那会把一条写坏的行当成架构层的错。
            untriageable.append(line_id)
            continue
        reference_value = (reference_snapshot or {}).get(channel)
        if not isinstance(reference_value, (int, float)):
            untriageable.append(line_id)
            continue
        if _claimed_position_matches_reference(op, float(threshold),
                                               float(reference_value)):
            not_delivered.append(line_id)
        else:
            misread.append(line_id)

    if misread:
        # 只要有一条是「读错了参考」，就该回架构层 —— 那条行的前提是错的，
        # 在它底下调实现是白费。
        return {
            "attribution": "curriculum",
            "misread_lines": misread,
            "not_delivered_lines": not_delivered,
            "untriageable_lines": untriageable,
            "reason": (f"机制行 {misread} 声称的位置**参考自己就不在**"
                       f"（差出 {MECHANISM_BAND_FACTOR} 倍带外）。\n"
                       "    这不只是记错一个数：实现会被要求去追一个参考"
                       "自己都达不到的目标。\n"
                       "    这是课程表对参考行为的解释错了 —— 该重开那条决策，"
                       "在一个错的前提底下调实现是白费。"),
        }
    return {
        "attribution": "implementation",
        "not_delivered_lines": not_delivered,
        "untriageable_lines": untriageable,
        "reason": (
            (f"机制行 {not_delivered} 声称的位置与参考实际位置对得上 —— "
             "课程表的读法没问题，没做到的是实现。\n"
             if not_delivered else "")
            + (f"机制行 {untriageable} 分流不了（参考侧没有同通道读数，"
               "或这条行解析不成可核验的断言）；按实现失败处理。\n"
               "    两种误判的代价不对称：判成课程表缺陷会让架构层连续重写"
               "课程表，而实现\n"
               "    还是同一个退化回退（一个不会响的失败）；判成实现失败"
               "会把状态推到 blocked\n"
               "    并交给人（一个会响的失败）。\n"
               if untriageable else "")
            + "    补一张经验卡片记下这个退化回退，**不要改课程表**。"),
    }


def judge_mechanism(milestone_doc: Optional[Dict[str, Any]],
                    signature: Optional[Dict[str, Any]],
                    decision_id: str,
                    previous_snapshot: Optional[Dict[str, float]] = None,
                    reference_snapshot: Optional[Dict[str, float]] = None,
                    ) -> Dict[str, Any]:
    """机制裁决：本次改动有没有把它声称的机制做出来。

    没给 milestone 或 signature 时返回 unknown —— 工具不替调用方假设
    「没查就是通过」。

    四种状态：
      verified  全部机制行成立且字段确实朝声称方向移动
      refuted   有机制行阈值不成立 —— 机制没落地
      vacuous   阈值成立但字段没动 —— 不是本次改动造成的，平凡通过
      unknown   无从核验
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

    results = [verify_mechanism_line(line, signature, previous_snapshot,
                                    reference_snapshot=reference_snapshot)
               for line in lines]

    if all(r["verified"] is True for r in results):
        status = "verified"
        reason = f"{len(results)} 条机制行全部成立，且字段确实朝声称方向移动"
    elif any(r["verified"] is False and not r.get("vacuous") for r in results):
        status = "refuted"
        failed = [r["id"] for r in results
                  if r["verified"] is False and not r.get("vacuous")]
        reason = (f"机制行 {failed} 阈值不成立 —— 这次的增益来自别处，"
                  "不能把它附带的解释当成已验证的机制")
    elif any(r.get("vacuous") for r in results):
        status = "vacuous"
        vacuous = [r["id"] for r in results if r.get("vacuous")]
        reason = (f"机制行 {vacuous} 是平凡通过的：阈值成立，但那个字段"
                  "并非因本次改动而变化。增益同样另有来源")
    else:
        status = "unknown"
        pending = [r["id"] for r in results if r["verified"] is None]
        reason = f"机制行 {pending} 无法求值，未构成验证"

    out: Dict[str, Any] = {"status": status, "lines": results,
                           "reason": reason}

    # 机制没落地时，把「该修实现」和「该改课程表」分开。
    #
    # 此前这里什么都不分 —— refuted / vacuous 一律落进 streak_refuted，
    # 也就是默认判「课程表解释错」，于是架构层被连续叫回来重写课程表，
    # 而下一轮实现还是同一个退化回退。structure_probe 这条路能分流，
    # 但它要 --structure-report，是可选参数：没传时默认路径仍然走反。
    #
    # 现在分流不依赖任何可选输入 —— 判据是参考快照，而它在 init 就必填。
    if status in ("refuted", "vacuous"):
        triage = attribute_mechanism_failure(results, reference_snapshot)
        if triage["attribution"] == "implementation":
            out["attribution"] = "implementation"
            out["reason"] = reason + "\n  " + triage["reason"]
        elif triage["attribution"] == "curriculum":
            out["attribution"] = "curriculum"
            out["reason"] = reason + "\n  " + triage["reason"]
        out["triage"] = triage

    return out


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


#: 连续多少次无果的重开之后拒绝再开。
#:
#: 上游取 2（`reopen_max_fruitless`），并且**计数从磁盘读** —— 进程重启后
#: 仍然有效。我们这边同理：计数存进 state，不随任何一次 reopen 归零。
#: 不设上限的话「重开 → 几轮不够 → 停滞 → 再重开」可以永久循环，而每一轮的
#: status 看起来都健康（三个计数器都是 0/3）。
REOPEN_MAX_FRUITLESS = 2

#: 重开定价的形状：`<Δ百分比> via <来源>`。
#:
#: 上游的规矩是「没有价格就没有重开」：`reopen_price: <Δ> via
#: probe:<path>#<n> | lever:<name> | KB:<entry>`，拿它去比对本步探针读数、
#: 定价表或知识库条目，对不上就拒绝、记 UNPRICED_REOPEN、课程表原样不动。
#: 实证：一次审计凭「等待指令落在哪里」这种推断翻掉了分段切分轴，整组决策
#: 重做，而那次重开本身从未被定价。
RE_REOPEN_PRICE = re.compile(
    r"^\s*(?P<delta>-?\d+(?:\.\d+)?)\s*%?\s*via\s+"
    r"(?P<kind>lever|probe|kb|ablation)\s*[:：]\s*(?P<ref>\S.*?)\s*$",
    re.IGNORECASE)


#: 封存清单的文件名。由 seal_reference.py 定义，两边必须一致。
MANIFEST_FILENAME = "sealed_manifest.json"


def _work_dir_from_state(state: Dict[str, Any],
                         state_path: Path) -> Optional[str]:
    """封存工作目录在哪。顺序与 _find_sealed_list 一致 —— 两处找的必须是
    同一个目录，否则自检校验的和 hook 拦的不是同一份封存。"""
    recorded = state.get("sealed_paths")
    if isinstance(recorded, str) and recorded.strip():
        return str(Path(recorded).resolve().parent)
    env_dir = os.environ.get("FORGE_WORK_DIR")
    if env_dir:
        return env_dir
    return str(state_path.resolve().parent)


def check_reference_unchanged(work_dir: Optional[str]) -> List[str]:
    """裁决之前确认参考没变过。返回问题列表，空表示没变。

    为什么要在 record 这一刻查：**本轮的加速比是相对参考算的**。参考变了，
    这个百分比就不再是它声称的那个意思 —— 而变化是静默的：结果文件的形状、
    字段、cv 全都正常。

    三种变化里，「新增文件」那一种最不像问题却最危险：新文件没进封存清单，
    于是 hook **不会拦**对它的读取 —— 参考树上多出一个 agent 可以随便读的
    源文件，而所有「比对已记录的那些」的校验全绿。`opp_install` 那类来源
    尤其容易碰上：那是一个活的安装目录，另一个任务重装一次算子包就会往里
    加文件。

    拿不到工作目录或清单时返回空表 —— 这一条不硬拒。封存清单是 `seal` 的
    产物，而有些正当路径（内置实现作参考、老 state）本来就没有它。硬拒会
    把那些用法一起卡住，而人会反射性地去绕过参数。缺失由调用方告警。
    """
    if not work_dir:
        return []
    manifest_path = Path(work_dir) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"封存清单 {manifest_path} 读不出来：{exc}。"
                "读不出来就无法确认参考没变过，而本轮的加速比是相对它算的"]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import seal_reference as _seal  # noqa: PLC0415

    drifted = _seal.verify_manifest(manifest)
    if not drifted:
        return []
    lines = [f"参考已发生 {len(drifted)} 处变化，本轮裁决不成立："]
    for item in drifted[:6]:
        lines.append(f"    {item['relative']}: {item['reason']}")
    lines.append("    本轮的加速比是相对参考算的 —— 参考变了，"
                 "这个百分比就不再是它声称的那个意思。")
    lines.append("    重新跑一次 seal，并重测噪声底与参考签名。")
    return ["\n".join(lines)]


def _load_hwkb():
    """按路径装载 forge-hwkb 的 hwkb.py。

    卡片的 frontmatter 解析、认证字段的判据都在那边。在这里重写一份的
    代价不是多几十行，而是两份判据会分头漂移 —— 这类分头漂移正是 L-DENY
    当初失效的形态（写方和读方各拼各的路径）。
    """
    path = (Path(__file__).resolve().parent.parent.parent
            / "forge-hwkb" / "scripts" / "hwkb.py")
    if not path.is_file():
        raise CampaignError(f"找不到 {path}，无法核对知识库卡片")
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location("hwkb", path)
    if spec is None or spec.loader is None:
        raise CampaignError(f"{path} 无法装载")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: 从 certified_by.delta 里抠出一个百分数。
#:
#: 卡片的 delta 是人写的自由文本（`"-24.3%"`、`"mte2_pct 61.2 -> 31.0"`、
#: `"总时间 -24.3%"`），所以这里只认带百分号的那一种。抠不出来就当作
#: 「这张卡片没有记下可比的量」—— 不是当作 0，也不是当作通过。
RE_CARD_DELTA_PCT = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")


def find_kb_card(kb_root: Path, card_id: str,
                 arch_dirs: List[str]) -> Optional[Path]:
    """在知识库里按 id 找卡片。不猜架构 —— 挨个架构目录找。

    目录名由调用方从 hwkb.ARCH_DIRS 取，不在这里再抄一份：抄一份就会有
    两个「架构目录叫什么」的真源，而它们分头漂移时这里只会悄悄找不到卡片。
    """
    for sub in arch_dirs:
        candidate = kb_root / sub / f"{card_id}.md"
        if candidate.is_file():
            return candidate
    return None


def verify_kb_price(ref: str, delta: float,
                    kb_root: Optional[Path]) -> Tuple[bool, str]:
    """知识库卡片能不能当一次重开的价格。

    这是「读知识库能省一次 GPU job」那条经济激励的落点 —— 但它只有在
    卡片真的被核对过时才成立。上游那句话说得很直接：**一条转述过来的
    处方不是价格**。先前这里无条件返回通过，附一句「请确认该卡片确实带
    探针认证」—— 把判定交给了那个正想重开的人，而他没有任何动机去查。

    三件事都成立才通过：
      卡片在        知识库里真有这个 id
      带认证        certified_by 齐全，且该张卡片本身过准入校验
      量对得上      卡片记下的实测变化量 ≥ 这次声称的 Δ
                    （卡片只能为它测过的那个幅度背书）
    """
    if kb_root is None:
        return False, (
            f"引用了知识库条目 {ref}，但没给 --kb-root，无法核对。\n"
            "    一条转述过来的处方不是价格 —— 它要么被核对过，"
            "要么不算数。\n"
            "    加 --kb-root <forge-hwkb/references>，"
            "或改用一次 lever/probe 的实测读数。")
    if not kb_root.is_dir():
        return False, f"知识库根 {kb_root} 不存在"

    try:
        hwkb = _load_hwkb()
    except CampaignError as exc:
        return False, str(exc)

    card_path = find_kb_card(kb_root, ref, list(hwkb.ARCH_DIRS.values()))
    if card_path is None:
        return False, (
            f"知识库里没有卡片 {ref}（找过 "
            f"{sorted(hwkb.ARCH_DIRS.values())} 各目录）。\n"
            "    引用一张不存在的卡片，和没有价格是同一件事。")

    try:
        meta, body = hwkb.load_card(card_path)
    except ValueError as exc:
        return False, f"卡片 {card_path} 读不出来: {exc}"

    issues = hwkb.check_admission(meta, body, kb_root)
    issues += hwkb.check_certification_against_ledger(meta)[0]
    if issues:
        return False, (
            f"卡片 {ref} 自己不过准入校验（{len(issues)} 项）：\n    "
            + "\n    ".join(str(i) for i in issues[:4])
            + "\n    不过准入的卡片不能当价格 —— 它背后那次测量是否成立"
              "正是准入在查的事。")

    recorded = str((meta.get("certified_by") or {}).get("delta", ""))
    match = RE_CARD_DELTA_PCT.search(recorded)
    if not match:
        return False, (
            f"卡片 {ref} 的 certified_by.delta 是 {recorded!r}，"
            "里面没有百分数。\n"
            "    它可能记了别的量（比如 'mte2_pct 61.2 -> 31.0'），"
            "那是机制证据，不是时间增益。\n"
            "    用它定价需要一个时间上的百分比；没有就得重新测一次。")

    card_delta = abs(float(match.group(1)))
    if delta > card_delta + 1e-9:
        return False, (
            f"声称 Δ {delta}%，但卡片 {ref} 记下的只有 {card_delta}%。\n"
            "    卡片只能为它测过的那个幅度背书。要更大的数就得自己测。")

    return True, (f"卡片 {ref} 带认证且过准入，记录幅度 {card_delta}% "
                  f"≥ 声称的 {delta}% —— 这次重开不必另测一遍")


def verify_reopen_price(price: str, state: str,
                        lever_store: Optional[str] = None,
                        kb_root: Optional[str] = None) -> Tuple[bool, str]:
    """核对重开价格：这个 Δ 是测出来的，还是写上去的。

    返回 (是否通过, 说明)。通过的条件是三件事都成立：
      形状对  `<Δ%> via lever:<名字>` 这种形状，能解析出数值与来源
      引用在  来源确实存在（定价表里有这个手段 / 知识库里有这条卡片）
      超噪声  Δ 必须超过噪声底，否则这个「价格」本身就在噪声里
    """
    match = RE_REOPEN_PRICE.match(price or "")
    if not match:
        return False, (
            f"价格 {price!r} 不是 `<Δ%> via <来源>` 的形状。\n"
            "    来源只能是 lever:<手段名> / probe:<路径> / kb:<卡片 id> / "
            "ablation:<消融名>。\n"
            "    例：--price \"8.4% via lever:ub_resident\"")

    delta = abs(float(match.group("delta")))
    kind = match.group("kind").lower()
    ref = match.group("ref")

    state_path = Path(state)
    noise = 0.0
    try:
        doc = json.loads(state_path.read_text(encoding="utf-8"))
        noise = float(doc.get("noise_floor_pct", 0.0) or 0.0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    if noise > 0 and delta <= noise:
        return False, (
            f"价格 {delta}% 没有超过噪声底 {noise}% —— "
            "这个「价格」本身就落在测量抖动里。\n"
            "    重开要花掉整条下游的实现，理由不能是一个测不出来的数。")

    if kind == "lever":
        store_path = (Path(lever_store) if lever_store
                      else state_path.parent / "levers.json")
        if not store_path.is_file():
            return False, (
                f"找不到定价表 {store_path} —— 无法核对手段 {ref!r} 的价格。\n"
                "    先用 lever_probe.py 给现有手段定价（这也是停滞之后的"
                "第一件事），\n"
                "    或用 --lever-store 指定定价表位置。")
        try:
            store = json.loads(store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"定价表 {store_path} 读不出来: {exc}"
        names: Set[str] = set()
        for variant in store.get("variants") or []:
            if isinstance(variant, dict):
                for lever in variant.get("levers") or []:
                    names.add(str(lever))
        if ref not in names:
            return False, (
                f"定价表里没有手段 {ref!r}；已登记的是 {sorted(names)}。\n"
                "    引用一个没测过的手段，等于把推断当价格用。")
        return True, f"手段 {ref} 在定价表里，价格 {delta}% 超过噪声底 {noise}%"

    if kind == "kb":
        return verify_kb_price(ref, delta,
                               Path(kb_root) if kb_root else None)
    return True, f"引用 {kind}:{ref}（Δ {delta}%）"


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

    # 原子组成员要互相牵连。
    #
    # `atomic_group` 标的是「必须一次提交落地」的决策集，V9 校验它们同属一个
    # milestone，`forge-coder.md` 也明说任何真子集单独落地都会退化。但闭包
    # 早先只走 depends_on —— 于是重开组里一个成员，兄弟成员的台账仍然有效，
    # 树上留着一个原子组的真子集，而退化会被归因给刚重开的那条决策。
    #
    # 上游把 companions 与 depends_on 并列进不动点（invalidation.py），
    # 理由写在注释里：companions 必须落在同一个 commit 里。
    groups: Dict[str, List[str]] = {}
    for did, decision in decisions.items():
        group = decision.get("atomic_group")
        if isinstance(group, str) and group.strip():
            groups.setdefault(group.strip(), []).append(did)
    for members in groups.values():
        for a in members:
            for b in members:
                if a != b and b not in reverse[a]:
                    reverse[a].append(b)

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


def rollback_target(state: Dict[str, Any],
                    invalidated: Set[str]) -> Optional[Dict[str, Any]]:
    """重开之后，实现该从哪一份产物接着做。

    为什么要指出来：重开把整条下游的台账作废了，但**工作树还在原地** ——
    上面带着那些刚被作废的改动。编码层如果从当前树接着改，它就是在一个
    已经被判定建立在错假设上的实现之上继续叠加，而归因会把后面的结果算到
    新架构假设头上。

    取「作废范围之外、最后一个机制已核实的产物」：
      作废范围之外   在范围内的那些正是被判定不可信的
      机制已核实     `counts_toward_best` 为真的那些 —— 只是变快了的不算，
                     那种产物自己就解释不清
      最后一个       时间上最近的，也就是走得最远的那个起点

    找不到就返回 None，调用方据此说「从参考基线重新开始」。不要在这里编一个
    看起来合理的回退点 —— 一个错的回退目标比没有目标更糟。
    """
    best_entry: Optional[Dict[str, Any]] = None
    for entry in generation_entries(state):
        if entry.get("invalidated"):
            continue
        if entry.get("decision") in invalidated:
            continue
        if not entry.get("counts_toward_best"):
            continue
        if not entry.get("artifact_sha"):
            # 没有代码身份的记录指不出任何具体产物。它可以是个合格的读数，
            # 但不能当回退目标 —— 回退目标必须是一份能被找回来的东西。
            continue
        best_entry = entry
    if best_entry is None:
        return None
    return {
        "decision": best_entry.get("decision"),
        "artifact_sha": best_entry.get("artifact_sha"),
        "gain": float(best_entry.get("gain") or 1.0),
        "at": best_entry.get("at"),
    }


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
        entry.get("decision") for entry in generation_entries(state)
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
    pending = state.get("pending_confirmation")
    if pending:
        return {
            "action": "confirm-best",
            "reason": (f"决策 {pending.get('decision')} 的增益 "
                       f"{pending.get('gain')} 已达目标，但只测过一次。"
                       + str(pending.get("reason", ""))),
        }

    # M0 对账没做过 / 没通过 → 先把它做完。
    #
    # 为什么排在这么前面：它查的是整个 campaign 里最容易悄悄出错的那件事 ——
    # 课程表漏掉了某个超带通道。漏掉的后果不是少一条门禁，而是**门禁永远
    # 不会因它触发**：门禁只拿课程表自己声明的值做比较。于是那个通道上的
    # 问题可以一路活到 promote，而每一轮的数都正常。
    #
    # 做成「下一步动作」而不是硬停机：硬停会在 M0 还没跑完任何一轮时就把
    # 流程卡住（那时还没有本轮签名可对账），而一个在正当起点上就报错的
    # 门禁会被反射性绕过。
    audit = state.get("m0_audit")
    phase_now = state.get("state")
    if phase_now in (STATE_IMPLEMENT, STATE_SHIPPED) and (
            not isinstance(audit, dict)
            or int(audit.get("generation", -1)) != int(
                state.get("generation", 0))):
        if any(e.get("counts_toward_best") for e in generation_entries(state)):
            return {
                "action": "audit",
                "reason": ("本代还没做过 M0 对账。门禁只拿课程表自己声明的"
                           "值做比较 —— 课程表漏掉的通道不会触发任何东西，"
                           "而那种漏可以一路活到 promote，每轮的数都正常。"
                           "跑 `campaign.py audit --signature <本轮签名>`。"),
            }
    if isinstance(audit, dict) and audit.get("passed") is False \
            and int(audit.get("generation", -1)) == int(
                state.get("generation", 0)):
        return {
            "action": "audit",
            "reason": (f"M0 对账未通过：{len(audit.get('unclaimed') or [])} "
                       "个通道差了带外而课程表里没有机制行锚着它们"
                       f"（{audit.get('unclaimed')}）。"
                       "三择一：补机制行 / 补 hwkb 卡片 / reopen。"),
        }

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
    if phase == STATE_BLOCKED:
        return {
            "action": "fix-implementation",
            "reason": (state.get("stall_detail")
                       or "连续实现失败，需要人介入；这不消耗架构重入机会"),
        }
    if phase == STATE_STALLED:
        reasons = state.get("stall_reasons") or [{
            "kind": state.get("stall_reason", STALL_NO_IMPROVEMENT),
            "detail": state.get("stall_detail", ""),
        }]
        titles = {
            STALL_NO_IMPROVEMENT: "内层无进展",
            STALL_GRAPH_EXHAUSTED: "决策图走穷",
            STALL_GATE_IMPASSE: "门禁僵持",
            STALL_MECHANISM_UNVERIFIED: "机制无法核验",
            STALL_MECHANISM_REFUTED: "机制被证伪",
            STALL_IMPL_FAILURE: "连续实现失败",
        }
        listed = "；".join(
            f"{titles.get(r['kind'], r['kind'])}（{r['detail']}）"
            for r in reasons)
        kinds = {r["kind"] for r in reasons}
        if STALL_MECHANISM_REFUTED in kinds:
            tail = ("交回架构层：先重新分析参考、修正机制假设 —— "
                    "连续多次「更快但机制不成立」说明对参考行为的解释错了，"
                    "补参数重跑解决不了")
            action = "re-analyze"
        elif kinds == {STALL_MECHANISM_UNVERIFIED}:
            # 这一种不必惊动架构层：课程表没问题，是调用少了参数。
            # 动作标签也要说准 —— 它是机器可读的那一个，标成 re-analyze
            # 会让架构层白跑一轮分析。
            tail = ("补齐 record 的 --signature / --milestone 后重跑 —— "
                    "这是流程缺口，不必改课程表")
            action = "fix-invocation"
        else:
            tail = ("交回架构层：做消融定价、必要时用 reopen "
                    "重开 M0 的架构决策，而不是在当前架构下继续硬调")
            action = "re-analyze"
        return {"action": action, "reason": f"{listed}。{tail}"}
    return {"action": "implement", "reason": "继续实现当前决策行"}


# ---------------------------------------------------------------------------
# 状态读写
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _elapsed_hours(state: Dict[str, Any]) -> Optional[float]:
    """自 init 起过了多少小时。算不出返回 None。

    算不出的原因可以是老 state 没有 created_at、或者那个字段被手改坏了。
    两种都返回 None 而不是 0 —— 返回 0 会让墙钟预算这道闸静默失效，
    而失效时输出照样是「预算充裕」。
    """
    raw = state.get("created_at")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        started = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - started
    return delta.total_seconds() / 3600.0


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
              budget: Optional[int] = None,
              reference_snapshot: Optional[Dict[str, float]] = None,
              device: Optional[Dict[str, Any]] = None,
              wall_hours: Optional[float] = None
              ) -> Dict[str, Any]:
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
        # 参考自己的架构签名快照 = 第 0 轮。机制行做差时若没有更近的
        # 有效台账快照，就拿它当基线。
        "reference_snapshot": reference_snapshot or None,
        # 代际之间**不换卡**：换了之后参考签名、噪声底、绝对耗时全部
        # 不可比，而这些不可比是静默的 —— 数字照常产出，只是错了。
        "device": device or {"device_id": None},
        "target_gain": target_gain,
        "budget": {"max_records": budget, "used": 0,
                   "max_wall_hours": wall_hours},
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
            # 当前这段连续里，两种成因各占多少。两者该做的事不同，
            # 不拆开就只能给出一句笼统的提示，而那句提示会把架构层
            # 引向错误的动作。
            "streak_refuted": 0,
            "streak_unknown": 0,
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


#: 评测自己判定「这次跑成了」的 gating 取值。其余一律不接受。
#:
#: 这两个之外还有 failed / target_regression / generalization_regression。
#: 不检查 gating 的后果很具体：泛化 shape 上算错或退化时，评测已经判了
#: failed / generalization_regression，而 comparison.precision_passed 和
#: comparison.speedup 都只覆盖 target shape —— 于是裁决照样 accept 并抬高 best。
ACCEPTABLE_GATINGS = frozenset({"fully_passed", "partial_passed"})


def check_result_admissible(doc: Dict[str, Any],
                            state: Dict[str, Any]) -> List[str]:
    """结果文件能不能进裁决。返回拒绝理由列表，空列表表示可以。

    裁决权在脚本手里，但输入权在调用方手里。这三条查的是「这份结果是不是
    本次 campaign 的、在对的卡上、跑成了的那一次测量」—— 不查的话，裁决
    严格建立在一份没人验证过来源的 JSON 上。
    """
    problems: List[str] = []

    gating = doc.get("gating")
    if gating is not None and gating not in ACCEPTABLE_GATINGS:
        problems.append(
            f"评测自己判定 gating={gating!r}，不是 "
            f"{sorted(ACCEPTABLE_GATINGS)} 之一。\n"
            "    failed = 编译/精度/运行出错；*_regression = 某个 shape 退化。\n"
            "    注意 comparison.precision_passed 与 comparison.speedup 只覆盖 "
            "target shape，\n"
            "    泛化 shape 上的算错与退化只体现在 gating 里 —— "
            "不看它就会把这两种情况当成功收下")

    result_device = _result_device_id(doc)
    state_device = (state.get("device") or {}).get("device_id")
    if result_device is not None and state_device is not None \
            and int(result_device) != int(state_device):
        problems.append(
            f"这份结果测在 device {result_device}，而本 campaign 绑的是 "
            f"device {state_device}。\n"
            "    换卡之后参考签名、噪声底、绝对耗时全部不可比，"
            "而这种不可比是静默的 —— 所以这里拒绝")

    # baseline 不许来自缓存。
    #
    # 配对计时的全部纪律都建立在「两侧在同一个 job 里、位置互为镜像」之上：
    # burn-in 把器件加热到稳态，A-B-B-A 抵掉位置偏置（实测 2.7%，方向恒定、
    # 多测不收敛，而目标增益只有 5%）。
    #
    # baseline 一旦来自上一次 job 的缓存，这两个机制同时失效 —— 而且是静默
    # 失效：结果文件的形状、字段、cv 都完全正常，只是那个 baseline 是在另一
    # 个温度、另一个位置、可能另一个时间点测的。于是 2.7% 的偏置原封不动地
    # 回到了一个 5% 的判据里。
    measurement = doc.get("measurement")
    if isinstance(measurement, dict) and measurement.get("baseline_from_cache"):
        problems.append(
            "这份结果的 baseline 来自缓存（measurement.baseline_from_cache "
            "为真）。\n"
            "    配对计时要求两侧在同一个 job 里、位置互为镜像 —— "
            "缓存的 baseline 让 burn-in 与\n"
            "    A-B-B-A 镜像同时失效，而失效是静默的：字段、cv 全都正常，"
            "只是那个数来自\n"
            "    另一个温度、另一个位置。实测位置偏置 2.7%（方向恒定、"
            "多测不收敛），\n"
            "    而目标增益只有 5% 这个量级。\n"
            "    去掉 --baseline-cache 重测一次。")

    return problems


def _result_device_id(doc: Dict[str, Any]) -> Optional[int]:
    for key in ("device_id", "device"):
        value = doc.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, dict) and isinstance(value.get("device_id"), int):
            return value["device_id"]
    return None


def result_cv_values(doc: Dict[str, Any]) -> List[float]:
    """这份结果里各 shape 的变异系数。用来复核存量噪声底还成不成立。"""
    out: List[float] = []
    shape_results = doc.get("shape_results")
    if isinstance(shape_results, dict):
        for group in shape_results.values():
            if not isinstance(group, list):
                continue
            for item in group:
                if isinstance(item, dict):
                    value = item.get("cv_pct")
                    if isinstance(value, (int, float)) and value > 0:
                        out.append(float(value))
    return out


def precision_from_report(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """读误差比报告。

    这条判据优先于评测结果里的容差比对，原因有两个：

    1. 容差比对分不出「在参考舍入的地方舍入」和「用 bf16 累加 / tf32 相乘」
       —— 1e-3 下两者都过。
    2. 容差比对的锚是 baseline 的输出，而 promote 之后 baseline 变成上一代
       产物，锚点跟着移动：**每代允许再漂 1e-3，N 代之后与数学真值的距离
       无界，而每一代都显示「精度通过」**。误差比的锚是 float64 真值，
       与代际无关。

    `unavailable` 不是通过。没有参考语义时这一轮的精度**没有被判过**。
    """
    if not isinstance(report, dict):
        return {"status": "absent",
                "reason": "没有误差比报告（record 未传 --precision-report）"}
    verdict = report.get("verdict")
    if verdict not in ("pass", "fail", "unavailable"):
        return {"status": "absent",
                "reason": f"误差比报告里的 verdict 不认识: {verdict!r}"}
    return {
        "status": verdict,
        "reason": str(report.get("reason", "")),
        "anchor": report.get("anchor"),
    }


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

def generation_entries(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """只取**本代**的台账条目。

    台账是跨代累积的 —— promote 不清空它，这是对的：要能回看整条自我提升
    链条。但「读台账回答关于本代的问题」时必须按代过滤，因为 promote 把
    参考换成了上一代的产出，于是跨代的数全部不可比：

      best（增益与耗时）  上一代的 1.4x 是相对**旧参考**的。混进本代会让
                          一个还没开始优化的新代直接带着 1.4x，而 promote
                          正是特意把 best 归零的。
      机制做差的基线      上一代的签名来自另一个二进制。拿它做差，两代之间
                          的差异会被算到本代某一次改动头上。
      决策图是否走穷      上一代试过的决策在本代是全新的（参考变了，同一个
                          决策的结论可能完全不同）。不过滤的话，promote 之后
                          第一次 record 就会报「每条决策都已试过」。
      机制是否已核实      上一代核实过不代表本代核实过。不过滤会让一个零核实
                          的代被封进产物库时标上 mechanism_verified=true。

    条目里没有 generation 字段时当作本代。那种条目只会出现在手工改过的
    state 里；当作本代是保守选择 —— 宁可多算一条，不要静默丢掉证据。
    """
    current = int(state.get("generation", 0) or 0)
    out: List[Dict[str, Any]] = []
    for entry in state.get("ledger", []):
        if not isinstance(entry, dict):
            continue
        gen = entry.get("generation")
        if isinstance(gen, (int, float)) and int(gen) != current:
            continue
        out.append(entry)
    return out


#: 改动被保留下来的裁决。只有这些代表「当前代码的样子」。
#:
#: revert / regression 的代码已经撤回，它们的签名记录的是一个**不复存在**
#: 的状态。拿它当基线，下一轮真正的改动会被算成「没动」—— 而那一轮
#: 恰恰是有效的。
KEPT_VERDICTS = frozenset({VERDICT_ACCEPT, VERDICT_ACCEPT_UNVERIFIED})


def previous_snapshot(state: Dict[str, Any]) -> Tuple[Optional[Dict[str, float]], str]:
    """取做差用的「上一轮」快照，并说明它是哪一轮。

    「上一轮」指的是**当前代码所处的状态**，所以只有被保留下来的改动才算：

      accept / accept_unverified   代码留下了，是当前状态的一部分
      revert / regression          代码撤回了，那个签名对应的状态已不存在
      invalidated（reopen 过）     建立在旧架构假设上

    漏掉这个判断的后果很具体：某一轮因增益落在噪声里被回滚，但它碰巧把
    某个停顿压到了低位；下一轮真的把那个停顿从高位压下来时，会拿已撤回
    那轮的低位值当基线，算出「没动」，于是一个真实的机制被判成平凡通过。

    都没有就退回参考自己的签名 —— 那就是第 0 轮。
    """
    for entry in reversed(generation_entries(state)):
        if entry.get("invalidated"):
            continue
        if entry.get("verdict") not in KEPT_VERDICTS:
            continue
        snapshot = entry.get("signature_snapshot")
        if snapshot:
            label = f"上一轮（决策 {entry.get('decision')}）"
            sha = entry.get("artifact_sha")
            if sha:
                label += f"，产物 {sha}"
            return snapshot, label
    reference = state.get("reference_snapshot")
    if reference:
        return reference, "参考实现"
    return None, "无"


def _milestone_fingerprint(milestone_doc: Optional[Dict[str, Any]]) -> Optional[str]:
    """课程表的内容指纹。判据改了要能看出来。"""
    if milestone_doc is None:
        return None
    body = json.dumps(milestone_doc, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def directory_fingerprint(path: Path) -> str:
    """一个候选目录的内容指纹。

    用它代替 git sha：候选目录不一定在版本控制里（评测要的是装好的 OPP
    安装路径），但「这一轮测的是哪一份代码」必须可核实。
    按相对路径排序后把每个文件的路径与内容摘要串起来。
    """
    digest = hashlib.sha256()
    if path.is_file():
        files = [path]
        root = path.parent
    else:
        files = [p for p in sorted(path.rglob("*")) if p.is_file()]
        root = path
    for item in files:
        try:
            relative = item.relative_to(root)
        except ValueError:
            relative = Path(item.name)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        try:
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()[:16]


def apply_result(state: Dict[str, Any], doc: Dict[str, Any], decision: str,
                 milestone_doc: Optional[Dict[str, Any]] = None,
                 signature: Optional[Dict[str, Any]] = None,
                 artifact_sha: str = "",
                 base_sha: str = "",
                 milestone_sha: str = "",
                 structure_report: Optional[Dict[str, Any]] = None,
                 precision_report: Optional[Dict[str, Any]] = None,
                 fence_report: Optional[Dict[str, Any]] = None,
                 recipes_used: Optional[List[str]] = None,
                 recipes_invented: Optional[List[str]] = None,
                 symptom: str = "",
                 signature_path: str = "") -> Dict[str, Any]:
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
        # 顺序证据靠它做差。存快照而不是路径 —— 路径会失效，而那次测量
        # 过去了就补不回来。
        "signature_snapshot": (snapshot_signature(signature)
                               if signature else None),
        "signature_path": signature_path or None,
        # 代码身份。没有它，`previous_snapshot` 的整个前提
        # 「被保留下来的最后一条就是当前代码的样子」**无法核实** ——
        # 编码层该回滚却没回滚、或在一棵更早的树上接着改，做差基线就是错的，
        # 而算出来的百分比看着完全正常。这是机制核验链条上最底层的假设。
        #
        # 上游每轮记 sha（落地提交）与 base_sha（构建在哪个 HEAD 上，
        # 原话：是血统不是顺序 —— reset / 重开之后下一轮的基是更早的树）。
        "artifact_sha": artifact_sha or None,
        "base_sha": base_sha or None,
        # 判据用的是哪一份课程表。阈值被改宽、或机制行被删掉，
        # 早先在台账里一点痕迹都没有 —— 配合阈值不绑参考，那是一条完整的
        # 「挪球门」路径。
        "milestone_sha256": milestone_sha or None,
    }

    # 围栏与设备占比先算好、先记进台账，不管后面走哪条分支。
    #
    # 为什么不等到 accept 分支里才算：一条因精度回滚的记录，如果围栏也命中，
    # 那两件事都该留在台账里 —— 只记前者的话，「这一轮其实是在转发」这个
    # 事实会随着那条回滚记录一起被归档掉。
    fence = judge_fence(fence_report)
    entry["fence"] = fence
    device_share = reconcile_device_share(doc, signature)
    entry["device_share"] = device_share

    precision = _precision_passed(doc)
    measured = False

    # 误差比优先于容差比对。
    ratio_verdict = precision_from_report(precision_report)
    entry["precision"] = ratio_verdict
    if ratio_verdict["status"] == "fail":
        # 误差比说不过，容差说过 —— 这正是容差分不出的那一类（降精度换速度）。
        precision = False
    elif ratio_verdict["status"] == "pass":
        precision = True

    if precision is not True:
        # 注意是 `is not True`，不是 `is False`。
        #
        # 字段缺失时 _precision_passed 返回 None，而早先只拦 False ——
        # 于是一份没有精度字段的结果直接落到增益判据，精度门禁静默消失。
        # 「没查」不等于「查过且通过」，这是整套判据里最不该含糊的一条。
        entry["verdict"] = VERDICT_REVERT
        if ratio_verdict["status"] == "fail":
            entry["reason"] = (
                "精度不达标（误差比判据），回滚；算错的 kernel 再快也无意义。"
                + ratio_verdict["reason"])
        elif precision is False:
            entry["reason"] = "精度不达标，回滚；算错的 kernel 再快也无意义"
        else:
            entry["reason"] = (
                "评测结果里没有精度字段（comparison.precision_passed / "
                "evolved.precision_passed），无法确认算得对不对；"
                "缺失不等于通过，回滚")
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
                # 结构先判，停顿后判。
                #
                # 为什么这个顺序：停顿比例能被别的改动顶下去（减小 tile、
                # 改对齐都会让 mte2_wait 降），于是一个单缓冲回退的实现可以
                # 凭一个恰好变好的停顿数拿到「机制已核实」，而双缓冲从未存在。
                # 结构在不在是二值事实，顶不下去。
                structure = judge_structure(structure_report)
                entry["structure"] = structure
                baseline, baseline_label = previous_snapshot(state)
                mechanism = judge_mechanism(
                    milestone_doc, signature, decision, baseline,
                    reference_snapshot=state.get("reference_snapshot"))
                if structure["status"] == "absent":
                    # 结构不在 —— 覆盖停顿的结论。
                    #
                    # 归因方向要紧：这是**实现没做到**，不是课程表解释错了。
                    # 上游的分流规则很明确 ——「声称无冲突布局而冲突仍高 ⇒
                    # 学生没产出无冲突版本（退化回退）⇒ 加一张经验卡片，
                    # 不要改课程表」。把它归到课程表那一侧，结果是改对的
                    # 东西被改坏、改坏的东西没人碰。
                    mechanism = {
                        "status": "refuted",
                        "lines": mechanism.get("lines", []),
                        "compared_against": baseline_label,
                        "attribution": "implementation",
                        "reason": (structure["reason"]
                                   + "。停顿指标可能恰好变好（减小 tile、改"
                                     "对齐都会让它降），但那不是这条机制"),
                    }
                mechanism["compared_against"] = baseline_label
                entry["mechanism"] = mechanism

                # 课程表与台账对账。两份文件此前从不互看。
                coupling = check_gate_coupling(
                    milestone_doc, decision, mechanism)
                if coupling is not None:
                    entry["gate_coupling"] = coupling

                # 围栏与设备占比：两道「这个数算不算数」的前置条件。
                #
                # 它们和机制核验是不同性质的东西。机制核验问「变快的原因是不是
                # 设计的那个」；这两道问的是更靠前的问题 ——「这个百分比到底
                # 是不是这份 kernel 在这张卡上挣来的」。所以它们不走
                # accept_unverified 那条路（那条路的含义是「数是真的，解释
                # 不清」），而是直接压住 counts_toward_best。
                blockers: List[str] = []
                if fence["status"] in ("clean", "clean_with_channels"):
                    # 通道项不压 best。它是「请说明」，不是「你转发了」——
                    # 把它当阻断项会让一批正当实现过不去，而人会反射性地
                    # 整条豁免掉围栏，那之后真正的转发也无人把守。
                    pass
                elif fence["status"] == "violated":
                    # 转发是实现缺陷，不是课程表缺陷 —— 归因方向同结构核验。
                    entry["verdict"] = VERDICT_REVERT
                    entry["failure_kind"] = "fence"
                    # 这一轮不算「测出来了」。
                    #
                    # measured 为真会走 `elif measured` 那条分支，而那条分支
                    # 把实现失败计数**清零** —— 对一次转发来说方向正好相反。
                    # 更根本的是：围栏命中意味着这次测量量的不是这份 kernel，
                    # 所以「测出来了」这句话对它并不成立。
                    measured = False
                    entry["reason"] = (
                        f"围栏命中，回滚：{fence['reason']}。\n"
                        "  这一轮的加速比与精度都不作数 —— 它们不是这份 kernel "
                        "挣来的。\n"
                        "  这是实现问题（该修实现），不是课程表对参考行为的"
                        "解释错了")
                else:
                    blockers.append(f"围栏未查（{fence['reason']}）")
                if device_share is not None and not device_share["ok"]:
                    blockers.append(f"设备占比过低（{device_share['reason']}）")

                if entry.get("verdict") == VERDICT_REVERT:
                    pass
                elif mechanism["status"] == "verified" and not blockers:
                    entry["verdict"] = VERDICT_ACCEPT
                    entry["counts_toward_best"] = True
                    entry["kb_admissible"] = True
                    entry["reason"] = (
                        f"增益 {delta:+.2f}% 超过噪声底 {noise:.2f}%，"
                        f"且机制已核实（{mechanism['reason']}）")
                elif mechanism["status"] == "verified":
                    entry["verdict"] = VERDICT_ACCEPT_UNVERIFIED
                    entry["reason"] = (
                        f"增益 {delta:+.2f}% 超过噪声底 {noise:.2f}%、"
                        f"机制也核实了，但这个数本身还不能用：\n  "
                        + "\n  ".join(blockers)
                        + "\n  保留该改动，但不计入 best —— "
                          "「变快了」这件事本身尚未成立")
                else:
                    entry["verdict"] = VERDICT_ACCEPT_UNVERIFIED
                    entry["reason"] = (
                        f"增益 {delta:+.2f}% 超过噪声底 {noise:.2f}%，"
                        f"但机制未核实：{mechanism['reason']}。"
                        "保留该改动，但不计入 best，也不得作为入库依据 —— "
                        "变快了不等于设计的机制落地了")
                    if blockers:
                        entry["reason"] += ("\n  另有：" + "；".join(blockers))
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
        "consecutive": 0, "threshold": DEFAULT_UNVERIFIED_THRESHOLD,
        "streak_refuted": 0, "streak_unknown": 0})

    if entry["verdict"] in (VERDICT_ACCEPT, VERDICT_ACCEPT_UNVERIFIED):
        # 注意这里**不**清零停滞计数 —— 清零的条件在下面，是「抬高了 best」。
        #
        # 早先任何 accept / accept_unverified 都清零，而 best 只在
        # `gain > best` 时才动。于是编码层每轮都交出 +3%（相对参考、不是
        # 相对上一轮）时：每轮 accept → 停滞清零；best 第一轮停在 1.03 之后
        # 再不动；状态**永远是 implement**，而 status 每轮都说「继续实现」。
        # 循环就这么空转下去。
        #
        # 上游的判据是「窗口内有没有新的下界」（parse.py 的 detect_plateau）：
        # 重复拿同样大小的增益也算停滞。这才是架构层该被叫回来的信号。
        impl["consecutive"] = 0
        if entry["verdict"] == VERDICT_ACCEPT:
            unver["consecutive"] = 0
            unver["streak_refuted"] = 0
            unver["streak_unknown"] = 0
        else:
            # 连续若干次「变快了但机制不算数」。两种成因分开记：
            #   refuted  机制核验了但为假 —— 架构信号，该改课程表的假设
            #   unknown  机制没被核验     —— 流程缺口，该补 record 的参数
            # 合并计数就只能给一句笼统提示，而那句提示会把架构层引向
            # 错误的动作。
            unver["consecutive"] = int(unver.get("consecutive", 0)) + 1
            status = ((entry.get("mechanism") or {}).get("status")
                      if isinstance(entry.get("mechanism"), dict) else None)
            # vacuous 与 refuted 同归一桶：两者都说明增益另有来源，
            # 都该把控制权交回架构层。成因差别在行级 reason 里说清。
            mech = entry.get("mechanism")
            attribution = (mech.get("attribution")
                           if isinstance(mech, dict) else None)
            if attribution == "implementation":
                # 归因到实现的 refuted **不走架构重入**。
                #
                # 上游的分流规则：课程表写对了、实现没做到 → 加一张经验卡片，
                # 不要改课程表。把它算进 streak_refuted 会连续三次把架构层
                # 叫回来重写课程表，而下一轮实现还是同一个退化回退 ——
                # 改对的东西被改坏、改坏的东西没人碰。
                impl["consecutive"] = int(impl.get("consecutive", 0)) + 1
                unver["consecutive"] = max(
                    0, int(unver.get("consecutive", 0)) - 1)
            else:
                key = ("streak_refuted" if status in ("refuted", "vacuous")
                       else "streak_unknown")
                unver[key] = int(unver.get(key, 0)) + 1
        # 只有机制核实过的改动才抬高 best。未核实的保留但不计入 ——
        # 否则 promote 时会把一个解释不清的结果封成下一代的参考。
        raised_best = False
        entry_gain_for_patch = entry.get("gain")
        if entry["counts_toward_best"] and entry["gain"] is not None:
            best = state.setdefault("best", {})
            if entry["gain"] > float(best.get("gain", 1.0)):
                raised_best = True
                best["gain"] = entry["gain"]
                best["time_us"] = entry.get("time_us")
                best["decision"] = decision
                # 新高 = 第一次读数。确认读数由 `record --confirm` 追加 ——
                # 它不能走这条路：确认复测的是**同一份代码**，机制必然
                # 「没有因本次改动而移动」，于是会被判 vacuous、
                # counts_toward_best=False，根本到不了这里。
                best["readings"] = [entry["gain"]]
        # 停滞 = 没有新的下界。抬高了 best 才清零。
        if raised_best:
            stag["since_improvement"] = 0
            # 上一次重开带来了新下界 —— 它不是徒劳的。
            state.setdefault("reopen", {})["fruitless"] = 0
            # 把那次课程表修订的结果回填。
            #
            # 不回填的话 outcome 永远停在 pending，于是 status 会把一次
            # **成功的**修订也列进「已试过的方向」去劝退下一代 —— 一张
            # 只记投入不记产出的账，用起来比没有更糟。
            # 循环变量**不能**叫 entry —— 它会遮蔽本函数正在构造的那个
            # entry，于是函数返回台账里的一条旧记录。这类遮蔽不报错，
            # 只是悄悄换掉返回值。
            for past in reversed(state.get("ledger", [])):
                patch = past.get("patch")
                if isinstance(patch, dict) and patch.get("outcome") == "pending":
                    patch["outcome"] = "accepted"
                    patch["closed_by_gain"] = entry_gain_for_patch
                    break
        else:
            stag["since_improvement"] = int(stag.get("since_improvement", 0)) + 1
    elif measured:
        # 测出来了但不够 —— 同样是「没有新下界」
        stag["since_improvement"] = int(stag.get("since_improvement", 0)) + 1
        impl["consecutive"] = 0
    else:
        # 精度不过 / 评测缺失是实现问题，不消耗架构重入机会。
        # 但它必须改变动作 —— 见 _advance_state 里的 STATE_BLOCKED。
        impl["consecutive"] = int(impl.get("consecutive", 0)) + 1

    state.setdefault("ledger", []).append(entry)
    _advance_state(state, milestone_doc)
    return entry


#: 确认一个 best 需要几次读数。
#:
#: 上游取 3 并以中位数作最终 claim。本仓此前取 2 —— 两读时只能「两次都要
#: 达标」，而那条判据对一次偏低抽样没有抵抗力：偏低和偏高一样常见，于是一个
#: 真达标的结果会被一次运气差的复测否掉，然后再测一次，机时反而更费。
#:
#: 三读 + 中位数同时解决两个方向：一次偏高进不来，一次偏低也拉不下去。
CONFIRM_READINGS = 3

#: 三次读数之间允许差多少（相对噪声底的倍数）。
#:
#: 上游在这个位置查的是「三个样本是不是都落在同一台机器上 —— 那是一次抽样
#: 不是三次」。这条在我们这里**无法照搬**：本仓按设计单卡（`--device-id`
#: 必填、代际之间不换卡），跨机器采样会直接破坏可比性，所以样本必然同机。
#:
#: 它想防的事可以换一个量来查：三次读数彼此的离散度。如果三读之间差得比
#: 噪声底允许的还多，那它们不是同一个量的三次抽样 —— 中位数对它们没有意义，
#: 该先去查环境而不是取中位数。
CONFIRM_SPREAD_FACTOR = 3.0


def confirmation_spread_ok(readings: Sequence[Any],
                           noise_floor_pct: float) -> Tuple[bool, str]:
    """三次读数彼此够接近吗。

    返回 (是否够接近, 说明)。噪声底为 0（还没量）时不判 —— 没有判据可用，
    而编一个出来比不判更糟。
    """
    numeric = [r for r in readings if isinstance(r, (int, float))]
    if len(numeric) < 2 or noise_floor_pct <= 0:
        return True, "读数不足或噪声底未量，离散度不判"
    low, high = min(numeric), max(numeric)
    if low <= 0:
        return True, "读数里有非正值，离散度不判"
    spread_pct = (high - low) / low * 100.0
    cap = noise_floor_pct * CONFIRM_SPREAD_FACTOR
    if spread_pct <= cap:
        return True, (f"读数离散度 {spread_pct:.2f}%，在 "
                      f"{cap:.2f}%（噪声底 ×{CONFIRM_SPREAD_FACTOR}）以内")
    return False, (
        f"读数离散度 {spread_pct:.2f}% 超出 {cap:.2f}%"
        f"（噪声底 {noise_floor_pct:.2f}% × {CONFIRM_SPREAD_FACTOR}）。\n"
        f"    读数：{numeric}\n"
        "    差这么多说明它们不是同一个量的几次抽样 —— 中位数对它们没有"
        "意义。\n"
        "    先查环境（别的进程占卡、降频、散热），而不是取中位数收工。")


def _best_is_confirmed(state: Dict[str, Any]) -> bool:
    """best 是否经过独立复测。

    三个条件：
      够次数     至少 CONFIRM_READINGS 次读数
      中位达标   中位数达到目标（不是「每一次都达标」—— 见下）
      够集中     读数彼此的离散度在噪声底的几倍以内

    为什么从「每一次都达标」改成「中位达标」：前者对一次**偏低**抽样没有
    抵抗力，而偏低和偏高一样常见。于是一个真达标的结果会被一次运气差的
    复测否掉，然后再测一次 —— 机时反而更费，而且那次否决本身没有信息量。

    离散度那一条补的是前者被拿掉之后留下的口子：中位数只在三读确实是同一个
    量的三次抽样时才有意义。
    """
    best = state.get("best") or {}
    readings = best.get("readings")
    if not isinstance(readings, list) or len(readings) < CONFIRM_READINGS:
        return False
    numeric = sorted(r for r in readings if isinstance(r, (int, float)))
    if len(numeric) < CONFIRM_READINGS:
        return False
    target = float(state.get("target_gain", DEFAULT_PHASE_TARGET_GAIN))
    median = (numeric[len(numeric) // 2] if len(numeric) % 2
              else (numeric[len(numeric) // 2 - 1]
                    + numeric[len(numeric) // 2]) / 2)
    if median < target:
        return False
    ok, _detail = confirmation_spread_ok(
        numeric, float(state.get("noise_floor_pct", 0.0) or 0.0))
    return ok


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
    target = float(state.get("target_gain", 1.0))
    if best_gain >= target and not _best_is_confirmed(state):
        # 看似达标不能直接收工。
        #
        # best 是历史最大值，而每次 record 都是一次抽样 —— 预算几十次的
        # campaign 里「抽到一个偏高读数」是必然事件。拿它转 shipped 再
        # promote，就把一次运气封成了下一代参考，且之后无法回溯：
        # promote 会把 best 归零、噪声底置为待重测，那个虚高的数就消失了，
        # 而后面每一代的增益都相对它算。
        state["state"] = STATE_IMPLEMENT
        state["pending_confirmation"] = {
            "decision": (state.get("best") or {}).get("decision"),
            "gain": best_gain,
            "readings": list((state.get("best") or {}).get("readings") or []),
            "reason": ("已达目标，但这个增益只测过一次。对同一份产物再测一次"
                       "（同一个决策 id 再 record 一遍），两次都达标才收工。"),
        }
        state.pop("stall_reason", None)
        state.pop("stall_reasons", None)
        state.pop("stall_detail", None)
        return
    state.pop("pending_confirmation", None)
    if best_gain >= target:
        state["state"] = STATE_SHIPPED
        state.pop("stall_reason", None)
        state.pop("stall_reasons", None)
        state.pop("stall_detail", None)
        return

    # 连续实现失败要**改变动作**。
    #
    # 早先 implementation_failures 只递增、只在 status 里打印，_advance_state
    # 从不读它 —— 于是连续 30 次精度不过，next_action 每次都说 implement，
    # 预算一条条被消耗，没有任何人被告知。
    #
    # 「不消耗架构重入机会」这个判断是对的（实现 bug 不该花掉一次架构重入），
    # 但它被实现成了「什么都不触发」。上游的做法是连续 3 次 revert 就
    # return "stuck" 并买一次 teacher step；我们没有常驻 driver，所以进一个
    # 独立状态让人介入。
    impl = state.get("implementation_failures", {})
    impl_cap = int(impl.get("threshold", DEFAULT_IMPL_FAILURE_THRESHOLD))
    if int(impl.get("consecutive", 0)) >= impl_cap:
        state["state"] = STATE_BLOCKED
        state["stall_reason"] = STALL_IMPL_FAILURE
        state["stall_detail"] = (
            f"连续 {impl.get('consecutive')} 次实现失败"
            f"（精度不过 / 没测出数），已达阈值 {impl_cap}。\n"
            "  这不是架构问题，所以不消耗架构重入机会 —— 但也不该继续"
            "闷头实现：\n"
            "  同一个错重复几十次会把预算吃光，而每轮状态都显示正常。\n"
            "  先看最近几条台账的 failure_kind 与 reason，修对了再 record；\n"
            "  确认是环境问题就停下来上报，不要绕过。")
        return

    budget = state.get("budget", {})
    cap = budget.get("max_records")
    if isinstance(cap, int) and cap > 0 and int(budget.get("used", 0)) >= cap:
        state["state"] = STATE_STOPPED
        state["stop_reason"] = (
            f"预算耗尽：已用 {budget['used']}/{cap} 次评测。"
            "继续请用 set-budget 提高上限")
        return

    # 墙钟预算。
    #
    # 次数预算挡不住一种很常见的消耗：每轮都很快、但轮数没涨而时间在走
    # （构建失败重试、环境排障、等卡）。那些都不消耗 `used`，于是一个跑了
    # 两天的 campaign 看起来预算还很充裕。
    #
    # 没有常驻进程，所以只能按 `created_at` 算差 —— 这是个近似：它把中间
    # 停机的时间也算进去了。取这个近似而不是不做，理由是它的误差方向是
    # **保守**的（报得比实际机时多，于是提前提醒），而反过来会漏。
    elapsed = _elapsed_hours(state)
    wall_cap = budget.get("max_wall_hours")
    if (isinstance(wall_cap, (int, float)) and wall_cap > 0
            and elapsed is not None and elapsed >= wall_cap):
        state["state"] = STATE_STOPPED
        state["stop_reason"] = (
            f"墙钟预算耗尽：自 init 起已过 {elapsed:.1f} 小时"
            f"（上限 {wall_cap}）。\n"
            "  注意这个数按 created_at 算差，含中间停机的时间 —— "
            "它是机时的上界，不是机时本身。\n"
            "  继续请用 set-budget --max-wall-hours 提高上限")
        return

    reasons: List[Dict[str, str]] = []

    # 机制相关的两条排最前：机制不算数时 best 卡在 1.0、其他信号也都是
    # 降级的，先修这个。两条分开报，因为该做的事完全不同。
    unver = state.get("mechanism_unverified", {})
    if int(unver.get("consecutive", 0)) >= int(
            unver.get("threshold", DEFAULT_UNVERIFIED_THRESHOLD)):
        refuted = int(unver.get("streak_refuted", 0))
        unknown = int(unver.get("streak_unknown", 0))
        total = int(unver.get("consecutive", 0))

        # 被证伪排在没核验之前：它是实打实的架构信号，而没核验只是流程缺口。
        if refuted:
            reasons.append({
                "kind": STALL_MECHANISM_REFUTED,
                "detail": (
                    f"最近 {total} 次 accept_unverified 里有 {refuted} 次"
                    "**机制核验为假**：改动确实更快，但它声称的机制没有发生。"
                    "这不是参数没传 —— 是课程表里对参考行为的解释错了，"
                    "而实现碰对了别的东西。该做的是重新分析参考、改写机制"
                    "假设（必要时 reopen 相关决策），不是补参数重跑"),
            })
        if unknown:
            reasons.append({
                "kind": STALL_MECHANISM_UNVERIFIED,
                "detail": (
                    f"最近 {total} 次 accept_unverified 里有 {unknown} 次"
                    "**机制无法核验**：多半是 record 没带 --signature / "
                    "--milestone，也可能是签名里缺该字段或签名自判不可信。"
                    "补齐后重跑即可；在此之前 best 会一直卡在 1.0，"
                    "campaign 永远到不了 shipped"),
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


#: 死路表的文件名。由 forge-hwkb 的 hwkb.py 定义，两边必须一致。
NOGO_FILENAME = "nogo.json"


def write_nogo(kb_root: Path, state: Dict[str, Any],
               entry: Dict[str, Any]) -> int:
    """直接写入 hwkb 的死路表。

    参数是**知识库根目录**，不是文件路径。文件名与锁的位置都由程序拼，
    这样互斥是结构性的，不依赖调用方传对路径 ——

      hwkb.py   锁 <kb_root>/.lock，写 <kb_root>/nogo.json
      这里      锁 <kb_root>/.lock，写 <kb_root>/nogo.json

    若这里改成按任意文件路径取其父目录作锁，调用方把文件放到别处时，
    两边锁的就是不同的文件，而加锁要防的正是这件事。

    格式由 hwkb.py 定义（`{"entries": [...]}`），改格式时两边要一起改。
    """
    # exclusive 是「对一个目录加排他锁」的通用原语（约定 <dir>/.lock），
    # 与 hwkb.exclusive 同一套约定，所以锁的是同一个文件。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import reference_library as _lib  # noqa: PLC0415

    with _lib.exclusive(kb_root):
        return _write_nogo_locked(kb_root / NOGO_FILENAME, state, entry)


def _write_nogo_locked(nogo_path: Path, state: Dict[str, Any],
                       entry: Dict[str, Any]) -> int:
    """write_nogo 的实体。调用方必须已持锁。"""
    entries: List[Dict[str, Any]] = []
    if nogo_path.exists():
        # 读不出来就停手。这个函数是「读全部 → 追加一条 → 整份写回」，
        # 把损坏文件读成空表意味着下一行写入会清掉整张死路表，只剩新加的
        # 那一条。死路表的作用是防止后续代次重走已证伪的路；它被悄悄清空
        # 之后，那些路会被一条条重试，每条都要花掉真实的评测预算。
        try:
            text = nogo_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CampaignError(f"无法读取死路表 {nogo_path}: {exc}") from exc
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CampaignError(
                f"死路表 {nogo_path} 不是合法 JSON: {exc}\n"
                "  多半是上一次写入被打断留下了截断的文件。\n"
                "  这里不会当成空表继续 —— 那样会把整张表覆盖成只有一条，\n"
                "  之前所有证伪过的方向都会被重新试一遍。\n"
                "  请先人工确认该文件（必要时从 git 历史恢复）。") from exc
        if not isinstance(doc, dict):
            raise CampaignError(
                f"死路表 {nogo_path} 的顶层不是对象，无法解析出 entries")
        existing = doc.get("entries")
        if existing is None:
            entries = []
        elif isinstance(existing, list):
            entries = existing
        else:
            raise CampaignError(f"死路表 {nogo_path} 里的 entries 不是数组")

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

def _load_signature(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise CampaignError(f"无法读取签名 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CampaignError(f"签名 {path} 不是合法 JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise CampaignError(f"签名 {path} 顶层必须是对象")
    return doc


def cmd_set_reference_signature(args: argparse.Namespace) -> int:
    """登记参考实现自己的架构签名，作为机制做差的第 0 轮基线。"""
    path = Path(args.state)
    try:
        state = load_state(path)
        signature = _load_signature(args.signature)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    # 与 init 同一道门。这条路径不是补充手段而是**主要**手段：promote 会把
    # reference_snapshot 清空，所以第 1 代起每一代都只能靠它登记参考签名。
    # 门只守在 init 上等于只守住了第 0 代。
    if signature is not None and not args.allow_untrusted_signature:
        reasons = _signature_blocking_reasons(signature)
        if reasons:
            print(f"错误: {args.signature} 这份签名自己判了测量不可信"
                  "（trust.trustworthy=false），不能作为做差基线。",
                  file=sys.stderr)
            for reason in reasons:
                print(f"  - {reason}", file=sys.stderr)
            print("  不可信的签名作基线，之后每一轮机制做差都建立在错的起点上，"
                  "而且每个数看起来都正常。\n"
                  "  先按签名里 trust 的提示重采，或加 "
                  "--allow-untrusted-signature 明确接受。", file=sys.stderr)
            return 2

    snapshot = snapshot_signature(signature) if signature else {}
    if not snapshot:
        print("错误: 该签名里没有任何有效读数，无法作为做差基线。\n"
              "  多半是 signature_probe 判了测量不可信、或 metric 缺失。",
              file=sys.stderr)
        return 1

    state["reference_snapshot"] = snapshot
    state.setdefault("ledger", []).append({
        "at": _now(), "generation": state.get("generation", 0),
        "input_shape": state.get("input_shape"), "decision": None,
        "gain": None, "verdict": "reference_signature",
        "reason": f"登记参考签名快照，含 {len(snapshot)} 个字段",
        "counts_toward_best": False, "kb_admissible": False,
    })
    save_state(path, state)
    print(f"已登记参考签名快照：{len(snapshot)} 个字段")
    print("机制行做差时，没有更近的有效台账快照就以它为基线（第 0 轮）。")
    return 0


def _signature_blocking_reasons(doc: Dict[str, Any]) -> List[str]:
    """签名自己判了「这次测量不可信」时，把阻塞原因取出来。

    signature_probe 在降频、ICache 缺失率过高、关键 metric 缺失等情况下会把
    trustworthy 置 false 并扣留架构假设。那份签名若被当作第 0 轮基线，后面
    每一轮的机制做差都建立在一个错的起点上，而且因为每个数看起来都正常，
    错误不会自己冒出来。
    """
    trust = doc.get("trust")
    if not isinstance(trust, dict):
        return []
    if trust.get("trustworthy") is False:
        reasons = trust.get("blocking_reasons")
        if isinstance(reasons, list) and reasons:
            return [str(r) for r in reasons]
        return ["签名里 trust.trustworthy=false，但未写明阻塞原因"]
    return []


def _find_sealed_list(explicit: Optional[str], state: str) -> Optional[Path]:
    """找封存清单。与 hook 的查找顺序保持一致，否则自检测的不是同一个东西。"""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_path = os.environ.get("FORGE_SEALED_PATHS")
    if env_path:
        candidates.append(Path(env_path))
    work_dir = os.environ.get("FORGE_WORK_DIR")
    if work_dir:
        candidates.append(Path(work_dir) / SEALED_PATHS_FILENAME)
    state_dir = Path(state).resolve().parent
    candidates.append(state_dir / SEALED_PATHS_FILENAME)
    candidates.append(Path("forge_work") / SEALED_PATHS_FILENAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _find_reference_spec(explicit: Optional[str], state: str,
                         reference_path: Optional[str]) -> Optional[Path]:
    """找出本次 campaign 对应的 reference_spec.json。

    不强制命令行传：实际流程里它就躺在工作目录里（prepare 写的），
    自动找到就能把门禁用上；找不到才退回告警。门禁只在证据确实存在时
    才生效，这样既堵住了洞，也不会把没有 spec 的用法整个卡死。
    """
    if explicit:
        return Path(explicit)
    candidates: List[Path] = []
    if reference_path:
        ref = Path(reference_path)
        candidates += [ref / SPEC_FILENAME, ref.parent / SPEC_FILENAME]
    candidates.append(Path(state).resolve().parent / SPEC_FILENAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _check_arch(arch: str, allow_unknown: bool) -> Optional[str]:
    """架构串对不上就别开 campaign。返回错误文本，没问题返回 None。"""
    if arch in KNOWN_ARCHS or allow_unknown:
        return None
    return (
        f"架构串 {arch!r} 不在已知列表 {list(KNOWN_ARCHS)} 里。\n"
        "  架构成熟卡片按这个串分目录存放。填错的后果是 campaign 能一路跑完，\n"
        "  但每张卡片入库时都被 forge-hwkb 判 A-ARCH 拒掉 —— 几十轮攒下来的\n"
        "  经验一条也存不住，而这件事要跑很久才会暴露。\n"
        "  怎么查本机该填哪个：\n"
        "    npu-smi info                                   先看芯片型号（如 910B3）\n"
        "    grep -r NpuArch \"$ASCEND_HOME_PATH\"/*/data/platform_config/*.ini\n"
        "    NpuArch=2201 对应 DAV_2201，NpuArch=3510 对应 DAV_3510\n"
        "  确实是新硬件时加 --allow-unknown-arch，但要同时在 forge-hwkb 的\n"
        "  ARCH_DIRS 里把这个架构登记上，否则入库问题照旧。"
    )


def _check_reference_spec(spec_path: Optional[Path], kind: str,
                          reference_id: str,
                          allow_unverified: bool) -> Tuple[Optional[str], List[str]]:
    """核对参考定义。返回 (拒绝理由, 告警列表)。

    要挡住的是这一种情形：参考声明为「CANN 内置实现」，而「空 OPP 目录能让
    aclnn 落到内置实现」这件事根本没在本机证实过。没证实就意味着第 0 轮基线
    测的可能压根不是内置实现，后面所有加速比都是对着一个未知对象算的。
    """
    warnings: List[str] = []
    if spec_path is None:
        if kind == KIND_BUILTIN:
            warnings.append(
                f"没找到 {SPEC_FILENAME}，无法确认派发是否真的落到内置实现。\n"
                "    建议先跑 reference_adapter.py prepare/verify 产出并确认它，"
                "再用 --reference-spec 指过来。"
            )
        return None, warnings

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return f"无法读取参考定义 {spec_path}: {exc}", warnings
    except json.JSONDecodeError as exc:
        return f"{spec_path} 不是合法 JSON: {exc}", warnings
    if not isinstance(spec, dict):
        return f"{spec_path} 的顶层必须是对象", warnings

    spec_kind = spec.get("kind")
    if spec_kind != kind:
        warnings.append(
            f"--reference-kind 是 {kind}，而 {spec_path.name} 里写的是 {spec_kind}。"
            "两者应当一致，请确认用的是同一个参考。")
    spec_id = spec.get("reference_id")
    if spec_id and spec_id != reference_id:
        warnings.append(
            f"--reference-id 是 {reference_id!r}，而 {spec_path.name} 里是 "
            f"{spec_id!r}。目前没有任何逻辑按这个值做匹配，不影响运行，"
            "但两处不一致容易让人以为它们必须对齐 —— 建议统一成算子名。")

    # 两者不一致时，这份 spec 多半根本不是本次参考的定义（工作目录里留着上一个
    # 参考的 spec 就会这样）。那种情况下拿它的 dispatch_verified 去拦 init 是错的：
    # 建议会变成「去确认 aclnn 派发」，而本次参考压根不是 aclnn。
    # 所以只在**声明的来源**是内置实现时才要求派发确认；对不上的事由上面的
    # 告警负责提醒，让人去确认用的是不是同一份 spec。
    if kind == KIND_BUILTIN and spec.get("dispatch_verified") is not True:
        if allow_unverified:
            warnings.append(
                "派发行为未确认，但已用 --allow-unverified-dispatch 放行。"
                "第 0 轮基线可能测的不是内置实现，请在 milestone 里记录这件事。")
        else:
            return (
                f"{spec_path} 里 dispatch_verified 不是 true，"
                "即「空 OPP 目录会让 aclnn 落到内置实现」这件事还没在本机证实过。\n"
                "  没证实就开 campaign，第 0 轮基线测的可能根本不是内置实现，"
                "后面每一轮加速比都是对着一个未知对象算出来的。\n"
                "  确认并回写：\n"
                "    python3 reference_adapter.py verify "
                f"--spec {spec_path} \\\n"
                "        --confirm-dispatch --evidence \"<msprof 里看到的算子名等证据>\"\n"
                "  确实要先跑起来再说时加 --allow-unverified-dispatch。"
            ), warnings
    return None, warnings


HOOK_RELATIVE = Path("hooks") / "no-code-boundary.sh"


def probe_boundary(sealed_list: Path) -> Tuple[bool, str]:
    """真的过一遍 hook，确认边界此刻拦得住。

    为什么要实测而不是查文件存在：边界由三样东西合起来生效 —— hook 脚本、
    它在 agent 工具层的注册、以及封存清单。查前两样只能知道「文件在」，
    而实际失效的方式恰恰是「文件都在但没被调用」或「清单是空的」。

    做法：拿清单里第一条被封路径构造一次 Read 调用喂给 hook，期望它返回 2。
    返回 0 说明这条路此刻是通的 —— 那么整个 campaign 都会在无防护下跑。
    """
    plugin_root = Path(__file__).resolve().parents[3]
    hook = plugin_root / HOOK_RELATIVE
    if not hook.is_file():
        return False, f"找不到边界 hook 脚本：{hook}"

    sealed: List[str] = []
    try:
        for line in sealed_list.read_text(encoding="utf-8").splitlines():
            token = line.split("#", 1)[0].strip()
            if token:
                sealed.append(token)
    except OSError as exc:
        return False, f"读不到封存清单 {sealed_list}: {exc}"
    if not sealed:
        return False, f"封存清单 {sealed_list} 是空的 —— 没有任何路径被保护"

    # 挑一个具体文件而不是目录：目录可能恰好不存在，而 hook 对 Read 走的是
    # 路径归一化那条分支。
    target = next((p for p in sealed if Path(p).is_file()), sealed[-1])
    payload = json.dumps({"tool_name": "Read",
                          "tool_input": {"file_path": target}})
    env = dict(os.environ)
    env["FORGE_SEALED_PATHS"] = str(sealed_list)
    env.pop("FORGE_BOUNDARY_HOOK_DISABLE", None)
    try:
        proc = subprocess.run(["bash", str(hook)], input=payload, text=True,
                              capture_output=True, timeout=30, env=env,
                              check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"执行 hook 失败: {exc}"
    if proc.returncode == 2:
        return True, f"实测拦住了对 {target} 的读取（hook 返回 2）"
    return False, (
        f"hook 对一条**已封存**的路径返回了 {proc.returncode}，期望 2。\n"
        f"    被测路径: {target}\n"
        f"    hook: {hook}\n"
        f"    stderr: {(proc.stderr or '').strip()[:300]}")


def lint_knowledge_base(kb_root: Optional[str]) -> List[str]:
    """开跑之前把知识库整棵树过一遍 lint。返回问题列表，空表示干净。

    为什么放在 init 这一刻：知识库是**唯一跨 campaign 存活**的东西，而它被
    读的时机是架构层写课程表的时候 —— 那时一张越界的卡片（留着参考的专有
    符号名、或读起来像上一次的 campaign 日记）会直接进到架构层的输入里。

    此时拦下来的代价是最小的：还没有花任何机时。跑到中途才发现，意味着那张
    卡片已经参与过几轮决策，而哪几轮受了影响已经无从分辨。

    拿不到 hwkb 或知识库根时返回空表 —— 这一条不硬拒（知识库是可选的，
    一个全新环境里它本来就是空的）。缺失由调用方告警。
    """
    if not kb_root:
        return []
    root = Path(kb_root)
    if not root.is_dir():
        return [f"知识库根 {root} 不存在 —— 开跑之前的 lint 跳过了"]
    try:
        hwkb = _load_hwkb()
    except CampaignError as exc:
        return [f"装载不了 hwkb，开跑之前的 lint 跳过了：{exc}"]

    denylist = hwkb.load_denylist(root)
    operator_names = hwkb.load_operator_names(root)
    problems: List[str] = []
    unreadable: List[Any] = []
    count = 0
    for path, meta, body in hwkb.iter_cards(root, None, unreadable):
        count += 1
        issues = hwkb.check_no_code_boundary(
            meta, body, denylist, operator_names)
        for issue in issues:
            problems.append(f"{path.name}: {issue}")
    for path, reason in unreadable:
        problems.append(f"{path.name}: 无法解析（{reason}）")
    if not problems:
        return []
    return [
        f"知识库有 {len(problems)} 项问题（共 {count} 张卡片），"
        "开跑之前先修掉：\n    "
        + "\n    ".join(problems[:8])
        + ("\n    ……" if len(problems) > 8 else "")
        + "\n    知识库是唯一跨 campaign 存活的东西，而它被读的时机是架构层"
          "写课程表的时候 ——\n"
          "    一张越界的卡片会直接进到架构层的输入里。此时拦下来还没花"
          "任何机时；\n"
          "    跑到中途才发现，那张卡片已经参与过几轮决策，而哪几轮受了影响"
          "无从分辨。\n"
          "    修法：hwkb.py lint --all（--quarantine 可把屡次不过检的移出"
          "读集）。"
    ]


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

    arch_error = _check_arch(args.arch, args.allow_unknown_arch)
    if arch_error:
        print(f"错误: {arch_error}", file=sys.stderr)
        return 2

    try:
        reference_signature = _load_signature(args.reference_signature)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if reference_signature is not None:
        reasons = _signature_blocking_reasons(reference_signature)
        if reasons and not args.allow_untrusted_signature:
            print(f"错误: {args.reference_signature} 这份签名自己判了测量不可信"
                  "（trust.trustworthy=false），不能作为第 0 轮基线。",
                  file=sys.stderr)
            for reason in reasons:
                print(f"  - {reason}", file=sys.stderr)
            print("  不可信的签名作基线，后面每一轮机制做差都建立在错的起点上，"
                  "而且每个数看起来都正常，错误不会自己冒出来。\n"
                  "  先按签名里 trust 的提示重采（常见原因：降频、"
                  "metric 没采齐、取到了冷启动行），或加 "
                  "--allow-untrusted-signature 明确接受。", file=sys.stderr)
            return 2

    spec_path = _find_reference_spec(
        args.reference_spec, args.state, args.reference_path)
    refusal, spec_warnings = _check_reference_spec(
        spec_path, args.reference_kind, args.reference_id,
        args.allow_unverified_dispatch)
    if refusal:
        print(f"错误: {refusal}", file=sys.stderr)
        return 2
    for warning in spec_warnings:
        print(f"警告: {warning}", file=sys.stderr)

    # 边界自检。整套方法建立在「参考只能测不能读」上，而这件事此前从没在
    # 开 campaign 时验证过 —— 默认安装不注册 hook、清单被清理、路径拼错，
    # 三种情况都会让整个 campaign 在无防护下跑完，而所有文档都说边界生效。
    sealed_list = _find_sealed_list(args.sealed_paths, args.state)
    if sealed_list is None:
        if not args.allow_no_boundary:
            print("错误: 找不到封存清单（sealed_paths.txt），"
                  "无法确认 no-code 边界是否生效。", file=sys.stderr)
            print("  先封存参考：", file=sys.stderr)
            print("    python3 seal_reference.py seal --artifact <中转目录> \\",
                  file=sys.stderr)
            print("        --op-name <算子> --work-dir <工作目录>", file=sys.stderr)
            print("  或用 --sealed-paths 指定清单位置。", file=sys.stderr)
            print("  参考确实是不可读的二进制、无需封存时加 --allow-no-boundary。",
                  file=sys.stderr)
            return 2
        print("警告: 未封存任何参考，no-code 边界不生效。"
              "架构结论将无法证明是从硬件行为反推的。", file=sys.stderr)
    else:
        ok, detail = probe_boundary(sealed_list)
        if ok:
            print(f"边界自检通过：{detail}")
        elif args.allow_no_boundary:
            print(f"警告: 边界自检未通过：{detail}", file=sys.stderr)
            print("  已按 --allow-no-boundary 继续。本次 campaign 的架构结论"
                  "无法证明不是抄的。", file=sys.stderr)
        else:
            print(f"错误: 边界自检未通过：{detail}", file=sys.stderr)
            print("  边界此刻拦不住读参考源码。整套方法的前提是"
                  "「架构决策从硬件行为反推」——", file=sys.stderr)
            print("  这个前提破了，结论就无法迁移到没有开源实现的算子上，"
                  "而那正是它最有价值的场景。", file=sys.stderr)
            print("  常见成因：init.sh 没有为当前工具注册 PreToolUse hook"
                  "（默认工具就不注册）、", file=sys.stderr)
            print("  清单被清理、或 FORGE_SEALED_PATHS 指错。",
                  file=sys.stderr)
            return 2

    reference_snapshot = (snapshot_signature(reference_signature)
                          if reference_signature else None)

    # 开跑之前把知识库过一遍。此时拦下来还没花任何机时。
    kb_problems = lint_knowledge_base(args.kb_root)
    if kb_problems and not args.allow_dirty_kb:
        for problem in kb_problems:
            print(f"错误: {problem}", file=sys.stderr)
        print("  确认要带着这些问题开跑时加 --allow-dirty-kb。",
              file=sys.stderr)
        return 1
    for problem in kb_problems:
        print(f"警告: {problem}", file=sys.stderr)

    state = new_state(
        operator=args.operator, input_shape=args.input_shape, arch=args.arch,
        reference={
            "kind": args.reference_kind, "id": args.reference_id,
            "path": args.reference_path, "sealed": True, "generation": 0,
        },
        noise_floor_pct=args.noise_floor, target_gain=args.target_gain,
        stagnation_threshold=args.stagnation_threshold, budget=args.budget,
        reference_snapshot=reference_snapshot,
        device={"device_id": args.device_id},
        wall_hours=args.wall_hours,
    )
    if args.milestone_path:
        state["milestone_path"] = str(Path(args.milestone_path).resolve())
    if sealed_list is not None:
        state["sealed_paths"] = str(sealed_list.resolve())
    save_state(path, state)
    print(f"已创建 campaign: {args.operator} @ {args.arch}")
    print(f"  输入形状  {args.input_shape}")
    print(f"  参考      {args.reference_kind}:{args.reference_id}（封闭，只测不读）")
    if spec_path is not None:
        print(f"  参考定义  {spec_path}")
    print(f"  噪声底    {args.noise_floor}%")
    print(f"  目标增益  {args.target_gain}")
    if args.budget:
        print(f"  预算      {args.budget} 次评测")
    else:
        print("  预算      不限（循环没有次数出口，"
              "停滞判据失效时会一直跑）", file=sys.stderr)
    print(f"  设备      device {args.device_id}")
    action = next_action(state)
    print(f"\n下一步: {action['action']} — {action['reason']}")
    return 0


def _evolved_time_us(doc: Dict[str, Any]) -> Optional[float]:
    """取候选侧耗时。与 apply_result 用同一个口径。"""
    evolved = doc.get("evolved")
    if isinstance(evolved, dict):
        value = evolved.get("time_us")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def confirm_best(state: Dict[str, Any], doc: Dict[str, Any],
                 decision: str) -> Dict[str, Any]:
    """并入一次**确认复测**的读数。

    为什么不能复用 record：确认复测跑的是同一份产物，代码一个字没改。
    于是机制核验必然判「那个字段并非因本次改动而变化」（vacuous），
    这一轮会被记成 accept_unverified、不计入 best —— 确认永远无法完成。

    确认读数的语义也不同：它不是一个新候选。它不该推进停滞计数、不该
    消耗实现失败计数、不该登记死路，只回答一件事 ——「上次那个增益，
    再测一次还在不在」。
    """
    best = state.setdefault("best", {})
    if best.get("decision") != decision:
        raise CampaignError(
            f"best 当前记的是决策 {best.get('decision')!r}，不是 {decision!r}。"
            "确认复测必须针对拿到 best 的那个决策、那份产物。")

    gain = _extract_gain(doc)
    if gain is None:
        raise CampaignError("确认复测的结果里取不到增益")
    precision = _precision_passed(doc)
    if precision is not True:
        raise CampaignError(
            "确认复测的精度没过（或结果里没有精度字段）——"
            "这说明上次那轮的精度结论本身不可靠")

    readings = best.setdefault("readings", [])
    readings.append(gain)
    target = float(state.get("target_gain", DEFAULT_PHASE_TARGET_GAIN))
    entry = {
        "at": _now(), "generation": state.get("generation", 0),
        "input_shape": state.get("input_shape"), "decision": decision,
        "gain": gain, "time_us": _evolved_time_us(doc),
        "verdict": VERDICT_CONFIRM,
        # 明确记下「这是对哪一轮读数的复测」，而不是一次新候选。
        "reading_of": decision,
        "reason": (f"确认复测：增益 {(gain - 1) * 100:+.2f}%"
                   f"（目标 {target}）。该决策累计读数 {readings}"),
        "counts_toward_best": False,
        "kb_admissible": False,
    }
    state.setdefault("ledger", []).append(entry)
    # best 的 gain 取**中位数**。
    #
    # 三个读数下中位数同时丢掉一个偏高和一个偏低的离群点，这正是它存在的
    # 理由。此前取最小值，那是两读时的保守选择 —— 但三读时取最小意味着
    # 一次偏低抽样就能把结论拉下来，而偏低抽样和偏高抽样一样常见。
    #
    # 不取最大：那是让运气决定 best。不取平均：一个离群值会按权重渗进来。
    numeric = sorted(r for r in readings if isinstance(r, (int, float)))
    best["gain"] = numeric[len(numeric) // 2] if len(numeric) % 2 \
        else (numeric[len(numeric) // 2 - 1] + numeric[len(numeric) // 2]) / 2
    _advance_state(state, None)
    return entry


#: 停顿类机制行的默认带宽：参考实测值的多少倍。
#:
#: 取 1.25 —— 比噪声底高一档，但不足以让「两倍于参考的等待量」过关。
#: 这个数来自论文实现的「机制行带宽纪律」，它的理由是实证：一条锚在参考值
#: 1.6 的行被写成 ≤6.0，病灶最后是靠并排读源码才暴露的。
MECHANISM_BAND_FACTOR = 1.25

#: 本轮读数与参考差到多少倍，就必须被显式裁定。
#:
#: 论文实现取 2×，理由是实证：一个 stall 指标 12.7 倍于参考而其余全部在带内，
#: 因为课程表没写那条机制行，整段预算花在了无辜的嫌疑人上。
M0_AUDIT_FACTOR = 2.0


def audit_against_reference(state: Dict[str, Any],
                            signature: Dict[str, Any],
                            factor: float = M0_AUDIT_FACTOR
                            ) -> List[Dict[str, Any]]:
    """把本轮签名与参考签名逐通道并排，找出差得最多的那些。

    为什么需要这一步：门禁验的是「实现有没有建出课程表说的东西」，它**只拿
    课程表自己声明的值**做比较。于是整个门禁体系里没有任何地方回答
    「课程表本身画对了吗」—— 课程表没写到的通道可以是参考的任意倍数而不
    触发任何东西，而机制核验只看课程表挑出来写的那几条。

    具体形态：参考的某个等待占比是 0.3、候选是 6.0（20 倍），只要课程表没写
    这条机制行，候选照样 accept、入知识库、封成下一代参考。
    """
    reference = state.get("reference_snapshot") or {}
    current = snapshot_signature(signature) if signature else {}
    findings: List[Dict[str, Any]] = []
    for channel, ref_value in sorted(reference.items()):
        cur_value = current.get(channel)
        if not isinstance(ref_value, (int, float)) \
                or not isinstance(cur_value, (int, float)):
            continue
        # 两边都接近 0 时倍数没有意义（0.01 vs 0.03 是 3 倍但无关紧要）
        if abs(ref_value) < _MOVE_EPS and abs(cur_value) < _MOVE_EPS:
            continue
        denominator = max(abs(ref_value), _MOVE_EPS)
        ratio = abs(cur_value) / denominator
        if ratio >= factor:
            findings.append({
                "channel": channel,
                "reference": ref_value,
                "current": cur_value,
                "ratio": round(ratio, 2),
            })
    findings.sort(key=lambda f: -f["ratio"])
    return findings


def cmd_audit(args: argparse.Namespace) -> int:
    """M0 对账：本轮读数与参考并排，每个差 ≥N 倍的通道都要有人认领。"""
    path = Path(args.state)
    try:
        state = load_state(path)
        signature = _load_signature(args.signature)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    if signature is None:
        print("错误: 需要 --signature（本轮的 signature_probe 输出）",
              file=sys.stderr)
        return 2
    if not state.get("reference_snapshot"):
        print("错误: state 里没有参考签名快照，无法对账。\n"
              "  先 campaign.py set-reference-signature --signature <参考签名>",
              file=sys.stderr)
        return 2

    milestone_doc = None
    milestone_path = args.milestone or state.get("milestone_path")
    if milestone_path:
        try:
            milestone_doc = load_milestone(Path(milestone_path))
        except CampaignError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 2

    findings = audit_against_reference(state, signature, args.factor)
    # 课程表里已经有机制行锚着的通道，算「已被认领」
    claimed: Set[str] = set()
    if milestone_doc is not None:
        for _mid, category, line in _iter_all_gate_lines(milestone_doc):
            if category != "mechanism_lines":
                continue
            anchor = line.get("anchor") or {}
            expr = anchor.get("expr") if isinstance(anchor, dict) else None
            if not isinstance(expr, str):
                continue
            for finding in findings:
                field = finding["channel"].rsplit(":", 1)[-1]
                if field and field in expr:
                    claimed.add(finding["channel"])

    unclaimed = [f for f in findings if f["channel"] not in claimed]

    if args.as_json:
        print(json.dumps({
            "factor": args.factor,
            "findings": findings,
            "claimed": sorted(claimed),
            "unclaimed": unclaimed,
            "passed": not unclaimed,
        }, ensure_ascii=False, indent=2))
        return 0 if not unclaimed else 1

    sys.stdout.flush()
    print(f"M0 对账：与参考相差 ≥{args.factor}× 的通道 {len(findings)} 个")
    for finding in findings:
        mark = "已认领" if finding["channel"] in claimed else "**无人认领**"
        print(f"  [{mark}] {finding['channel']}")
        print(f"      参考 {finding['reference']} → 本轮 "
              f"{finding['current']}（{finding['ratio']}×）")
    if not findings or not unclaimed:
        # 把结果记进 state。
        #
        # 不记的话，「这一代到底有没有做过 M0 对账」无从得知 —— audit 是个
        # 纯读的命令，跑完什么痕迹都不留。而它查的是整个 campaign 里最容易
        # 悄悄出错的那件事：课程表漏掉了某个超带通道，于是门禁永远不会因它
        # 触发，而所有数看起来都正常。
        state.setdefault("m0_audit", {})
        state["m0_audit"] = {
            "at": _now(),
            "generation": state.get("generation", 0),
            "factor": args.factor,
            "finding_count": len(findings),
            "unclaimed": [],
            "passed": True,
        }
        save_state(path, state)
        if not findings:
            print("  （无）所有参考通道都在 "
                  f"{args.factor}× 带内。")
        else:
            print("\n对账通过：每个超带通道都有机制行锚着。")
        return 0

    state["m0_audit"] = {
        "at": _now(),
        "generation": state.get("generation", 0),
        "factor": args.factor,
        "finding_count": len(findings),
        "unclaimed": [f["channel"] for f in unclaimed],
        "passed": False,
    }
    save_state(path, state)

    sys.stdout.flush()
    print(f"\n对账未通过：{len(unclaimed)} 个通道差了 {args.factor} 倍以上，"
          "而课程表里没有任何机制行锚着它们。", file=sys.stderr)
    print("  门禁只拿课程表自己声明的值做比较，所以课程表没写到的通道"
          "不会触发任何东西。", file=sys.stderr)
    print("  对每一个无人认领的通道三择一：", file=sys.stderr)
    print("    (a) M0 里已有具名决策接手 → 补一条机制行锚住它",
          file=sys.stderr)
    print("    (b) 实现缺陷（课程表写对了、实现没做到）→ 补一张 hwkb 卡片，"
          "不要改课程表", file=sys.stderr)
    print("    (c) 课程表缺陷（没写/写错）→ 走 reopen 修订课程表",
          file=sys.stderr)
    return 1


def cmd_neutral(args: argparse.Namespace) -> int:
    """记一条「改了代码但没测」的条目。

    它要 `--sha` 而**不要** `--result`：没有结果可读，这正是这条命令存在的
    理由。它做的唯一一件事是把当前的代码身份接到台账末尾，让下一轮的做差
    基线仍然指得对。

    不消耗评测预算（没评测），不动停滞计数（没有新下界也没有失败），
    不抬高 best（没有增益）。
    """
    path = Path(args.state)
    try:
        state = load_state(path)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    sha = (args.sha or "").strip()
    if not sha and args.artifact:
        try:
            sha = directory_fingerprint(Path(args.artifact))
        except OSError as exc:
            print(f"错误: 算不出 {args.artifact} 的内容指纹: {exc}",
                  file=sys.stderr)
            return 2
    if not sha:
        print("错误: 需要 --sha 或 --artifact 之一。\n"
              "  这条命令的全部作用就是把**当前的代码身份**接到台账末尾 ——"
              "没有身份就没有意义。", file=sys.stderr)
        return 2

    previous = previous_artifact_sha(state)
    if previous and previous == sha:
        print(f"错误: 指纹与台账里上一条相同（{sha[:12]}）—— 代码没有变过。\n"
              "  那么不需要记这一条：做差基线本来就指着它。\n"
              "  如果你以为改过了，先确认改动真的落到了被测的那份产物上"
              "（常见原因：改完忘了重装 OPP）。", file=sys.stderr)
        return 1

    entry = {
        "at": _now(),
        "generation": state.get("generation", 0),
        "input_shape": state.get("input_shape"),
        "decision": args.decision,
        "gain": None,
        "time_us": None,
        "verdict": VERDICT_NEUTRAL,
        "artifact_sha": sha,
        "base_sha": previous or None,
        "reason": (f"中性记录：{args.reason or '改动不产生可测的时间变化'}。"
                   "只记代码身份，不记性能 —— 下一轮的做差基线据此取"),
        "counts_toward_best": False,
        "kb_admissible": False,
    }
    state.setdefault("ledger", []).append(entry)
    save_state(path, state)
    print(f"已记中性条目：决策 {args.decision}，指纹 {sha[:12]}")
    print(f"  上一条指纹 {(previous or '无')[:12]} → 做差基线已接到这一条")
    print("  不消耗评测预算，不动停滞计数，不抬高 best。")
    return 0


def previous_artifact_sha(state: Dict[str, Any]) -> Optional[str]:
    """台账里最后一条有代码身份的记录的指纹。

    只看本代、只看未作废 —— 作废的那些建立在旧架构假设上，拿它们当基线会把
    两套假设之间的差异算到本次改动头上。
    """
    for entry in reversed(generation_entries(state)):
        if entry.get("invalidated"):
            continue
        sha = entry.get("artifact_sha")
        if isinstance(sha, str) and sha.strip():
            return sha
    return None


def cmd_confirm(args: argparse.Namespace) -> int:
    path = Path(args.state)
    try:
        state = load_state(path)
        doc = json.loads(Path(args.result).read_text(encoding="utf-8"))
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError) as exc:
        print(f"错误: 无法读取 {args.result}: {exc}", file=sys.stderr)
        return 2

    problems = check_result_admissible(doc, state)
    if problems:
        print("错误: 这份结果不能用于确认复测：", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    try:
        entry = confirm_best(state, doc, args.decision)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    save_state(path, state)

    best = state.get("best", {})
    readings = best.get("readings", [])
    print(f"确认复测记录：{entry['reason']}")
    if _best_is_confirmed(state):
        print(f"  已确认：{len(readings)} 次读数全部达标，"
              f"best 取最小值 {best.get('gain')}")
    else:
        target = float(state.get("target_gain", DEFAULT_PHASE_TARGET_GAIN))
        low = [r for r in readings if r < target]
        print(f"  未确认：{len(low)} 次读数低于目标 {target}（{low}）。")
        print("  上次那个达标读数是一次偏高抽样 —— 这正是复测要查出来的。")
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

    problems = check_result_admissible(doc, state)
    # 参考变了就不裁决。本轮的加速比是相对参考算的 —— 参考变了，这个百分比
    # 就不再是它声称的那个意思，而变化是静默的（结果文件一切正常）。
    problems += check_reference_unchanged(
        _work_dir_from_state(state, path))
    if problems and not args.allow_unverified_result:
        print(f"错误: 这份评测结果不能用于裁决（{len(problems)} 项）：",
              file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("  确认要据此裁决时加 --allow-unverified-result，"
              "并在 milestone 里写明理由。", file=sys.stderr)
        return 1
    for problem in problems:
        print(f"警告: {problem}", file=sys.stderr)

    # 噪声底是 init 时量一次存进 state 的，之后整个 campaign 复用。
    # 环境会变（同卡来了别的进程、发生降频），所以每轮拿本次结果的 cv
    # 复核一次：实测抖动已经高过存量噪声底时，这一轮的判据是失准的。
    stored_floor = float(state.get("noise_floor_pct", 0.0) or 0.0)
    cv_values = result_cv_values(doc)
    if cv_values and stored_floor > 0:
        worst = max(cv_values)
        # 按**倍数**判，不按「超过一点」判。
        #
        # 存量噪声底本来就是从某一次的 cv 量出来的，之后每轮的 cv 在它上下
        # 浮动是正常抽样波动。只要高一点就拒绝，这道门会天天响，然后被
        # 反射性豁免 —— 那就回到了静默放行。
        # 要拦的是「环境真的变了」：抖动高到让存量噪声底不再是个可信的界。
        if worst > stored_floor * STALE_FLOOR_FACTOR:
            if args.allow_stale_noise_floor:
                print(f"警告: 本次实测抖动 {worst:.2f}% 高于存量噪声底 "
                      f"{stored_floor:.2f}%，已按 --allow-stale-noise-floor 放行。",
                      file=sys.stderr)
            else:
                print(f"错误: 本次实测抖动 {worst:.2f}% 已达存量噪声底 "
                      f"{stored_floor:.2f}% 的 {worst / stored_floor:.1f} 倍。",
                      file=sys.stderr)
                print("  噪声底是 init 时量的，这一轮的环境比那时吵 —— "
                      "按旧判据裁决会把抖动当成真实改进。", file=sys.stderr)
                print("  先查环境（同卡有没有别的进程、是否降频），"
                      "再重量并 set-noise-floor；", file=sys.stderr)
                print("  确认这次抖动是偶发、不代表环境变化时加 "
                      "--allow-stale-noise-floor。", file=sys.stderr)
                return 1
        elif worst > stored_floor:
            print(f"提示: 本次实测抖动 {worst:.2f}% 略高于存量噪声底 "
                  f"{stored_floor:.2f}%（正常抽样波动范围内）。"
                  "连续多轮如此就该重量一次。", file=sys.stderr)

    try:
        signature = _load_signature(args.signature)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    # milestone 路径优先用命令行，其次用 init 时记在 state 里的那个。
    #
    # 为什么要有后备：漏传 --milestone 会同时关掉「决策图走穷」和「门禁僵持」
    # 两个外层重入口（两个函数在 milestone_doc 为 None 时直接返回"没问题"），
    # 并把本轮降级为 accept_unverified。一个可选参数静默关掉两道闸，
    # 而最该触发重入的那种调用方式恰恰就是漏传它。
    milestone_path = args.milestone or state.get("milestone_path")
    if not milestone_path:
        # 不拒绝，但必须说出来、而且要留痕。
        #
        # 缺了课程表，「决策图走穷」和「门禁僵持」两个外层重入口都会静默
        # 返回「没问题」（两个函数在 milestone_doc 为 None 时直接早退），
        # 机制也一律判为无法核验。三个重入口只剩一个。
        #
        # 早先这里是直接拒绝的，但那样会把 record 的其他正当用法一起卡住，
        # 结果是人反射性地加豁免参数 —— 那和静默失效是同一件事，只是多敲几个字。
        # 真正的缺陷是「失效了没人知道」，所以改成告警 + 在台账里标记本轮
        # 哪两道闸没被评估过。
        print("警告: 本轮没有 milestone。两个外层重入口（决策图走穷 / 门禁僵持）"
              "不会被评估，", file=sys.stderr)
        print("      机制也一律判为无法核验（不计入 best）。"
              "台账里会标记这一轮。", file=sys.stderr)
        print("      用 --milestone <路径>，或在 init 时 --milestone-path "
              "记进 state 以免每次都要传。", file=sys.stderr)
    try:
        milestone_doc = load_milestone(
            Path(milestone_path) if milestone_path else None)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    # 这一轮测的是哪一份代码。
    #
    # 不记的话，「被保留下来的最后一条就是当前代码的样子」这个前提无法核实，
    # 而机制做差整个建立在它上面。候选目录不一定在版本控制里（评测要的是
    # 装好的 OPP 安装路径），所以用内容指纹代替 git sha。
    artifact_sha = args.artifact_sha or ""
    if not artifact_sha and args.artifact:
        artifact_path = Path(args.artifact)
        if artifact_path.exists():
            artifact_sha = directory_fingerprint(artifact_path)
        else:
            print(f"警告: --artifact 指向的 {artifact_path} 不存在，"
                  "本轮没有代码身份记录。", file=sys.stderr)
    if not artifact_sha:
        print("警告: 本轮没有记录代码身份（--artifact 或 --artifact-sha）。",
              file=sys.stderr)
        print("      机制做差假设「上一条保留下来的就是当前代码」—— "
              "没有指纹这件事无法核实：", file=sys.stderr)
        print("      该回滚却没回滚、或在更早的树上接着改，"
              "基线就是错的而百分比看着正常。", file=sys.stderr)

    previous_sha = None
    for entry_before in reversed(generation_entries(state)):
        if entry_before.get("artifact_sha"):
            previous_sha = entry_before["artifact_sha"]
            break
    if artifact_sha and previous_sha and artifact_sha == previous_sha:
        print(f"警告: 本轮产物指纹与上一轮相同（{artifact_sha}）—— "
              "代码没有变化。", file=sys.stderr)
        print("      对未改动的树重测只是多抽一次噪声；"
              "当成新候选保留会让 best 由运气决定。", file=sys.stderr)
        print("      要记一次确认读数请用 campaign.py confirm。",
              file=sys.stderr)

    precision_report = None
    if args.precision_report:
        try:
            precision_report = json.loads(
                Path(args.precision_report).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"错误: 无法读取精度报告 {args.precision_report}: {exc}",
                  file=sys.stderr)
            return 2
        except json.JSONDecodeError as exc:
            print(f"错误: {args.precision_report} 不是合法 JSON: {exc}",
                  file=sys.stderr)
            return 2
        if precision_report.get("verdict") == "unavailable":
            print("警告: 精度误差比判不了（缺参考语义）。本轮只有容差比对，",
                  file=sys.stderr)
            print("      而容差分不出「在参考舍入的地方舍入」和"
                  "「用 bf16 累加 / tf32 相乘」。", file=sys.stderr)
            print("      另外容差的锚是 baseline 的输出 —— promote 之后它变成"
                  "上一代产物，", file=sys.stderr)
            print("      于是每代允许再漂一个容差，N 代之后与真值的距离无界。",
                  file=sys.stderr)
    else:
        print("提示: 本轮没有误差比报告。容差比对的锚会随代际漂移 ——",
              file=sys.stderr)
        print("      先跑 precision_gate.py --output <报告>，"
              "再 record --precision-report <报告>。", file=sys.stderr)

    structure_report = None
    if args.structure_report:
        try:
            structure_report = json.loads(
                Path(args.structure_report).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"错误: 无法读取结构报告 {args.structure_report}: {exc}",
                  file=sys.stderr)
            return 2
        except json.JSONDecodeError as exc:
            print(f"错误: {args.structure_report} 不是合法 JSON: {exc}",
                  file=sys.stderr)
            return 2
    else:
        print("提示: 本轮没有结构核验报告。机制核验只能靠停顿指标反驳，",
              file=sys.stderr)
        print("      而停顿能被别的改动顶下去（减小 tile、改对齐都会让它降）",
              file=sys.stderr)
        print("      —— 一个退化回退的实现可以凭一个恰好变好的停顿数拿到"
              "「机制已核实」。", file=sys.stderr)
        print("      先跑 structure_probe.py --output <报告>，"
              "再 record --structure-report <报告>。", file=sys.stderr)

    fence_report = None
    if args.fence_report:
        try:
            fence_report = json.loads(
                Path(args.fence_report).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"错误: 无法读取围栏报告 {args.fence_report}: {exc}",
                  file=sys.stderr)
            return 2
        except json.JSONDecodeError as exc:
            print(f"错误: {args.fence_report} 不是合法 JSON: {exc}",
                  file=sys.stderr)
            return 2
    else:
        print("提示: 本轮没有围栏报告 —— 这一轮没有被查过"
              "「是不是转发给了参考实现」。", file=sys.stderr)
        print("      候选直接调厂商实现时，加速比、精度、所有门禁全过，"
              "而一行 kernel 都没写；", file=sys.stderr)
        print("      自我提升场景更隐蔽：第 n 代转发给第 n−1 代，"
              "链条可以空转很多轮，", file=sys.stderr)
        print("      每轮都「没有退化」。", file=sys.stderr)
        print("      围栏是静态扫描，不花机时：", file=sys.stderr)
        print("        python3 <forge-probes>/scripts/fence_check.py "
              "--candidate <候选> \\", file=sys.stderr)
        print("            --op-name <算子> --json > fence.json",
              file=sys.stderr)
        print("      然后 record --fence-report fence.json。"
              "没有它这一轮不会抬高 best。", file=sys.stderr)

    entry = apply_result(
        state, doc, args.decision, milestone_doc=milestone_doc,
        signature=signature, recipes_used=args.recipe,
        recipes_invented=args.invented, symptom=args.symptom,
        signature_path=args.signature or "",
        artifact_sha=artifact_sha,
        base_sha=args.base_sha or previous_sha or "",
        milestone_sha=_milestone_fingerprint(milestone_doc) or "",
        structure_report=structure_report,
        precision_report=precision_report,
        fence_report=fence_report,
    )
    if milestone_doc is None:
        # 让「这一轮是在没有课程表的情况下裁决的」可查，否则事后无法区分
        # 「重入口没触发」和「重入口根本没被评估」。
        entry["gates_not_evaluated"] = ["graph_exhausted", "gate_impasse"]
    else:
        # 判据换过没有。阈值被改宽、或机制行被删掉，早先在台账里一点痕迹
        # 都没有 —— 那是一条完整的「挪球门」路径。
        current_ms = entry.get("milestone_sha256")
        previous_ms = None
        for before in reversed(generation_entries(state)[:-1]):
            if before.get("milestone_sha256"):
                previous_ms = before["milestone_sha256"]
                break
        if previous_ms and current_ms and previous_ms != current_ms:
            print(f"注意: 本轮的判据与上一轮不是同一份课程表"
                  f"（{previous_ms} → {current_ms}）。", file=sys.stderr)
            print("      两轮的裁决不可直接相比；"
                  "若这是一次修订，请确认它走过 reopen。", file=sys.stderr)
    save_state(path, state)

    print(f"决策 {args.decision}: {entry['verdict']}")
    print(f"  {entry['reason']}")

    structure = entry.get("structure")
    if isinstance(structure, dict) and structure.get("status") != "unchecked":
        marks = {"ok": "结构成立", "absent": "结构不在"}
        print(f"\n  结构核验（{marks.get(structure['status'], structure['status'])}）："
              f"{structure['reason']}")
        if structure["status"] == "absent":
            print("        这是实现问题 —— 该修实现或给实现补一张经验卡片，")
            print("        **不要**改课程表：改措辞不会让一个「不在」"
                  "变成「在」。")

    mech_for_attr = entry.get("mechanism")
    if isinstance(mech_for_attr, dict) and mech_for_attr.get("attribution"):
        labels = {"implementation": "实现没做到（课程表读对了参考）",
                  "curriculum": "课程表读错了参考"}
        attribution = mech_for_attr["attribution"]
        print(f"\n  归因：{labels.get(attribution, attribution)}")
        if attribution == "implementation":
            print("        补一张经验卡片记下这个退化回退，不要改课程表。")
        else:
            print("        reopen 那条决策 —— 在一个错的前提底下调实现是白费。")

    coupling = entry.get("gate_coupling")
    if isinstance(coupling, dict):
        print(f"\n  课程表对账：{coupling['detail']}", file=sys.stderr)

    mechanism = entry.get("mechanism")
    if isinstance(mechanism, dict) and mechanism.get("lines"):
        print(f"\n  机制核验（做差基线：{mechanism.get('compared_against', '无')}）:")
        for line in mechanism["lines"]:
            if line.get("vacuous"):
                mark = "平凡通过"
            else:
                mark = {True: "成立", False: "不成立",
                        None: "无法求值"}[line["verified"]]
            print(f"    [{mark}] {line['id']} ({line.get('label')}): {line['reason']}")

    if entry["verdict"] == VERDICT_ACCEPT_UNVERIFIED:
        print("\n  注意: 该改动保留但不计入 best，也不得作为知识库入库依据。")
        print("        要让它算数，补上 --signature 与 --milestone 重新核验机制。")

    if entry["verdict"] in (VERDICT_REVERT, VERDICT_REGRESSION):
        # 精度不过 / 压根没测出数，这两种是**实现**失败，不是这个方向走不通。
        #
        # 死路表的前提是「测过且无效」（hwkb 的 nogo add 也拒绝无证据的条目）。
        # 把一次精度 bug 登记成死路，等于判定这条架构方向不可行 —— 后续代次
        # 不会再试它，而它其实从未被真正测量过。原先这类条目还会被盖上
        # [MEASURED] 的章，看起来证据充分。
        failure_kind = entry.get("failure_kind")
        if failure_kind in ("precision", "no_measurement"):
            label = {"precision": "精度不达标",
                     "no_measurement": "没有有效测量"}[failure_kind]
            print(f"\n  不登记死路：本次是{label}，属实现失败。")
            print("        死路表只收「测过且确实无效」的方向 —— 收进一个没测过的，")
            print("        这条架构方向会被永久跳过，而它从未被真正测量。")
            print("        先把实现修对再测一次；若修不好，由架构层决定要不要弃用。")
        elif args.nogo_kb_root:
            try:
                count = write_nogo(Path(args.nogo_kb_root), state, entry)
            except CampaignError as exc:
                print(f"\n错误: 登记死路失败: {exc}", file=sys.stderr)
                return 1
            print(f"\n  已登记死路（当前 {count} 条）: "
                  f"{Path(args.nogo_kb_root) / NOGO_FILENAME}")
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

    # 徒劳重开上限。计数**不随任何一次 reopen 归零** —— 下面那三个计数器
    # 归零是对的（新架构假设要给足机会），但「重开了几次、几次白干」必须
    # 跨重开存活，否则「重开 → 几轮不够 → 停滞 → 再重开」可以永久循环，
    # 而每一轮的 status 看起来都健康。
    reopen_stats = state.setdefault("reopen", {"total": 0, "fruitless": 0})
    if int(reopen_stats.get("fruitless", 0)) >= REOPEN_MAX_FRUITLESS \
            and not args.force:
        print(f"错误: 已连续 {reopen_stats['fruitless']} 次重开没有带来新的"
              f"下界（上限 {REOPEN_MAX_FRUITLESS}）。", file=sys.stderr)
        print("  再开一次很可能还是原地打转 —— 每次重开都会作废整条下游的"
              "实现、把 best 归零。", file=sys.stderr)
        print("  该做的是换一件事：调低目标增益、换 shape、"
              "或者承认这个方向到此为止。", file=sys.stderr)
        print("  确实要再开一次时加 --force（会重置这个计数）。",
              file=sys.stderr)
        return 1

    # 重开定价。没有价格就没有重开。
    if not args.price and not args.force:
        print("错误: 重开必须定价。", file=sys.stderr)
        print("  重开会按依赖反向闭包作废整条下游的实现、把 best 归零、"
              "三个计数器清零 ——", file=sys.stderr)
        print("  这是整套流程里代价最大的单个动作，理由不能是一句自由文本。",
              file=sys.stderr)
        print("  形状：--price \"<Δ%> via lever:<手段名>\"", file=sys.stderr)
        print("        来源也可以是 probe:<路径> / kb:<卡片 id> / "
              "ablation:<消融名>", file=sys.stderr)
        print("  停滞之后的第一件事本来就是用 lever_probe.py 给现有手段定价，"
              "拿那个数填进来。", file=sys.stderr)
        return 2
    if args.price:
        ok, detail = verify_reopen_price(args.price, args.state,
                                         args.lever_store, args.kb_root)
        if not ok:
            if args.force:
                print(f"警告: 价格核对未通过：{detail}", file=sys.stderr)
                print("  已按 --force 继续。", file=sys.stderr)
            else:
                print(f"错误: 价格核对未通过：{detail}", file=sys.stderr)
                return 1
        else:
            print(f"价格已核对：{detail}")

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
    for entry in generation_entries(state):
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
    unver_reset = state.setdefault("mechanism_unverified", {})
    unver_reset["consecutive"] = 0
    unver_reset["streak_refuted"] = 0
    unver_reset["streak_unknown"] = 0
    state.pop("stall_reason", None)
    state.pop("stall_reasons", None)
    state.pop("stall_detail", None)
    rollback = rollback_target(state, invalidated)
    state.setdefault("ledger", []).append({
        "at": _now(),
        "generation": state.get("generation", 0),
        "input_shape": state.get("input_shape"),
        "decision": args.decision,
        "gain": None,
        # 重开是一次**课程表修订**，所以记成 patch 类条目而不只是一行 reason。
        # 不记的话，「这条决策被改过、从哪个课程表指纹改到哪个、结果如何」
        # 在台账里完全无痕 —— 而 status 要据此把已证伪的课程表方向列出来，
        # 否则下一代会把同一个方向原封不动再试一遍。
        "verdict": "reopen",
        "patch": {
            "locus": args.decision,
            "direction": args.reason or "未填",
            "milestone_sha_before": _milestone_fingerprint(milestone_doc),
            "invalidated_decisions": sorted(invalidated),
            "price": args.price or None,
            # 结果留空，由下一次抬高 best 或下一次重开回填。
            "outcome": "pending",
        },
        "reason": (f"重开决策 {args.decision}；按 depends_on 反向闭包作废 "
                   f"{len(affected)} 条下游决策的 {count} 条台账记录。"
                   f"理由：{args.reason or '未填'}"),
        "invalidated_decisions": sorted(invalidated),
        "price": args.price or None,
        "rollback_to": rollback,
        "counts_toward_best": False,
        "kb_admissible": False,
    })
    reopen_stats["total"] = int(reopen_stats.get("total", 0)) + 1
    # 本次重开先按「无果」计入；下一次抬高 best 时由 apply_result 清零。
    # 方向取保守侧：没有证据说明它有用之前，就当它没用。
    reopen_stats["fruitless"] = (0 if args.force
                                 else int(reopen_stats.get("fruitless", 0)) + 1)
    reopen_stats["last_price"] = args.price or None
    save_state(path, state)

    print(f"已重开 {args.decision}")
    if affected:
        print(f"  受影响的下游决策: {', '.join(sorted(affected))}")
    else:
        print("  没有下游决策依赖它")
    print(f"  作废台账记录: {count} 条")
    print(f"  best 重算为: {best_gain:.4f}")
    if rollback is not None:
        print(f"\n  回退到: {rollback['artifact_sha']}"
              f"（决策 {rollback['decision']}，增益 {rollback['gain']:.4f}）")
        print("        这是作废范围之外、最后一个机制已核实的产物。")
        print("        重开之后的实现应当从它开始，而不是从当前工作树 ——")
        print("        当前树上带着那些刚被作废的改动。")
    else:
        print("\n  回退到: 没有可回退的产物（本代还没有机制已核实的记录，"
              "或它们都在作废范围内）")
        print("        从参考基线重新开始。")
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
    if args.max_records is not None:
        budget["max_records"] = args.max_records
    if args.max_wall_hours is not None:
        budget["max_wall_hours"] = args.max_wall_hours
    if args.max_records is None and args.max_wall_hours is None:
        print("错误: 至少要给 --max-records 或 --max-wall-hours 之一",
              file=sys.stderr)
        return 2
    if state.get("state") == STATE_STOPPED and "预算耗尽" in str(
            state.get("stop_reason", "")):
        state["state"] = STATE_IMPLEMENT
        state.pop("stop_reason", None)
    save_state(path, state)
    if args.max_records is not None:
        print(f"预算已设为 {args.max_records} 次评测"
              f"（已用 {budget.get('used', 0)}）")
    if args.max_wall_hours is not None:
        elapsed = _elapsed_hours(state)
        used = f"{elapsed:.1f}" if elapsed is not None else "?"
        print(f"墙钟预算已设为 {args.max_wall_hours} 小时（已过 {used}）")
    return 0


def generation_next(state: Dict[str, Any]) -> int:
    """本次将要封存的产物属于第几代。promote 之前 generation 还没加。"""
    return int(state.get("generation", 0))


def reference_time_us(state: Dict[str, Any]) -> Optional[float]:
    """反推参考自己的耗时。

    评测给的是 gain = t_参考 / t_候选，台账里存了候选的 time_us，
    两者相乘即得参考耗时。没有单独记一份参考耗时，是因为它本来就是
    从同一次配对测量里来的 —— 另存一份反而会出现两个可能不一致的数。
    """
    for entry in state.get("ledger", []):
        if entry.get("invalidated"):
            continue
        gain = entry.get("gain")
        time_us = entry.get("time_us")
        if (isinstance(gain, (int, float)) and gain > 0
                and isinstance(time_us, (int, float)) and time_us > 0):
            return gain * time_us
    return None


def ensure_reference_in_library(lib, library_root: Path,
                                state: Dict[str, Any]) -> Optional[Any]:
    """库里还没有这个算子时，先把参考补登为 gen0。

    不补的话血统会从我们自己的 kernel 开始，「累计相对 gen0」读起来就像
    是相对自己 —— 而整条自我提升链条的意义恰恰在于相对最初那个参考。
    """
    # 存在性判断交给 add_generation 在锁内做（only_if_new=True），不要在这里
    # 先读索引再调过去登记：那两步之间没有锁，另一个并行任务可以在这段时间里
    # 建好同一个键，于是这次登记会把参考接在别人的链上成为 gen1，血统错位。
    # 并行跑多任务、共用产物库是这个插件明确支持的用法，所以这个窗口会被撞上。
    reference = state.get("reference") or {}
    snapshot = state.get("reference_snapshot")
    signature_file = None
    if snapshot:
        # add_generation 吃的是签名文件；参考快照已经是压好的形状，
        # 这里临时落一个等价文件交给它，避免为此再开一条入口。
        import tempfile
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({channel: {"channel": channel, "value": value,
                             "present": True}
                   for channel, value in snapshot.items()}, handle)
        handle.close()
        signature_file = handle.name

    try:
        return lib.add_generation(
            library_root,
            operator=str(state.get("operator")),
            arch=str(state.get("arch")),
            input_shape=str(state.get("input_shape")),
            kind=str(reference.get("kind", "builtin_aclnn")),
            identifier=str(reference.get("id", state.get("operator"))),
            artifact=(reference.get("path")
                      if reference.get("kind") not in lib.ARTIFACTLESS_KINDS
                      else None),
            signature=signature_file,
            only_if_new=True,
            time_us=reference_time_us(state),
            note="由 campaign promote 自动补登的初始参考",
        )
    finally:
        if signature_file:
            try:
                os.unlink(signature_file)
            except OSError:
                pass


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
        debts = harvest_debt(state)
        if debts:
            # 这条分支只告警，不拦。拦了的话，一个该停的 campaign 连
            # 「我收敛了」都没法记下来 —— 状态机会卡在 implement，而那
            # 比丢几条经验更糟：之后所有停止条件的判断都从一个错的状态出发。
            print(f"\n注意: 本代还有 {len(debts)} 笔未结的知识欠账。"
                  "链条到此为止，但这些结论仍然值得入库 ——")
            print("      它们是这一代唯一留得下来的东西。"
                  "跑 `campaign.py harvest` 看清单。")
        return 0

    if not _best_is_confirmed(state) and not args.force:
        readings = (state.get("best") or {}).get("readings") or []
        print(f"错误: best 只有 {len(readings)} 次读数（{readings}），"
              "没有经过独立复测。", file=sys.stderr)
        print("  promote 会把这份产物复制进库、带上校验和与血统、"
              "成为下一代的参考，", file=sys.stderr)
        print("  而之后每一代的增益都相对它算。同时 promote 会把 best 归零、"
              "噪声底置为待重测 ——", file=sys.stderr)
        print("  也就是说那个读数本身从此消失，一次运气好的抽样封进去之后"
              "无法回溯。", file=sys.stderr)
        print("  对同一份产物再 record 一次（同一个决策 id），"
              "两次都达标后再 promote；", file=sys.stderr)
        print("  确认要据单次读数提升时加 --force。", file=sys.stderr)
        return 1

    # 知识欠账必须先结清。
    #
    # 这道闸的位置是算过的：排在确认复测**之后**、正式封存**之前**。
    #
    #   在确认之后 —— 读数本身不可信时，先让人去写一张以它为据的卡片
    #                  比不写更糟：那张卡片会被别的 campaign 当成已核实的
    #                  经验用。所以先确认数，再谈入库。
    #   在封存之前 —— promote 把 best 归零、噪声底置为待重测、产物封成下一代
    #                  的参考。台账还在，但产出这些结论的会话上下文已经没了：
    #                  「增益超噪声底且机制已核实」背后的推理过程，过了这一刻
    #                  就只剩一行 reason。这是最后一个还来得及的时刻。
    debts = harvest_debt(state)
    if debts and not args.allow_unharvested and not args.force:
        print(f"错误: 本代有 {len(debts)} 笔未结的知识欠账，先结清再 promote。",
              file=sys.stderr)
        print("", file=sys.stderr)
        print(render_harvest(debts), file=sys.stderr)
        return 1

    # 封存进库：产物复制进去 + 校验和 + 血统。
    # 只记路径是不够的 —— 那个目录被清掉、被覆盖，第 n 代就对不上
    # 第 n-1 代了，而且没有任何地方能看出参考变过。
    library_record = None
    library_seeded = None
    if True:
        # 只看本代：上一代核实过的机制不能替本代背书。这个标志会写进
        # 产物库，而库的准入门禁靠它区分「机制已核实的产物」和「只是跑得快」。
        best_verified = any(
            e.get("counts_toward_best") and not e.get("invalidated")
            for e in generation_entries(state))
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import reference_library as lib  # noqa: PLC0415

            seeded = ensure_reference_in_library(
                lib, Path(args.library), state)
            manifest = lib.add_generation(
                Path(args.library),
                operator=str(state.get("operator")),
                arch=str(state.get("arch")),
                input_shape=str(state.get("input_shape")),
                kind="sealed_kernel",
                identifier=f"{state.get('operator')}_gen{generation_next(state)}",
                artifact=args.artifact,
                signature=args.signature,
                time_us=state.get("best", {}).get("time_us"),
                gain=gain,
                device_id=(state.get("device") or {}).get("device_id"),
                mechanism_verified=best_verified,
                note=args.note,
            )
            library_record = manifest
            library_seeded = seeded
        except Exception as exc:  # noqa: BLE001 - 入库失败不该吞掉 promote
            print(f"错误: 封存进库失败: {exc}", file=sys.stderr)
            print("  promote 未执行 —— 产物没进库就提升，下一代会拿不到它。",
                  file=sys.stderr)
            return 2

    # 新参考必须真的被封存，`sealed: True` 不能只是个字段。
    #
    # promote 之前这里只写一个标志位，而全仓只有 reference_library.ingest
    # 会调 seal_reference.seal —— promote 走的是 add_generation，不封。
    # 于是第 1 代起，编码 agent 可以直接 Read 第 0 代那份自己写的 kernel
    # （它就是新参考），而台账里写着「源码不跨代边界」。
    #
    # 另外参考路径要指向**库里的只读拷贝**，不是工作区里那个候选目录 ——
    # 后者可读可写，改动它会直接改掉下一代的计时基线。
    generation = int(state.get("generation", 0)) + 1
    sealed_source = args.artifact
    if library_record and library_record.get("artifact_dir"):
        candidate = Path(args.library) / library_record["artifact_dir"]
        if candidate.is_dir():
            sealed_source = str(candidate)

    reseal_note = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import seal_reference as sealer  # noqa: PLC0415

        work_dir = Path(args.state).resolve().parent
        manifest = sealer.seal(
            Path(sealed_source),
            f"{state.get('operator')}_gen{generation - 1}",
            work_dir,
        )
        reseal_note = (f"已封存新参考：{manifest['sealed_file_count']} 个文件"
                       f"（{sealed_source}）")
    except Exception as exc:  # noqa: BLE001 - 封存失败不能让 promote 悄悄过去
        print(f"错误: 新参考封存失败: {exc}", file=sys.stderr)
        print("  promote 未完成。不封存就提升，下一代可以直接读上一代的源码，", 
              file=sys.stderr)
        print("  而「架构结论从硬件行为反推」这个前提就破了。", file=sys.stderr)
        return 2

    state["generation"] = generation
    state["reference"] = {
        "kind": "sealed_kernel",
        "id": f"{state.get('operator')}_gen{generation - 1}",
        "path": sealed_source, "sealed": True,
        "generation": generation - 1, "promoted_gain": gain,
        "library_dir": (str(Path(args.library)
                            / f"{state.get('arch')}")
                        if library_record else None),
        "library_key": library_record["key"] if library_record else None,
    }
    state["state"] = STATE_ANALYZE
    state["cursor"] = {"milestone": None, "decision": None, "iteration": 0}
    state.setdefault("stagnation", {})["since_improvement"] = 0
    state.setdefault("implementation_failures", {})["consecutive"] = 0
    unver_reset = state.setdefault("mechanism_unverified", {})
    unver_reset["consecutive"] = 0
    unver_reset["streak_refuted"] = 0
    unver_reset["streak_unknown"] = 0
    state["target_gain"] = args.phase_target_gain
    state["best"] = {"gain": 1.0, "time_us": None, "artifact": None}
    # 参考换了，噪声底和参考签名都必须重取 —— 新参考是另一个二进制，
    # 拿上一代的签名做差会把两代之间的差异算到本代改动头上。
    state["noise_floor_status"] = NOISE_PENDING
    state["reference_snapshot"] = None
    # device 原样保留：代际之间不换卡
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
    if reseal_note:
        print(f"  {reseal_note}")
    if library_record:
        if library_seeded:
            print(f"  补登参考  gen{library_seeded['generation']} "
                  f"{library_seeded['kind']}:{library_seeded['id']}"
                  "（库里原本没有这个算子）")
        print(f"  已入库    {library_record['key']} gen{library_record['generation']}"
              f"（{len(library_record['artifact_files'])} 个文件）")
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
    print(f"机制不算数 {unver.get('consecutive')}/{unver.get('threshold')}"
          f"（其中被证伪 {unver.get('streak_refuted', 0)}，"
          f"无法核验 {unver.get('streak_unknown', 0)}）")
    cap = budget.get("max_records")
    print(f"预算      {budget.get('used', 0)}/{cap if cap else '不限'}")
    wall_cap = budget.get("max_wall_hours")
    elapsed = _elapsed_hours(state)
    if wall_cap or elapsed is not None:
        used = f"{elapsed:.1f}" if elapsed is not None else "?"
        print(f"墙钟      {used}/{wall_cap if wall_cap else '不限'} 小时"
              + ("" if elapsed is None else
                 "（按 created_at 算差，含停机时间 —— 是机时的上界）"))
    device = state.get("device") or {}
    if device.get("device_id") is not None:
        print(f"设备      device {device['device_id']}")
    print(f"台账条目  {len(state.get('ledger', []))}")

    # 已证伪的课程表方向。
    #
    # 这些方向每一个都花掉了一次重开的代价（作废整条下游、best 归零）。
    # 不列出来的话，下一代的架构层看不到它们，会把同一个方向原封不动再试
    # 一遍 —— 而那一遍的代价和上一遍一样大。
    refuted_patches = [
        entry["patch"] for entry in state.get("ledger", [])
        if isinstance(entry.get("patch"), dict)
        and entry["patch"].get("outcome") != "accepted"
    ]
    if refuted_patches:
        print(f"\n已试过的课程表修订（{len(refuted_patches)} 次）：")
        for item in refuted_patches:
            print(f"  · {item.get('locus')}: {item.get('direction')}"
                  f"  [{item.get('outcome', 'pending')}]")
            if item.get("price"):
                print(f"      当时的定价: {item['price']}")
        print("  这些方向每一个都花掉了一次重开的代价（作废整条下游、"
              "best 归零）。")
        print("  换方向之前先读一遍 —— 重试同一个方向的代价和上一次一样大。")
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


#: 一个决策烧掉多少轮评测之后，它的过程本身就值得入库。
#:
#: 上游叫 `kb_tripwire_rounds`，默认 6。理由不是「花得多就该记」，而是
#: **花得多说明那一轮有东西没被预见到** —— 课程表以为一步能到，实际走了
#: 六步。中间那五次失败里藏着一条硬件事实，而它只存在于那次会话的上下文里。
KB_TRIPWIRE_ROUNDS = 6


def harvest_debt(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """算出「该入库而还没入库」的欠账。

    为什么要自动算：知识沉淀排在流程第 11 步，也就是一次长 campaign 的
    末尾 —— 正是上下文最紧、最容易被跳过的时刻。本仓就是证据：整套
    dav2201 流程建完，库里 1 张卡。靠提示词提醒的事，到那个时刻不会发生。

    两类欠账，各自对应一类「已经付过代价、但结论没留下」的情形：

      admissible   台账判了 kb_admissible（增益超噪声底**且**机制已核实），
                   却没有任何 recipes_invented。这是一条已经被证实的
                   硬件经验，代价已经付了，只差写下来。

      tripwire     一个决策烧掉 ≥ KB_TRIPWIRE_ROUNDS 轮才拿到接受判定。
                   花得多说明有东西没被预见到，而那个发现只存在于
                   那次会话里 —— 这一代一结束就没了。

    只看本代。上一代的欠账在上一次 promote 时就该结清，再算一遍只会让
    列表永远非空，然后 promote 那道闸被反射性豁免掉。
    """
    entries = [e for e in generation_entries(state) if not e.get("invalidated")]
    debts: List[Dict[str, Any]] = []

    # 本代一共写下了哪些卡片。按决策分不开 —— 一张卡片常常是几轮的合并
    # 结论，所以这里只问「这一代到底写了没有」。
    invented_anywhere = any(e.get("recipes_invented") for e in entries)

    for entry in entries:
        if not entry.get("kb_admissible"):
            continue
        if entry.get("recipes_invented"):
            continue
        mechanism = entry.get("mechanism") or {}
        debts.append({
            "kind": "admissible",
            "decision": entry.get("decision"),
            "gain": entry.get("gain"),
            "symptom": entry.get("symptom") or "",
            "mechanism_reason": mechanism.get("reason") or "",
            "mechanism_lines": mechanism.get("lines") or [],
            "signature_path": entry.get("signature_path"),
            "why": ("台账判了 kb_admissible —— 增益超噪声底且机制已核实。"
                    "这条经验的代价已经付过了，只差写下来"),
        })

    attempts: Dict[str, int] = {}
    accepted_at: Dict[str, int] = {}
    for entry in entries:
        decision = entry.get("decision")
        if not decision:
            continue
        attempts[decision] = attempts.get(decision, 0) + 1
        if (entry.get("verdict") in (VERDICT_ACCEPT, VERDICT_ACCEPT_UNVERIFIED)
                and decision not in accepted_at):
            accepted_at[decision] = attempts[decision]
    for decision, rounds in sorted(accepted_at.items()):
        if rounds < KB_TRIPWIRE_ROUNDS:
            continue
        if invented_anywhere:
            # 这一代确实写了卡片，就不再为「花得多」单独记一笔欠账。
            # 这类欠账的目的是让人去写，写过了就不必再催。
            continue
        debts.append({
            "kind": "tripwire",
            "decision": decision,
            "rounds": rounds,
            "why": (f"这个决策烧掉 {rounds} 轮才拿到接受判定（阈值 "
                    f"{KB_TRIPWIRE_ROUNDS}）。花得多说明有东西没被预见到，"
                    "而那个发现只存在于那次会话的上下文里"),
        })
    return debts


def render_harvest(debts: List[Dict[str, Any]]) -> str:
    """把欠账渲染成「照着它就能写卡片」的样子。

    不只列条目号。欠账清单要是还得让人回头翻台账找读数，到那一步就没人写了 ——
    所以把症状、机制结论、机制行、签名文件路径都带上。
    """
    if not debts:
        return "没有未结的知识欠账。"
    lines = [f"{len(debts)} 笔未结的知识欠账：", ""]
    for index, debt in enumerate(debts, 1):
        lines.append(f"[{index}] {debt['kind']}  决策 {debt.get('decision')}")
        lines.append(f"    为什么: {debt['why']}")
        if debt.get("gain") is not None:
            lines.append(f"    增益:   {(float(debt['gain']) - 1) * 100:+.2f}%")
        if debt.get("symptom"):
            lines.append(f"    症状:   {debt['symptom']}")
        if debt.get("mechanism_reason"):
            lines.append(f"    机制:   {debt['mechanism_reason']}")
        for line in debt.get("mechanism_lines") or []:
            anchor = (line.get("anchor") or {}) if isinstance(line, dict) else {}
            label = line.get("label") if isinstance(line, dict) else line
            lines.append(f"    机制行: {label}  "
                         f"{anchor.get('expr', '')}")
        if debt.get("signature_path"):
            lines.append(f"    签名:   {debt['signature_path']}"
                         "（前后通道读数在这里）")
        lines.append("")
    lines += [
        "怎么结清：",
        "  1. 按上面的读数起草卡片（正文中文，四小节：机制/触发症状/"
        "最小可运行例/已知坑）",
        "  2. 参考侧细节写进 <kb_root>/provenance/<卡片 id>.md，"
        "卡片里写 provenance_archive 指针",
        "  3. python3 <forge-hwkb>/scripts/hwkb.py validate --card draft.md",
        "  4. python3 <forge-hwkb>/scripts/hwkb.py add --card draft.md",
        "  5. record 时带 --invented <卡片 id>，这笔欠账才算结清",
        "",
        "确实不值得入库时（比如那条增益来自一次纯参数调整，没有可复用的"
        "硬件事实），",
        "promote 加 --allow-unharvested 显式放过 —— 它会写进台账，"
        "而不是悄悄消失。",
    ]
    return "\n".join(lines)


def cmd_harvest(args: argparse.Namespace) -> int:
    path = Path(args.state)
    try:
        state = load_state(path)
    except CampaignError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    debts = harvest_debt(state)
    if args.as_json:
        print(json.dumps({"debt_count": len(debts), "debts": debts},
                         ensure_ascii=False, indent=2))
    else:
        print(render_harvest(debts))
    return 1 if debts else 0


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
    p_init.add_argument("--kb-root", default=None,
                        help="知识库根（forge-hwkb/references）。给了就在开跑"
                             "之前把整棵树过一遍 lint —— 知识库是唯一跨 "
                             "campaign 存活的东西，而一张越界的卡片会直接进到"
                             "架构层的输入里")
    p_init.add_argument("--allow-dirty-kb", action="store_true",
                        help="带着知识库的 lint 问题开跑")
    p_init.add_argument("--wall-hours", type=float, default=None,
                        help="墙钟预算（小时，自 init 起算）。不给则不限。"
                             "次数预算挡不住「每轮都很快但时间在走」那种"
                             "消耗 —— 构建失败重试、环境排障、等卡都不"
                             "消耗次数")
    p_init.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help=f"评测次数上限（默认 {DEFAULT_BUDGET}）。"
                             "传 0 表示不限 —— 但不限意味着循环没有任何出口，"
                             "停滞判据失效时会一直跑下去")
    p_init.add_argument("--device-id", type=int, required=True,
                        help="本 campaign 使用的卡号。一个任务一张卡；"
                             "代际之间不换卡 —— 换了之后参考签名、噪声底、"
                             "绝对耗时全部不可比，而这些不可比是静默的")
    p_init.add_argument("--reference-signature", default=None,
                        help="参考实现的 signature_probe 输出。机制行要靠它"
                             "做差（第 0 轮基线）；不给则首轮无法构成顺序证据")
    p_init.add_argument("--reference-spec", default=None,
                        help=f"reference_adapter.py prepare 产出的 "
                             f"{SPEC_FILENAME}。不给则在工作目录里自动查找；"
                             "找到后会校验派发是否已确认")
    p_init.add_argument("--allow-unknown-arch", action="store_true",
                        help="接受不在已知列表里的架构串。"
                             "新硬件才用，且要同时在 forge-hwkb 登记该架构")
    p_init.add_argument("--allow-untrusted-signature", action="store_true",
                        help="接受自判不可信的参考签名作第 0 轮基线。"
                             "这会让后续所有机制做差建立在错的起点上")
    p_init.add_argument("--allow-unverified-dispatch", action="store_true",
                        help="参考声明为内置实现但派发未确认时仍然开 campaign")
    p_init.add_argument("--sealed-paths", default=None,
                        help="封存清单位置。不给则在工作目录与 "
                             "FORGE_SEALED_PATHS 里找")
    p_init.add_argument("--allow-no-boundary", action="store_true",
                        help="边界自检不通过时仍然开 campaign。"
                             "架构结论将无法证明不是抄的")
    p_init.add_argument("--milestone-path", default=None,
                        help="课程表路径，记进 state。record 不传 --milestone "
                             "时用它 —— 漏传会静默关掉两个外层重入口")
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
    p_record.add_argument("--precision-report", default=None,
                          help="precision_gate.py 的输出（对 float64 真值的"
                               "误差比）。它优先于评测结果里的容差比对 —— "
                               "容差分不出降精度换速度，而且锚点会随代际漂移")
    p_record.add_argument("--structure-report", default=None,
                          help="structure_probe.py 的输出。结构断言不成立时"
                               "直接判机制被证伪并归因到实现 —— "
                               "这一关在任何性能数之前就能给出结论")
    p_record.add_argument("--fence-report", default=None,
                          help="fence_check.py --json 的输出。围栏命中直接回滚"
                               "并归因到实现；**没有它这一轮不会抬高 best** —— "
                               "转发给参考实现时所有门禁都会过，"
                               "而一行 kernel 都没写")
    p_record.add_argument("--artifact", default=None,
                          help="本轮候选的目录（已装好的 OPP 路径）。"
                               "用来算内容指纹，记进台账作代码身份 —— "
                               "机制做差的基线靠它才可核实")
    p_record.add_argument("--artifact-sha", default=None,
                          help="直接给代码身份指纹（有 git sha 时可用它）")
    p_record.add_argument("--base-sha", default=None,
                          help="本轮构建在哪个版本之上。不给则取上一轮的指纹；"
                               "reopen 之后应显式给更早那个")
    p_record.add_argument("--allow-unverified-result", action="store_true",
                          help="接受 gating 不合格或卡号不符的结果。"
                               "裁决会建立在一份来源未核实的测量上")
    p_record.add_argument("--allow-stale-noise-floor", action="store_true",
                          help="本次实测抖动高于存量噪声底时仍然裁决")
    p_record.add_argument("--milestone", default=None,
                          help="milestone.json；机制核验与走穷/僵持判定需要它")
    p_record.add_argument("--recipe", action="append", default=[],
                          help="本次用到的外来经验卡片 id，可重复")
    p_record.add_argument("--invented", action="append", default=[],
                          help="本次新总结的经验卡片 id，可重复")
    p_record.add_argument("--symptom", default="",
                          help="本次的 profiler 症状，回滚时写进死路表")
    p_record.add_argument("--nogo-kb-root", default=None,
                          help="知识库根目录；给了就在回滚时直接写入其中的"
                               "死路表。传目录而非文件路径，是为了让它与 "
                               "hwkb.py 锁同一个文件 —— 互斥要靠结构保证，"
                               "不能指望调用方传对路径")
    p_record.set_defaults(func=cmd_record)

    p_audit = sub.add_parser(
        "audit", help="M0 对账：本轮读数与参考逐通道并排", parents=[common])
    p_audit.add_argument("--signature", required=True,
                         help="本轮的 signature_probe 输出")
    p_audit.add_argument("--milestone", default=None,
                         help="课程表；不给则用 init 记进 state 的那个")
    p_audit.add_argument("--factor", type=float, default=M0_AUDIT_FACTOR,
                         help=f"差到多少倍就必须被裁定（默认 {M0_AUDIT_FACTOR}）")
    p_audit.set_defaults(func=cmd_audit)

    p_confirm = sub.add_parser(
        "confirm", help="并入一次确认复测（同一份产物再测一次）",
        parents=[common])
    p_confirm.add_argument("--result", required=True,
                           help="对同一份产物重新评测的结果")
    p_confirm.add_argument("--decision", required=True,
                           help="拿到 best 的那个决策 id")
    p_confirm.set_defaults(func=cmd_confirm)

    p_neutral = sub.add_parser(
        "neutral", help="记一条「改了代码但没测」的条目", parents=[common])
    p_neutral.add_argument("--decision", required=True)
    p_neutral.add_argument("--sha", default=None,
                           help="当前代码身份。与 --artifact 二选一")
    p_neutral.add_argument("--artifact", default=None,
                           help="候选目录，由它算内容指纹")
    p_neutral.add_argument("--reason", default="",
                           help="为什么这次改动不需要测（重命名、拆函数、"
                                "补注释、提常量而取值不变……）")
    p_neutral.set_defaults(func=cmd_neutral)

    p_reopen = sub.add_parser("reopen", help="重开架构决策并作废下游实现",
                              parents=[common])
    p_reopen.add_argument("--milestone", required=True)
    p_reopen.add_argument("--decision", required=True, help="要重开的决策 id")
    p_reopen.add_argument("--price", default=None,
                          help="重开的价格：`<Δ%%> via lever:<手段名>`。"
                               "来源也可以是 probe:<路径> / kb:<卡片 id> / "
                               "ablation:<消融名>。"
                               "会去定价表核对引用存在且 Δ 超过噪声底")
    p_reopen.add_argument("--kb-root", default=None,
                          help="知识库根（forge-hwkb/references）。"
                               "用 kb:<卡片 id> 定价时必填 —— "
                               "一条转述过来的处方不是价格，要核对才算数")
    p_reopen.add_argument("--lever-store", default=None,
                          help="定价表位置，默认取 state 同目录的 levers.json")
    p_reopen.add_argument("--force", action="store_true",
                          help="跳过定价与徒劳上限。每次重开都作废整条下游，"
                               "用它之前请在 milestone 里写明理由")
    p_reopen.add_argument("--reason", default="")
    p_reopen.set_defaults(func=cmd_reopen)

    p_rs = sub.add_parser("set-reference-signature",
                          help="登记参考签名快照，作为机制做差的第 0 轮基线",
                          parents=[common])
    p_rs.add_argument("--allow-untrusted-signature", action="store_true",
                      help="接受自判不可信的签名作做差基线。"
                           "这会让之后所有机制做差建立在错的起点上")
    p_rs.add_argument("--signature", required=True,
                      help="参考实现的 signature_probe 输出")
    p_rs.set_defaults(func=cmd_set_reference_signature)

    p_nf = sub.add_parser("set-noise-floor", help="更新实测噪声底",
                          parents=[common])
    p_nf.add_argument("--value", type=float, required=True)
    p_nf.set_defaults(func=cmd_set_noise_floor)

    p_bg = sub.add_parser("set-budget", help="设置或提高评测次数上限",
                          parents=[common])
    p_bg.add_argument("--max-records", type=int, default=None)
    p_bg.add_argument("--max-wall-hours", type=float, default=None,
                      help="自 init 起的墙钟上限。次数预算挡不住"
                           "「每轮都很快但时间在走」那种消耗 ——"
                           "构建失败重试、环境排障、等卡都不消耗次数")
    p_bg.set_defaults(func=cmd_set_budget)

    p_promote = sub.add_parser("promote", help="封存本代产出为下一代参考",
                               parents=[common])
    p_promote.add_argument("--artifact", required=True)
    p_promote.add_argument("--phase-target-gain", type=float,
                           default=DEFAULT_PHASE_TARGET_GAIN)
    p_promote.add_argument("--library", required=True,
                           help="封存产物库根目录。产物会被复制进库并记校验和"
                                "与血统。必填 —— 只留路径等于没留：那个目录"
                                "被清掉之后下一代就拿不到上一代的产物，"
                                "而自我提升的前提正是它还拿得到")
    p_promote.add_argument("--signature", default=None,
                           help="本代产物的架构签名，供日后检测漂移")
    p_promote.add_argument("--note", default="")
    p_promote.add_argument("--force", action="store_true",
                           help="即使增益未超噪声底也强行提升")
    p_promote.add_argument("--allow-unharvested", action="store_true",
                           help="带着未结的知识欠账提升。"
                                "欠账确实不值得入库时用它 —— "
                                "它会写进台账，而不是悄悄消失")
    p_promote.set_defaults(func=cmd_promote)

    p_harvest = sub.add_parser(
        "harvest", help="列出「该入库而还没入库」的欠账", parents=[common])
    p_harvest.set_defaults(func=cmd_harvest)

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
