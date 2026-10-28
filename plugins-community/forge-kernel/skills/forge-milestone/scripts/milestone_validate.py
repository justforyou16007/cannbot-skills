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
"""milestone.json 校验器。

ForgeKernel 的 milestone 是「架构决策的课程表」：arch agent 把从参考实现
profiler 行为里反推出的架构决策写成有序的 M0 -> Mk，coding agent 一次实现
一行。本校验器是把论文中「agent 自己的主张不算证据」这条规则真正落地的
地方 —— 它独立于提示词运行，agent 说服不了它。

校验分十条规则，每条对应论文中一项明确要求：

  V1  五元组完整      每个决策必须有 action/purpose/trigger/cost/depends_on
  V2  决策必须有触发器  没有 profiler 症状激活的决策是不被支持的主张
  V3  派生图无环       depends_on 构成 DAG，且引用的 id 必须存在
  V4  数字必须有出处    [MEASURED, job N] 或 [ESTIMATE] + 推导
  V5  门禁三类行齐全    correctness / performance / mechanism
  V6  未决行必须销账    DEFERRED / INFEASIBLE / N/A，各有各的举证责任
  V7  门禁锚点来源受限  只能是参考实测、硬件规格、或两者派生的表达式
  V8  M0 不设绝对性能阈值  该阶段只能用相对参考的比率
  V9  原子组同批落地    atomic_group 的成员不能跨 milestone
  V10 架构决策须命名验证通道  必须指出哪个机器可读字段能验证它真的执行了
  V11 机制行须有标签与停顿锚点 闭合词汇表 + 停顿类指标 + 顺序证据

用法:
    python3 milestone_validate.py milestone.json
    python3 milestone_validate.py milestone.json --json      # 机器可读输出
    python3 milestone_validate.py milestone.json --rules V2,V6  # 只跑指定规则

退出码: 0 = 全部通过; 1 = 有违规; 2 = 文件无法解析
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

FIVE_TUPLE_FIELDS = ("action", "purpose", "trigger", "cost", "depends_on")

LINE_CATEGORIES = ("correctness_lines", "performance_lines", "mechanism_lines")

#: 门禁行的终态。OPEN 表示尚未销账，是 V6 要抓的目标。
RESOLVED_STATUSES = {"PASS", "FAIL"}
DISCHARGE_STATUSES = {"DEFERRED", "INFEASIBLE", "N/A"}
ALL_STATUSES = RESOLVED_STATUSES | DISCHARGE_STATUSES | {"OPEN"}

#: V7：门禁锚点只允许这三类来源。
ALLOWED_ANCHOR_KINDS = {"reference_measurement", "hardware_spec", "derived"}

ARCHITECTURAL = "architectural"
TACTICAL = "tactical"

#: V4：出处标记的两种合法形态。
RE_MEASURED = re.compile(r"^\[MEASURED,\s*job\s+\S+\]")
RE_ESTIMATE = re.compile(r"^\[ESTIMATE\]")

#: V8：M0 阶段禁止出现的绝对时间单位。相对比率（speedup/ratio/x）不在此列。
RE_ABSOLUTE_TIME = re.compile(
    r"\d+(?:\.\d+)?\s*(?:us|µs|ms|ns|us/iter|cycles?)\b", re.IGNORECASE
)

#: V11：机制标签的闭合词汇表，以及每个标签必需的停顿锚点字段。
#:
#: 论文 Appendix B 要求这类机制使用闭合词汇表，且每个标签都要有停顿锚定的
#: 验证。锚点取停顿类字段而非占比类，是因为占比下降有多种成因（包括别的段
#: 变慢），而停顿下降直接说明「不再有人在等」—— 那才是机制落地的证据。
#:
#: 字段定义见 ops/ops-profiling/references/csv_fields_reference.md；
#: 词汇表的完整说明见 references/mechanism-labels.md。改这里要同步改那里。
#: MTE 类字段在 msprof 里写作 ai*_mte2_wait_ratio —— 星号是通配，
#: Cube 与 Vector 各有一份。锚点只写 mte2_wait_ratio 会同时匹配到两个，
#: campaign 的机制核验无法判断说的是哪个单元，会判为无法求值。
#: 所以这里要求带前缀的全名，两种前缀都接受。
MECHANISM_LABELS: Dict[str, Tuple[str, ...]] = {
    "CV_HANDOFF": ("aic_cube_wait_ratio", "aiv_vec_wait_ratio"),
    "SUPPLY_KEEPUP": ("aiv_mte2_wait_ratio", "aic_mte2_wait_ratio"),
    "WRITEBACK_OVERLAP": ("aiv_mte3_wait_ratio", "aic_mte3_wait_ratio"),
    "CONSUMER_PHASING": ("aiv_vec_wait_ratio",),
    "L1_RESIDENCY": ("aic_mte1_wait_ratio", "aic_mte1_ratio"),
    "BANK_LAYOUT": ("aiv_vec_bank_cflt_ratio",),
    "BANKGROUP_STRIDE": ("aiv_vec_bankgroup_cflt_ratio",),
    "RESOURCE_SERIALIZATION": ("aiv_vec_resc_cflt_ratio",),
    "CUBE_SATURATION": ("aic_cube_wait_ratio",),
    "L2_LOCALITY": ("total_hit_rate",),
    "OUTPUT_ALIGNMENT": ("aic_fixpipe_ratio",),
    "EXPOSURE_REDUCTION": ("exposed_pct",),
}


class Violation:
    """一条违规。rule 是规则号，path 定位到出问题的字段。"""

    def __init__(self, rule: str, path: str, message: str) -> None:
        self.rule = rule
        self.path = path
        self.message = message

    def as_dict(self) -> Dict[str, str]:
        return {"rule": self.rule, "path": self.path, "message": self.message}

    def __str__(self) -> str:
        return f"[{self.rule}] {self.path}: {self.message}"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _is_blank(value: Any) -> bool:
    """空值判定。空字符串、纯空白、None 都算缺失；0 和 False 不算。"""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _provenance_ok(marker: Any) -> Tuple[bool, str]:
    """检查一个 provenance 标记是否合法。

    返回 (是否合法, 失败原因)。合法形态有两种：
      [MEASURED, job N]        —— 实测，必须指明是哪次 job
      [ESTIMATE] <推导过程>     —— 估算，必须附推导，否则就是凭空断言
    """
    if _is_blank(marker):
        return False, "缺少 provenance 标记"
    if not isinstance(marker, str):
        return False, f"provenance 必须是字符串，实际是 {type(marker).__name__}"
    text = marker.strip()
    if RE_MEASURED.match(text):
        return True, ""
    if RE_ESTIMATE.match(text):
        # [ESTIMATE] 后面必须跟推导，否则和没有出处等价
        remainder = RE_ESTIMATE.sub("", text, count=1).strip()
        if not remainder:
            return False, "[ESTIMATE] 必须附推导过程，裸标记不构成出处"
        return True, ""
    return False, (
        f"出处标记格式非法: {text!r}；"
        "只接受 '[MEASURED, job N]' 或 '[ESTIMATE] <推导>'"
    )


def _iter_decisions(doc: Dict[str, Any]) -> Iterable[Tuple[str, Dict, Dict]]:
    """遍历所有决策，产出 (milestone_id, milestone, decision)。"""
    for milestone in doc.get("milestones", []) or []:
        if not isinstance(milestone, dict):
            continue
        mid = milestone.get("id", "<无 id>")
        for decision in milestone.get("decisions", []) or []:
            if isinstance(decision, dict):
                yield mid, milestone, decision


def _iter_gate_lines(doc: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict]]:
    """遍历所有门禁行，产出 (milestone_id, category, line)。"""
    for milestone in doc.get("milestones", []) or []:
        if not isinstance(milestone, dict):
            continue
        mid = milestone.get("id", "<无 id>")
        gate = milestone.get("gate") or {}
        if not isinstance(gate, dict):
            continue
        for category in LINE_CATEGORIES:
            for line in gate.get(category, []) or []:
                if isinstance(line, dict):
                    yield mid, category, line


# ---------------------------------------------------------------------------
# V1 — 五元组完整
# ---------------------------------------------------------------------------

def check_five_tuple(doc: Dict[str, Any]) -> List[Violation]:
    """论文 §3.2：每个决策用自然语言陈述其 action/purpose/trigger/cost/dependence。

    缺任何一项，这个决策就无法被独立评估，也无法在 patch 后重执行。
    """
    out: List[Violation] = []
    for mid, _m, decision in _iter_decisions(doc):
        did = decision.get("id", "<无 id>")
        for field in FIVE_TUPLE_FIELDS:
            if field not in decision:
                out.append(Violation(
                    "V1", f"{mid}.{did}",
                    f"五元组缺字段 '{field}'",
                ))
            elif field != "depends_on" and _is_blank(decision.get(field)):
                out.append(Violation(
                    "V1", f"{mid}.{did}.{field}",
                    f"五元组字段 '{field}' 为空",
                ))
        # depends_on 允许是空列表（根决策），但必须是列表
        deps = decision.get("depends_on")
        if "depends_on" in decision and not isinstance(deps, list):
            out.append(Violation(
                "V1", f"{mid}.{did}.depends_on",
                f"depends_on 必须是列表，实际是 {type(deps).__name__}",
            ))
    return out


# ---------------------------------------------------------------------------
# V2 — 决策必须有触发器
# ---------------------------------------------------------------------------

def check_trigger(doc: Dict[str, Any]) -> List[Violation]:
    """论文 §3.2：没有 trigger —— 即激活它的 profiler 症状 —— 的决策是不被支持的主张。

    这是 ForgeKernel 区别于「凭经验堆优化」的核心。每个架构决策都必须能回答
    「是观测到了什么才让你认为需要它」。
    """
    out: List[Violation] = []
    for mid, _m, decision in _iter_decisions(doc):
        did = decision.get("id", "<无 id>")
        trigger = decision.get("trigger")
        if _is_blank(trigger):
            # V1 已经报过缺失，这里只处理「有字段但内容不成立」
            continue
        if not isinstance(trigger, dict):
            out.append(Violation(
                "V2", f"{mid}.{did}.trigger",
                "trigger 必须是对象，至少含 symptom 与 channel",
            ))
            continue
        if _is_blank(trigger.get("symptom")):
            out.append(Violation(
                "V2", f"{mid}.{did}.trigger.symptom",
                "缺少触发症状：说不出是观测到什么才需要这个决策，该决策不成立",
            ))
        if _is_blank(trigger.get("channel")):
            out.append(Violation(
                "V2", f"{mid}.{did}.trigger.channel",
                "缺少机器可读通道：症状必须能被程序复核，不能只是文字描述",
            ))
        observed = trigger.get("observed")
        if isinstance(observed, dict):
            ok, reason = _provenance_ok(observed.get("provenance"))
            if not ok:
                out.append(Violation(
                    "V2", f"{mid}.{did}.trigger.observed.provenance", reason,
                ))
    return out


# ---------------------------------------------------------------------------
# V3 — 派生图无环
# ---------------------------------------------------------------------------

def check_decision_graph(doc: Dict[str, Any]) -> List[Violation]:
    """论文 Appendix C：决策间的派生边构成 DAG，这才允许 patch 后部分重执行。

    有环意味着无法确定重执行的起点；引用不存在的 id 意味着依赖链断裂。
    """
    out: List[Violation] = []
    decisions: Dict[str, List[str]] = {}
    locations: Dict[str, str] = {}
    duplicates: List[Tuple[str, str]] = []

    for mid, _m, decision in _iter_decisions(doc):
        raw_id = decision.get("id")
        if _is_blank(raw_id):
            continue
        did = str(raw_id)
        if did in decisions:
            duplicates.append((did, f"{mid}.{did}"))
            continue
        deps = decision.get("depends_on")
        decisions[did] = [d for d in deps if isinstance(d, str)] if isinstance(deps, list) else []
        locations[did] = f"{mid}.{did}"

    for did, path in duplicates:
        out.append(Violation(
            "V3", path, f"决策 id '{did}' 重复，id 必须全局唯一",
        ))

    # 悬空引用
    for did, deps in decisions.items():
        for dep in deps:
            if dep not in decisions:
                out.append(Violation(
                    "V3", f"{locations[did]}.depends_on",
                    f"依赖的决策 '{dep}' 不存在",
                ))

    # 环检测：迭代式 DFS，三色标记
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {did: WHITE for did in decisions}

    for root in decisions:
        if color[root] != WHITE:
            continue
        # 栈元素: (节点, 待访问依赖的迭代器)
        stack: List[Tuple[str, Iterable[str]]] = [(root, iter(decisions[root]))]
        path_stack: List[str] = [root]
        color[root] = GRAY
        while stack:
            node, it = stack[-1]
            advanced = False
            for dep in it:
                if dep not in decisions:
                    continue  # 悬空引用已单独报过
                if color[dep] == GRAY:
                    # 找到环，截出环上的部分便于定位
                    try:
                        cycle_start = path_stack.index(dep)
                        cycle = path_stack[cycle_start:] + [dep]
                    except ValueError:
                        cycle = [dep]
                    out.append(Violation(
                        "V3", locations.get(dep, dep),
                        "决策图存在环，无法确定 patch 后的重执行起点: "
                        + " -> ".join(cycle),
                    ))
                    continue
                if color[dep] == WHITE:
                    color[dep] = GRAY
                    stack.append((dep, iter(decisions[dep])))
                    path_stack.append(dep)
                    advanced = True
                    break
            if not advanced:
                color[node] = BLACK
                stack.pop()
                if path_stack:
                    path_stack.pop()

    return out


# ---------------------------------------------------------------------------
# V4 — 数字必须有出处
# ---------------------------------------------------------------------------

def check_provenance(doc: Dict[str, Any]) -> List[Violation]:
    """论文 Appendix C：每个 milestone 数字携带出处标记，连同它的推导。

    检查两处带数值的结构：资源拓扑契约、门禁行的锚点。
    """
    out: List[Violation] = []

    contract = doc.get("resource_topology_contract") or {}
    if isinstance(contract, dict):
        for key, entry in contract.items():
            if not isinstance(entry, dict):
                out.append(Violation(
                    "V4", f"resource_topology_contract.{key}",
                    "契约条目必须是含 value 与 provenance 的对象",
                ))
                continue
            ok, reason = _provenance_ok(entry.get("provenance"))
            if not ok:
                out.append(Violation(
                    "V4", f"resource_topology_contract.{key}.provenance", reason,
                ))

    for mid, category, line in _iter_gate_lines(doc):
        lid = line.get("id", "<无 id>")
        anchor = line.get("anchor")
        # 只有 performance 行强制要出处；correctness/mechanism 行若带锚点也一并检查
        if not isinstance(anchor, dict):
            if category == "performance_lines":
                out.append(Violation(
                    "V4", f"{mid}.{category}.{lid}.anchor",
                    "性能行必须带锚点对象（含 kind/expr/provenance）",
                ))
            continue
        ok, reason = _provenance_ok(anchor.get("provenance"))
        if not ok:
            out.append(Violation(
                "V4", f"{mid}.{category}.{lid}.anchor.provenance", reason,
            ))

    return out


# ---------------------------------------------------------------------------
# V5 — 门禁三类行齐全
# ---------------------------------------------------------------------------

def check_gate_completeness(doc: Dict[str, Any]) -> List[Violation]:
    """论文 §3.2：每个 milestone 关闭时的门禁含正确性行、带出处的性能行，
    以及一条机制验证行 —— 那条 profiler 锚定的谓词，用来区分「机制真的落地了」
    和「只是碰巧变快了」。

    三类缺一，门禁就失去它要起的作用。
    """
    out: List[Violation] = []
    for milestone in doc.get("milestones", []) or []:
        if not isinstance(milestone, dict):
            continue
        mid = milestone.get("id", "<无 id>")
        gate = milestone.get("gate")
        if not isinstance(gate, dict):
            out.append(Violation(
                "V5", f"{mid}.gate", "缺少门禁定义",
            ))
            continue
        for category in LINE_CATEGORIES:
            lines = gate.get(category)
            if not isinstance(lines, list) or not lines:
                out.append(Violation(
                    "V5", f"{mid}.gate.{category}",
                    f"门禁缺少 '{category}'；三类行缺一则无法区分真机制与巧合加速",
                ))
    return out


# ---------------------------------------------------------------------------
# V6 — 未决行必须销账
# ---------------------------------------------------------------------------

def check_line_discharge(doc: Dict[str, Any]) -> List[Violation]:
    """论文 Appendix B：未解决的门禁行必须销账成三种判定之一，各有举证责任。

      DEFERRED    需要三件证据：参考实现确实达到了该指标、当前手段改不动它、
                  且有一个具名的下游手段来接这个未解的测试
      INFEASIBLE  需要逐指标论证 + 证据表明目标被排除
      N/A         只适用于结构上不适用的情形

    销账而不举证，等于把问题藏起来。
    """
    out: List[Violation] = []
    for mid, category, line in _iter_gate_lines(doc):
        lid = line.get("id", "<无 id>")
        base = f"{mid}.{category}.{lid}"
        status = line.get("status")

        if _is_blank(status):
            out.append(Violation("V6", f"{base}.status", "门禁行缺少 status"))
            continue
        if status not in ALL_STATUSES:
            out.append(Violation(
                "V6", f"{base}.status",
                f"非法 status '{status}'；合法取值: {sorted(ALL_STATUSES)}",
            ))
            continue
        if status == "OPEN":
            out.append(Violation(
                "V6", base,
                "门禁行仍为 OPEN：关闭 milestone 前必须销账为 "
                "DEFERRED / INFEASIBLE / N/A 并附证据",
            ))
            continue
        if status not in DISCHARGE_STATUSES:
            continue

        # 以下是三种销账各自的举证责任
        if _is_blank(line.get("evidence")):
            out.append(Violation(
                "V6", f"{base}.evidence",
                f"销账为 {status} 必须附证据",
            ))

        if status == "DEFERRED":
            if _is_blank(line.get("downstream_lever")):
                out.append(Violation(
                    "V6", f"{base}.downstream_lever",
                    "DEFERRED 必须指名接手这个未解测试的下游手段，"
                    "否则该测试就是被丢弃而非推迟",
                ))
            if _is_blank(line.get("reference_achieves")):
                out.append(Violation(
                    "V6", f"{base}.reference_achieves",
                    "DEFERRED 必须举证参考实现确实达到了该指标",
                ))
            if _is_blank(line.get("current_levers_cannot")):
                out.append(Violation(
                    "V6", f"{base}.current_levers_cannot",
                    "DEFERRED 必须举证当前手段改不动它",
                ))
        elif status == "INFEASIBLE":
            argument = line.get("per_metric_argument")
            if not isinstance(argument, list) or not argument:
                out.append(Violation(
                    "V6", f"{base}.per_metric_argument",
                    "INFEASIBLE 必须给出逐指标论证（非空列表）",
                ))
        elif status == "N/A":
            if _is_blank(line.get("structural_reason")):
                out.append(Violation(
                    "V6", f"{base}.structural_reason",
                    "N/A 只适用于结构上不适用的情形，必须说明结构原因",
                ))
    return out


# ---------------------------------------------------------------------------
# V7 — 门禁锚点来源受限
# ---------------------------------------------------------------------------

def check_anchor_sources(doc: Dict[str, Any]) -> List[Violation]:
    """论文 Appendix C：门禁锚点只能由参考实测、硬件规格，或从这两者派生的
    表达式构成。

    这条把「目标是怎么来的」锁死在可核验的来源上 —— agent 不能自己定一个
    好听的目标再去达成它。
    """
    out: List[Violation] = []
    for mid, category, line in _iter_gate_lines(doc):
        lid = line.get("id", "<无 id>")
        anchor = line.get("anchor")
        if not isinstance(anchor, dict):
            continue  # 缺失由 V4 负责
        base = f"{mid}.{category}.{lid}.anchor"
        kind = anchor.get("kind")
        if _is_blank(kind):
            out.append(Violation("V7", f"{base}.kind", "锚点缺少 kind"))
            continue
        if kind not in ALLOWED_ANCHOR_KINDS:
            out.append(Violation(
                "V7", f"{base}.kind",
                f"非法锚点来源 '{kind}'；只允许 {sorted(ALLOWED_ANCHOR_KINDS)}",
            ))
            continue
        if _is_blank(anchor.get("expr")):
            out.append(Violation(
                "V7", f"{base}.expr", "锚点必须给出可核验的表达式",
            ))
        if kind == "derived" and _is_blank(anchor.get("derived_from")):
            out.append(Violation(
                "V7", f"{base}.derived_from",
                "derived 锚点必须指明派生自哪些参考实测或硬件规格",
            ))
    return out


# ---------------------------------------------------------------------------
# V8 — M0 不设绝对性能阈值
# ---------------------------------------------------------------------------

def check_m0_no_absolute_threshold(doc: Dict[str, Any]) -> List[Violation]:
    """论文 Appendix C：M0 门禁检查正确性、架构谓词和机制行，不含绝对性能阈值；
    该阶段用相对参考的比率提供跨 job 比较。

    道理是：M0 装的是架构决策，架构刚落地时还没调过，拿绝对时间卡它会把
    正确的架构当成失败砍掉。
    """
    out: List[Violation] = []
    for milestone in doc.get("milestones", []) or []:
        if not isinstance(milestone, dict):
            continue
        if milestone.get("id") != "M0":
            continue
        gate = milestone.get("gate") or {}
        if not isinstance(gate, dict):
            continue
        for line in gate.get("performance_lines", []) or []:
            if not isinstance(line, dict):
                continue
            lid = line.get("id", "<无 id>")
            anchor = line.get("anchor") or {}
            expr = anchor.get("expr") if isinstance(anchor, dict) else None
            haystack = " ".join(
                str(x) for x in (line.get("statement"), expr) if x
            )
            match = RE_ABSOLUTE_TIME.search(haystack)
            if match:
                out.append(Violation(
                    "V8", f"M0.performance_lines.{lid}",
                    f"M0 性能行出现绝对时间阈值 '{match.group(0)}'；"
                    "该阶段只能用相对参考的比率，架构尚未调优，"
                    "绝对阈值会把正确架构误判为失败",
                ))
    return out


# ---------------------------------------------------------------------------
# V9 — 原子组同批落地
# ---------------------------------------------------------------------------

def check_atomic_groups(doc: Dict[str, Any]) -> List[Violation]:
    """论文 Appendix C：原子组标注标识必须一次提交落地的决策集，
    因为它的任何真子集都会导致退化。

    组成员若散落在不同 milestone，就不可能同批落地。
    """
    out: List[Violation] = []
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for mid, _m, decision in _iter_decisions(doc):
        group = decision.get("atomic_group")
        if _is_blank(group):
            continue
        groups.setdefault(str(group), []).append((mid, decision.get("id", "<无 id>")))

    for group, members in groups.items():
        milestones = {mid for mid, _ in members}
        if len(milestones) > 1:
            detail = ", ".join(f"{mid}.{did}" for mid, did in members)
            out.append(Violation(
                "V9", f"atomic_group.{group}",
                f"原子组成员跨 {len(milestones)} 个 milestone（{detail}），"
                "无法同批落地；任何真子集单独落地都会退化",
            ))
        if len(members) < 2:
            out.append(Violation(
                "V9", f"atomic_group.{group}",
                "原子组只有一个成员，该标注无意义；"
                "原子组用于标识必须一起落地的决策集",
            ))
    return out


# ---------------------------------------------------------------------------
# V10 — 架构决策须命名验证通道
# ---------------------------------------------------------------------------

def check_verification_channel(doc: Dict[str, Any]) -> List[Violation]:
    """论文 Appendix C：每个架构决策命名验证其执行的机器可读通道。

    「我实现了持久化循环」不算数，要能指出哪个字段会因此改变。战术决策
    不强制，因为它们通常由上层架构决策的通道一并覆盖。
    """
    out: List[Violation] = []
    for mid, _m, decision in _iter_decisions(doc):
        if decision.get("layer") != ARCHITECTURAL:
            continue
        did = decision.get("id", "<无 id>")
        channel = decision.get("verification_channel")
        if _is_blank(channel):
            out.append(Violation(
                "V10", f"{mid}.{did}.verification_channel",
                "架构决策必须命名验证其执行的机器可读通道，"
                "否则无法区分「真的落地了」和「只是声称落地了」",
            ))
    return out


# ---------------------------------------------------------------------------
# V11 — 机制行须有闭合词汇表标签与停顿锚点
# ---------------------------------------------------------------------------

def check_mechanism_labels(doc: Dict[str, Any]) -> List[Violation]:
    """论文 Appendix B：这类机制使用闭合词汇表标签，每个标签要求停顿锚定的
    验证与记录在案的顺序证据。

    不加这条约束时，机制行会退化成「变快了」的换一种说法 —— 而区分
    「我设计的机制真的发生了」与「碰巧变快了」正是机制行唯一的用途。

    锚点必须是停顿类字段。占比下降有多种成因（包括别的段变慢了），
    停顿下降才直接说明「不再有人在等」。
    """
    out: List[Violation] = []
    for mid, category, line in _iter_gate_lines(doc):
        if category != "mechanism_lines":
            continue
        lid = line.get("id", "<无 id>")
        base = f"{mid}.mechanism_lines.{lid}"

        label = line.get("label")
        if _is_blank(label):
            out.append(Violation(
                "V11", f"{base}.label",
                "机制行缺少 label。没有闭合词汇表标签，这条行无法被程序核验，"
                f"等于只是换个说法说「变快了」；可选标签: {sorted(MECHANISM_LABELS)}",
            ))
            continue

        label = str(label).strip()
        required = MECHANISM_LABELS.get(label)
        if required is None:
            out.append(Violation(
                "V11", f"{base}.label",
                f"'{label}' 不在闭合词汇表内。自造标签等于没有标签；"
                f"可选: {sorted(MECHANISM_LABELS)}。"
                "确需新增请先按 references/mechanism-labels.md 的条件评估",
            ))
            continue

        anchor = line.get("anchor")
        expr = anchor.get("expr") if isinstance(anchor, dict) else None
        haystack = " ".join(str(x) for x in (expr, line.get("statement")) if x)
        if not any(token in haystack for token in required):
            out.append(Violation(
                "V11", f"{base}.anchor.expr",
                f"标签 '{label}' 要求锚定停顿指标 {list(required)}，"
                "但锚点表达式里没有出现它们中的任何一个。"
                "占比类指标不能替代：占比下降可能只是别的段变慢了，"
                "停顿下降才说明机制真的生效",
            ))

        if _is_blank(line.get("ordering_evidence")):
            out.append(Violation(
                "V11", f"{base}.ordering_evidence",
                "缺少顺序证据。单独一个数说明不了变化方向，"
                "需记录改动前后两次读数及它们来自同配置的事实",
            ))

    return out


# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------

RULES = {
    "V1": ("五元组完整", check_five_tuple),
    "V2": ("决策必须有触发器", check_trigger),
    "V3": ("派生图无环", check_decision_graph),
    "V4": ("数字必须有出处", check_provenance),
    "V5": ("门禁三类行齐全", check_gate_completeness),
    "V6": ("未决行必须销账", check_line_discharge),
    "V7": ("门禁锚点来源受限", check_anchor_sources),
    "V8": ("M0 不设绝对性能阈值", check_m0_no_absolute_threshold),
    "V9": ("原子组同批落地", check_atomic_groups),
    "V10": ("架构决策须命名验证通道", check_verification_channel),
    "V11": ("机制行须有标签与停顿锚点", check_mechanism_labels),
}


def validate(doc: Dict[str, Any],
             rules: Optional[Iterable[str]] = None) -> List[Violation]:
    """跑校验，返回全部违规。rules 为 None 时跑全部规则。"""
    selected = list(rules) if rules else list(RULES)
    out: List[Violation] = []
    for rule in selected:
        entry = RULES.get(rule)
        if entry is None:
            out.append(Violation(
                "ARG", "--rules", f"未知规则 '{rule}'；可选: {sorted(RULES)}",
            ))
            continue
        out.extend(entry[1](doc))
    return out


def _load(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"错误: 无法读取 {path}: {exc}") from exc
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"错误: {path} 不是合法 JSON，第 {exc.lineno} 行第 {exc.colno} 列: {exc.msg}"
        ) from exc
    if not isinstance(doc, dict):
        raise SystemExit(f"错误: {path} 顶层必须是对象，实际是 {type(doc).__name__}")
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="校验 ForgeKernel milestone.json 是否满足论文规定的门禁义务",
    )
    parser.add_argument("milestone", type=Path, nargs="?",
                        help="milestone.json 路径")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="以 JSON 输出结果，供脚本消费")
    parser.add_argument("--rules", type=str, default="",
                        help="只跑指定规则，逗号分隔，如 V2,V6")
    parser.add_argument("--list-rules", action="store_true",
                        help="列出全部规则后退出")
    args = parser.parse_args()

    if args.list_rules:
        for rule, (title, _fn) in RULES.items():
            print(f"{rule:<4} {title}")
        return 0

    if args.milestone is None:
        parser.error("缺少 milestone.json 路径（或用 --list-rules 查看规则）")

    try:
        doc = _load(args.milestone)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    rules = [r.strip() for r in args.rules.split(",") if r.strip()] or None
    violations = validate(doc, rules)

    if args.as_json:
        print(json.dumps({
            "passed": not violations,
            "violation_count": len(violations),
            "violations": [v.as_dict() for v in violations],
        }, ensure_ascii=False, indent=2))
        return 1 if violations else 0

    if not violations:
        count = sum(1 for _ in _iter_decisions(doc))
        print(f"通过：{args.milestone} 满足全部门禁义务（共 {count} 条决策）")
        return 0

    # 按规则分组输出，便于定位
    by_rule: Dict[str, List[Violation]] = {}
    for v in violations:
        by_rule.setdefault(v.rule, []).append(v)

    print(f"未通过：发现 {len(violations)} 条违规\n", file=sys.stderr)
    for rule in sorted(by_rule, key=lambda r: (len(r), r)):
        title = RULES.get(rule, ("参数错误", None))[0]
        print(f"— {rule} {title} —", file=sys.stderr)
        for v in by_rule[rule]:
            print(f"  {v.path}: {v.message}", file=sys.stderr)
        print(file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
