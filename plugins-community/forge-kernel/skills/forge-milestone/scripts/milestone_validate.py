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
#: 算「已销账」的状态。
#:
#: 刻意**不含** FAIL。FAIL 看起来像一个终态，实际零举证义务：早先
#: RESOLVED_STATUSES 含它，而销账检查对已销账状态直接跳过 —— 于是把全部
#: 门禁行改成 FAIL，校验通过、`gate_impasse` 也不报，一个 milestone 可以
#: 带着满页 FAIL 合法关闭。
#:
#: 没中就必须 DEFERRED（交给具名的下游手段，三条举证）或 INFEASIBLE
#: （宣告做不到，也要举证）。两条都有举证责任，FAIL 没有。
RESOLVED_STATUSES = {"PASS"}
DISCHARGE_STATUSES = {"DEFERRED", "INFEASIBLE", "N/A"}
ALL_STATUSES = RESOLVED_STATUSES | DISCHARGE_STATUSES | {"OPEN"}

#: V7：门禁锚点只允许这三类来源。
ALLOWED_ANCHOR_KINDS = {"reference_measurement", "hardware_spec", "derived"}

ARCHITECTURAL = "architectural"
TACTICAL = "tactical"

#: V4：出处标记的两种合法形态。
#: 机制锚点要求的形状：一个「通道 比较符 数值」。与 campaign.py 的
#: RE_MECH_EXPR 对应 —— 核验端只认这一种形状，所以这一端就要把别的形状拦掉。
RE_MECH_COMPARISON = re.compile(
    r"[\w:.()%/\-]+?\s*(?:<=|>=|==|!=|<|>)\s*-?\d+(?:\.\d+)?")
RE_BOOL_CONNECTIVE = re.compile(r"\b(?:and|or)\b|&&|\|\|", re.IGNORECASE)

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

def _is_m0(milestone_id: Any) -> bool:
    return str(milestone_id).strip().upper().startswith("M0")


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
            if category == "performance_lines" and _is_m0(mid):
                # M0 不要求性能行，而且不该有。
                #
                # 第一版能跑到哪里事先无法知道：设高了不可达、设低了没意义。
                # 架构是否真的生效由**机制行**（锚到参考的比率）判断。
                # 强制要求的唯一产物是训练架构层发明一条敷衍的性能行 ——
                # 要么写成几乎必然不达标的数（只能标 DEFERRED 走三条举证），
                # 要么写成毫无约束力的数再配一个合法出处。
                if isinstance(lines, list) and lines:
                    out.append(Violation(
                        "V5", f"{mid}.gate.{category}",
                        "M0 不应有性能行：第一版能跑到哪里事先无法知道，"
                        "设高了不可达、设低了没意义。架构是否生效由机制行"
                        "（锚到参考的比率）判断",
                    ))
                continue
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
    milestones = [m for m in (doc.get("milestones") or [])
                  if isinstance(m, dict)]
    last_milestone_id = milestones[-1].get("id") if milestones else None
    milestone_order = {}
    all_decision_ids = set()
    for index, milestone in enumerate(milestones):
        milestone_order[milestone.get("id")] = index
        for decision in milestone.get("decisions") or []:
            if isinstance(decision, dict) and decision.get("id"):
                all_decision_ids.add(decision["id"])
                milestone_order[decision["id"]] = index

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
        if status == "PASS":
            # 干净的 PASS 也要说清是哪次测量把它关掉的。
            #
            # anchor.provenance 说的是**阈值**的来源，不是「关住这条行的那次
            # 测量」。两者不是一回事：阈值可以来自参考实测，而这条行有没有
            # 达标是本次 job 的事。不要求的话，PASS 是一个零举证的自述状态，
            # 把所有行改成 PASS 就能让门禁僵持这个重入口消失。
            if _is_blank(line.get("closed_by")):
                out.append(Violation(
                    "V6", f"{base}.closed_by",
                    "PASS 必须写 closed_by，说明是哪次测量把它关掉的"
                    "（如 \"[MEASURED, job 7]\"）。"
                    "anchor.provenance 说的是阈值的来源，不是这次达标的证据",
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
            if category == "correctness_lines":
                out.append(Violation(
                    "V6", base,
                    "正确性行不得推迟。算错的 kernel 再快也无意义，"
                    "这类测试没有「交给下游手段」这个出口",
                ))
            if mid == last_milestone_id:
                out.append(Violation(
                    "V6", base,
                    f"{mid} 是最后一个 milestone，它的门禁行不能是 DEFERRED —— "
                    "没有更靠后的手段可以接手了。"
                    "要么在本 milestone 内关掉，要么宣告 INFEASIBLE 并举证",
                ))
            lever = line.get("downstream_lever")
            if lever and lever not in all_decision_ids:
                out.append(Violation(
                    "V6", f"{base}.downstream_lever",
                    f"接手的下游手段 {lever!r} 在课程表里不存在 —— "
                    "这条测试实际是被丢弃了，而不是推迟了",
                ))
            elif lever and milestone_order.get(lever, -1) <= \
                    milestone_order.get(mid, 0):
                out.append(Violation(
                    "V6", f"{base}.downstream_lever",
                    f"接手的 {lever!r} 不在更靠后的 milestone 里。"
                    "手里已经握着手段还推迟，说明这个 milestone 没做完",
                ))
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
        # 扫 M0 的**全部**门禁行，不只性能行。
        #
        # V5 已经规定 M0 不该有性能行，所以只扫那一类等于这条规则作废 ——
        # 而绝对时间阈值一样可以写进机制行或正确性行（「mte2 等待必须低于
        # 20us」），危害完全相同：架构刚落地没调过，拿绝对时间卡它会把正确
        # 的架构当失败砍掉。
        for category in LINE_CATEGORIES:
            for line in gate.get(category, []) or []:
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
                        "V8", f"M0.{category}.{lid}",
                        f"M0 门禁行出现绝对时间阈值 '{match.group(0)}'；"
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

        # 复合条件在这里就拦住，而不是留到 record 时才发现核验不了。
        #
        # 核验端（campaign.py 的 verify_mechanism_line）只认单个
        # 「通道 比较符 数值」。写成 `a < 4.2 and b < 0.3` 的话，这条机制行
        # 会在 record 时判为「无法核验」—— 到那时课程表已经写完、代码也写完了，
        # 而失败原因看起来像是实现问题。两端的判据要一致，门禁才有意义。
        if isinstance(expr, str) and expr.strip():
            matches = RE_MECH_COMPARISON.findall(expr)
            if len(matches) > 1 or RE_BOOL_CONNECTIVE.search(expr):
                out.append(Violation(
                    "V11", f"{base}.anchor.expr",
                    f"锚点表达式含多个条件：{expr!r}。"
                    "机制核验只认单个「通道 比较符 数值」—— 复合条件会被判为"
                    "无法核验，这条机制行拿不到顺序证据。"
                    "请拆成多条机制行，每条锚定一个停顿字段",
                ))
                continue

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

def check_document_shape(doc: Dict[str, Any]) -> List[Violation]:
    """课程表得先是一份课程表。

    为什么单独立一条规则：其余十一条规则都在「遍历决策/门禁行」的循环里做
    检查，而空文档、键名拼错、门禁行写成一句话这三种情况下，那些循环一次
    都不会执行 —— 于是零违规，输出「通过：满足全部门禁义务（共 0 条决策）」。

    「没东西可查」被当成了「查过了，没问题」。这是整条门禁链最脆的一环：
    `forge-coder` 的规则是「validate 不通过就不要开始实现」，它拿到 rc=0
    就会在一份空课程表上开工，而后续每一条机制行都无从核验。

    所以这里只问三个问题：有没有 milestones、里面有没有决策、门禁行是不是
    结构化对象。
    """
    out: List[Violation] = []

    milestones = doc.get("milestones")
    if milestones is None:
        known = sorted(k for k in doc if isinstance(k, str))
        out.append(Violation(
            "V0", "milestones",
            "文档里没有 milestones 字段，没有任何内容可校验 —— "
            "「零违规」在这里不等于通过。"
            + (f"顶层实际有的键: {known}（是不是拼错了？）" if known
               else "文档是空的。")))
        return out

    if not isinstance(milestones, list):
        out.append(Violation(
            "V0", "milestones",
            f"milestones 必须是数组，实际是 {type(milestones).__name__}"))
        return out

    if not milestones:
        out.append(Violation(
            "V0", "milestones",
            "milestones 是空数组。一份没有里程碑的课程表无法指导实现，"
            "也没有任何门禁义务可查"))
        return out

    total_decisions = 0
    for idx, milestone in enumerate(milestones):
        if not isinstance(milestone, dict):
            out.append(Violation(
                "V0", f"milestones[{idx}]",
                f"里程碑必须是对象，实际是 {type(milestone).__name__}；"
                "非对象条目会被其余规则整条跳过，等于不受任何约束"))
            continue
        mid = milestone.get("id", f"<milestones[{idx}] 无 id>")

        decisions = milestone.get("decisions")
        if isinstance(decisions, list):
            for d_idx, decision in enumerate(decisions):
                if isinstance(decision, dict):
                    total_decisions += 1
                else:
                    out.append(Violation(
                        "V0", f"{mid}.decisions[{d_idx}]",
                        f"决策必须是对象，实际是 {type(decision).__name__}；"
                        "非对象条目会被五元组、触发器、依赖图等规则整条跳过"))

        gate = milestone.get("gate")
        if gate is not None and not isinstance(gate, dict):
            out.append(Violation(
                "V0", f"{mid}.gate",
                f"gate 必须是对象，实际是 {type(gate).__name__}；"
                "非对象的 gate 会让门禁相关的全部规则失效"))
            continue
        if isinstance(gate, dict):
            for category in LINE_CATEGORIES:
                lines = gate.get(category)
                if lines is None:
                    continue
                if not isinstance(lines, list):
                    out.append(Violation(
                        "V0", f"{mid}.gate.{category}",
                        f"{category} 必须是数组，实际是 "
                        f"{type(lines).__name__}"))
                    continue
                for l_idx, line in enumerate(lines):
                    if not isinstance(line, dict):
                        out.append(Violation(
                            "V0", f"{mid}.gate.{category}[{l_idx}]",
                            f"门禁行必须是对象（含 id/statement/anchor），"
                            f"实际是 {type(line).__name__}: {line!r}。"
                            "写成一句话的门禁行会被全部规则跳过 —— "
                            "它读起来像一条义务，实际不受任何核验"))

    if total_decisions == 0:
        out.append(Violation(
            "V0", "milestones[].decisions",
            "整份课程表里没有一条决策。没有决策就没有可实现的东西，"
            "也没有可核验的机制"))
    return out



# ---------------------------------------------------------------------------
# V12 — 课程表不得含实现体
# ---------------------------------------------------------------------------

#: 实现体的痕迹。课程表写的是**决策与契约**，不是代码。
#:
#: 论文把三个产物的边界当成「保证参照的代码不会进入 kernel」的构件：
#: milestone（决策、参数与契约，**从不含实现体**）/ 硬件知识库（唯一含代码的
#: 产物）/ meta-milestone（蒸馏策略）。
#:
#: 为什么这条是承重墙：架构 agent 读过参考的硬件行为（在上游的 source 模式下
#: 甚至读过源码），然后写课程表；而编码 agent 读课程表。所以 milestone 是
#: 参考实现**唯一可能的泄漏通道** —— 它一旦能装代码，前面那道 no-code 边界
#: 就有了一条绕过去的路（逃生舱一开、或封存漏挑了文件，它就是唯一的那道墙）。
RE_IMPLEMENTATION_BODY = re.compile(
    r"```"                                   # markdown 代码块
    r"|__aicore__|__global__|__gm__|__ubuf__"  # AscendC/CUDA 的存储与函数限定
    r"|\bAscendC\s*::"                       # 命名空间直接调用
    r"|\bfor\s*\([^)]*;[^)]*;"                # 三段式 for
    r"|\bwhile\s*\([^)]{0,80}\)\s*\{"        # while (...) {
    r"|\bif\s*\([^)]{0,80}\)\s*\{"           # if (...) {
    r"|\b(?:class|struct)\s+[A-Z]\w*\s*(?::\s*[^{]{0,60})?\{"
)

#: 允许出现代码形状的字段。契约本来就要写函数签名与表达式。
IMPLEMENTATION_EXEMPT_KEYS = {
    "expr", "verification_channel", "channel", "structural_anchor",
    "derived_from", "signature", "provenance", "ref", "closed_by",
}


def _walk_strings(node: Any, path: str = "",
                  exempt: Optional[set] = None):
    """遍历文档里所有字符串，产出 (路径, 文本)。"""
    exempt = exempt or set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key in exempt:
                continue
            yield from _walk_strings(value, f"{path}.{key}" if path else str(key),
                                     exempt)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk_strings(value, f"{path}[{index}]", exempt)
    elif isinstance(node, str):
        yield path, node


def check_no_implementation_body(doc: Dict[str, Any]) -> List[Violation]:
    """课程表里不得出现实现体。

    课程表写的是「做什么、为什么、什么观测触发它、代价是什么、依赖什么」。
    一旦它开始装代码，它就成了参考实现绕过 no-code 边界的通道 —— 而那条边界
    是整套方法的地基（架构结论必须是从硬件行为反推的）。
    """
    out: List[Violation] = []
    for path, text in _walk_strings(doc, exempt=IMPLEMENTATION_EXEMPT_KEYS):
        match = RE_IMPLEMENTATION_BODY.search(text)
        if match:
            snippet = match.group(0)[:40].replace("\n", " ")
            out.append(Violation(
                "V12", path,
                f"出现实现体的痕迹 {snippet!r}。课程表写决策与契约，不写代码 —— "
                "它是参考实现唯一可能的泄漏通道，一旦能装代码，"
                "no-code 边界就有了一条绕过去的路。"
                "代码放实现侧；可复用的机制放知识库卡片的「最小可运行例」",
            ))
    return out


# ---------------------------------------------------------------------------
# V13 — 四条轴必须表态
# ---------------------------------------------------------------------------

#: 必须表态的结构轴，各自一张闭合词汇表。
#:
#: 上游有四条正交轴（同步 / 取数 / 相位 / 操作数源），每条**必须表态**，
#: 哪怕表态为 `none`。差别在于：我们的 12 个机制标签是「写了才查」，
#: 而这四条是「**不写就违规**」。
#:
#: 为什么要这一层：架构层只要**不写**某条机制行，那个维度就完全不受约束 ——
#: V5 只要求每个 milestone 有至少一条机制行。上游记录的病例正是这个形状：
#: 一个把流式交接裁定成全员集合的 kernel 以三联症通过 M0 并轮轮硬调，
#: 而「对比同步签名」本应在第一个绿轮就拉响警报。
ADJUDICATION_AXES: Dict[str, Dict[str, Any]] = {
    "cv_handoff": {
        "values": {"flag_pair", "queue", "none"},
        "label": "Cube 与 Vector 的交接方式",
        "requires_label": "CV_HANDOFF",
    },
    "supply": {
        "values": {"double_buffer", "serial", "none"},
        "label": "搬运供给的组织方式",
        "requires_label": "SUPPLY_KEEPUP",
    },
    "consumer_phase": {
        "values": {"ping_pong", "convoy", "none"},
        "label": "消费侧的分阶段方式",
        "requires_label": "CONSUMER_PHASING",
    },
    "operand_source": {
        "values": {"l1_direct", "ub_roundtrip", "none"},
        "label": "Cube 操作数的来源",
        "requires_label": "L1_RESIDENCY",
    },
}


def check_adjudications(doc: Dict[str, Any]) -> List[Violation]:
    """四条结构轴必须表态，非 none 的轴要有对应机制行锚着。"""
    out: List[Violation] = []
    adjudications = doc.get("adjudications")
    if not isinstance(adjudications, dict):
        out.append(Violation(
            "V13", "adjudications",
            "缺少 adjudications。四条结构轴必须逐条表态"
            f"（{sorted(ADJUDICATION_AXES)}），**哪怕表态为 none** —— "
            "不写就意味着那个维度完全不受约束，而 V5 只要求有至少一条机制行",
        ))
        return out

    labels_used = {
        str(line.get("label", "")).strip().upper()
        for _mid, category, line in _iter_gate_lines(doc)
        if category == "mechanism_lines"
    }

    for axis, spec in ADJUDICATION_AXES.items():
        if axis not in adjudications:
            out.append(Violation(
                "V13", f"adjudications.{axis}",
                f"没有对 {spec['label']} 表态。"
                f"可选: {sorted(spec['values'])}（确实不涉及就写 none）",
            ))
            continue
        value = str(adjudications[axis]).strip()
        if value not in spec["values"]:
            # 刻意和「完全没裁定」分开报：词汇表外的值说明作者表过态但用词
            # 不在表里，该做的是对齐用词，而不是补一次裁定。
            out.append(Violation(
                "V13", f"adjudications.{axis}",
                f"未知取值 {value!r}（{spec['label']}）。"
                f"闭合词汇表: {sorted(spec['values'])}",
            ))
            continue
        if value != "none" and spec["requires_label"] not in labels_used:
            out.append(Violation(
                "V13", f"adjudications.{axis}",
                f"裁定为 {value!r}（非 none），"
                f"但没有任何 {spec['requires_label']} 机制行锚着它。"
                "表了态就要有一条可核验的行 —— 否则这个裁定只是一句声明",
            ))
    return out


# ---------------------------------------------------------------------------
# V14 — 结构类门禁行必须先定价
# ---------------------------------------------------------------------------

#: 带结构断言、但没说「没有它慢多少」的门禁行。
#:
#: 上游的规矩：SASS 形状说的是参考长什么样；**只有探针测出这个形状值多少钱，
#: 它才成为门禁行**，否则它属于决策正文里的描述。实证：某版 M0 门禁带了
#: 35 条未定价的形状行，把「长得像参考」变成学生必须用探针证明的契约 ——
#: 八轮、两小时、零 kernel 工作；而唯一被定过价的那条是合法门禁行，
#: 那条从未定价的寄存器切分**实际上比替代方案更慢**。
def check_priced_structural_lines(doc: Dict[str, Any]) -> List[Violation]:
    out: List[Violation] = []
    for mid, category, line in _iter_gate_lines(doc):
        if category != "mechanism_lines":
            continue
        anchor = line.get("structural_anchor")
        if not anchor:
            continue
        lid = line.get("id", "<无 id>")
        base = f"{mid}.mechanism_lines.{lid}"
        priced = line.get("priced")
        if not isinstance(priced, dict):
            out.append(Violation(
                "V14", f"{base}.priced",
                "带 structural_anchor 的门禁行必须先定价："
                "`priced: {delta_pct, source}`。"
                "结构形状说的是参考长什么样 —— 只有测出「没有它慢多少」，"
                "它才成为验收线，否则它属于决策正文里的描述",
            ))
            continue
        delta = priced.get("delta_pct")
        if not isinstance(delta, (int, float)) or delta <= 0:
            out.append(Violation(
                "V14", f"{base}.priced.delta_pct",
                f"价格 {delta!r} 不是一个正数百分比。"
                "没有它慢多少？这个数要来自消融或定价探针",
            ))
        if _is_blank(priced.get("source")):
            out.append(Violation(
                "V14", f"{base}.priced.source",
                "价格必须说明出处（`job N` / `lever:<名字>` / `KB:<卡片 id>`）—— "
                "一个没有出处的价格和没有价格一样",
            ))
    return out


# ---------------------------------------------------------------------------
# V15 — 不得把参考的分配结果钉成验收线
# ---------------------------------------------------------------------------

#: 「指标 == 具体数值」这种形状。
#:
#: 上游机械拒绝把 `regs_per_thread == N` 这类**在参考上观测到的分配结果**
#: 写成门禁行：那些是观测结果，不是要求。门禁只写上界与「溢出为零」，
#: 切分数字留给实现自己选。
#:
#: 实证：某版钉死了一组切分数字，学生照做，它的 producer 装不下同一个
#: milestone 要求的索引预取，首次跑绿就大量溢出；同算子的另一份课程表写明
#: 「切分数字是编译器的分配结果、不是验收线」，学生选了别的切分、零溢出。
RE_PINNED_ALLOCATION = re.compile(
    r"(?P<field>[\w:.()%/\-]*(?:block_dim|regs?_per_thread|tile_num|"
    r"block_num|core_num|buffer_num)[\w:.()%/\-]*)"
    r"\s*==\s*(?P<value>\d+)",
    re.IGNORECASE)


def check_no_pinned_allocation(doc: Dict[str, Any]) -> List[Violation]:
    out: List[Violation] = []
    for mid, category, line in _iter_gate_lines(doc):
        lid = line.get("id", "<无 id>")
        anchor = line.get("anchor") or {}
        expr = anchor.get("expr") if isinstance(anchor, dict) else None
        if not isinstance(expr, str):
            continue
        match = RE_PINNED_ALLOCATION.search(expr)
        if match:
            field = match.group("field")
            value = match.group("value")
            out.append(Violation(
                "V15", f"{mid}.{category}.{lid}.anchor.expr",
                f"把 {field} == {value} 写成了验收线。"
                "这是在参考上**观测到的分配结果**，不是要求 —— "
                f"钉死它会让实现被绑在一组具体数字上，"
                f"而那组数字在候选的资源预算下可能装不下。\n"
                f"    改成上界 + 配对的溢出/冲突行，例如 "
                f"`{field} <= {value}` 且「无溢出」，具体取值留给实现选",
            ))
    return out


# ---------------------------------------------------------------------------
# V16 — 与上一版做差：阈值不得悄悄放宽
# ---------------------------------------------------------------------------

def _gate_line_index(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for mid, category, line in _iter_gate_lines(doc):
        key = f"{mid}.{category}.{line.get('id', '<无 id>')}"
        out[key] = line
    return out


def _threshold_of(line: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    anchor = line.get("anchor") or {}
    expr = anchor.get("expr") if isinstance(anchor, dict) else None
    if not isinstance(expr, str):
        return None
    match = re.search(r"(?P<op><=|>=|<|>)\s*(?P<value>-?\d+(?:\.\d+)?)", expr)
    if not match:
        return None
    return match.group("op"), float(match.group("value"))


def check_against_previous(doc: Dict[str, Any],
                           previous: Optional[Dict[str, Any]]
                           ) -> List[Violation]:
    """补丁模式：和上一版做差，拦住「悄悄挪球门」。

    不做差的话，两种改动完全无痕：
      放宽阈值   把 `< 4.2` 改成 `< 6.0`，于是实现带着两倍于参考的等待量过关
      删掉机制行 那条机制从此不被核验，而 record 会判「无法核验」后放过

    上游的双模校验还把「只改数值的重调」与「换了 VERIFY 类别的新主张」分开
    （原话：一条保留数字但换了类别的行，是旧主张的词穿着新衣服）。
    """
    out: List[Violation] = []
    if previous is None:
        return out

    before = _gate_line_index(previous)
    after = _gate_line_index(doc)

    for key, old_line in before.items():
        if key not in after:
            if str(old_line.get("status")) == "PASS":
                # 已经关掉的行被删掉是正常的收尾
                continue
            out.append(Violation(
                "V16", key,
                "这条门禁行在上一版里有、这一版没了，而它还没销账。"
                "删掉一条机制行等于让那条机制从此不被核验 —— "
                "要放弃它请显式销账为 INFEASIBLE 并举证",
            ))
            continue

        new_line = after[key]
        old_threshold = _threshold_of(old_line)
        new_threshold = _threshold_of(new_line)
        if not old_threshold or not new_threshold:
            continue
        old_op, old_value = old_threshold
        new_op, new_value = new_threshold
        if old_op != new_op:
            out.append(Violation(
                "V16", f"{key}.anchor.expr",
                f"比较符从 {old_op} 改成了 {new_op}。"
                "保留数字但换了方向的行，是旧主张穿着新衣服 —— "
                "这是一个新主张，要重新举证",
            ))
            continue
        looser = ((old_op in ("<", "<=") and new_value > old_value)
                  or (old_op in (">", ">=") and new_value < old_value))
        if looser and _is_blank(new_line.get("band_rationale")):
            out.append(Violation(
                "V16", f"{key}.anchor.expr",
                f"阈值从 {old_value} 放宽到 {new_value}，而没有写 "
                "band_rationale。\n"
                "    放宽验收线是「挪球门」—— 实现没变而门禁过了，"
                "增益的解释就是错的。\n"
                "    确有理由（比如上一版那个数取自另一个 shape）请写明，"
                "那样至少是一个有记录的决定",
            ))
    return out


RULES = {
    "V0": ("课程表结构完整", check_document_shape),
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
    "V12": ("课程表不得含实现体", check_no_implementation_body),
    "V13": ("四条结构轴必须表态", check_adjudications),
    "V14": ("结构类门禁行必须先定价", check_priced_structural_lines),
    "V15": ("不得把参考的分配结果钉成验收线", check_no_pinned_allocation),
}

#: 需要上一版才能跑的规则。单独列出来，因为它们在 authoring 模式下无从比较。
PATCH_RULES = {
    "V16": ("与上一版做差：阈值不得悄悄放宽", check_against_previous),
}


def validate(doc: Dict[str, Any],
             rules: Optional[Iterable[str]] = None,
             previous: Optional[Dict[str, Any]] = None) -> List[Violation]:
    """跑校验，返回全部违规。

    rules 为 None 时跑全部规则。给了 previous（上一版课程表）就额外跑补丁
    规则 —— 它们在没有上一版时无从比较，所以不在默认集合里。

    上游把这件事做成双模：新蒸馏的 v1 全量严格，补丁只约束它新增的门禁行。
    我们这边的形态是「全量规则 + 给了上一版才跑的做差规则」，效果一样：
    不做差的话，放宽阈值与删掉机制行这两种改动完全无痕。
    """
    known = dict(RULES)
    known.update(PATCH_RULES)
    if rules:
        selected = list(rules)
    else:
        selected = list(RULES)
        if previous is not None:
            selected += list(PATCH_RULES)

    out: List[Violation] = []
    for rule in selected:
        entry = known.get(rule)
        if entry is None:
            out.append(Violation(
                "ARG", "--rules", f"未知规则 '{rule}'；可选: {sorted(known)}",
            ))
            continue
        if rule in PATCH_RULES:
            if previous is None:
                out.append(Violation(
                    "ARG", "--against",
                    f"规则 {rule} 需要上一版课程表才能比较，"
                    "请用 --against <上一版 milestone.json>",
                ))
                continue
            out.extend(entry[1](doc, previous))
        else:
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


def build_parser() -> argparse.ArgumentParser:
    """构造命令行。独立出来是为了让文档命令门禁能拿到它 ——
    门禁拿真实 parser 去解析文档里的示例，parser 藏在 main 里就够不着。
    """
    parser = argparse.ArgumentParser(
        description="校验 ForgeKernel milestone.json 是否满足论文规定的门禁义务",
    )
    parser.add_argument("milestone", type=Path, nargs="?",
                        help="milestone.json 路径")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="以 JSON 输出结果，供脚本消费")
    parser.add_argument("--rules", type=str, default="",
                        help="只跑指定规则，逗号分隔，如 V2,V6")
    parser.add_argument("--against", type=Path, default=None,
                        help="上一版 milestone.json。给了它会额外跑做差规则："
                             "拦住悄悄放宽阈值与删掉机制行 —— "
                             "这两种改动不做差完全无痕")
    parser.add_argument("--list-rules", action="store_true",
                        help="列出全部规则后退出")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_rules:
        for rule, (title, _fn) in RULES.items():
            print(f"{rule:<4} {title}")
        for rule, (title, _fn) in PATCH_RULES.items():
            print(f"{rule:<4} {title}（需要 --against）")
        return 0

    if args.milestone is None:
        parser.error("缺少 milestone.json 路径（或用 --list-rules 查看规则）")

    try:
        doc = _load(args.milestone)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    previous = None
    if args.against is not None:
        try:
            previous = _load(args.against)
        except SystemExit as exc:
            print(f"错误: 读不到上一版课程表 {args.against}: {exc}",
                  file=sys.stderr)
            return 2

    rules = [r.strip() for r in args.rules.split(",") if r.strip()] or None
    violations = validate(doc, rules, previous)

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
