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
"""milestone_validate.py 的单元测试。

测试策略：先确认随包样例能通过（避免校验器过严把合法输入也拦了），
再对每条规则各构造一个违规输入，确认对应规则会报错。

一个校验器最常见的失败模式不是漏报而是「从不报错」—— 所以每个用例都
断言具体规则号命中，而不只是断言「有违规」。

运行: python3 -m pytest plugins-community/forge-kernel/tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent / "skills" / "forge-milestone"
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import milestone_validate as mv  # noqa: E402

EXAMPLE_PATH = SKILL_ROOT / "references" / "example-milestone.json"


@pytest.fixture
def doc():
    """每个用例拿到一份独立的合法样例副本，改坏它来触发规则。"""
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def rules_hit(violations):
    """从违规列表里取出命中的规则号集合。"""
    return {v.rule for v in violations}


def first_decision(d):
    return d["milestones"][0]["decisions"][0]


# ---------------------------------------------------------------------------
# 基线：合法样例必须通过
# ---------------------------------------------------------------------------

def test_example_is_valid(doc):
    """随包样例必须零违规。它同时是文档，被 agent 当模板用，不能是坏的。"""
    violations = mv.validate(doc)
    assert violations == [], "样例不合法:\n" + "\n".join(str(v) for v in violations)


def test_all_rules_registered():
    """规则表与文档声明的保持一致。"""
    assert sorted(mv.RULES) == sorted(
        ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
         "V11", "V12", "V13", "V14", "V15"]
    )
    # 做差规则单列：没有上一版时它们无从比较，所以不在默认集合里
    assert sorted(mv.PATCH_RULES) == ["V16"]


# ---------------------------------------------------------------------------
# V11 机制行须有标签与停顿锚点
# ---------------------------------------------------------------------------

def mechanism_line(doc, milestone=0):
    return doc["milestones"][milestone]["gate"]["mechanism_lines"][0]


def test_v11_missing_label(doc):
    """没有标签的机制行无法被程序核验，等于换个说法说「变快了」。"""
    del mechanism_line(doc)["label"]
    assert "V11" in rules_hit(mv.validate(doc, ["V11"]))


def test_v11_label_outside_vocabulary(doc):
    """自造标签等于没有标签。"""
    mechanism_line(doc)["label"] = "OVERLAP_IMPROVED"
    violations = mv.validate(doc, ["V11"])
    assert "V11" in rules_hit(violations)
    assert any("闭合词汇表" in v.message for v in violations)


def test_v11_ratio_anchor_rejected(doc):
    """占比下降可能只是别的段变慢了，不能替代停顿指标。"""
    line = mechanism_line(doc)
    assert line["label"] == "SUPPLY_KEEPUP"
    line["anchor"]["expr"] = "msprof:PipeUtilization.csv:aiv_mte2_ratio < 61.2"
    violations = mv.validate(doc, ["V11"])
    assert "V11" in rules_hit(violations)
    assert any("停顿" in v.message for v in violations)


def test_v11_correct_stall_anchor_accepted(doc):
    assert "V11" not in rules_hit(mv.validate(doc, ["V11"]))


def test_v11_missing_ordering_evidence(doc):
    """单独一个数说明不了变化方向。"""
    del mechanism_line(doc)["ordering_evidence"]
    assert "V11" in rules_hit(mv.validate(doc, ["V11"]))


@pytest.mark.parametrize("label,anchor_field", [
    ("CV_HANDOFF", "aic_cube_wait_ratio"),
    ("SUPPLY_KEEPUP", "aiv_mte2_wait_ratio"),
    ("WRITEBACK_OVERLAP", "aiv_mte3_wait_ratio"),
    ("BANK_LAYOUT", "aiv_vec_bank_cflt_ratio"),
    ("L2_LOCALITY", "total_hit_rate"),
    ("OUTPUT_ALIGNMENT", "aic_fixpipe_ratio"),
])
def test_v11_each_label_accepts_its_anchor(doc, label, anchor_field):
    line = mechanism_line(doc)
    line["label"] = label
    line["anchor"]["expr"] = f"msprof:X.csv:{anchor_field} < 1.0"
    assert "V11" not in rules_hit(mv.validate(doc, ["V11"]))


def test_v11_requires_unit_prefix_on_mte_fields(doc):
    """不带 aic_/aiv_ 前缀的 MTE 字段会同时匹配两个单元，必须拒。"""
    line = mechanism_line(doc)
    line["label"] = "SUPPLY_KEEPUP"
    line["anchor"]["expr"] = "msprof:X.csv:mte2_wait_ratio < 4.2"
    assert "V11" in rules_hit(mv.validate(doc, ["V11"]))


def test_v11_label_rejects_wrong_anchor(doc):
    """标签与锚点必须对应，不能张冠李戴。"""
    line = mechanism_line(doc)
    line["label"] = "BANK_LAYOUT"
    line["anchor"]["expr"] = "msprof:X.csv:aiv_mte2_wait_ratio < 1.0"
    assert "V11" in rules_hit(mv.validate(doc, ["V11"]))


def test_v11_only_applies_to_mechanism_lines(doc):
    """正确性行与性能行不需要机制标签。"""
    doc["milestones"][0]["gate"]["correctness_lines"][0].pop("label", None)
    doc["milestones"][1]["gate"]["performance_lines"][0].pop("label", None)
    assert "V11" not in rules_hit(mv.validate(doc, ["V11"]))


def test_v11_vocabulary_is_documented():
    """词汇表每个标签都要有必需锚点，空列表等于没约束。"""
    for label, anchors in mv.MECHANISM_LABELS.items():
        assert anchors, f"{label} 没有指定必需锚点"


# ---------------------------------------------------------------------------
# V1 五元组完整
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["action", "purpose", "trigger", "cost", "depends_on"])
def test_v1_missing_tuple_field(doc, field):
    del first_decision(doc)[field]
    assert "V1" in rules_hit(mv.validate(doc, ["V1"]))


def test_v1_blank_field_is_missing(doc):
    """空白字符串等同于没填。"""
    first_decision(doc)["purpose"] = "   "
    assert "V1" in rules_hit(mv.validate(doc, ["V1"]))


def test_v1_depends_on_must_be_list(doc):
    first_decision(doc)["depends_on"] = "D0.0"
    assert "V1" in rules_hit(mv.validate(doc, ["V1"]))


def test_v1_empty_depends_on_is_fine(doc):
    """根决策没有依赖是正常的，不该报错。"""
    first_decision(doc)["depends_on"] = []
    assert "V1" not in rules_hit(mv.validate(doc, ["V1"]))


# ---------------------------------------------------------------------------
# V2 决策必须有触发器
# ---------------------------------------------------------------------------

def test_v2_missing_symptom(doc):
    """说不出观测到什么就要做这个决策 —— 论文称之为不被支持的主张。"""
    del first_decision(doc)["trigger"]["symptom"]
    assert "V2" in rules_hit(mv.validate(doc, ["V2"]))


def test_v2_missing_channel(doc):
    """症状必须能被程序复核，光有文字描述不够。"""
    del first_decision(doc)["trigger"]["channel"]
    assert "V2" in rules_hit(mv.validate(doc, ["V2"]))


def test_v2_trigger_must_be_object(doc):
    first_decision(doc)["trigger"] = "MTE2 太高"
    assert "V2" in rules_hit(mv.validate(doc, ["V2"]))


def test_v2_observed_needs_provenance(doc):
    first_decision(doc)["trigger"]["observed"]["provenance"] = "我觉得是这样"
    assert "V2" in rules_hit(mv.validate(doc, ["V2"]))


# ---------------------------------------------------------------------------
# V3 派生图无环
# ---------------------------------------------------------------------------

def test_v3_self_cycle(doc):
    first_decision(doc)["depends_on"] = ["D0.1"]
    assert "V3" in rules_hit(mv.validate(doc, ["V3"]))


def test_v3_two_node_cycle(doc):
    """D0.1 -> D0.2 -> D0.1。有环就无法确定 patch 后从哪里重执行。"""
    decisions = doc["milestones"][0]["decisions"]
    decisions[0]["depends_on"] = ["D0.2"]
    decisions[1]["depends_on"] = ["D0.1"]
    violations = mv.validate(doc, ["V3"])
    assert "V3" in rules_hit(violations)
    assert any("环" in v.message for v in violations)


def test_v3_dangling_reference(doc):
    first_decision(doc)["depends_on"] = ["D9.9"]
    violations = mv.validate(doc, ["V3"])
    assert "V3" in rules_hit(violations)
    assert any("不存在" in v.message for v in violations)


def test_v3_duplicate_id(doc):
    doc["milestones"][0]["decisions"][1]["id"] = "D0.1"
    violations = mv.validate(doc, ["V3"])
    assert "V3" in rules_hit(violations)
    assert any("重复" in v.message for v in violations)


def test_v3_cross_milestone_dag_is_fine(doc):
    """M1 的决策依赖 M0 的决策是正常形态，不该误报成环。"""
    assert "V3" not in rules_hit(mv.validate(doc, ["V3"]))


# ---------------------------------------------------------------------------
# V4 数字必须有出处
# ---------------------------------------------------------------------------

def test_v4_contract_missing_provenance(doc):
    del doc["resource_topology_contract"]["block_dim"]["provenance"]
    assert "V4" in rules_hit(mv.validate(doc, ["V4"]))


def test_v4_bare_estimate_rejected(doc):
    """裸 [ESTIMATE] 不附推导，和没有出处等价。"""
    doc["resource_topology_contract"]["block_dim"]["provenance"] = "[ESTIMATE]"
    assert "V4" in rules_hit(mv.validate(doc, ["V4"]))


def test_v4_estimate_with_derivation_accepted(doc):
    doc["resource_topology_contract"]["block_dim"]["provenance"] = (
        "[ESTIMATE] 由 core 数 24 乘 2 得到"
    )
    assert "V4" not in rules_hit(mv.validate(doc, ["V4"]))


def test_v4_measured_requires_job_id(doc):
    """[MEASURED] 必须说明是哪次 job，否则无法复核。"""
    doc["resource_topology_contract"]["block_dim"]["provenance"] = "[MEASURED]"
    assert "V4" in rules_hit(mv.validate(doc, ["V4"]))


def test_v4_performance_line_needs_anchor(doc):
    del doc["milestones"][1]["gate"]["performance_lines"][0]["anchor"]
    assert "V4" in rules_hit(mv.validate(doc, ["V4"]))


# ---------------------------------------------------------------------------
# V5 门禁三类行齐全
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category",
                         ["correctness_lines", "performance_lines", "mechanism_lines"])
def test_v5_missing_line_category(doc, category):
    """三类行缺一则无法区分真机制与巧合加速。

    在 M1 上测而不是 M0：M0 现在**不该**有性能行（第一版能跑到哪里事先无法
    知道，设高了不可达、设低了没意义），所以 M0 的性能行为空是对的。
    """
    doc["milestones"][1]["gate"][category] = []
    assert "V5" in rules_hit(mv.validate(doc, ["V5"]))


def test_v5_m0_must_not_have_performance_lines(doc):
    """反过来：给 M0 加一条性能行也是违规。

    强制要求它的唯一产物是训练架构层发明一条敷衍的性能行 —— 要么写成几乎
    必然不达标的数（只能走 DEFERRED 的三条举证），要么写成毫无约束力的数
    再配一个合法出处。
    """
    doc["milestones"][0]["gate"]["performance_lines"] = [{
        "id": "P9", "statement": "至少 0.3x", "status": "PASS",
        "closed_by": "[MEASURED, job 1]",
        "anchor": {"kind": "derived", "derived_from": "瞎猜",
                   "expr": "evaluation_results.json:comparison.speedup >= 0.3",
                   "provenance": "[ESTIMATE] 随手写的"},
    }]
    assert "V5" in rules_hit(mv.validate(doc, ["V5"]))


def test_v5_missing_gate(doc):
    del doc["milestones"][0]["gate"]
    assert "V5" in rules_hit(mv.validate(doc, ["V5"]))


# ---------------------------------------------------------------------------
# V6 未决行必须销账
# ---------------------------------------------------------------------------

def test_v6_open_line_blocks_close(doc):
    doc["milestones"][0]["gate"]["correctness_lines"][0]["status"] = "OPEN"
    assert "V6" in rules_hit(mv.validate(doc, ["V6"]))


def test_v6_invalid_status(doc):
    doc["milestones"][0]["gate"]["correctness_lines"][0]["status"] = "MAYBE"
    assert "V6" in rules_hit(mv.validate(doc, ["V6"]))


def test_v6_deferred_needs_downstream_lever(doc):
    """DEFERRED 不指名接手的下游手段，就是丢弃而非推迟。"""
    line = doc["milestones"][1]["gate"]["mechanism_lines"][0]
    assert line["status"] == "DEFERRED"
    del line["downstream_lever"]
    assert "V6" in rules_hit(mv.validate(doc, ["V6"]))


@pytest.mark.parametrize("field",
                         ["evidence", "reference_achieves", "current_levers_cannot"])
def test_v6_deferred_evidence_obligations(doc, field):
    line = doc["milestones"][1]["gate"]["mechanism_lines"][0]
    del line[field]
    assert "V6" in rules_hit(mv.validate(doc, ["V6"]))


def test_v6_infeasible_needs_per_metric_argument(doc):
    line = doc["milestones"][1]["gate"]["mechanism_lines"][0]
    line["status"] = "INFEASIBLE"
    assert "V6" in rules_hit(mv.validate(doc, ["V6"]))


def test_v6_infeasible_with_argument_accepted(doc):
    line = doc["milestones"][1]["gate"]["mechanism_lines"][0]
    line["status"] = "INFEASIBLE"
    line["per_metric_argument"] = ["UB 容量 192KB 已被行常驻占满，无法再分配"]
    assert "V6" not in rules_hit(mv.validate(doc, ["V6"]))


def test_v6_na_needs_structural_reason(doc):
    line = doc["milestones"][1]["gate"]["mechanism_lines"][0]
    line["status"] = "N/A"
    assert "V6" in rules_hit(mv.validate(doc, ["V6"]))


def test_v6_na_with_structural_reason_accepted(doc):
    line = doc["milestones"][1]["gate"]["mechanism_lines"][0]
    line["status"] = "N/A"
    line["structural_reason"] = "该算子无 Cube 参与，Cube 相关谓词结构上不适用"
    assert "V6" not in rules_hit(mv.validate(doc, ["V6"]))


# ---------------------------------------------------------------------------
# V7 门禁锚点来源受限
# ---------------------------------------------------------------------------

def test_v7_illegal_anchor_kind(doc):
    """agent 不能自己定一个好听的目标再去达成它。"""
    doc["milestones"][1]["gate"]["performance_lines"][0]["anchor"]["kind"] = "agent_judgment"
    assert "V7" in rules_hit(mv.validate(doc, ["V7"]))


def test_v7_anchor_needs_expr(doc):
    del doc["milestones"][1]["gate"]["performance_lines"][0]["anchor"]["expr"]
    assert "V7" in rules_hit(mv.validate(doc, ["V7"]))


def test_v7_derived_needs_source(doc):
    anchor = doc["milestones"][1]["gate"]["performance_lines"][0]["anchor"]
    assert anchor["kind"] == "derived"
    del anchor["derived_from"]
    assert "V7" in rules_hit(mv.validate(doc, ["V7"]))


# ---------------------------------------------------------------------------
# V8 M0 不设绝对性能阈值
# ---------------------------------------------------------------------------

def test_v8_absolute_threshold_in_m0_rejected(doc):
    """架构刚落地还没调，拿绝对时间卡它会把正确架构误判成失败。

    在 M0 的**机制行**上测：M0 现在没有性能行了（V5 禁止），而绝对时间阈值
    一样能写进机制行，危害完全相同 —— 所以 V8 要扫 M0 的全部门禁行。
    """
    line = doc["milestones"][0]["gate"]["mechanism_lines"][0]
    line["statement"] = "device time 必须低于 300 us"
    assert "V8" in rules_hit(mv.validate(doc, ["V8"]))


def test_v8_absolute_threshold_in_anchor_expr_rejected(doc):
    anchor = doc["milestones"][0]["gate"]["mechanism_lines"][0]["anchor"]
    anchor["expr"] = "evolved.time_us < 300us"
    assert "V8" in rules_hit(mv.validate(doc, ["V8"]))


def test_v8_also_scans_m0_correctness_lines(doc):
    """绝对阈值写进正确性行同样要拦 —— 只扫性能行等于这条规则作废。"""
    line = doc["milestones"][0]["gate"]["correctness_lines"][0]
    line["statement"] = "端到端必须在 50us 内返回"
    assert "V8" in rules_hit(mv.validate(doc, ["V8"]))


def test_v8_ratio_in_m0_accepted(doc):
    """相对参考的比率是 M0 允许的表达方式。"""
    assert "V8" not in rules_hit(mv.validate(doc, ["V8"]))


def test_v8_absolute_threshold_allowed_after_m0(doc):
    """M1 及以后不受此限 —— 那时架构已定，绝对阈值有意义。"""
    line = doc["milestones"][1]["gate"]["performance_lines"][0]
    line["statement"] = "device time 必须低于 300 us"
    assert "V8" not in rules_hit(mv.validate(doc, ["V8"]))


# ---------------------------------------------------------------------------
# V9 原子组同批落地
# ---------------------------------------------------------------------------

def test_v9_group_split_across_milestones(doc):
    """原子组散落在不同 milestone 就不可能同批落地。"""
    doc["milestones"][1]["decisions"][0]["atomic_group"] = "g1"
    violations = mv.validate(doc, ["V9"])
    assert "V9" in rules_hit(violations)
    assert any("跨" in v.message for v in violations)


def test_v9_single_member_group_is_meaningless(doc):
    doc["milestones"][0]["decisions"][1]["atomic_group"] = "g2"
    violations = mv.validate(doc, ["V9"])
    assert "V9" in rules_hit(violations)
    assert any("一个成员" in v.message for v in violations)


def test_v9_wellformed_group_accepted(doc):
    assert "V9" not in rules_hit(mv.validate(doc, ["V9"]))


# ---------------------------------------------------------------------------
# V10 架构决策须命名验证通道
# ---------------------------------------------------------------------------

def test_v10_architectural_decision_needs_channel(doc):
    del first_decision(doc)["verification_channel"]
    assert "V10" in rules_hit(mv.validate(doc, ["V10"]))


def test_v10_tactical_decision_exempt(doc):
    """战术决策通常由上层架构决策的通道一并覆盖，不强制。"""
    tactical = doc["milestones"][1]["decisions"][0]
    assert tactical["layer"] == "tactical"
    del tactical["verification_channel"]
    assert "V10" not in rules_hit(mv.validate(doc, ["V10"]))


# ---------------------------------------------------------------------------
# 健壮性：畸形输入不能让校验器崩溃
# ---------------------------------------------------------------------------

def test_empty_document_does_not_crash():
    violations = mv.validate({})
    assert isinstance(violations, list)


def test_malformed_milestones_do_not_crash():
    doc = {"milestones": ["不是对象", None, 42, {"id": "M0"}]}
    violations = mv.validate(doc)
    assert isinstance(violations, list)


def test_unknown_rule_reported():
    violations = mv.validate({}, ["V99"])
    assert "ARG" in rules_hit(violations)


def test_decisions_not_a_list_does_not_crash():
    doc = {"milestones": [{"id": "M0", "decisions": "坏数据", "gate": {}}]}
    violations = mv.validate(doc)
    assert isinstance(violations, list)


# ---------------------------------------------------------------------------
# V0 课程表结构完整
#
# 其余十一条规则都在「遍历决策/门禁行」的循环里做检查。空文档、键名拼错、
# 门禁行写成一句话这三种情况下，那些循环一次都不会执行 —— 于是零违规，
# 输出「通过」。而 forge-coder 的规则是「validate 不通过就不要开始实现」，
# 它拿到 rc=0 就会在一份空课程表上开工。
# ---------------------------------------------------------------------------

def rules_of(violations):
    return {v.rule for v in violations}


def test_empty_document_does_not_pass():
    assert "V0" in rules_of(mv.validate({}))


def test_misspelled_top_level_key_is_caught():
    violations = mv.validate({"mileStones": [{"id": "M0"}]})
    assert "V0" in rules_of(violations)
    # 报错要指出实际有哪些键，否则光说「没有 milestones」帮不上忙
    assert any("mileStones" in v.message for v in violations)


def test_empty_milestone_list_does_not_pass():
    assert "V0" in rules_of(mv.validate({"milestones": []}))


def test_milestone_without_any_decision_does_not_pass():
    violations = mv.validate({"milestones": [{"id": "M0", "decisions": []}]})
    assert "V0" in rules_of(violations)


def test_prose_gate_line_is_rejected_not_skipped(doc):
    """写成一句话的门禁行读起来像义务，实际不受任何核验。"""
    doc["milestones"][0]["gate"]["mechanism_lines"].append("机制落地了，快了 30%")
    violations = mv.validate(doc)
    assert "V0" in rules_of(violations)
    assert any("门禁行必须是对象" in v.message for v in violations)


def test_non_dict_decision_is_rejected_not_skipped(doc):
    doc["milestones"][0]["decisions"].append("D9.9 顺手把 tiling 调一下")
    violations = mv.validate(doc)
    assert "V0" in rules_of(violations)


def test_non_dict_milestone_is_rejected_not_skipped(doc):
    doc["milestones"].append("M9 收尾")
    assert "V0" in rules_of(mv.validate(doc))


# ---------------------------------------------------------------------------
# V11 复合锚点
#
# 核验端只认单个「通道 比较符 数值」。写成复合条件的话，record 时会判为
# 「无法核验」—— 而那时课程表和代码都写完了，失败看起来像实现问题。
# 两端判据要一致，门禁才有意义。
# ---------------------------------------------------------------------------

def test_compound_mechanism_anchor_is_rejected(doc):
    line = mechanism_line(doc, milestone=1)
    line["anchor"]["expr"] = (
        "msprof:ResourceConflictRatio.csv:aiv_mte2_wait_ratio < 4.2 "
        "and msprof:ResourceConflictRatio.csv:aiv_mte3_wait_ratio < 0.3")
    violations = mv.validate(doc, rules=["V11"])
    assert "V11" in rules_of(violations)
    assert any("多个条件" in v.message for v in violations)


def test_single_mechanism_anchor_still_passes(doc):
    violations = mv.validate(doc, rules=["V11"])
    assert violations == [], [str(v) for v in violations]


# ---------------------------------------------------------------------------
# FAIL 不再是「已销账」
#
# 它看起来像一个终态，实际零举证义务：早先 RESOLVED_STATUSES 含它，而销账
# 检查对已销账状态直接跳过 —— 于是把全部门禁行改成 FAIL，校验通过、
# gate_impasse 也不报，一个 milestone 可以带着满页 FAIL 合法关闭。
# ---------------------------------------------------------------------------

def test_fail_is_not_a_legal_status(doc):
    doc["milestones"][0]["gate"]["mechanism_lines"][0]["status"] = "FAIL"
    assert "V6" in rules_hit(mv.validate(doc, ["V6"]))


def test_all_fail_milestone_does_not_validate(doc):
    for m in doc["milestones"]:
        for cat in mv.LINE_CATEGORIES:
            for ln in m.get("gate", {}).get(cat, []):
                ln["status"] = "FAIL"
    assert mv.validate(doc) != []


def test_pass_requires_closed_by(doc):
    """干净的 PASS 也要说清是哪次测量把它关掉的。

    anchor.provenance 说的是**阈值**的来源，不是「这次达标」的证据。
    不要求的话 PASS 是零举证的自述状态，把所有行改成 PASS 就能让
    「门禁僵持」这个重入口消失。
    """
    doc["milestones"][0]["gate"]["mechanism_lines"][0].pop("closed_by", None)
    assert "V6" in rules_hit(mv.validate(doc, ["V6"]))


# ---------------------------------------------------------------------------
# DEFERRED 的举证义务
# ---------------------------------------------------------------------------

def deferred_line(doc, milestone=1):
    lines = doc["milestones"][milestone]["gate"]["mechanism_lines"]
    ln = lines[0]
    ln["status"] = "DEFERRED"
    ln.pop("closed_by", None)
    ln["evidence"] = "job 7 实测"
    ln["reference_achieves"] = "参考该通道 0.24"
    ln["current_levers_cannot"] = "当前手段改不动写回时序"
    return ln


def test_deferred_to_a_nonexistent_lever_is_rejected(doc):
    """指向不存在的下游手段 = 这条测试被丢弃了，不是推迟了。

    随包的 example-milestone.json 原先就是这个样子（指向不存在的 D2.1），
    而 validate 返回 rc=0「满足全部门禁义务」—— 而 SKILL.md 教架构层拿它
    当模板起草，coder 的规则是「validate 不通过就不要开始实现」。
    """
    ln = deferred_line(doc)
    ln["downstream_lever"] = "D9.9"
    assert "V6" in rules_hit(mv.validate(doc, ["V6"]))


def test_deferred_to_an_earlier_lever_is_rejected(doc):
    """手里已经握着手段还推迟，说明这个 milestone 没做完。"""
    ln = deferred_line(doc)
    ln["downstream_lever"] = "D0.1"
    assert "V6" in rules_hit(mv.validate(doc, ["V6"]))


def test_last_milestone_cannot_defer(doc):
    """最后一个 milestone 没有更靠后的手段可以接手了。"""
    ln = deferred_line(doc, milestone=len(doc["milestones"]) - 1)
    ln["downstream_lever"] = "D0.1"
    assert "V6" in rules_hit(mv.validate(doc, ["V6"]))


def test_correctness_lines_cannot_be_deferred(doc):
    """算错的 kernel 再快也无意义，这类测试没有「交给下游」这个出口。"""
    ln = doc["milestones"][1]["gate"]["correctness_lines"][0]
    ln["status"] = "DEFERRED"
    ln.pop("closed_by", None)
    ln["evidence"] = "x"
    ln["downstream_lever"] = "D2.1"
    ln["reference_achieves"] = "x"
    ln["current_levers_cannot"] = "x"
    assert "V6" in rules_hit(mv.validate(doc, ["V6"]))


# ---------------------------------------------------------------------------
# V12 课程表不得含实现体
#
# 课程表是参考实现**唯一可能的泄漏通道**：架构层读过参考的硬件行为（上游的
# source 模式下甚至读过源码），然后写课程表；而编码层读课程表。它一旦能装
# 代码，前面那道 no-code 边界就有了一条绕过去的路。
# ---------------------------------------------------------------------------

def test_code_fence_in_a_decision_is_rejected(doc):
    doc["milestones"][0]["decisions"][0]["action"] = (
        "按这样写：```cpp\nfor (int i = 0; i < N; i++) {}\n```")
    assert "V12" in rules_hit(mv.validate(doc, ["V12"]))


def test_aicore_qualifier_is_rejected(doc):
    doc["milestones"][0]["decisions"][0]["purpose"] = (
        "在 __aicore__ inline void Process() 里做双缓冲")
    assert "V12" in rules_hit(mv.validate(doc, ["V12"]))


def test_loop_body_is_rejected(doc):
    doc["milestones"][0]["decisions"][0]["cost"] = (
        "多一次循环：for (int i = 0; i < tileNum; i++) { Compute(i); }")
    assert "V12" in rules_hit(mv.validate(doc, ["V12"]))


def test_anchor_expr_is_exempt(doc):
    """契约本来就要写表达式 —— 锚点不能因为长得像代码被拦。"""
    assert "V12" not in rules_hit(mv.validate(doc, ["V12"]))


def test_structural_anchor_is_exempt(doc):
    line = doc["milestones"][0]["gate"]["mechanism_lines"][0]
    line["structural_anchor"] = ["buffer_depth:2"]
    line["priced"] = {"delta_pct": 8.4, "source": "lever:double_buffer"}
    assert "V12" not in rules_hit(mv.validate(doc, ["V12"]))


# ---------------------------------------------------------------------------
# V13 四条结构轴必须表态
# ---------------------------------------------------------------------------

def test_missing_adjudications_is_rejected(doc):
    doc.pop("adjudications", None)
    assert "V13" in rules_hit(mv.validate(doc, ["V13"]))


def test_each_axis_must_be_present(doc):
    doc["adjudications"].pop("supply", None)
    violations = mv.validate(doc, ["V13"])
    assert "V13" in rules_hit(violations)
    assert any("supply" in v.path for v in violations)


def test_explicit_none_is_a_valid_adjudication(doc):
    """表态为 none 是合法的 —— 要的是「表过态」，不是「有这个机制」。"""
    doc["adjudications"] = {k: "none" for k in mv.ADJUDICATION_AXES}
    assert "V13" not in rules_hit(mv.validate(doc, ["V13"]))


def test_value_outside_the_vocabulary_is_reported_separately(doc):
    """词汇表外的值说明作者表过态但用词不对 —— 该对齐用词，不是补一次裁定。"""
    doc["adjudications"]["supply"] = "我自己想的一种供给方式"
    violations = mv.validate(doc, ["V13"])
    assert any("未知取值" in v.message for v in violations)


def test_non_none_axis_needs_a_mechanism_line(doc):
    """表了态就要有一条可核验的行，否则这个裁定只是一句声明。"""
    doc["adjudications"]["cv_handoff"] = "flag_pair"   # 但没有 CV_HANDOFF 行
    violations = mv.validate(doc, ["V13"])
    assert any("CV_HANDOFF" in v.message for v in violations)


# ---------------------------------------------------------------------------
# V14 结构类门禁行必须先定价
# ---------------------------------------------------------------------------

def test_structural_anchor_without_price_is_rejected(doc):
    doc["milestones"][0]["gate"]["mechanism_lines"][0]["structural_anchor"] = \
        ["buffer_depth:2"]
    assert "V14" in rules_hit(mv.validate(doc, ["V14"]))


def test_priced_structural_anchor_passes(doc):
    line = doc["milestones"][0]["gate"]["mechanism_lines"][0]
    line["structural_anchor"] = ["buffer_depth:2"]
    line["priced"] = {"delta_pct": 8.4, "source": "lever:double_buffer"}
    assert "V14" not in rules_hit(mv.validate(doc, ["V14"]))


def test_price_needs_a_source(doc):
    """一个没有出处的价格和没有价格一样。"""
    line = doc["milestones"][0]["gate"]["mechanism_lines"][0]
    line["structural_anchor"] = ["buffer_depth:2"]
    line["priced"] = {"delta_pct": 8.4}
    assert "V14" in rules_hit(mv.validate(doc, ["V14"]))


def test_price_must_be_positive(doc):
    line = doc["milestones"][0]["gate"]["mechanism_lines"][0]
    line["structural_anchor"] = ["buffer_depth:2"]
    line["priced"] = {"delta_pct": 0, "source": "job 7"}
    assert "V14" in rules_hit(mv.validate(doc, ["V14"]))


def test_line_without_structural_anchor_needs_no_price(doc):
    assert "V14" not in rules_hit(mv.validate(doc, ["V14"]))


# ---------------------------------------------------------------------------
# V15 不得把参考的分配结果钉成验收线
# ---------------------------------------------------------------------------

def test_pinned_block_dim_is_rejected(doc):
    """block_dim == 48 是在参考上观测到的分配结果，不是要求。"""
    line = doc["milestones"][1]["gate"]["performance_lines"][0]
    line["anchor"]["expr"] = "msprof:PipeUtilization.csv:block_dim == 48"
    violations = mv.validate(doc, ["V15"])
    assert "V15" in rules_hit(violations)
    assert any("观测到" in v.message for v in violations)


def test_pinned_buffer_num_is_rejected(doc):
    line = doc["milestones"][1]["gate"]["performance_lines"][0]
    line["anchor"]["expr"] = "buffer_num == 2"
    assert "V15" in rules_hit(mv.validate(doc, ["V15"]))


def test_upper_bound_on_the_same_field_is_allowed(doc):
    """门禁只写上界，具体取值留给实现选。"""
    line = doc["milestones"][1]["gate"]["performance_lines"][0]
    line["anchor"]["expr"] = "msprof:PipeUtilization.csv:block_dim <= 48"
    assert "V15" not in rules_hit(mv.validate(doc, ["V15"]))


# ---------------------------------------------------------------------------
# V16 与上一版做差
#
# 放宽阈值与删掉机制行这两种改动，不做差完全无痕。
# ---------------------------------------------------------------------------

def test_widened_threshold_is_caught(doc):
    import copy  # noqa: PLC0415
    previous = copy.deepcopy(doc)
    doc["milestones"][0]["gate"]["mechanism_lines"][0]["anchor"]["expr"] = \
        "msprof:ResourceConflictRatio.csv:aiv_mte2_wait_ratio < 99.0"
    violations = mv.validate(doc, ["V16"], previous)
    assert "V16" in rules_hit(violations)
    assert any("放宽" in v.message for v in violations)


def test_widening_with_a_rationale_is_allowed(doc):
    """确有理由请写明 —— 那样至少是一个有记录的决定。"""
    import copy  # noqa: PLC0415
    previous = copy.deepcopy(doc)
    line = doc["milestones"][0]["gate"]["mechanism_lines"][0]
    line["anchor"]["expr"] = \
        "msprof:ResourceConflictRatio.csv:aiv_mte2_wait_ratio < 99.0"
    line["band_rationale"] = "上一版那个数取自另一个 shape"
    assert "V16" not in rules_hit(mv.validate(doc, ["V16"], previous))


def test_tightening_a_threshold_is_always_allowed(doc):
    import copy  # noqa: PLC0415
    previous = copy.deepcopy(doc)
    doc["milestones"][0]["gate"]["mechanism_lines"][0]["anchor"]["expr"] = \
        "msprof:ResourceConflictRatio.csv:aiv_mte2_wait_ratio < 1.0"
    assert "V16" not in rules_hit(mv.validate(doc, ["V16"], previous))


def test_deleting_an_unresolved_mechanism_line_is_caught(doc):
    """删掉一条机制行等于让那条机制从此不被核验。"""
    import copy  # noqa: PLC0415
    previous = copy.deepcopy(doc)
    previous["milestones"][0]["gate"]["mechanism_lines"][0]["status"] = "OPEN"
    doc["milestones"][0]["gate"]["mechanism_lines"] = []
    assert "V16" in rules_hit(mv.validate(doc, ["V16"], previous))


def test_flipping_the_comparator_is_a_new_claim(doc):
    """保留数字但换了方向的行，是旧主张穿着新衣服。"""
    import copy  # noqa: PLC0415
    previous = copy.deepcopy(doc)
    doc["milestones"][0]["gate"]["mechanism_lines"][0]["anchor"]["expr"] = \
        "msprof:ResourceConflictRatio.csv:aiv_mte2_wait_ratio > 4.2"
    violations = mv.validate(doc, ["V16"], previous)
    assert any("新衣服" in v.message for v in violations)


def test_patch_rules_need_the_previous_version(doc):
    violations = mv.validate(doc, ["V16"], None)
    assert any(v.rule == "ARG" for v in violations)


def test_patch_rules_run_automatically_when_against_is_given(doc):
    import copy  # noqa: PLC0415
    previous = copy.deepcopy(doc)
    doc["milestones"][0]["gate"]["mechanism_lines"][0]["anchor"]["expr"] = \
        "msprof:ResourceConflictRatio.csv:aiv_mte2_wait_ratio < 99.0"
    # 不显式指定规则，做差规则也要跑
    assert "V16" in rules_hit(mv.validate(doc, None, previous))
