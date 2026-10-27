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
        ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11"]
    )


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
    ("SUPPLY_KEEPUP", "mte2_wait_ratio"),
    ("WRITEBACK_OVERLAP", "mte3_wait_ratio"),
    ("BANK_LAYOUT", "aiv_vec_bank_cflt_ratio"),
    ("L2_LOCALITY", "total_hit_rate"),
    ("OUTPUT_ALIGNMENT", "aic_fixpipe_ratio"),
])
def test_v11_each_label_accepts_its_anchor(doc, label, anchor_field):
    line = mechanism_line(doc)
    line["label"] = label
    line["anchor"]["expr"] = f"msprof:X.csv:{anchor_field} < 1.0"
    assert "V11" not in rules_hit(mv.validate(doc, ["V11"]))


def test_v11_label_rejects_wrong_anchor(doc):
    """标签与锚点必须对应，不能张冠李戴。"""
    line = mechanism_line(doc)
    line["label"] = "BANK_LAYOUT"
    line["anchor"]["expr"] = "msprof:X.csv:aiv_mte2_wait_ratio < 1.0"
    assert "V11" in rules_hit(mv.validate(doc, ["V11"]))


def test_v11_only_applies_to_mechanism_lines(doc):
    """正确性行与性能行不需要机制标签。"""
    doc["milestones"][0]["gate"]["correctness_lines"][0].pop("label", None)
    doc["milestones"][0]["gate"]["performance_lines"][0].pop("label", None)
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
    del doc["milestones"][0]["gate"]["performance_lines"][0]["anchor"]
    assert "V4" in rules_hit(mv.validate(doc, ["V4"]))


# ---------------------------------------------------------------------------
# V5 门禁三类行齐全
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category",
                         ["correctness_lines", "performance_lines", "mechanism_lines"])
def test_v5_missing_line_category(doc, category):
    doc["milestones"][0]["gate"][category] = []
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
    doc["milestones"][0]["gate"]["performance_lines"][0]["anchor"]["kind"] = "agent_judgment"
    assert "V7" in rules_hit(mv.validate(doc, ["V7"]))


def test_v7_anchor_needs_expr(doc):
    del doc["milestones"][0]["gate"]["performance_lines"][0]["anchor"]["expr"]
    assert "V7" in rules_hit(mv.validate(doc, ["V7"]))


def test_v7_derived_needs_source(doc):
    anchor = doc["milestones"][0]["gate"]["performance_lines"][0]["anchor"]
    assert anchor["kind"] == "derived"
    del anchor["derived_from"]
    assert "V7" in rules_hit(mv.validate(doc, ["V7"]))


# ---------------------------------------------------------------------------
# V8 M0 不设绝对性能阈值
# ---------------------------------------------------------------------------

def test_v8_absolute_threshold_in_m0_rejected(doc):
    """架构刚落地还没调，拿绝对时间卡它会把正确架构误判成失败。"""
    line = doc["milestones"][0]["gate"]["performance_lines"][0]
    line["statement"] = "device time 必须低于 300 us"
    assert "V8" in rules_hit(mv.validate(doc, ["V8"]))


def test_v8_absolute_threshold_in_anchor_expr_rejected(doc):
    anchor = doc["milestones"][0]["gate"]["performance_lines"][0]["anchor"]
    anchor["expr"] = "evolved.time_us < 300us"
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
