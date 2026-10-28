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
"""campaign.py 与 noise_floor.py 的单元测试。

这两个脚本合起来决定「什么样的改动算数」。测试围绕五件容易出错的判断：

  边界     增益恰好等于噪声底时该不该收？不该 —— 等于噪声意味着无法区分。
  优先级   精度不过的候选，增益再高也要回滚。
  机制     变快了不等于设计的机制落地了。没核实的不能计入 best，
           更不能作为入库依据 —— 否则一个巧合会被其他算子照着做决策。
  归因     停滞只数「测出来了但不够」。编译 bug 不该消耗架构重入机会。
  重启范围 重开一个决策时，依赖它的下游实现必须一并作废；
           漏一条，新架构上就还挂着旧假设的实现。

运行: python3 -m pytest plugins-community/forge-kernel/tests/ -v
"""

import argparse
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = (Path(__file__).resolve().parent.parent
           / "skills" / "forge-campaign" / "scripts")
sys.path.insert(0, str(SCRIPTS))

import campaign  # noqa: E402
import noise_floor  # noqa: E402


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def fresh_state(**kw):
    state = campaign.new_state(
        operator="demo_op", input_shape="B=8,S=4096,fp16", arch="DAV_2201",
        reference={"kind": "builtin_aclnn", "id": "aclnnDemo", "sealed": True},
        noise_floor_pct=kw.get("noise", 2.0),
        target_gain=kw.get("target", 1.5),
        stagnation_threshold=kw.get("threshold", 3),
        budget=kw.get("budget"),
    )
    state["state"] = campaign.STATE_IMPLEMENT
    return state


def result(speedup=None, precision=True, baseline_us=None, evolved_us=None):
    doc = {"comparison": {"precision_passed": precision}}
    if speedup is not None:
        doc["comparison"]["speedup"] = speedup
    if baseline_us is not None:
        doc["baseline"] = {"time_us": baseline_us}
    if evolved_us is not None:
        doc["evolved"] = {"time_us": evolved_us, "precision_passed": precision}
    return doc


def signature(stall_value=0.9, trustworthy=True, present=True):
    """造一份最小的架构签名，只含机制核验需要的那个停顿字段。"""
    return {
        "trust": {"trustworthy": trustworthy},
        "stalls": {
            "wait": {
                "mte2_wait": {
                    "value": stall_value if present else None,
                    "channel": "msprof:ResourceConflictRatio.csv:aiv_mte2_wait_ratio",
                    "present": present,
                }
            }
        },
    }


def milestone(decisions=None, mechanism_expr=None, gate_extra=None):
    """造一份最小的 milestone。decisions 是 {id: [依赖...]}。"""
    decisions = decisions or {"D0.1": []}
    expr = mechanism_expr or (
        "msprof:ResourceConflictRatio.csv:aiv_mte2_wait_ratio < 4.2")
    gate = {
        "correctness_lines": [{"id": "C1", "status": "PASS"}],
        "performance_lines": [{"id": "P1", "status": "PASS"}],
        "mechanism_lines": [{
            "id": "M1", "label": "SUPPLY_KEEPUP", "status": "PASS",
            "anchor": {"kind": "reference_measurement", "expr": expr},
            "ordering_evidence": "job 1 为 4.2，job 2 为 0.9",
        }],
    }
    if gate_extra:
        gate.update(gate_extra)
    return {
        "milestones": [{
            "id": "M0", "kind": "foundation",
            "decisions": [
                {"id": did, "layer": "architectural", "depends_on": deps}
                for did, deps in decisions.items()
            ],
            "gate": gate,
        }]
    }


# ---------------------------------------------------------------------------
# 增益判据
# ---------------------------------------------------------------------------

def test_gain_above_noise_is_accepted():
    assert campaign.decide_candidate(1.10, 2.0) == campaign.VERDICT_ACCEPT


def test_gain_within_noise_is_reverted():
    assert campaign.decide_candidate(1.01, 2.0) == campaign.VERDICT_REVERT


def test_gain_exactly_at_noise_floor_is_reverted():
    """边界用例：恰好等于噪声底不算超过，必须拒。"""
    assert campaign.decide_candidate(1.02, 2.0) == campaign.VERDICT_REVERT


def test_gain_just_above_noise_floor_is_accepted():
    assert campaign.decide_candidate(1.0201, 2.0) == campaign.VERDICT_ACCEPT


def test_clear_regression_is_flagged():
    assert campaign.decide_candidate(0.90, 2.0) == campaign.VERDICT_REGRESSION


def test_small_regression_within_noise_is_revert_not_regression():
    assert campaign.decide_candidate(0.99, 2.0) == campaign.VERDICT_REVERT


def test_negative_noise_floor_rejected():
    with pytest.raises(ValueError):
        campaign.decide_candidate(1.1, -1.0)


def test_zero_noise_floor_accepts_any_positive_gain():
    assert campaign.decide_candidate(1.0001, 0.0) == campaign.VERDICT_ACCEPT


# ---------------------------------------------------------------------------
# 自我提升终止判据
# ---------------------------------------------------------------------------

def test_promotion_continues_while_gain_clears_noise():
    assert campaign.should_stop_promotion(1.10, 2.0) is False


def test_promotion_stops_when_gain_falls_into_noise():
    assert campaign.should_stop_promotion(1.01, 2.0) is True


def test_promotion_stops_on_regression():
    assert campaign.should_stop_promotion(0.80, 2.0) is True


# ---------------------------------------------------------------------------
# 机制核验：把 profiling 接进裁决
# ---------------------------------------------------------------------------

def test_signature_lookup_finds_value_by_channel():
    value, channel = campaign.lookup_signature_value(
        signature(0.9), "aiv_mte2_wait_ratio")
    assert value == 0.9
    assert "ResourceConflictRatio" in channel


def test_signature_lookup_skips_absent_fields():
    """没采到与值为 0 必须分开 —— 否则缺失会被读成「没有停顿」。"""
    value, _ = campaign.lookup_signature_value(
        signature(present=False), "aiv_mte2_wait_ratio")
    assert value is None


def test_mechanism_line_verified_when_expr_holds():
    line = milestone()["milestones"][0]["gate"]["mechanism_lines"][0]
    out = campaign.verify_mechanism_line(line, signature(0.9))
    assert out["verified"] is True


def test_mechanism_line_refuted_when_expr_fails():
    """机制没落地，说明这次的增益来自别处。"""
    line = milestone()["milestones"][0]["gate"]["mechanism_lines"][0]
    out = campaign.verify_mechanism_line(line, signature(9.9))
    assert out["verified"] is False


def test_mechanism_line_unknown_when_field_absent():
    """无法求值不等于不成立，但也不构成验证。"""
    line = milestone()["milestones"][0]["gate"]["mechanism_lines"][0]
    out = campaign.verify_mechanism_line(line, signature(present=False))
    assert out["verified"] is None
    assert "缺失不等于不成立" in out["reason"]


def test_mechanism_line_unknown_on_unparseable_expr():
    line = {"id": "M1", "anchor": {"expr": "机制应该更好"}}
    assert campaign.verify_mechanism_line(line, signature())["verified"] is None


@pytest.mark.parametrize("op,actual,threshold,expected", [
    ("<", 0.9, 4.2, True), ("<", 9.9, 4.2, False),
    (">", 9.9, 4.2, True), (">", 0.9, 4.2, False),
    ("<=", 4.2, 4.2, True), (">=", 4.2, 4.2, True),
    ("==", 4.2, 4.2, True), ("!=", 4.2, 4.2, False),
])
def test_all_comparators(op, actual, threshold, expected):
    line = {"id": "M1", "anchor": {
        "expr": f"msprof:X.csv:aiv_mte2_wait_ratio {op} {threshold}"}}
    assert campaign.verify_mechanism_line(
        line, signature(actual))["verified"] is expected


def test_mechanism_unknown_without_inputs():
    """没给 milestone/signature 时不替调用方假设「没查就是通过」。"""
    out = campaign.judge_mechanism(None, None, "D0.1")
    assert out["status"] == "unknown"
    assert "未提供" in out["reason"]


def test_mechanism_unknown_when_signature_untrustworthy():
    """签名自判不可信时，据此核验机制没有意义。"""
    out = campaign.judge_mechanism(
        milestone(), signature(trustworthy=False), "D0.1")
    assert out["status"] == "unknown"
    assert "不可信" in out["reason"]


def test_mechanism_unknown_when_decision_not_in_milestone():
    out = campaign.judge_mechanism(milestone(), signature(), "D9.9")
    assert out["status"] == "unknown"


# ---------------------------------------------------------------------------
# 裁决：机制进回路后的四种结果
# ---------------------------------------------------------------------------

def test_verified_mechanism_yields_accept():
    state = fresh_state()
    entry = campaign.apply_result(
        state, result(speedup=1.20), "D0.1",
        milestone_doc=milestone(), signature=signature(0.9))
    assert entry["verdict"] == campaign.VERDICT_ACCEPT
    assert entry["counts_toward_best"] is True
    assert entry["kb_admissible"] is True
    assert state["best"]["gain"] == pytest.approx(1.20)


def test_refuted_mechanism_yields_accept_unverified():
    """核心用例：变快了但机制没落地 —— 保留，但不许影响 best 与知识库。"""
    state = fresh_state()
    entry = campaign.apply_result(
        state, result(speedup=1.20), "D0.1",
        milestone_doc=milestone(), signature=signature(9.9))
    assert entry["verdict"] == campaign.VERDICT_ACCEPT_UNVERIFIED
    assert entry["counts_toward_best"] is False
    assert entry["kb_admissible"] is False
    assert state["best"]["gain"] == 1.0, "未核实的增益不得抬高 best"


def test_unverified_mechanism_yields_accept_unverified():
    state = fresh_state()
    entry = campaign.apply_result(
        state, result(speedup=1.20), "D0.1",
        milestone_doc=milestone(), signature=signature(present=False))
    assert entry["verdict"] == campaign.VERDICT_ACCEPT_UNVERIFIED
    assert state["best"]["gain"] == 1.0


def test_missing_probe_inputs_yield_accept_unverified():
    """不传 signature/milestone 时不能当成核实过。"""
    state = fresh_state()
    entry = campaign.apply_result(state, result(speedup=1.20), "D0.1")
    assert entry["verdict"] == campaign.VERDICT_ACCEPT_UNVERIFIED
    assert entry["kb_admissible"] is False


def test_unverified_does_not_trigger_ship():
    """未核实的增益不得让 campaign 进入可提升状态。"""
    state = fresh_state(target=1.1)
    campaign.apply_result(state, result(speedup=1.50), "D0.1")
    assert state["state"] != campaign.STATE_SHIPPED


def test_verified_triggers_ship():
    state = fresh_state(target=1.1)
    campaign.apply_result(state, result(speedup=1.50), "D0.1",
                          milestone_doc=milestone(), signature=signature(0.9))
    assert state["state"] == campaign.STATE_SHIPPED


def test_precision_failure_reverts_regardless_of_speed():
    state = fresh_state()
    entry = campaign.apply_result(
        state, result(speedup=5.0, precision=False), "D0.1",
        milestone_doc=milestone(), signature=signature(0.9))
    assert entry["verdict"] == campaign.VERDICT_REVERT
    assert "精度" in entry["reason"]
    assert state["best"]["gain"] == 1.0


def test_missing_measurement_reverts():
    state = fresh_state()
    entry = campaign.apply_result(state, {"comparison": {}}, "D0.1")
    assert entry["verdict"] == campaign.VERDICT_REVERT
    assert "无测量支持" in entry["reason"]


def test_gain_computed_from_times_when_speedup_absent():
    state = fresh_state()
    entry = campaign.apply_result(
        state, result(baseline_us=400.0, evolved_us=200.0), "D0.1",
        milestone_doc=milestone(), signature=signature(0.9))
    assert entry["gain"] == pytest.approx(2.0)
    assert entry["verdict"] == campaign.VERDICT_ACCEPT


def test_best_not_downgraded_by_weaker_accept():
    state = fresh_state(target=10.0)
    for speedup in (1.80, 1.10):
        campaign.apply_result(state, result(speedup=speedup), "D0.1",
                              milestone_doc=milestone(), signature=signature(0.9))
    assert state["best"]["gain"] == pytest.approx(1.80)


# ---------------------------------------------------------------------------
# 停滞归因：实现问题不该消耗架构重入机会
# ---------------------------------------------------------------------------

def test_stagnation_counts_only_measured_shortfalls():
    state = fresh_state(threshold=3)
    for _ in range(3):
        campaign.apply_result(state, result(speedup=1.001), "D0.1")
    assert state["state"] == campaign.STATE_STALLED
    assert state["stall_reason"] == campaign.STALL_NO_IMPROVEMENT


def test_precision_failures_do_not_trigger_architecture_reentry():
    """连续编译/精度 bug 是实现问题，不该白费一次架构重入。"""
    state = fresh_state(threshold=3)
    for _ in range(5):
        campaign.apply_result(state, result(speedup=5.0, precision=False), "D0.1")
    assert state["state"] != campaign.STATE_STALLED
    assert state["stagnation"]["since_improvement"] == 0
    assert state["implementation_failures"]["consecutive"] == 5


def test_missing_measurement_counts_as_implementation_failure():
    state = fresh_state(threshold=3)
    for _ in range(3):
        campaign.apply_result(state, {"comparison": {}}, "D0.1")
    assert state["state"] != campaign.STATE_STALLED
    assert state["implementation_failures"]["consecutive"] == 3


def test_accept_resets_both_counters():
    state = fresh_state(threshold=3)
    campaign.apply_result(state, result(speedup=1.001), "D0.1")
    campaign.apply_result(state, result(speedup=5.0, precision=False), "D0.2")
    campaign.apply_result(state, result(speedup=1.30), "D0.3",
                          milestone_doc=milestone(), signature=signature(0.9))
    assert state["stagnation"]["since_improvement"] == 0
    assert state["implementation_failures"]["consecutive"] == 0


# ---------------------------------------------------------------------------
# 外层重入的另外两个口子
# ---------------------------------------------------------------------------

def test_graph_exhaustion_triggers_stall():
    """决策图走穷但没达标 —— 必须有人提示回架构层。"""
    doc = milestone({"D0.1": [], "D0.2": ["D0.1"]})
    state = fresh_state(target=10.0, threshold=99)
    campaign.apply_result(state, result(speedup=1.30), "D0.1",
                          milestone_doc=doc, signature=signature(0.9))
    assert state["state"] == campaign.STATE_IMPLEMENT
    campaign.apply_result(state, result(speedup=1.35), "D0.2",
                          milestone_doc=doc, signature=signature(0.9))
    assert state["state"] == campaign.STATE_STALLED
    assert state["stall_reason"] == campaign.STALL_GRAPH_EXHAUSTED


def test_open_gate_line_triggers_impasse():
    """用两条决策，避免同时触发走穷，单独验证僵持这一条。"""
    doc = milestone({"D0.1": [], "D0.2": ["D0.1"]})
    doc["milestones"][0]["gate"]["correctness_lines"][0]["status"] = "OPEN"
    state = fresh_state(target=10.0, threshold=99)
    campaign.apply_result(state, result(speedup=1.30), "D0.1",
                          milestone_doc=doc, signature=signature(0.9))
    assert state["state"] == campaign.STATE_STALLED
    assert state["stall_reason"] == campaign.STALL_GATE_IMPASSE


def test_multiple_stall_reasons_all_recorded():
    """三个重入口可能同时成立，只报一个会让架构层丢线索。"""
    doc = milestone({"D0.1": []})
    doc["milestones"][0]["gate"]["correctness_lines"][0]["status"] = "OPEN"
    state = fresh_state(target=10.0, threshold=1)
    campaign.apply_result(state, result(speedup=1.001), "D0.1",
                          milestone_doc=doc)
    kinds = {r["kind"] for r in state["stall_reasons"]}
    assert campaign.STALL_GATE_IMPASSE in kinds
    assert campaign.STALL_GRAPH_EXHAUSTED in kinds
    assert campaign.STALL_NO_IMPROVEMENT in kinds
    # 主因取最具体的那个：僵持点名到具体门禁行
    assert state["stall_reason"] == campaign.STALL_GATE_IMPASSE


# ---------------------------------------------------------------------------
# 第四个重入口：机制从未核实
#
# 昇腾上计时与 profiling 必须分两次跑，于是多出一种论文里不存在的失效方式：
# 一直不传 --signature，每轮判 accept_unverified —— 不计入 best 所以永远
# 到不了 shipped，又清零停滞计数所以永远不转 stalled，循环一直空转。
# ---------------------------------------------------------------------------

def test_consecutive_unverified_eventually_stalls():
    state = fresh_state(target=10.0, threshold=99)
    for i in range(3):
        campaign.apply_result(state, result(speedup=1.10), f"D0.{i}")
    assert state["state"] == campaign.STATE_STALLED
    assert state["stall_reason"] == campaign.STALL_MECHANISM_UNVERIFIED


def test_unverified_stall_names_the_missing_flags():
    """提示必须点名缺了什么，否则 agent 还是不知道该补哪一步。"""
    state = fresh_state(target=10.0, threshold=99)
    for i in range(3):
        campaign.apply_result(state, result(speedup=1.10), f"D0.{i}")
    detail = state["stall_reasons"][0]["detail"]
    assert "--signature" in detail
    assert "--milestone" in detail


# ---------------------------------------------------------------------------
# 「没核验」与「被证伪」是两回事
#
# 两者都走 accept_unverified，但该做的事相反：
#   没核验  补 record 的参数即可，课程表没问题
#   被证伪  课程表里对参考行为的解释错了，补参数解决不了
# 混成一条提示会把架构层引向错误的动作 —— 它会去补参数重跑，而真正该改的
# 机制假设没人碰。
# ---------------------------------------------------------------------------

def test_refuted_mechanism_gets_its_own_stall_kind():
    state = fresh_state(target=10.0, threshold=99)
    for _ in range(3):
        campaign.apply_result(state, result(speedup=1.30), "D0.1",
                              milestone_doc=milestone(), signature=signature(9.9))
    assert state["state"] == campaign.STATE_STALLED
    assert state["stall_reason"] == campaign.STALL_MECHANISM_REFUTED


def test_refuted_detail_points_at_the_curriculum_not_the_flags():
    """核心用例：被证伪时不能说「通常是没带 --signature」。"""
    state = fresh_state(target=10.0, threshold=99)
    for _ in range(3):
        campaign.apply_result(state, result(speedup=1.30), "D0.1",
                              milestone_doc=milestone(), signature=signature(9.9))
    detail = state["stall_reasons"][0]["detail"]
    assert "机制核验为假" in detail
    assert "解释错了" in detail
    assert "不是参数没传" in detail


def test_refuted_action_sends_work_to_architecture():
    state = fresh_state(target=10.0, threshold=99)
    for _ in range(3):
        campaign.apply_result(state, result(speedup=1.30), "D0.1",
                              milestone_doc=milestone(), signature=signature(9.9))
    action = campaign.next_action(state)
    assert action["action"] == "re-analyze"
    assert "修正机制假设" in action["reason"]


def test_pure_unverified_action_is_fix_invocation():
    """纯流程缺口不该惊动架构层 —— 标成 re-analyze 会让它白跑一轮分析。"""
    state = fresh_state(target=10.0, threshold=99)
    for _ in range(3):
        campaign.apply_result(state, result(speedup=1.30), "D0.1")
    action = campaign.next_action(state)
    assert action["action"] == "fix-invocation"
    assert "不必改课程表" in action["reason"]


def test_mixed_streak_reports_both_reasons():
    """两种成因并存时都要报，且被证伪排在前面（它是架构信号）。"""
    state = fresh_state(target=10.0, threshold=99)
    campaign.apply_result(state, result(speedup=1.30), "D0.1")
    campaign.apply_result(state, result(speedup=1.30), "D0.1")
    campaign.apply_result(state, result(speedup=1.30), "D0.1",
                          milestone_doc=milestone(), signature=signature(9.9))
    kinds = [r["kind"] for r in state["stall_reasons"]]
    assert campaign.STALL_MECHANISM_REFUTED in kinds
    assert campaign.STALL_MECHANISM_UNVERIFIED in kinds
    assert kinds.index(campaign.STALL_MECHANISM_REFUTED) < \
        kinds.index(campaign.STALL_MECHANISM_UNVERIFIED)
    assert campaign.next_action(state)["action"] == "re-analyze"


def test_streak_breakdown_tracked():
    state = fresh_state(target=10.0, threshold=99)
    campaign.apply_result(state, result(speedup=1.30), "D0.1")
    campaign.apply_result(state, result(speedup=1.30), "D0.1",
                          milestone_doc=milestone(), signature=signature(9.9))
    unver = state["mechanism_unverified"]
    assert unver["streak_unknown"] == 1
    assert unver["streak_refuted"] == 1
    assert unver["consecutive"] == 2


def test_verified_accept_clears_the_breakdown():
    state = fresh_state(target=10.0, threshold=99)
    campaign.apply_result(state, result(speedup=1.30), "D0.1",
                          milestone_doc=milestone(), signature=signature(9.9))
    campaign.apply_result(state, result(speedup=1.40), "D0.1",
                          milestone_doc=milestone(), signature=signature(0.9))
    unver = state["mechanism_unverified"]
    assert unver["consecutive"] == 0
    assert unver["streak_refuted"] == 0
    assert unver["streak_unknown"] == 0


def test_untrustworthy_signature_counts_as_unknown_not_refuted():
    """签名自判不可信时无从判断机制成没成立，不能算证伪。"""
    state = fresh_state(target=10.0, threshold=99)
    campaign.apply_result(
        state, result(speedup=1.30), "D0.1",
        milestone_doc=milestone(), signature=signature(9.9, trustworthy=False))
    unver = state["mechanism_unverified"]
    assert unver["streak_unknown"] == 1
    assert unver["streak_refuted"] == 0


def test_verified_accept_resets_unverified_counter():
    state = fresh_state(target=10.0, threshold=99)
    campaign.apply_result(state, result(speedup=1.10), "D0.1")
    campaign.apply_result(state, result(speedup=1.10), "D0.2")
    assert state["mechanism_unverified"]["consecutive"] == 2
    campaign.apply_result(state, result(speedup=1.20), "D0.1",
                          milestone_doc=milestone(), signature=signature(0.9))
    assert state["mechanism_unverified"]["consecutive"] == 0


def test_unverified_stall_takes_priority():
    """机制没核实时 best 卡在 1.0、其他信号都是降级的，应排最前。"""
    doc = milestone({"D0.1": []})
    doc["milestones"][0]["gate"]["correctness_lines"][0]["status"] = "OPEN"
    state = fresh_state(target=10.0, threshold=1)
    for _ in range(3):
        campaign.apply_result(state, result(speedup=1.10), "D0.1",
                              milestone_doc=doc)
    assert state["stall_reason"] == campaign.STALL_MECHANISM_UNVERIFIED


def test_reverts_do_not_reset_unverified_counter():
    """回滚既不确认也不否认 profiling 是否接上，应保持中立。"""
    state = fresh_state(target=10.0, threshold=99)
    campaign.apply_result(state, result(speedup=1.10), "D0.1")
    campaign.apply_result(state, result(speedup=1.001), "D0.2")
    assert state["mechanism_unverified"]["consecutive"] == 1


# ---------------------------------------------------------------------------
# 走穷判定不能把管理条目算成「试过」
# ---------------------------------------------------------------------------

def test_reopen_entry_not_counted_as_tried(tmp_path):
    """刚重开的叶子决策还没重做，不该立刻被判走穷。"""
    doc = milestone({"D0.1": [], "D0.2": []})
    state = fresh_state(target=10.0, threshold=99)
    for did in ("D0.1", "D0.2"):
        campaign.apply_result(state, result(speedup=1.10), did,
                              milestone_doc=doc, signature=signature(0.9))

    state_file = tmp_path / "s.json"
    milestone_file = tmp_path / "m.json"
    campaign.save_state(state_file, state)
    milestone_file.write_text(json.dumps(doc), encoding="utf-8")

    import argparse
    campaign.cmd_reopen(argparse.Namespace(
        state=str(state_file), milestone=str(milestone_file),
        decision="D0.2", reason="", as_json=False))

    reloaded = campaign.load_state(state_file)
    campaign.apply_result(reloaded, result(speedup=1.10), "D0.1",
                          milestone_doc=doc, signature=signature(0.9))
    assert reloaded["state"] != campaign.STATE_STALLED, \
        "D0.2 刚重开尚未重做，决策图没有走穷"


@pytest.mark.parametrize("verdict", ["reopen", "promote", "stop", "noise_floor"])
def test_management_entries_excluded_from_tried(verdict):
    doc = milestone({"D0.1": [], "D0.2": []})
    state = fresh_state(target=10.0, threshold=99)
    campaign.apply_result(state, result(speedup=1.10), "D0.1",
                          milestone_doc=doc, signature=signature(0.9))
    state["ledger"].append({"decision": "D0.2", "verdict": verdict})
    assert campaign.graph_exhausted(state, doc) is False


def test_genuine_exhaustion_still_detected():
    doc = milestone({"D0.1": [], "D0.2": []})
    state = fresh_state(target=10.0, threshold=99)
    for did in ("D0.1", "D0.2"):
        campaign.apply_result(state, result(speedup=1.10), did,
                              milestone_doc=doc, signature=signature(0.9))
    assert campaign.graph_exhausted(state, doc) is True


def test_stall_flags_cleared_on_recovery():
    doc = milestone({"D0.1": [], "D0.2": ["D0.1"]})
    state = fresh_state(target=10.0, threshold=1)
    campaign.apply_result(state, result(speedup=1.001), "D0.1",
                          milestone_doc=doc)
    assert state["state"] == campaign.STATE_STALLED
    campaign.apply_result(state, result(speedup=1.30), "D0.2",
                          milestone_doc=milestone({"D0.1": [], "D0.2": ["D0.1"],
                                                   "D0.3": []}),
                          signature=signature(0.9))
    assert state["state"] == campaign.STATE_IMPLEMENT
    assert "stall_reason" not in state


def test_deferred_to_nonexistent_lever_triggers_impasse():
    """DEFERRED 指向不存在的手段，该测试实际是被丢弃了。"""
    doc = milestone({"D0.1": []})
    doc["milestones"][0]["gate"]["mechanism_lines"][0].update(
        status="DEFERRED", downstream_lever="D9.9")
    assert campaign.gate_impasse(doc) is not None
    assert "被丢弃" in campaign.gate_impasse(doc)


def test_deferred_to_existing_lever_is_not_impasse():
    doc = milestone({"D0.1": [], "D2.1": ["D0.1"]})
    doc["milestones"][0]["gate"]["mechanism_lines"][0].update(
        status="DEFERRED", downstream_lever="D2.1")
    assert campaign.gate_impasse(doc) is None


def test_target_reached_wins_over_stall():
    """达标优先于一切停滞判定。"""
    doc = milestone({"D0.1": []})
    doc["milestones"][0]["gate"]["correctness_lines"][0]["status"] = "OPEN"
    state = fresh_state(target=1.1)
    campaign.apply_result(state, result(speedup=1.50), "D0.1",
                          milestone_doc=doc, signature=signature(0.9))
    assert state["state"] == campaign.STATE_SHIPPED


@pytest.mark.parametrize("phase,expected", [
    (campaign.STATE_ANALYZE, "analyze"),
    (campaign.STATE_IMPLEMENT, "implement"),
    (campaign.STATE_STALLED, "re-analyze"),
    (campaign.STATE_SHIPPED, "promote"),
    (campaign.STATE_STOPPED, "stop"),
])
def test_next_action_per_state(phase, expected):
    assert campaign.next_action({"state": phase})["action"] == expected


def test_stalled_action_mentions_reopening_architecture():
    action = campaign.next_action({
        "state": campaign.STATE_STALLED,
        "stall_reason": campaign.STALL_NO_IMPROVEMENT,
        "stagnation": {"since_improvement": 3},
    })
    assert "reopen" in action["reason"]
    assert "M0" in action["reason"]


# ---------------------------------------------------------------------------
# 重启范围：依赖图反向闭包
# ---------------------------------------------------------------------------

def test_dependents_closure_is_transitive():
    doc = milestone({"D0.1": [], "D0.2": ["D0.1"], "D0.3": ["D0.2"],
                     "D0.4": []})
    assert campaign.dependents_closure(doc, "D0.1") == {"D0.2", "D0.3"}


def test_dependents_closure_excludes_root():
    doc = milestone({"D0.1": [], "D0.2": ["D0.1"]})
    assert "D0.1" not in campaign.dependents_closure(doc, "D0.1")


def test_dependents_closure_empty_for_leaf():
    doc = milestone({"D0.1": [], "D0.2": ["D0.1"]})
    assert campaign.dependents_closure(doc, "D0.2") == set()


def test_dependents_closure_unknown_decision_errors():
    with pytest.raises(campaign.CampaignError, match="没有决策"):
        campaign.dependents_closure(milestone(), "D9.9")


def test_reopen_invalidates_downstream(tmp_path):
    """重开 M0 的决策时，建立在它之上的实现必须一并作废。"""
    doc = milestone({"D0.1": [], "D0.2": ["D0.1"], "D0.3": []})
    state_file = tmp_path / "state.json"
    state = fresh_state(target=10.0, threshold=99)
    for did in ("D0.1", "D0.2", "D0.3"):
        campaign.apply_result(state, result(speedup=1.30), did,
                              milestone_doc=doc, signature=signature(0.9))
    campaign.save_state(state_file, state)

    milestone_file = tmp_path / "m.json"
    milestone_file.write_text(json.dumps(doc), encoding="utf-8")

    import argparse
    code = campaign.cmd_reopen(argparse.Namespace(
        state=str(state_file), milestone=str(milestone_file),
        decision="D0.1", reason="拓扑决策有缺陷", as_json=False))
    assert code == 0

    reloaded = json.loads(state_file.read_text(encoding="utf-8"))
    by_decision = {}
    for entry in reloaded["ledger"]:
        if entry.get("decision") and entry.get("verdict") != "reopen":
            by_decision[entry["decision"]] = entry
    assert by_decision["D0.1"]["invalidated"] is True
    assert by_decision["D0.2"]["invalidated"] is True, "下游必须一并作废"
    assert "invalidated" not in by_decision["D0.3"], "无关决策不该被牵连"
    assert reloaded["state"] == campaign.STATE_ANALYZE


def test_reopen_recomputes_best(tmp_path):
    """best 必须按剩余有效条目重算，gain 与 time_us 要一起取。"""
    doc = milestone({"D0.1": [], "D0.2": ["D0.1"], "D0.3": []})
    state = fresh_state(target=10.0, threshold=99)
    campaign.apply_result(
        state, result(speedup=1.80, evolved_us=250.0), "D0.2",
        milestone_doc=doc, signature=signature(0.9))
    campaign.apply_result(
        state, result(speedup=1.20, evolved_us=380.0), "D0.3",
        milestone_doc=doc, signature=signature(0.9))
    assert state["best"]["gain"] == pytest.approx(1.80)
    assert state["best"]["time_us"] == pytest.approx(250.0)

    state_file = tmp_path / "s.json"
    milestone_file = tmp_path / "m.json"
    campaign.save_state(state_file, state)
    milestone_file.write_text(json.dumps(doc), encoding="utf-8")

    import argparse
    campaign.cmd_reopen(argparse.Namespace(
        state=str(state_file), milestone=str(milestone_file),
        decision="D0.1", reason="", as_json=False))

    reloaded = json.loads(state_file.read_text(encoding="utf-8"))
    assert reloaded["best"]["gain"] == pytest.approx(1.20), \
        "1.80 来自已作废的 D0.2，必须退回到 D0.3 的 1.20"
    assert reloaded["best"]["time_us"] == pytest.approx(380.0), \
        "time_us 要跟着 gain 一起重算，不能被抹成 None"


def test_reopen_resets_all_counters(tmp_path):
    """三个计数器一起归零，与 promote 的处理对齐。"""
    doc = milestone({"D0.1": []})
    state = fresh_state(target=10.0, threshold=99)
    campaign.apply_result(state, result(speedup=1.10), "D0.1")          # unverified
    campaign.apply_result(state, result(speedup=1.001), "D0.1")         # stagnation
    campaign.apply_result(state, result(precision=False, speedup=2.0), "D0.1")
    assert state["mechanism_unverified"]["consecutive"] > 0

    state_file = tmp_path / "s.json"
    milestone_file = tmp_path / "m.json"
    campaign.save_state(state_file, state)
    milestone_file.write_text(json.dumps(doc), encoding="utf-8")

    import argparse
    campaign.cmd_reopen(argparse.Namespace(
        state=str(state_file), milestone=str(milestone_file),
        decision="D0.1", reason="", as_json=False))

    reloaded = json.loads(state_file.read_text(encoding="utf-8"))
    assert reloaded["stagnation"]["since_improvement"] == 0
    assert reloaded["implementation_failures"]["consecutive"] == 0
    assert reloaded["mechanism_unverified"]["consecutive"] == 0


def test_ledger_entry_carries_time_us():
    """台账要能自给自足地重算 best，所以 time_us 必须落在条目里。"""
    state = fresh_state()
    entry = campaign.apply_result(
        state, result(speedup=1.30, evolved_us=350.0), "D0.1",
        milestone_doc=milestone(), signature=signature(0.9))
    assert entry["time_us"] == pytest.approx(350.0)


# ---------------------------------------------------------------------------
# 死路表登记
# ---------------------------------------------------------------------------

def test_nogo_command_is_copy_pasteable():
    state = fresh_state()
    entry = campaign.apply_result(state, result(speedup=1.001), "D0.1",
                                  symptom="MTE2 占比过高")
    command = campaign.nogo_command(state, entry)
    assert "hwkb.py nogo add" in command
    assert "MTE2 占比过高" in command
    assert "D0.1" in command
    assert "--arch DAV_2201" in command


def test_nogo_written_directly(tmp_path):
    """靠 agent 自觉登记是不牢的 —— 连续失败时最容易跳过记账。"""
    state = fresh_state()
    entry = campaign.apply_result(state, result(speedup=1.001), "D0.1",
                                  symptom="MTE2 占比过高")
    nogo = tmp_path / "nogo.json"
    assert campaign.write_nogo(nogo, state, entry) == 1

    doc = json.loads(nogo.read_text(encoding="utf-8"))
    assert doc["entries"][0]["symptom"] == "MTE2 占比过高"
    assert doc["entries"][0]["tried"] == "D0.1"
    assert doc["entries"][0]["input_shape"] == "B=8,S=4096,fp16"


def test_nogo_appends_to_existing(tmp_path):
    nogo = tmp_path / "nogo.json"
    nogo.write_text(json.dumps({"entries": [{"symptom": "旧的"}]}),
                    encoding="utf-8")
    state = fresh_state()
    entry = campaign.apply_result(state, result(speedup=1.001), "D0.1")
    assert campaign.write_nogo(nogo, state, entry) == 2


# ---------------------------------------------------------------------------
# 知识库的返回路径
# ---------------------------------------------------------------------------

def test_recipes_recorded_in_ledger():
    state = fresh_state()
    entry = campaign.apply_result(
        state, result(speedup=1.30), "D0.1",
        milestone_doc=milestone(), signature=signature(0.9),
        recipes_used=["R-2201-001"], recipes_invented=["R-2201-009"])
    assert entry["recipes_used"] == ["R-2201-001"]
    assert entry["recipes_invented"] == ["R-2201-009"]


def test_recipe_summary_separates_foreign_and_invented():
    """没有这份汇总就无法回答「知识库到底有没有在帮忙」。"""
    state = fresh_state(target=10.0)
    campaign.apply_result(state, result(speedup=1.30), "D0.1",
                          milestone_doc=milestone(), signature=signature(0.9),
                          recipes_used=["R-A"], recipes_invented=["R-X"])
    campaign.apply_result(state, result(speedup=1.40), "D0.2",
                          milestone_doc=milestone(), signature=signature(0.9),
                          recipes_used=["R-A", "R-B"])
    summary = campaign.summarise_recipes(state)
    assert summary["foreign_recipes"] == {"R-A": 2, "R-B": 1}
    assert summary["invented_recipes"] == {"R-X": 1}
    assert summary["accepted_total"] == 2
    assert summary["foreign_assist_rate"] == pytest.approx(1.0)


def test_recipe_summary_skips_invalidated():
    state = fresh_state(target=10.0)
    campaign.apply_result(state, result(speedup=1.30), "D0.1",
                          recipes_used=["R-A"])
    state["ledger"][0]["invalidated"] = True
    assert campaign.summarise_recipes(state)["foreign_recipes"] == {}


def test_foreign_assist_rate_zero_without_recipes():
    state = fresh_state(target=10.0)
    campaign.apply_result(state, result(speedup=1.30), "D0.1")
    assert campaign.summarise_recipes(state)["foreign_assist_rate"] == 0.0


# ---------------------------------------------------------------------------
# campaign 身份：input shape
# ---------------------------------------------------------------------------

def test_input_shape_stored_in_state():
    assert fresh_state()["input_shape"] == "B=8,S=4096,fp16"


def test_input_shape_recorded_on_every_entry():
    """一份 state 跨 shape 复用会把不同 shape 的结果混进同一个 best。"""
    state = fresh_state()
    entry = campaign.apply_result(state, result(speedup=1.30), "D0.1")
    assert entry["input_shape"] == "B=8,S=4096,fp16"


# ---------------------------------------------------------------------------
# 预算
# ---------------------------------------------------------------------------

def test_budget_counts_every_record():
    state = fresh_state(budget=3)
    campaign.apply_result(state, result(speedup=1.001), "D0.1")
    assert state["budget"]["used"] == 1


def test_budget_exhaustion_stops_campaign():
    state = fresh_state(budget=2, threshold=99)
    for _ in range(2):
        campaign.apply_result(state, result(speedup=1.001), "D0.1")
    assert state["state"] == campaign.STATE_STOPPED
    assert "预算耗尽" in state["stop_reason"]


def test_unlimited_budget_never_stops():
    state = fresh_state(threshold=99)
    for _ in range(10):
        campaign.apply_result(state, result(speedup=1.001), "D0.1")
    assert state["state"] != campaign.STATE_STOPPED


# ---------------------------------------------------------------------------
# 端到端 CLI
# ---------------------------------------------------------------------------

def run(argv):
    import contextlib
    import io
    saved = sys.argv
    sys.argv = ["campaign.py"] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                return campaign.main(), buf.getvalue()
            except SystemExit as exc:
                return int(exc.code or 0), buf.getvalue()
    finally:
        sys.argv = saved


def init_argv(state_file, **kw):
    argv = ["--state", str(state_file), "init",
            "--operator", "demo_op", "--input-shape", "B=8,S=4096",
            "--arch", "DAV_2201",
            "--reference-kind", "builtin_aclnn", "--reference-id", "aclnnDemo",
            "--noise-floor", str(kw.get("noise", 2.0))]
    if "target" in kw:
        argv += ["--target-gain", str(kw["target"])]
    if "budget" in kw:
        argv += ["--budget", str(kw["budget"])]
    return argv


def test_cli_init_requires_input_shape(tmp_path):
    code, _ = run(["--state", str(tmp_path / "s.json"), "init",
                   "--operator", "demo_op", "--arch", "DAV_2201",
                   "--reference-kind", "builtin_aclnn",
                   "--reference-id", "aclnnDemo", "--noise-floor", "2.0"])
    assert code == 2


# ---------------------------------------------------------------------------
# --state 的位置
#
# 只挂顶层时，argparse 要求全局参数写在子命令之前，而人和 agent 的直觉
# 都是写在后面。写错只报「unrecognized arguments」，不说该放哪 ——
# agent 拿到这个报错往往不去调位置，而是转而自己判断，可这套设计的要害
# 恰恰是不让 agent 自己裁决。
# ---------------------------------------------------------------------------

def test_state_accepted_after_subcommand(tmp_path):
    state_file = tmp_path / "s.json"
    code, _ = run(["init", "--state", str(state_file),
                   "--operator", "demo_op", "--input-shape", "B=8",
                   "--arch", "DAV_2201", "--reference-kind", "builtin_aclnn",
                   "--reference-id", "aclnnDemo", "--noise-floor", "2.0"])
    assert code == 0
    assert state_file.exists()


def test_state_accepted_before_subcommand(tmp_path):
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0
    assert state_file.exists()


def test_subcommand_state_wins_when_both_given(tmp_path):
    """两处都给时以子命令那个为准，顶层默认值不得悄悄覆盖回去。"""
    real = tmp_path / "real.json"
    assert run(init_argv(real))[0] == 0
    code, out = run(["--state", str(tmp_path / "wrong.json"),
                     "status", "--state", str(real)])
    assert code == 0
    assert "demo_op" in out


@pytest.mark.parametrize("command", [
    "status", "record", "reopen", "set-noise-floor", "set-budget",
    "promote", "ledger",
])
def test_every_subcommand_accepts_state(command):
    """新增子命令时容易漏挂，这里逐个兜住。"""
    parser = campaign.build_parser()
    sub = [a for a in parser._actions
           if isinstance(a, argparse._SubParsersAction)][0]
    options = set()
    for action in sub.choices[command]._actions:
        options.update(action.option_strings)
    assert "--state" in options, f"{command} 没有挂 --state"
    assert "--json" in options, f"{command} 没有挂 --json"


def test_recovery_hint_is_copy_pasteable(tmp_path):
    """程序打印的恢复命令必须照抄就能跑，否则 agent 会开始猜。"""
    state_file = tmp_path / "s.json"
    run(init_argv(state_file))
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["noise_floor_status"] = campaign.NOISE_PENDING
    state_file.write_text(json.dumps(state), encoding="utf-8")

    res = tmp_path / "ev.json"
    res.write_text(json.dumps(result(speedup=1.30)), encoding="utf-8")
    _, out = run(["--state", str(state_file), "record",
                  "--result", str(res), "--decision", "D1.1"])

    hint = [ln for ln in out.splitlines() if "set-noise-floor" in ln][0]
    argv = hint.strip().replace("campaign.py ", "", 1).split()
    # 把 <实测值> 这个占位符换成真实数值，其余原样照抄
    argv = ["2.7" if a.startswith("<") else a for a in argv]
    assert run(argv)[0] == 0, f"恢复提示照抄跑不通: {hint}"


def test_cli_init_then_status(tmp_path):
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0
    code, out = run(["--state", str(state_file), "status"])
    assert code == 0
    assert "B=8,S=4096" in out


def test_cli_init_refuses_overwrite(tmp_path):
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0
    assert run(init_argv(state_file))[0] == 2
    assert run(init_argv(state_file) + ["--force"])[0] == 0


def test_cli_warns_on_zero_noise_floor(tmp_path):
    _, out = run(init_argv(tmp_path / "s.json", noise=0))
    assert "警告" in out


def test_cli_record_prints_nogo_hint_on_revert(tmp_path):
    state_file = tmp_path / "s.json"
    run(init_argv(state_file))
    res = tmp_path / "ev.json"
    res.write_text(json.dumps(result(speedup=1.001)), encoding="utf-8")
    _, out = run(["--state", str(state_file), "record", "--result", str(res),
                  "--decision", "D0.1", "--symptom", "MTE2 高"])
    assert "hwkb.py nogo add" in out


def test_cli_record_writes_nogo_when_store_given(tmp_path):
    state_file = tmp_path / "s.json"
    nogo = tmp_path / "nogo.json"
    run(init_argv(state_file))
    res = tmp_path / "ev.json"
    res.write_text(json.dumps(result(speedup=1.001)), encoding="utf-8")
    run(["--state", str(state_file), "record", "--result", str(res),
         "--decision", "D0.1", "--symptom", "MTE2 高",
         "--nogo-store", str(nogo)])
    assert json.loads(nogo.read_text(encoding="utf-8"))["entries"]


def test_cli_record_reports_mechanism(tmp_path):
    state_file = tmp_path / "s.json"
    run(init_argv(state_file, target=10.0))
    res = tmp_path / "ev.json"
    res.write_text(json.dumps(result(speedup=1.30)), encoding="utf-8")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(signature(0.9)), encoding="utf-8")
    mil = tmp_path / "m.json"
    mil.write_text(json.dumps(milestone()), encoding="utf-8")

    _, out = run(["--state", str(state_file), "record", "--result", str(res),
                  "--decision", "D0.1", "--signature", str(sig),
                  "--milestone", str(mil)])
    assert "机制核验" in out
    assert "成立" in out


def test_cli_warns_when_mechanism_unverified(tmp_path):
    state_file = tmp_path / "s.json"
    run(init_argv(state_file, target=10.0))
    res = tmp_path / "ev.json"
    res.write_text(json.dumps(result(speedup=1.30)), encoding="utf-8")
    _, out = run(["--state", str(state_file), "record", "--result", str(res),
                  "--decision", "D0.1"])
    assert "不计入 best" in out


def test_cli_promote_marks_noise_floor_pending(tmp_path):
    """新参考是另一个二进制，噪声底必须重测。"""
    state_file = tmp_path / "s.json"
    run(init_argv(state_file, target=1.1))
    res = tmp_path / "ev.json"
    res.write_text(json.dumps(result(speedup=1.25)), encoding="utf-8")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(signature(0.9)), encoding="utf-8")
    mil = tmp_path / "m.json"
    mil.write_text(json.dumps(milestone()), encoding="utf-8")
    run(["--state", str(state_file), "record", "--result", str(res),
         "--decision", "D0.1", "--signature", str(sig), "--milestone", str(mil)])

    code, out = run(["--state", str(state_file), "promote",
                     "--artifact", "/tmp/best"])
    assert code == 0
    promoted = json.loads(state_file.read_text(encoding="utf-8"))
    assert promoted["generation"] == 1
    assert promoted["noise_floor_status"] == campaign.NOISE_PENDING
    assert "重测" in out


def test_cli_record_refused_while_noise_floor_pending(tmp_path):
    state_file = tmp_path / "s.json"
    run(init_argv(state_file))
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["noise_floor_status"] = campaign.NOISE_PENDING
    state_file.write_text(json.dumps(state), encoding="utf-8")

    res = tmp_path / "ev.json"
    res.write_text(json.dumps(result(speedup=1.30)), encoding="utf-8")
    code, out = run(["--state", str(state_file), "record",
                     "--result", str(res), "--decision", "D1.1"])
    assert code == 1
    assert "重测" in out


def test_cli_set_noise_floor_clears_pending(tmp_path):
    state_file = tmp_path / "s.json"
    run(init_argv(state_file))
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["noise_floor_status"] = campaign.NOISE_PENDING
    state_file.write_text(json.dumps(state), encoding="utf-8")

    assert run(["--state", str(state_file), "set-noise-floor",
                "--value", "2.7"])[0] == 0
    reloaded = json.loads(state_file.read_text(encoding="utf-8"))
    assert reloaded["noise_floor_pct"] == 2.7
    assert reloaded["noise_floor_status"] == campaign.NOISE_MEASURED


def test_cli_set_budget_resumes_stopped_campaign(tmp_path):
    state_file = tmp_path / "s.json"
    run(init_argv(state_file, budget=1))
    res = tmp_path / "ev.json"
    res.write_text(json.dumps(result(speedup=1.001)), encoding="utf-8")
    run(["--state", str(state_file), "record", "--result", str(res),
         "--decision", "D0.1"])
    assert json.loads(state_file.read_text())["state"] == campaign.STATE_STOPPED

    run(["--state", str(state_file), "set-budget", "--max-records", "10"])
    assert json.loads(state_file.read_text())["state"] != campaign.STATE_STOPPED


def test_cli_ledger_recipes_summary(tmp_path):
    state_file = tmp_path / "s.json"
    run(init_argv(state_file, target=10.0))
    res = tmp_path / "ev.json"
    res.write_text(json.dumps(result(speedup=1.30)), encoding="utf-8")
    run(["--state", str(state_file), "record", "--result", str(res),
         "--decision", "D0.1", "--recipe", "R-2201-001"])
    _, out = run(["--state", str(state_file), "ledger", "--recipes"])
    assert "R-2201-001" in out
    assert "外来经验" in out


def test_cli_record_refused_after_stop(tmp_path):
    state_file = tmp_path / "s.json"
    run(init_argv(state_file))
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["state"] = campaign.STATE_STOPPED
    state_file.write_text(json.dumps(state), encoding="utf-8")
    res = tmp_path / "ev.json"
    res.write_text(json.dumps(result(speedup=2.0)), encoding="utf-8")
    assert run(["--state", str(state_file), "record", "--result", str(res),
                "--decision", "D1.1"])[0] == 1


def test_corrupt_state_file_reports_clearly(tmp_path):
    state_file = tmp_path / "s.json"
    state_file.write_text("这不是 JSON", encoding="utf-8")
    code, out = run(["--state", str(state_file), "status"])
    assert code == 2
    assert "损坏" in out


def test_corrupt_milestone_reports_clearly(tmp_path):
    bad = tmp_path / "m.json"
    bad.write_text("不是 JSON", encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="不是合法 JSON"):
        campaign.load_milestone(bad)


# ---------------------------------------------------------------------------
# 噪声底
# ---------------------------------------------------------------------------

def test_noise_floor_needs_minimum_samples():
    with pytest.raises(noise_floor.NoiseFloorError, match="样本不足"):
        noise_floor.compute([1.0, 1.01])


def test_noise_floor_on_identical_readings_is_zero():
    assert noise_floor.compute([1.0, 1.0, 1.0])["floor_pct"] == pytest.approx(0.0)


def test_noise_floor_grows_with_spread():
    tight = noise_floor.compute([1.00, 1.01, 0.99])
    loose = noise_floor.compute([1.00, 1.20, 0.80])
    assert loose["floor_pct"] > tight["floor_pct"]


def test_noise_floor_uses_conservative_of_two_statistics():
    out = noise_floor.compute([1.0, 1.0, 1.0, 1.0, 1.3])
    assert out["floor_pct"] == pytest.approx(
        max(out["cv_pct"], out["half_range_pct"]))


def test_noise_floor_warns_when_environment_noisy():
    out = noise_floor.compute([1.0, 1.5, 0.6])
    assert any("不稳定" in w for w in out["warnings"])


def test_noise_floor_ignores_invalid_readings():
    with pytest.raises(noise_floor.NoiseFloorError):
        noise_floor.compute([1.0, -1.0, 0.0])


def test_ratio_extracted_from_speedup():
    assert noise_floor.ratio_from_result(
        {"comparison": {"speedup": 1.5}}) == pytest.approx(1.5)


def test_ratio_computed_from_times():
    assert noise_floor.ratio_from_result(
        {"baseline": {"time_us": 400}, "evolved": {"time_us": 200}}
    ) == pytest.approx(2.0)


def test_ratio_none_when_unavailable():
    assert noise_floor.ratio_from_result({"comparison": {}}) is None
    assert noise_floor.ratio_from_result({}) is None


def test_ratio_none_on_zero_time():
    assert noise_floor.ratio_from_result(
        {"baseline": {"time_us": 0}, "evolved": {"time_us": 200}}) is None


def test_cv_extracted_from_top_level():
    assert noise_floor.extract_cv_values({"cv_pct": 1.8}) == [1.8]


def test_cv_extracted_from_both_sides():
    values = noise_floor.extract_cv_values({
        "baseline": {"cv_pct": 1.2}, "evolved": {"cv_pct": 2.4}})
    assert sorted(values) == [1.2, 2.4]


def test_cv_extracted_from_shape_results():
    values = noise_floor.extract_cv_values({
        "shape_results": {"target": [{"cv_pct": 1.0}, {"cv_pct": 3.0}],
                          "generalization": [{"cv_pct": 2.0}]}})
    assert sorted(values) == [1.0, 2.0, 3.0]


def test_cv_floor_takes_the_worst_reading():
    out = noise_floor.compute_from_cv([1.0, 3.5, 2.0])
    assert out["floor_pct"] == pytest.approx(3.5)
    assert out["method"] == "cv_pct"


def test_cv_floor_errors_when_absent():
    with pytest.raises(noise_floor.NoiseFloorError, match="没有 cv_pct"):
        noise_floor.compute_from_cv([])


def test_cv_floor_warns_when_noisy():
    out = noise_floor.compute_from_cv([12.0])
    assert any("降频" in w for w in out["warnings"])


def test_cv_ignores_malformed_entries():
    values = noise_floor.extract_cv_values({
        "cv_pct": "不是数字", "baseline": "不是对象",
        "shape_results": {"target": ["不是对象", {"cv_pct": 2.0}]}})
    assert values == [2.0]


def test_spread_method_is_labelled():
    assert noise_floor.compute([1.0, 1.01, 0.99])["method"] == "ratio_spread"


def test_measured_floor_drives_acceptance():
    """两个脚本的接缝：噪声底不是常数，是量出来的。"""
    measured = noise_floor.compute([1.00, 1.02, 0.98, 1.01, 0.99])
    floor = measured["floor_pct"]
    assert campaign.decide_candidate(
        1.0 + floor / 200.0, floor) != campaign.VERDICT_ACCEPT
    assert campaign.decide_candidate(
        1.0 + floor / 50.0, floor) == campaign.VERDICT_ACCEPT
