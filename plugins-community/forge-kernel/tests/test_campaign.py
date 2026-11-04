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

#: 参考实现的同字段读数。机制行要靠它做差 —— 没有基线就只是个阈值比较，
#: 拦不住「这个 kernel 的停顿本来就低」。
REFERENCE_STALL = 4.2

CHANNEL = "msprof:ResourceConflictRatio.csv:aiv_mte2_wait_ratio"


def write_clean_fence(tmp_path, name="fence.json"):
    path = tmp_path / name
    path.write_text(json.dumps(clean_fence()), encoding="utf-8")
    return path


def misread_reference():
    """一份把参考读错了的课程表 + 与之配套的 state。

    课程表声称参考在 `< 1.0`，而参考快照里那个通道实测 18.0 —— 差出
    1.25 倍带外。这不只是记错一个数：实现会被要求去追一个参考自己都
    达不到的目标。这才是「课程表缺陷」，该重开那条决策。

    对照组是默认 fixture（声称 4.2、参考 4.2、实现 9.9）：那是论文那个
    例子本身 —— 声称做到了而停顿仍高 ⇒ **实现退化**，该补经验卡片而不是
    改课程表。两者判然不同，所以要有两套 fixture。
    """
    state = fresh_state(target=10.0, threshold=99, ref_stall=18.0)
    doc = milestone(mechanism_expr=f"{CHANNEL} < 1.0")
    return state, doc


def clean_fence(files_scanned=3):
    """一份干净的围栏报告（fence_check.py --json 的形状）。"""
    return {"findings": [], "files_scanned": files_scanned}


def apply(state, doc, decision, **kw):
    """走一次 record 判定，默认带一份干净的围栏报告。

    为什么默认带：真实流程里围栏**总是**跑（静态扫描，不花机时），所以
    「带着干净围栏」才是默认情形。让 96 个调用点各自穿一遍这个参数，只会
    把噪声加进每个用例而不增加覆盖 —— 围栏自身的行为由下面 test_fence_*
    那组用例直接打 campaign.apply_result 来盯。

    想测「没有围栏证据」时直接调 campaign.apply_result，别走这个包装。
    """
    kw.setdefault("fence_report", clean_fence())
    return campaign.apply_result(state, doc, decision, **kw)


def fresh_state(**kw):
    ref = kw.get("ref_stall", REFERENCE_STALL)
    state = campaign.new_state(
        operator="demo_op", input_shape="B=8,S=4096,fp16", arch="DAV_2201",
        reference={"kind": "builtin_aclnn", "id": "aclnnDemo", "sealed": True},
        noise_floor_pct=kw.get("noise", 2.0),
        target_gain=kw.get("target", 1.5),
        stagnation_threshold=kw.get("threshold", 3),
        budget=kw.get("budget"),
        reference_snapshot=({CHANNEL: ref} if ref is not None else None),
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
                    "channel": CHANNEL,
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


def baseline(value=REFERENCE_STALL):
    return {CHANNEL: value}


def test_mechanism_line_verified_when_expr_holds():
    """阈值成立 + 字段朝声称方向移动过 = 机制确实落地。"""
    line = milestone()["milestones"][0]["gate"]["mechanism_lines"][0]
    out = campaign.verify_mechanism_line(line, signature(0.9), baseline())
    assert out["verified"] is True
    assert out["moved_pct"] < 0


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


@pytest.mark.parametrize("op,previous,actual,threshold,expected", [
    # 基线都取得离 actual 足够远，把移动量这一项满足掉，单独验阈值逻辑
    ("<", 9.9, 0.9, 4.2, True), ("<", 1.0, 9.9, 4.2, False),
    (">", 1.0, 9.9, 4.2, True), (">", 9.9, 0.9, 4.2, False),
    ("<=", 9.9, 4.2, 4.2, True), (">=", 1.0, 4.2, 4.2, True),
    ("==", 1.0, 4.2, 4.2, True), ("!=", 1.0, 4.2, 4.2, False),
])
def test_all_comparators(op, previous, actual, threshold, expected):
    line = {"id": "M1", "anchor": {
        "expr": f"{CHANNEL} {op} {threshold}"}}
    out = campaign.verify_mechanism_line(
        line, signature(actual), baseline(previous))
    assert out["verified"] is expected


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
# 顺序证据：阈值成立不够，字段还得确实动过
#
# 只查阈值的话，一个本来停顿就很低的 kernel 会让机制行平凡成立 ——
# 门禁看起来过了，实际什么都没验证。论文要的是「改动前后两次读数」这个序对。
# ---------------------------------------------------------------------------

def test_vacuous_when_field_did_not_move():
    """核心用例：这个 kernel 的停顿本来就是 0.9，改动做的是别的事。"""
    line = milestone()["milestones"][0]["gate"]["mechanism_lines"][0]
    out = campaign.verify_mechanism_line(line, signature(0.9), baseline(0.9))
    assert out["verified"] is False
    assert out["vacuous"] is True
    assert "本来就是" in out["reason"]
    assert "平凡通过" in out["reason"]


def test_vacuous_when_movement_below_threshold():
    """动了但幅度低于判据，同样算没动。"""
    line = milestone()["milestones"][0]["gate"]["mechanism_lines"][0]
    out = campaign.verify_mechanism_line(line, signature(0.98), baseline(1.0))
    assert out["vacuous"] is True


def test_movement_just_above_threshold_verifies():
    line = milestone()["milestones"][0]["gate"]["mechanism_lines"][0]
    out = campaign.verify_mechanism_line(line, signature(0.90), baseline(1.0))
    assert out["verified"] is True


def test_vacuous_when_direction_is_wrong():
    """阈值仍成立，但字段朝反方向动了 —— 机制没朝声称的方向起作用。"""
    line = milestone()["milestones"][0]["gate"]["mechanism_lines"][0]
    out = campaign.verify_mechanism_line(line, signature(3.0), baseline(1.0))
    assert out["verified"] is False
    assert out["vacuous"] is True
    assert "方向" in out["reason"]


def test_direction_derived_from_operator():
    """方向从比较符推出，不另立一张「每个标签该增该减」的表。"""
    assert campaign._expected_direction("<") == -1
    assert campaign._expected_direction("<=") == -1
    assert campaign._expected_direction(">") == +1
    assert campaign._expected_direction(">=") == +1
    assert campaign._expected_direction("==") is None


def test_increase_direction_verified():
    """L2 命中率这类指标声称的是「升上去」，用 > 表达。"""
    line = {"id": "M1", "label": "L2_LOCALITY", "anchor": {
        "expr": "msprof:L2Cache.csv:total_hit_rate > 80"}}
    sig = {"trust": {"trustworthy": True},
           "cache": {"hit": {"value": 92.0, "present": True,
                             "channel": "msprof:L2Cache.csv:total_hit_rate"}}}
    out = campaign.verify_mechanism_line(
        line, sig, {"msprof:L2Cache.csv:total_hit_rate": 45.0})
    assert out["verified"] is True


def test_unknown_without_any_baseline():
    """没有上一轮也没有参考签名时，顺序证据无从谈起 —— 不能算核实。"""
    line = milestone()["milestones"][0]["gate"]["mechanism_lines"][0]
    out = campaign.verify_mechanism_line(line, signature(0.9), None)
    assert out["verified"] is None
    assert "不构成顺序证据" in out["reason"]


def test_judge_reports_vacuous_status():
    out = campaign.judge_mechanism(
        milestone(), signature(0.9), "D0.1", baseline(0.9))
    assert out["status"] == "vacuous"
    assert "平凡通过" in out["reason"]


def test_refuted_takes_priority_over_vacuous():
    """阈值不成立比平凡通过更严重，状态要报前者。"""
    doc = milestone()
    doc["milestones"][0]["gate"]["mechanism_lines"].append({
        "id": "M2", "label": "SUPPLY_KEEPUP", "status": "PASS",
        "anchor": {"expr": f"{CHANNEL} < 0.1"},
        "ordering_evidence": "x",
    })
    out = campaign.judge_mechanism(doc, signature(0.9), "D0.1", baseline(0.9))
    assert out["status"] == "refuted"


# ---------------------------------------------------------------------------
# 做差基线从哪来
# ---------------------------------------------------------------------------

def test_baseline_falls_back_to_reference():
    """首轮没有上一轮，参考自己的签名就是第 0 轮。"""
    state = fresh_state()
    snapshot, label = campaign.previous_snapshot(state)
    assert snapshot[CHANNEL] == REFERENCE_STALL
    assert label == "参考实现"


def test_baseline_prefers_latest_ledger_entry():
    state = fresh_state(target=10.0)
    apply(state, result(speedup=1.30), "D0.1",
                          milestone_doc=milestone(), signature=signature(2.0))
    snapshot, label = campaign.previous_snapshot(state)
    assert snapshot[CHANNEL] == 2.0
    assert "上一轮" in label


def test_invalidated_entries_not_used_as_baseline():
    """已作废的条目建立在旧架构假设上，拿它做差会把变化归错因。"""
    state = fresh_state(target=10.0)
    apply(state, result(speedup=1.30), "D0.1",
                          milestone_doc=milestone(), signature=signature(2.0))
    state["ledger"][-1]["invalidated"] = True
    snapshot, label = campaign.previous_snapshot(state)
    assert snapshot[CHANNEL] == REFERENCE_STALL
    assert label == "参考实现"


def test_no_baseline_at_all():
    state = fresh_state(ref_stall=None)
    snapshot, label = campaign.previous_snapshot(state)
    assert snapshot is None
    assert label == "无"


def test_snapshot_skips_absent_fields():
    """缺失的字段不进快照，下一轮做差时如实报缺，而不是拿 0 去减。"""
    snapshot = campaign.snapshot_signature(signature(present=False))
    assert CHANNEL not in snapshot


def test_snapshot_includes_derived_exposure():
    sig = {"exposure": {"available": True, "exposed_pct": 14.1,
                        "exposed_us": 64.4, "covered_us": 392.1}}
    snapshot = campaign.snapshot_signature(sig)
    assert snapshot["derived:exposure.exposed_pct"] == 14.1


def test_ledger_carries_signature_snapshot():
    state = fresh_state()
    entry = apply(
        state, result(speedup=1.30), "D0.1",
        milestone_doc=milestone(), signature=signature(0.9))
    assert entry["signature_snapshot"][CHANNEL] == 0.9


def test_mechanism_records_which_baseline_was_used():
    """台账要能回答「这次是跟谁比的」。"""
    state = fresh_state()
    entry = apply(
        state, result(speedup=1.30), "D0.1",
        milestone_doc=milestone(), signature=signature(0.9))
    assert entry["mechanism"]["compared_against"] == "参考实现"


def test_vacuous_on_a_well_read_line_is_an_implementation_failure():
    """平凡通过 + 课程表读对了参考 ⇒ 实现没做到，不是架构信号。

    这条用例原先断言 streak_refuted（架构重入）。改掉的理由是那个归因
    方向是反的：课程表声称参考坐在 4.2，参考实测就是 4.2 —— 读法没问题。
    把它算成架构信号会把架构层连续叫回来重写一份本来就对的课程表，
    而实现还是同一个退化回退。
    """
    state = fresh_state(target=10.0, threshold=99)
    apply(state, result(speedup=1.30), "D0.1",
                          milestone_doc=milestone(),
                          signature=signature(REFERENCE_STALL))
    assert state["mechanism_unverified"]["streak_refuted"] == 0
    assert state["implementation_failures"]["consecutive"] == 1


def test_vacuous_on_a_misread_line_is_an_architectural_signal():
    """同样是平凡通过，但课程表读错了参考 ⇒ 该回架构层。

    要两轮才能构造出 vacuous：第一轮的基线是参考快照（18.0），字段降到
    0.5 是一次真实移动（verified）；第二轮字段原地不动，这才是平凡通过。
    """
    state, doc = misread_reference()
    apply(state, result(speedup=1.30), "D0.1",
                          milestone_doc=doc, signature=signature(0.5))
    apply(state, result(speedup=1.30), "D0.1",
                          milestone_doc=doc, signature=signature(0.5))
    assert state["mechanism_unverified"]["streak_refuted"] == 1
    assert state["mechanism_unverified"]["streak_unknown"] == 0


def test_promote_clears_reference_snapshot(tmp_path):
    """新参考是另一个二进制，拿旧签名做差会把两代差异算到本代头上。"""
    state_file = tmp_path / "s.json"
    run(init_argv(state_file, target=1.1,
                  reference_signature=write_reference_signature(tmp_path)))
    res = tmp_path / "ev.json"
    res.write_text(json.dumps(result(speedup=1.25)), encoding="utf-8")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(signature(0.9)), encoding="utf-8")
    mil = tmp_path / "m.json"
    mil.write_text(json.dumps(milestone()), encoding="utf-8")
    run(["--state", str(state_file), "record", "--result", str(res),
         "--decision", "D0.1", "--signature", str(sig),
         "--milestone", str(mil), "--allow-stale-noise-floor",
         # 这条增益被判 kb_admissible，所以它欠一张卡片。不还的话
         # promote 会被知识欠账那道闸拦住，而这两个用例查的是别的事。
         "--invented", "R-2201-001",
         # 同理：没有围栏证据的一轮不抬高 best，promote 会先被「best 未复测」
         # 拦住。围栏是静态扫描，真实流程里每轮都跑。
         "--fence-report", str(write_clean_fence(tmp_path))])
    # 确认复测：promote 要求 best 经过独立复测，而确认走单独的子命令
    run(["--state", str(state_file), "confirm", "--result", str(res),
         "--decision", "D0.1"])
    artifact = tmp_path / "b"
    (artifact / "op_kernel").mkdir(parents=True)
    (artifact / "op_kernel" / "k.cpp").write_text("// k\n", encoding="utf-8")
    run(["--state", str(state_file), "promote", "--artifact", str(artifact),
         "--library", str(tmp_path / "lib")])

    promoted = json.loads(state_file.read_text(encoding="utf-8"))
    assert promoted["reference_snapshot"] is None


def test_cli_set_reference_signature(tmp_path):
    state_file = tmp_path / "s.json"
    run(init_argv(state_file))
    sig = write_reference_signature(tmp_path, 7.7)
    code, _ = run(["--state", str(state_file), "set-reference-signature",
                   "--signature", str(sig)])
    assert code == 0
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["reference_snapshot"][CHANNEL] == 7.7


def test_cli_set_reference_signature_rejects_empty(tmp_path):
    """签名里一个有效读数都没有时，它不能充当做差基线。"""
    state_file = tmp_path / "s.json"
    run(init_argv(state_file))
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps(signature(present=False)), encoding="utf-8")
    code, out = run(["--state", str(state_file), "set-reference-signature",
                     "--signature", str(empty)])
    assert code == 1
    assert "无法作为做差基线" in out


# ---------------------------------------------------------------------------
# 裁决：机制进回路后的四种结果
# ---------------------------------------------------------------------------

def test_verified_mechanism_yields_accept():
    state = fresh_state()
    entry = apply(
        state, result(speedup=1.20), "D0.1",
        milestone_doc=milestone(), signature=signature(0.9))
    assert entry["verdict"] == campaign.VERDICT_ACCEPT
    assert entry["counts_toward_best"] is True
    assert entry["kb_admissible"] is True
    assert state["best"]["gain"] == pytest.approx(1.20)


def test_refuted_mechanism_yields_accept_unverified():
    """核心用例：变快了但机制没落地 —— 保留，但不许影响 best 与知识库。"""
    state = fresh_state()
    entry = apply(
        state, result(speedup=1.20), "D0.1",
        milestone_doc=milestone(), signature=signature(9.9))
    assert entry["verdict"] == campaign.VERDICT_ACCEPT_UNVERIFIED
    assert entry["counts_toward_best"] is False
    assert entry["kb_admissible"] is False
    assert state["best"]["gain"] == 1.0, "未核实的增益不得抬高 best"


def test_unverified_mechanism_yields_accept_unverified():
    state = fresh_state()
    entry = apply(
        state, result(speedup=1.20), "D0.1",
        milestone_doc=milestone(), signature=signature(present=False))
    assert entry["verdict"] == campaign.VERDICT_ACCEPT_UNVERIFIED
    assert state["best"]["gain"] == 1.0


def test_missing_probe_inputs_yield_accept_unverified():
    """不传 signature/milestone 时不能当成核实过。"""
    state = fresh_state()
    entry = apply(state, result(speedup=1.20), "D0.1")
    assert entry["verdict"] == campaign.VERDICT_ACCEPT_UNVERIFIED
    assert entry["kb_admissible"] is False


def test_unverified_does_not_trigger_ship():
    """未核实的增益不得让 campaign 进入可提升状态。"""
    state = fresh_state(target=1.1)
    apply(state, result(speedup=1.50), "D0.1")
    assert state["state"] != campaign.STATE_SHIPPED


def test_single_reading_does_not_ship(tmp_path=None):
    """看似达标不能直接收工 —— best 是历史最大值，而每次 record 是一次抽样。

    预算几十次的 campaign 里「抽到一个偏高读数」是必然事件。拿它转 shipped
    再 promote，就把一次运气封成了下一代参考，而 promote 会把 best 归零、
    噪声底置为待重测 —— 那个读数本身从此消失，无法回溯。
    """
    state = fresh_state(target=1.05)
    apply(state, result(1.20), "D0.1",
                          milestone_doc=milestone(), signature=signature(0.9))
    assert state["state"] == campaign.STATE_IMPLEMENT
    assert state["pending_confirmation"]["decision"] == "D0.1"
    assert campaign.next_action(state)["action"] == "confirm-best"


def test_second_matching_reading_ships(tmp_path=None):
    """同一个决策再测一次、仍然达标 —— 这才收工。"""
    state = fresh_state(target=1.05)
    ms = milestone()
    apply(state, result(1.20), "D0.1", milestone_doc=ms,
                          signature=signature(0.9))
    assert state["state"] == campaign.STATE_IMPLEMENT

    # 确认复测走独立入口：它跑的是同一份代码，机制必然「没动」——
    # 再走一次 record 会被判 vacuous、不计入 best，确认永远无法完成。
    campaign.confirm_best(state, result(1.18), "D0.1")
    assert state["state"] == campaign.STATE_SHIPPED
    assert state.get("pending_confirmation") is None
    # best 取最小那次读数：多次抽样取最大就是让运气决定 best
    assert state["best"]["gain"] == pytest.approx(1.18)


def test_confirmation_requires_every_reading_to_clear_target():
    """两次都要达标，不是平均达标。

    平均会让一次偏高读数把一次不达标的拉过线 —— 那正是要防的情形。
    """
    state = fresh_state(target=1.10)
    state["best"] = {"gain": 1.20, "decision": "D0.1",
                     "readings": [1.20, 1.02]}
    assert campaign._best_is_confirmed(state) is False
    state["best"]["readings"] = [1.20, 1.12]
    assert campaign._best_is_confirmed(state) is True


def test_verified_triggers_ship():
    """机制核实 + 达标 + **两次读数都达标** → shipped。

    第二次 record 是确认复测；单次读数只会转 pending_confirmation，
    见 test_single_reading_does_not_ship。
    """
    state = fresh_state(target=1.1)
    ms = milestone()
    apply(state, result(speedup=1.50), "D0.1",
                          milestone_doc=ms, signature=signature(0.9))
    campaign.confirm_best(state, result(speedup=1.45), "D0.1")
    assert state["state"] == campaign.STATE_SHIPPED


def test_precision_failure_reverts_regardless_of_speed():
    state = fresh_state()
    entry = apply(
        state, result(speedup=5.0, precision=False), "D0.1",
        milestone_doc=milestone(), signature=signature(0.9))
    assert entry["verdict"] == campaign.VERDICT_REVERT
    assert "精度" in entry["reason"]
    assert state["best"]["gain"] == 1.0


def test_missing_measurement_reverts():
    """精度过了但没测出数 —— 这一条要隔离「没有测量」这一种失败。

    夹具里显式写上 precision_passed，否则会先被精度门禁拦下（字段缺失
    本身也判 revert，见下一条），测到的就不是这条路径了。
    """
    state = fresh_state()
    entry = apply(
        state, {"comparison": {"precision_passed": True}}, "D0.1")
    assert entry["verdict"] == campaign.VERDICT_REVERT
    assert "无测量支持" in entry["reason"]
    assert entry["failure_kind"] == "no_measurement"


def test_missing_precision_field_reverts():
    """字段缺失不等于精度通过。

    早先只拦 precision is False，于是一份没有精度字段的结果直接落到增益
    判据 —— 精度门禁静默消失，而算错的 kernel 再快也无意义。
    """
    state = fresh_state()
    entry = apply(
        state, {"comparison": {"speedup": 1.5}}, "D0.1")
    assert entry["verdict"] == campaign.VERDICT_REVERT
    assert "没有精度字段" in entry["reason"]
    assert entry["failure_kind"] == "precision"


def test_gain_computed_from_times_when_speedup_absent():
    state = fresh_state()
    entry = apply(
        state, result(baseline_us=400.0, evolved_us=200.0), "D0.1",
        milestone_doc=milestone(), signature=signature(0.9))
    assert entry["gain"] == pytest.approx(2.0)
    assert entry["verdict"] == campaign.VERDICT_ACCEPT


def test_best_not_downgraded_by_weaker_accept():
    state = fresh_state(target=10.0)
    for speedup in (1.80, 1.10):
        apply(state, result(speedup=speedup), "D0.1",
                              milestone_doc=milestone(), signature=signature(0.9))
    assert state["best"]["gain"] == pytest.approx(1.80)


# ---------------------------------------------------------------------------
# 停滞归因：实现问题不该消耗架构重入机会
# ---------------------------------------------------------------------------

def test_stagnation_counts_only_measured_shortfalls():
    state = fresh_state(threshold=3)
    for _ in range(3):
        apply(state, result(speedup=1.001), "D0.1")
    assert state["state"] == campaign.STATE_STALLED
    assert state["stall_reason"] == campaign.STALL_NO_IMPROVEMENT


def test_precision_failures_do_not_trigger_architecture_reentry():
    """连续编译/精度 bug 是实现问题，不该白费一次架构重入。"""
    state = fresh_state(threshold=3)
    for _ in range(5):
        apply(state, result(speedup=5.0, precision=False), "D0.1")
    assert state["state"] != campaign.STATE_STALLED
    assert state["stagnation"]["since_improvement"] == 0
    assert state["implementation_failures"]["consecutive"] == 5


def test_missing_measurement_counts_as_implementation_failure():
    state = fresh_state(threshold=3)
    for _ in range(3):
        apply(state, {"comparison": {}}, "D0.1")
    assert state["state"] != campaign.STATE_STALLED
    assert state["implementation_failures"]["consecutive"] == 3


def test_raising_best_resets_both_counters():
    """抬高 best 才清零停滞 —— 决策必须在 milestone 里，否则机制核验不了。"""
    state = fresh_state(threshold=3)
    ms = milestone({"D0.1": []})
    apply(state, result(speedup=1.001), "D0.1", milestone_doc=ms)
    apply(state, result(speedup=5.0, precision=False), "D0.1",
                          milestone_doc=ms)
    apply(state, result(speedup=1.30), "D0.1",
                          milestone_doc=ms, signature=signature(0.9))
    assert state["stagnation"]["since_improvement"] == 0
    assert state["implementation_failures"]["consecutive"] == 0


def test_repeated_same_size_gain_still_counts_as_stagnation():
    """重复拿同样大小的增益也算停滞。

    早先任何 accept / accept_unverified 都清零停滞，而 best 只在 gain >
    best 时才动。于是每轮都交出 +3%（相对参考、不是相对上一轮）时：
    每轮 accept → 停滞清零；best 第一轮之后再不动；状态永远是 implement，
    而 status 每轮都说「继续实现」—— 循环就这么空转下去。

    上游的判据是「窗口内有没有新的下界」，这条用例钉的就是它。
    """
    state = fresh_state(threshold=3, noise=2.0)
    ms = milestone({"D0.1": []})
    for _ in range(4):
        apply(state, result(speedup=1.03), "D0.1",
                              milestone_doc=ms, signature=signature(0.9))
    # 第一轮抬到 1.03，之后三轮都没有新下界
    assert state["best"]["gain"] == pytest.approx(1.03)
    assert state["stagnation"]["since_improvement"] == 3
    assert state["state"] == campaign.STATE_STALLED


def test_accept_unverified_does_not_reset_stagnation():
    """机制没核实的改动不抬 best，所以也不该清零停滞。"""
    state = fresh_state(threshold=3)
    ms = milestone({"D0.1": []})
    apply(state, result(speedup=1.001), "D0.1", milestone_doc=ms)
    before = state["stagnation"]["since_improvement"]
    # 决策不在课程表里 → 机制无法核验 → accept_unverified
    apply(state, result(speedup=1.30), "D9.9", milestone_doc=ms)
    assert state["stagnation"]["since_improvement"] == before + 1


# ---------------------------------------------------------------------------
# 连续实现失败要改变动作
#
# 早先 implementation_failures 只递增、只在 status 里打印，_advance_state
# 从不读它 —— 连续 30 次精度不过，next_action 每次都说 implement，预算一条条
# 被消耗，没有任何人被告知。
# ---------------------------------------------------------------------------

def test_consecutive_implementation_failures_block():
    state = fresh_state()
    ms = milestone({"D0.1": []})
    for _ in range(campaign.DEFAULT_IMPL_FAILURE_THRESHOLD):
        apply(state, result(speedup=5.0, precision=False),
                              "D0.1", milestone_doc=ms)
    assert state["state"] == campaign.STATE_BLOCKED
    assert campaign.next_action(state)["action"] == "fix-implementation"


def test_blocked_does_not_consume_architect_reentry():
    """实现 bug 不该花掉一次架构重入机会 —— 它进独立状态。"""
    state = fresh_state()
    ms = milestone({"D0.1": []})
    for _ in range(campaign.DEFAULT_IMPL_FAILURE_THRESHOLD):
        apply(state, result(speedup=5.0, precision=False),
                              "D0.1", milestone_doc=ms)
    assert state["state"] != campaign.STATE_STALLED
    assert state["stall_reason"] == campaign.STALL_IMPL_FAILURE


# ---------------------------------------------------------------------------
# 外层重入的另外两个口子
# ---------------------------------------------------------------------------

def test_graph_exhaustion_triggers_stall():
    """决策图走穷但没达标 —— 必须有人提示回架构层。"""
    doc = milestone({"D0.1": [], "D0.2": ["D0.1"]})
    state = fresh_state(target=10.0, threshold=99)
    apply(state, result(speedup=1.30), "D0.1",
                          milestone_doc=doc, signature=signature(0.9))
    assert state["state"] == campaign.STATE_IMPLEMENT
    apply(state, result(speedup=1.35), "D0.2",
                          milestone_doc=doc, signature=signature(0.9))
    assert state["state"] == campaign.STATE_STALLED
    assert state["stall_reason"] == campaign.STALL_GRAPH_EXHAUSTED


def test_open_gate_line_triggers_impasse():
    """用两条决策，避免同时触发走穷，单独验证僵持这一条。"""
    doc = milestone({"D0.1": [], "D0.2": ["D0.1"]})
    doc["milestones"][0]["gate"]["correctness_lines"][0]["status"] = "OPEN"
    state = fresh_state(target=10.0, threshold=99)
    apply(state, result(speedup=1.30), "D0.1",
                          milestone_doc=doc, signature=signature(0.9))
    assert state["state"] == campaign.STATE_STALLED
    assert state["stall_reason"] == campaign.STALL_GATE_IMPASSE


def test_multiple_stall_reasons_all_recorded():
    """三个重入口可能同时成立，只报一个会让架构层丢线索。"""
    doc = milestone({"D0.1": []})
    doc["milestones"][0]["gate"]["correctness_lines"][0]["status"] = "OPEN"
    state = fresh_state(target=10.0, threshold=1)
    apply(state, result(speedup=1.001), "D0.1",
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
        apply(state, result(speedup=1.10), f"D0.{i}")
    assert state["state"] == campaign.STATE_STALLED
    assert state["stall_reason"] == campaign.STALL_MECHANISM_UNVERIFIED


def test_unverified_stall_names_the_missing_flags():
    """提示必须点名缺了什么，否则 agent 还是不知道该补哪一步。"""
    state = fresh_state(target=10.0, threshold=99)
    for i in range(3):
        apply(state, result(speedup=1.10), f"D0.{i}")
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
    """只有「读错了参考」这一类才走架构重入。"""
    state, doc = misread_reference()
    for _ in range(3):
        apply(state, result(speedup=1.30), "D0.1",
                              milestone_doc=doc, signature=signature(30.0))
    assert state["state"] == campaign.STATE_STALLED
    assert state["stall_reason"] == campaign.STALL_MECHANISM_REFUTED


def test_refuted_on_a_well_read_line_does_not_reach_architecture():
    """核心用例：实现声称做到了而停顿仍高，是实现问题。

    连续三次之后状态该转 blocked（停下来交给人），**不是** stalled
    （把架构层叫回来重写课程表）。
    """
    state = fresh_state(target=10.0, threshold=99)
    for _ in range(3):
        apply(state, result(speedup=1.30), "D0.1",
                              milestone_doc=milestone(), signature=signature(9.9))
    assert state["mechanism_unverified"]["streak_refuted"] == 0
    assert state["stall_reason"] != campaign.STALL_MECHANISM_REFUTED


def test_refuted_detail_points_at_the_curriculum_not_the_flags():
    """被证伪时不能说「通常是没带 --signature」。"""
    state, doc = misread_reference()
    for _ in range(3):
        apply(state, result(speedup=1.30), "D0.1",
                              milestone_doc=doc, signature=signature(30.0))
    detail = state["stall_reasons"][0]["detail"]
    assert "机制核验为假" in detail
    assert "解释错了" in detail
    assert "不是参数没传" in detail


def test_refuted_action_sends_work_to_architecture():
    state, doc = misread_reference()
    for _ in range(3):
        apply(state, result(speedup=1.30), "D0.1",
                              milestone_doc=doc, signature=signature(30.0))
    action = campaign.next_action(state)
    assert action["action"] == "re-analyze"
    assert "修正机制假设" in action["reason"]


def test_pure_unverified_action_is_fix_invocation():
    """纯流程缺口不该惊动架构层 —— 标成 re-analyze 会让它白跑一轮分析。"""
    state = fresh_state(target=10.0, threshold=99)
    for _ in range(3):
        apply(state, result(speedup=1.30), "D0.1")
    action = campaign.next_action(state)
    assert action["action"] == "fix-invocation"
    assert "不必改课程表" in action["reason"]


def test_mixed_streak_reports_both_reasons():
    """两种成因并存时都要报，且被证伪排在前面（它是架构信号）。"""
    state, doc = misread_reference()
    apply(state, result(speedup=1.30), "D0.1")
    apply(state, result(speedup=1.30), "D0.1")
    apply(state, result(speedup=1.30), "D0.1",
                          milestone_doc=doc, signature=signature(30.0))
    kinds = [r["kind"] for r in state["stall_reasons"]]
    assert campaign.STALL_MECHANISM_REFUTED in kinds
    assert campaign.STALL_MECHANISM_UNVERIFIED in kinds
    assert kinds.index(campaign.STALL_MECHANISM_REFUTED) < \
        kinds.index(campaign.STALL_MECHANISM_UNVERIFIED)
    assert campaign.next_action(state)["action"] == "re-analyze"


def test_streak_breakdown_tracked():
    state, doc = misread_reference()
    apply(state, result(speedup=1.30), "D0.1")
    apply(state, result(speedup=1.30), "D0.1",
                          milestone_doc=doc, signature=signature(30.0))
    unver = state["mechanism_unverified"]
    assert unver["streak_unknown"] == 1
    assert unver["streak_refuted"] == 1
    assert unver["consecutive"] == 2


def test_verified_accept_clears_the_breakdown():
    state = fresh_state(target=10.0, threshold=99)
    apply(state, result(speedup=1.30), "D0.1",
                          milestone_doc=milestone(), signature=signature(9.9))
    apply(state, result(speedup=1.40), "D0.1",
                          milestone_doc=milestone(), signature=signature(0.9))
    unver = state["mechanism_unverified"]
    assert unver["consecutive"] == 0
    assert unver["streak_refuted"] == 0
    assert unver["streak_unknown"] == 0


def test_untrustworthy_signature_counts_as_unknown_not_refuted():
    """签名自判不可信时无从判断机制成没成立，不能算证伪。"""
    state = fresh_state(target=10.0, threshold=99)
    apply(
        state, result(speedup=1.30), "D0.1",
        milestone_doc=milestone(), signature=signature(9.9, trustworthy=False))
    unver = state["mechanism_unverified"]
    assert unver["streak_unknown"] == 1
    assert unver["streak_refuted"] == 0


def test_verified_accept_resets_unverified_counter():
    state = fresh_state(target=10.0, threshold=99)
    apply(state, result(speedup=1.10), "D0.1")
    apply(state, result(speedup=1.10), "D0.2")
    assert state["mechanism_unverified"]["consecutive"] == 2
    apply(state, result(speedup=1.20), "D0.1",
                          milestone_doc=milestone(), signature=signature(0.9))
    assert state["mechanism_unverified"]["consecutive"] == 0


def test_unverified_stall_takes_priority():
    """机制没核实时 best 卡在 1.0、其他信号都是降级的，应排最前。"""
    doc = milestone({"D0.1": []})
    doc["milestones"][0]["gate"]["correctness_lines"][0]["status"] = "OPEN"
    state = fresh_state(target=10.0, threshold=1)
    for _ in range(3):
        apply(state, result(speedup=1.10), "D0.1",
                              milestone_doc=doc)
    assert state["stall_reason"] == campaign.STALL_MECHANISM_UNVERIFIED


def test_reverts_do_not_reset_unverified_counter():
    """回滚既不确认也不否认 profiling 是否接上，应保持中立。"""
    state = fresh_state(target=10.0, threshold=99)
    apply(state, result(speedup=1.10), "D0.1")
    apply(state, result(speedup=1.001), "D0.2")
    assert state["mechanism_unverified"]["consecutive"] == 1


# ---------------------------------------------------------------------------
# 走穷判定不能把管理条目算成「试过」
# ---------------------------------------------------------------------------

def test_reopen_entry_not_counted_as_tried(tmp_path):
    """刚重开的叶子决策还没重做，不该立刻被判走穷。"""
    doc = milestone({"D0.1": [], "D0.2": []})
    state = fresh_state(target=10.0, threshold=99)
    for did in ("D0.1", "D0.2"):
        apply(state, result(speedup=1.10), did,
                              milestone_doc=doc, signature=signature(0.9))

    state_file = tmp_path / "s.json"
    milestone_file = tmp_path / "m.json"
    campaign.save_state(state_file, state)
    milestone_file.write_text(json.dumps(doc), encoding="utf-8")

    import argparse
    campaign.cmd_reopen(argparse.Namespace(
        state=str(state_file), milestone=str(milestone_file),
        decision="D0.2", reason="", as_json=False,
        price=None, lever_store=None, kb_root=None, force=True))

    reloaded = campaign.load_state(state_file)
    apply(reloaded, result(speedup=1.10), "D0.1",
                          milestone_doc=doc, signature=signature(0.9))
    assert reloaded["state"] != campaign.STATE_STALLED, \
        "D0.2 刚重开尚未重做，决策图没有走穷"


@pytest.mark.parametrize("verdict", ["reopen", "promote", "stop", "noise_floor"])
def test_management_entries_excluded_from_tried(verdict):
    doc = milestone({"D0.1": [], "D0.2": []})
    state = fresh_state(target=10.0, threshold=99)
    apply(state, result(speedup=1.10), "D0.1",
                          milestone_doc=doc, signature=signature(0.9))
    state["ledger"].append({"decision": "D0.2", "verdict": verdict})
    assert campaign.graph_exhausted(state, doc) is False


def test_genuine_exhaustion_still_detected():
    doc = milestone({"D0.1": [], "D0.2": []})
    state = fresh_state(target=10.0, threshold=99)
    for did in ("D0.1", "D0.2"):
        apply(state, result(speedup=1.10), did,
                              milestone_doc=doc, signature=signature(0.9))
    assert campaign.graph_exhausted(state, doc) is True


def test_stall_flags_cleared_on_recovery():
    doc = milestone({"D0.1": [], "D0.2": ["D0.1"]})
    state = fresh_state(target=10.0, threshold=1)
    apply(state, result(speedup=1.001), "D0.1",
                          milestone_doc=doc)
    assert state["state"] == campaign.STATE_STALLED
    apply(state, result(speedup=1.30), "D0.2",
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
    apply(state, result(speedup=1.50), "D0.1",
                          milestone_doc=doc, signature=signature(0.9))
    campaign.confirm_best(state, result(speedup=1.48), "D0.1")
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
        apply(state, result(speedup=1.30), did,
                              milestone_doc=doc, signature=signature(0.9))
    campaign.save_state(state_file, state)

    milestone_file = tmp_path / "m.json"
    milestone_file.write_text(json.dumps(doc), encoding="utf-8")

    import argparse
    code = campaign.cmd_reopen(argparse.Namespace(
        state=str(state_file), milestone=str(milestone_file),
        decision="D0.1", reason="拓扑决策有缺陷", as_json=False,
        price=None, lever_store=None, kb_root=None, force=True))
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
    # 两条决策共用一条机制行，所以第二条要相对第一条继续下降；
    # 停在原地就是平凡通过（阈值成立但不是它造成的）。
    apply(
        state, result(speedup=1.80, evolved_us=250.0), "D0.2",
        milestone_doc=doc, signature=signature(2.0))
    apply(
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
        decision="D0.1", reason="", as_json=False,
        price=None, lever_store=None, kb_root=None, force=True))

    reloaded = json.loads(state_file.read_text(encoding="utf-8"))
    assert reloaded["best"]["gain"] == pytest.approx(1.20), \
        "1.80 来自已作废的 D0.2，必须退回到 D0.3 的 1.20"
    assert reloaded["best"]["time_us"] == pytest.approx(380.0), \
        "time_us 要跟着 gain 一起重算，不能被抹成 None"


def test_reopen_resets_all_counters(tmp_path):
    """三个计数器一起归零，与 promote 的处理对齐。"""
    doc = milestone({"D0.1": []})
    state = fresh_state(target=10.0, threshold=99)
    apply(state, result(speedup=1.10), "D0.1")          # unverified
    apply(state, result(speedup=1.001), "D0.1")         # stagnation
    apply(state, result(precision=False, speedup=2.0), "D0.1")
    assert state["mechanism_unverified"]["consecutive"] > 0

    state_file = tmp_path / "s.json"
    milestone_file = tmp_path / "m.json"
    campaign.save_state(state_file, state)
    milestone_file.write_text(json.dumps(doc), encoding="utf-8")

    import argparse
    campaign.cmd_reopen(argparse.Namespace(
        state=str(state_file), milestone=str(milestone_file),
        decision="D0.1", reason="", as_json=False,
        price=None, lever_store=None, kb_root=None, force=True))

    reloaded = json.loads(state_file.read_text(encoding="utf-8"))
    assert reloaded["stagnation"]["since_improvement"] == 0
    assert reloaded["implementation_failures"]["consecutive"] == 0
    assert reloaded["mechanism_unverified"]["consecutive"] == 0


def test_ledger_entry_carries_time_us():
    """台账要能自给自足地重算 best，所以 time_us 必须落在条目里。"""
    state = fresh_state()
    entry = apply(
        state, result(speedup=1.30, evolved_us=350.0), "D0.1",
        milestone_doc=milestone(), signature=signature(0.9))
    assert entry["time_us"] == pytest.approx(350.0)


# ---------------------------------------------------------------------------
# 死路表登记
# ---------------------------------------------------------------------------

def test_nogo_command_is_copy_pasteable():
    state = fresh_state()
    entry = apply(state, result(speedup=1.001), "D0.1",
                                  symptom="MTE2 占比过高")
    command = campaign.nogo_command(state, entry)
    assert "hwkb.py nogo add" in command
    assert "MTE2 占比过高" in command
    assert "D0.1" in command
    assert "--arch DAV_2201" in command


def test_nogo_written_directly(tmp_path):
    """靠 agent 自觉登记是不牢的 —— 连续失败时最容易跳过记账。"""
    state = fresh_state()
    entry = apply(state, result(speedup=1.001), "D0.1",
                                  symptom="MTE2 占比过高")
    kb_root = tmp_path / "kb"
    assert campaign.write_nogo(kb_root, state, entry) == 1

    doc = json.loads((kb_root / campaign.NOGO_FILENAME).read_text(
        encoding="utf-8"))
    assert doc["entries"][0]["symptom"] == "MTE2 占比过高"
    assert doc["entries"][0]["tried"] == "D0.1"
    assert doc["entries"][0]["input_shape"] == "B=8,S=4096,fp16"


def test_nogo_appends_to_existing(tmp_path):
    kb_root = tmp_path / "kb"
    kb_root.mkdir()
    (kb_root / campaign.NOGO_FILENAME).write_text(
        json.dumps({"entries": [{"symptom": "旧的"}]}), encoding="utf-8")
    state = fresh_state()
    entry = apply(state, result(speedup=1.001), "D0.1")
    assert campaign.write_nogo(kb_root, state, entry) == 2


def test_nogo_locks_the_same_file_as_hwkb(tmp_path):
    """互斥要靠结构保证：两边都锁 <kb_root>/.lock，不随调用方的路径漂。"""
    kb_root = tmp_path / "kb"
    state = fresh_state()
    entry = apply(state, result(speedup=1.001), "D0.1")
    campaign.write_nogo(kb_root, state, entry)
    assert (kb_root / ".lock").exists()
    assert (kb_root / campaign.NOGO_FILENAME).exists()


# ---------------------------------------------------------------------------
# 知识库的返回路径
# ---------------------------------------------------------------------------

def test_recipes_recorded_in_ledger():
    state = fresh_state()
    entry = apply(
        state, result(speedup=1.30), "D0.1",
        milestone_doc=milestone(), signature=signature(0.9),
        recipes_used=["R-2201-001"], recipes_invented=["R-2201-009"])
    assert entry["recipes_used"] == ["R-2201-001"]
    assert entry["recipes_invented"] == ["R-2201-009"]


def test_recipe_summary_separates_foreign_and_invented():
    """没有这份汇总就无法回答「知识库到底有没有在帮忙」。"""
    state = fresh_state(target=10.0)
    apply(state, result(speedup=1.30), "D0.1",
                          milestone_doc=milestone(), signature=signature(0.9),
                          recipes_used=["R-A"], recipes_invented=["R-X"])
    apply(state, result(speedup=1.40), "D0.2",
                          milestone_doc=milestone(), signature=signature(0.9),
                          recipes_used=["R-A", "R-B"])
    summary = campaign.summarise_recipes(state)
    assert summary["foreign_recipes"] == {"R-A": 2, "R-B": 1}
    assert summary["invented_recipes"] == {"R-X": 1}
    assert summary["accepted_total"] == 2
    assert summary["foreign_assist_rate"] == pytest.approx(1.0)


def test_recipe_summary_skips_invalidated():
    state = fresh_state(target=10.0)
    apply(state, result(speedup=1.30), "D0.1",
                          recipes_used=["R-A"])
    state["ledger"][0]["invalidated"] = True
    assert campaign.summarise_recipes(state)["foreign_recipes"] == {}


def test_foreign_assist_rate_zero_without_recipes():
    state = fresh_state(target=10.0)
    apply(state, result(speedup=1.30), "D0.1")
    assert campaign.summarise_recipes(state)["foreign_assist_rate"] == 0.0


# ---------------------------------------------------------------------------
# campaign 身份：input shape
# ---------------------------------------------------------------------------

def test_input_shape_stored_in_state():
    assert fresh_state()["input_shape"] == "B=8,S=4096,fp16"


def test_input_shape_recorded_on_every_entry():
    """一份 state 跨 shape 复用会把不同 shape 的结果混进同一个 best。"""
    state = fresh_state()
    entry = apply(state, result(speedup=1.30), "D0.1")
    assert entry["input_shape"] == "B=8,S=4096,fp16"


# ---------------------------------------------------------------------------
# 预算
# ---------------------------------------------------------------------------

def test_budget_counts_every_record():
    state = fresh_state(budget=3)
    apply(state, result(speedup=1.001), "D0.1")
    assert state["budget"]["used"] == 1


def test_budget_exhaustion_stops_campaign():
    state = fresh_state(budget=2, threshold=99)
    for _ in range(2):
        apply(state, result(speedup=1.001), "D0.1")
    assert state["state"] == campaign.STATE_STOPPED
    assert "预算耗尽" in state["stop_reason"]


def test_unlimited_budget_never_stops():
    state = fresh_state(threshold=99)
    for _ in range(10):
        apply(state, result(speedup=1.001), "D0.1")
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


def write_reference_signature(tmp_path, value=REFERENCE_STALL):
    """参考自己的签名 = 机制做差的第 0 轮基线。"""
    path = tmp_path / "ref_sig.json"
    path.write_text(json.dumps(signature(value)), encoding="utf-8")
    return path


def init_argv(state_file, **kw):
    """构造一条 init 命令。

    默认带 --allow-no-boundary：这些用例测的是状态机与裁决逻辑，没有封存
    参考，而 init 会实测一次 no-code 边界（见 test_boundary_self_check_*
    那几条专门测它）。不带这个参数的话每条用例都要先铺一份封存清单。
    """
    argv = ["--state", str(state_file), "init", "--allow-no-boundary",
            "--operator", "demo_op", "--input-shape", "B=8,S=4096",
            "--arch", "DAV_2201",
            "--reference-kind", "builtin_aclnn", "--reference-id", "aclnnDemo",
            "--noise-floor", str(kw.get("noise", 2.0)),
            "--device-id", str(kw.get("device_id", 0))]
    if "target" in kw:
        argv += ["--target-gain", str(kw["target"])]
    if "budget" in kw:
        argv += ["--budget", str(kw["budget"])]
    if kw.get("reference_signature"):
        argv += ["--reference-signature", str(kw["reference_signature"])]
    return argv


def test_cli_init_requires_input_shape(tmp_path):
    code, _ = run(["--state", str(tmp_path / "s.json"), "init",
                   "--operator", "demo_op", "--arch", "DAV_2201",
                   "--reference-kind", "builtin_aclnn",
                   "--reference-id", "aclnnDemo", "--noise-floor", "2.0",
                   "--device-id", "0"])
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
    code, _ = run(["init", "--state", str(state_file), "--allow-no-boundary",
                   "--operator", "demo_op", "--input-shape", "B=8",
                   "--arch", "DAV_2201", "--reference-kind", "builtin_aclnn",
                   "--reference-id", "aclnnDemo", "--noise-floor", "2.0",
                   "--device-id", "0"])
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


def test_cli_record_writes_nogo_when_kb_root_given(tmp_path):
    state_file = tmp_path / "s.json"
    kb_root = tmp_path / "kb"
    run(init_argv(state_file))
    res = tmp_path / "ev.json"
    res.write_text(json.dumps(result(speedup=1.001)), encoding="utf-8")
    run(["--state", str(state_file), "record", "--result", str(res),
         "--decision", "D0.1", "--symptom", "MTE2 高",
         "--nogo-kb-root", str(kb_root)])
    nogo = kb_root / campaign.NOGO_FILENAME
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
    """新参考是另一个二进制，噪声底与参考签名都必须重取。"""
    state_file = tmp_path / "s.json"
    run(init_argv(state_file, target=1.1,
                  reference_signature=write_reference_signature(tmp_path)))
    res = tmp_path / "ev.json"
    res.write_text(json.dumps(result(speedup=1.25)), encoding="utf-8")
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(signature(0.9)), encoding="utf-8")
    mil = tmp_path / "m.json"
    mil.write_text(json.dumps(milestone()), encoding="utf-8")
    run(["--state", str(state_file), "record", "--result", str(res),
         "--decision", "D0.1", "--signature", str(sig),
         "--milestone", str(mil), "--allow-stale-noise-floor",
         # 这条增益被判 kb_admissible，所以它欠一张卡片。不还的话
         # promote 会被知识欠账那道闸拦住，而这两个用例查的是别的事。
         "--invented", "R-2201-001",
         # 同理：没有围栏证据的一轮不抬高 best，promote 会先被「best 未复测」
         # 拦住。围栏是静态扫描，真实流程里每轮都跑。
         "--fence-report", str(write_clean_fence(tmp_path))])
    # 确认复测：promote 要求 best 经过独立复测，而确认走单独的子命令
    run(["--state", str(state_file), "confirm", "--result", str(res),
         "--decision", "D0.1"])

    artifact = tmp_path / "best"
    (artifact / "op_kernel").mkdir(parents=True)
    (artifact / "op_kernel" / "k.cpp").write_text("// k\n", encoding="utf-8")
    code, out = run(["--state", str(state_file), "promote",
                     "--artifact", str(artifact),
                     "--library", str(tmp_path / "lib")])
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


def test_noise_floor_on_identical_readings_is_refused():
    """完全相同的比值不是「环境干净」，是这些数不是独立测出来的。

    这条用例原先断言返回 0.00%。改掉的理由：噪声底 0 的含义是「任何正增益
    都算真实改进」，而重复测量给出一模一样的比值在物理上不会发生 —— 实际
    成因是自评场景下 speedup 被钉成常数 1.0。返回 0 并提示「把这个数传给
    campaign init」会让测量抖动被当成加速收进知识库。
    """
    with pytest.raises(noise_floor.NoiseFloorError, match="完全相同"):
        noise_floor.compute([1.0, 1.0, 1.0])


def test_noise_floor_grows_with_spread():
    tight = noise_floor.compute([1.00, 1.01, 0.99])
    loose = noise_floor.compute([1.00, 1.20, 0.80])
    assert loose["floor_pct"] > tight["floor_pct"]


def test_noise_floor_uses_conservative_of_two_statistics():
    out = noise_floor.compute([1.0, 1.0, 1.0, 1.0, 1.3])
    assert out["floor_pct"] == pytest.approx(
        max(out["cv_pct"], out["spread_pct"]))


def test_noise_floor_spread_matches_the_paper_formula():
    """口径必须是「全极差比最小值」，不是「半极差比中位数」。

    早先写的是后者 —— 比值都在 1.0 附近时 median ≈ min，那个 /2.0 让同一批
    数据算出的噪声底只有论文的一半，判据系统性偏松一倍。
    """
    out = noise_floor.compute([1.00, 1.02, 1.04])
    assert out["spread_pct"] == pytest.approx((1.04 - 1.00) / 1.00 * 100.0)


def test_noise_floor_has_a_hard_minimum():
    """实测离散很小也不能把判据定得很低。

    论文实现把每代噪声底取成 max(1.5%, spread)，称之为「循环的平台期地板」：
    样本少时离散度本身就是个不稳的估计。我们这边还叠着一个方向恒定的
    位置偏置（约 2.7%），低于下限的噪声底在它面前没有意义。
    """
    out = noise_floor.compute([1.000, 1.001, 1.002])
    assert out["floor_pct"] == pytest.approx(noise_floor.NOISE_FLOOR_MIN_PCT)
    assert out["floor_clamped"] is True
    assert any("硬下限" in w for w in out["warnings"])


def test_clamp_does_not_mask_identical_readings():
    """钳位不能把「读数完全相同」这个无效输入信号盖住。"""
    with pytest.raises(noise_floor.NoiseFloorError, match="完全相同"):
        noise_floor.compute([1.0, 1.0, 1.0])


def test_cv_method_also_has_the_hard_minimum():
    out = noise_floor.compute_from_cv([0.3, 0.4])
    assert out["floor_pct"] == pytest.approx(noise_floor.NOISE_FLOOR_MIN_PCT)
    assert out["measured_pct"] == pytest.approx(0.4)


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


# ---------------------------------------------------------------------------
# 回归：被回滚的轮次不作基线
#
# 某一轮因增益落在噪声里被回滚，但它碰巧把某个停顿压到了低位。下一轮真的
# 把那个停顿从高位压下来时，若拿已撤回那轮的低位值当基线，会算出「没动」，
# 于是一个真实的机制被判成平凡通过 —— 而结论看着完全合理。
# ---------------------------------------------------------------------------

def test_reverted_round_is_not_used_as_baseline():
    doc = milestone({"D0.1": [], "D0.2": []})
    state = fresh_state(ref_stall=10.0, target=99.0)
    # 第一轮被回滚（增益在噪声内），但它的签名显示停顿已降到 2.0
    apply(state, result(speedup=1.001), "D0.1",
                          milestone_doc=doc, signature=signature(2.0))
    assert state["ledger"][0]["verdict"] == campaign.VERDICT_REVERT

    # 第二轮真的把 10.0 降到 2.0
    entry = apply(state, result(speedup=1.50), "D0.2",
                                  milestone_doc=doc,
                                  signature=signature(2.0))
    assert entry["mechanism"]["compared_against"] == "参考实现"
    assert entry["verdict"] == campaign.VERDICT_ACCEPT


def test_regression_round_is_not_used_as_baseline():
    state = fresh_state(ref_stall=10.0, target=99.0)
    apply(state, result(speedup=0.5), "D0.1",
                          milestone_doc=milestone(), signature=signature(2.0))
    assert state["ledger"][0]["verdict"] == campaign.VERDICT_REGRESSION
    snapshot, label = campaign.previous_snapshot(state)
    assert label == "参考实现"


def test_kept_round_is_used_as_baseline():
    """accept_unverified 的改动留在代码里，应当作基线。"""
    state = fresh_state(ref_stall=10.0, target=99.0)
    apply(state, result(speedup=1.50), "D0.1",
                          signature=signature(5.0))   # 无 milestone → unverified
    assert state["ledger"][0]["verdict"] == campaign.VERDICT_ACCEPT_UNVERIFIED
    snapshot, label = campaign.previous_snapshot(state)
    assert snapshot[CHANNEL] == 5.0
    assert "上一轮" in label


# ---------------------------------------------------------------------------
# 回归：同名指标跨硬件单元不得混比
#
# msprof 的 ai*_mte2_wait_ratio 在 Cube 与 Vector 上各有一份。锚点只写
# mte2_wait_ratio 时两边都匹配，挑错一个就会拿 Cube 的基线配 Vector 的
# 实测 —— 算出的百分比看着很有说服力，实际毫无意义。
# ---------------------------------------------------------------------------

AIC = "msprof:ResourceConflictRatio.csv:aic_mte2_wait_ratio"
AIV = "msprof:ResourceConflictRatio.csv:aiv_mte2_wait_ratio"


def two_unit_signature(aic_value, aiv_value):
    return {"trust": {"trustworthy": True}, "stalls": {"wait": {
        "cube": {"value": aic_value, "channel": AIC, "present": True},
        "vec": {"value": aiv_value, "channel": AIV, "present": True}}}}


def test_ambiguous_field_is_refused():
    line = {"id": "M1", "label": "SUPPLY_KEEPUP",
            "anchor": {"expr": "msprof:X.csv:mte2_wait_ratio < 4.0"}}
    out = campaign.verify_mechanism_line(
        line, two_unit_signature(2.0, 9.0), {AIC: 10.0, AIV: 9.2})
    assert out["verified"] is None
    assert "匹配到 2 个通道" in out["reason"]
    assert "单元前缀" in out["reason"]


def test_qualified_field_picks_the_right_unit():
    """写全名时取的就是那个单元：Vector 几乎没动，应判平凡通过。"""
    line = {"id": "M1", "label": "SUPPLY_KEEPUP",
            "anchor": {"expr": f"{AIV} < 9.5"}}
    out = campaign.verify_mechanism_line(
        line, two_unit_signature(2.0, 9.0), {AIC: 10.0, AIV: 9.2})
    assert out["vacuous"] is True
    assert out["actual"] == 9.0
    assert out["previous"] == 9.2


def test_baseline_matched_by_exact_channel():
    """基线按命中的通道精确取，不按字段名模糊匹配。"""
    line = {"id": "M1", "anchor": {"expr": f"{AIV} < 9.5"}}
    out = campaign.verify_mechanism_line(
        line, two_unit_signature(2.0, 9.0), {AIC: 10.0})   # 基线只有 Cube
    assert out["verified"] is None
    assert "同一通道" in out["reason"]


def test_lookup_matches_returns_all_units():
    matches = campaign.lookup_signature_matches(
        two_unit_signature(2.0, 9.0), "mte2_wait_ratio")
    assert len(matches) == 2
    assert {c for _v, c in matches} == {AIC, AIV}


def test_lookup_matches_dedupes_channels():
    """同一通道在签名里出现多次（嵌套结构）只算一个。"""
    sig = {"trust": {"trustworthy": True},
           "a": {"value": 1.0, "channel": AIV, "present": True},
           "b": {"nested": {"value": 1.0, "channel": AIV, "present": True}}}
    assert len(campaign.lookup_signature_matches(sig, "aiv_mte2")) == 1


# ===========================================================================
# init 的前置校验
#
# 这三道门挡的都是「能一路跑完、但结论全错」的情形。它们的共同点是要跑很久
# 才会暴露：架构串填错要等到第一张卡片入库被拒；签名不可信要等到机制做差
# 得出莫名其妙的结论；派发未确认则根本不知道基线测的是谁。
# ===========================================================================

def test_init_rejects_unknown_arch(tmp_path):
    argv = init_argv(tmp_path / "s.json")
    argv[argv.index("--arch") + 1] = "ascend910b"
    code, out = run(argv)
    assert code == 2
    assert "DAV_2201" in out


def test_init_accepts_unknown_arch_when_explicitly_allowed(tmp_path):
    argv = init_argv(tmp_path / "s.json") + ["--allow-unknown-arch"]
    argv[argv.index("--arch") + 1] = "DAV_9999"
    code, _ = run(argv)
    assert code == 0


def test_known_archs_match_the_knowledge_base():
    """架构串的真正来源是 forge-hwkb 的 ARCH_DIRS，两边必须一致。

    不一致的后果是 campaign 能跑完而卡片永远入不了库 —— 所以这里用一条
    用例盯住，不靠人记得同步改两个文件。
    """
    hwkb_scripts = (Path(__file__).resolve().parent.parent
                    / "skills" / "forge-hwkb" / "scripts")
    sys.path.insert(0, str(hwkb_scripts))
    import hwkb  # noqa: PLC0415

    assert set(campaign.KNOWN_ARCHS) == set(hwkb.ARCH_DIRS)


def test_reference_spec_constants_match_the_adapter():
    """campaign 里这两个常量是从 reference_adapter 抄来的，不能漂。"""
    import reference_adapter  # noqa: PLC0415

    assert campaign.SPEC_FILENAME == reference_adapter.SPEC_FILENAME
    assert campaign.KIND_BUILTIN == reference_adapter.KIND_BUILTIN


def test_init_rejects_untrusted_reference_signature(tmp_path):
    """签名自己判了测量不可信，就不能当第 0 轮基线。"""
    sig_path = tmp_path / "bad_sig.json"
    sig_path.write_text(json.dumps({
        "trust": {"trustworthy": False,
                  "blocking_reasons": ["缺少 metric: ResourceConflictRatio"]},
    }), encoding="utf-8")

    code, out = run(init_argv(tmp_path / "s.json",
                              reference_signature=sig_path))
    assert code == 2
    assert "ResourceConflictRatio" in out


def test_init_accepts_untrusted_signature_when_explicitly_allowed(tmp_path):
    sig_path = tmp_path / "bad_sig.json"
    sig_path.write_text(json.dumps({"trust": {"trustworthy": False}}),
                        encoding="utf-8")
    code, _ = run(init_argv(tmp_path / "s.json", reference_signature=sig_path)
                  + ["--allow-untrusted-signature"])
    assert code == 0


def test_init_accepts_trustworthy_signature(tmp_path):
    code, _ = run(init_argv(
        tmp_path / "s.json",
        reference_signature=write_reference_signature(tmp_path)))
    assert code == 0


# ---------------------------------------------------------------------------
# 派发确认
#
# builtin_aclnn 这种参考依赖「空 OPP 目录让 aclnn 落到内置实现」这一行为。
# 它没在本机证实过时，第 0 轮基线测的可能根本不是内置实现，后面每一轮
# 加速比都是对着一个未知对象算出来的。
# ---------------------------------------------------------------------------

def write_spec(tmp_path, dispatch_verified, kind="builtin_aclnn",
               reference_id="demo_op"):
    path = tmp_path / campaign.SPEC_FILENAME
    path.write_text(json.dumps({
        "kind": kind, "op_name": "demo_op", "reference_id": reference_id,
        "baseline_path": str(tmp_path / "null_opp"),
        "dispatch_verified": dispatch_verified,
    }), encoding="utf-8")
    return path


def test_init_rejects_unverified_dispatch(tmp_path):
    spec = write_spec(tmp_path, False)
    code, out = run(init_argv(tmp_path / "s.json")
                    + ["--reference-spec", str(spec)])
    assert code == 2
    assert "dispatch_verified" in out


def test_init_accepts_verified_dispatch(tmp_path):
    spec = write_spec(tmp_path, True)
    code, _ = run(init_argv(tmp_path / "s.json")
                  + ["--reference-spec", str(spec)])
    assert code == 0


def test_init_allows_unverified_dispatch_on_request(tmp_path):
    spec = write_spec(tmp_path, False)
    code, out = run(init_argv(tmp_path / "s.json")
                    + ["--reference-spec", str(spec),
                       "--allow-unverified-dispatch"])
    assert code == 0
    assert "milestone" in out


def test_init_finds_spec_next_to_the_state_file(tmp_path):
    """实际流程里 spec 就躺在工作目录里，不传也该被用上。"""
    write_spec(tmp_path, False)
    code, out = run(init_argv(tmp_path / "s.json"))
    assert code == 2
    assert "dispatch_verified" in out


def test_init_warns_when_reference_id_differs_from_spec(tmp_path):
    """quickstart 用 aclnnXxx、prepare 默认写算子名，两值并存容易让人误解。"""
    spec = write_spec(tmp_path, True, reference_id="demo_op")
    argv = init_argv(tmp_path / "s.json") + ["--reference-spec", str(spec)]
    code, out = run(argv)
    assert code == 0
    assert "--reference-id" in out and "建议统一成算子名" in out


def test_init_without_spec_still_works_but_warns(tmp_path):
    """没有 spec 时不卡死流程，但要说清少了哪一道确认。"""
    code, out = run(init_argv(tmp_path / "s.json"))
    assert code == 0
    assert campaign.SPEC_FILENAME in out


# ===========================================================================
# 噪声底：崩掉的评测不能量出一个数来
#
# 评测崩溃时写进结果文件的是 time_us=-1 和 cv_pct=0.0。那个 0.0 在形态上
# 和「环境极其干净」完全一样，于是脚本会一声不响地输出 0.00%，还提示把它
# 传给 campaign init。噪声底 0 的含义是「任何正增益都算真实改进」。
# ===========================================================================

def eval_result(gating="partial_passed", speedup=1.02, cv=2.4,
                baseline_us=314.4, evolved_us=308.2):
    return {
        "gating": gating,
        "shape_results": {"target": [{
            "name": "s0", "baseline_time_us": baseline_us,
            "evolved_time_us": evolved_us, "speedup": speedup,
            "cv_pct": cv, "precision_passed": True,
        }]},
        "baseline": {"time_us": baseline_us, "cv_pct": 0.0},
        "evolved": {"time_us": evolved_us, "cv_pct": cv},
        "comparison": {"speedup": speedup, "cv_pct": cv},
    }


def write_result(tmp_path, name, **kw):
    path = tmp_path / name
    path.write_text(json.dumps(eval_result(**kw)), encoding="utf-8")
    return path


def test_noise_floor_refuses_failed_evaluation(tmp_path):
    path = write_result(tmp_path, "failed.json", gating="failed",
                        speedup=0.0, cv=0.0, baseline_us=-1, evolved_us=-1)
    with pytest.raises(noise_floor.NoiseFloorError, match="gating=failed"):
        noise_floor._load_docs([path])


def test_noise_floor_refuses_result_without_any_positive_time(tmp_path):
    path = write_result(tmp_path, "crashed.json", gating="partial_passed",
                        baseline_us=-1, evolved_us=-1)
    with pytest.raises(noise_floor.NoiseFloorError, match="time_us"):
        noise_floor._load_docs([path])


def test_noise_floor_refuses_all_zero_cv():
    """cv 全 0 不是「环境干净」，是这个字段没被真正填上。"""
    with pytest.raises(noise_floor.NoiseFloorError, match="全是 0"):
        noise_floor.compute_from_cv([0.0, 0.0, 0.0])


def test_noise_floor_warns_when_below_paper_lower_bound():
    """偏低比偏高危险：判据会把抖动当成真实改进收下。

    compute() 一直有这条下限告警，compute_from_cv() 当初漏了 —— 两条
    路径不对称，而实际用的恰恰是后者。
    """
    out = noise_floor.compute_from_cv([0.2, 0.3])
    assert any("低于" in w for w in out["warnings"])


def test_noise_floor_rejects_spread_on_self_evaluation(tmp_path):
    """自评时 speedup 被钉成 1.0，比值离散法结构性地量不出东西。"""
    paths = [write_result(tmp_path, f"run{i}.json", speedup=1.0,
                          baseline_us=314.38 + i, evolved_us=319.84 + i,
                          cv=2.3 + i * 0.3)
             for i in range(3)]
    docs = noise_floor._load_docs(paths)
    with pytest.raises(noise_floor.NoiseFloorError, match="自评"):
        noise_floor._ratios_from_docs(docs, [str(p) for p in paths])


def test_noise_floor_cv_method_works_on_self_evaluation(tmp_path):
    """同一批自评结果用 cv 法照样能量出噪声底 —— cv 不经过 speedup。"""
    paths = [write_result(tmp_path, f"run{i}.json", speedup=1.0,
                          baseline_us=314.38, evolved_us=319.84,
                          cv=[2.31, 2.95, 2.63][i])
             for i in range(3)]
    docs = noise_floor._load_docs(paths)
    values = [v for doc in docs for v in noise_floor.extract_cv_values(doc)]
    assert noise_floor.compute_from_cv(values)["floor_pct"] == pytest.approx(2.95)


def test_forced_self_baseline_not_flagged_when_times_match(tmp_path):
    """两侧耗时本来就相等时，speedup=1.0 是真的持平，不是被钉的。"""
    path = write_result(tmp_path, "equal.json", speedup=1.0,
                        baseline_us=300.0, evolved_us=300.0)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert noise_floor.detect_forced_self_baseline(doc) == []


# ---------------------------------------------------------------------------
# 复合锚点：两种连接词都要认
#
# `and`/`or` 需要词边界（否则 "operand" 里的 "and" 会命中），而 `&&`/`||`
# 两侧没有单词字符，加 \b 会让这两个分支永远匹配不上 —— 一条用 && 写的
# 复合条件就会只验第一个子句再报「成立」，正是这道门要拦的情形。
# ---------------------------------------------------------------------------

def mech_line(expr):
    return {"id": "L1", "label": "ub_resident_reuse", "anchor": {"expr": expr}}


def test_compound_anchor_with_and_is_not_verified():
    out = campaign.verify_mechanism_line(
        mech_line(f"{CHANNEL} < 4.2 and other_field < 0.3"),
        signature(0.9), {CHANNEL: REFERENCE_STALL})
    assert out["verified"] is None
    assert "多个条件" in out["reason"]


def test_compound_anchor_with_c_style_and_is_not_verified():
    out = campaign.verify_mechanism_line(
        mech_line(f"{CHANNEL} < 4.2 && mechanism_ok"),
        signature(0.9), {CHANNEL: REFERENCE_STALL})
    assert out["verified"] is None, out


def test_compound_anchor_with_c_style_or_is_not_verified():
    out = campaign.verify_mechanism_line(
        mech_line(f"{CHANNEL} < 4.2 || fallback"),
        signature(0.9), {CHANNEL: REFERENCE_STALL})
    assert out["verified"] is None, out


def test_word_operand_is_not_mistaken_for_a_connective():
    """「operand」里含 and，不能据此把单条件判成复合条件。"""
    out = campaign.verify_mechanism_line(
        {"id": "L1", "label": "ub_resident_reuse",
         "anchor": {"expr": f"{CHANNEL} < 4.2"}},
        signature(0.9), {CHANNEL: REFERENCE_STALL})
    assert out["verified"] is True, out


def test_single_anchor_still_verifies():
    out = campaign.verify_mechanism_line(
        mech_line(f"{CHANNEL} < 4.2"),
        signature(0.9), {CHANNEL: REFERENCE_STALL})
    assert out["verified"] is True, out


# ---------------------------------------------------------------------------
# 不可信签名：set-reference-signature 走的是同一道门
#
# promote 会把 reference_snapshot 清空，所以第 1 代起每一代都只能靠这条命令
# 登记参考签名。门只守在 init 上等于只守住了第 0 代。
# ---------------------------------------------------------------------------

def test_set_reference_signature_rejects_untrusted(tmp_path):
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0

    sig_path = tmp_path / "bad.json"
    sig_path.write_text(json.dumps({
        "trust": {"trustworthy": False,
                  "blocking_reasons": ["发生 DVFS 降频"]},
        "stalls": {"wait": {"mte2_wait": {
            "value": 0.9, "channel": CHANNEL, "present": True}}},
    }), encoding="utf-8")

    code, out = run(["--state", str(state_file), "set-reference-signature",
                     "--signature", str(sig_path)])
    assert code == 2
    assert "DVFS" in out


def test_set_reference_signature_accepts_untrusted_when_allowed(tmp_path):
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0

    sig_path = tmp_path / "bad.json"
    sig_path.write_text(json.dumps({
        "trust": {"trustworthy": False, "blocking_reasons": ["降频"]},
        "stalls": {"wait": {"mte2_wait": {
            "value": 0.9, "channel": CHANNEL, "present": True}}},
    }), encoding="utf-8")

    code, _ = run(["--state", str(state_file), "set-reference-signature",
                   "--signature", str(sig_path),
                   "--allow-untrusted-signature"])
    assert code == 0


def test_set_reference_signature_accepts_trustworthy(tmp_path):
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0
    code, _ = run(["--state", str(state_file), "set-reference-signature",
                   "--signature", str(write_reference_signature(tmp_path))])
    assert code == 0


# ---------------------------------------------------------------------------
# 台账按代过滤
#
# 台账跨代累积（promote 不清空，这是对的 —— 要能回看整条链）。但读台账
# 回答关于本代的问题时必须按代过滤：promote 换了参考，跨代的数全不可比。
# ---------------------------------------------------------------------------

def ledger_entry(generation, **kw):
    base = {"generation": generation, "decision": "D0.1",
            "verdict": campaign.VERDICT_ACCEPT, "gain": 1.4,
            "time_us": 210.0, "counts_toward_best": True,
            "kb_admissible": True, "invalidated": False}
    base.update(kw)
    return base


def test_generation_entries_filters_by_current_generation():
    state = fresh_state()
    state["generation"] = 1
    state["ledger"] = [ledger_entry(0), ledger_entry(1, decision="D1.1")]
    kept = campaign.generation_entries(state)
    assert [e["decision"] for e in kept] == ["D1.1"]


def test_generation_entries_keeps_entries_without_generation():
    """老 state 没记代号时当作本代 —— 宁可多算一条，不要静默丢掉证据。"""
    state = fresh_state()
    entry = ledger_entry(0)
    del entry["generation"]
    state["ledger"] = [entry]
    assert len(campaign.generation_entries(state)) == 1


def test_graph_not_exhausted_right_after_promote():
    """上一代试过的决策在本代是全新的 —— 参考换了，结论可能完全不同。"""
    doc = milestone({"D0.1": [], "D0.2": []})
    state = fresh_state()
    state["ledger"] = [ledger_entry(0, decision="D0.1"),
                       ledger_entry(0, decision="D0.2")]
    assert campaign.graph_exhausted(state, doc) is True

    state["generation"] = 1
    assert campaign.graph_exhausted(state, doc) is False


def test_previous_snapshot_does_not_cross_generations():
    """上一代的签名来自另一个二进制，拿它做差会把两代差异算到本代头上。"""
    state = fresh_state()
    state["ledger"] = [ledger_entry(0, signature_snapshot={CHANNEL: 1.1})]
    state["generation"] = 1
    snapshot, label = campaign.previous_snapshot(state)
    assert label == "参考实现", label


# ===========================================================================
# 结果文件的入口门禁
#
# 裁决权在脚本手里，但输入权在调用方手里。不查来源的话，裁决严格建立在
# 一份没人验证过的 JSON 上。
# ===========================================================================

def test_record_refuses_failed_gating(tmp_path):
    """评测自己判了 failed，裁决不能当成功收下。"""
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0
    result = write_result(tmp_path, "r.json", gating="failed")
    code, out = run(["--state", str(state_file), "record",
                     "--result", str(result), "--decision", "D0.1"])
    assert code == 1
    assert "gating" in out


def test_record_refuses_generalization_regression(tmp_path):
    """泛化 shape 退化只体现在 gating 里 —— comparison.* 只覆盖 target。"""
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0
    result = write_result(tmp_path, "r.json",
                          gating="generalization_regression")
    code, _ = run(["--state", str(state_file), "record",
                   "--result", str(result), "--decision", "D0.1"])
    assert code == 1


def test_record_refuses_result_from_another_device(tmp_path):
    """换卡之后参考签名、噪声底、绝对耗时全部不可比，而不可比是静默的。"""
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file, device_id=4))[0] == 0
    doc = eval_result()
    doc["device_id"] = 7
    result = tmp_path / "r.json"
    result.write_text(json.dumps(doc), encoding="utf-8")
    code, out = run(["--state", str(state_file), "record",
                     "--result", str(result), "--decision", "D0.1"])
    assert code == 1
    assert "device 7" in out


def test_record_accepts_matching_device(tmp_path):
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file, device_id=4))[0] == 0
    doc = eval_result()
    doc["device_id"] = 4
    result = tmp_path / "r.json"
    result.write_text(json.dumps(doc), encoding="utf-8")
    code, _ = run(["--state", str(state_file), "record",
                   "--result", str(result), "--decision", "D0.1"])
    assert code == 0


def test_record_refuses_when_measured_noise_exceeds_stored_floor(tmp_path):
    """噪声底是 init 时量的；环境变吵之后旧判据就失准了。

    判据是倍数而不是「超过一点」—— 后者会让这道门天天响，然后被反射性豁免。
    """
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file, noise=2.0))[0] == 0
    result = write_result(tmp_path, "r.json", cv=9.0)
    code, out = run(["--state", str(state_file), "record",
                     "--result", str(result), "--decision", "D0.1"])
    assert code == 1
    assert "9.00%" in out
    assert "倍" in out


def test_record_only_hints_when_noise_is_slightly_above_floor(tmp_path):
    """略高于噪声底是正常抽样波动，提示即可，不该拦。"""
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file, noise=2.0))[0] == 0
    result = write_result(tmp_path, "r.json", cv=2.4)
    code, out = run(["--state", str(state_file), "record",
                     "--result", str(result), "--decision", "D0.1"])
    assert code == 0
    assert "略高于" in out


def test_record_allows_stale_floor_on_request(tmp_path):
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file, noise=2.0))[0] == 0
    result = write_result(tmp_path, "r.json", cv=9.0)
    code, _ = run(["--state", str(state_file), "record",
                   "--result", str(result), "--decision", "D0.1",
                   "--allow-stale-noise-floor"])
    assert code == 0


def test_record_without_milestone_stamps_the_ledger(tmp_path):
    """漏传 milestone 会关掉两个重入口 —— 这件事必须留痕，否则事后分不清
    「重入口没触发」和「重入口根本没被评估」。"""
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0
    result = write_result(tmp_path, "r.json")
    code, out = run(["--state", str(state_file), "record",
                     "--result", str(result), "--decision", "D0.1"])
    assert code == 0
    assert "外层重入口" in out
    ledger = json.loads(state_file.read_text(encoding="utf-8"))["ledger"]
    assert ledger[-1]["gates_not_evaluated"] == ["graph_exhausted",
                                                 "gate_impasse"]


def test_init_records_milestone_path_for_later_records(tmp_path):
    """记进 state 之后 record 不必每次都传 —— 省掉那个最容易漏的参数。"""
    ms = tmp_path / "ms.json"
    ms.write_text(json.dumps(milestone()), encoding="utf-8")
    state_file = tmp_path / "s.json"
    code, _ = run(init_argv(state_file) + ["--milestone-path", str(ms)])
    assert code == 0
    assert json.loads(state_file.read_text())["milestone_path"] == str(ms.resolve())

    result = write_result(tmp_path, "r.json")
    code, out = run(["--state", str(state_file), "record",
                     "--result", str(result), "--decision", "D0.1"])
    assert code == 0
    assert "外层重入口" not in out


def test_budget_has_a_default(tmp_path):
    """不限预算意味着循环没有任何次数出口。"""
    state_file = tmp_path / "s.json"
    argv = [a for a in init_argv(state_file)]
    code, _ = run(argv)
    assert code == 0
    assert json.loads(state_file.read_text())["budget"]["max_records"] == \
        campaign.DEFAULT_BUDGET


# ===========================================================================
# 边界自检
#
# 整套方法建立在「参考只能测不能读」上，而这件事此前从没在开 campaign 时
# 验证过。三种失效都会让整个 campaign 在无防护下跑完，而所有文档都说
# 边界生效：默认安装不注册 hook、清单被清理、FORGE_SEALED_PATHS 指错。
# ===========================================================================

def seal_a_reference(tmp_path):
    """铺一份真的封存产物，返回清单路径。"""
    scripts = (Path(__file__).resolve().parent.parent
               / "skills" / "forge-campaign" / "scripts")
    sys.path.insert(0, str(scripts))
    import seal_reference  # noqa: PLC0415

    artifact = tmp_path / "ref"
    artifact.mkdir()
    (artifact / "k.cpp").write_text("class KernelSecretCore {};\n",
                                    encoding="utf-8")
    work_dir = tmp_path / "forge_work"
    seal_reference.seal(artifact, "demo_op", work_dir,
                        use_public_api_filter=False, kb_root=tmp_path / "kb")
    return work_dir / "sealed_paths.txt"


def test_boundary_self_check_passes_with_a_real_seal(tmp_path):
    sealed = seal_a_reference(tmp_path)
    argv = [a for a in init_argv(tmp_path / "s.json")
            if a != "--allow-no-boundary"]
    code, out = run(argv + ["--sealed-paths", str(sealed)])
    assert code == 0, out
    assert "边界自检通过" in out


def test_boundary_self_check_refuses_without_a_seal(tmp_path):
    """没封存就开 campaign —— 架构结论无法证明是反推的。"""
    argv = [a for a in init_argv(tmp_path / "s.json")
            if a != "--allow-no-boundary"]
    code, out = run(argv)
    assert code == 2
    assert "封存清单" in out


def test_boundary_self_check_refuses_an_empty_list(tmp_path):
    """清单在但是空的 —— 没有任何路径被保护。"""
    sealed = tmp_path / "sealed_paths.txt"
    sealed.write_text("# 只有注释\n", encoding="utf-8")
    argv = [a for a in init_argv(tmp_path / "s.json")
            if a != "--allow-no-boundary"]
    code, out = run(argv + ["--sealed-paths", str(sealed)])
    assert code == 2
    assert "空的" in out


def test_boundary_self_check_records_the_list_in_state(tmp_path):
    sealed = seal_a_reference(tmp_path)
    argv = [a for a in init_argv(tmp_path / "s.json")
            if a != "--allow-no-boundary"]
    assert run(argv + ["--sealed-paths", str(sealed)])[0] == 0
    state = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert state["sealed_paths"] == str(sealed.resolve())


def test_probe_boundary_detects_a_hook_that_does_not_block(tmp_path):
    """hook 对已封存路径返回 0 时必须被识别出来。

    这是「默认安装没注册 PreToolUse」的等价形态：脚本在、清单在、
    但那条路此刻是通的。
    """
    sealed = seal_a_reference(tmp_path)
    # 把逃生舱打开来模拟「hook 存在但不拦」
    import os  # noqa: PLC0415
    saved = os.environ.get("FORGE_BOUNDARY_HOOK_DISABLE")
    os.environ["FORGE_BOUNDARY_HOOK_DISABLE"] = "1"
    try:
        ok, detail = campaign.probe_boundary(sealed)
    finally:
        if saved is None:
            os.environ.pop("FORGE_BOUNDARY_HOOK_DISABLE", None)
        else:
            os.environ["FORGE_BOUNDARY_HOOK_DISABLE"] = saved
    # probe_boundary 自己会清掉逃生舱，所以它应当仍然拦得住
    assert ok is True, detail


# ===========================================================================
# promote 必须真的封存新参考
#
# 此前 promote 只写 state["reference"]["sealed"] = True 这个字段，而全仓只有
# reference_library.ingest 会调 seal_reference.seal —— promote 走的是
# add_generation，不封。于是第 1 代起编码 agent 可以直接读第 0 代那份自己
# 写的 kernel（它就是新参考），而台账里写着「源码不跨代边界」。
# ===========================================================================

def shipped_state(tmp_path, artifact):
    return {
        "schema_version": campaign.SCHEMA_VERSION,
        "operator": "demo", "input_shape": "B=8", "arch": "DAV_2201",
        "reference": {"kind": "builtin_aclnn", "id": "ref", "sealed": True,
                      "generation": 0},
        "state": campaign.STATE_SHIPPED, "generation": 0,
        "noise_floor_pct": 2.0, "target_gain": 1.05,
        # 两次读数都达标 —— promote 要求 best 经过独立复测
        "best": {"gain": 1.30, "time_us": 210.0, "artifact": str(artifact),
                 "decision": "D0.1", "readings": [1.30, 1.28]},
        "budget": {"max_records": 60, "used": 3},
        "stagnation": {"since_improvement": 0},
        "cursor": {"milestone": None, "decision": None, "iteration": 0},
        # 这一代的知识已经收割过（写了一张卡片）—— 否则 promote 会先被
        # 知识欠账那道闸拦住，而这些用例查的是封存行为。
        "ledger": [{"generation": 0, "decision": "D0.1", "verdict": "accept",
                    "gain": 1.30, "time_us": 210.0,
                    "counts_toward_best": True, "kb_admissible": True,
                    "recipes_invented": ["R-2201-001"]}],
        "device": {"device_id": 0},
    }


def test_promote_seals_the_new_reference(tmp_path):
    artifact = tmp_path / "art"
    artifact.mkdir()
    (artifact / "k.cpp").write_text("class MyGen0Kernel {};\n", encoding="utf-8")
    state_file = tmp_path / "s.json"
    state_file.write_text(json.dumps(shipped_state(tmp_path, artifact)),
                          encoding="utf-8")

    code, out = run(["--state", str(state_file), "promote",
                     "--artifact", str(artifact),
                     "--library", str(tmp_path / "lib"), "--note", "t"])
    assert code == 0, out
    assert "已封存新参考" in out

    sealed = tmp_path / "sealed_paths.txt"
    assert sealed.is_file(), "promote 没有产出封存清单"
    body = sealed.read_text(encoding="utf-8")
    assert "k.cpp" in body


def test_promote_points_the_reference_at_the_library_copy(tmp_path):
    """参考路径要指向库里的只读拷贝，不是工作区里的候选目录。

    后者可读可写 —— 改动它会直接改掉下一代的计时基线。
    """
    artifact = tmp_path / "art"
    artifact.mkdir()
    (artifact / "k.cpp").write_text("class MyGen0Kernel {};\n", encoding="utf-8")
    state_file = tmp_path / "s.json"
    state_file.write_text(json.dumps(shipped_state(tmp_path, artifact)),
                          encoding="utf-8")
    library = tmp_path / "lib"

    assert run(["--state", str(state_file), "promote",
                "--artifact", str(artifact), "--library", str(library),
                "--note", "t"])[0] == 0

    ref_path = json.loads(state_file.read_text())["reference"]["path"]
    assert str(library) in ref_path
    assert ref_path != str(artifact)


def test_generation_manifest_records_its_own_directory(tmp_path):
    """manifest 要能告诉调用方自己在库里的哪个目录。

    索引里一直有这个字段，manifest 里没有 —— 于是拿到 manifest 的调用方
    只能自己拼目录名，拼错就静默退回候选目录（promote 就这么退回过）。
    """
    import reference_library as rl  # noqa: PLC0415

    artifact = tmp_path / "art"
    artifact.mkdir()
    (artifact / "k.cpp").write_text("x\n", encoding="utf-8")
    manifest = rl.add_generation(
        tmp_path / "lib", operator="demo", arch="DAV_2201",
        input_shape="B=8", kind="sealed_kernel", identifier="g0",
        artifact=str(artifact), mechanism_verified=True)
    assert manifest["dir"]
    assert (tmp_path / "lib" / manifest["artifact_dir"]).is_dir()


# ===========================================================================
# 确认复测
#
# best 是历史最大值，而每次 record 是一次抽样。确认复测回答的是「上次那个
# 增益，再测一次还在不在」—— 它不是一次新候选：跑的是同一份代码，所以
# 机制必然「没有因本次改动而变化」，走 record 会被判 vacuous、不计入 best，
# 确认永远无法完成。所以它必须是独立入口。
# ===========================================================================

def test_confirm_requires_the_same_decision():
    state = fresh_state(target=1.05)
    state["best"] = {"gain": 1.20, "decision": "D0.1", "readings": [1.20]}
    with pytest.raises(campaign.CampaignError, match="D0.1"):
        campaign.confirm_best(state, result(1.18), "D0.2")


def test_confirm_does_not_count_toward_best_or_kb():
    state = fresh_state(target=1.05)
    apply(state, result(1.20), "D0.1",
                          milestone_doc=milestone(), signature=signature(0.9))
    entry = campaign.confirm_best(state, result(1.18), "D0.1")
    assert entry["verdict"] == campaign.VERDICT_CONFIRM
    assert entry["counts_toward_best"] is False
    assert entry["kb_admissible"] is False
    assert entry["reading_of"] == "D0.1"


def test_confirm_takes_the_minimum_reading():
    """多次抽样取最大就是让运气决定 best；取最小才是「至少能稳定做到多少」。"""
    state = fresh_state(target=1.05)
    apply(state, result(1.30), "D0.1",
                          milestone_doc=milestone(), signature=signature(0.9))
    assert state["best"]["gain"] == pytest.approx(1.30)
    campaign.confirm_best(state, result(1.12), "D0.1")
    assert state["best"]["gain"] == pytest.approx(1.12)


def test_confirm_that_falls_short_does_not_ship():
    """复测没达标 —— 上次那个读数就是一次偏高抽样，这正是复测要查的。"""
    state = fresh_state(target=1.10)
    apply(state, result(1.20), "D0.1",
                          milestone_doc=milestone(), signature=signature(0.9))
    campaign.confirm_best(state, result(1.02), "D0.1")
    assert state["state"] != campaign.STATE_SHIPPED
    assert campaign._best_is_confirmed(state) is False


def test_confirm_refuses_failed_precision():
    state = fresh_state(target=1.05)
    apply(state, result(1.20), "D0.1",
                          milestone_doc=milestone(), signature=signature(0.9))
    with pytest.raises(campaign.CampaignError, match="精度"):
        campaign.confirm_best(state, result(1.18, precision=False), "D0.1")


def test_promote_refuses_unconfirmed_best(tmp_path):
    artifact = tmp_path / "art"
    artifact.mkdir()
    (artifact / "k.cpp").write_text("x\n", encoding="utf-8")
    base = shipped_state(tmp_path, artifact)
    base["best"]["readings"] = [1.30]      # 只测过一次
    state_file = tmp_path / "s.json"
    state_file.write_text(json.dumps(base), encoding="utf-8")

    code, out = run(["--state", str(state_file), "promote",
                     "--artifact", str(artifact),
                     "--library", str(tmp_path / "lib")])
    assert code == 1
    assert "独立复测" in out


def test_promote_allows_unconfirmed_best_with_force(tmp_path):
    artifact = tmp_path / "art"
    artifact.mkdir()
    (artifact / "k.cpp").write_text("x\n", encoding="utf-8")
    base = shipped_state(tmp_path, artifact)
    base["best"]["readings"] = [1.30]
    state_file = tmp_path / "s.json"
    state_file.write_text(json.dumps(base), encoding="utf-8")

    code, _ = run(["--state", str(state_file), "promote",
                   "--artifact", str(artifact),
                   "--library", str(tmp_path / "lib"), "--force"])
    assert code == 0


# ===========================================================================
# 机制行阈值要受参考实测值约束
#
# 不约束的后果是实测出来的：一条锚在参考值 1.6 的停顿行被写成 ≤6.0，于是
# 实现以**两倍于参考**的等待量舒舒服服坐在带内，轮轮过门禁。带宽宽出参考
# 几倍的验收线只剩仪式价值。
# ===========================================================================

def banded_line(threshold, rationale=None):
    line = {"id": "M1", "label": "supply_keepup",
            "anchor": {"expr": f"{CHANNEL} < {threshold}"}}
    if rationale:
        line["band_rationale"] = rationale
    return line


def test_threshold_far_wider_than_reference_is_refuted():
    out = campaign.verify_mechanism_line(
        banded_line(6.0), signature(5.0), {CHANNEL: 5.5},
        reference_snapshot={CHANNEL: 1.6})
    assert out["verified"] is False
    assert "宽出参考实测值" in out["reason"]


def test_threshold_inside_the_band_verifies():
    out = campaign.verify_mechanism_line(
        banded_line(2.0), signature(1.1), {CHANNEL: 5.5},
        reference_snapshot={CHANNEL: 1.6})
    assert out["verified"] is True
    assert out["band_cap"] == pytest.approx(1.6 * campaign.MECHANISM_BAND_FACTOR)


def test_band_rationale_allows_a_wider_threshold():
    """更宽必须是一个有记录的决定，而不是一个随手写大的数字。"""
    out = campaign.verify_mechanism_line(
        banded_line(6.0, "本形状下参考自身也在 5.x；1.6 是另一 shape 的读数"),
        signature(5.0), {CHANNEL: 5.5}, reference_snapshot={CHANNEL: 1.6})
    assert out["verified"] is True


def test_band_uses_reference_not_last_round():
    """带宽要绑**参考**值，不是上一轮值。

    拿上一轮读数当锚会让带宽随实现一起漂：上一轮坐在 5.5（参考 1.6），
    下一轮的带宽就变成 6.9 —— 参考从此退出判据。
    """
    out = campaign.verify_mechanism_line(
        banded_line(6.0), signature(5.0), {CHANNEL: 5.5},
        reference_snapshot={CHANNEL: 1.6})
    assert out["verified"] is False
    assert out["reference_value"] == pytest.approx(1.6)


def test_lower_bound_thresholds_are_not_banded():
    """下界型（越大越好）语义相反，参考值在它那一侧不构成同样的约束。"""
    line = {"id": "M1", "label": "cube_saturation",
            "anchor": {"expr": f"{CHANNEL} > 0.5"}}
    out = campaign.verify_mechanism_line(
        line, signature(2.0), {CHANNEL: 0.2},
        reference_snapshot={CHANNEL: 0.1})
    assert out.get("band_cap") is None


# ===========================================================================
# M0 对账
#
# 门禁只拿课程表自己声明的值做比较，所以课程表没写到的通道可以是参考的
# 任意倍数而不触发任何东西。
# ===========================================================================

def test_audit_finds_unclaimed_channels():
    other = "msprof:ResourceConflictRatio.csv:aiv_mte3_wait_ratio"
    state = fresh_state()
    state["reference_snapshot"] = {CHANNEL: 4.2, other: 0.3}
    sig = {"trust": {"trustworthy": True}, "stalls": {"wait": {
        "mte2_wait": {"value": 1.1, "channel": CHANNEL, "present": True},
        "mte3_wait": {"value": 6.0, "channel": other, "present": True}}}}
    findings = campaign.audit_against_reference(state, sig)
    assert [f["channel"] for f in findings] == [other]
    assert findings[0]["ratio"] == pytest.approx(20.0)


def test_audit_ignores_channels_inside_the_band():
    state = fresh_state()
    state["reference_snapshot"] = {CHANNEL: 4.2}
    sig = signature(4.5)
    assert campaign.audit_against_reference(state, sig) == []


def test_audit_ignores_two_near_zero_values():
    """0.01 vs 0.03 是 3 倍，但两边都接近 0 时倍数没有意义。"""
    state = fresh_state()
    state["reference_snapshot"] = {CHANNEL: 0.0}
    sig = signature(0.0)
    assert campaign.audit_against_reference(state, sig) == []


# ===========================================================================
# 重开必须定价
#
# 重开会按依赖反向闭包作废整条下游的实现、把 best 归零、三个计数器清零 ——
# 这是整套流程里代价最大的单个动作，而早先它只要一句自由文本理由。
# 上游的规矩是「没有价格就没有重开」：拿 Δ 去比对探针读数或定价表，
# 对不上就拒绝、课程表原样不动。
# ===========================================================================

def lever_store(tmp_path, *names, noise=2.0):
    path = tmp_path / "levers.json"
    path.write_text(json.dumps({
        "schema_version": "1.0",
        "baseline": {"label": "baseline", "time_us": 300.0},
        "noise_floor_pct": noise,
        "variants": [{"levers": [n], "time_us": 280.0} for n in names],
    }), encoding="utf-8")
    return path


def reopen_args(state_file, milestone_file, **kw):
    import argparse  # noqa: PLC0415
    base = dict(state=str(state_file), milestone=str(milestone_file),
                decision="D0.1", reason="x", as_json=False,
                price=None, lever_store=None, kb_root=None, force=False)
    base.update(kw)
    return argparse.Namespace(**base)


def reopen_fixture(tmp_path):
    doc = milestone({"D0.1": [], "D0.2": ["D0.1"]})
    state = fresh_state(target=10.0, threshold=99)
    apply(state, result(speedup=1.30), "D0.1",
                          milestone_doc=doc, signature=signature(0.9))
    state_file = tmp_path / "state.json"
    campaign.save_state(state_file, state)
    milestone_file = tmp_path / "m.json"
    milestone_file.write_text(json.dumps(doc), encoding="utf-8")
    return state_file, milestone_file


def test_reopen_without_price_is_refused(tmp_path):
    state_file, milestone_file = reopen_fixture(tmp_path)
    assert campaign.cmd_reopen(reopen_args(state_file, milestone_file)) == 2


def test_reopen_price_must_reference_a_priced_lever(tmp_path):
    """引用一个没测过的手段，等于把推断当价格用。"""
    state_file, milestone_file = reopen_fixture(tmp_path)
    lever_store(tmp_path, "ub_resident")
    code = campaign.cmd_reopen(reopen_args(
        state_file, milestone_file, price="8.4% via lever:never_measured",
        lever_store=str(tmp_path / "levers.json")))
    assert code == 1


def test_reopen_price_must_clear_the_noise_floor(tmp_path):
    """价格本身落在噪声里，就不构成重开的理由。"""
    state_file, milestone_file = reopen_fixture(tmp_path)
    lever_store(tmp_path, "ub_resident")
    code = campaign.cmd_reopen(reopen_args(
        state_file, milestone_file, price="0.5% via lever:ub_resident",
        lever_store=str(tmp_path / "levers.json")))
    assert code == 1


def test_reopen_with_a_verified_price_proceeds(tmp_path):
    state_file, milestone_file = reopen_fixture(tmp_path)
    lever_store(tmp_path, "ub_resident")
    code = campaign.cmd_reopen(reopen_args(
        state_file, milestone_file, price="8.4% via lever:ub_resident",
        lever_store=str(tmp_path / "levers.json")))
    assert code == 0
    reloaded = json.loads(state_file.read_text(encoding="utf-8"))
    assert reloaded["reopen"]["total"] == 1
    assert reloaded["ledger"][-1]["price"] == "8.4% via lever:ub_resident"


def test_reopen_price_shape_is_checked():
    ok, detail = campaign.verify_reopen_price("因为我觉得不对", "nonexistent")
    assert ok is False
    assert "形状" in detail


def test_fruitless_reopens_are_capped(tmp_path):
    """连续无果的重开要拦住 —— 否则「重开→停滞→再重开」可以永久循环，
    而每一轮的 status 看起来都健康。"""
    state_file, milestone_file = reopen_fixture(tmp_path)
    lever_store(tmp_path, "ub_resident")
    args = dict(price="8.4% via lever:ub_resident",
                lever_store=str(tmp_path / "levers.json"))
    for _ in range(campaign.REOPEN_MAX_FRUITLESS):
        assert campaign.cmd_reopen(
            reopen_args(state_file, milestone_file, **args)) == 0
    assert campaign.cmd_reopen(
        reopen_args(state_file, milestone_file, **args)) == 1


def test_reopen_counter_survives_across_reopens(tmp_path):
    """这个计数不能随重开归零 —— 否则上限等于不存在。"""
    state_file, milestone_file = reopen_fixture(tmp_path)
    lever_store(tmp_path, "ub_resident")
    campaign.cmd_reopen(reopen_args(
        state_file, milestone_file, price="8.4% via lever:ub_resident",
        lever_store=str(tmp_path / "levers.json")))
    reloaded = json.loads(state_file.read_text(encoding="utf-8"))
    assert reloaded["reopen"]["fruitless"] == 1
    # 而三个停滞类计数器确实该归零（新架构假设要给足机会）
    assert reloaded["stagnation"]["since_improvement"] == 0


def test_raising_best_clears_the_fruitless_counter():
    """上一次重开带来了新下界 —— 它不是徒劳的。"""
    state = fresh_state()
    state["reopen"] = {"total": 1, "fruitless": 1}
    apply(state, result(speedup=1.30), "D0.1",
                          milestone_doc=milestone({"D0.1": []}),
                          signature=signature(0.9))
    assert state["reopen"]["fruitless"] == 0


# ---------------------------------------------------------------------------
# 原子组成员互相牵连
#
# atomic_group 标的是「必须一次提交落地」的决策集，而闭包早先只走
# depends_on —— 重开组里一个成员，兄弟成员的台账仍然有效，树上留着一个
# 原子组的真子集，而退化会被归因给刚重开的那条决策。
# ---------------------------------------------------------------------------

def test_atomic_group_members_are_invalidated_together():
    doc = milestone({"D1.1": [], "D1.2": []})
    for d in doc["milestones"][0]["decisions"]:
        d["atomic_group"] = "g1"
    affected = campaign.dependents_closure(doc, "D1.1")
    assert "D1.2" in affected


def test_atomic_group_does_not_pull_in_unrelated_decisions():
    doc = milestone({"D1.1": [], "D1.2": [], "D1.3": []})
    decisions = doc["milestones"][0]["decisions"]
    decisions[0]["atomic_group"] = "g1"
    decisions[1]["atomic_group"] = "g1"
    affected = campaign.dependents_closure(doc, "D1.1")
    assert "D1.2" in affected
    assert "D1.3" not in affected


# ===========================================================================
# 代码身份
#
# `previous_snapshot` 的整个前提是「被保留下来的最后一条就是当前代码的样子」。
# 没有指纹这件事无法核实：编码层该回滚却没回滚、或在一棵更早的树上接着改，
# 做差基线就是错的，而算出来的百分比看着完全正常。
# ===========================================================================

def test_directory_fingerprint_changes_with_content(tmp_path):
    root = tmp_path / "cand"
    root.mkdir()
    (root / "k.cpp").write_text("v1\n", encoding="utf-8")
    first = campaign.directory_fingerprint(root)
    (root / "k.cpp").write_text("v2\n", encoding="utf-8")
    assert campaign.directory_fingerprint(root) != first


def test_directory_fingerprint_is_stable(tmp_path):
    root = tmp_path / "cand"
    (root / "sub").mkdir(parents=True)
    (root / "k.cpp").write_text("x\n", encoding="utf-8")
    (root / "sub" / "t.h").write_text("y\n", encoding="utf-8")
    assert campaign.directory_fingerprint(root) == \
        campaign.directory_fingerprint(root)


def test_fingerprint_notices_a_new_file(tmp_path):
    """多出来的文件也是变化 —— 只比已知文件会漏掉新增。"""
    root = tmp_path / "cand"
    root.mkdir()
    (root / "k.cpp").write_text("x\n", encoding="utf-8")
    before = campaign.directory_fingerprint(root)
    (root / "extra.cpp").write_text("z\n", encoding="utf-8")
    assert campaign.directory_fingerprint(root) != before


def test_record_stores_code_identity(tmp_path):
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0
    artifact = tmp_path / "cand"
    artifact.mkdir()
    (artifact / "k.cpp").write_text("x\n", encoding="utf-8")
    res = write_result(tmp_path, "r.json")

    code, _ = run(["--state", str(state_file), "record", "--result", str(res),
                   "--decision", "D0.1", "--artifact", str(artifact)])
    assert code == 0
    entry = json.loads(state_file.read_text(encoding="utf-8"))["ledger"][-1]
    assert entry["artifact_sha"] == campaign.directory_fingerprint(artifact)


def test_record_warns_when_code_did_not_change(tmp_path):
    """对未改动的树重测只是多抽一次噪声；当成新候选会让 best 由运气决定。"""
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0
    artifact = tmp_path / "cand"
    artifact.mkdir()
    (artifact / "k.cpp").write_text("x\n", encoding="utf-8")
    res = write_result(tmp_path, "r.json")
    args = ["--state", str(state_file), "record", "--result", str(res),
            "--decision", "D0.1", "--artifact", str(artifact)]
    run(args)
    _, out = run(args)
    assert "代码没有变化" in out
    assert "campaign.py confirm" in out


def test_record_warns_without_code_identity(tmp_path):
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0
    res = write_result(tmp_path, "r.json")
    _, out = run(["--state", str(state_file), "record", "--result", str(res),
                  "--decision", "D0.1"])
    assert "代码身份" in out


def test_record_records_the_milestone_fingerprint(tmp_path):
    ms = tmp_path / "m.json"
    ms.write_text(json.dumps(milestone()), encoding="utf-8")
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0
    res = write_result(tmp_path, "r.json")
    run(["--state", str(state_file), "record", "--result", str(res),
         "--decision", "D0.1", "--milestone", str(ms)])
    entry = json.loads(state_file.read_text(encoding="utf-8"))["ledger"][-1]
    assert entry["milestone_sha256"]


def test_record_flags_a_changed_milestone(tmp_path):
    """阈值被改宽、或机制行被删掉，早先在台账里一点痕迹都没有。"""
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0
    res = write_result(tmp_path, "r.json")
    ms = tmp_path / "m.json"

    ms.write_text(json.dumps(milestone()), encoding="utf-8")
    run(["--state", str(state_file), "record", "--result", str(res),
         "--decision", "D0.1", "--milestone", str(ms)])

    widened = milestone(mechanism_expr=f"{CHANNEL} < 99.0")
    ms.write_text(json.dumps(widened), encoding="utf-8")
    _, out = run(["--state", str(state_file), "record", "--result", str(res),
                  "--decision", "D0.1", "--milestone", str(ms)])
    assert "不是同一份课程表" in out


def test_previous_snapshot_label_names_the_artifact():
    state = fresh_state()
    state["ledger"] = [{
        "generation": 0, "decision": "D0.1", "verdict": campaign.VERDICT_ACCEPT,
        "signature_snapshot": {CHANNEL: 1.1}, "artifact_sha": "abc123",
        "counts_toward_best": True,
    }]
    _, label = campaign.previous_snapshot(state)
    assert "abc123" in label


# ===========================================================================
# 结构核验进裁决
#
# 停顿指标能被别的改动顶下去（减小 tile、改对齐都会让它降），于是一个单缓冲
# 回退的实现可以凭一个恰好变好的停顿数拿到「机制已核实」，而双缓冲从未存在。
# 结构在不在是二值事实，顶不下去 —— 所以它排在停顿之前，而且结论优先。
# ===========================================================================

def structure_report(*verdicts):
    return {"schema_version": "1.0", "assertions": [
        {"assertion": f"a{i}", "verdict": v, "detail": "", "observed": None}
        for i, v in enumerate(verdicts)]}


def test_absent_structure_overrides_a_passing_stall():
    """停顿成立、方向也对，但结构不在 —— 不能判「机制已核实」。"""
    state = fresh_state()
    entry = apply(
        state, result(speedup=1.30), "D0.1",
        milestone_doc=milestone({"D0.1": []}), signature=signature(0.9),
        structure_report=structure_report("fail"))
    assert entry["structure"]["status"] == "absent"
    assert entry["mechanism"]["status"] == "refuted"
    assert entry["verdict"] == campaign.VERDICT_ACCEPT_UNVERIFIED
    assert entry["counts_toward_best"] is False


def test_absent_structure_is_attributed_to_implementation():
    """归因方向要紧：这是实现没做到，不是课程表解释错了。

    归到课程表那一侧的后果很具体：连续三次把架构层叫回来重写课程表，
    而下一轮实现还是同一个退化回退 —— 改对的东西被改坏、改坏的东西没人碰。
    """
    state = fresh_state()
    entry = apply(
        state, result(speedup=1.30), "D0.1",
        milestone_doc=milestone({"D0.1": []}), signature=signature(0.9),
        structure_report=structure_report("fail"))
    assert entry["mechanism"]["attribution"] == "implementation"


def test_absent_structure_does_not_consume_architect_reentry():
    """实现退化该计入实现失败，不该推高「机制被证伪」的连击。"""
    state = fresh_state()
    for _ in range(3):
        apply(
            state, result(speedup=1.30), "D0.1",
            milestone_doc=milestone({"D0.1": []}), signature=signature(0.9),
            structure_report=structure_report("fail"))
    assert state["mechanism_unverified"]["streak_refuted"] == 0
    assert state["implementation_failures"]["consecutive"] >= 1


def test_present_structure_lets_the_stall_verdict_stand():
    state = fresh_state()
    entry = apply(
        state, result(speedup=1.30), "D0.1",
        milestone_doc=milestone({"D0.1": []}), signature=signature(0.9),
        structure_report=structure_report("pass", "pass"))
    assert entry["structure"]["status"] == "ok"
    assert entry["mechanism"]["status"] == "verified"
    assert entry["verdict"] == campaign.VERDICT_ACCEPT


def test_unchecked_structure_does_not_change_the_verdict():
    """没查不构成任何结论 —— 既不该让它过，也不该让它挂。"""
    state = fresh_state()
    entry = apply(
        state, result(speedup=1.30), "D0.1",
        milestone_doc=milestone({"D0.1": []}), signature=signature(0.9))
    assert entry["structure"]["status"] == "unchecked"
    assert entry["verdict"] == campaign.VERDICT_ACCEPT


def test_unknown_structure_assertions_are_not_a_pass():
    state = fresh_state()
    entry = apply(
        state, result(speedup=1.30), "D0.1",
        milestone_doc=milestone({"D0.1": []}), signature=signature(0.9),
        structure_report=structure_report("pass", "unknown"))
    assert entry["structure"]["status"] == "unchecked"


def test_judge_structure_distinguishes_three_states():
    assert campaign.judge_structure(None)["status"] == "unchecked"
    assert campaign.judge_structure(
        structure_report("pass"))["status"] == "ok"
    assert campaign.judge_structure(
        structure_report("fail"))["status"] == "absent"
    assert campaign.judge_structure(
        structure_report("unknown"))["status"] == "unchecked"


def test_record_hints_when_no_structure_report(tmp_path):
    state_file = tmp_path / "s.json"
    assert run(init_argv(state_file))[0] == 0
    res = write_result(tmp_path, "r.json")
    _, out = run(["--state", str(state_file), "record", "--result", str(res),
                  "--decision", "D0.1"])
    assert "结构核验报告" in out


# ---------------------------------------------------------------------------
# 知识库卡片当重开价格
#
# 上游那句话很直接：**一条转述过来的处方不是价格**。先前 kind == "kb" 这
# 条分支无条件返回通过，附一句「请确认该卡片确实带探针认证」—— 把判定交给
# 了那个正想重开的人，而他没有任何动机去查。
# ---------------------------------------------------------------------------

def _kb_with_card(tmp_path, delta="-24.3%", card_id="R-2201-001"):
    """造一个最小知识库：一张过准入的卡片 + 它的归档。"""
    import yaml  # noqa: PLC0415
    kb_root = tmp_path / "kb"
    (kb_root / "dav2201").mkdir(parents=True)
    (kb_root / "provenance").mkdir(parents=True)
    (kb_root / "provenance" / f"{card_id}.md").write_text(
        "归档", encoding="utf-8")
    meta = {
        "id": card_id,
        "arch": "DAV_2201",
        "mechanism": "UB 常驻复用",
        "symptom": "MTE2 占比显著高于 VEC",
        "symptom_channel": "evaluation_results.json:baseline.pipeline.mte2_pct",
        "operator_families": [],
        "provenance": [{"kind": "public_doc", "ref": "ops/npu-arch/SKILL.md"}],
        "certified_by": {"job": "job 7", "arch": "DAV_2201", "delta": delta},
        "provenance_archive": f"provenance/{card_id}.md",
        "status": "active",
    }
    body = ("\n## 机制\nm\n\n## 触发症状\ns\n\n## 最小可运行例\ne\n\n"
            "## 已知坑\nk\n")
    (kb_root / "dav2201" / f"{card_id}.md").write_text(
        "---\n" + yaml.safe_dump(meta, allow_unicode=True) + "---\n" + body,
        encoding="utf-8")
    return kb_root


def _state_with_noise(tmp_path, noise=2.0):
    state_file = tmp_path / "state.json"
    campaign.save_state(state_file, fresh_state(noise=noise))
    return state_file


def test_kb_price_without_a_kb_root_is_refused(tmp_path):
    """没给库根就无法核对 —— 不能据此放行。"""
    ok, detail = campaign.verify_reopen_price(
        "8.4% via kb:R-2201-001", str(_state_with_noise(tmp_path)))
    assert not ok
    assert "不是价格" in detail


def test_kb_price_for_a_missing_card_is_refused(tmp_path):
    kb_root = _kb_with_card(tmp_path)
    ok, detail = campaign.verify_reopen_price(
        "8.4% via kb:R-2201-999", str(_state_with_noise(tmp_path)),
        kb_root=str(kb_root))
    assert not ok
    assert "没有卡片" in detail


def test_kb_price_for_a_card_that_fails_admission_is_refused(tmp_path):
    """不过准入的卡片不能当价格 —— 它背后那次测量是否成立正是准入在查的事。"""
    kb_root = _kb_with_card(tmp_path)
    (kb_root / "provenance" / "R-2201-001.md").unlink()   # 归档没了
    ok, detail = campaign.verify_reopen_price(
        "8.4% via kb:R-2201-001", str(_state_with_noise(tmp_path)),
        kb_root=str(kb_root))
    assert not ok
    assert "不过准入" in detail


def test_kb_price_cannot_exceed_what_the_card_measured(tmp_path):
    """卡片只能为它测过的那个幅度背书。"""
    kb_root = _kb_with_card(tmp_path, delta="-5.0%")
    ok, detail = campaign.verify_reopen_price(
        "8.4% via kb:R-2201-001", str(_state_with_noise(tmp_path)),
        kb_root=str(kb_root))
    assert not ok
    assert "只有 5.0%" in detail


def test_kb_price_needs_a_percentage_in_the_card(tmp_path):
    """`mte2_pct 61.2 -> 31.0` 是机制证据，不是时间增益。"""
    kb_root = _kb_with_card(tmp_path, delta="mte2_pct 61.2 -> 31.0")
    ok, detail = campaign.verify_reopen_price(
        "8.4% via kb:R-2201-001", str(_state_with_noise(tmp_path)),
        kb_root=str(kb_root))
    assert not ok
    assert "没有百分数" in detail


def test_kb_price_within_the_card_passes(tmp_path):
    """这是「读知识库能省一次 GPU job」那条经济激励的落点。"""
    kb_root = _kb_with_card(tmp_path, delta="-24.3%")
    ok, detail = campaign.verify_reopen_price(
        "8.4% via kb:R-2201-001", str(_state_with_noise(tmp_path)),
        kb_root=str(kb_root))
    assert ok
    assert "不必另测一遍" in detail


def test_kb_price_still_has_to_clear_the_noise_floor(tmp_path):
    """噪声底那道闸在知识库这条路上同样生效。"""
    kb_root = _kb_with_card(tmp_path, delta="-24.3%")
    ok, detail = campaign.verify_reopen_price(
        "1.0% via kb:R-2201-001", str(_state_with_noise(tmp_path, noise=2.0)),
        kb_root=str(kb_root))
    assert not ok
    assert "噪声底" in detail


def test_card_lookup_uses_hwkb_arch_dirs(tmp_path):
    """目录名只有一个真源。

    在这里抄一份 {"DAV_2201": "dav2201"}，两份映射分头漂移时这里只会悄悄
    找不到卡片 —— 而「找不到卡片」会被读成「这张卡片不存在」。
    """
    hwkb = campaign._load_hwkb()
    kb_root = _kb_with_card(tmp_path)
    found = campaign.find_kb_card(kb_root, "R-2201-001",
                                  list(hwkb.ARCH_DIRS.values()))
    assert found is not None and found.name == "R-2201-001.md"


# ---------------------------------------------------------------------------
# 知识欠账
#
# 本仓自己就是这条规则的理由：整套 dav2201 流程建完，库里 1 张卡。知识沉淀
# 排在流程第 11 步 —— 一次长 campaign 的末尾，上下文最紧、最容易被跳过的
# 时刻。靠提示词提醒的事，到那个时刻不会发生。
# ---------------------------------------------------------------------------

def _harvest_state(entries):
    state = fresh_state()
    state["generation"] = 0
    state["ledger"] = [dict(generation=0, **e) for e in entries]
    return state


def test_admissible_entry_without_a_card_is_a_debt():
    debts = campaign.harvest_debt(_harvest_state([
        {"decision": "D0.1", "verdict": "accept", "gain": 1.20,
         "kb_admissible": True, "counts_toward_best": True},
    ]))
    assert [d["kind"] for d in debts] == ["admissible"]
    assert debts[0]["decision"] == "D0.1"


def test_admissible_entry_with_a_card_is_settled():
    debts = campaign.harvest_debt(_harvest_state([
        {"decision": "D0.1", "verdict": "accept", "gain": 1.20,
         "kb_admissible": True, "recipes_invented": ["R-2201-001"]},
    ]))
    assert debts == []


def test_entry_that_is_not_admissible_is_not_a_debt():
    """机制未核实的改动不该催人写卡片 —— 那张卡片会把一个未核实的解释
    封成已验证的经验，比不写更糟。"""
    debts = campaign.harvest_debt(_harvest_state([
        {"decision": "D0.1", "verdict": "accept_unverified", "gain": 1.20,
         "kb_admissible": False},
    ]))
    assert debts == []


def test_invalidated_entry_is_not_a_debt():
    debts = campaign.harvest_debt(_harvest_state([
        {"decision": "D0.1", "verdict": "accept", "gain": 1.20,
         "kb_admissible": True, "invalidated": True},
    ]))
    assert debts == []


def test_debt_does_not_cross_generations():
    """上一代的欠账在上一次 promote 就该结清。

    再算一遍只会让列表永远非空，然后 promote 那道闸被反射性豁免掉。
    """
    state = fresh_state()
    state["generation"] = 1
    state["ledger"] = [
        {"generation": 0, "decision": "D0.1", "verdict": "accept",
         "gain": 1.20, "kb_admissible": True},
    ]
    assert campaign.harvest_debt(state) == []


def test_a_decision_that_burned_many_rounds_is_a_debt():
    """花得多说明有东西没被预见到，而那个发现只存在于那次会话里。"""
    entries = [{"decision": "D0.1", "verdict": "revert", "gain": 1.0,
                "kb_admissible": False}
               for _ in range(campaign.KB_TRIPWIRE_ROUNDS - 1)]
    entries.append({"decision": "D0.1", "verdict": "accept_unverified",
                    "gain": 1.20, "kb_admissible": False})
    debts = campaign.harvest_debt(_harvest_state(entries))
    assert [d["kind"] for d in debts] == ["tripwire"]
    assert debts[0]["rounds"] == campaign.KB_TRIPWIRE_ROUNDS


def test_a_quick_decision_is_not_a_tripwire_debt():
    debts = campaign.harvest_debt(_harvest_state([
        {"decision": "D0.1", "verdict": "accept_unverified", "gain": 1.20,
         "kb_admissible": False},
    ]))
    assert debts == []


def test_writing_any_card_settles_the_tripwire_debt():
    """这类欠账的目的是让人去写，写过了就不必再催。"""
    entries = [{"decision": "D0.1", "verdict": "revert", "gain": 1.0,
                "kb_admissible": False}
               for _ in range(campaign.KB_TRIPWIRE_ROUNDS - 1)]
    entries.append({"decision": "D0.1", "verdict": "accept_unverified",
                    "gain": 1.20, "kb_admissible": False,
                    "recipes_invented": ["R-2201-007"]})
    assert campaign.harvest_debt(_harvest_state(entries)) == []


def test_harvest_output_carries_the_readings_needed_to_write_the_card():
    """欠账清单要是还得让人回头翻台账找读数，到那一步就没人写了。"""
    debts = campaign.harvest_debt(_harvest_state([
        {"decision": "D0.1", "verdict": "accept", "gain": 1.20,
         "kb_admissible": True, "symptom": "MTE2 占比 61%",
         "signature_path": "/tmp/sig.json",
         "mechanism": {"status": "verified", "reason": "mte2_wait 18% -> 4%",
                       "lines": [{"label": "SUPPLY_KEEPUP",
                                  "anchor": {"expr": "aiv_mte2_wait < 4.2"}}]},
         },
    ]))
    text = campaign.render_harvest(debts)
    assert "MTE2 占比 61%" in text
    assert "mte2_wait 18% -> 4%" in text
    assert "SUPPLY_KEEPUP" in text
    assert "/tmp/sig.json" in text
    assert "怎么结清" in text


def test_harvest_exit_code_signals_outstanding_debt(tmp_path, capsys):
    import argparse  # noqa: PLC0415
    state_file = tmp_path / "s.json"
    campaign.save_state(state_file, _harvest_state([
        {"decision": "D0.1", "verdict": "accept", "gain": 1.20,
         "kb_admissible": True},
    ]))
    args = argparse.Namespace(state=str(state_file), as_json=False)
    assert campaign.cmd_harvest(args) == 1
    capsys.readouterr()

    campaign.save_state(state_file, _harvest_state([]))
    assert campaign.cmd_harvest(args) == 0


def test_promote_refuses_while_knowledge_debt_is_outstanding(tmp_path):
    """promote 是这一代的闭幕动作，也是最后一个还来得及的时刻。"""
    artifact = tmp_path / "art"
    artifact.mkdir()
    (artifact / "k.cpp").write_text("class K {};\n", encoding="utf-8")
    state = shipped_state(tmp_path, artifact)
    state["ledger"][0].pop("recipes_invented")
    state_file = tmp_path / "s.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")

    code, out = run(["--state", str(state_file), "promote",
                     "--artifact", str(artifact),
                     "--library", str(tmp_path / "lib")])
    assert code == 1
    assert "知识欠账" in out
    # 没有封存：欠账未结时这一代不该被关掉
    assert json.loads(state_file.read_text())["generation"] == 0


def test_promote_allows_unharvested_explicitly(tmp_path):
    """确实不值得入库时要有一条显式出口 —— 而且它会写进台账。"""
    artifact = tmp_path / "art"
    artifact.mkdir()
    (artifact / "k.cpp").write_text("class K {};\n", encoding="utf-8")
    state = shipped_state(tmp_path, artifact)
    state["ledger"][0].pop("recipes_invented")
    state_file = tmp_path / "s.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")

    code, out = run(["--state", str(state_file), "promote",
                     "--artifact", str(artifact),
                     "--library", str(tmp_path / "lib"),
                     "--allow-unharvested"])
    assert code == 0, out
    assert json.loads(state_file.read_text())["generation"] == 1


def test_convergence_stop_warns_but_does_not_block(tmp_path):
    """该停的 campaign 必须能记下「我收敛了」。

    拦住它的后果比丢几条经验更糟：状态机会卡在 implement，之后所有停止
    条件的判断都从一个错的状态出发。
    """
    artifact = tmp_path / "art"
    artifact.mkdir()
    (artifact / "k.cpp").write_text("class K {};\n", encoding="utf-8")
    state = shipped_state(tmp_path, artifact)
    state["ledger"][0].pop("recipes_invented")
    state["best"]["gain"] = 1.005          # 增益落在噪声底以内
    state["best"]["readings"] = [1.005, 1.004]
    state_file = tmp_path / "s.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")

    code, out = run(["--state", str(state_file), "promote",
                     "--artifact", str(artifact),
                     "--library", str(tmp_path / "lib")])
    assert code == 0, out
    assert "收敛" in out
    assert "知识欠账" in out                # 告警，不是拦截
    assert json.loads(state_file.read_text())["state"] == campaign.STATE_STOPPED


# ---------------------------------------------------------------------------
# 围栏裁决
#
# 围栏脚本本来就有，文档也写明它排在任何计时之前 —— 但做裁决的 record 此前
# 完全不看它。于是这道纪律是纯自愿的：跑不跑、结果如何，对裁决没有影响。
# 这正是本轮一直在修的那种形态：一道看起来在、但关不上的门。
# ---------------------------------------------------------------------------

def _verified_accept(state, **kw):
    """一次「增益达标 + 机制已核实」的 record。围栏状态由调用方给。"""
    return campaign.apply_result(
        state, result(speedup=1.30), "D0.1",
        milestone_doc=milestone(), signature=signature(0.1),
        **kw)


def test_clean_fence_lets_a_verified_accept_raise_best():
    state = fresh_state()
    entry = _verified_accept(state, fence_report=clean_fence())
    assert entry["verdict"] == campaign.VERDICT_ACCEPT
    assert entry["counts_toward_best"] is True
    assert entry["fence"]["status"] == "clean"


def test_missing_fence_report_does_not_raise_best():
    """没有围栏证据的一轮不抬高 best。

    best 是 promote 封进下一代参考的那个东西。一个转发给厂商实现的候选
    被封成参考之后，整条自我提升链都建立在厂商实现上。
    """
    state = fresh_state()
    entry = _verified_accept(state)          # 不传 fence_report
    assert entry["verdict"] == campaign.VERDICT_ACCEPT_UNVERIFIED
    assert entry["counts_toward_best"] is not True
    assert entry["fence"]["status"] == "unchecked"
    assert "围栏未查" in entry["reason"]
    # 机制确实核实了 —— 拦的不是机制，是「这个数算不算数」
    assert entry["mechanism"]["status"] == "verified"


def test_fence_violation_reverts_the_round():
    """围栏命中时加速比与精度都不作数 —— 它们不是这份 kernel 挣来的。"""
    state = fresh_state()
    entry = _verified_accept(state, fence_report={
        "findings": [{"rule": "F-ACLNN", "file": "k.cpp", "line": 12}],
        "files_scanned": 3})
    assert entry["verdict"] == campaign.VERDICT_REVERT
    assert entry["failure_kind"] == "fence"
    assert entry["fence"]["status"] == "violated"


def test_fence_violation_is_attributed_to_the_implementation():
    """转发是实现缺陷，不是课程表缺陷 —— 归因方向同结构核验。

    归错的后果很具体：把架构层叫回来重写课程表，而下一轮实现还是同一个转发。
    """
    state = fresh_state()
    _verified_accept(state, fence_report={
        "findings": [{"rule": "F-ACLNN"}], "files_scanned": 3})
    assert state["implementation_failures"]["consecutive"] == 1
    assert state["mechanism_unverified"]["streak_refuted"] == 0


def test_unrecognizable_fence_json_is_not_read_as_clean():
    """拿一份认不出的 JSON 当围栏证据，和没有证据是同一件事。"""
    state = fresh_state()
    entry = _verified_accept(state, fence_report={"ok": True})
    assert entry["fence"]["status"] == "unchecked"
    assert entry["counts_toward_best"] is not True


def test_fence_findings_list_the_rules_that_fired():
    """报告要点名哪条规则命中 —— 否则无从判断是误伤还是真转发。"""
    verdict = campaign.judge_fence({
        "findings": [{"rule": "F-ACLNN"}, {"rule": "F-REF-PATH"},
                     {"rule": "F-ACLNN"}],
        "files_scanned": 5})
    assert verdict["status"] == "violated"
    assert verdict["rules"] == ["F-ACLNN", "F-REF-PATH"]
    assert verdict["count"] == 3


# ---------------------------------------------------------------------------
# 设备占比对账
#
# 两个数早就都在手：signature 的 task_duration_us 来自 msprof（设备侧），
# time_us 来自评测的 NPU Event 计时（墙上）。此前没有任何地方把它们放在
# 一起看过。
# ---------------------------------------------------------------------------

def _sig_with_duration(duration_us):
    doc = signature(0.1)
    doc["launch"] = {"task_duration_us": {
        "value": duration_us, "channel": "msprof:OpBasicInfo.csv"}}
    return doc


def test_healthy_device_share_passes():
    state = fresh_state()
    entry = campaign.apply_result(
        state, result(baseline_us=300.0, evolved_us=200.0), "D0.1",
        milestone_doc=milestone(), signature=_sig_with_duration(180.0),
        fence_report=clean_fence())
    assert entry["device_share"]["ok"] is True
    assert entry["counts_toward_best"] is True


def test_host_dominated_round_does_not_raise_best():
    """设备执行只占墙上时间一小部分时，那个百分比量的是 host 开销。

    而机制核验读的是 msprof 的设备侧通道。两者量的东西几乎不相交时，
    「增益超噪声底 + 机制已核实」就是两个互不相关的事实被钉在一起。
    """
    state = fresh_state()
    entry = campaign.apply_result(
        state, result(baseline_us=300.0, evolved_us=200.0), "D0.1",
        milestone_doc=milestone(), signature=_sig_with_duration(10.0),
        fence_report=clean_fence())
    assert entry["device_share"]["ok"] is False
    assert entry["counts_toward_best"] is not True
    assert "设备占比过低" in entry["reason"]


def test_device_share_is_none_when_the_number_is_missing():
    """对不了账不等于通过 —— 但也不等于不通过。两者要分开。"""
    state = fresh_state()
    entry = campaign.apply_result(
        state, result(speedup=1.30), "D0.1",
        milestone_doc=milestone(), signature=signature(0.1),
        fence_report=clean_fence())
    # signature() 里没有 task_duration_us，而 speedup 形态下也没有绝对耗时
    assert entry["device_share"] is None
    # 对不了账不该把一轮本来合格的记录压下去
    assert entry["counts_toward_best"] is True


def test_device_share_boundary_is_inclusive():
    reconciled = campaign.reconcile_device_share(
        result(baseline_us=300.0, evolved_us=100.0),
        _sig_with_duration(20.0))
    assert reconciled is not None
    assert reconciled["device_share"] == campaign.DEVICE_SHARE_MIN
    assert reconciled["ok"] is True


def test_device_share_explains_what_to_do():
    reconciled = campaign.reconcile_device_share(
        result(baseline_us=300.0, evolved_us=100.0),
        _sig_with_duration(5.0))
    assert reconciled is not None
    assert "host" in reconciled["reason"]
    assert "num_trials" in reconciled["reason"]


# ---------------------------------------------------------------------------
# baseline 不许来自缓存
#
# 配对计时的全部纪律建立在「两侧在同一个 job 里、位置互为镜像」之上。
# 缓存的 baseline 让 burn-in 与 A-B-B-A 镜像同时失效 —— 而且是静默失效。
# ---------------------------------------------------------------------------

def test_cached_baseline_is_refused():
    doc = result(speedup=1.30)
    doc["measurement"] = {"mirror_rounds": 2, "baseline_from_cache": True}
    problems = campaign.check_result_admissible(doc, fresh_state())
    assert any("缓存" in p for p in problems)


def test_freshly_measured_baseline_is_accepted():
    doc = result(speedup=1.30)
    doc["measurement"] = {"mirror_rounds": 2, "baseline_from_cache": False}
    assert campaign.check_result_admissible(doc, fresh_state()) == []


def test_result_without_a_measurement_block_is_not_refused():
    """老结果文件没有 measurement 段。缺字段不该被读成「用了缓存」。"""
    assert campaign.check_result_admissible(result(speedup=1.30),
                                            fresh_state()) == []


def test_cached_baseline_rejection_names_the_position_bias():
    """拒绝理由要给出可核对的量，否则会被当成洁癖而豁免掉。"""
    doc = result(speedup=1.30)
    doc["measurement"] = {"baseline_from_cache": True}
    problems = campaign.check_result_admissible(doc, fresh_state())
    joined = "\n".join(problems)
    assert "2.7%" in joined
    assert "--baseline-cache" in joined


# ---------------------------------------------------------------------------
# 机制失败的归因分流
#
# 上游的规则：「声称无冲突布局而冲突仍高 ⇒ 学生没做到（退化回退）⇒ 加知识库
# 卡片，不要改课程表；只有『参考侧该字段本来就不在声称的位置』才是课程表缺陷」
#
# 此前这条分流只有 structure_probe 一条路能触发，而那要 --structure-report
# （可选参数）。没传时 refuted / vacuous 一律落进 streak_refuted —— 也就是
# 默认判「课程表解释错」，正是这条规则要防的方向。
# ---------------------------------------------------------------------------

def _line(line_id="M1", channel=None, threshold=4.2, op="<", vacuous=False):
    return {"id": line_id, "channel": channel or CHANNEL,
            "threshold": threshold, "operator": op,
            "verified": False, "vacuous": vacuous}


def test_claimed_position_at_the_reference_is_an_implementation_failure():
    """课程表写 `< 4.2`、参考实测 4.2 —— 这是「追平参考」，读法没问题。

    用「参考是否满足这条行的阈值」来判会在这里就错：`4.2 < 4.2` 为假，
    于是一条完全正当的行被判成「读错了参考」。而这是最常见的写法。
    """
    triage = campaign.attribute_mechanism_failure(
        [_line(threshold=4.2)], {CHANNEL: 4.2})
    assert triage["attribution"] == "implementation"
    assert triage["not_delivered_lines"] == ["M1"]


def test_claimed_position_far_below_the_reference_is_a_curriculum_defect():
    """课程表声称参考在 1.0，参考自己是 18.0 —— 差出带外。

    这不只是记错一个数：实现会被要求去追一个参考自己都达不到的目标。
    """
    triage = campaign.attribute_mechanism_failure(
        [_line(threshold=1.0)], {CHANNEL: 18.0})
    assert triage["attribution"] == "curriculum"
    assert triage["misread_lines"] == ["M1"]
    assert "参考自己就不在" in triage["reason"]


def test_a_threshold_looser_than_the_reference_is_still_well_read():
    """阈值比参考宽不构成「读错参考」—— 带宽纪律那条规则管这件事。"""
    triage = campaign.attribute_mechanism_failure(
        [_line(threshold=5.0)], {CHANNEL: 4.2})
    assert triage["attribution"] == "implementation"


def test_band_boundary_is_the_same_factor_as_the_band_discipline():
    """用的是同一个常数 —— 两处对「差多远算差太远」的判断不该各有一套。"""
    reference = 10.0
    inside = reference / campaign.MECHANISM_BAND_FACTOR
    assert campaign.attribute_mechanism_failure(
        [_line(threshold=inside)], {CHANNEL: reference}
    )["attribution"] == "implementation"
    assert campaign.attribute_mechanism_failure(
        [_line(threshold=inside * 0.99)], {CHANNEL: reference}
    )["attribution"] == "curriculum"


def test_lower_bound_lines_flip_the_direction():
    """下界型（> / >=）的语义相反：声称得比参考高太多才是读错了。"""
    assert campaign.attribute_mechanism_failure(
        [_line(threshold=10.0, op=">")], {CHANNEL: 9.0}
    )["attribution"] == "implementation"
    assert campaign.attribute_mechanism_failure(
        [_line(threshold=100.0, op=">")], {CHANNEL: 9.0}
    )["attribution"] == "curriculum"


def test_near_zero_reference_cannot_discriminate():
    """参考侧本来就接近 0 时，「声称的位置」无从比较 —— 任何上界都在它之上。"""
    triage = campaign.attribute_mechanism_failure(
        [_line(threshold=4.2)], {CHANNEL: 0.0})
    assert triage["attribution"] == "implementation"


def test_missing_reference_reading_defaults_to_implementation():
    """判不了时偏向实现失败 —— 因为那是一个**会响**的失败。

    判成课程表缺陷而其实是实现：架构层被连续叫回来重写课程表，而实现还是
    同一个退化回退（不会响）。判成实现失败而其实是课程表：实现失败计数爬到
    阈值 → 状态转 blocked → 停下来交给人（会响）。
    """
    triage = campaign.attribute_mechanism_failure([_line()], None)
    assert triage["attribution"] == "implementation"
    assert triage["untriageable_lines"] == ["M1"]
    assert "代价不对称" in triage["reason"]


def test_unparseable_line_is_untriageable_not_a_curriculum_defect():
    """一条写坏的机制行不该被算成架构层的错。"""
    triage = campaign.attribute_mechanism_failure(
        [{"id": "M1", "verified": False}], {CHANNEL: 4.2})
    assert triage["attribution"] == "implementation"
    assert triage["untriageable_lines"] == ["M1"]


def test_any_misread_line_sends_the_round_to_architecture():
    """只要有一条前提是错的，在它底下调实现就是白费。"""
    triage = campaign.attribute_mechanism_failure(
        [_line("M1", threshold=4.2), _line("M2", threshold=1.0)],
        {CHANNEL: 4.2})
    # M2 的通道与 M1 相同，参考 4.2；阈值 1.0 落在带外
    assert triage["attribution"] == "curriculum"
    assert triage["misread_lines"] == ["M2"]


def test_verified_lines_produce_no_attribution():
    assert campaign.attribute_mechanism_failure(
        [{"id": "M1", "verified": True}], {CHANNEL: 4.2}
    )["attribution"] is None


def test_triage_does_not_need_a_structure_report():
    """分流不依赖任何可选输入 —— 判据是参考快照，而它在 init 就必填。

    这是本条修复的要点：此前只有 structure_probe 能分流，而它是可选的，
    于是**默认路径**仍然走反。
    """
    state = fresh_state(target=10.0, threshold=99)
    entry = apply(state, result(speedup=1.30), "D0.1",
                  milestone_doc=milestone(), signature=signature(9.9))
    assert entry["structure"]["status"] == "unchecked"     # 没传结构报告
    assert entry["mechanism"]["attribution"] == "implementation"
    assert state["implementation_failures"]["consecutive"] == 1
    assert state["mechanism_unverified"]["streak_refuted"] == 0


def test_structure_report_still_overrides_with_the_same_attribution():
    """结构核验那条路继续有效，而且两条路给出同一个归因。"""
    state = fresh_state(target=10.0, threshold=99)
    entry = apply(state, result(speedup=1.30), "D0.1",
                  milestone_doc=milestone(), signature=signature(0.9),
                  structure_report=structure_report("fail"))
    assert entry["mechanism"]["attribution"] == "implementation"
    assert state["implementation_failures"]["consecutive"] == 1


# ---------------------------------------------------------------------------
# 围栏的第三态：阻断项为空，但有未说明的通道
# ---------------------------------------------------------------------------

def test_clean_with_channels_still_raises_best():
    """通道项是「请说明」，不是「你转发了」。

    把它当阻断项会让一批正当实现过不去，而人会反射性地整条豁免掉围栏 ——
    那之后真正的转发也无人把守。
    """
    state = fresh_state()
    entry = _verified_accept(state, fence_report={
        "findings": [], "channels": [{"rule": "F-FFI"}], "files_scanned": 3})
    assert entry["fence"]["status"] == "clean_with_channels"
    assert entry["counts_toward_best"] is True


def test_clean_with_channels_is_recorded_distinctly():
    """「阻断项为空」和「围栏干净」不是一回事，台账要分得开。

    读成 clean 等于让那条路永久免检 —— 而它正是按算子名写的规则看不见的
    那条路。
    """
    verdict = campaign.judge_fence({
        "findings": [], "channels": [{"rule": "F-DLOPEN"},
                                     {"rule": "F-OPAPI-SO"}],
        "files_scanned": 4})
    assert verdict["status"] == "clean_with_channels"
    assert verdict["rules"] == ["F-DLOPEN", "F-OPAPI-SO"]
    assert "转发不需要写出名字" in verdict["reason"]


def test_truly_clean_fence_has_no_channel_note():
    verdict = campaign.judge_fence({"findings": [], "channels": [],
                                    "files_scanned": 3})
    assert verdict["status"] == "clean"


def test_blocking_findings_win_over_channels():
    """两者并存时按阻断处理 —— 阻断项是转发的标志，不是一条待说明的路。"""
    verdict = campaign.judge_fence({
        "findings": [{"rule": "F-ACLNN"}],
        "channels": [{"rule": "F-FFI"}], "files_scanned": 3})
    assert verdict["status"] == "violated"


# ---------------------------------------------------------------------------
# 课程表与台账对账
#
# 两份文件此前**从不互看**：课程表自己写 status: PASS，台账自己记
# mechanism.status。于是会出现一个很具体的情形 ——
#   课程表：M1 机制行 status=PASS
#   台账：  那一轮的 mechanism.status=refuted
# 读课程表的人得到「这条机制已经核实」，而唯一做过核验的那次说的是相反的话。
# 课程表是三个产物里唯一会被下一轮当作**前提**的那个。
# ---------------------------------------------------------------------------

def test_pass_line_contradicting_a_refuted_mechanism_is_reported():
    state, doc = misread_reference()
    entry = apply(state, result(speedup=1.30), "D0.1",
                  milestone_doc=doc, signature=signature(30.0))
    assert entry["mechanism"]["status"] in ("refuted", "vacuous")
    coupling = entry["gate_coupling"]
    assert coupling["claimed_pass"] == ["M1"]
    assert "说相反的话" in coupling["detail"]


def test_verified_mechanism_produces_no_contradiction():
    state = fresh_state()
    entry = apply(state, result(speedup=1.30), "D0.1",
                  milestone_doc=milestone(), signature=signature(0.1))
    assert entry["mechanism"]["status"] == "verified"
    assert entry.get("gate_coupling") is None


def test_unknown_mechanism_is_not_a_contradiction():
    """unknown 不构成「相反的话」—— 它只是没核验过。

    把它也报成矛盾会让每一轮没带 --signature 的 record 都多一条告警，
    而那条告警指不出任何该修的东西。
    """
    state = fresh_state()
    entry = apply(state, result(speedup=1.30), "D0.1")
    assert entry.get("gate_coupling") is None


def test_contradiction_says_which_side_to_fix():
    """报出矛盾而不说该修哪一侧，等于把判断又推回给人。

    该修哪一侧完全由归因决定，而归因本轮已经算出来了。
    """
    state, doc = misread_reference()
    entry = apply(state, result(speedup=1.30), "D0.1",
                  milestone_doc=doc, signature=signature(30.0))
    detail = entry["gate_coupling"]["detail"]
    assert "reopen" in detail
    assert "不要改课程表" in detail or "原样不动" in detail


def test_contradiction_check_does_not_rewrite_the_milestone():
    """record 不该去写课程表 —— 那是架构层的产物。

    AGENTS.md 的禁止项里写着编排层与编码层都不许改 milestone。对账只报告。
    """
    import copy  # noqa: PLC0415
    state, doc = misread_reference()
    before = copy.deepcopy(doc)
    apply(state, result(speedup=1.30), "D0.1",
          milestone_doc=doc, signature=signature(30.0))
    assert doc == before


def test_milestone_without_pass_lines_has_nothing_to_contradict():
    state, doc = misread_reference()
    for m in doc["milestones"]:
        for line in m["gate"]["mechanism_lines"]:
            line["status"] = "OPEN"
    entry = apply(state, result(speedup=1.30), "D0.1",
                  milestone_doc=doc, signature=signature(30.0))
    assert entry.get("gate_coupling") is None


# ---------------------------------------------------------------------------
# 课程表修订的台账与回退目标
# ---------------------------------------------------------------------------

def _reopen(state_file, milestone_file, **kw):
    return campaign.cmd_reopen(reopen_args(state_file, milestone_file, **kw))


def test_reopen_records_a_patch_entry(tmp_path):
    """重开是一次课程表修订。不记成 patch，这件事在台账里完全无痕。"""
    state_file, milestone_file = reopen_fixture(tmp_path)
    assert _reopen(state_file, milestone_file, force=True,
                   reason="分段切分轴判错了") == 0
    state = json.loads(state_file.read_text())
    patch = next(e["patch"] for e in state["ledger"] if e.get("patch"))
    assert patch["locus"] == "D0.1"
    assert patch["direction"] == "分段切分轴判错了"
    assert patch["outcome"] == "pending"
    assert patch["milestone_sha_before"]


def test_patch_outcome_is_backfilled_when_the_reopen_pays_off(tmp_path):
    """只记投入不记产出的账，用起来比没有更糟。

    不回填的话 outcome 永远停在 pending，于是 status 会把一次**成功的**
    修订也列进「已试过的方向」去劝退下一代。
    """
    state_file, milestone_file = reopen_fixture(tmp_path)
    _reopen(state_file, milestone_file, force=True, reason="改拓扑")
    state = campaign.load_state(state_file)
    apply(state, result(speedup=1.40), "D0.1",
          milestone_doc=milestone(), signature=signature(0.1))
    patch = next(e["patch"] for e in state["ledger"] if e.get("patch"))
    assert patch["outcome"] == "accepted"
    assert patch["closed_by_gain"] == pytest.approx(1.40)


def test_patch_outcome_stays_pending_without_a_new_lower_bound(tmp_path):
    """没抬高 best 就不算这次修订成功 —— 它还在待观察。"""
    state_file, milestone_file = reopen_fixture(tmp_path)
    _reopen(state_file, milestone_file, force=True, reason="改拓扑")
    state = campaign.load_state(state_file)
    apply(state, result(speedup=1.001), "D0.1",
          milestone_doc=milestone(), signature=signature(0.1))
    patch = next(e["patch"] for e in state["ledger"] if e.get("patch"))
    assert patch["outcome"] == "pending"


def test_apply_result_returns_the_new_entry_not_a_ledger_row(tmp_path):
    """回填那段用过 `for entry in ...`，把函数正在构造的 entry 遮蔽掉了。

    这类遮蔽不报错，只是悄悄把返回值换成台账里的一条旧记录 —— 而调用方
    拿它去读 mechanism / verdict。
    """
    state_file, milestone_file = reopen_fixture(tmp_path)
    _reopen(state_file, milestone_file, force=True, reason="改拓扑")
    state = campaign.load_state(state_file)
    entry = apply(state, result(speedup=1.40), "D0.1",
                  milestone_doc=milestone(), signature=signature(0.1))
    assert entry["decision"] == "D0.1"
    assert entry["verdict"] == campaign.VERDICT_ACCEPT
    assert entry is state["ledger"][-1]


def test_rollback_target_is_the_last_verified_artifact_outside_the_closure():
    """重开把下游台账作废了，但**工作树还在原地**。

    编码层如果从当前树接着改，它就是在一个已被判定建立在错假设上的实现
    之上继续叠加，而归因会把后面的结果算到新架构假设头上。
    """
    state = fresh_state()
    state["ledger"] = [
        {"generation": 0, "decision": "D0.1", "verdict": "accept",
         "gain": 1.10, "counts_toward_best": True, "artifact_sha": "aaa111"},
        {"generation": 0, "decision": "D0.2", "verdict": "accept",
         "gain": 1.20, "counts_toward_best": True, "artifact_sha": "bbb222"},
    ]
    target = campaign.rollback_target(state, {"D0.2"})
    assert target is not None
    assert target["artifact_sha"] == "aaa111"       # D0.2 在作废范围内


def test_rollback_target_skips_unverified_rounds():
    """只是变快了的产物自己就解释不清，不能当回退起点。"""
    state = fresh_state()
    state["ledger"] = [
        {"generation": 0, "decision": "D0.1", "verdict": "accept_unverified",
         "gain": 1.30, "counts_toward_best": False, "artifact_sha": "aaa111"},
    ]
    assert campaign.rollback_target(state, set()) is None


def test_rollback_target_skips_rounds_without_a_code_identity():
    """回退目标必须是一份能被找回来的东西。"""
    state = fresh_state()
    state["ledger"] = [
        {"generation": 0, "decision": "D0.1", "verdict": "accept",
         "gain": 1.10, "counts_toward_best": True},      # 没有 artifact_sha
    ]
    assert campaign.rollback_target(state, set()) is None


def test_rollback_target_does_not_cross_generations():
    state = fresh_state()
    state["generation"] = 1
    state["ledger"] = [
        {"generation": 0, "decision": "D0.1", "verdict": "accept",
         "gain": 1.10, "counts_toward_best": True, "artifact_sha": "aaa111"},
    ]
    assert campaign.rollback_target(state, set()) is None


# ---------------------------------------------------------------------------
# 墙钟预算
#
# 次数预算挡不住一种很常见的消耗：每轮都很快、但轮数没涨而时间在走
# （构建失败重试、环境排障、等卡）。那些都不消耗 used。
# ---------------------------------------------------------------------------

def _state_started_hours_ago(hours, wall_cap=None):
    from datetime import datetime, timedelta, timezone  # noqa: PLC0415
    state = fresh_state()
    state["created_at"] = (datetime.now(timezone.utc)
                           - timedelta(hours=hours)).isoformat()
    if wall_cap is not None:
        state["budget"]["max_wall_hours"] = wall_cap
    return state


def test_elapsed_hours_is_computed_from_created_at():
    state = _state_started_hours_ago(3.0)
    elapsed = campaign._elapsed_hours(state)
    assert elapsed is not None and 2.9 < elapsed < 3.1


def test_elapsed_hours_is_none_when_the_timestamp_is_unusable():
    """返回 0 会让这道闸静默失效，而失效时输出照样是「预算充裕」。"""
    assert campaign._elapsed_hours({}) is None
    assert campaign._elapsed_hours({"created_at": "前天"}) is None
    assert campaign._elapsed_hours({"created_at": ""}) is None


def test_wall_budget_stops_the_campaign():
    state = _state_started_hours_ago(5.0, wall_cap=4.0)
    campaign._advance_state(state, None)
    assert state["state"] == campaign.STATE_STOPPED
    assert "墙钟预算耗尽" in state["stop_reason"]


def test_wall_budget_says_the_number_is_an_upper_bound():
    """按 created_at 算差把停机时间也算进去了 —— 这件事要说出来。"""
    state = _state_started_hours_ago(5.0, wall_cap=4.0)
    campaign._advance_state(state, None)
    assert "上界" in state["stop_reason"]


def test_wall_budget_does_not_fire_below_the_cap():
    state = _state_started_hours_ago(1.0, wall_cap=4.0)
    campaign._advance_state(state, None)
    assert state["state"] != campaign.STATE_STOPPED


def test_no_wall_cap_means_no_wall_limit():
    state = _state_started_hours_ago(1000.0)
    campaign._advance_state(state, None)
    assert state["state"] != campaign.STATE_STOPPED


# ---------------------------------------------------------------------------
# M0 对账强制化
#
# audit 查的是整个 campaign 里最容易悄悄出错的那件事：课程表漏掉了某个超带
# 通道。漏掉的后果不是少一条门禁，而是**门禁永远不会因它触发** —— 门禁只拿
# 课程表自己声明的值做比较。于是那个通道上的问题可以一路活到 promote，
# 而每一轮的数都正常。
#
# 做成「下一步动作」而不是硬停机：硬停会在 M0 还没跑完任何一轮时就把流程
# 卡住（那时还没有本轮签名可对账），而一个在正当起点上就报错的门禁会被
# 反射性绕过。
# ---------------------------------------------------------------------------

def _implementing(audit=None, generation=0, accepted_gen=0):
    state = fresh_state()
    state["state"] = campaign.STATE_IMPLEMENT
    state["generation"] = generation
    state["ledger"] = [{"generation": accepted_gen, "decision": "D0.1",
                        "verdict": "accept", "gain": 1.2,
                        "counts_toward_best": True}]
    if audit is not None:
        state["m0_audit"] = audit
    return state


def test_next_action_asks_for_audit_when_it_never_ran():
    action = campaign.next_action(_implementing())
    assert action["action"] == "audit"
    assert "不会触发任何东西" in action["reason"]


def test_next_action_proceeds_once_audit_passed():
    state = _implementing({"generation": 0, "passed": True, "unclaimed": []})
    assert campaign.next_action(state)["action"] == "implement"


def test_next_action_asks_again_when_audit_did_not_pass():
    state = _implementing({"generation": 0, "passed": False,
                           "unclaimed": ["msprof:X.csv:y"]})
    action = campaign.next_action(state)
    assert action["action"] == "audit"
    assert "msprof:X.csv:y" in action["reason"]
    assert "三择一" in action["reason"]


def test_audit_from_a_previous_generation_does_not_count():
    """新参考是另一个二进制，上一代的对账结论对本代不成立。"""
    state = _implementing({"generation": 0, "passed": True, "unclaimed": []},
                          generation=1, accepted_gen=1)
    assert campaign.next_action(state)["action"] == "audit"


def test_audit_is_not_demanded_before_there_is_anything_to_audit():
    """本代还没有机制已核实的读数时，对不了账 —— 这时要求 audit 是在
    一个正当起点上报错，而那种门禁会被反射性绕过。"""
    state = _implementing(generation=1, accepted_gen=0)
    assert campaign.next_action(state)["action"] != "audit"


def test_audit_is_not_demanded_during_analysis():
    """架构层还在写课程表时没有本轮签名。"""
    state = _implementing()
    state["state"] = campaign.STATE_ANALYZE
    assert campaign.next_action(state)["action"] != "audit"


def test_audit_stamps_the_state_so_it_leaves_a_trace(tmp_path):
    """audit 是个纯读命令，不记的话「这一代到底对过账没有」无从得知。"""
    import argparse  # noqa: PLC0415
    state = fresh_state()
    state_file = tmp_path / "s.json"
    campaign.save_state(state_file, state)
    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps(signature(4.2)), encoding="utf-8")
    mil = tmp_path / "m.json"
    mil.write_text(json.dumps(milestone()), encoding="utf-8")

    rc = campaign.cmd_audit(argparse.Namespace(
        state=str(state_file), signature=str(sig), milestone=str(mil),
        factor=campaign.M0_AUDIT_FACTOR, as_json=False))
    stamped = json.loads(state_file.read_text())["m0_audit"]
    assert stamped["generation"] == 0
    assert stamped["passed"] is (rc == 0)
    assert "at" in stamped
