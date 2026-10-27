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

这两个脚本合起来决定「什么样的改动算数」。测试围绕三个容易出错的判断：

  边界   增益恰好等于噪声底时该不该收？不该 —— 等于噪声意味着无法区分。
  优先级 精度不过的候选，增益再高也要回滚。
  收敛   自我提升什么时候停，停了之后状态还能不能被误推进。

运行: python3 -m pytest plugins-community/forge-kernel/tests/ -v
"""

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
# 接受判据
# ---------------------------------------------------------------------------

def test_gain_above_noise_is_accepted():
    assert campaign.decide_candidate(1.10, 2.0) == campaign.VERDICT_ACCEPT


def test_gain_within_noise_is_reverted():
    """增益落在噪声里，和没改一样 —— 收下它等于把抖动当成果。"""
    assert campaign.decide_candidate(1.01, 2.0) == campaign.VERDICT_REVERT


def test_gain_exactly_at_noise_floor_is_reverted():
    """边界用例：恰好等于噪声底不算超过，必须拒。"""
    assert campaign.decide_candidate(1.02, 2.0) == campaign.VERDICT_REVERT


def test_gain_just_above_noise_floor_is_accepted():
    assert campaign.decide_candidate(1.0201, 2.0) == campaign.VERDICT_ACCEPT


def test_clear_regression_is_flagged():
    assert campaign.decide_candidate(0.90, 2.0) == campaign.VERDICT_REGRESSION


def test_small_regression_within_noise_is_revert_not_regression():
    """小幅变慢也可能是抖动，不该当成明确退化。"""
    assert campaign.decide_candidate(0.99, 2.0) == campaign.VERDICT_REVERT


def test_negative_noise_floor_rejected():
    with pytest.raises(ValueError):
        campaign.decide_candidate(1.1, -1.0)


def test_zero_noise_floor_accepts_any_positive_gain():
    """噪声底为 0 时任何正增益都收 —— 这正是 init 要警告的危险配置。"""
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
# 下一步动作
# ---------------------------------------------------------------------------

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
    """停滞时的提示必须指向重开架构决策，而不是继续在原架构下硬调 ——
    这是 ForgeKernel 区别于普通调优循环的关键动作。"""
    action = campaign.next_action({
        "state": campaign.STATE_STALLED,
        "stagnation": {"since_improvement": 3},
    })
    assert "M0" in action["reason"]


# ---------------------------------------------------------------------------
# 结果并入
# ---------------------------------------------------------------------------

def fresh_state(**kw):
    state = campaign.new_state(
        operator="demo_op", arch="DAV_2201",
        reference={"kind": "builtin_aclnn", "id": "aclnnDemo", "sealed": True},
        noise_floor_pct=kw.get("noise", 2.0),
        target_gain=kw.get("target", 1.5),
        stagnation_threshold=kw.get("threshold", 3),
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


def test_accept_updates_best_and_resets_stagnation():
    state = fresh_state()
    state["stagnation"]["since_improvement"] = 2
    entry = campaign.apply_result(state, result(speedup=1.20), "D0.1")
    assert entry["verdict"] == campaign.VERDICT_ACCEPT
    assert state["best"]["gain"] == pytest.approx(1.20)
    assert state["stagnation"]["since_improvement"] == 0


def test_precision_failure_reverts_regardless_of_speed():
    """核心用例：算错的 kernel 再快也无意义，连增益都不该看。"""
    state = fresh_state()
    entry = campaign.apply_result(
        state, result(speedup=5.0, precision=False), "D0.1")
    assert entry["verdict"] == campaign.VERDICT_REVERT
    assert "精度" in entry["reason"]
    assert state["best"]["gain"] == 1.0  # 未被这个快但错的结果污染


def test_missing_measurement_reverts():
    """没有测量支持的改动一律回滚 —— agent 声称有效不算数。"""
    state = fresh_state()
    entry = campaign.apply_result(state, {"comparison": {}}, "D0.1")
    assert entry["verdict"] == campaign.VERDICT_REVERT
    assert "无测量支持" in entry["reason"]


def test_gain_computed_from_times_when_speedup_absent():
    state = fresh_state()
    entry = campaign.apply_result(
        state, result(baseline_us=400.0, evolved_us=200.0), "D0.1")
    assert entry["gain"] == pytest.approx(2.0)
    assert entry["verdict"] == campaign.VERDICT_ACCEPT


def test_stagnation_accumulates_then_stalls():
    """连续 3 轮无改进后交回架构层。"""
    state = fresh_state(threshold=3)
    for _ in range(2):
        campaign.apply_result(state, result(speedup=1.001), "D0.1")
        assert state["state"] == campaign.STATE_IMPLEMENT
    campaign.apply_result(state, result(speedup=1.001), "D0.1")
    assert state["state"] == campaign.STATE_STALLED


def test_accept_in_between_resets_stall_countdown():
    state = fresh_state(threshold=3)
    campaign.apply_result(state, result(speedup=1.001), "D0.1")
    campaign.apply_result(state, result(speedup=1.001), "D0.2")
    campaign.apply_result(state, result(speedup=1.30), "D0.3")   # 有进展
    campaign.apply_result(state, result(speedup=1.001), "D0.4")
    assert state["state"] == campaign.STATE_IMPLEMENT
    assert state["stagnation"]["since_improvement"] == 1


def test_reaching_target_gain_ships():
    state = fresh_state(target=1.5)
    campaign.apply_result(state, result(speedup=1.60), "D0.1")
    assert state["state"] == campaign.STATE_SHIPPED


def test_best_not_downgraded_by_weaker_accept():
    state = fresh_state(target=10.0)
    campaign.apply_result(state, result(speedup=1.80), "D0.1")
    campaign.apply_result(state, result(speedup=1.10), "D0.2")
    assert state["best"]["gain"] == pytest.approx(1.80)


def test_ledger_records_every_verdict():
    state = fresh_state()
    campaign.apply_result(state, result(speedup=1.30), "D0.1")
    campaign.apply_result(state, result(speedup=1.001), "D0.2")
    assert len(state["ledger"]) == 2
    assert all(e.get("reason") for e in state["ledger"]), "每条裁决都要有理由"


# ---------------------------------------------------------------------------
# 端到端：init -> record -> promote
# ---------------------------------------------------------------------------

def run(_tmp_path, argv):
    """用真实 argv 跑一次 CLI，返回退出码。"""
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


def test_cli_init_then_status(tmp_path):
    state_file = tmp_path / "state.json"
    code, _ = run(tmp_path, [
        "--state", str(state_file), "init",
        "--operator", "demo_op", "--arch", "DAV_2201",
        "--reference-kind", "builtin_aclnn", "--reference-id", "aclnnDemo",
        "--noise-floor", "2.0", "--target-gain", "1.5",
    ])
    assert code == 0
    assert state_file.exists()

    code, out = run(tmp_path, ["--state", str(state_file), "status"])
    assert code == 0
    assert "analyze" in out


def test_cli_init_refuses_overwrite(tmp_path):
    state_file = tmp_path / "state.json"
    base = ["--state", str(state_file), "init",
            "--operator", "demo_op", "--arch", "DAV_2201",
            "--reference-kind", "builtin_aclnn", "--reference-id", "aclnnDemo",
            "--noise-floor", "2.0"]
    assert run(tmp_path, base)[0] == 0
    assert run(tmp_path, base)[0] == 2            # 不许静默覆盖进度
    assert run(tmp_path, base + ["--force"])[0] == 0


def test_cli_warns_on_zero_noise_floor(tmp_path):
    state_file = tmp_path / "state.json"
    _, out = run(tmp_path, [
        "--state", str(state_file), "init",
        "--operator", "demo_op", "--arch", "DAV_2201",
        "--reference-kind", "builtin_aclnn", "--reference-id", "aclnnDemo",
        "--noise-floor", "0",
    ])
    assert "警告" in out


def test_cli_record_and_promote(tmp_path):
    state_file = tmp_path / "state.json"
    run(tmp_path, [
        "--state", str(state_file), "init",
        "--operator", "demo_op", "--arch", "DAV_2201",
        "--reference-kind", "builtin_aclnn", "--reference-id", "aclnnDemo",
        "--noise-floor", "2.0", "--target-gain", "1.1",
    ])

    res = tmp_path / "eval.json"
    res.write_text(json.dumps(result(speedup=1.25)), encoding="utf-8")
    code, _ = run(tmp_path, ["--state", str(state_file), "record",
                             "--result", str(res), "--decision", "D0.1"])
    assert code == 0

    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["state"] == campaign.STATE_SHIPPED

    code, _out = run(tmp_path, ["--state", str(state_file), "promote",
                               "--artifact", "/tmp/best_kernel"])
    assert code == 0
    promoted = json.loads(state_file.read_text(encoding="utf-8"))
    assert promoted["generation"] == 1
    assert promoted["reference"]["kind"] == "sealed_kernel"
    assert promoted["reference"]["sealed"] is True
    assert promoted["state"] == campaign.STATE_ANALYZE
    assert promoted["best"]["gain"] == 1.0, "新一代应从零开始计增益"


def test_cli_promote_stops_when_gain_in_noise(tmp_path):
    """自我提升的收敛路径：增益跌进噪声就停，不再无谓地开下一代。"""
    state_file = tmp_path / "state.json"
    run(tmp_path, [
        "--state", str(state_file), "init",
        "--operator", "demo_op", "--arch", "DAV_2201",
        "--reference-kind", "builtin_aclnn", "--reference-id", "aclnnDemo",
        "--noise-floor", "5.0", "--target-gain", "1.001",
    ])
    res = tmp_path / "eval.json"
    res.write_text(json.dumps(result(speedup=1.002)), encoding="utf-8")
    run(tmp_path, ["--state", str(state_file), "record",
                   "--result", str(res), "--decision", "D0.1"])

    code, out = run(tmp_path, ["--state", str(state_file), "promote",
                               "--artifact", "/tmp/k"])
    assert code == 0
    assert "收敛" in out
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["state"] == campaign.STATE_STOPPED
    assert state["generation"] == 0, "收敛时不应推进代次"


def test_cli_record_refused_after_stop(tmp_path):
    state_file = tmp_path / "state.json"
    run(tmp_path, [
        "--state", str(state_file), "init",
        "--operator", "demo_op", "--arch", "DAV_2201",
        "--reference-kind", "builtin_aclnn", "--reference-id", "aclnnDemo",
        "--noise-floor", "2.0",
    ])
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["state"] = campaign.STATE_STOPPED
    state_file.write_text(json.dumps(state), encoding="utf-8")

    res = tmp_path / "eval.json"
    res.write_text(json.dumps(result(speedup=2.0)), encoding="utf-8")
    code, _ = run(tmp_path, ["--state", str(state_file), "record",
                             "--result", str(res), "--decision", "D1.1"])
    assert code == 1


def test_corrupt_state_file_reports_clearly(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text("这不是 JSON", encoding="utf-8")
    code, out = run(tmp_path, ["--state", str(state_file), "status"])
    assert code == 2
    assert "损坏" in out


# ---------------------------------------------------------------------------
# 噪声底
# ---------------------------------------------------------------------------

def test_noise_floor_needs_minimum_samples():
    with pytest.raises(noise_floor.NoiseFloorError, match="样本不足"):
        noise_floor.compute([1.0, 1.01])


def test_noise_floor_on_identical_readings_is_zero():
    out = noise_floor.compute([1.0, 1.0, 1.0])
    assert out["floor_pct"] == pytest.approx(0.0)


def test_noise_floor_grows_with_spread():
    tight = noise_floor.compute([1.00, 1.01, 0.99])
    loose = noise_floor.compute([1.00, 1.20, 0.80])
    assert loose["floor_pct"] > tight["floor_pct"]


def test_noise_floor_uses_conservative_of_two_statistics():
    """取变异系数与半极差的较大者：一个离群点不应被平均掉。"""
    out = noise_floor.compute([1.0, 1.0, 1.0, 1.0, 1.3])
    assert out["floor_pct"] == pytest.approx(
        max(out["cv_pct"], out["half_range_pct"]))


def test_noise_floor_warns_when_environment_noisy():
    out = noise_floor.compute([1.0, 1.5, 0.6])
    assert out["warnings"], "离散这么大必须提醒环境可能不稳定"
    assert any("不稳定" in w for w in out["warnings"])


def test_noise_floor_ignores_invalid_readings():
    with pytest.raises(noise_floor.NoiseFloorError):
        noise_floor.compute([1.0, -1.0, 0.0])  # 有效样本不足 3


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


# ---------------------------------------------------------------------------
# 噪声底：首选的 cv_pct 路径
# ---------------------------------------------------------------------------

def test_cv_extracted_from_top_level():
    assert noise_floor.extract_cv_values({"cv_pct": 1.8}) == [1.8]


def test_cv_extracted_from_both_sides():
    values = noise_floor.extract_cv_values({
        "baseline": {"cv_pct": 1.2},
        "evolved": {"cv_pct": 2.4},
    })
    assert sorted(values) == [1.2, 2.4]


def test_cv_extracted_from_shape_results():
    """多 shape 评测时每个 shape 各有一个 cv，都要收进来。"""
    values = noise_floor.extract_cv_values({
        "shape_results": {
            "target": [{"cv_pct": 1.0}, {"cv_pct": 3.0}],
            "generalization": [{"cv_pct": 2.0}],
        }
    })
    assert sorted(values) == [1.0, 2.0, 3.0]


def test_cv_floor_takes_the_worst_reading():
    """取最大而非平均：取宽松的读数会把线划得太低，抖动就被收进来了。"""
    out = noise_floor.compute_from_cv([1.0, 3.5, 2.0])
    assert out["floor_pct"] == pytest.approx(3.5)
    assert out["method"] == "cv_pct"


def test_cv_floor_errors_when_absent():
    with pytest.raises(noise_floor.NoiseFloorError, match="没有 cv_pct"):
        noise_floor.compute_from_cv([])


def test_cv_floor_warns_when_noisy():
    out = noise_floor.compute_from_cv([12.0])
    assert out["warnings"]
    assert any("降频" in w for w in out["warnings"])


def test_cv_ignores_malformed_entries():
    values = noise_floor.extract_cv_values({
        "cv_pct": "不是数字",
        "baseline": "不是对象",
        "shape_results": {"target": ["不是对象", {"cv_pct": 2.0}]},
    })
    assert values == [2.0]


def test_spread_method_is_labelled():
    assert noise_floor.compute([1.0, 1.01, 0.99])["method"] == "ratio_spread"


# ---------------------------------------------------------------------------
# 两者衔接：量出来的噪声底直接喂给接受判据
# ---------------------------------------------------------------------------

def test_measured_floor_drives_acceptance():
    """这是两个脚本的接缝：噪声底不是常数，是量出来的。"""
    measured = noise_floor.compute([1.00, 1.02, 0.98, 1.01, 0.99])
    floor = measured["floor_pct"]
    # 落在实测噪声内的增益必须被拒
    assert campaign.decide_candidate(1.0 + floor / 200.0, floor) != campaign.VERDICT_ACCEPT
    # 明显超出的才收
    assert campaign.decide_candidate(1.0 + floor / 50.0, floor) == campaign.VERDICT_ACCEPT
