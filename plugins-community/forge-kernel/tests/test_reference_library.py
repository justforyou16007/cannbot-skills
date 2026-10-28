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
"""reference_library.py 的单元测试。

这个库存在的理由是自我提升的一个前提：**上一代的产出，下一代还拿得到**。
此前 promote 只记路径，那个目录被清掉、被覆盖，第 n 代就对不上第 n-1 代，
而且没有任何地方能看出参考变过。

测试盯住四件事：

  留存   产物是复制进库的，源目录删掉之后库里还在
  可核   改一个字节就能查出来 —— 参考一变，所有相对它的加速比就失效了，
         这件事必须可发现
  准入   机制未核实的产出不得入库作参考，与知识库的准入闸同理
  身份   不同 shape / 不同架构是不同的条目，不能混成一条血统

运行: python3 -m pytest plugins-community/forge-kernel/tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = (Path(__file__).resolve().parent.parent
           / "skills" / "forge-campaign" / "scripts")
sys.path.insert(0, str(SCRIPTS))

import reference_library as rl  # noqa: E402

CHANNEL = "msprof:ResourceConflictRatio.csv:aiv_mte2_wait_ratio"


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def write_signature(tmp_path, value, name="sig.json"):
    path = tmp_path / name
    path.write_text(json.dumps({
        "trust": {"trustworthy": True},
        "stalls": {"wait": {"w": {
            "value": value, "channel": CHANNEL, "present": True}}},
    }), encoding="utf-8")
    return path


def make_artifact(tmp_path, name="kernel", content="// k\n"):
    root = tmp_path / name / "op_kernel"
    root.mkdir(parents=True, exist_ok=True)
    (root / "k.cpp").write_text(content, encoding="utf-8")
    return tmp_path / name


IDENTITY = dict(operator="add_layer_norm", arch="DAV_2201",
                input_shape="B=8,S=4096,fp16")


def seed_reference(library, tmp_path, time_us=456.5, stall=4.2):
    return rl.add_generation(
        library, kind="builtin_aclnn", identifier="aclnnAddLayerNorm",
        signature=str(write_signature(tmp_path, stall, "ref.json")),
        time_us=time_us, **IDENTITY)


# ---------------------------------------------------------------------------
# 身份
# ---------------------------------------------------------------------------

def test_shape_key_is_readable_and_unique():
    """键要人看得懂（list 时有用），又不能撞。"""
    key = rl.shape_key("B=8,S=4096,fp16")
    assert "B=8" in key
    assert key != rl.shape_key("B=16,S=4096,fp16")


def test_shape_key_ignores_whitespace():
    assert rl.shape_key("B=8, S=4096") == rl.shape_key("B=8,S=4096")


def test_different_shapes_are_different_entries(tmp_path):
    """同一算子的不同 shape 不能混成一条血统 —— 它们的参考与目标都不同。"""
    library = tmp_path / "L"
    rl.add_generation(library, kind="builtin_aclnn", identifier="a",
                      operator="op", arch="DAV_2201", input_shape="B=8")
    rl.add_generation(library, kind="builtin_aclnn", identifier="a",
                      operator="op", arch="DAV_2201", input_shape="B=16")
    assert len(rl.load_index(library)["entries"]) == 2


def test_different_arch_are_different_entries(tmp_path):
    library = tmp_path / "L"
    rl.add_generation(library, kind="builtin_aclnn", identifier="a",
                      operator="op", arch="DAV_2201", input_shape="B=8")
    rl.add_generation(library, kind="builtin_aclnn", identifier="a",
                      operator="op", arch="DAV_3510", input_shape="B=8")
    assert len(rl.load_index(library)["entries"]) == 2


def test_environment_recorded(tmp_path):
    """不同 CANN 版本下的参考不是同一个东西，环境必须留痕。"""
    library = tmp_path / "L"
    manifest = seed_reference(library, tmp_path)
    assert "cann_version" in manifest["environment"]
    assert "ascend_home" in manifest["environment"]


def test_unknown_environment_is_labelled_not_guessed(monkeypatch):
    """探不到就如实标 unknown，不猜。"""
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    env = rl.detect_environment()
    assert env["cann_version"] == "unknown"
    assert env["ascend_home"] == "unknown"


# ---------------------------------------------------------------------------
# 留存：产物复制进库
# ---------------------------------------------------------------------------

def test_artifact_copied_not_linked(tmp_path):
    """源目录删掉之后，库里还要拿得出这一代 —— 这正是自我提升的前提。"""
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    artifact = make_artifact(tmp_path)
    rl.add_generation(library, kind="sealed_kernel", identifier="gen1",
                      artifact=str(artifact), time_us=350.0,
                      mechanism_verified=True, **IDENTITY)

    import shutil
    shutil.rmtree(artifact)

    record = rl.resolve(library, **IDENTITY)
    assert Path(record["artifact_dir"], "op_kernel", "k.cpp").is_file()


def test_builtin_has_no_artifact(tmp_path):
    """厂商实现不归我们保管，只记可观测身份。"""
    library = tmp_path / "L"
    manifest = seed_reference(library, tmp_path)
    assert manifest["artifact_present"] is False
    assert manifest["artifact_files"] == []


def test_builtin_rejects_artifact(tmp_path):
    library = tmp_path / "L"
    with pytest.raises(rl.LibraryError, match="属于厂商"):
        rl.add_generation(library, kind="builtin_aclnn", identifier="a",
                          artifact=str(make_artifact(tmp_path)), **IDENTITY)


def test_sealed_kernel_requires_artifact(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    with pytest.raises(rl.LibraryError, match="必须提供 --artifact"):
        rl.add_generation(library, kind="sealed_kernel", identifier="g1",
                          mechanism_verified=True, **IDENTITY)


def test_empty_artifact_rejected(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(rl.LibraryError, match="为空"):
        rl.add_generation(library, kind="sealed_kernel", identifier="g1",
                          artifact=str(empty), mechanism_verified=True,
                          **IDENTITY)


# ---------------------------------------------------------------------------
# 可核：改一个字节就要查得出来
# ---------------------------------------------------------------------------

def test_verify_passes_on_untouched_library(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    rl.add_generation(library, kind="sealed_kernel", identifier="g1",
                      artifact=str(make_artifact(tmp_path)), time_us=350.0,
                      mechanism_verified=True, **IDENTITY)
    problems, checked = rl.verify(library)
    assert problems == []
    assert checked == 2


def test_verify_detects_tampering(tmp_path):
    """参考一变，之前所有相对它的加速比全部失效，必须可发现。"""
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    rl.add_generation(library, kind="sealed_kernel", identifier="g1",
                      artifact=str(make_artifact(tmp_path)), time_us=350.0,
                      mechanism_verified=True, **IDENTITY)

    record = rl.resolve(library, **IDENTITY)
    target = Path(record["artifact_dir"], "op_kernel", "k.cpp")
    target.write_text("// 被改了\n", encoding="utf-8")

    problems, _ = rl.verify(library)
    assert problems
    assert "内容已变更" in problems[0]


def test_verify_detects_deletion(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    rl.add_generation(library, kind="sealed_kernel", identifier="g1",
                      artifact=str(make_artifact(tmp_path)), time_us=350.0,
                      mechanism_verified=True, **IDENTITY)
    record = rl.resolve(library, **IDENTITY)
    Path(record["artifact_dir"], "op_kernel", "k.cpp").unlink()
    problems, _ = rl.verify(library)
    assert any("已删除" in p for p in problems)


def test_verify_scopes_to_operator(tmp_path):
    library = tmp_path / "L"
    rl.add_generation(library, kind="builtin_aclnn", identifier="a",
                      operator="op_a", arch="DAV_2201", input_shape="B=8")
    rl.add_generation(library, kind="builtin_aclnn", identifier="b",
                      operator="op_b", arch="DAV_2201", input_shape="B=8")
    _, checked = rl.verify(library, operator="op_a")
    assert checked == 1


# ---------------------------------------------------------------------------
# 签名漂移：厂商实现唯一的变更检测手段
# ---------------------------------------------------------------------------

def test_drift_stable_when_signature_unchanged(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path, stall=4.2)
    report = rl.signature_drift(
        library, generation=0,
        current_signature=str(write_signature(tmp_path, 4.2, "now.json")),
        **IDENTITY)
    assert report["stable"] is True


def test_drift_detected_after_cann_upgrade(tmp_path):
    """CANN 升级后内置实现变了，签名会漂 —— 那就说明参考不是原来那个。"""
    library = tmp_path / "L"
    seed_reference(library, tmp_path, stall=4.2)
    report = rl.signature_drift(
        library, generation=0,
        current_signature=str(write_signature(tmp_path, 4.8, "now.json")),
        **IDENTITY)
    assert report["stable"] is False
    assert report["drifted"][0]["channel"] == CHANNEL


def test_small_drift_tolerated(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path, stall=4.2)
    report = rl.signature_drift(
        library, generation=0,
        current_signature=str(write_signature(tmp_path, 4.25, "now.json")),
        **IDENTITY)
    assert report["stable"] is True


def test_drift_needs_sealed_signature(tmp_path):
    library = tmp_path / "L"
    rl.add_generation(library, kind="builtin_aclnn", identifier="a",
                      time_us=1.0, **IDENTITY)
    with pytest.raises(rl.LibraryError, match="没有封存签名"):
        rl.signature_drift(
            library, generation=0,
            current_signature=str(write_signature(tmp_path, 4.2)),
            **IDENTITY)


# ---------------------------------------------------------------------------
# 准入：机制未核实的不得入库
# ---------------------------------------------------------------------------

def test_unverified_mechanism_rejected(tmp_path):
    """封存一个解释不清的结果，后续各代都会建立在它之上而无人知其所以然。"""
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    with pytest.raises(rl.LibraryError, match="机制未核实"):
        rl.add_generation(library, kind="sealed_kernel", identifier="g1",
                          artifact=str(make_artifact(tmp_path)),
                          mechanism_verified=False, **IDENTITY)


def test_verified_mechanism_accepted(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    manifest = rl.add_generation(
        library, kind="sealed_kernel", identifier="g1",
        artifact=str(make_artifact(tmp_path)), time_us=350.0,
        mechanism_verified=True, **IDENTITY)
    assert manifest["generation"] == 1


def test_gen0_exempt_from_mechanism_gate(tmp_path):
    """gen0 是外来参考，没有「机制」可言。"""
    library = tmp_path / "L"
    manifest = seed_reference(library, tmp_path)
    assert manifest["generation"] == 0
    assert manifest["mechanism_verified"] is None


# ---------------------------------------------------------------------------
# 血统与最优
# ---------------------------------------------------------------------------

def test_lineage_records_parent_and_gain(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    manifest = rl.add_generation(
        library, kind="sealed_kernel", identifier="g1",
        artifact=str(make_artifact(tmp_path)), time_us=350.0, gain=1.30,
        mechanism_verified=True, **IDENTITY)
    assert manifest["promoted_from"] == 0
    assert manifest["gain_over_parent"] == 1.30


def test_best_is_fastest_measured(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path, time_us=456.5)
    rl.add_generation(library, kind="sealed_kernel", identifier="g1",
                      artifact=str(make_artifact(tmp_path, "k1")),
                      time_us=350.0, mechanism_verified=True, **IDENTITY)
    rl.add_generation(library, kind="sealed_kernel", identifier="g2",
                      artifact=str(make_artifact(tmp_path, "k2")),
                      time_us=400.0, mechanism_verified=True, **IDENTITY)
    assert rl.resolve(library, **IDENTITY)["generation"] == 1


def test_untimed_generations_excluded_from_best(tmp_path):
    """没测过的不能和测过的放在一起比。"""
    library = tmp_path / "L"
    seed_reference(library, tmp_path, time_us=456.5)
    rl.add_generation(library, kind="sealed_kernel", identifier="g1",
                      artifact=str(make_artifact(tmp_path)), time_us=None,
                      mechanism_verified=True, **IDENTITY)
    assert rl.resolve(library, **IDENTITY)["generation"] == 0


def test_resolve_specific_generation(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    rl.add_generation(library, kind="sealed_kernel", identifier="g1",
                      artifact=str(make_artifact(tmp_path)), time_us=350.0,
                      mechanism_verified=True, **IDENTITY)
    assert rl.resolve(library, generation=0, **IDENTITY)["generation"] == 0


def test_resolve_unknown_entry_lists_what_exists(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    with pytest.raises(rl.LibraryError) as exc:
        rl.resolve(library, operator="nope", arch="DAV_2201",
                   input_shape="B=8")
    assert "已有" in str(exc.value)


def test_resolve_empty_library_says_so(tmp_path):
    with pytest.raises(rl.LibraryError, match="库是空的"):
        rl.resolve(tmp_path / "L", **IDENTITY)


# ---------------------------------------------------------------------------
# cann-bench 挂载
# ---------------------------------------------------------------------------

def test_bench_baseline_is_annotation_not_artifact(tmp_path):
    """bench 的 baseline_perf_us 是评分基准线，不参与 best 评选。"""
    library = tmp_path / "L"
    rl.add_generation(
        library, kind="builtin_aclnn", identifier="a", time_us=456.5,
        bench_task="tasks/level2/add_layer_norm", bench_baseline_us=500.0,
        **IDENTITY)
    record = rl.resolve(library, **IDENTITY)
    assert record["bench_baseline_us"] == 500.0
    assert record["generation"] == 0   # best 仍按实测，不看 bench 基准线


def test_bench_task_recorded(tmp_path):
    library = tmp_path / "L"
    rl.add_generation(library, kind="builtin_aclnn", identifier="a",
                      bench_task="tasks/level4/sparse_attn", **IDENTITY)
    entry = rl.load_index(library)["entries"][
        rl.entry_key(IDENTITY["arch"], IDENTITY["operator"],
                     IDENTITY["input_shape"])]
    assert entry["bench_task"] == "tasks/level4/sparse_attn"


# ---------------------------------------------------------------------------
# 健壮性
# ---------------------------------------------------------------------------

def test_unknown_kind_rejected(tmp_path):
    with pytest.raises(rl.LibraryError, match="未知来源"):
        rl.add_generation(tmp_path / "L", kind="magic", identifier="a",
                          **IDENTITY)


def test_corrupt_index_reports_clearly(tmp_path):
    library = tmp_path / "L"
    library.mkdir()
    rl.index_path(library).write_text("不是 JSON", encoding="utf-8")
    with pytest.raises(rl.LibraryError, match="不是合法 JSON"):
        rl.load_index(library)


def test_missing_artifact_source_reports_clearly(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    with pytest.raises(rl.LibraryError, match="不存在"):
        rl.add_generation(library, kind="sealed_kernel", identifier="g1",
                          artifact=str(tmp_path / "nope"),
                          mechanism_verified=True, **IDENTITY)


def test_index_roundtrip(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    assert len(rl.load_index(library)["entries"]) == 1


def test_signature_snapshot_skips_absent(tmp_path):
    path = tmp_path / "sig.json"
    path.write_text(json.dumps({
        "stalls": {"wait": {"w": {"value": None, "channel": CHANNEL,
                                  "present": False}}}}), encoding="utf-8")
    assert rl.load_signature_snapshot(str(path)) is None
