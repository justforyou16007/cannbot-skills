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
    """整条链没有增益记录时，退回绝对耗时比较。"""
    library = tmp_path / "L"
    seed_reference(library, tmp_path, time_us=456.5)
    rl.add_generation(library, kind="sealed_kernel", identifier="g1",
                      artifact=str(make_artifact(tmp_path, "k1")),
                      time_us=350.0, mechanism_verified=True, **IDENTITY)
    rl.add_generation(library, kind="sealed_kernel", identifier="g2",
                      artifact=str(make_artifact(tmp_path, "k2")),
                      time_us=400.0, mechanism_verified=True, **IDENTITY)
    record = rl.resolve(library, **IDENTITY)
    assert record["generation"] == 1


# ---------------------------------------------------------------------------
# 最优的评选依据：累计增益跨环境可比，绝对耗时不可比
#
# 多卡机器上卡与卡之间有系统性差异，不同时段的并发水平也不同。
# gain_over_parent 是同一次配对测量里的比值，哪张卡上测都成立 ——
# 这正是论文坚持同 job 配对计时的理由。
# ---------------------------------------------------------------------------

def test_best_prefers_cumulative_gain(tmp_path):
    """有增益记录时按累计增益选，不看绝对耗时。"""
    library = tmp_path / "L"
    seed_reference(library, tmp_path, time_us=456.5)
    # gen1 在慢卡上测，绝对耗时更大，但增益更高
    rl.add_generation(library, kind="sealed_kernel", identifier="g1",
                      artifact=str(make_artifact(tmp_path, "k1")),
                      time_us=900.0, gain=1.50, device_id=3,
                      mechanism_verified=True, **IDENTITY)
    record = rl.resolve(library, **IDENTITY)
    assert record["generation"] == 1, "不该因为绝对耗时大就判它更差"

    index = rl.load_index(library)
    entry = index["entries"][rl.entry_key(**IDENTITY)]
    assert "跨环境可比" in entry["best_basis"]


def test_cumulative_gain_multiplies_along_chain(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path, time_us=100.0)
    for idx, gain in enumerate((1.20, 1.10), start=1):
        rl.add_generation(library, kind="sealed_kernel", identifier=f"g{idx}",
                          artifact=str(make_artifact(tmp_path, f"k{idx}")),
                          gain=gain, mechanism_verified=True, **IDENTITY)
    entry = rl.load_index(library)["entries"][rl.entry_key(**IDENTITY)]
    assert rl.cumulative_gain(entry["generations"], 2) == pytest.approx(1.32)


def test_gen0_cumulative_gain_is_one(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    entry = rl.load_index(library)["entries"][rl.entry_key(**IDENTITY)]
    assert rl.cumulative_gain(entry["generations"], 0) == 1.0


def test_basis_flags_when_only_absolute_time(tmp_path):
    """退回绝对耗时时要讲明它只在同一环境内可比。"""
    library = tmp_path / "L"
    seed_reference(library, tmp_path, time_us=456.5)
    entry = rl.load_index(library)["entries"][rl.entry_key(**IDENTITY)]
    _, basis = rl.pick_best(entry)
    assert "仅同一环境内可比" in basis


def test_device_id_recorded(tmp_path):
    """多卡机器上卡间有系统性差异，记下来才知道哪些数字可比。"""
    library = tmp_path / "L"
    manifest = rl.add_generation(library, kind="builtin_aclnn", identifier="a",
                                 device_id=3, **IDENTITY)
    assert manifest["device_id"] == 3


# ---------------------------------------------------------------------------
# 算子族：按类型检索的基础
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operator,expected", [
    ("add_layer_norm", "reduction"),
    ("softmax_v2", "reduction"),
    ("matmul_nz", "matmul"),
    ("batch_matmul", "matmul"),
    ("flash_attn_fwd", "attention"),
    ("mla_prefill", "attention"),
    ("conv2d_fwd", "convolution"),
    ("fast_gelu", "elementwise"),
    ("topk_v2", "sort"),
    # 量化变体归入主族：仓内分类体系把 matmul_mxfp4 这类归 matmul
    ("fake_quant_matmul", "matmul"),
    ("quantize_per_tensor", "quantization"),
    ("transpose_nd", "conversion"),
])
def test_family_inference(operator, expected):
    assert rl.infer_family(operator) == expected


def test_family_inference_returns_none_when_unsure():
    """推不出来就返回 None，不硬塞一个族 —— 塞错比没有更糟。"""
    assert rl.infer_family("zzz_custom_thing") is None


def test_explicit_family_overrides_inference(tmp_path):
    library = tmp_path / "L"
    rl.add_generation(library, kind="builtin_aclnn", identifier="a",
                      family="custom_family", **IDENTITY)
    entry = rl.load_index(library)["entries"][rl.entry_key(**IDENTITY)]
    assert entry["family"] == "custom_family"
    assert entry["family_source"] == "explicit"


def test_inferred_family_is_labelled(tmp_path):
    """推断的要标明来源，查的人才知道该不该信。"""
    library = tmp_path / "L"
    seed_reference(library, tmp_path)
    entry = rl.load_index(library)["entries"][rl.entry_key(**IDENTITY)]
    assert entry["family_source"] == "inferred"


# ---------------------------------------------------------------------------
# 按类型检索
# ---------------------------------------------------------------------------

def _populate(library, tmp_path):
    specs = [
        ("add_layer_norm", "DAV_2201", "B=8,S=4096"),
        ("softmax_v2", "DAV_3510", "S=8192"),
        ("matmul_nz", "DAV_2201", "M=4096,K=4096"),
        ("flash_attn_fwd", "DAV_2201", "B=2,S=4096"),
    ]
    for operator, arch, shape in specs:
        rl.add_generation(library, kind="builtin_aclnn",
                          identifier=f"aclnn_{operator}", time_us=100.0,
                          operator=operator, arch=arch, input_shape=shape)


def test_find_by_family(tmp_path):
    """forge-kernel 要迭代某类算子时，先问这一类有没有现成参考。"""
    library = tmp_path / "L"
    _populate(library, tmp_path)
    hits = rl.find(library, family="reduction")
    assert {h["operator"] for h in hits} == {"add_layer_norm", "softmax_v2"}


def test_find_by_arch(tmp_path):
    library = tmp_path / "L"
    _populate(library, tmp_path)
    hits = rl.find(library, arch="DAV_3510")
    assert len(hits) == 1


def test_find_combines_filters(tmp_path):
    library = tmp_path / "L"
    _populate(library, tmp_path)
    hits = rl.find(library, family="reduction", arch="DAV_2201")
    assert {h["operator"] for h in hits} == {"add_layer_norm"}


def test_find_by_operator_substring(tmp_path):
    library = tmp_path / "L"
    _populate(library, tmp_path)
    assert len(rl.find(library, operator="attn")) == 1


def test_find_by_shape_substring(tmp_path):
    library = tmp_path / "L"
    _populate(library, tmp_path)
    assert len(rl.find(library, shape="S=4096")) == 2


def test_find_require_artifact_excludes_builtin(tmp_path):
    """厂商实现没有产物可取，要拿来当起点改写时得排除掉。"""
    library = tmp_path / "L"
    _populate(library, tmp_path)
    assert rl.find(library, require_artifact=True) == []


def test_find_returns_best_generation(tmp_path):
    library = tmp_path / "L"
    seed_reference(library, tmp_path, time_us=456.5)
    rl.add_generation(library, kind="sealed_kernel", identifier="g1",
                      artifact=str(make_artifact(tmp_path)), gain=1.30,
                      time_us=350.0, mechanism_verified=True, **IDENTITY)
    hit = rl.find(library, family="reduction")[0]
    assert hit["best_gen"] == 1
    assert hit["cumulative_gain"] == pytest.approx(1.30)


def test_find_empty_on_no_match(tmp_path):
    library = tmp_path / "L"
    _populate(library, tmp_path)
    assert rl.find(library, family="nonexistent") == []


# ---------------------------------------------------------------------------
# 自定义参考入库
# ---------------------------------------------------------------------------

def custom_reference(tmp_path):
    root = tmp_path / "myref"
    (root / "op_kernel").mkdir(parents=True)
    (root / "op_kernel" / "k.cpp").write_text(
        "#include \"kernel_operator.h\"\n"
        "#define MYREF_TILE_BYTES 16384\n"
        "class MyProprietaryCore { };\n", encoding="utf-8")
    (root / "op_host").mkdir()
    (root / "op_host" / "tiling.cpp").write_text("// t\n", encoding="utf-8")
    return root


def test_ingest_registers_as_gen0(tmp_path):
    library = tmp_path / "L"
    record = rl.ingest(library, source=custom_reference(tmp_path),
                       identifier="my_impl", **IDENTITY)
    assert record["generation"] == 0
    assert record["kind"] == "opp_install"


def test_ingest_copies_artifact(tmp_path):
    library = tmp_path / "L"
    rl.ingest(library, source=custom_reference(tmp_path), **IDENTITY)
    resolved = rl.resolve(library, **IDENTITY)
    assert Path(resolved["artifact_dir"], "op_kernel", "k.cpp").is_file()


def test_ingest_seals_the_reference(tmp_path):
    """有源码的参考必须封存，否则 no-code 边界形同虚设。"""
    library = tmp_path / "L"
    record = rl.ingest(library, source=custom_reference(tmp_path), **IDENTITY)
    seal = record["seal"]
    assert Path(seal["sealed_paths"]).is_file()
    assert Path(seal["denylist"]).is_file()
    denylist = Path(seal["denylist"]).read_text(encoding="utf-8")
    assert "MyProprietaryCore" in denylist
    assert "MYREF_TILE_BYTES" in denylist


def test_ingest_denylist_excludes_common_api(tmp_path):
    library = tmp_path / "L"
    record = rl.ingest(library, source=custom_reference(tmp_path), **IDENTITY)
    denylist = Path(record["seal"]["denylist"]).read_text(encoding="utf-8")
    assert "GlobalTensor" not in denylist


def test_ingest_rejects_missing_dir(tmp_path):
    with pytest.raises(rl.LibraryError, match="不存在"):
        rl.ingest(tmp_path / "L", source=tmp_path / "nope", **IDENTITY)


def test_ingest_rejects_file(tmp_path):
    path = tmp_path / "a.cpp"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(rl.LibraryError, match="必须是目录"):
        rl.ingest(tmp_path / "L", source=path, **IDENTITY)


def test_ingest_rejects_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(rl.LibraryError, match="没有任何文件"):
        rl.ingest(tmp_path / "L", source=empty, **IDENTITY)


def test_ingest_infers_family(tmp_path):
    library = tmp_path / "L"
    rl.ingest(library, source=custom_reference(tmp_path), **IDENTITY)
    assert rl.find(library, family="reduction")


def test_ingest_accepts_explicit_family(tmp_path):
    library = tmp_path / "L"
    rl.ingest(library, source=custom_reference(tmp_path),
              family="my_family", **IDENTITY)
    assert rl.find(library, family="my_family")


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
