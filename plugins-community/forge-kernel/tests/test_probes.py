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
"""fence_check.py 与 lever_probe.py 的单元测试。

围栏的测试重心在**误伤**。一次误伤会让人去关掉围栏，关掉之后真正的
转发就无人把守了 —— 所以「不该拦的别拦」的用例比「该拦的要拦」还多：
注释里提到厂商接口、候选调用自己的同名算子、定义同名 kernel，
这些全都是正当的。

定价工具的测试重心在**边界与耦合**。定价的产出是上界而不是预期收益，
落在噪声内的手段必须被判为「测不出来」，两项相加对不上账时必须指出
是耦合而不是算错。

运行: python3 -m pytest plugins-community/forge-kernel/tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = (Path(__file__).resolve().parent.parent
           / "skills" / "forge-probes" / "scripts")
sys.path.insert(0, str(SCRIPTS))

import fence_check as fc  # noqa: E402
import lever_probe as lp  # noqa: E402


# ===========================================================================
# 围栏
# ===========================================================================

def candidate(tmp_path, **files) -> Path:
    root = tmp_path / "cand"
    root.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = root / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def run_fence(root, op_name="add_layer_norm", **kw):
    return fc.check(root, op_name, kw.pop("reference_paths", []),
                    kw.pop("denylist", []), kw.pop("exempt", ()))


def rules_of(result):
    return {f["rule"] for f in result["findings"]}


# ---------------------------------------------------------------------------
# 该拦的要拦
# ---------------------------------------------------------------------------

def test_aclnn_two_phase_call_blocked(tmp_path):
    """aclnn 两段式调用是调用厂商实现的标志。"""
    root = candidate(tmp_path, **{"op_kernel__k.cpp": (
        "void launch() {\n"
        "    uint64_t ws = 0;\n"
        "    aclnnAddLayerNormGetWorkspaceSize(a, b, &ws, nullptr);\n"
        "    aclnnAddLayerNorm(nullptr, ws, nullptr, nullptr);\n"
        "}\n")})
    result = run_fence(root)
    assert result["passed"] is False
    assert "F-ACLNN" in rules_of(result)
    assert len(result["findings"]) == 2


def test_torch_npu_vendor_op_blocked(tmp_path):
    root = candidate(tmp_path, **{"custom.py":
                                  "import torch_npu\n"
                                  "def run(x):\n"
                                  "    return torch_npu.npu_add_layer_norm(x)\n"})
    assert "F-TORCH-NPU" in rules_of(run_fence(root))


def test_framework_implementation_blocked(tmp_path):
    root = candidate(tmp_path, **{"custom.py":
                                  "import torch\n"
                                  "def run(x):\n"
                                  "    return torch.nn.functional.add_layer_norm(x)\n"})
    assert "F-TORCH-FUNC" in rules_of(run_fence(root))


def test_aten_dispatch_blocked(tmp_path):
    root = candidate(tmp_path, **{"custom.py":
                                  "import torch\n"
                                  "y = torch.ops.aten.add_layer_norm(x)\n"})
    assert "F-ATEN" in rules_of(run_fence(root))


def test_opp_path_rewrite_blocked(tmp_path):
    """候选自己改派发路径，可以把调用导向参考实现。"""
    root = candidate(tmp_path, **{"custom.py":
                                  "import os\n"
                                  "os.environ['ASCEND_CUSTOM_OPP_PATH'] = '/ref/vendors/custom'\n"})
    assert "F-OPP-SWITCH" in rules_of(run_fence(root))


def test_opp_path_export_blocked(tmp_path):
    root = candidate(tmp_path, **{"build.sh":
                                  "export ASCEND_CUSTOM_OPP_PATH=/ref/vendors/custom\n"})
    assert "F-OPP-SWITCH" in rules_of(run_fence(root))


def test_opp_path_read_only_allowed(tmp_path):
    """只读不写是正当的，例如打日志确认当前派发路径。"""
    root = candidate(tmp_path, **{"custom.py":
                                  "import os\n"
                                  "print(os.environ.get('ASCEND_CUSTOM_OPP_PATH'))\n"})
    assert run_fence(root)["passed"] is True


def test_reference_path_reference_blocked(tmp_path):
    root = candidate(tmp_path, **{"build.sh":
                                  "cp /sealed/reference/libop.so ./\n"})
    result = run_fence(root, reference_paths=["/sealed/reference"])
    assert "F-REF-PATH" in rules_of(result)


def test_denylist_symbol_blocked(tmp_path):
    root = candidate(tmp_path, **{"op_kernel__k.cpp":
                                  "ProprietaryRefCore core;\ncore.Run();\n"})
    result = run_fence(root, denylist=["ProprietaryRefCore"])
    assert "F-DENY" in rules_of(result)


def test_camel_case_derived_from_snake(tmp_path):
    """add_layer_norm 要能认出 aclnnAddLayerNorm 这种写法。"""
    assert fc.name_variants("add_layer_norm")["camel"] == "AddLayerNorm"
    assert fc.name_variants("fused-attn score")["camel"] == "FusedAttnScore"


# ---------------------------------------------------------------------------
# 不该拦的别拦（误伤会让人关掉围栏）
# ---------------------------------------------------------------------------

def test_own_kernel_definition_allowed(tmp_path):
    """候选必须定义同名算子，那是它的本职工作。"""
    root = candidate(tmp_path, **{"op_kernel__k.cpp": (
        "#include \"kernel_operator.h\"\n"
        "extern \"C\" __global__ __aicore__ void add_layer_norm(GM_ADDR x) {\n"
        "    AscendC::GlobalTensor<half> xGm;\n"
        "}\n")})
    assert run_fence(root)["passed"] is True


def test_own_custom_namespace_allowed(tmp_path):
    """torch.ops.cust.<op> 是候选自己注册的算子，不是厂商实现。"""
    root = candidate(tmp_path, **{"custom.py":
                                  "import torch\n"
                                  "y = torch.ops.cust.add_layer_norm(x)\n"})
    assert run_fence(root)["passed"] is True


def test_cpp_line_comment_allowed(tmp_path):
    """注释里解释「我们没用 aclnnXxx」不该被判越界。"""
    root = candidate(tmp_path, **{"op_kernel__k.cpp":
                                  "// 本实现不调用 aclnnAddLayerNorm，自己做两趟统计\n"
                                  "void run() {}\n"})
    assert run_fence(root)["passed"] is True


def test_cpp_block_comment_allowed(tmp_path):
    root = candidate(tmp_path, **{"op_kernel__k.cpp": (
        "/*\n"
        " * 对照组曾用 aclnnAddLayerNorm，本版不再使用。\n"
        " */\n"
        "void run() {}\n")})
    assert run_fence(root)["passed"] is True


def test_inline_block_comment_does_not_hide_code(tmp_path):
    """块注释后面的真实调用仍要被抓到，不能被注释处理吃掉。"""
    root = candidate(tmp_path, **{"op_kernel__k.cpp":
                                  "/* note */ aclnnAddLayerNorm(a);\n"})
    assert "F-ACLNN" in rules_of(run_fence(root))


def test_python_comment_allowed(tmp_path):
    root = candidate(tmp_path, **{"custom.py":
                                  "# 不要用 torch_npu.npu_add_layer_norm\n"
                                  "y = torch.ops.cust.add_layer_norm(x)\n"})
    assert run_fence(root)["passed"] is True


def test_different_operator_not_flagged(tmp_path):
    """别的算子的 aclnn 调用与本次无关。"""
    root = candidate(tmp_path, **{"op_kernel__k.cpp": "aclnnSoftmax(a);\n"})
    assert run_fence(root, op_name="add_layer_norm")["passed"] is True


def test_build_dirs_skipped(tmp_path):
    """构建产物不是候选的实现。"""
    root = tmp_path / "cand"
    (root / "build_out").mkdir(parents=True)
    (root / "build_out" / "gen.cpp").write_text(
        "aclnnAddLayerNorm(a);\n", encoding="utf-8")
    (root / "k.cpp").write_text("void run() {}\n", encoding="utf-8")
    assert run_fence(root)["passed"] is True


# ---------------------------------------------------------------------------
# 豁免与错误处理
# ---------------------------------------------------------------------------

def test_exempt_moves_finding_to_waived(tmp_path):
    root = candidate(tmp_path, **{"custom.py":
                                  "y = torch.nn.functional.add_layer_norm(x)\n"})
    result = run_fence(root, exempt=["F-TORCH-FUNC"])
    assert result["passed"] is True
    assert len(result["waived"]) == 1


def test_missing_candidate_errors(tmp_path):
    with pytest.raises(fc.FenceError, match="不存在"):
        run_fence(tmp_path / "nope")


def test_no_scannable_files_errors(tmp_path):
    root = tmp_path / "cand"
    root.mkdir()
    (root / "data.bin").write_bytes(b"\x00\x01")
    with pytest.raises(fc.FenceError, match="没有可扫描的源文件"):
        run_fence(root)


def test_findings_carry_location(tmp_path):
    """没有 file:line 就无法当场判断是不是误伤。"""
    root = candidate(tmp_path, **{"op_kernel__k.cpp":
                                  "void a() {}\naclnnAddLayerNorm(x);\n"})
    finding = run_fence(root)["findings"][0]
    assert finding["line"] == 2
    assert finding["file"].endswith("k.cpp")
    assert finding["reason"]


def test_spec_supplies_op_name_and_paths(tmp_path):
    spec = tmp_path / "reference_spec.json"
    spec.write_text(json.dumps({
        "op_name": "add_layer_norm", "baseline_path": "/sealed/ref"}),
        encoding="utf-8")
    op_name, paths = fc.load_reference_paths(spec)
    assert op_name == "add_layer_norm"
    assert paths == ["/sealed/ref"]


def test_missing_spec_points_at_the_step_that_creates_it(tmp_path):
    """错误信息要能直接操作，否则用户会卡在这一步。"""
    with pytest.raises(fc.FenceError) as exc:
        fc.load_reference_paths(tmp_path / "nope.json")
    message = str(exc.value)
    assert "reference_adapter.py prepare" in message
    assert "--op-name" in message


# ===========================================================================
# 手段定价
# ===========================================================================

def store_with(tmp_path, baseline=456.5, noise=2.3, variants=()):
    store = lp.new_store(baseline, noise)
    for levers, time_us in variants:
        lp.add_variant(store, list(levers), time_us)
    path = tmp_path / "levers.json"
    lp.save_store(path, store)
    return store, path


def test_pricing_reports_ceiling_not_expectation(tmp_path):
    """定价给的是上界：删掉它最多省这么多，不是优化它能拿这么多。"""
    store, _ = store_with(tmp_path, variants=[(["drop_store"], 429.1)])
    report = lp.price_levers(store)
    item = report["levers"][0]
    assert item["gain_pct"] == pytest.approx(6.0, abs=0.01)
    assert "不会超过" in item["meaning"]
    assert report["ceiling_pct"] == pytest.approx(6.0, abs=0.01)


def test_gain_within_noise_is_not_actionable(tmp_path):
    """落在噪声内的手段没有可测量的空间，不该投迭代。"""
    store, _ = store_with(tmp_path, variants=[(["drop_mask"], 451.0)])
    report = lp.price_levers(store)
    assert report["levers"][0]["verdict"] == "测不出来"
    assert report["actionable"] == []
    assert "drop_mask" in report["not_worth_pursuing"]


def test_gain_exactly_at_noise_floor_not_actionable(tmp_path):
    """边界取保守，与 campaign 的接受判据一致。"""
    baseline = 100.0
    at_floor = baseline * (1 - 2.3 / 100.0)
    store, _ = store_with(tmp_path, baseline=baseline,
                          variants=[(["x"], at_floor)])
    assert lp.price_levers(store)["actionable"] == []


def test_negative_gain_flagged(tmp_path):
    """删掉它反而更慢，通常说明它改善了别的东西。"""
    store, _ = store_with(tmp_path, variants=[(["drop_align"], 480.0)])
    item = lp.price_levers(store)["levers"][0]
    assert item["verdict"] == "反而更慢"


def test_coupling_detected_as_subadditive(tmp_path):
    """两项抢同一份时间时，按相加规划会对不上账。"""
    store, _ = store_with(tmp_path, variants=[
        (["a"], 429.1), (["b"], 440.0), (["a", "b"], 430.0)])
    coupling = lp.price_levers(store)["couplings"][0]
    assert coupling["status"] == "耦合抵消"
    assert coupling["observed_pct"] < coupling["additive_pct"]


def test_synergy_detected(tmp_path):
    """协同的两项应作为原子组同批落地。"""
    store, _ = store_with(tmp_path, variants=[
        (["a"], 440.0), (["b"], 445.0), (["a", "b"], 400.0)])
    coupling = lp.price_levers(store)["couplings"][0]
    assert coupling["status"] == "协同"
    assert "原子组" in coupling["meaning"]


def test_independent_when_additive_holds(tmp_path):
    store, _ = store_with(tmp_path, baseline=100.0, variants=[
        (["a"], 90.0), (["b"], 80.0), (["a", "b"], 70.0)])
    assert lp.price_levers(store)["couplings"][0]["status"] == "独立"


def test_coupling_incomplete_without_singles(tmp_path):
    store, _ = store_with(tmp_path, variants=[(["a", "b"], 400.0)])
    coupling = lp.price_levers(store)["couplings"][0]
    assert coupling["status"] == "incomplete"
    assert "单项" in coupling["meaning"]


def test_missing_probes_listed(tmp_path):
    store, _ = store_with(tmp_path, variants=[(["a"], 440.0), (["b"], 445.0)])
    pending = lp.missing_probes(store)
    assert any("a" in p and "b" in p for p in pending)


def test_all_levers_null_points_back_to_architecture(tmp_path):
    """所有手段都没空间，本身就是结论：该回 M0 改架构。"""
    store, _ = store_with(tmp_path, variants=[
        (["a"], 455.0), (["b"], 454.0)])
    report = lp.price_levers(store)
    assert report["actionable"] == []
    assert report["ceiling_pct"] == 0.0


def test_variants_marked_probe_only(tmp_path):
    """探针变体会产出错误结果，必须标明不得上线。"""
    store, _ = store_with(tmp_path)
    assert store["probe_only"] is True
    assert "不是候选" in store["warning"]


def test_duplicate_variant_overwrites(tmp_path):
    store, _ = store_with(tmp_path, variants=[(["a"], 440.0)])
    lp.add_variant(store, ["a"], 430.0)
    assert len(store["variants"]) == 1
    assert store["variants"][0]["time_us"] == 430.0


def test_lever_order_does_not_matter(tmp_path):
    store, _ = store_with(tmp_path, variants=[(["a", "b"], 400.0)])
    lp.add_variant(store, ["b", "a"], 390.0)
    assert len(store["variants"]) == 1


def test_empty_store_errors(tmp_path):
    store, _ = store_with(tmp_path)
    with pytest.raises(lp.LeverError, match="没有任何变体"):
        lp.price_levers(store)


def test_invalid_baseline_rejected():
    with pytest.raises(lp.LeverError):
        lp.new_store(0.0, 2.3)
    with pytest.raises(lp.LeverError):
        lp.new_store(100.0, -1.0)


def test_variant_needs_a_lever(tmp_path):
    store, _ = store_with(tmp_path)
    with pytest.raises(lp.LeverError, match="至少关联一个手段"):
        lp.add_variant(store, [], 400.0)


def test_store_roundtrip(tmp_path):
    _, path = store_with(tmp_path, variants=[(["a"], 440.0)])
    reloaded = lp.load_store(path)
    assert reloaded["variants"][0]["levers"] == ["a"]


def test_corrupt_store_reports_clearly(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("不是 JSON", encoding="utf-8")
    with pytest.raises(lp.LeverError, match="不是合法 JSON"):
        lp.load_store(path)
