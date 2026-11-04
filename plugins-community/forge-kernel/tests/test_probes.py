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


# ---------------------------------------------------------------------------
# 名词匹配要带词边界
#
# 裸子串匹配会让清单里的短名字吃掉候选里更长的名字：清单有 DataCopyEx 时，
# 候选里的 DataCopyExtParams 会被判越界，而后者是另一个符号。这种误报的
# 代价是围栏被整条豁免掉（--exempt F-DENY），真正的转发也跟着放过去。
# ---------------------------------------------------------------------------

def test_denylist_does_not_match_longer_identifier(tmp_path):
    root = candidate(tmp_path, **{"op_kernel__k.cpp": (
        "void f() {\n"
        "    AscendC::DataCopyExtParams p{1, 1024, 0, 0, 0};\n"
        "    AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, 0};\n"
        "}\n")})
    result = run_fence(root, denylist=["DataCopyEx", "DataCopyPad"])
    assert result["passed"] is True, result["findings"]


def test_denylist_still_matches_exact_identifier(tmp_path):
    """带了词边界之后，该拦的还要拦。"""
    root = candidate(tmp_path, **{"op_kernel__k.cpp": (
        "void f() {\n"
        "    AscendC::DataCopyEx(dst, src, 128);\n"
        "}\n")})
    result = run_fence(root, denylist=["DataCopyEx"])
    assert "F-DENY" in rules_of(result)


def test_denylist_token_inside_longer_name_is_not_a_hit(tmp_path):
    """候选自己起的名字里含参考的类名片段，不算引用参考的符号。"""
    root = candidate(tmp_path, **{"op_kernel__k.cpp": (
        "class MyOwnAddLayerNormKernel {};\n")})
    result = run_fence(root, denylist=["AddLayerNormKernelCore", "KernelAddLayerNorm"])
    assert result["passed"] is True, result["findings"]


def test_same_token_twice_on_one_line_reported_once(tmp_path):
    root = candidate(tmp_path, **{"op_kernel__k.cpp": (
        "void f() { RefSecretCore(); RefSecretCore(); }\n")})
    result = run_fence(root, denylist=["RefSecretCore"])
    assert len([f for f in result["findings"] if f["rule"] == "F-DENY"]) == 1


def test_missing_denylist_file_is_an_error_not_an_empty_list(tmp_path):
    """显式给了清单却读不到时必须报错。

    静默退回空清单的后果是 F-DENY 整条检查消失，而围栏照样打印「通过」——
    一个照抄参考内部命名的候选会拿到干净的通行证。路径打错一个字母就这样。
    """
    with pytest.raises(fc.FenceError, match="找不到专有名词清单"):
        fc.load_denylist(tmp_path / "nope.txt")


def test_no_denylist_option_is_still_allowed():
    assert fc.load_denylist(None) == []


# ---------------------------------------------------------------------------
# 通道规则：与算子名无关的那一组
#
# 按算子名生成的五条规则只能抓住「照名字直接调」。而转发**不需要写出名字**
# —— ctypes.CDLL("libopapi.so").aclnnAddLayerNorm 里那个名字是运行时拼出来
# 的字符串，importlib.import_module(cfg.backend) 里连字符串都没有。
# 于是五条全绿，而候选一行 kernel 都没写。
#
# #8 之后围栏能压住 best，也就是能决定一轮算不算数 —— 漏掉的那条路现在是
# 「静默地让一个转发实现成为下一代参考」。
# ---------------------------------------------------------------------------

def _channel_rules(result):
    return {f["rule"] for f in result["channels"]}


def test_dlopen_is_reported_as_a_channel(tmp_path):
    root = candidate(tmp_path, **{"op_kernel__k.cpp": (
        'void *h = dlopen("libopapi.so", RTLD_LAZY);\n')})
    result = run_fence(root)
    assert "F-DLOPEN" in _channel_rules(result)
    assert "F-OPAPI-SO" in _channel_rules(result)


def test_ctypes_is_reported_as_a_channel(tmp_path):
    root = candidate(tmp_path, **{"op_kernel__k.py": (
        'lib = ctypes.CDLL("libopapi.so")\n')})
    assert "F-FFI" in _channel_rules(run_fence(root))


def test_dynamic_import_is_reported_as_a_channel(tmp_path):
    root = candidate(tmp_path, **{"op_kernel__k.py": (
        'm = importlib.import_module(cfg.backend)\n')})
    assert "F-DYNIMPORT" in _channel_rules(run_fence(root))


def test_subprocess_is_reported_as_a_channel(tmp_path):
    root = candidate(tmp_path, **{"op_kernel__k.py": (
        'subprocess.run([helper, "--run"])\n')})
    assert "F-SUBPROCESS" in _channel_rules(run_fence(root))


def test_another_operators_aclnn_is_a_channel_not_a_block(tmp_path):
    """转发给**另一个**厂商算子同样不是自己实现，但判不了是不是正当。

    这条用例原先断言「别的算子的 aclnn 调用与本次无关」。那句话对被测的
    kernel 并不成立 —— 如果 kernel 体里调 aclnnSoftmax 去做一部分工作，
    测到的时间就包含厂商代码。

    但也不能直接判不通过：候选自己的测试代码、setup/teardown 里出现别的
    aclnn 调用是正当的，而围栏的误伤代价高于漏判（一次误伤会让人整条
    --exempt 关掉）。所以给 advisory 档：报出来，不压 best。
    """
    root = candidate(tmp_path, **{"op_kernel__k.cpp": "aclnnSoftmax(a);\n"})
    result = run_fence(root, op_name="add_layer_norm")
    assert result["passed"] is True          # 不判不通过
    assert "F-ACLNN-ANY" in _channel_rules(result)   # 但要报出来


def test_channels_do_not_flip_passed(tmp_path):
    root = candidate(tmp_path, **{"op_kernel__k.py": (
        'lib = ctypes.CDLL("./my_own_kernel.so")\n')})
    result = run_fence(root)
    assert result["passed"] is True
    assert result["channels"]


def test_blocking_rules_still_flip_passed(tmp_path):
    """阻断档没有被这次改动放松。"""
    root = candidate(tmp_path, **{"op_kernel__k.cpp": (
        "aclnnAddLayerNorm(a, b);\n")})
    result = run_fence(root)
    assert result["passed"] is False
    assert "F-ACLNN" in rules_of(result)


def test_channel_rules_fire_even_with_a_wrong_op_name(tmp_path):
    """这是通道规则存在的理由之一。

    按算子名生成的五条规则只要算子名写错一个字母就**全体失效**，而那种
    失效是静默的 —— 围栏报「通过」。通道规则不由算子名派生，所以照样响。
    """
    root = candidate(tmp_path, **{"op_kernel__k.cpp": (
        'dlsym(h, "aclnnAddLayerNorm");\n')})
    result = run_fence(root, op_name="完全不相干的算子名")
    assert "F-DLOPEN" in _channel_rules(result)


def test_third_party_is_scanned(tmp_path):
    """third_party 曾在 SKIP_DIRS 里，那是围栏上最大的一个洞。

    把一份转发实现放进 third_party/、再从候选 import 它，整棵目录连扫都不扫
    —— 候选代码里只剩一行 import，没有任何 aclnn 字样。

    「第三方代码不是候选的实现」这句话本身对，但它回答的不是围栏在问的问题。
    围栏问的是「这次测到的加速比是不是这份 kernel 挣来的」—— 从哪个目录
    转发出去，对这个问题没有区别。
    """
    root = tmp_path / "cand"
    (root / "third_party" / "shim").mkdir(parents=True)
    (root / "third_party" / "shim" / "fwd.cpp").write_text(
        "aclnnAddLayerNorm(a, b);\n", encoding="utf-8")
    (root / "op_kernel").mkdir(parents=True)
    (root / "op_kernel" / "k.cpp").write_text(
        "void run() {}\n", encoding="utf-8")
    result = run_fence(root)
    assert result["passed"] is False
    assert "F-ACLNN" in rules_of(result)


def test_build_dirs_are_still_skipped(tmp_path):
    """构建产物仍然跳过 —— 拿掉 third_party 不等于什么都扫。"""
    root = tmp_path / "cand"
    (root / "build_out").mkdir(parents=True)
    (root / "build_out" / "gen.cpp").write_text(
        "aclnnAddLayerNorm(a);\n", encoding="utf-8")
    (root / "k.cpp").write_text("void run() {}\n", encoding="utf-8")
    assert run_fence(root)["passed"] is True


# ---------------------------------------------------------------------------
# 手段分类：两条腿
#
# 单项定价（「只删这一项」）只能回答「它单独值多少」。优化循环实际要问的是
# 「全上之后，再动它还有没有收益」—— 这两个问题的答案可以相反。
#
# 关键的一类是 coupled：单独删掉省不出时间，但别人依赖它。只有单项腿时，
# 它的读数落在噪声里 → 判「测不出来」→ 进 not_worth_pursuing → 工具主动
# 告诉你不要在一个承重的方向上投入。这不是漏报，是反向报错。
# ---------------------------------------------------------------------------

def _store(noise=2.0, baseline=100.0, **variants):
    """variants: {"a": 时间, "a|b": 时间} —— 竖线分隔多个手段。"""
    store = lp.new_store(baseline, noise)
    for key, time_us in variants.items():
        lp.add_variant(store, key.split("|"), time_us)
    return store


def test_coupled_lever_is_found_only_with_both_legs():
    """a 单独删没用（省 1%，噪声内），但把它从全删里拿掉就少省 20%。"""
    # 三次测量就够了。注意「全删但留下 a」和「只删 b」是同一个变体
    # （universe = {a, b}，所以 universe − {a} == {b}）——
    # 两条腿在两个手段的情形下共用这一次测量。
    store = _store(
        **{"a": 99.0,        # 只删 a：省 1%，落在噪声内
           "b": 80.0,        # 只删 b：省 20%（也就是「全删但留下 a」）
           "a|b": 50.0})     # 全删：省 50%
    classes = lp.classify_levers(store)
    assert classes["levers"]["a"]["class"] == "coupled"
    assert classes["levers"]["a"]["single_pct"] == pytest.approx(1.0)
    assert classes["levers"]["a"]["marginal_pct"] == pytest.approx(30.0)


def test_coupled_lever_is_not_listed_as_not_worth_pursuing():
    """这是整条改动的要点 —— 此前它会被列进「不要再投」。"""
    store = _store(**{"a": 99.0, "b": 80.0, "a|b": 50.0})
    report = lp.price_levers(store)
    assert "a" not in report["not_worth_pursuing"]
    assert "a" in report["must_keep"]


def test_independent_lever_pays_off_on_both_legs():
    store = _store(**{"a": 80.0, "b": 75.0, "a|b": 55.0})
    classes = lp.classify_levers(store)
    assert classes["levers"]["a"]["class"] == "independent"


def test_redundant_lever_pays_off_alone_but_not_at_the_margin():
    """单独有用，但别人已经吃掉了它的收益 —— 不要重复投入。"""
    store = _store(**{"a": 80.0, "b": 80.0, "a|b": 80.0})
    classes = lp.classify_levers(store)
    assert classes["levers"]["a"]["class"] == "redundant"


def test_inert_lever_pays_off_on_neither_leg():
    store = _store(**{"a": 99.5, "b": 80.0, "a|b": 80.0})
    classes = lp.classify_levers(store)
    assert classes["levers"]["a"]["class"] == "inert"
    assert "a" in lp.price_levers(store)["not_worth_pursuing"]


def test_harmful_lever_is_classified_before_the_margin_is_consulted():
    """删掉它反而更慢 —— 这个结论不需要边际读数。"""
    store = _store(**{"a": 130.0, "b": 80.0, "a|b": 110.0})
    classes = lp.classify_levers(store)
    assert classes["levers"]["a"]["class"] == "harmful"


def test_missing_remove_leg_is_unknown_not_inert():
    """缺一条腿时判不了。拿单项读数顶替，正是 coupled 被判成 inert 那条路。"""
    store = _store(**{"a": 99.0, "b": 80.0})     # 没有「全删」那次
    classes = lp.classify_levers(store)
    assert classes["levers"]["a"]["class"] == "unknown"
    assert classes["levers"]["a"]["marginal_pct"] is None


def test_missing_measurement_hint_is_runnable():
    """提示要说清补哪一次测量，否则这条分类永远停在 unknown。"""
    store = _store(**{"a": 99.0, "b": 80.0})
    info = lp.classify_levers(store)["levers"]["a"]
    assert any("--lever" in m for m in info["missing_measurements"])


# ---------------------------------------------------------------------------
# 变体指纹
# ---------------------------------------------------------------------------

def test_variant_identical_to_baseline_is_refused():
    """没改过的变体，测出来的差值全是噪声 —— 而它会被当成价格用。"""
    store = lp.new_store(100.0, 2.0, variant_hash="deadbeef")
    with pytest.raises(lp.LeverError, match="没有被真正改过"):
        lp.add_variant(store, ["a"], 95.0, variant_hash="deadbeef")


def test_variant_with_a_different_hash_is_accepted():
    store = lp.new_store(100.0, 2.0, variant_hash="deadbeef")
    entry = lp.add_variant(store, ["a"], 95.0, variant_hash="cafe")
    assert entry["variant_hash"] == "cafe"


def test_hash_check_is_skipped_when_either_side_is_absent():
    """没有指纹时不能拒 —— 那会把老 store 与不传指纹的正当用法一起卡住。"""
    store = lp.new_store(100.0, 2.0)          # 基线无指纹
    assert lp.add_variant(store, ["a"], 95.0, variant_hash="x")
    store2 = lp.new_store(100.0, 2.0, variant_hash="x")
    assert lp.add_variant(store2, ["a"], 95.0)   # 变体无指纹


def test_refusal_names_the_three_common_causes():
    """报错要说清怎么发生的，否则会被当成工具挑剔而绕过去。"""
    store = lp.new_store(100.0, 2.0, variant_hash="same")
    try:
        lp.add_variant(store, ["a"], 95.0, variant_hash="same")
    except lp.LeverError as exc:
        text = str(exc)
    assert "没被编译进去" in text
    assert "重装 OPP" in text
