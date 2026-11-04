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
"""评测脚本是**生成**出来的，所以它的语法要单独检查。

`evaluate_ops_direct.py` 不直接跑评测：它拼一段 Python 源码，写进临时文件，
再起子进程执行。那段源码是用 f-string 拼的，而 f-string 里的 `{{` / `}}`、
缩进、引号嵌套都很容易写错。

**这类错误只在设备上才暴露。** 本机没有 torch_npu，整个 CI 也没有 —— 于是
一次拼错的改动会一路通过全部测试，直到有人在 910B 上跑评测，拿到一个
SyntaxError。那一刻的代价是：环境已经布好、产物已经装好、一次机时窗口
已经排到，而失败原因是一个本可以在本机 0.01 秒查出来的语法错。

所以这个文件做一件很小的事：把生成出来的脚本交给 `ast.parse`。
它不验证行为（那需要硬件），只验证「这段代码是合法的 Python」。

运行: python3 -m pytest plugins-community/forge-kernel/tests/ -v
"""

import ast
import sys
from pathlib import Path

import pytest

SCRIPTS = (Path(__file__).resolve().parent.parent / "skills"
           / "forge-evaluation" / "scripts")
sys.path.insert(0, str(SCRIPTS))

import evaluate_ops_direct as ev  # noqa: E402


def _cfg(**kw):
    """造一份最小配置。"""
    base = dict(
        call_spec={"op_namespace": "cust", "op_func": "add_layer_norm"},
        install_path="/opt/ref_opp",
        shape_plan=[],
        device_id=0,
        task_type="AiCore",
        profile_dir="/tmp/prof",
        num_trials=20,
        tag="evolved",
    )
    base.update(kw)
    return ev.DirectVersionConfig(**base)


def _script(cfg, opp_path="/opt/ref_opp", lib_path="/opt/lib"):
    return ev._build_direct_eval_script(cfg, opp_path, lib_path, "[]")


def test_generated_script_is_valid_python():
    """最基本的那一条：拼出来的东西能被解析。"""
    ast.parse(_script(_cfg()))


@pytest.mark.parametrize("warmup,trials", [(0, 1), (20, 20), (3, 100)])
def test_valid_for_every_measurement_setting(warmup, trials):
    ast.parse(_script(_cfg(warmup=warmup, num_trials=trials)))


@pytest.mark.parametrize("tag", ["baseline", "evolved"])
def test_valid_for_both_sides(tag):
    """两侧走同一个生成器，但 tag 会进字符串字面量与文件名。"""
    ast.parse(_script(_cfg(tag=tag)))


def test_valid_with_a_path_containing_special_characters():
    """路径进的是 repr()，但只要有一处漏了 repr 就会拼出非法代码。"""
    ast.parse(_script(_cfg(profile_dir="/tmp/a b/c'd"),
                      opp_path="/opt/ref's opp"))


def test_each_fragment_is_individually_parseable():
    """逐段查，这样报错能指到是哪一段拼坏的。

    整段一起查时，ast 的报错位置落在拼好之后的行号上，而那个行号对不回
    任何一个源文件 —— 查起来要先把生成结果 dump 出来再数行。
    """
    cfg = _cfg()
    fragments = {
        "env_setup": ev._dscript_env_setup(cfg, "/opt/o", "/opt/l", "[]"),
        "shape_loop": ev._dscript_shape_loop(cfg),
        "output": ev._dscript_output(),
    }
    for name, text in fragments.items():
        # 片段自己不是完整程序（缩进从中间开始），所以按「补一个外层」来查：
        # 这样仍能抓住括号不配对、引号不闭合、f-string 占位写错这几类。
        wrapped = "def _probe():\n" + "\n".join(
            "    " + line for line in text.splitlines())
        try:
            ast.parse(wrapped)
        except SyntaxError as exc:
            pytest.fail(f"片段 {name} 拼出了非法 Python：{exc}\n"
                        f"  这类错误只在设备上才暴露 —— 本机没有 torch_npu，"
                        f"所以必须靠这道检查。")


def test_measure_fragment_mentions_both_timing_windows():
    """两个计量窗口都要在生成出来的代码里。

    不是为了测行为（那需要硬件），而是为了防一种具体的退化：有人改这段时
    把新加的摊销窗口删掉，而所有测试照样绿 —— 因为没有任何测试提到它。
    """
    text = ev._dscript_perf_measure(_cfg())
    assert "_measure_events" in text
    assert "host_enqueue_us" in text
    assert "device_amortized_us" in text


# ---------------------------------------------------------------------------
# 压力分布与输入轮换
#
# 这两件事的**行为**需要在 NPU 上才能验证（本机没有 torch）。这里只验证
# 本机能验证的那部分：规格表自身的一致性、参数能传到底、生成的代码合法。
# ---------------------------------------------------------------------------

def test_stress_profiles_are_documented():
    """每个分布都要有一句说明它针对哪类降精度。

    不写的话，三个名字看起来只是三档强度，而它们针对的是不同机制。
    """
    assert set(ev.STRESS_PROFILES) == {"normal", "wide", "extreme"}
    for name, note in ev.STRESS_PROFILES.items():
        assert note.strip(), f"{name} 没有说明"


def test_unknown_stress_profile_is_refused():
    with pytest.raises(ValueError, match="不认识的压力分布"):
        ev.generate_inputs({}, "/tmp/x", {"target": [], "generalization": []},
                           stress_profile="随便写的")


def test_cli_exposes_both_knobs():
    """参数要真的能从命令行传进来。

    不查的话，加了 dataclass 字段却忘了挂 argparse 选项 —— 那两个旋钮
    就只有默认值可用，而默认值恰好是「不施加压力、只轮换两份」。
    """
    import inspect  # noqa: PLC0415
    source = inspect.getsource(ev.main)
    assert "stress_profile=args.stress_profile" in source
    assert "input_rotation=args.input_rotation" in source
    # 选项本身也要在
    whole = inspect.getsource(ev)
    assert '"--stress-profile"' in whole
    assert '"--input-rotation"' in whole


def test_measurement_metadata_records_what_was_measured():
    """压力分布改变**测的是什么** —— 不写进结果的话，一份 normal 分布下的
    精度结论和一份 extreme 分布下的看起来一模一样，而两者强度差很远。"""
    import inspect  # noqa: PLC0415
    source = inspect.getsource(ev)
    assert '"stress_profile": getattr(cfg, "stress_profile"' in source
    assert '"input_rotation": getattr(cfg, "input_rotation"' in source


def test_generated_script_checks_for_input_mutation():
    """这一段的存在本身要被钉住 —— 否则有人改这块时删掉它，测试照样全绿。"""
    text = ev._dscript_perf_measure(_cfg())
    assert "_input_digest" in text
    assert "inputs_mutated" in text


def test_digest_placeholders_survive_into_the_generated_code():
    """f-string 里的 `{...}` 在**生成时**就会求值。

    内层那个 f-string 必须写成 `{{...}}`，否则生成阶段就会 NameError ——
    这正是这道语法门禁第一次抓到的错。
    """
    text = ev._dscript_perf_measure(_cfg())
    assert "{tuple(item.shape)}" in text        # 落成了运行时占位
    assert "{sample:.12e}" in text
