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
"""precision_gate.py 的判据测试。

这里只测**判据本身**（纯算术），不测 torch 管道 —— 后者需要 torch 与评测
产物，本机没有。这个切分是有意的：判据是唯一决定对错的地方，而管道只是
把张量搬进来。

判据要同时做到两件相反的事：

  抓住降精度   bf16 累加 / tf32 相乘 / fp16 中间量，容差比对放过的那一类
  不误杀诚实   归约顺序不同、参考本身精确时的一次舍入

两者在最大绝对差上长得很像，所以判据用的是「对 float64 真值的误差比」。

运行: python3 -m pytest plugins-community/forge-kernel/tests/ -v
"""

import sys
from pathlib import Path

import pytest

SCRIPTS = (Path(__file__).resolve().parent.parent
           / "skills" / "forge-evaluation" / "scripts")
sys.path.insert(0, str(SCRIPTS))

import precision_gate as pg  # noqa: E402


# ---------------------------------------------------------------------------
# 抓住降精度
# ---------------------------------------------------------------------------

def test_bf16_accumulation_is_caught():
    """参考误差 1e-7、候选 1e-3 —— 这个幅度不是归约顺序造成的。"""
    out = pg.judge_errors(1e-7, 1e-3, "float32")
    assert out["verdict"] == "fail"
    assert "降了精度" in out["reason"]


def test_overflow_to_inf_is_a_precision_failure():
    """真值有限而候选是 inf —— 中间量溢出了（典型是 fp16）。"""
    out = pg.judge_errors(1e-7, float("inf"), "float16")
    assert out["verdict"] == "fail"
    assert "溢出" in out["reason"]


def test_nan_is_a_precision_failure():
    out = pg.judge_errors(1e-7, float("nan"), "float32")
    assert out["verdict"] == "fail"


# ---------------------------------------------------------------------------
# 不误杀诚实实现
# ---------------------------------------------------------------------------

def test_different_reduction_order_passes():
    """归约顺序不同本来就产生这个量级的差，不该被判降精度。"""
    assert pg.judge_errors(1e-7, 2e-7, "float32")["verdict"] == "pass"


def test_exact_reference_does_not_fail_one_rounding():
    """参考精确时（拷贝、gather，e_ref ≈ 0）没有地板会把每个舍入一次的
    诚实实现全判失败。"""
    out = pg.judge_errors(0.0, 5e-6, "float32")
    assert out["verdict"] == "pass"
    assert out["floor"] >= pg.FLOOR_REL


def test_floor_scales_with_output_dtype():
    """fp16 输出本来只有 11 位有效位，拿 fp32 的地板卡它没有意义。"""
    assert pg.error_floor("float16") > pg.error_floor("float32")
    assert pg.error_floor("bfloat16") > pg.error_floor("float16")


def test_fp16_output_tolerates_its_own_roundoff():
    """一个 fp16 输出的 e_cand 在它自己的舍入量级上，应当通过。"""
    out = pg.judge_errors(0.0, pg.UNIT_ROUNDOFF["float16"] * 0.9, "float16")
    assert out["verdict"] == "pass"


# ---------------------------------------------------------------------------
# 缺失不等于通过
# ---------------------------------------------------------------------------

def test_missing_candidate_error_is_unavailable_not_pass():
    out = pg.judge_errors(1e-7, None, "float32")
    assert out["verdict"] == "unavailable"
    assert "不等于通过" in out["reason"]


def test_missing_reference_error_is_unavailable_not_pass():
    out = pg.judge_errors(None, 1e-7, "float32")
    assert out["verdict"] == "unavailable"


def test_no_torch_reference_is_unavailable_not_pass():
    """没有参考语义时这一轮的精度**没有被判过**。"""
    out = pg.evaluate(Path("/nonexistent"), Path("/a"), Path("/b"), None)
    assert out["verdict"] == "unavailable"
    assert "缺失不等于通过" in out["reason"]


# ---------------------------------------------------------------------------
# 判据的形状
# ---------------------------------------------------------------------------

def test_limit_is_k_times_reference_plus_floor():
    out = pg.judge_errors(1e-6, 1e-9, "float32", k_ratio=3.0)
    assert out["limit"] == pytest.approx(
        3.0 * 1e-6 + pg.error_floor("float32"))


def test_k_ratio_is_configurable_for_calibration():
    """常数要能调 —— 上游那组是在 B200 上标定的，换到昇腾要重标。

    误差量级要取在**地板之上**，否则 K 不起作用（见下一条）。
    """
    strict = pg.judge_errors(1e-4, 5e-4, "float32", k_ratio=1.0)
    loose = pg.judge_errors(1e-4, 5e-4, "float32", k_ratio=100.0)
    assert strict["verdict"] == "fail"
    assert loose["verdict"] == "pass"


def test_floor_dominates_below_its_own_scale():
    """小误差下地板完全主导，K 不起作用 —— 这是设计如此，不是 bug。

    fp32 的 FLOOR_REL ≈ 7.6e-6，远大于它自身的舍入单位 6e-8。上游的理由：
    一个用 fast-math exp2/log2 且换了归约顺序的 fp32 输出离真值约 1e-6，
    而它的参考离真值约 1e-7 —— 比值 20，但那根本不是降精度（tf32 自己的
    舍入就是 2^-11）。没有地板的话这一类会被全部误杀。
    """
    # 比值 20，远超 K=3，但两者都在地板之下 → 仍然通过
    out = pg.judge_errors(1e-7, 2e-6, "float32")
    assert out["verdict"] == "pass"
    assert out["e_cand"] / out["e_ref"] == pytest.approx(20.0)
    assert out["floor"] > out["e_cand"]


def test_unknown_dtype_falls_back_without_crashing():
    out = pg.judge_errors(1e-7, 2e-7, "torch.some_future_dtype")
    assert out["verdict"] == "pass"


def test_dtype_name_accepts_torch_prefix():
    assert pg.unit_roundoff("torch.float16") == pg.UNIT_ROUNDOFF["float16"]


def test_calibration_note_says_it_needs_recalibration():
    """常数没标定之前，这句话必须跟在报告里。"""
    note = pg.calibration_note()
    assert "重新标定" in note
    assert "K_RATIO" in note


def test_headroom_is_reported_for_calibration():
    """标定要看 e_cand/limit 的分布，所以这个比值要出现在报告里。"""
    out = pg.judge_errors(1e-7, 1e-7, "float32")
    assert "headroom_ratio" in out
