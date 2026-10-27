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
"""signature_probe.py 的单元测试。

这个探针最要命的失效方式不是崩溃，而是**悄悄把缺失当成零**。
msprof 没采到某个指标时，若按 0 处理，读出来就是「完全没有停顿」、
「完全没有冲突」—— 一个本该引发追查的信号被反转成了健康信号。

所以测试的重心是三件事：

  缺失与零要分开   present=False 和 value=0.0 必须是两种状态
  不可信要扣留     降频、ICache 颠簸、停顿数据缺失时不得输出架构假设
  判定要有出处     每个阈值判定都对得上 csv_fields_reference.md

运行: python3 -m pytest plugins-community/forge-kernel/tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = (Path(__file__).resolve().parent.parent
           / "skills" / "forge-probes" / "scripts")
sys.path.insert(0, str(SCRIPTS))

import signature_probe as sp  # noqa: E402


# ---------------------------------------------------------------------------
# 夹具：合成 msprof 归档目录
# ---------------------------------------------------------------------------

PIPE_COLUMNS = (
    "Op Name,Op Type,Task Duration(us),Block Dim,Mix Block Dim,"
    "Current Freq,Rated Freq,aic_time(us),aiv_time(us),"
    "aiv_vec_ratio,aic_cube_ratio,aiv_scalar_ratio,aic_mte1_ratio,"
    "aiv_mte2_ratio,aiv_mte3_ratio,aic_fixpipe_ratio,aiv_icache_miss_rate"
)

PIPE_DEFAULTS = {
    "Op Name": "AddLayerNorm", "Op Type": "AddLayerNorm",
    "Task Duration(us)": "456.5", "Block Dim": "48", "Mix Block Dim": "N/A",
    "Current Freq": "1800", "Rated Freq": "1800",
    "aic_time(us)": "0", "aiv_time(us)": "392.1",
    "aiv_vec_ratio": "28.4", "aic_cube_ratio": "0",
    "aiv_scalar_ratio": "9.2", "aic_mte1_ratio": "0",
    "aiv_mte2_ratio": "61.2", "aiv_mte3_ratio": "14.0",
    "aic_fixpipe_ratio": "0", "aiv_icache_miss_rate": "3.1",
}

CONFLICT_COLUMNS = (
    "Op Name,aiv_vec_total_cflt_ratio,aiv_vec_bankgroup_cflt_ratio,"
    "aiv_vec_bank_cflt_ratio,aiv_vec_resc_cflt_ratio,aiv_vec_mte_cflt_ratio,"
    "aiv_vec_wait_ratio,aiv_mte2_wait_ratio,aiv_mte3_wait_ratio"
)

CONFLICT_DEFAULTS = {
    "Op Name": "AddLayerNorm", "aiv_vec_total_cflt_ratio": "7.2",
    "aiv_vec_bankgroup_cflt_ratio": "1.1", "aiv_vec_bank_cflt_ratio": "4.5",
    "aiv_vec_resc_cflt_ratio": "1.0", "aiv_vec_mte_cflt_ratio": "0.6",
    "aiv_vec_wait_ratio": "31.8", "aiv_mte2_wait_ratio": "4.2",
    "aiv_mte3_wait_ratio": "2.0",
}

MEMORY_COLUMNS = (
    "Op Name,GM_to_UB_datas(KB),UB_to_GM_datas(KB),"
    "read_main_memory_datas(KB),write_main_memory_datas(KB),"
    "GM_to_UB_bw_usage_rate(%),UB_to_GM_bw_usage_rate(%)"
)

MEMORY_DEFAULTS = {
    "Op Name": "AddLayerNorm", "GM_to_UB_datas(KB)": "32768",
    "UB_to_GM_datas(KB)": "8192", "read_main_memory_datas(KB)": "32768",
    "write_main_memory_datas(KB)": "8192",
    "GM_to_UB_bw_usage_rate(%)": "38.5", "UB_to_GM_bw_usage_rate(%)": "61.0",
}


def _csv(columns: str, values: dict) -> str:
    names = columns.split(",")
    row = ",".join(str(values.get(n, "")) for n in names)
    return f"{columns}\n{row}\n"


def write_archive(root: Path, *, pipe=None, conflict=None, memory=None,
                  l2_hit="42.0", include=("PipeUtilization", "Memory",
                                          "ResourceConflictRatio", "L2Cache")):
    """造一个归档目录。include 控制哪些 metric 存在，用来测缺失场景。"""
    root.mkdir(parents=True, exist_ok=True)

    if "PipeUtilization" in include:
        values = dict(PIPE_DEFAULTS)
        values.update(pipe or {})
        (root / "op_summary_PipeUtilization.csv").write_text(
            _csv(PIPE_COLUMNS, values), encoding="utf-8")

    if "ResourceConflictRatio" in include:
        values = dict(CONFLICT_DEFAULTS)
        values.update(conflict or {})
        (root / "op_summary_ResourceConflictRatio.csv").write_text(
            _csv(CONFLICT_COLUMNS, values), encoding="utf-8")

    if "Memory" in include:
        values = dict(MEMORY_DEFAULTS)
        values.update(memory or {})
        (root / "op_summary_Memory.csv").write_text(
            _csv(MEMORY_COLUMNS, values), encoding="utf-8")

    if "L2Cache" in include:
        (root / "op_summary_L2Cache.csv").write_text(
            f"Op Name,aiv_total_hit_rate(%)\nAddLayerNorm,{l2_hit}\n",
            encoding="utf-8")

    return root


@pytest.fixture
def archive(tmp_path):
    return write_archive(tmp_path / "round1")


def build(root, **kw):
    return sp.build_signature(None, Path(root), kw.pop("op_name", "AddLayerNorm"),
                              kw.pop("min_gm_bytes", None))


def claims(sig):
    return " ".join(h["claim"] for h in sig["architectural_hypotheses"])


# ---------------------------------------------------------------------------
# 核心：缺失不等于零
# ---------------------------------------------------------------------------

def test_absent_column_is_not_zero(archive):
    """未采到的列 present=False，而不是 value=0。

    这是本探针存在的首要理由。混淆两者会把「没数据」读成「很健康」。
    """
    sig = build(archive)
    cube_wait = sig["stalls"]["wait"]["cube_wait"]
    assert cube_wait["present"] is False
    assert cube_wait["value"] is None


def test_real_zero_is_present(archive):
    """真的是 0 的列 present=True，value=0.0。"""
    cube = build(archive)["pipeline"]["cube"]
    assert cube["present"] is True
    assert cube["value"] == 0.0


def test_na_parsed_as_missing_not_zero(archive):
    """Mix Block Dim 的 N/A 表示非 MIX 算子，不是 0 个从核。"""
    sig = build(archive)
    assert sig["launch"]["mix_block_dim"]["value"] is None
    assert sig["launch"]["role_split"]["value"] == "single_role"


def test_mix_block_dim_with_value_means_mix(tmp_path):
    archive = write_archive(tmp_path / "r", pipe={"Mix Block Dim": "24"})
    sig = build(archive)
    assert sig["launch"]["role_split"]["value"] == "mix"
    assert "核间分工" in claims(sig)


@pytest.mark.parametrize("raw", ["", "N/A", "NA", "-", "null", "None", "  "])
def test_safe_float_rejects_placeholders(raw):
    assert sp.safe_float(raw) is None


def test_safe_float_accepts_zero():
    assert sp.safe_float("0") == 0.0
    assert sp.safe_float("0.0") == 0.0


# ---------------------------------------------------------------------------
# 可信度闸门
# ---------------------------------------------------------------------------

def test_downclocking_blocks_hypotheses(tmp_path):
    """降频会按不同比例拉长各段，占比失真 —— 必须扣留架构假设。"""
    archive = write_archive(tmp_path / "r", pipe={"Current Freq": "1200"})
    sig = build(archive)
    assert sig["trust"]["trustworthy"] is False
    assert sig["architectural_hypotheses"] == []
    assert "hypotheses_withheld" in sig
    assert any("降频" in r for r in sig["trust"]["blocking_reasons"])


def test_icache_thrash_blocks_hypotheses(tmp_path):
    archive = write_archive(tmp_path / "r", pipe={"aiv_icache_miss_rate": "22.0"})
    sig = build(archive)
    assert sig["trust"]["trustworthy"] is False
    assert sig["architectural_hypotheses"] == []


def test_missing_stall_metric_blocks_hypotheses(tmp_path):
    """停顿数据缺失是致命的：没有它，架构判断失去一半依据。"""
    archive = write_archive(
        tmp_path / "r", include=("PipeUtilization", "Memory", "L2Cache"))
    sig = build(archive)
    assert sig["trust"]["trustworthy"] is False
    assert sig["architectural_hypotheses"] == []


def test_missing_optional_metric_only_warns(tmp_path):
    """MemoryL0 之类缺了只告警，不该因此扣留全部结论。"""
    archive = write_archive(tmp_path / "r")
    sig = build(archive)
    assert sig["trust"]["trustworthy"] is True
    coverage = [c for c in sig["trust"]["checks"] if c["name"] == "metric_coverage"][0]
    assert coverage["verdict"] == "warn"


def test_healthy_measurement_passes(tmp_path):
    archive = write_archive(tmp_path / "r")
    sig = build(archive)
    assert sig["trust"]["trustworthy"] is True
    assert sig["architectural_hypotheses"]


# ---------------------------------------------------------------------------
# 阈值判定（出处：csv_fields_reference.md）
# ---------------------------------------------------------------------------

def test_mte2_bound_flagged(archive):
    sig = build(archive)
    flags = [f["flag"] for f in sig["pipeline"]["flags"]]
    assert "MTE2 Bound" in flags


def test_limiting_resource_is_highest_pipe(archive):
    assert build(archive)["pipeline"]["limiting"]["value"] == "mte2"


def test_fixpipe_excluded_from_limiting(tmp_path):
    """fixpipe 反映对齐问题而非算力占用，不该被选作限制资源。"""
    archive = write_archive(tmp_path / "r", pipe={
        "aic_fixpipe_ratio": "90.0", "aiv_mte2_ratio": "20.0",
        "aiv_vec_ratio": "10.0"})
    sig = build(archive)
    assert sig["pipeline"]["limiting"]["value"] != "fixpipe"
    assert any(f["flag"] == "地址未对齐" for f in sig["pipeline"]["flags"])


def test_bank_conflict_flagged(archive):
    """bank 冲突是可观测的 —— 之前文档里写成「无对应字段」是错的。"""
    sig = build(archive)
    assert any("bank 冲突" in f["flag"] for f in sig["stalls"]["flags"])
    assert "bank 冲突" in claims(sig)


def test_low_l2_hit_flagged(tmp_path):
    archive = write_archive(tmp_path / "r", l2_hit="30.0")
    sig = build(archive)
    assert any("需优化" in f["flag"] for f in sig["cache"]["flags"])


def test_good_l2_hit_not_flagged(tmp_path):
    archive = write_archive(tmp_path / "r", l2_hit="92.0")
    assert build(archive)["cache"]["flags"] == []


# ---------------------------------------------------------------------------
# 搬运量 vs 搬运方式 —— 两者改法不同，不能混
# ---------------------------------------------------------------------------

def test_low_bandwidth_points_at_tactical_layer(archive):
    """搬运占比高但带宽利用率低 = 搬运方式问题，属战术层。"""
    assert "战术层" in claims(build(archive))


def test_saturated_bandwidth_points_at_architecture(tmp_path):
    """搬运占比高且带宽已满 = 只能减少搬运量，属架构层。"""
    archive = write_archive(tmp_path / "r",
                            memory={"GM_to_UB_bw_usage_rate(%)": "88.0"})
    text = claims(build(archive))
    assert "减少搬运总量" in text
    assert "架构层" in text


def test_missing_bandwidth_degrades_to_weak_claim(tmp_path):
    """没有带宽数据时不能二选一，应如实降级为弱结论。"""
    archive = write_archive(
        tmp_path / "r", include=("PipeUtilization", "ResourceConflictRatio", "L2Cache"))
    sig = build(archive)
    weak = [h for h in sig["architectural_hypotheses"] if h["strength"] == "weak"]
    assert any("无法区分" in h["claim"] for h in weak)


# ---------------------------------------------------------------------------
# 数据复用倍数
# ---------------------------------------------------------------------------

def test_reuse_factor_computed(archive):
    """32768KB 实搬 ÷ 8MB 理论下界 = 4 倍冗余。"""
    sig = build(archive, min_gm_bytes=8 * 1024 * 1024)
    assert sig["memory"]["reuse_factor"]["value"] == pytest.approx(4.0)
    assert "4.0 倍" in claims(sig)


def test_reuse_factor_absent_without_baseline(archive):
    sig = build(archive)
    assert sig["memory"]["reuse_factor"]["value"] is None
    questions = " ".join(q["question"] for q in sig["open_questions"])
    assert "复用倍数" in questions


def test_reuse_near_one_not_flagged(archive):
    """几乎无冗余时不该报「有提升空间」。"""
    sig = build(archive, min_gm_bytes=32768 * 1024)
    assert sig["memory"]["reuse_factor"]["value"] == pytest.approx(1.0)
    assert "冗余 GM 读取" not in claims(sig)


# ---------------------------------------------------------------------------
# 暴露时间（timeline probe 替代）
# ---------------------------------------------------------------------------

def test_exposure_computed(archive):
    exposure = build(archive)["exposure"]
    assert exposure["available"] is True
    assert exposure["exposed_us"] == pytest.approx(456.5 - 392.1)


def test_exposure_uses_max_not_sum(tmp_path):
    """aic 与 aiv 并行，相加会高估覆盖、低估暴露 —— 那会掩盖真问题。"""
    archive = write_archive(tmp_path / "r", pipe={
        "Task Duration(us)": "100", "aic_time(us)": "60", "aiv_time(us)": "55"})
    exposure = build(archive)["exposure"]
    assert exposure["covered_us"] == 60.0       # max，不是 115
    assert exposure["exposed_us"] == pytest.approx(40.0)


def test_serial_latency_bound_detected(tmp_path):
    archive = write_archive(tmp_path / "r", pipe={
        "Task Duration(us)": "100", "aiv_time(us)": "40", "aic_time(us)": "0"})
    sig = build(archive)
    assert sig["exposure"]["serial_latency_bound"] is True
    assert "串行延迟受限" in claims(sig)


def test_not_serial_when_well_covered(tmp_path):
    archive = write_archive(tmp_path / "r", pipe={
        "Task Duration(us)": "100", "aiv_time(us)": "95", "aic_time(us)": "0"})
    assert build(archive)["exposure"]["serial_latency_bound"] is False


def test_exposure_inconsistency_reported_not_clamped(tmp_path):
    """覆盖时间大于总时长说明数据不一致，应如实报告而非裁剪成 0。"""
    archive = write_archive(tmp_path / "r", pipe={
        "Task Duration(us)": "50", "aiv_time(us)": "80", "aic_time(us)": "0"})
    exposure = build(archive)["exposure"]
    assert exposure["available"] is False
    assert "不一致" in exposure["reason"]


def test_serial_bound_claim_is_weak(tmp_path):
    """暴露时间只说有多少闲置，不说闲置在哪 —— 不能当强结论。"""
    archive = write_archive(tmp_path / "r", pipe={
        "Task Duration(us)": "100", "aiv_time(us)": "40", "aic_time(us)": "0"})
    sig = build(archive)
    serial = [h for h in sig["architectural_hypotheses"] if "串行" in h["claim"]]
    assert serial and serial[0]["strength"] == "weak"
    assert "消融" in serial[0]["caveat"]


# ---------------------------------------------------------------------------
# 通道字符串：milestone 的 trigger 要直接引用它
# ---------------------------------------------------------------------------

def test_channel_strings_name_source_csv(archive):
    sig = build(archive)
    channel = sig["pipeline"]["mte2"]["channel"]
    assert channel.startswith("msprof:")
    assert "PipeUtilization.csv" in channel
    assert "aiv_mte2_ratio" in channel


def test_every_hypothesis_carries_a_channel(archive):
    """没有通道的假设无法写进 milestone 的 trigger。"""
    for item in build(archive)["architectural_hypotheses"]:
        assert item["channel"], f"假设缺通道: {item['claim']}"


# ---------------------------------------------------------------------------
# 开放问题
# ---------------------------------------------------------------------------

def test_open_questions_are_answerable(archive):
    """开放问题必须带「为何答不了」和「怎么才能答」，否则只是免责声明。"""
    questions = build(archive)["open_questions"]
    assert questions
    for q in questions:
        assert q["why_unresolvable"]
        assert q["proposed_probe"]


def test_tile_shape_remains_open(archive):
    text = " ".join(q["question"] for q in build(archive)["open_questions"])
    assert "tile" in text.lower()


# ---------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------

def test_empty_dir_reports_search_paths(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(sp.ProbeError) as exc:
        sp.build_signature(None, empty, "AddLayerNorm", None)
    assert "op_summary_<Metric>.csv" in str(exc.value)


def test_unknown_op_name_reports_available_files(archive):
    with pytest.raises(sp.ProbeError) as exc:
        sp.build_signature(None, Path(archive), "NoSuchOperator", None)
    assert "没有匹配到算子" in str(exc.value)


def test_op_name_substring_match(archive):
    sig = sp.build_signature(None, Path(archive), "LayerNorm", None)
    assert sig["op_name"] == "AddLayerNorm"


def test_longest_row_picked_without_op_name(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    (root / "op_summary_PipeUtilization.csv").write_text(
        "Op Name,Task Duration(us),aiv_mte2_ratio\n"
        "TinyOp,1.0,10\nBigOp,900.0,61.2\n", encoding="utf-8")
    sig = sp.build_signature(None, root, None, None)
    assert sig["op_name"] == "BigOp"


def test_output_is_json_serialisable(archive):
    json.dumps(build(archive), ensure_ascii=False)


def test_text_render_does_not_crash(archive):
    assert "架构假设" in sp.render_text(build(archive))


def test_text_render_shows_withheld_reason(tmp_path):
    archive = write_archive(tmp_path / "r", pipe={"Current Freq": "1000"})
    assert "扣留" in sp.render_text(build(archive))
