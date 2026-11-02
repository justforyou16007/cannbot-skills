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


# ===========================================================================
# 采集态目录：多出来的 PROF_GROUP_* 一层，以及多次采集并存
# ===========================================================================

def write_collection(root: Path, group: str, **kw) -> Path:
    """造一个采集态目录。

    实机落盘比文档里记的多一层 group：
        <root>/PROF_GROUP_<时间戳>/PROF_<Metric>/PROF_000001_x/
            mindstudio_profiler_output/op_summary_*.csv
    """
    import shutil  # noqa: PLC0415

    staging = root / f"_staging_{group}"
    write_archive(staging, **kw)
    for csv_file in sorted(staging.glob("op_summary_*.csv")):
        metric = csv_file.stem.replace("op_summary_", "")
        dest = (root / group / f"PROF_{metric}" / "PROF_000001_x"
                / "mindstudio_profiler_output" / "op_summary_1.csv")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(csv_file.read_text(encoding="utf-8"), encoding="utf-8")
    shutil.rmtree(staging)
    return root


def test_prof_group_layer_is_tolerated(tmp_path):
    """文档记的树不带 group 层，实机带。写死一层就会「采到了却说没找到」。"""
    root = write_collection(tmp_path / "prof_ref", "PROF_GROUP_20261030_171959")
    found, _ = sp.locate_metric_csvs(root, None)
    assert "PipeUtilization" in found
    sig = sp.build_signature(root, None, "AddLayerNorm", None)
    assert sig["op_name"] == "AddLayerNorm"


def test_multiple_collections_refuse_to_guess(tmp_path):
    """多次采集并存时不许静默挑一次。

    早先是排序取第一个，于是时间最早、可能只采到两个 metric 的那次被选中。
    缺 metric 的签名里停顿数据是缺失的，读起来像「完全没有停顿」—— 而
    下游未必检查这件事，于是一个错的基线能一路跑下去。
    """
    root = tmp_path / "prof_ref"
    write_collection(root, "PROF_GROUP_20261030_171708",
                     include=("PipeUtilization",))
    write_collection(root, "PROF_GROUP_20261030_171959")

    with pytest.raises(sp.ProbeError) as excinfo:
        sp.locate_metric_csvs(root, None)
    message = str(excinfo.value)
    assert "PROF_GROUP_20261030_171708" in message
    assert "PROF_GROUP_20261030_171959" in message
    assert "--group" in message


def test_group_option_selects_the_complete_collection(tmp_path):
    root = tmp_path / "prof_ref"
    write_collection(root, "PROF_GROUP_20261030_171708",
                     include=("PipeUtilization",))
    write_collection(root, "PROF_GROUP_20261030_171959")

    found, missing = sp.locate_metric_csvs(
        root, None, "PROF_GROUP_20261030_171959")
    assert "ResourceConflictRatio" in found
    assert "ResourceConflictRatio" not in missing


def test_unknown_group_lists_what_is_available(tmp_path):
    root = write_collection(tmp_path / "prof_ref", "PROF_GROUP_A")
    with pytest.raises(sp.ProbeError) as excinfo:
        sp.locate_metric_csvs(root, None, "PROF_GROUP_NOPE")
    assert "PROF_GROUP_A" in str(excinfo.value)


# ===========================================================================
# 冷启动行
#
# 一次采集里同一个算子有十来行（每次执行一行），头几次是冷启动。冷启动行
# 的核执行时间会超过任务总时长 —— 这在物理上不可能，是计时窗口没对齐。
# 它实测比稳态高 75%，而它看起来只是一个正常的耗时数字；这个数会成为整个
# campaign 的参考基线，后面每一轮候选都在和一个虚高的数做比。
# ===========================================================================

PIPE_COLUMNS_WITH_TASK_ID = PIPE_COLUMNS + ",Task ID"


def write_multi_row_pipe(root: Path, rows) -> Path:
    """rows: [(task_id, duration, aiv_time), ...]"""
    root.mkdir(parents=True, exist_ok=True)
    names = PIPE_COLUMNS_WITH_TASK_ID.split(",")
    lines = [PIPE_COLUMNS_WITH_TASK_ID]
    for task_id, duration, aiv in rows:
        values = dict(PIPE_DEFAULTS)
        values.update({"Task Duration(us)": str(duration),
                       "aiv_time(us)": str(aiv), "Task ID": str(task_id)})
        lines.append(",".join(str(values.get(n, "")) for n in names))
    (root / "op_summary_PipeUtilization.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return root


#: 实测的十行：12 是冷启动（aiv_time > Task Duration），15 起进入稳态。
REAL_ROWS = [(12, 408.52, 478.15), (13, 293.00, 215.30), (14, 284.36, 213.68),
             (15, 233.16, 215.65), (16, 231.88, 214.00), (17, 234.18, 215.00),
             (18, 233.30, 214.50), (19, 232.16, 213.90), (20, 234.24, 215.20),
             (21, 233.44, 214.80)]


def test_cold_start_row_is_not_used(tmp_path):
    root = write_archive(tmp_path / "r", include=("Memory", "L2Cache",
                                                  "ResourceConflictRatio"))
    write_multi_row_pipe(root, REAL_ROWS)

    sig = sp.build_signature(None, root, "AddLayerNorm", None)
    picked = sig["launch"]["task_duration_us"]["value"]
    assert picked != pytest.approx(408.52), "取到了冷启动行"
    assert 230.0 < picked < 240.0, picked


def test_row_selection_is_written_into_the_signature(tmp_path):
    """用了哪一行决定全部后续结论，不该是个看不见的内部决定。"""
    root = write_archive(tmp_path / "r", include=("Memory", "L2Cache",
                                                  "ResourceConflictRatio"))
    write_multi_row_pipe(root, REAL_ROWS)

    sig = sp.build_signature(None, root, "AddLayerNorm", None)
    info = sig["row_selection"]["PipeUtilization"]
    assert info["candidates"] == 10
    assert info["dropped_cold_start"] == 1
    assert info["picked_task_id"] is not None
    assert info["duration_max_us"] == pytest.approx(293.00)


def test_steady_from_narrows_to_given_task_id(tmp_path):
    root = write_archive(tmp_path / "r", include=("Memory", "L2Cache",
                                                  "ResourceConflictRatio"))
    write_multi_row_pipe(root, REAL_ROWS)

    sig = sp.build_signature(None, root, "AddLayerNorm", None,
                             None, 15)
    info = sig["row_selection"]["PipeUtilization"]
    assert info["dropped_by_steady_from"] == 3
    assert info["duration_max_us"] == pytest.approx(234.24)


def test_steady_state_check_appears_in_trust(tmp_path):
    root = write_archive(tmp_path / "r", include=("Memory", "L2Cache",
                                                  "ResourceConflictRatio"))
    write_multi_row_pipe(root, REAL_ROWS)

    sig = sp.build_signature(None, root, "AddLayerNorm", None)
    names = {c["name"] for c in sig["trust"]["checks"]}
    assert "steady_state" in names


def test_all_rows_cold_start_is_untrustworthy(tmp_path):
    """整次采集都没有稳态数据时，不许挑一行蒙过去。"""
    root = write_archive(tmp_path / "r", include=("Memory", "L2Cache",
                                                  "ResourceConflictRatio"))
    write_multi_row_pipe(root, [(12, 408.52, 478.15), (13, 400.0, 450.0)])

    sig = sp.build_signature(None, root, "AddLayerNorm", None)
    assert sig["trust"]["trustworthy"] is False
    assert sig["architectural_hypotheses"] == []


def test_longest_op_is_identified_before_steady_state_pick(tmp_path):
    """不给算子名时先认出被测算子，再在它自己的多次执行里挑稳态行。

    跨算子取中位数没有意义 —— 那是把 TinyOp 和 BigOp 混在一起算。
    """
    root = tmp_path / "r"
    root.mkdir()
    names = PIPE_COLUMNS_WITH_TASK_ID.split(",")
    lines = [PIPE_COLUMNS_WITH_TASK_ID]
    for op, task_id, duration in [("TinyOp", 1, 1.0), ("BigOp", 2, 900.0),
                                  ("BigOp", 3, 890.0), ("TinyOp", 4, 1.2)]:
        values = dict(PIPE_DEFAULTS)
        values.update({"Op Name": op, "Op Type": op, "Task ID": str(task_id),
                       "Task Duration(us)": str(duration), "aiv_time(us)": "0.5"})
        lines.append(",".join(str(values.get(n, "")) for n in names))
    (root / "op_summary_PipeUtilization.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    sig = sp.build_signature(None, root, None, None)
    assert sig["op_name"] == "BigOp"
    assert sig["row_selection"]["PipeUtilization"]["candidates"] == 2


def test_unrankable_rows_do_not_claim_a_steady_state_pick(tmp_path):
    """没有可解析的耗时列时，两道冷启动防线都失效，可信度不能报「通过」。

    否则签名里会出现一个并不存在的稳态选择：既没丢冷启动行，也没取中位数，
    实际取的是第一行 —— 而那正是冷启动行。
    """
    root = tmp_path / "r"
    root.mkdir()
    (root / "op_summary_PipeUtilization.csv").write_text(
        "Op Name,Task Duration(us),aiv_mte2_ratio,Current Freq,Rated Freq\n"
        "AddLayerNorm,N/A,61.2,1800,1800\n"
        "AddLayerNorm,N/A,12.0,1800,1800\n", encoding="utf-8")

    sig = sp.build_signature(None, root, "AddLayerNorm", None)
    info = sig["row_selection"]["PipeUtilization"]
    assert info["duration_ranked"] is False
    assert "Task Duration" in info["note"]

    steady = [c for c in sig["trust"]["checks"] if c["name"] == "steady_state"]
    assert steady and steady[0]["verdict"] == "unknown"


# ===========================================================================
# 比例列的口径：0-1 小数还是百分数
#
# msprof 同一类数据有两种写法，列名会告诉你是哪种：带 `(%)` 的已是百分数，
# 不带的是 0-1 小数（本仓 ops-profiling/scripts/perf_summary.py 正是对后者
# 乘 100 再显示）。
#
# 探针的阈值全是百分数。把 0-1 原始值直接拿去比，这些闸门一个都不可能触发 ——
# 最大可能值 1.0 连最低的阈值 3 都到不了。而输出看着完全正常：限制资源会
# 显示 0.874 而不是 87.4%，可信度检查里的 ICache 那条也就此失效。
# ===========================================================================

FRACTION_SCALE = {
    "aiv_vec_ratio": "0.874", "aic_cube_ratio": "0",
    "aiv_scalar_ratio": "0.12", "aic_mte1_ratio": "0",
    "aiv_mte2_ratio": "0.55", "aiv_mte3_ratio": "0.20",
    "aic_fixpipe_ratio": "0", "aiv_icache_miss_rate": "0.18",
}
FRACTION_CONFLICT = {
    "aiv_vec_total_cflt_ratio": "0.16",
    "aiv_vec_bankgroup_cflt_ratio": "0.142",
    "aiv_vec_bank_cflt_ratio": "0.045",
    "aiv_vec_resc_cflt_ratio": "0.01", "aiv_vec_mte_cflt_ratio": "0.006",
    "aiv_vec_wait_ratio": "0.318", "aiv_mte2_wait_ratio": "0.042",
    "aiv_mte3_wait_ratio": "0.02",
}
PERCENT_SCALE = {k: str(float(v) * 100) for k, v in FRACTION_SCALE.items()}
PERCENT_CONFLICT = {k: str(float(v) * 100) for k, v in FRACTION_CONFLICT.items()}


def flags_of(sig):
    return ({f["flag"] for f in sig["pipeline"]["flags"]}
            | {f["flag"] for f in sig["stalls"]["flags"]})


def test_fraction_scale_csv_is_normalised_to_percent(tmp_path):
    root = write_archive(tmp_path / "frac", pipe=FRACTION_SCALE,
                         conflict=FRACTION_CONFLICT)
    sig = sp.build_signature(None, root, "AddLayerNorm", None)

    assert sig["ratio_scale"]["factor"] == 100.0
    assert sig["pipeline"]["vec"]["value"] == pytest.approx(87.4)
    assert sig["pipeline"]["limiting"]["ratio"] == pytest.approx(87.4)
    # 修复前这些闸门一个都不会触发
    assert "VEC Bound" in flags_of(sig)
    assert "MTE2 Bound" in flags_of(sig)
    assert "冲突严重" in flags_of(sig)


def test_percent_scale_csv_is_left_alone(tmp_path):
    root = write_archive(tmp_path / "pct", pipe=PERCENT_SCALE,
                         conflict=PERCENT_CONFLICT)
    sig = sp.build_signature(None, root, "AddLayerNorm", None)

    assert sig["ratio_scale"]["factor"] == 1.0
    assert sig["pipeline"]["vec"]["value"] == pytest.approx(87.4)


def test_both_conventions_give_the_same_reading(tmp_path):
    """同一份测量换个写法，结论必须一致 —— 否则结论取决于 CSV 的格式。"""
    frac = sp.build_signature(
        None, write_archive(tmp_path / "a", pipe=FRACTION_SCALE,
                            conflict=FRACTION_CONFLICT),
        "AddLayerNorm", None)
    pct = sp.build_signature(
        None, write_archive(tmp_path / "b", pipe=PERCENT_SCALE,
                            conflict=PERCENT_CONFLICT),
        "AddLayerNorm", None)

    assert flags_of(frac) == flags_of(pct)
    assert (frac["pipeline"]["limiting"]["value"]
            == pct["pipeline"]["limiting"]["value"])
    for name in ("vec", "mte2", "scalar"):
        assert (frac["pipeline"][name]["value"]
                == pytest.approx(pct["pipeline"][name]["value"]))


def test_icache_trust_gate_fires_on_fraction_scale(tmp_path):
    """ICache 那条是可信度硬闸门，小数口径下原先永远不可能触发。"""
    root = write_archive(tmp_path / "frac", pipe=FRACTION_SCALE,
                         conflict=FRACTION_CONFLICT)
    sig = sp.build_signature(None, root, "AddLayerNorm", None)

    icache = [c for c in sig["trust"]["checks"] if c["name"] == "icache"]
    assert icache and icache[0]["verdict"] == "fail"
    assert sig["trust"]["trustworthy"] is False


def test_percent_suffixed_columns_are_not_scaled(tmp_path):
    """带 (%) 的列本来就是百分数，不能再乘 100。"""
    root = write_archive(tmp_path / "frac", pipe=FRACTION_SCALE,
                         conflict=FRACTION_CONFLICT, l2_hit="42.0")
    sig = sp.build_signature(None, root, "AddLayerNorm", None)

    assert sig["cache"]["total_hit_rate"]["value"] == pytest.approx(42.0)
    assert (sig["memory"]["bw_usage"]["gm_to_ub"]["value"]
            == pytest.approx(38.5))


def test_ratio_scale_basis_is_recorded(tmp_path):
    """判定依据要写出来 —— 判错了所有阈值判定都跟着失效，而输出看着正常。"""
    root = write_archive(tmp_path / "frac", pipe=FRACTION_SCALE,
                         conflict=FRACTION_CONFLICT)
    sig = sp.build_signature(None, root, "AddLayerNorm", None)
    assert "0-1" in sig["ratio_scale"]["basis"]


def test_unknown_ratio_scale_is_blocking(tmp_path):
    """口径判不出来时必须进可信度检查，不能只留在 ratio_scale 里等人去看。

    判错的方向是「闸门永不触发」：占比、等待率、冲突率的阈值全都对不上，
    输出里一条告警都没有，看着像一切正常。
    """
    root = tmp_path / "r"
    root.mkdir()
    # 只有冲突率，没有流水段占比可作参照
    (root / "op_summary_ResourceConflictRatio.csv").write_text(
        "Op Name,aiv_vec_total_cflt_ratio,aiv_mte2_wait_ratio\n"
        "AddLayerNorm,0.16,0.042\n", encoding="utf-8")

    sig = sp.build_signature(None, root, "AddLayerNorm", None)
    assert sig["ratio_scale"]["factor"] == 1.0
    assert "未经确认" in sig["ratio_scale"]["basis"]

    scale_checks = [c for c in sig["trust"]["checks"]
                    if c["name"] == "ratio_scale"]
    assert scale_checks and scale_checks[0]["verdict"] == "fail"
    assert sig["trust"]["trustworthy"] is False
    assert sig["architectural_hypotheses"] == []
