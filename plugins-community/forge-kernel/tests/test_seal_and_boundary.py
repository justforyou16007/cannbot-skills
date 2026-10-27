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
"""封存与 no-code 边界 hook 的测试。

这一层是整套方法的地基：参考实现只能测、不能读。地基塌了，后面所有
「架构是反推出来的」说法都不成立 —— 所以这里的用例比别处更偏执，
正反两面都要钉死：

  该拦的要拦   读参考源码、grep 参考目录、拿它当 Glob 起点
  不该拦的别拦  读无关文件、跑无关命令、尚未封存任何参考时

最后一条尤其重要：误拦会让人去关掉 hook，关掉之后真正的越界就无人把守了。

运行: python3 -m pytest plugins-community/forge-kernel/tests/ -v
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
HOOK = PLUGIN_ROOT / "hooks" / "no-code-boundary.sh"
SEAL_SCRIPT = (PLUGIN_ROOT / "skills" / "forge-campaign" / "scripts"
               / "seal_reference.py")

sys.path.insert(0, str(SEAL_SCRIPT.parent))

import seal_reference  # noqa: E402


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

REF_SOURCE = """\
#include "kernel_operator.h"
#define REF_PRIVATE_TILE_BYTES 32768
class ProprietaryAttentionCore {
public:
    __aicore__ inline void Init(GM_ADDR x) { xGm.SetGlobalBuffer((__gm__ half*)x); }
private:
    AscendC::GlobalTensor<half> xGm;
    AscendC::LocalTensor<float> buf;
};
"""


@pytest.fixture
def sealed(tmp_path):
    """造一份假的参考实现并封存，返回 (work_dir, 参考目录, 源码文件)。"""
    artifact = tmp_path / "ref_impl" / "op_kernel"
    artifact.mkdir(parents=True)
    source = artifact / "ref_kernel.cpp"
    source.write_text(REF_SOURCE, encoding="utf-8")

    work_dir = tmp_path / "forge_work"
    seal_reference.seal(artifact.parent, "demo_attn", work_dir)
    return work_dir, artifact.parent, source


def call_hook(work_dir, payload, disable=False):
    """按 Claude Code 的方式调用 hook，返回 (退出码, stderr)。"""
    env = dict(os.environ)
    env["FORGE_SEALED_PATHS"] = str(work_dir / "sealed_paths.txt")
    if disable:
        env["FORGE_BOUNDARY_HOOK_DISABLE"] = "1"
    proc = subprocess.run(
        ["bash", str(HOOK)],
        input=json.dumps(payload), capture_output=True, text=True,
        env=env, timeout=30, check=False,
    )
    return proc.returncode, proc.stderr


# ---------------------------------------------------------------------------
# 封存产物
# ---------------------------------------------------------------------------

def test_seal_extracts_reference_specific_identifiers(sealed):
    work_dir, _, _ = sealed
    tokens = seal_reference.DENYLIST_FILENAME
    text = (work_dir / tokens).read_text(encoding="utf-8")
    assert "ProprietaryAttentionCore" in text
    assert "REF_PRIVATE_TILE_BYTES" in text


def test_seal_does_not_flag_common_ascendc_api(sealed):
    """通用 API 名不该进禁用名单 —— 误伤会让 lint 变成噪声而被忽略。"""
    work_dir, _, _ = sealed
    text = (work_dir / seal_reference.DENYLIST_FILENAME).read_text(encoding="utf-8")
    for api in ("GlobalTensor", "LocalTensor", "SetGlobalBuffer"):
        assert api not in text, f"{api} 是通用 API，不应被当作参考专有名词"


def test_seal_lists_source_files_as_off_limits(sealed):
    work_dir, ref_dir, source = sealed
    listed = (work_dir / seal_reference.SEALED_PATHS_FILENAME).read_text(encoding="utf-8")
    assert str(ref_dir.resolve()) in listed
    assert str(source.resolve()) in listed


def test_seal_rejects_missing_artifact(tmp_path):
    with pytest.raises(seal_reference.SealError, match="不存在"):
        seal_reference.seal(tmp_path / "nope", "demo", tmp_path / "w")


def test_seal_rejects_empty_artifact(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(seal_reference.SealError, match="没有任何文件"):
        seal_reference.seal(empty, "demo", tmp_path / "w")


def test_identifier_extraction_skips_short_names():
    assert seal_reference.extract_identifiers("class Foo {};") == set()


def test_manifest_detects_tampering(sealed, tmp_path, capsys):
    """参考被改过就要报出来 —— 参考变了，之前的加速比全不可比。"""
    import argparse
    work_dir, _, source = sealed
    args = argparse.Namespace(work_dir=str(work_dir), as_json=False)
    assert seal_reference.cmd_manifest(args) == 0

    source.write_text(REF_SOURCE + "// 动了一下\n", encoding="utf-8")
    assert seal_reference.cmd_manifest(args) == 1
    assert "已变更" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 该拦的要拦
# ---------------------------------------------------------------------------

def test_hook_blocks_reading_reference_source(sealed):
    work_dir, _, source = sealed
    code, err = call_hook(work_dir, {
        "tool_name": "Read", "tool_input": {"file_path": str(source)}})
    assert code == 2
    assert "目标路径位于封存的参考实现内" in err


def test_hook_blocks_grep_on_reference_dir(sealed):
    work_dir, ref_dir, _ = sealed
    code, err = call_hook(work_dir, {
        "tool_name": "Grep", "tool_input": {"pattern": "Tile", "path": str(ref_dir)}})
    assert code == 2


def test_hook_blocks_bash_touching_reference(sealed):
    work_dir, ref_dir, _ = sealed
    code, err = call_hook(work_dir, {
        "tool_name": "Bash",
        "tool_input": {"command": f"grep -rn Tile {ref_dir}/"}})
    assert code == 2
    assert "命令中引用了封存路径" in err


def test_hook_message_points_at_the_measuring_alternative(sealed):
    """拦截不能只说「不许」，要给出正确做法，否则 agent 会卡死在这里。"""
    work_dir, _, source = sealed
    _, err = call_hook(work_dir, {
        "tool_name": "Read", "tool_input": {"file_path": str(source)}})
    assert "msprof" in err
    assert "配对评测" in err


# ---------------------------------------------------------------------------
# 不该拦的别拦
# ---------------------------------------------------------------------------

def test_hook_allows_unrelated_file(sealed, tmp_path):
    work_dir, _, _ = sealed
    other = tmp_path / "my_kernel.cpp"
    other.write_text("// 我自己写的\n", encoding="utf-8")
    code, _ = call_hook(work_dir, {
        "tool_name": "Read", "tool_input": {"file_path": str(other)}})
    assert code == 0


def test_hook_allows_unrelated_command(sealed):
    work_dir, _, _ = sealed
    code, _ = call_hook(work_dir, {
        "tool_name": "Bash", "tool_input": {"command": "npu-smi info"}})
    assert code == 0


def test_hook_allows_everything_when_nothing_sealed(tmp_path):
    """尚未封存任何参考时不得妨碍正常工作。"""
    code, _ = call_hook(tmp_path / "no_such_dir", {
        "tool_name": "Read", "tool_input": {"file_path": "/etc/hostname"}})
    assert code == 0


def test_hook_survives_malformed_json(sealed):
    work_dir, _, _ = sealed
    proc = subprocess.run(
        ["bash", str(HOOK)], input="这不是 JSON", capture_output=True, text=True,
        env={**os.environ,
             "FORGE_SEALED_PATHS": str(work_dir / "sealed_paths.txt")},
        timeout=30, check=False,
    )
    assert proc.returncode == 0


def test_hook_survives_empty_tool_input(sealed):
    work_dir, _, _ = sealed
    code, _ = call_hook(work_dir, {"tool_name": "Read", "tool_input": {}})
    assert code == 0


def test_hook_escape_hatch_warns_without_blocking(sealed):
    """逃生舱只降级为告警，而且必须明说自己开着。"""
    work_dir, _, source = sealed
    code, err = call_hook(
        work_dir,
        {"tool_name": "Read", "tool_input": {"file_path": str(source)}},
        disable=True,
    )
    assert code == 0
    assert "逃生舱已开启" in err


# ---------------------------------------------------------------------------
# 与知识库 lint 的衔接
# ---------------------------------------------------------------------------

def test_sealed_denylist_feeds_knowledge_base_lint(sealed):
    """封存抽出的名词清单要能直接被知识库 lint 用上，两段不能脱节。"""
    hwkb_scripts = PLUGIN_ROOT / "skills" / "forge-hwkb" / "scripts"
    sys.path.insert(0, str(hwkb_scripts))
    import hwkb  # noqa: PLC0415

    work_dir, _, _ = sealed
    denylist = hwkb.load_denylist(work_dir)
    assert "ProprietaryAttentionCore" in denylist

    card = (
        "---\n"
        "id: R-2201-901\n"
        "arch: DAV_2201\n"
        "mechanism: 照搬 ProprietaryAttentionCore 的流水编排\n"
        "symptom: MTE2 偏高\n"
        "symptom_channel: evaluation_results.json:baseline.pipeline.mte2_pct\n"
        "---\n"
        "## 机制\n按参考做。\n"
    )
    meta, body = hwkb.parse_card(card)
    issues = hwkb.check_no_code_boundary(meta, body, denylist)
    assert "L-DENY" in {i.code for i in issues}
