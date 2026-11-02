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


# ---------------------------------------------------------------------------
# 封存范围：配置文件和源码一样要禁读
#
# 算子配置表（<op>.json）把参考实现了哪些分支、每个分支用什么 tiling key、
# 支持哪些输入形态组合全列了出来。那正是本该从硬件行为反推的结论，而且
# 照抄它连代码都不用读。早先只封源码后缀，这类文件是完全敞开的。
# ---------------------------------------------------------------------------

def test_config_json_is_sealed(tmp_path):
    artifact = tmp_path / "ref"
    artifact.mkdir()
    (artifact / "k.cpp").write_text(REF_SOURCE, encoding="utf-8")
    config = artifact / "add_layer_norm.json"
    config.write_text('{"tiling_keys": [1000, 1001]}', encoding="utf-8")

    work_dir = tmp_path / "forge_work"
    manifest = seal_reference.seal(artifact, "demo", work_dir,
                                   use_public_api_filter=False)

    assert manifest["config_file_count"] == 1
    sealed = (work_dir / "sealed_paths.txt").read_text(encoding="utf-8")
    assert str(config.resolve()) in sealed


def test_sealed_config_blocked_even_when_reached_by_real_path(tmp_path):
    """软链中转目录是推荐的封存做法，禁读必须落在软链指向的真实路径上。

    实际流程是「挑出本算子的文件 → 软链到中转目录 → 封中转目录」。
    若清单里只有中转目录这一个前缀，绕过软链直接读真实路径就放行了。
    """
    real = tmp_path / "cann" / "config"
    real.mkdir(parents=True)
    config = real / "add_layer_norm.json"
    config.write_text('{"tiling_keys": [1000]}', encoding="utf-8")

    stage = tmp_path / "ref_stage"
    stage.mkdir()
    (stage / "k.cpp").write_text(REF_SOURCE, encoding="utf-8")
    (stage / "add_layer_norm.json").symlink_to(config)

    work_dir = tmp_path / "forge_work"
    seal_reference.seal(stage, "demo", work_dir, use_public_api_filter=False)

    code, err = call_hook(
        work_dir,
        {"tool_name": "Read", "tool_input": {"file_path": str(config)}})
    assert code == 2
    assert "封存" in err


def test_symlink_to_sealed_source_is_blocked(tmp_path, sealed):
    """从别处软链到封存源码再读，同样算越界。"""
    work_dir, _, source = sealed
    shortcut = tmp_path / "shortcut.cpp"
    shortcut.symlink_to(source)

    code, _ = call_hook(
        work_dir,
        {"tool_name": "Read", "tool_input": {"file_path": str(shortcut)}})
    assert code == 2


# ---------------------------------------------------------------------------
# 专有名词：公开接口与通用词都不该进名单
#
# 误伤的代价不是多一行输出。围栏只能按规则整条豁免（--exempt F-DENY），
# 一次误伤就可能让整类检查被关掉，真正的转发也跟着放过去。
# ---------------------------------------------------------------------------

def test_public_api_names_excluded_from_denylist(tmp_path):
    headers = tmp_path / "ascendc" / "include"
    headers.mkdir(parents=True)
    (headers / "kernel_operator.h").write_text(
        "namespace AscendC {\n"
        "struct DataCopyExtParams {};\n"
        "void DataCopyPad(int a);\n"
        "void WholeReduceSum(int a);\n"
        "}\n", encoding="utf-8")

    artifact = tmp_path / "ref"
    artifact.mkdir()
    (artifact / "k.cpp").write_text(
        "class KernelSecretAttention {\n"
        "  void Run() {\n"
        "    AscendC::DataCopyPad(x);\n"
        "    AscendC::WholeReduceSum(y);\n"
        "    AscendC::DataCopyExtParams p;\n"
        "  }\n"
        "};\n", encoding="utf-8")

    work_dir = tmp_path / "forge_work"
    manifest = seal_reference.seal(artifact, "demo", work_dir,
                                   public_header_dirs=[str(headers)])

    denylist = (work_dir / "denylist.txt").read_text(encoding="utf-8").split()
    assert "KernelSecretAttention" in denylist
    for public_name in ("DataCopyPad", "WholeReduceSum", "DataCopyExtParams"):
        assert public_name not in denylist
    assert set(manifest["public_api_filter"]["dropped_tokens"]) >= {
        "DataCopyPad", "WholeReduceSum", "DataCopyExtParams"}


def test_generic_single_words_not_treated_as_proprietary(tmp_path):
    """`class Kernel {` 这种写法到处都有，Kernel 进名单等于拦住所有候选。"""
    artifact = tmp_path / "ref"
    artifact.mkdir()
    (artifact / "k.cpp").write_text(
        "class Kernel {};\n"
        "class Buffer {};\n"
        "struct Tiling {};\n"
        "#define BUFFER_NUM 2\n"
        "class KernelAddLayerNorm {};\n", encoding="utf-8")

    work_dir = tmp_path / "forge_work"
    seal_reference.seal(artifact, "demo", work_dir, use_public_api_filter=False)
    denylist = (work_dir / "denylist.txt").read_text(encoding="utf-8").split()

    assert "KernelAddLayerNorm" in denylist
    for generic in ("Kernel", "Buffer", "Tiling", "BUFFER_NUM"):
        assert generic not in denylist


def test_public_api_allowlist_is_cached_for_reuse(tmp_path):
    headers = tmp_path / "inc"
    headers.mkdir()
    (headers / "a.h").write_text("void DataCopyPad(int);\n", encoding="utf-8")
    artifact = tmp_path / "ref"
    artifact.mkdir()
    (artifact / "k.cpp").write_text("class KernelFoobar {};\n", encoding="utf-8")

    work_dir = tmp_path / "forge_work"
    seal_reference.seal(artifact, "demo", work_dir,
                        public_header_dirs=[str(headers)])
    cached = work_dir / "public_api_allowlist.txt"
    assert cached.is_file()
    assert "DataCopyPad" in cached.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# hook 的耗时
#
# 这个 hook 挂在 Read/Grep/Glob/Bash 上，每次工具调用都要等它返回，所以
# 它的耗时直接变成 agent 每一步的固定开销。实测过一版「清单 N 条 × 目标
# M 个」的嵌套 shell 循环：7583 条清单时单次调用 94 秒，而 hooks.json 给
# 的超时是 10 秒 —— 要么被杀、要么白等，agent 循环推不动。
# 这条用例盯住的就是别再退回那种写法。
# ---------------------------------------------------------------------------

def test_hook_stays_fast_with_a_long_sealed_list(tmp_path):
    import time  # noqa: PLC0415

    sealed_list = tmp_path / "sealed_paths.txt"
    sealed_list.write_text(
        "# 大清单\n"
        + "\n".join(f"/opt/ref/op_{i}/op_{i}.cpp" for i in range(4000)) + "\n",
        encoding="utf-8")

    work_dir = tmp_path  # call_hook 只用 work_dir/sealed_paths.txt
    start = time.monotonic()
    code, _ = call_hook(
        work_dir,
        {"tool_name": "Bash", "tool_input": {"command": "ls -la /tmp"}})
    elapsed = time.monotonic() - start

    assert code == 0
    assert elapsed < 5.0, f"4000 条清单下单次 hook 耗时 {elapsed:.1f}s，已接近超时"


def test_hook_still_blocks_with_a_long_sealed_list(tmp_path):
    """快了但不能漏。"""
    target = tmp_path / "op_3999.cpp"
    target.write_text("// ref\n", encoding="utf-8")
    sealed_list = tmp_path / "sealed_paths.txt"
    sealed_list.write_text(
        "\n".join(f"/opt/ref/op_{i}/op_{i}.cpp" for i in range(4000))
        + f"\n{target.resolve()}\n",
        encoding="utf-8")

    code, _ = call_hook(
        tmp_path,
        {"tool_name": "Read", "tool_input": {"file_path": str(target)}})
    assert code == 2


def test_reseal_removes_public_api_tokens_left_by_an_earlier_seal(tmp_path):
    """「重新封存一次」这个补救动作必须真的能去掉误伤源。

    名词清单是合并写入的（多个参考要并存）。若只对本次抽出的词做过滤，
    上一次封存（没带 --public-headers 那次）写进去的 DataCopyPad 会永远
    留在清单里 —— 文档让人重新封存，而重新封存对它无效，只能手工删文件。
    """
    headers = tmp_path / "inc"
    headers.mkdir()
    (headers / "a.h").write_text("void DataCopyPad(int);\n", encoding="utf-8")

    artifact = tmp_path / "ref"
    artifact.mkdir()
    (artifact / "k.cpp").write_text(
        "class KernelSecret { void f() { DataCopyPad(1); } };\n",
        encoding="utf-8")

    work_dir = tmp_path / "forge_work"

    # 第一次：没有白名单，公开接口名进了清单
    seal_reference.seal(artifact, "demo", work_dir, use_public_api_filter=False)
    first = (work_dir / "denylist.txt").read_text(encoding="utf-8").split()
    assert "DataCopyPad" in first

    # 第二次：带上公开头文件重封，旧的误伤源要被清掉
    manifest = seal_reference.seal(artifact, "demo", work_dir,
                                   public_header_dirs=[str(headers)])
    second = (work_dir / "denylist.txt").read_text(encoding="utf-8").split()
    assert "DataCopyPad" not in second
    assert "KernelSecret" in second
    assert "DataCopyPad" in manifest["public_api_filter"]["removed_stale_tokens"]


def test_reseal_keeps_hand_added_tokens(tmp_path):
    """手工追加的名词不能被重封弄丢 —— 清单是合并而非覆盖。"""
    artifact = tmp_path / "ref"
    artifact.mkdir()
    (artifact / "k.cpp").write_text("class KernelSecret {};\n", encoding="utf-8")
    work_dir = tmp_path / "forge_work"
    seal_reference.seal(artifact, "demo", work_dir, use_public_api_filter=False)

    denylist = work_dir / "denylist.txt"
    denylist.write_text(denylist.read_text(encoding="utf-8")
                        + "MyHandAddedMarker\n", encoding="utf-8")

    seal_reference.seal(artifact, "demo", work_dir, use_public_api_filter=False)
    assert "MyHandAddedMarker" in denylist.read_text(encoding="utf-8").split()


# ---------------------------------------------------------------------------
# 递归搜索的两个方向
#
# Grep/Glob 会从起点递归向下搜。所以「起点在封存目录内」和「起点是封存目录
# 的上层」都能读到参考源码，是一回事。只拦前者等于留了条绕过去的路。
#
# 代价是封存目录一旦落在工作目录下，`Grep path=.` 就成了「上层目录」。
# 所以中转目录要放在工作区之外 —— 位置选错时 seal 会当场警告。
# ---------------------------------------------------------------------------

def outside_layout(tmp_path):
    """推荐布局：中转目录在工作区之外。返回 (work_dir, 参考目录)。"""
    stage = tmp_path / "outside" / "ref_stage"
    stage.mkdir(parents=True)
    (stage / "ref.cpp").write_text(REF_SOURCE, encoding="utf-8")

    work_dir = tmp_path / "project" / "forge_work"
    work_dir.mkdir(parents=True)
    seal_reference.seal(stage, "demo", work_dir, use_public_api_filter=False)
    return work_dir, stage


def test_grep_starting_above_sealed_tree_is_blocked(tmp_path):
    work_dir, stage = outside_layout(tmp_path)
    code, err = call_hook(work_dir, {
        "tool_name": "Grep",
        "tool_input": {"pattern": "Proprietary", "path": str(stage.parent)}})
    assert code == 2
    assert "上层目录" in err


def test_grep_inside_own_work_tree_is_allowed(tmp_path):
    """中转目录在工作区外时，搜自己的代码不能被拦。

    误拦会让人把 hook 关掉，而关掉之后真正的越界就无人把守了。
    """
    work_dir, _ = outside_layout(tmp_path)
    own = tmp_path / "project" / "mycode"
    own.mkdir()
    code, _ = call_hook(work_dir, {
        "tool_name": "Grep",
        "tool_input": {"pattern": "x", "path": str(own)}})
    assert code == 0


def test_grep_without_path_uses_cwd(tmp_path, monkeypatch):
    """不给 path 的 grep 默认搜当前目录，这条路同样要判。"""
    work_dir, stage = outside_layout(tmp_path)
    monkeypatch.chdir(stage.parent)
    code, _ = call_hook(work_dir,
                        {"tool_name": "Grep", "tool_input": {"pattern": "x"}})
    assert code == 2


def test_read_of_a_parent_directory_is_not_treated_as_recursive(tmp_path):
    """Read 读一个具体文件不会递归，不该套用「上层目录」这条判定。"""
    work_dir, stage = outside_layout(tmp_path)
    sibling = stage.parent / "notes.md"
    sibling.write_text("# 自己的笔记\n", encoding="utf-8")
    code, _ = call_hook(work_dir, {
        "tool_name": "Read", "tool_input": {"file_path": str(sibling)}})
    assert code == 0


def test_seal_warns_when_staging_dir_is_inside_the_work_tree(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    """位置选错要在封存时就说清，不能等第一次被误拦时让人去猜。"""
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    stage = project / "ref_stage"
    stage.mkdir()
    (stage / "ref.cpp").write_text(REF_SOURCE, encoding="utf-8")

    sys.argv = ["seal_reference.py", "seal", "--artifact", str(stage),
                "--op-name", "demo", "--work-dir", str(project / "forge_work"),
                "--no-public-api-filter"]
    assert seal_reference.main() == 0
    err = capsys.readouterr().err
    assert "工作区之外" in err or "之内" in err
