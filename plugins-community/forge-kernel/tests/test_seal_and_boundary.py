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
    assert "封存的参考实现内" in err


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
    """尚未封存任何参考时不得妨碍正常工作。

    「没封存」的真实形态是**不设**那两个环境变量 —— 所以这里不走 call_hook
    （它总会设 FORGE_SEALED_PATHS）。设了却读不到是另一回事：那是配置事故，
    见 test_hook_fails_closed_when_declared_list_is_unreadable。
    """
    env = {k: v for k, v in os.environ.items()
           if k not in ("FORGE_SEALED_PATHS", "FORGE_WORK_DIR")}
    proc = subprocess.run(
        ["bash", str(HOOK)],
        input=json.dumps({"tool_name": "Read",
                          "tool_input": {"file_path": "/etc/hostname"}}),
        text=True, capture_output=True, env=env, cwd=str(tmp_path), check=False)
    assert proc.returncode == 0


def test_hook_fails_closed_when_declared_list_is_unreadable(tmp_path):
    """声明了清单位置却读不到 —— 边界此刻无效，必须拦而不是放行。

    这是打包事故、工作目录被清理、路径拼错的共同形态。静默放行会让整个
    campaign 在无防护下跑完，而所有文档都说边界生效。
    """
    code, err = call_hook(tmp_path / "no_such_dir", {
        "tool_name": "Read", "tool_input": {"file_path": "/etc/hostname"}})
    assert code == 2
    assert "读不到" in err


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

def _hwkb():
    hwkb_scripts = PLUGIN_ROOT / "skills" / "forge-hwkb" / "scripts"
    sys.path.insert(0, str(hwkb_scripts))
    import hwkb  # noqa: PLC0415
    return hwkb


def test_sealed_denylist_feeds_knowledge_base_lint(tmp_path):
    """封存抽出的名词清单要能被知识库 lint 用上，两段不能脱节。

    这条用例原先是**假的**：它把 work_dir 传进 hwkb.load_denylist 的
    kb_root 参数位，等于自己手工把两段接上，然后断言「接上了」。而生产
    路径上 seal 写的是 work_dir、hwkb 读的是 kb_root，中间没有任何一步
    搬运 —— L-DENY 在真实使用中一条都不报，lint 照样打印「通过」。

    现在按真实调用方式测：seal 自己把清单写进 kb_root，hwkb 用它**默认的**
    kb_root 去读。
    """
    hwkb = _hwkb()

    artifact = tmp_path / "ref"
    artifact.mkdir()
    (artifact / "k.cpp").write_text(REF_SOURCE, encoding="utf-8")
    work_dir = tmp_path / "forge_work"
    kb_root = tmp_path / "kb"

    seal_reference.seal(artifact, "demo_attn", work_dir,
                        use_public_api_filter=False, kb_root=kb_root)

    # 关键：从 kb_root 读，而不是从 work_dir 读
    denylist = hwkb.load_denylist(kb_root)
    assert "ProprietaryAttentionCore" in denylist
    assert hwkb.denylist_status(kb_root, denylist) is None

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


def test_seal_writes_denylist_to_the_default_kb_root(tmp_path, monkeypatch):
    """不传 --kb-root 时也要落到 hwkb 默认读的那个位置。

    这是接线的关键一环：实际流程里没人会传 --kb-root，而 hwkb 的所有文档
    调用都走默认 kb_root。两个默认值必须指向同一个文件。
    """
    hwkb = _hwkb()
    assert seal_reference.DEFAULT_KB_ROOT.resolve() == hwkb.DEFAULT_KB_ROOT.resolve()


def test_denylist_status_flags_missing_list(tmp_path):
    hwkb = _hwkb()
    status = hwkb.denylist_status(tmp_path, [])
    assert status and "未生效" in status


def test_denylist_status_flags_empty_list(tmp_path):
    hwkb = _hwkb()
    (tmp_path / "denylist.txt").write_text("# 只有注释\n", encoding="utf-8")
    status = hwkb.denylist_status(tmp_path, [])
    assert status and "形同不存在" in status


def test_lint_says_so_when_denylist_is_not_in_effect(tmp_path, capsys):
    """lint 打印「通过」时，必须说清这句话覆盖了什么。

    L-DENY 没生效而输出「未越过 no-code 边界」，是在给一个没做过的检查
    背书 —— 这正是本轮要消除的那种失效形态。
    """
    import argparse  # noqa: PLC0415
    hwkb = _hwkb()
    (tmp_path / "dav2201").mkdir(parents=True)

    args = argparse.Namespace(kb_root=str(tmp_path), card=None, arch=None,
                              quarantine=False, as_json=False)
    hwkb.cmd_lint(args)
    captured = capsys.readouterr()
    # 结论那一行要**点名**哪道闸没生效，不能只说「见上」——
    # 读结论的人得当场知道这句「通过」不覆盖什么。
    assert "L-DENY" in captured.out
    assert "未生效" in captured.out
    assert "这句话不覆盖它们" in captured.out
    assert "未生效" in captured.err


# ---------------------------------------------------------------------------
# 禁读清单要带真实父目录
#
# 清单原来只有「产物根目录」和「逐个文件」两种条目，而 hook 对 Grep / Glob
# 这类递归工具的判据是「请求路径是不是某个被封条目的祖先」。产物根是软链、
# 或源码散在产物之外（opp_install 那种来源很常见）时，真实父目录就既不等于
# 产物根、也不在逐文件清单里 —— 于是 `grep -rn class /real/parent/` 从两边
# 都漏掉。
# ---------------------------------------------------------------------------

def _sealed_lines(work_dir):
    text = (work_dir / "sealed_paths.txt").read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if ln and not ln.startswith("#")]


def test_nested_parent_dirs_are_sealed(tmp_path):
    artifact = tmp_path / "ref"
    (artifact / "op_kernel" / "inner").mkdir(parents=True)
    (artifact / "op_kernel" / "inner" / "k.cpp").write_text(
        "class KernelSecret {};\n", encoding="utf-8")
    work_dir = tmp_path / "w"
    seal_reference.seal(artifact, "demo", work_dir,
                        use_public_api_filter=False, kb_root=tmp_path / "kb")

    lines = _sealed_lines(work_dir)
    # 每一层都要在，漏掉中间任何一层，对那一层的递归请求就会放行
    assert str((artifact / "op_kernel" / "inner").resolve()) in lines
    assert str((artifact / "op_kernel").resolve()) in lines
    assert str(artifact.resolve()) in lines


def test_sealing_stops_at_the_artifact_root(tmp_path):
    """不能一路封到公共目录去 —— 那会把无关路径一起拦住。"""
    artifact = tmp_path / "ref"
    artifact.mkdir()
    (artifact / "k.cpp").write_text("class KernelSecret {};\n",
                                    encoding="utf-8")
    work_dir = tmp_path / "w"
    seal_reference.seal(artifact, "demo", work_dir,
                        use_public_api_filter=False, kb_root=tmp_path / "kb")
    lines = _sealed_lines(work_dir)
    assert str(tmp_path.resolve()) not in lines      # 产物的父目录不该被封
    assert "/tmp" not in lines


# ---------------------------------------------------------------------------
# check-tree：明文搬运
#
# 封存 hook 拦的是「读」。它拦不住一次成功的读之后发生的事 —— 封存漏挑了
# 一个文件（后缀不在 SOURCE_SUFFIXES 里）、或者在封存之前就已经读过。
# 那之后源码进了候选，而围栏（比对符号名）只在名字没改时才抓得到。
# ---------------------------------------------------------------------------

def _sealed_pair(tmp_path, reference_body):
    artifact = tmp_path / "ref"
    artifact.mkdir()
    (artifact / "kernel.cpp").write_text(reference_body, encoding="utf-8")
    work_dir = tmp_path / "w"
    seal_reference.seal(artifact, "demo", work_dir,
                        use_public_api_filter=False, kb_root=tmp_path / "kb")
    candidate = tmp_path / "cand"
    candidate.mkdir()
    return artifact, work_dir, candidate


REF_BODY = (
    "#include <cstdint>\n"
    "class SecretRefKernel {\n"
    "    __aicore__ inline void ProcessTileWithResidentBuffer(int32_t idx) {\n"
    "        DataCopy(sharedLocal, sharedGm[idx * tileBytes], tileBytes);\n"
    "    }\n"
    "};\n")


def test_clean_candidate_passes_check_tree(tmp_path):
    _a, work_dir, candidate = _sealed_pair(tmp_path, REF_BODY)
    (candidate / "mine.cpp").write_text(
        "void MyOwnKernel() { int x = 1; }\n", encoding="utf-8")
    report = seal_reference.check_tree(candidate, work_dir)
    assert report["passed"] is True


def test_byte_identical_file_is_caught(tmp_path):
    _a, work_dir, candidate = _sealed_pair(tmp_path, REF_BODY)
    (candidate / "copied.cpp").write_text(REF_BODY, encoding="utf-8")
    report = seal_reference.check_tree(candidate, work_dir)
    assert report["passed"] is False
    assert "T-SAME-FILE" in {f["rule"] for f in report["findings"]}


def test_single_distinctive_line_survives_reindentation(tmp_path):
    """归一化要压掉空白 —— 否则改一下缩进就绕过去了。"""
    _a, work_dir, candidate = _sealed_pair(tmp_path, REF_BODY)
    (candidate / "partial.cpp").write_text(
        "void f() {\n"
        "            DataCopy(sharedLocal, sharedGm[idx * tileBytes], "
        "tileBytes);\n"
        "}\n", encoding="utf-8")
    report = seal_reference.check_tree(candidate, work_dir)
    assert "T-SAME-LINES" in {f["rule"] for f in report["findings"]}


def test_short_lines_do_not_trigger(tmp_path):
    """`};`、`return;` 在任何两份 C++ 里都会撞。

    误报的代价是整条检查被豁免掉，所以短行必须丢。
    """
    _a, work_dir, candidate = _sealed_pair(tmp_path, REF_BODY)
    (candidate / "mine.cpp").write_text(
        "class Mine {\n};\n", encoding="utf-8")
    assert seal_reference.check_tree(candidate, work_dir)["passed"] is True


def test_comments_are_stripped_before_comparing(tmp_path):
    """注释是说明不是实现，而两份独立实现完全可以有同样的说明
    （都抄了同一份公开文档的措辞）。"""
    body = ("// 这一段注释足够长，长到超过特征行的字符下限很多很多\n"
            "void f() { int a = 1; }\n")
    artifact = tmp_path / "ref"
    artifact.mkdir()
    (artifact / "kernel.cpp").write_text(body, encoding="utf-8")
    work_dir = tmp_path / "w"
    seal_reference.seal(artifact, "demo", work_dir,
                        use_public_api_filter=False, kb_root=tmp_path / "kb")
    candidate = tmp_path / "cand"
    candidate.mkdir()
    (candidate / "mine.cpp").write_text(
        "// 这一段注释足够长，长到超过特征行的字符下限很多很多\n"
        "void g() { int b = 2; }\n", encoding="utf-8")
    assert seal_reference.check_tree(candidate, work_dir)["passed"] is True


def test_build_output_is_skipped(tmp_path):
    """构建产物里当然会有和参考相同的东西（同一套头文件生成），
    拿它们报搬运是纯误报。"""
    _a, work_dir, candidate = _sealed_pair(tmp_path, REF_BODY)
    (candidate / "build_out").mkdir()
    (candidate / "build_out" / "gen.cpp").write_text(REF_BODY,
                                                     encoding="utf-8")
    assert seal_reference.check_tree(candidate, work_dir)["passed"] is True


def test_third_party_is_not_skipped(tmp_path):
    """理由同 fence_check：被跳过的目录是漏过的那条路。"""
    _a, work_dir, candidate = _sealed_pair(tmp_path, REF_BODY)
    (candidate / "third_party").mkdir()
    (candidate / "third_party" / "shim.cpp").write_text(REF_BODY,
                                                        encoding="utf-8")
    assert seal_reference.check_tree(candidate, work_dir)["passed"] is False


def test_line_check_is_skipped_when_the_reference_is_gone(tmp_path):
    """参考可能已经被清掉。整文件摘要那一级不受影响（只用 manifest），
    而特征行那一级要读参考 —— 跳过时必须说出来，不能算成通过。"""
    import shutil  # noqa: PLC0415
    artifact, work_dir, candidate = _sealed_pair(
        tmp_path, REF_BODY)
    (candidate / "copied.cpp").write_text(REF_BODY, encoding="utf-8")
    shutil.rmtree(artifact)
    report = seal_reference.check_tree(candidate, work_dir)
    assert report["line_check"]["performed"] is False
    assert "跳过" in report["line_check"]["reason"]
    # 摘要那一级照样抓到
    assert "T-SAME-FILE" in {f["rule"] for f in report["findings"]}


def test_report_says_what_it_cannot_catch(tmp_path):
    """`findings` 为空不等于「没有搬运」—— 这句话要进报告，不能只写在文档里。"""
    _a, work_dir, candidate = _sealed_pair(tmp_path, REF_BODY)
    (candidate / "mine.cpp").write_text("void f() {}\n", encoding="utf-8")
    report = seal_reference.check_tree(candidate, work_dir)
    assert "不等于" in report["coverage_note"]


def test_check_tree_needs_a_manifest(tmp_path):
    candidate = tmp_path / "cand"
    candidate.mkdir()
    with pytest.raises(seal_reference.SealError, match="找不到封存清单"):
        seal_reference.check_tree(candidate, tmp_path / "nope")


# ---------------------------------------------------------------------------
# 封存校验的第三个方向：新增文件
#
# 原来只查「已删除」和「内容变更」。新增那个方向不是「记录对不上」这么轻 ——
# 新文件没进 manifest 就没进禁读清单，于是 hook **不会拦**对它的读取：
# 参考树上多出一个 agent 可以随便读的源文件，而所有「比对已记录的那些」的
# 校验全绿。
# ---------------------------------------------------------------------------

def _manifest_of(tmp_path, body="class KernelSecret {};\n"):
    artifact = tmp_path / "ref"
    artifact.mkdir()
    (artifact / "kernel.cpp").write_text(body, encoding="utf-8")
    work_dir = tmp_path / "w"
    seal_reference.seal(artifact, "demo", work_dir,
                        use_public_api_filter=False, kb_root=tmp_path / "kb")
    manifest = json.loads(
        (work_dir / "sealed_manifest.json").read_text(encoding="utf-8"))
    return artifact, work_dir, manifest


def test_unchanged_reference_verifies_clean(tmp_path):
    _a, _w, manifest = _manifest_of(tmp_path)
    assert seal_reference.verify_manifest(manifest) == []


def test_deleted_file_is_detected(tmp_path):
    artifact, _w, manifest = _manifest_of(tmp_path)
    (artifact / "kernel.cpp").unlink()
    drifted = seal_reference.verify_manifest(manifest)
    assert any(d["reason"] == "已删除" for d in drifted)


def test_modified_file_is_detected(tmp_path):
    artifact, _w, manifest = _manifest_of(tmp_path)
    (artifact / "kernel.cpp").write_text("class Other {};\n", encoding="utf-8")
    drifted = seal_reference.verify_manifest(manifest)
    assert any(d["reason"] == "内容已变更" for d in drifted)


def test_added_file_is_detected(tmp_path):
    """这是本轮补的方向。它在 opp_install 那类来源上很容易发生 ——
    那是一个活的安装目录，另一个任务重装一次算子包就会往里加文件。"""
    artifact, _w, manifest = _manifest_of(tmp_path)
    (artifact / "extra.cpp").write_text("class Sneaky {};\n", encoding="utf-8")
    drifted = seal_reference.verify_manifest(manifest)
    assert any("新增" in d["reason"] for d in drifted)


def test_added_file_message_says_the_hook_will_not_block_it(tmp_path):
    """报错要说清后果，否则这条会被当成洁癖。"""
    artifact, _w, manifest = _manifest_of(tmp_path)
    (artifact / "extra.cpp").write_text("class Sneaky {};\n", encoding="utf-8")
    drifted = seal_reference.verify_manifest(manifest)
    message = next(d["reason"] for d in drifted if "新增" in d["reason"])
    assert "hook 不会拦" in message


def test_added_non_source_file_is_ignored(tmp_path):
    """参考目录里多一个 .log / .so 不影响封闭性 —— 封的是可读的源与配置。"""
    artifact, _w, manifest = _manifest_of(tmp_path)
    (artifact / "build.log").write_text("ok\n", encoding="utf-8")
    assert seal_reference.verify_manifest(manifest) == []
