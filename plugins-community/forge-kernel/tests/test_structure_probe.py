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
"""structure_probe.py 的单元测试。

这个探针要挡住的是一种很具体的错判：停顿指标**能被别的改动顶下去**。
减小 tile、改对齐都会让 mte2_wait 降，于是一个单缓冲回退的实现可以凭一个
恰好变好的停顿数拿到「机制已核实」，而双缓冲从未存在。

所以测试的重心是三件事：

  退化能被抓住   声称双缓冲而实际单缓冲、同步少一侧，都要判不成立
  注释不算证据   注释里写「这里是 double buffer」不能让断言通过
  查不了≠通过    工具链不在时报错而不是放行

运行: python3 -m pytest plugins-community/forge-kernel/tests/ -v
"""

import sys
from pathlib import Path

import pytest

SCRIPTS = (Path(__file__).resolve().parent.parent
           / "skills" / "forge-probes" / "scripts")
sys.path.insert(0, str(SCRIPTS))

import structure_probe as sp  # noqa: E402


def candidate(tmp_path, body: str, name: str = "k.cpp") -> Path:
    root = tmp_path / "cand"
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(body, encoding="utf-8")
    return root


DOUBLE_BUFFERED = """
#include "kernel_operator.h"
class KernelMine {
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueue;
    void Init() { pipe.InitBuffer(inQueue, 2, TILE_BYTES); }
    void Process() {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    }
};
"""

SINGLE_BUFFERED = """
#include "kernel_operator.h"
class KernelMine {
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue;
    void Init() { pipe.InitBuffer(inQueue, 1, TILE_BYTES); }
    void Process() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0); }
};
"""


# ---------------------------------------------------------------------------
# 退化要被抓住
# ---------------------------------------------------------------------------

def test_single_buffer_fallback_is_caught(tmp_path):
    """「声明了多缓冲、实际是单缓冲回退」—— 停顿指标抓不到这个。"""
    root = candidate(tmp_path, SINGLE_BUFFERED)
    report = sp.probe(root, ["buffer_depth:2"], "dav-c220-vec", [],
                      tmp_path / "w")
    assert report["passed"] is False
    assert report["assertions"][0]["verdict"] == "fail"
    assert "单缓冲回退" in report["assertions"][0]["detail"]


def test_double_buffer_passes(tmp_path):
    root = candidate(tmp_path, DOUBLE_BUFFERED)
    report = sp.probe(root, ["buffer_depth:2"], "dav-c220-vec", [],
                      tmp_path / "w")
    assert report["passed"] is True


def test_unpaired_sync_is_caught(tmp_path):
    """少一侧说明同步链断了 —— 这个 case 可能恰好不出错，但不是设计要的结构。"""
    root = candidate(tmp_path, SINGLE_BUFFERED)
    report = sp.probe(root, ["flag_paired"], "dav-c220-vec", [], tmp_path / "w")
    assert report["assertions"][0]["verdict"] == "fail"
    assert report["assertions"][0]["observed"] == {"set_flag": 1,
                                                   "wait_flag": 0}


def test_no_sync_at_all_is_caught(tmp_path):
    root = candidate(tmp_path, "void f() { int x = 1; }")
    report = sp.probe(root, ["flag_paired"], "dav-c220-vec", [], tmp_path / "w")
    assert report["assertions"][0]["verdict"] == "fail"


def test_paired_sync_passes(tmp_path):
    root = candidate(tmp_path, DOUBLE_BUFFERED)
    report = sp.probe(root, ["flag_paired"], "dav-c220-vec", [], tmp_path / "w")
    assert report["assertions"][0]["verdict"] == "pass"


# ---------------------------------------------------------------------------
# 注释不算证据
# ---------------------------------------------------------------------------

def test_comment_claiming_the_structure_is_not_evidence(tmp_path):
    """注释里写「这里是 double buffer」不能让断言通过。"""
    root = candidate(tmp_path, """
// 这里用了 double buffer，深度 2
/* InitBuffer(inQueue, 2, TILE) */
class KernelMine {
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue;
};
""")
    report = sp.probe(root, ["buffer_depth:2"], "dav-c220-vec", [],
                      tmp_path / "w")
    assert report["assertions"][0]["verdict"] == "fail"


def test_block_comment_is_stripped(tmp_path):
    root = candidate(tmp_path, """
/* SetFlag<X>(0); WaitFlag<X>(0); */
void f() { int x = 1; }
""")
    report = sp.probe(root, ["flag_paired"], "dav-c220-vec", [], tmp_path / "w")
    assert report["assertions"][0]["verdict"] == "fail"


# ---------------------------------------------------------------------------
# 「查不了」不等于「通过」
# ---------------------------------------------------------------------------

def test_missing_toolchain_is_an_error_not_a_pass(tmp_path, monkeypatch):
    """IR 层断言拿不到工具链时报错。

    一条查不了的结构断言被当成通过，比没有这条断言更糟：它会让人以为
    核验过了。
    """
    monkeypatch.setattr(sp.shutil, "which", lambda _name: None)
    root = candidate(tmp_path, DOUBLE_BUFFERED)
    with pytest.raises(sp.StructureError, match="不等于"):
        sp.probe(root, ["operand_space:addrspace(6)"], "dav-c220-vec", [],
                 tmp_path / "w")


def test_missing_toolchain_can_be_accepted_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(sp.shutil, "which", lambda _name: None)
    root = candidate(tmp_path, DOUBLE_BUFFERED)
    report = sp.probe(root, ["operand_space:addrspace(6)"], "dav-c220-vec",
                      [], tmp_path / "w", allow_missing_toolchain=True)
    assert report["assertions"][0]["verdict"] == "unknown"
    # 查不了也不算通过
    assert report["passed"] is False


def test_unknown_is_counted_separately_from_failed(tmp_path, monkeypatch):
    """「有不成立的」与「有查不了的」要分开 —— 两者该做的事完全不同。"""
    monkeypatch.setattr(sp.shutil, "which", lambda _name: None)
    root = candidate(tmp_path, SINGLE_BUFFERED)
    report = sp.probe(root, ["buffer_depth:2", "operand_space:addrspace(6)"],
                      "dav-c220-vec", [], tmp_path / "w",
                      allow_missing_toolchain=True)
    assert report["failed_count"] == 1
    assert report["unknown_count"] == 1


def test_undeclared_depth_is_unknown_not_pass(tmp_path):
    """换了写法时查不到深度声明 —— 不要当成通过。"""
    root = candidate(tmp_path, "void f() { do_something(); }")
    report = sp.probe(root, ["buffer_depth:2"], "dav-c220-vec", [],
                      tmp_path / "w")
    assert report["assertions"][0]["verdict"] == "unknown"
    assert report["passed"] is False


# ---------------------------------------------------------------------------
# 边界
# ---------------------------------------------------------------------------

def test_no_sources_is_an_error(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(sp.StructureError, match="没有可扫的源文件"):
        sp.probe(root, ["flag_paired"], "dav-c220-vec", [], tmp_path / "w")


def test_build_dirs_are_skipped(tmp_path):
    """构建产物里的匹配说明不了候选自己的结构。"""
    root = candidate(tmp_path, SINGLE_BUFFERED)
    stale = root / "build"
    stale.mkdir()
    (stale / "old.cpp").write_text(DOUBLE_BUFFERED, encoding="utf-8")
    report = sp.probe(root, ["buffer_depth:2"], "dav-c220-vec", [],
                      tmp_path / "w")
    assert report["assertions"][0]["verdict"] == "fail", \
        "build/ 下的旧代码不该让断言通过"


def test_third_party_is_not_skipped(tmp_path):
    """third_party 不在跳过列表里 —— 从被跳过的目录 import 一份转发实现
    是围栏漏过的那条路，结构核验不该重复同一个错。"""
    assert "third_party" not in sp.SKIP_DIRS


def test_symbol_assertions(tmp_path):
    root = candidate(tmp_path, DOUBLE_BUFFERED)
    report = sp.probe(root, ["has:TQue", "absent:NotThere"],
                      "dav-c220-vec", [], tmp_path / "w")
    assert all(a["verdict"] == "pass" for a in report["assertions"])


def test_symbol_match_uses_word_boundaries(tmp_path):
    root = candidate(tmp_path, "void f() { TQueBindExtra x; }")
    report = sp.probe(root, ["has:TQueBind"], "dav-c220-vec", [],
                      tmp_path / "w")
    assert report["assertions"][0]["verdict"] == "fail"


def test_unknown_assertion_shape_is_rejected(tmp_path):
    root = candidate(tmp_path, DOUBLE_BUFFERED)
    with pytest.raises(sp.StructureError, match="不认识的断言"):
        sp.probe(root, ["nonsense:1"], "dav-c220-vec", [], tmp_path / "w")


def test_addrspace_names_are_documented():
    assert sp.ADDRSPACE_NAMES["1"] == "GM"
    assert sp.ADDRSPACE_NAMES["6"] == "UB"
