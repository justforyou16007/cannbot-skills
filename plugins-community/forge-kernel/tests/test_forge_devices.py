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
"""forge_devices.py 的单元测试。

多卡实验里最容易出的错不是崩溃，而是**噪声底在偏低的并发下量出**：
N 个 campaign 同时启动，第一个量噪声底时其余几个还没压上卡，量出来偏低；
之后满载运行，真实噪声更高，于是噪声里的「改进」被当成真的收下，
还会写进知识库和产物库污染后续各代。整个过程没有任何报错。

栅栏就是为这件事存在的，所以测试重点在它：少一个 peer 要能等出来并说清
后果，崩溃残留不能让它误判满员。

设备租约本身复用 ops-evaluation 的 device_lease，这里只验接得上、
接不上时报错能指出缺的是什么。

运行: python3 -m pytest plugins-community/forge-kernel/tests/ -v
"""

import json
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = (Path(__file__).resolve().parent.parent
           / "skills" / "forge-campaign" / "scripts")
sys.path.insert(0, str(SCRIPTS))

import forge_devices as fd  # noqa: E402


# ---------------------------------------------------------------------------
# 并发栅栏
# ---------------------------------------------------------------------------

def test_peer_registration_roundtrip(tmp_path):
    pool = tmp_path / "pool"
    fd.register_peer(pool, "s1", "add_layer_norm", 0, 4)
    peers = fd.list_peers(pool)
    assert len(peers) == 1
    assert peers[0]["operator"] == "add_layer_norm"
    assert peers[0]["device_id"] == 0


def test_unregister_removes_peer(tmp_path):
    pool = tmp_path / "pool"
    fd.register_peer(pool, "s1", "op", 0, 2)
    fd.unregister_peer(pool, "s1")
    assert fd.list_peers(pool) == []


def test_unregister_is_idempotent(tmp_path):
    fd.unregister_peer(tmp_path / "pool", "never_existed")


def test_list_peers_on_empty_pool(tmp_path):
    assert fd.list_peers(tmp_path / "nope") == []


def test_barrier_passes_when_all_bound(tmp_path):
    pool = tmp_path / "pool"
    for idx in range(3):
        fd.register_peer(pool, f"s{idx}", f"op{idx}", idx, 3)
    peers = fd.wait_for_peers(pool, 3, timeout=1.0, poll=0.05)
    assert len(peers) == 3


def test_barrier_times_out_when_short(tmp_path):
    """少一个 peer 就意味着噪声底会在偏低的并发下量出。"""
    pool = tmp_path / "pool"
    fd.register_peer(pool, "s1", "op1", 0, 3)
    fd.register_peer(pool, "s2", "op2", 1, 3)
    with pytest.raises(fd.DeviceError) as exc:
        fd.wait_for_peers(pool, 3, timeout=0.5, poll=0.05)
    assert "只等到 2 个" in str(exc.value)


def test_barrier_error_explains_the_consequence(tmp_path):
    """报错要说清为什么不能将就 —— 否则人会直接把 --expected 调小。"""
    pool = tmp_path / "pool"
    fd.register_peer(pool, "s1", "op1", 0, 2)
    with pytest.raises(fd.DeviceError) as exc:
        fd.wait_for_peers(pool, 2 + 1, timeout=0.3, poll=0.05)
    message = str(exc.value)
    assert "噪声底" in message
    assert "偏低" in message


def test_barrier_error_lists_who_is_bound(tmp_path):
    pool = tmp_path / "pool"
    fd.register_peer(pool, "s1", "add_layer_norm", 5, 2)
    with pytest.raises(fd.DeviceError) as exc:
        fd.wait_for_peers(pool, 2, timeout=0.3, poll=0.05)
    assert "add_layer_norm@dev5" in str(exc.value)


def test_barrier_accepts_more_than_expected(tmp_path):
    """实际起多了不该卡住 —— 噪声只会偏高，不会偏低。"""
    pool = tmp_path / "pool"
    for idx in range(4):
        fd.register_peer(pool, f"s{idx}", f"op{idx}", idx, 3)
    assert len(fd.wait_for_peers(pool, 3, timeout=0.5, poll=0.05)) == 4


# ---------------------------------------------------------------------------
# 崩溃残留
# ---------------------------------------------------------------------------

def test_stale_peer_is_cleaned(tmp_path):
    """崩溃进程留下的记录会让栅栏误判满员，必须清掉。"""
    pool = tmp_path / "pool"
    fd.register_peer(pool, "alive", "op", 0, 2)
    stale = fd.peers_dir(pool) / "dead.json"
    stale.write_text(json.dumps({
        "session_id": "dead", "operator": "ghost", "device_id": 9,
        "bound_ts": time.time() - fd.DEFAULT_STALE_AFTER - 10,
    }), encoding="utf-8")

    assert len(fd.list_peers(pool)) == 1
    assert not stale.exists(), "过期记录应被删除"


def test_fresh_peer_not_cleaned(tmp_path):
    pool = tmp_path / "pool"
    fd.register_peer(pool, "s1", "op", 0, 2)
    assert len(fd.list_peers(pool)) == 1
    assert len(fd.list_peers(pool)) == 1   # 再查一次仍在


def test_corrupt_peer_file_skipped(tmp_path):
    pool = tmp_path / "pool"
    fd.register_peer(pool, "good", "op", 0, 2)
    (fd.peers_dir(pool) / "bad.json").write_text("不是 JSON", encoding="utf-8")
    assert len(fd.list_peers(pool)) == 1


def test_stale_peer_does_not_satisfy_barrier(tmp_path):
    """这是最要命的一条：残留让栅栏误判满员，噪声底就在偏低并发下量了。"""
    pool = tmp_path / "pool"
    fd.register_peer(pool, "alive", "op", 0, 2)
    (fd.peers_dir(pool) / "dead.json").write_text(json.dumps({
        "session_id": "dead", "operator": "ghost", "device_id": 9,
        "bound_ts": time.time() - fd.DEFAULT_STALE_AFTER - 10,
    }), encoding="utf-8")
    with pytest.raises(fd.DeviceError):
        fd.wait_for_peers(pool, 2, timeout=0.3, poll=0.05)


# ---------------------------------------------------------------------------
# 跨 skill 导入
# ---------------------------------------------------------------------------

def test_device_lease_resolves_in_dev_layout():
    """开发布局下要找得到 ops-evaluation 的 device_lease。"""
    module = fd.load_device_lease()
    assert hasattr(module, "SessionDeviceLease")


def test_candidate_paths_cover_both_layouts():
    """安装后摊平、开发时分插件，两种布局都要认 —— 只写一种换个环境就挂。"""
    paths = [str(p) for p in fd._candidate_lease_paths()]
    assert any(p.endswith("skills/ops-evaluation/scripts") for p in paths)
    assert any("ops-perf-evolution" in p for p in paths)


def test_missing_lease_error_names_what_is_missing(monkeypatch):
    monkeypatch.setattr(fd, "_candidate_lease_paths",
                        lambda: [Path("/nonexistent/a"), Path("/nonexistent/b")])
    with pytest.raises(fd.DeviceError) as exc:
        fd.load_device_lease()
    message = str(exc.value)
    assert "ops-evaluation" in message
    assert "已查找" in message


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(argv):
    import contextlib
    import io
    saved = sys.argv
    sys.argv = ["forge_devices.py"] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                return fd.main(), buf.getvalue()
            except SystemExit as exc:
                return int(exc.code or 0), buf.getvalue()
    finally:
        sys.argv = saved


def test_cli_wait_peers_succeeds(tmp_path):
    pool = tmp_path / "pool"
    for idx in range(2):
        fd.register_peer(pool, f"s{idx}", f"op{idx}", idx, 2)
    code, out = run(["--pool", str(pool), "wait-peers", "--expected", "2",
                     "--timeout", "1"])
    assert code == 0
    assert "已全部绑定" in out


def test_cli_wait_peers_reports_shortfall(tmp_path):
    pool = tmp_path / "pool"
    fd.register_peer(pool, "s1", "op1", 0, 2)
    code, out = run(["--pool", str(pool), "wait-peers", "--expected", "2",
                     "--timeout", "0.3"])
    assert code == 1
    assert "噪声底" in out


def test_cli_status_lists_peers(tmp_path):
    pool = tmp_path / "pool"
    fd.register_peer(pool, "s1", "add_layer_norm", 3, 2)
    code, out = run(["--pool", str(pool), "status"])
    assert code == 0
    assert "add_layer_norm" in out
    assert "device 3" in out


def test_cli_run_requires_a_command(tmp_path):
    code, out = run(["--pool", str(tmp_path), "run"])
    assert code == 2
    assert "-- 之后" in out


def test_pool_accepted_in_both_positions():
    """与 campaign.py 同一套约定：全局参数写在子命令前后都能跑。"""
    parser = fd.build_parser()
    before = parser.parse_args(["--pool", "a", "status"])
    after = parser.parse_args(["status", "--pool", "a"])
    assert before.pool == "a"
    assert after.pool == "a"
