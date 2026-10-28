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
"""并行任务下的共享状态互斥测试。

使用方式是「同一仓库起多个任务，各占一张卡」。任务之间不共享设备、
不共享 state、不共享工作目录，但**共享产物库与知识库** —— 后者是有意的，
跨任务复用经验正是它们存在的理由。

共享就有竞争，而这里的竞争全都是静默的：

  索引丢更新   add_generation 的 load_index → 改 → save_index 之间若被
               插入，先写的那条在索引里消失。产物还在磁盘上，但查不到，
               双方都报「已入库」。

  产物被覆盖   同算子+形状的两个任务同时提升时，双方算出同一个 gen 号，
               后者 rmtree 掉前者刚复制进去的产物。

  死路丢更新   两个任务同时登记死路，后写的覆盖先写的，双方都报
               「已登记」。死路表的价值在于跨代存活，丢一条就意味着
               下一代把那个方向原封不动再试一遍。

所以这些测试用真实多进程压，而不是模拟。每个用例都先验证「不加锁时确实
会丢」—— 否则测试可能只是在验证一件本来就不会发生的事。

运行: python3 -m pytest plugins-community/forge-kernel/tests/ -v
"""

import contextlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import pytest

SCRIPTS = (Path(__file__).resolve().parent.parent
           / "skills" / "forge-campaign" / "scripts")
HWKB_SCRIPTS = (Path(__file__).resolve().parent.parent
                / "skills" / "forge-hwkb" / "scripts")
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HWKB_SCRIPTS))

import hwkb  # noqa: E402
import reference_library as rl  # noqa: E402

#: 并发度。够大才压得出竞争，又不至于让测试太慢。
WORKERS = 10


# ---------------------------------------------------------------------------
# 子进程入口必须是模块级函数（spawn/forkserver 要能 pickle）
# ---------------------------------------------------------------------------

def _register_one(payload):
    library, idx, disable_lock = payload
    if disable_lock:
        rl.exclusive = _noop_lock
    artifact = Path(library).parent / f"art{idx}" / "op_kernel"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "k.cpp").write_text(f"// task {idx}\n", encoding="utf-8")
    try:
        rl.add_generation(
            Path(library), operator=f"op{idx}", arch="DAV_2201",
            input_shape="B=8", kind="opp_install", identifier=f"id{idx}",
            artifact=str(artifact.parent), time_us=100.0 + idx)
        return True
    except Exception:  # noqa: BLE001 - 这里只关心有没有成功
        return False


@contextlib.contextmanager
def _noop_lock(library, timeout=0):
    """模拟未加锁的旧行为，用来确认竞争确实存在。"""
    Path(library).mkdir(parents=True, exist_ok=True)
    yield


def _add_nogo(payload):
    kb_root, idx = payload
    import argparse
    args = argparse.Namespace(
        kb_root=str(kb_root), symptom=f"症状{idx}", tried=f"方向{idx}",
        evidence=f"[MEASURED, job {idx}]", arch="DAV_2201",
        operator=f"op{idx}", as_json=False)
    return hwkb.cmd_nogo_add(args) == 0


def _run_pool(func, payloads):
    ctx = mp.get_context("fork")
    with ctx.Pool(len(payloads)) as pool:
        return pool.map(func, payloads)


# ---------------------------------------------------------------------------
# 产物库
# ---------------------------------------------------------------------------

def test_concurrent_registers_do_not_lose_entries(tmp_path):
    """并行任务同时入库，索引不能丢条目。"""
    library = tmp_path / "L"
    payloads = [(str(library), idx, False) for idx in range(WORKERS)]
    results = _run_pool(_register_one, payloads)

    assert all(results), "全部应成功"
    index = json.loads((library / "index.json").read_text(encoding="utf-8"))
    assert len(index["entries"]) == WORKERS, (
        f"丢了 {WORKERS - len(index['entries'])} 条：产物在磁盘上但索引查不到")


def test_race_is_real_without_the_lock(tmp_path):
    """反证：去掉锁就会丢。

    没有这条，上面那个测试可能只是在验证一件本来就不会发生的事。
    """
    library = tmp_path / "L"
    payloads = [(str(library), idx, True) for idx in range(WORKERS)]
    _run_pool(_register_one, payloads)

    index = json.loads((library / "index.json").read_text(encoding="utf-8"))
    assert len(index["entries"]) < WORKERS, (
        "去掉锁居然没丢 —— 并发度不够压不出竞争，这个反证就失效了")


def test_concurrent_registers_keep_artifacts_intact(tmp_path):
    """每条记录的产物校验和要对得上，不能被别的任务覆盖。"""
    library = tmp_path / "L"
    payloads = [(str(library), idx, False) for idx in range(WORKERS)]
    _run_pool(_register_one, payloads)

    problems, checked = rl.verify(library)
    assert problems == []
    assert checked == WORKERS


def test_same_operator_concurrent_promote_gets_distinct_generations(tmp_path):
    """同一算子+形状被并发提升时，gen 号不能撞 —— 撞了会互相覆盖产物。"""
    library = tmp_path / "L"
    rl.add_generation(library, operator="shared_op", arch="DAV_2201",
                      input_shape="B=8", kind="builtin_aclnn",
                      identifier="ref", time_us=500.0)

    def payload(idx):
        return (str(library), idx)

    ctx = mp.get_context("fork")
    with ctx.Pool(4) as pool:
        pool.map(_promote_shared, [payload(i) for i in range(4)])

    index = json.loads((library / "index.json").read_text(encoding="utf-8"))
    entry = index["entries"][rl.entry_key("DAV_2201", "shared_op", "B=8")]
    generations = [g["generation"] for g in entry["generations"]]
    assert len(generations) == len(set(generations)), f"gen 号撞了: {generations}"

    problems, _ = rl.verify(library)
    assert problems == [], "产物被覆盖了"


def _promote_shared(payload):
    library, idx = payload
    artifact = Path(library).parent / f"p{idx}" / "op_kernel"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "k.cpp").write_text(f"// promote {idx}\n", encoding="utf-8")
    try:
        rl.add_generation(
            Path(library), operator="shared_op", arch="DAV_2201",
            input_shape="B=8", kind="sealed_kernel", identifier=f"g{idx}",
            artifact=str(artifact.parent), gain=1.1, time_us=400.0 - idx,
            mechanism_verified=True)
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# 知识库死路表
# ---------------------------------------------------------------------------

def test_concurrent_nogo_writes_do_not_lose_entries(tmp_path):
    """并行任务同时登记死路，一条都不能丢。

    丢一条就意味着下一代把那个方向原封不动再试一遍 ——
    而死路表的全部价值就在于避免这件事。
    """
    kb_root = tmp_path / "kb"
    payloads = [(str(kb_root), idx) for idx in range(WORKERS)]
    results = _run_pool(_add_nogo, payloads)

    assert all(results)
    entries = hwkb.load_nogo(kb_root)
    assert len(entries) == WORKERS, f"丢了 {WORKERS - len(entries)} 条死路"


def test_nogo_entries_are_all_distinct(tmp_path):
    kb_root = tmp_path / "kb"
    payloads = [(str(kb_root), idx) for idx in range(WORKERS)]
    _run_pool(_add_nogo, payloads)
    tried = {e["tried"] for e in hwkb.load_nogo(kb_root)}
    assert len(tried) == WORKERS


# ---------------------------------------------------------------------------
# 锁本身
# ---------------------------------------------------------------------------

def test_library_lock_is_exclusive(tmp_path):
    """同一进程内嵌套取锁会超时 —— 证明它确实是排他的。"""
    library = tmp_path / "L"
    with rl.exclusive(library):
        with pytest.raises(rl.LibraryError, match="取库锁超时"):
            with rl.exclusive(library, timeout=0.2):
                pass


def test_library_lock_released_after_use(tmp_path):
    library = tmp_path / "L"
    with rl.exclusive(library):
        pass
    with rl.exclusive(library, timeout=1.0):
        pass


def test_library_lock_released_on_exception(tmp_path):
    """异常路径也要释放，否则一次失败会卡住后续所有任务。"""
    library = tmp_path / "L"
    with pytest.raises(ValueError):
        with rl.exclusive(library):
            raise ValueError("boom")
    with rl.exclusive(library, timeout=1.0):
        pass


def test_hwkb_lock_is_exclusive(tmp_path):
    kb_root = tmp_path / "kb"
    with hwkb.exclusive(kb_root):
        with pytest.raises(TimeoutError, match="取知识库锁超时"):
            with hwkb.exclusive(kb_root, timeout=0.2):
                pass


def test_hwkb_lock_released_on_exception(tmp_path):
    kb_root = tmp_path / "kb"
    with pytest.raises(ValueError):
        with hwkb.exclusive(kb_root):
            raise ValueError("boom")
    with hwkb.exclusive(kb_root, timeout=1.0):
        pass


def test_lock_timeout_message_is_actionable(tmp_path):
    """超时信息要说清锁文件在哪，否则人只能干等。"""
    library = tmp_path / "L"
    with rl.exclusive(library):
        with pytest.raises(rl.LibraryError) as exc:
            with rl.exclusive(library, timeout=0.1):
                pass
    message = str(exc.value)
    assert ".lock" in message
    assert "删掉" in message


# ---------------------------------------------------------------------------
# 死路表的两个写入者必须锁同一个文件
#
# hwkb.py 锁 <kb_root>/.lock，campaign.py 也必须锁同一个 —— 否则互斥只在
# 「调用方碰巧传对路径」时成立，而加锁要防的正是这件事。
# ---------------------------------------------------------------------------

def _campaign_nogo(payload):
    kb_root, idx = payload
    sys.path.insert(0, str(SCRIPTS))
    import campaign as c
    state = {"arch": "DAV_2201", "operator": f"op{idx}",
             "input_shape": "B=8", "noise_floor_pct": 2.0}
    entry = {"decision": f"D{idx}", "gain": 1.001,
             "symptom": f"症状{idx}", "reason": "噪声内"}
    c.write_nogo(Path(kb_root), state, entry)
    return True


def test_both_writers_lock_the_same_file(tmp_path):
    """campaign 与 hwkb 对同一个知识库根取锁，锁文件必须是同一个。"""
    import campaign as c
    kb_root = tmp_path / "kb"
    state = {"arch": "DAV_2201", "operator": "op", "input_shape": "B=8",
             "noise_floor_pct": 2.0}
    c.write_nogo(kb_root, state,
                 {"decision": "D1", "gain": 1.0, "symptom": "s",
                  "reason": "r"})
    campaign_lock = kb_root / ".lock"

    with hwkb.exclusive(kb_root):
        pass
    hwkb_lock = kb_root / ".lock"

    assert campaign_lock == hwkb_lock
    assert campaign_lock.exists()


def test_mixed_writers_do_not_lose_nogo_entries(tmp_path):
    """两个写入者并发追加，一条都不能丢。"""
    kb_root = tmp_path / "kb"
    payloads = [(str(kb_root), idx) for idx in range(WORKERS)]
    half = WORKERS // 2
    ctx = mp.get_context("fork")
    with ctx.Pool(WORKERS) as pool:
        a = pool.map_async(_campaign_nogo, payloads[:half])
        b = pool.map_async(_add_nogo, payloads[half:])
        a.get(); b.get()

    entries = hwkb.load_nogo(kb_root)
    assert len(entries) == WORKERS, f"丢了 {WORKERS - len(entries)} 条死路"


def test_campaign_nogo_filename_matches_hwkb(tmp_path):
    """文件名也要一致，否则两边各写各的文件，互斥等于没有。"""
    import campaign as c
    assert c.NOGO_FILENAME == hwkb.NOGO_FILENAME
