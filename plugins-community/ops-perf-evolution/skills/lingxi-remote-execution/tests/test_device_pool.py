#!/usr/bin/env python3
"""设备池测试：跨进程原子分配、sticky lease、心跳/TTL 回收、显式释放。"""

import json
import subprocess
import sys
import time
from pathlib import Path

from conftest import (WORKER, acquire_lease, run_worker, write_pool_config,
                      cmd_stage, simple_spec, submit_job, wait_terminal)


def test_concurrent_acquire_is_atomic(worker_root):
    """8 个进程抢 4 张卡：恰好 4 个成功，且设备互不重叠。"""
    procs = []
    for i in range(8):
        procs.append(subprocess.Popen(
            [sys.executable, str(WORKER), "--root", str(worker_root),
             "lease-acquire", "--session", "s%d" % i, "--owner", "op%d" % i,
             "--ttl", "600", "--timeout", "2"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    devices, rcs = [], []
    for p in procs:
        out, _ = p.communicate(timeout=30)
        payload = json.loads(out.decode().strip().splitlines()[-1])
        rcs.append(p.returncode)
        if p.returncode == 0:
            devices.append(payload["lease"]["device_id"])
    assert len([r for r in rcs if r == 0]) == 4, rcs
    assert len([r for r in rcs if r == 5]) == 4, rcs  # 其余等待超时
    assert sorted(devices) == [0, 1, 2, 3], devices   # 无重复分配


def test_sticky_lease_reuses_same_device(worker_root):
    """同一 session 重复 acquire 必须复用同一张卡（基线到复测绑卡不变）。"""
    first = acquire_lease(worker_root, "sess-A")
    second = acquire_lease(worker_root, "sess-A")
    assert second["device_id"] == first["device_id"]
    assert second["lease_id"] == first["lease_id"]


def test_concurrent_sticky_acquire_creates_only_one_lease(worker_root):
    """Concurrent acquires for one session must all return the same lease/card."""
    procs = [subprocess.Popen(
        [sys.executable, str(WORKER), "--root", str(worker_root),
         "lease-acquire", "--session", "same-session", "--owner", "same-op",
         "--ttl", "600", "--timeout", "10"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(8)]
    leases = []
    for proc in procs:
        out, err = proc.communicate(timeout=30)
        assert proc.returncode == 0, err.decode()
        leases.append(json.loads(out.decode().splitlines()[-1])["lease"])
    assert len({lease["lease_id"] for lease in leases}) == 1
    assert len({lease["device_id"] for lease in leases}) == 1
    rc, payload, _ = run_worker(worker_root, "pool-list")
    valid = [item for item in payload["pool"] if item["valid"]]
    assert len(valid) == 1


def test_no_sticky_allocates_preferred_free_device(worker_root):
    """--no-sticky 时可按 --device 指定空闲卡（仍受全局池约束）。"""
    acquire_lease(worker_root, "sess-A", device=0)
    rc, payload, _ = run_worker(
        worker_root, "lease-acquire", "--session", "sess-A", "--owner", "sess-A",
        "--ttl", "600", "--timeout", "2", "--device", "2", "--no-sticky")
    assert rc == 0, payload
    assert payload["lease"]["device_id"] == 2


def test_ttl_expiry_reclaims_device(worker_root):
    """心跳超时后租约过期，其他 session 可以拿到同一张卡。"""
    lease = acquire_lease(worker_root, "sess-A", ttl=600)
    # 人为把心跳拨回到 TTL 之前（模拟长时间无心跳）
    lease_path = worker_root / "pool" / ("device_%d.lease.json" % lease["device_id"])
    meta = json.loads(lease_path.read_text())
    meta["heartbeat_ts"] = time.time() - meta["ttl_s"] - 10
    lease_path.write_text(json.dumps(meta))
    rc, payload, _ = run_worker(worker_root, "pool-list")
    entry = [e for e in payload["pool"] if e["device_id"] == lease["device_id"]][0]
    assert entry["valid"] is False
    other = acquire_lease(worker_root, "sess-B")
    assert other["device_id"] == lease["device_id"]


def test_heartbeat_extends_lease(worker_root):
    lease = acquire_lease(worker_root, "sess-A", ttl=600)
    lease_path = worker_root / "pool" / ("device_%d.lease.json" % lease["device_id"])
    meta = json.loads(lease_path.read_text())
    meta["heartbeat_ts"] = time.time() - 599  # 马上要过期
    lease_path.write_text(json.dumps(meta))
    rc, payload, _ = run_worker(worker_root, "lease-heartbeat",
                                "--lease-id", lease["lease_id"])
    assert rc == 0
    refreshed = json.loads(lease_path.read_text())
    assert refreshed["heartbeat_ts"] > meta["heartbeat_ts"]


def test_release_frees_device(worker_root):
    lease = acquire_lease(worker_root, "sess-A")
    rc, payload, _ = run_worker(worker_root, "lease-release",
                                "--lease-id", lease["lease_id"])
    assert rc == 0
    other = acquire_lease(worker_root, "sess-B", timeout=2)
    assert other["device_id"] == lease["device_id"]


def test_runner_refreshes_lease_during_device_stage(worker_root):
    """远端 job 运行期间替 session 续租：本地断线不导致卡被 TTL 回收。"""
    lease = acquire_lease(worker_root, "sess-A", ttl=600)
    lease_path = worker_root / "pool" / ("device_%d.lease.json" % lease["device_id"])
    # 两个设备阶段：verify(慢) -> profile(快)。runner 在每个设备阶段开始前续租。
    spec = simple_spec("job-refresh", "sess-A", [
        cmd_stage("verify", ["sleep", "1"], needs_device=True,
                  failure_class="correctness"),
        cmd_stage("profile", ["true"], needs_device=True, failure_class="profile"),
    ])
    submit_job(worker_root, spec)
    # 等 verify 阶段进入 running（此时第一次续租已完成或即将完成）
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        rc, payload, _ = run_worker(worker_root, "job-status", "--job-id", "job-refresh")
        stages = {s["name"]: s["status"] for s in payload["state"]["stages"]}
        if stages.get("verify") == "running":
            break
        time.sleep(0.1)
    time.sleep(0.3)  # 留出第一次续租的时间
    # 回拨心跳：若无续租，剩余 2s 内会过期（verify 还要跑 ~1s，profile 的续租
    # 必然发生在回拨之后 -> 只要 profile 阶段成功就证明续租救回了租约）
    meta = json.loads(lease_path.read_text())
    meta["heartbeat_ts"] = time.time() - meta["ttl_s"] + 2
    lease_path.write_text(json.dumps(meta))
    state = wait_terminal(worker_root, "job-refresh", timeout=30)
    assert state["status"] == "succeeded", state
    profile_stage = [s for s in state["stages"] if s["name"] == "profile"][0]
    assert profile_stage["status"] == "ok"  # 租约若未被续上这里会 lease_lost
    refreshed = json.loads(lease_path.read_text())
    assert refreshed["heartbeat_ts"] > meta["heartbeat_ts"]


def test_long_device_stage_keeps_short_lease(worker_root):
    """A stage longer than TTL must keep the card; another session cannot overlap."""
    write_pool_config(worker_root, devices=(0,), max_build_slots=1)
    lease = acquire_lease(worker_root, "long-s", ttl=1.0)
    submit_job(worker_root, simple_spec(
        "long-job", "long-s",
        [cmd_stage("verify", ["sleep", "2.5"], needs_device=True,
                   failure_class="correctness", timeout_s=10)]))
    time.sleep(1.3)
    rc, payload, _ = run_worker(
        worker_root, "lease-acquire", "--session", "other-s", "--owner", "other",
        "--ttl", "10", "--timeout", "0.4")
    assert rc == 5, payload
    state = wait_terminal(worker_root, "long-job", timeout=15)
    assert state["status"] == "succeeded", state
    current = json.loads((worker_root / "pool" /
                          ("device_%d.lease.json" % lease["device_id"])).read_text())
    assert current["lease_id"] == lease["lease_id"]
