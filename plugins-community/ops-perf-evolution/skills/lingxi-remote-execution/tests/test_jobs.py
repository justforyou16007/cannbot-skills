#!/usr/bin/env python3
"""job 生命周期测试：同卡串行/异卡并行、build slot、幂等、取消（防 PID 复用）、
失败分类、状态恢复、私有 PYTHONPATH（pip --target，无全局安装）。"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from conftest import (acquire_lease, cmd_stage, make_minimal_wheel,
                      run_worker, simple_spec, stage_source, submit_job, wait_terminal)


MARKER_STAGE = (
    "import os, time;"
    "f=open('marker_{stage}.txt','w');"
    "f.write(os.environ.get('ASCEND_RT_VISIBLE_DEVICES','none'));"
    "f.write('\\n%d\\n%d' % (time.time()*1000, 0));"
    "f.close()"
)
# 写 [device, start_ms] / [device, end_ms]，用于同卡串行 / 异卡并行判定
TIMING_STAGE = (
    "import os, time, json;"
    "d=os.environ.get('ASCEND_RT_VISIBLE_DEVICES','none');"
    "t=time.time();"
    "time.sleep(0.8);"
    "open('timing_{stage}.txt','w').write(json.dumps([d, t, time.time()]))"
)


def test_job_success_stage_env_and_device(worker_root):
    """设备阶段的 ASCEND_RT_VISIBLE_DEVICES 来自 session lease，非设备阶段不设。"""
    lease = acquire_lease(worker_root, "sess-A")
    spec = simple_spec("job-ok", "sess-A", [
        cmd_stage("build", ["python3", "-c", MARKER_STAGE.format(stage="build")]),
        cmd_stage("verify",
                  ["python3", "-c", MARKER_STAGE.format(stage="verify")],
                  needs_device=True, failure_class="correctness"),
        cmd_stage("profile",
                  ["python3", "-c", MARKER_STAGE.format(stage="profile")],
                  needs_device=True, failure_class="profile"),
    ])
    submit_job(worker_root, spec)
    state = wait_terminal(worker_root, "job-ok")
    assert state["status"] == "succeeded", state
    assert state["device_id"] == lease["device_id"]
    job_dir = worker_root / "jobs" / "job-ok"
    assert (job_dir / "marker_build.txt").read_text().splitlines()[0] == "none"
    for stage in ("verify", "profile"):
        dev = (job_dir / ("marker_%s.txt" % stage)).read_text().splitlines()[0]
        assert dev == str(lease["device_id"]), (stage, dev)
    assert all(s["status"] == "ok" for s in state["stages"])
    # 产物清单包含声明的 glob
    assert "marker_verify.txt" in state["artifacts_manifest"]["files"]


def test_same_device_stages_serialize_across_jobs(worker_root):
    """同一 session 的两个 job：设备阶段在同一张卡上必须串行（不重叠）。"""
    acquire_lease(worker_root, "sess-A")
    timings = []
    for i in range(2):
        spec = simple_spec("job-ser-%d" % i, "sess-A", [
            cmd_stage("verify",
                      ["python3", "-c", TIMING_STAGE.format(stage="ser%d" % i)],
                      needs_device=True, failure_class="correctness"),
        ], artifacts=["timing_ser%d.txt" % i])
        submit_job(worker_root, spec)
        timings.append("timing_ser%d.txt" % i)
    for i in range(2):
        state = wait_terminal(worker_root, "job-ser-%d" % i)
        assert state["status"] == "succeeded", state
    job0 = json.loads((worker_root / "jobs" / "job-ser-0" / timings[0]).read_text())
    job1 = json.loads((worker_root / "jobs" / "job-ser-1" / timings[1]).read_text())
    # 时间区间不重叠（串行）；同卡
    assert job0[0] == job1[0]
    overlaps = job0[1] < job1[2] and job1[1] < job0[2]
    assert not overlaps, (job0, job1)


def test_different_devices_run_in_parallel(worker_root):
    """不同 session 绑不同卡：设备阶段必须重叠执行。"""
    lease_a = acquire_lease(worker_root, "sess-A", device=0)
    lease_b = acquire_lease(worker_root, "sess-B", device=1)
    assert lease_a["device_id"] != lease_b["device_id"]
    for sid, tag in (("sess-A", "parA"), ("sess-B", "parB")):
        spec = simple_spec("job-%s" % tag, sid, [
            cmd_stage("verify",
                      ["python3", "-c", TIMING_STAGE.format(stage=tag)],
                      needs_device=True, failure_class="correctness"),
        ], artifacts=["timing_%s.txt" % tag])
        submit_job(worker_root, spec)
    for tag in ("parA", "parB"):
        state = wait_terminal(worker_root, "job-%s" % tag)
        assert state["status"] == "succeeded", state
    a = json.loads((worker_root / "jobs" / "job-parA" / "timing_parA.txt").read_text())
    b = json.loads((worker_root / "jobs" / "job-parB" / "timing_parB.txt").read_text())
    assert a[0] != b[0]  # 两张不同的卡
    overlaps = a[1] < b[2] and b[1] < a[2]
    assert overlaps, (a, b)


def test_build_slot_semaphores_non_device_stages(worker_root, tmp_path):
    """max_build_slots=1 时，两个 job 的构建阶段互斥（不占设备锁但受限并发）。"""
    import conftest
    conftest.write_pool_config(worker_root, devices=(0, 1), max_build_slots=1)
    acquire_lease(worker_root, "sess-A")
    acquire_lease(worker_root, "sess-B", device=1)
    for sid, tag in (("sess-A", "slotA"), ("sess-B", "slotB")):
        spec = simple_spec("job-%s" % tag, sid, [
            cmd_stage("build", ["python3", "-c",
                                "import time,json;"
                                "t=time.time();time.sleep(0.6);"
                                "open('b_%s.txt','w').write(json.dumps([t,time.time()]))" % tag]),
        ], artifacts=["b_%s.txt" % tag])
        submit_job(worker_root, spec)
    for tag in ("slotA", "slotB"):
        state = wait_terminal(worker_root, "job-%s" % tag)
        assert state["status"] == "succeeded", state
    a = json.loads((worker_root / "jobs" / "job-slotA" / "b_slotA.txt").read_text())
    b = json.loads((worker_root / "jobs" / "job-slotB" / "b_slotB.txt").read_text())
    overlaps = a[0] < b[1] and b[0] < a[1]
    assert not overlaps, (a, b)


def test_submit_is_idempotent(worker_root):
    acquire_lease(worker_root, "sess-A")
    spec = simple_spec("job-idem", "sess-A", [
        cmd_stage("build", ["true"]),
    ])
    rc1, p1 = submit_job(worker_root, spec)
    rc2, p2 = submit_job(worker_root, spec)
    assert rc1 == 0 and rc2 == 0
    assert p1["idempotent"] is False
    assert p2["idempotent"] is True
    assert p2["job_id"] == "job-idem"
    state = wait_terminal(worker_root, "job-idem")
    assert state["status"] == "succeeded"


def test_concurrent_submit_spawns_exactly_one_runner(worker_root):
    """Concurrent identical submit calls must not start duplicate detached runners."""
    code = "open('runner_count.txt','a').write('run\\n')"
    spec = simple_spec("job-submit-race", "sess-A", [
        cmd_stage("build", ["python3", "-c", code]),
    ], artifacts=["runner_count.txt"])
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _index: submit_job(worker_root, spec), range(8)))
    assert sum(1 for _rc, payload in results if not payload["idempotent"]) == 1
    state = wait_terminal(worker_root, "job-submit-race")
    assert state["status"] == "succeeded", state
    lines = (worker_root / "jobs" / "job-submit-race" /
             "runner_count.txt").read_text().splitlines()
    assert lines == ["run"]


def test_submit_conflict_on_different_spec(worker_root):
    spec1 = simple_spec("job-x", "sess-A", [cmd_stage("build", ["true"])])
    spec2 = simple_spec("job-x", "sess-A", [cmd_stage("build", ["false"])])
    submit_job(worker_root, spec1)
    rc, payload = submit_job(worker_root, spec2, expect_rc=3)
    assert payload["code"] == 3


def test_cancel_running_job(worker_root):
    acquire_lease(worker_root, "sess-A")
    spec = simple_spec("job-cancel", "sess-A", [
        cmd_stage("verify", ["sleep", "30"], needs_device=True,
                  failure_class="correctness", timeout_s=120),
    ])
    submit_job(worker_root, spec)
    # 等 runner 真正开始跑设备阶段
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        rc, payload, _ = run_worker(worker_root, "job-status", "--job-id", "job-cancel")
        if payload["state"].get("status") == "running" and payload["state"].get("pgid"):
            break
        time.sleep(0.2)
    rc, payload, _ = run_worker(worker_root, "job-cancel", "--job-id", "job-cancel")
    assert rc == 0, payload
    state = wait_terminal(worker_root, "job-cancel", timeout=20)
    assert state["status"] == "cancelled", state
    assert state["failure_class"] == "cancelled"


def test_cancel_refuses_pid_reuse_victim(worker_root):
    """pgid 指向不属于本 job 的存活进程时，cancel 必须拒杀（防 PID 复用误杀）。"""
    acquire_lease(worker_root, "sess-A")
    spec = simple_spec("job-pidguard", "sess-A", [
        cmd_stage("build", ["true"]),
    ])
    submit_job(worker_root, spec)
    wait_terminal(worker_root, "job-pidguard")
    # 伪造状态：running + pgid 指向一个真实存在但无关的进程（sleep 300）
    victim = subprocess.Popen(["sleep", "300"])
    try:
        state_path = worker_root / "jobs" / "job-pidguard" / "state.json"
        state = json.loads(state_path.read_text())
        state["status"] = "running"
        state["pgid"] = victim.pid
        state["cancel_requested"] = False
        state_path.write_text(json.dumps(state))
        rc, payload, _ = run_worker(worker_root, "job-cancel",
                                    "--job-id", "job-pidguard")
        assert rc == 0
        assert payload["killed"] is False
        assert "refusing to kill" in payload["error"]
        assert victim.poll() is None  # 无辜进程未被杀
    finally:
        victim.terminate()
        victim.wait()


def test_runner_identity_uses_exact_job_id(monkeypatch):
    import remote_worker
    monkeypatch.setattr(remote_worker, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(remote_worker, "_proc_start_token", lambda _pid: "token")
    monkeypatch.setattr(
        remote_worker, "_pgid_cmdline",
        lambda _pid: ["python3", "remote_worker.py", "--root", "/tmp/r",
                      "_job-runner", "--job-id", "job10"])
    state = {"pgid": 123, "runner_start_token": "token"}
    assert remote_worker._runner_identity_matches(state, "job1") is False
    assert remote_worker._runner_identity_matches(state, "job10") is True


def test_failure_classification(worker_root):
    """correctness / build / profile / environment / lease_lost 各自归位。"""
    acquire_lease(worker_root, "sess-A")
    cases = [
        ("job-fail-correct", "sess-A",
         [cmd_stage("verify", ["python3", "-c", "raise SystemExit(1)"],
                    needs_device=True, failure_class="correctness")], "correctness"),
        ("job-fail-build", "sess-A",
         [cmd_stage("build", ["python3", "-c", "raise SystemExit(2)"],
                    failure_class="build")], "build"),
        ("job-fail-profile", "sess-A",
         [cmd_stage("build", ["true"], failure_class="build"),
          cmd_stage("profile", ["python3", "-c", "raise SystemExit(3)"],
                    needs_device=True, failure_class="profile")], "profile"),
        ("job-fail-env", "sess-A",
         [cmd_stage("build", ["definitely-not-a-command-xyz", "--flag"],
                    failure_class="build")], "environment"),
    ]
    for job_id, session, stages, expect in cases:
        submit_job(worker_root, simple_spec(job_id, session, stages))
        state = wait_terminal(worker_root, job_id)
        assert state["status"] == "failed", (job_id, state)
        assert state["failure_class"] == expect, (job_id, state)

    # lease_lost：session 没有租约却要求设备阶段
    submit_job(worker_root, simple_spec(
        "job-lease-lost", "sess-nolease",
        [cmd_stage("verify", ["true"], needs_device=True, failure_class="correctness")]))
    state = wait_terminal(worker_root, "job-lease-lost")
    assert state["status"] == "failed"
    assert state["failure_class"] == "lease_lost", state


def test_state_recovery_marks_dead_runner(worker_root):
    """runner 进程死亡但 state 仍是 running：job-status 标记 failed/environment。"""
    acquire_lease(worker_root, "sess-A")
    submit_job(worker_root, simple_spec("job-recover", "sess-A",
                                        [cmd_stage("build", ["true"])]))
    wait_terminal(worker_root, "job-recover")
    state_path = worker_root / "jobs" / "job-recover" / "state.json"
    state = json.loads(state_path.read_text())
    state["status"] = "running"
    state["pgid"] = 999999  # 不存在的进程
    state["started_ts"] = time.time() - 60
    state_path.write_text(json.dumps(state))
    rc, payload, _ = run_worker(worker_root, "job-status", "--job-id", "job-recover")
    assert rc == 0
    recovered = payload["state"]
    assert recovered["status"] == "failed"
    assert recovered["failure_class"] == "environment"


def test_private_pythonpath_no_global_install(worker_root, tmp_path):
    """pip 只允许 --target 到 job 私有目录；PYTHONPATH 指向该目录。"""
    acquire_lease(worker_root, "sess-A")
    source_dir = tmp_path / "source"
    wheel_dir = source_dir / "wheels"
    make_minimal_wheel(wheel_dir)

    probe = (
        "import os, sys, json;"
        "out={'pythonpath': os.environ.get('PYTHONPATH',''),"
        "     'import_ok': False};"
        "sys.path.insert(0, os.environ.get('PYTHONPATH','').split(':')[0]);"
        "\n"
        "try:\n"
        "    import canarypkg; out['import_ok']=True\n"
        "except Exception as exc:\n"
        "    out['import_err']=str(exc)\n"
        "open('pip_probe.txt','w').write(json.dumps(out))"
    )
    spec = simple_spec("job-pip", "sess-A", [
        {"name": "install", "kind": "pip_install", "needs_device": False,
         "failure_class": "build", "wheel": "bundle/wheels/*.whl",
         "target": "pylibs", "timeout_s": 120},
        cmd_stage("verify", ["python3", "-c", probe], needs_device=True,
                  failure_class="correctness",
                  env={"PYTHONPATH": "{job_dir}/pylibs:{job_dir}"}),
    ], artifacts=["pip_probe.txt"])
    stage_source(worker_root, spec, source_dir)
    submit_job(worker_root, spec)
    state = wait_terminal(worker_root, "job-pip", timeout=120)
    assert state["status"] == "succeeded", state
    job_dir = worker_root / "jobs" / "job-pip"
    # 1) 安装落在 job 私有目录
    assert (job_dir / "pylibs" / "canarypkg" / "__init__.py").exists()
    # 2) verify 阶段通过私有 PYTHONPATH 导入成功
    probe_result = json.loads((job_dir / "pip_probe.txt").read_text())
    assert probe_result["import_ok"] is True, probe_result
    assert str(job_dir / "pylibs") in probe_result["pythonpath"]
    # 3) pip 命令行带了 --target（日志可审计），不存在全局安装路径
    install_log = (job_dir / "logs" / "install.log").read_text()
    assert '"--target"' in install_log
    # 4) 当前解释器的全局环境未被写入
    probe_dir = str(job_dir / "pylibs")
    env_probe = subprocess.run(
        [sys.executable, "-c",
         "import sys, json; print(json.dumps([p for p in sys.path if 'site-packages' in p]))"],
        stdout=subprocess.PIPE, universal_newlines=True)
    for site_dir in json.loads(env_probe.stdout):
        assert probe_dir not in site_dir  # 私有目录不在全局 site-packages 里
        assert not (Path(site_dir) / "canarypkg").exists()


def test_user_cannot_override_device_env(worker_root):
    """spec 自带 ASCEND_RT_VISIBLE_DEVICES 必须被拒绝（防止绕过全局分配）。"""
    acquire_lease(worker_root, "sess-A")
    spec = simple_spec("job-env-attack", "sess-A", [
        cmd_stage("verify", ["true"], needs_device=True,
                  env={"ASCEND_RT_VISIBLE_DEVICES": "9"}),
    ])
    rc, payload = submit_job(worker_root, spec, expect_rc=2)
    assert "reserved" in payload["error"]
