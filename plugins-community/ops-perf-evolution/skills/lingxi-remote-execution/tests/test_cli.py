#!/usr/bin/env python3
"""本地 CLI 测试：--help、local 传输端到端、preflight（假 npu-smi/env）、
recover、batch（多算子受设备池约束）。"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from conftest import (REMOTE_EXEC, make_fake_bin, make_fake_env_script,
                      make_fake_npu_smi, run_cli, write_host_config)
import remote_exec
import remote_worker


def test_cli_help_all_subcommands():
    proc = subprocess.run([sys.executable, str(REMOTE_EXEC), "--help"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0
    for cmd in ("deploy", "preflight", "lease", "job", "recover", "batch"):
        proc = subprocess.run(
            [sys.executable, str(REMOTE_EXEC), cmd, "--help"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.returncode == 0, cmd
        # 子命令的 --help 必须真的有内容
        assert proc.stdout, cmd
    for sub in ("acquire", "heartbeat", "release", "list"):
        proc = subprocess.run(
            [sys.executable, str(REMOTE_EXEC), "lease", sub, "--help"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.returncode == 0, sub
    for sub in ("submit", "status", "cancel", "fetch", "list", "cleanup"):
        proc = subprocess.run(
            [sys.executable, str(REMOTE_EXEC), "job", sub, "--help"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.returncode == 0, sub


@pytest.fixture
def local_host(tmp_path, monkeypatch):
    """local 传输 + 假 npu-smi/cmake/make/g++ + 假 CANN env 脚本的完整环境。"""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    make_fake_npu_smi(tmp_path, devices=(0, 1))
    for tool in ("cmake", "make", "g++", "msprof"):
        make_fake_bin(tmp_path, tool, "exit 0\n")
    env_script = make_fake_env_script(tmp_path)
    monkeypatch.setenv("PATH", str(bin_dir) + ":" + os.environ.get("PATH", ""))
    root = tmp_path / "remote_root"
    cfg = write_host_config(tmp_path, transport="local", devices=(0, 1),
                            env_script=env_script, remote_root=root)
    return cfg, root, env_script


def test_local_deploy_and_preflight(local_host):
    cfg, root, env_script = local_host
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a", "deploy")
    assert rc == 0, (payload, err)
    assert (root / "lib" / "remote_worker.py").exists()

    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a", "preflight")
    assert rc == 0, (payload, err)
    assert payload["ok"] is True, payload
    by_name = {c["name"]: c for c in payload["checks"]}
    assert by_name["env_script"]["ok"] is True
    assert "fake_ascend" in by_name["env_script"]["detail"]
    assert payload["devices"] == [0, 1]
    # pool 配置已写入：设备列表来自配置（或 npu-smi 自动发现）
    pool_cfg = json.loads((root / "pool" / "pool_config.json").read_text())
    assert pool_cfg["devices"] == [0, 1]
    assert pool_cfg["env_script"] == str(env_script)


def test_npu_detection_prefers_mapping_and_skips_mcu(monkeypatch):
    """设备映射中的 MCU 行及主表里的温度数字都不能被当成独立卡。"""
    class Result(object):
        def __init__(self, stdout, returncode=0):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode

    calls = []

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        return Result(
            "NPU ID  Chip ID  Chip Logic ID  Chip Name\n"
            "0       0        0              Ascend 910B3\n"
            "0       1        -              Mcu\n"
            "2       0        2              Ascend 910B3\n")

    monkeypatch.setattr(remote_worker.shutil, "which", lambda *_args, **_kwargs: "/bin/npu-smi")
    monkeypatch.setattr(remote_worker.subprocess, "run", fake_run)
    assert remote_worker._detect_npu_devices({"PATH": "/bin"}) == [0, 2]
    assert calls == [["/bin/npu-smi", "info", "-m"]]


def test_npu_detection_main_table_fallback_uses_leading_id_only(monkeypatch):
    class Result(object):
        def __init__(self, stdout, returncode):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode

    def fake_run(argv, **_kwargs):
        if argv[-1] == "-m":
            return Result("", 1)
        return Result(
            "| NPU     Name | Health | Power(W) Temp(C) |\n"
            "| 3       910B3 | OK | 95 42 |\n"
            "| 7       910B3 | OK | 88 55 |\n", 0)

    monkeypatch.setattr(remote_worker.shutil, "which", lambda *_args, **_kwargs: "/bin/npu-smi")
    monkeypatch.setattr(remote_worker.subprocess, "run", fake_run)
    assert remote_worker._detect_npu_devices({"PATH": "/bin"}) == [3, 7]


def test_local_preflight_reports_missing_env(local_host, tmp_path):
    cfg, root, _ = local_host
    # 改成不存在的 env_script
    cfg_data = json.loads(cfg.read_text())
    cfg_data["hosts"]["npu-a"]["env_script"] = str(tmp_path / "nope.sh")
    cfg.write_text(json.dumps(cfg_data))
    run_cli("--config", str(cfg), "--host", "npu-a", "deploy")
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a", "preflight")
    assert rc == 1  # preflight 有失败项 -> 非零
    by_name = {c["name"]: c for c in payload["checks"]}
    assert by_name["env_script"]["ok"] is False


def test_preflight_checks_remote_evaluator_paths(local_host, tmp_path):
    cfg, _root, _ = local_host
    cfg_data = json.loads(cfg.read_text())
    cfg_data["hosts"]["npu-a"]["required_paths"] = [str(tmp_path / "missing_verify.py")]
    cfg.write_text(json.dumps(cfg_data))
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a", "preflight")
    assert rc == 1, (payload, err)
    required = [c for c in payload["checks"] if c["name"] == "required_path"]
    assert required == [{"name": "required_path", "ok": False,
                         "detail": str(tmp_path / "missing_verify.py")}]


def test_preflight_refuses_pool_change_while_lease_is_active(local_host, tmp_path):
    cfg, root, _ = local_host
    state_file = tmp_path / "pool-state.json"
    assert run_cli("--config", str(cfg), "--host", "npu-a", "preflight")[0] == 0
    rc, payload, err = run_cli(
        "--config", str(cfg), "--host", "npu-a", "--state-file", str(state_file),
        "lease", "acquire", "--session", "pool-s", "--owner", "PoolOp",
        "--device", "0")
    assert rc == 0, (payload, err)
    cfg_data = json.loads(cfg.read_text())
    cfg_data["hosts"]["npu-a"]["devices"] = [1]
    cfg.write_text(json.dumps(cfg_data))
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a", "preflight")
    assert rc == 1, (payload, err)
    assert "active leases" in payload["error"]
    stored = json.loads((root / "pool" / "pool_config.json").read_text())
    assert stored["devices"] == [0, 1]


def test_local_job_flow_and_recover(local_host, tmp_path):
    cfg, root, _ = local_host
    state_file = tmp_path / "st.json"
    run_cli("--config", str(cfg), "--host", "npu-a", "preflight")
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a",
                               "--state-file", str(state_file),
                               "lease", "acquire", "--session", "sess1",
                               "--owner", "FastGELU")
    assert rc == 0, (payload, err)
    lease = payload["lease"]

    spec = {
        "job_id": "j1", "session_id": "sess1",
        "stages": [
            {"name": "build", "kind": "cmd", "needs_device": False,
             "failure_class": "build", "commands": [["python3", "-c", "print('b')"]],
             "env": {}, "cwd": "", "timeout_s": 30},
            {"name": "verify", "kind": "cmd", "needs_device": True,
             "failure_class": "correctness",
             "commands": [["python3", "-c",
                           "import os;open('ok.txt','w').write("
                           "os.environ['ASCEND_RT_VISIBLE_DEVICES'])"]],
             "env": {}, "cwd": "", "timeout_s": 30},
        ],
        "artifacts": ["ok.txt"],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a",
                               "--state-file", str(state_file),
                               "job", "submit", "--job-id", "j1",
                               "--session", "sess1",
                               "--stage-json", str(spec_path),
                               "--wait", "--poll-interval", "1")
    assert rc == 0, (payload, err)
    assert payload["state"]["status"] == "succeeded"

    # recover：断线重连后能看到 lease / job 状态
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a",
                               "--state-file", str(state_file),
                               "recover", "--session", "sess1")
    assert rc == 0, (payload, err)
    assert payload["jobs"][0]["job_id"] == "j1"
    assert any(e["device_id"] == lease["device_id"] and e["valid"]
               for e in payload["pool"])

    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a",
                               "job", "fetch", "--job-id", "j1",
                               "--dest", str(tmp_path / "out"))
    assert rc == 0, (payload, err)
    assert (tmp_path / "out" / "ok.txt").read_text().strip() == str(lease["device_id"])

    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a",
                               "lease", "release", "--lease-id", lease["lease_id"])
    assert rc == 0, (payload, err)
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a",
                               "lease", "list")
    assert rc == 0, (payload, err)
    assert all(not e["lease"] for e in payload["pool"])


def test_batch_bounded_by_device_pool(local_host, tmp_path):
    """3 个算子 session、2 张卡：任何时刻最多 2 个持有租约（活跃算子受池约束）。"""
    cfg, root, _ = local_host
    run_cli("--config", str(cfg), "--host", "npu-a", "preflight")
    # 设备阶段记录"进入/退出"时间戳，用于验证同时最多 2 个并发
    probe = (
        "import json,os,time;"
        "t=time.time();time.sleep(0.6);"
        "open('active_%s.txt' % os.environ['ASCEND_RT_VISIBLE_DEVICES'],'a')"
        ".write(json.dumps([t, time.time()]) + '\\n')"
    )
    sessions = []
    for i in range(3):
        spec = {
            "job_id": "b%d-job" % i, "session_id": "b%d" % i,
            "stages": [
                {"name": "verify", "kind": "cmd", "needs_device": True,
                 "failure_class": "correctness",
                 "commands": [["python3", "-c", probe]],
                 "env": {}, "cwd": "", "timeout_s": 60},
            ],
            "artifacts": ["active_*.txt"],
        }
        spec_path = tmp_path / ("spec_b%d.json" % i)
        spec_path.write_text(json.dumps(spec))
        sessions.append({
            "session_id": "b%d" % i, "owner": "op%d" % i,
            "job_id": "b%d-job" % i, "stage_json": str(spec_path),
            "artifacts": ["active_*.txt"],
        })
    plan = {"sessions": sessions}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a",
                               "--state-file", str(tmp_path / "st.json"),
                               "batch", "--plan", str(plan_path),
                               "--poll-interval", "1",
                               "--fetch-dir", str(tmp_path / "fetched"))
    assert rc == 0, (payload, err)
    statuses = {sid: out["status"] for sid, out in payload["batch"].items()}
    assert set(statuses.values()) == {"succeeded"}, payload

    # 汇总两个 job 目录里的设备阶段时间区间：并发设备数 <= 2
    intervals = []
    for job_dir in (root / "jobs").iterdir():
        for f in job_dir.glob("active_*.txt"):
            for line in f.read_text().splitlines():
                intervals.append(json.loads(line))
    intervals.sort()
    max_concurrent = 0
    events = []
    for s, e in intervals:
        events.append((s, 1))
        events.append((e, -1))
    events.sort()
    cur = 0
    for _, delta in events:
        cur += delta
        max_concurrent = max(max_concurrent, cur)
    assert max_concurrent <= 2, max_concurrent
    # 至少真的并行过（否则约束没有意义）
    assert max_concurrent == 2, max_concurrent


def test_ascendc_preset_real_layout_private_install_and_fetch(local_host, tmp_path):
    """The actual preset must build from bundle/, install privately, and restore outputs."""
    cfg, root, _ = local_host
    state_file = tmp_path / "state.json"
    assert run_cli("--config", str(cfg), "--host", "npu-a", "preflight")[0] == 0
    rc, lease_payload, err = run_cli(
        "--config", str(cfg), "--host", "npu-a", "--state-file", str(state_file),
        "lease", "acquire", "--session", "preset-s", "--owner", "PresetOp")
    assert rc == 0, (lease_payload, err)

    source = tmp_path / "candidate"
    kernel = source / "kernel"
    kernel.mkdir(parents=True)
    (kernel / "setup.py").write_text(
        "from setuptools import setup\n"
        "setup(name='presetpkg', version='0.0.1', py_modules=['presetpkg'])\n",
        encoding="utf-8")
    (kernel / "presetpkg.py").write_text("VALUE = 42\n", encoding="utf-8")
    (source / "verify.py").write_text(
        "import json, presetpkg\n"
        "open('evaluation_results.json','w').write(json.dumps({'value': presetpkg.VALUE}))\n",
        encoding="utf-8")
    (source / "profile.py").write_text(
        "import json\nopen('performance.json','w').write(json.dumps({'us': 12.5}))\n",
        encoding="utf-8")
    output = tmp_path / "restored"
    rc, payload, err = run_cli(
        "--config", str(cfg), "--host", "npu-a", "--state-file", str(state_file),
        "job", "submit", "--job-id", "preset-job", "--session", "preset-s",
        "--src", str(source), "--preset", "ascendc-eval", "--kernel-dir", "kernel",
        "--verify-cmd", '["python3","verify.py"]',
        "--profile-cmd", '["python3","profile.py"]',
        "--artifacts", "evaluation_results.json", "performance.json",
        "--wait", "--poll-interval", "1", "--fetch", "--fetch-dest", str(output),
        timeout=240)
    assert rc == 0, (payload, err)
    assert payload["state"]["status"] == "succeeded", payload
    assert json.loads((output / "evaluation_results.json").read_text())["value"] == 42
    assert json.loads((output / "performance.json").read_text())["us"] == 12.5
    assert not (output / "bundle" / "performance.json").exists()
    spec = json.loads((root / "jobs" / "preset-job" / "spec.json").read_text())
    assert spec["stages"][0]["commands"][0][1] == "{bundle_dir}/kernel"
    assert spec["stages"][3]["commands"] == [["python3", "verify.py"]]


def test_msprof_compare_artifacts_are_collected_into_job(local_host, tmp_path):
    """--compare 的 /tmp 原始数据必须在释放设备锁后归档并可回传。"""
    cfg, root, _ = local_host
    state_file = tmp_path / "profile-state.json"
    assert run_cli("--config", str(cfg), "--host", "npu-a", "preflight")[0] == 0
    rc, lease_payload, err = run_cli(
        "--config", str(cfg), "--host", "npu-a", "--state-file", str(state_file),
        "lease", "acquire", "--session", "profile-s", "--owner", "ProfileOp")
    assert rc == 0, (lease_payload, err)

    source = tmp_path / "profile-candidate"
    kernel = source / "kernel"
    kernel.mkdir(parents=True)
    (kernel / "setup.py").write_text(
        "from setuptools import setup\nsetup(name='profilepkg', version='0.0.1')\n",
        encoding="utf-8")
    tag = "lingxi_" + tmp_path.name.replace("-", "_")
    ref_container = Path("/tmp") / ("msprof_ref_bundle_c0_" + tag)
    asc_container = Path("/tmp") / ("msprof_asc_bundle_c0_" + tag)
    profiler = tmp_path / "msprof_perf_summary.py"
    profiler.write_text(
        "import json, pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "ref = pathlib.Path(%r) / 'PROF_GROUP_ref'\n" % str(ref_container) +
        "asc = pathlib.Path(%r) / 'PROF_GROUP_asc'\n" % str(asc_container) +
        "ref.mkdir(parents=True); asc.mkdir(parents=True)\n"
        "(ref/'op_summary_ref.csv').write_text('metric,value\\npipe,1\\n')\n"
        "(asc/'op_summary_asc.csv').write_text('metric,value\\npipe,2\\n')\n"
        "data={'per_case':[{'ref_prof_dir':str(ref),'asc_prof_dir':str(asc)}]}\n"
        "(out/'performance.json').write_text(json.dumps(data))\n",
        encoding="utf-8")
    output = tmp_path / "profile-restored"
    try:
        profile_cmd = json.dumps([
            "python3", str(profiler), "--compare", "--output-dir", "{bundle_dir}"
        ])
        rc, payload, err = run_cli(
            "--config", str(cfg), "--host", "npu-a", "--state-file", str(state_file),
            "job", "submit", "--job-id", "profile-job", "--session", "profile-s",
            "--src", str(source), "--preset", "ascendc-eval", "--kernel-dir", "kernel",
            "--verify-cmd", '["true"]', "--profile-cmd", profile_cmd,
            "--artifacts", "performance.json", "--wait", "--poll-interval", "1",
            "--fetch", "--fetch-dest", str(output), timeout=240)
        assert rc == 0, (payload, err)
        assert list(output.glob("profiling/case_0000_ref/PROF_GROUP_ref/op_summary_*.csv"))
        assert list(output.glob("profiling/case_0000_asc/PROF_GROUP_asc/op_summary_*.csv"))
        spec = json.loads((root / "jobs" / "profile-job" / "spec.json").read_text())
        profile_stage = next(stage for stage in spec["stages"] if stage["name"] == "profile")
        collect_stage = next(stage for stage in spec["stages"]
                             if stage["name"] == "profile_collect")
        assert "--keep-prof" in profile_stage["commands"][0]
        assert collect_stage["needs_device"] is False
        assert "bundle/profiling" in spec["artifacts"]
    finally:
        shutil.rmtree(str(ref_container), ignore_errors=True)
        shutil.rmtree(str(asc_container), ignore_errors=True)


def test_source_digest_makes_job_idempotency_content_aware(local_host, tmp_path):
    cfg, root, _ = local_host
    assert run_cli("--config", str(cfg), "--host", "npu-a", "preflight")[0] == 0
    source = tmp_path / "source"
    source.mkdir()
    value = source / "value.txt"
    value.write_text("A", encoding="utf-8")
    spec = {
        "job_id": "content-job", "session_id": "content-s",
        "stages": [{"name": "build", "kind": "cmd", "needs_device": False,
                    "failure_class": "build",
                    "commands": [["python3", "-c", "open('seen.txt','w').write(open('bundle/value.txt').read())"]],
                    "cwd": "", "env": {}, "timeout_s": 30}],
        "artifacts": ["seen.txt"],
    }
    spec_path = tmp_path / "content-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    common = ["--config", str(cfg), "--host", "npu-a",
              "--state-file", str(tmp_path / "content-state.json"),
              "job", "submit", "--job-id", "content-job", "--session", "content-s",
              "--stage-json", str(spec_path), "--src", str(source)]
    rc, payload, err = run_cli(*(common + ["--wait", "--poll-interval", "1"]))
    assert rc == 0, (payload, err)
    rc, payload, err = run_cli(*common)
    assert rc == 0 and payload["idempotent"] is True, (payload, err)
    value.write_text("B", encoding="utf-8")
    rc, payload, err = run_cli(*common)
    assert rc == 3, (payload, err)
    assert (root / "jobs" / "content-job" / "bundle" / "value.txt").read_text() == "A"


def test_repeated_sources_keep_deterministic_extra_layout(local_host, tmp_path):
    """第二个源码目录必须进入 extra_0，且本地/远端摘要计算口径一致。"""
    cfg, root, _ = local_host
    assert run_cli("--config", str(cfg), "--host", "npu-a", "preflight")[0] == 0
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.txt").write_text("A", encoding="utf-8")
    (second / "b.txt").write_text("B", encoding="utf-8")
    spec = {
        "job_id": "multi-source", "session_id": "source-session",
        "stages": [{"name": "build", "kind": "cmd", "needs_device": False,
                    "failure_class": "build",
                    "commands": [["python3", "-c",
                                  "open('joined.txt','w').write("
                                  "open('bundle/a.txt').read()+"
                                  "open('bundle/extra_0/b.txt').read())"]],
                    "cwd": "", "env": {}, "timeout_s": 30}],
        "artifacts": ["joined.txt"],
    }
    spec_path = tmp_path / "multi-source.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    rc, payload, err = run_cli(
        "--config", str(cfg), "--host", "npu-a",
        "--state-file", str(tmp_path / "multi-state.json"),
        "job", "submit", "--job-id", "multi-source", "--session", "source-session",
        "--src", str(first), "--src", str(second), "--stage-json", str(spec_path),
        "--wait", "--poll-interval", "1")
    assert rc == 0, (payload, err)
    job_dir = root / "jobs" / "multi-source"
    assert (job_dir / "joined.txt").read_text() == "AB"
    stored = json.loads((job_dir / "spec.json").read_text())
    assert stored["source_digest"] == remote_exec.source_bundle_sha256([first, second])


def test_failed_job_wait_returns_nonzero(local_host, tmp_path):
    cfg, _root, _ = local_host
    assert run_cli("--config", str(cfg), "--host", "npu-a", "preflight")[0] == 0
    spec = {"job_id": "failed-cli", "session_id": "s-fail",
            "stages": [{"name": "build", "kind": "cmd", "needs_device": False,
                        "failure_class": "build", "commands": [["false"]],
                        "cwd": "", "env": {}, "timeout_s": 30}], "artifacts": []}
    spec_path = tmp_path / "failed.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    rc, payload, err = run_cli(
        "--config", str(cfg), "--host", "npu-a", "--state-file", str(tmp_path / "s.json"),
        "job", "submit", "--job-id", "failed-cli", "--session", "s-fail",
        "--stage-json", str(spec_path), "--wait", "--poll-interval", "1")
    assert rc == 1, (payload, err)
    assert payload["state"]["status"] == "failed"


def test_batch_rejects_malformed_entry_before_threads(local_host, tmp_path):
    cfg, _root, _ = local_host
    assert run_cli("--config", str(cfg), "--host", "npu-a", "preflight")[0] == 0
    plan_path = tmp_path / "bad-plan.json"
    plan_path.write_text(json.dumps({"sessions": [{"owner": "missing-id"}]}))
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a",
                               "batch", "--plan", str(plan_path))
    assert rc == 2, (payload, err)
    assert "session_id" in payload["error"]


def test_recover_refuses_silent_device_change(local_host, tmp_path):
    cfg, _root, _ = local_host
    state_file = tmp_path / "recover-state.json"
    assert run_cli("--config", str(cfg), "--host", "npu-a", "preflight")[0] == 0
    rc, payload, _ = run_cli(
        "--config", str(cfg), "--host", "npu-a", "--state-file", str(state_file),
        "lease", "acquire", "--session", "recover-s", "--owner", "RecoverOp",
        "--device", "0")
    old_lease = payload["lease"]
    run_cli("--config", str(cfg), "--host", "npu-a", "--state-file", str(state_file),
            "lease", "release", "--lease-id", old_lease["lease_id"])
    run_cli("--config", str(cfg), "--host", "npu-a", "--state-file", str(tmp_path / "b.json"),
            "lease", "acquire", "--session", "blocker", "--owner", "Blocker", "--device", "0")
    rc, payload, err = run_cli(
        "--config", str(cfg), "--host", "npu-a", "--state-file", str(state_file),
        "recover", "--session", "recover-s", "--reattach")
    assert rc == 1, (payload, err)
    assert payload["requires_new_baseline"] is True
    assert "different card was not accepted" in payload["reattach_error"]


def test_local_state_updates_are_atomic(tmp_path):
    state_file = tmp_path / "parallel-state.json"
    threads = [threading.Thread(
        target=remote_exec.record_session,
        args=(str(state_file), "h", "s%d" % i), kwargs={"device_id": i})
        for i in range(30)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state = json.loads(state_file.read_text())
    assert len(state["sessions"]) == 30
