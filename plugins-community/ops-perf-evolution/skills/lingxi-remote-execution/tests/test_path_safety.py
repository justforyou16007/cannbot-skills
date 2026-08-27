#!/usr/bin/env python3
"""路径安全测试：所有相对路径必须限制在 root/job 目录内，
拒绝 `..`、绝对逃逸与符号链接逃逸；删除只针对单个 job 目录。"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import remote_worker  # noqa: E402

from conftest import acquire_lease, cmd_stage, run_worker, simple_spec, submit_job  # noqa: E402


# --- safe_join 单元测试 ---

def test_safe_join_rejects_escape(tmp_path):
    root = tmp_path / "root"
    (root / "jobs" / "j1").mkdir(parents=True)
    for bad in ("../..", "a/../../b", "/etc/passwd", "..", "a//b", "a/./b", "~/x", "a\\b"):
        with pytest.raises(remote_worker.WorkerError):
            remote_worker.safe_join(root, bad)


def test_safe_join_rejects_symlink_escape(tmp_path):
    root = tmp_path / "root"
    job = root / "jobs" / "j1"
    job.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    os.symlink(str(outside), str(job / "evil"))
    with pytest.raises(remote_worker.WorkerError):
        remote_worker.safe_join(job, "evil/secret.txt")
    # 指向 root 内部的符号链接不拒绝（无逃逸）
    os.symlink("plain", str(job / "good"))
    (job / "plain").mkdir()
    assert remote_worker.safe_join(job, "good/x") == job / "good" / "x"


def test_safe_join_allows_normal_relative(tmp_path):
    root = tmp_path / "root"
    job = root / "jobs" / "j1"
    job.mkdir(parents=True)
    assert remote_worker.safe_join(job, "kernel/build") == job / "kernel" / "build"


# --- spec 校验（submit 时拦截） ---

def test_submit_rejects_path_escape_cwd(worker_root):
    spec = simple_spec("job-escape-cwd", "sess-A", [
        cmd_stage("build", ["true"], cwd="../../.."),
    ])
    rc, payload = submit_job(worker_root, spec, expect_rc=2)
    assert "escape" in payload["error"] or "unsafe" in payload["error"]


def test_submit_rejects_pip_wheel_escape(worker_root):
    spec = simple_spec("job-escape-wheel", "sess-A", [
        {"name": "install", "kind": "pip_install", "failure_class": "build",
         "wheel": "../../etc/*.whl", "target": "pylibs"},
    ])
    submit_job(worker_root, spec, expect_rc=2)


def test_submit_rejects_artifacts_escape(worker_root):
    spec = simple_spec("job-escape-art", "sess-A", [
        cmd_stage("build", ["true"]),
    ], artifacts=["../../etc/*"])
    submit_job(worker_root, spec, expect_rc=2)


def test_artifact_glob_cannot_follow_symlink_outside_job(worker_root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    code = "import os; os.symlink(%r, 'evil')" % str(outside)
    spec = simple_spec("job-art-link", "sess-A", [
        cmd_stage("build", ["python3", "-c", code]),
    ], artifacts=["evil/*"])
    submit_job(worker_root, spec)
    from conftest import wait_terminal
    state = wait_terminal(worker_root, "job-art-link")
    assert state["status"] == "failed", state
    assert state["failure_class"] == "environment"
    assert not (state.get("artifacts_manifest") or {}).get("files")


def test_submit_rejects_shell_metachar_program(worker_root):
    spec = simple_spec("job-shell-attack", "sess-A", [
        cmd_stage("build", ["bash -c 'rm -rf /'", "x"]),
    ])
    submit_job(worker_root, spec, expect_rc=2)


def test_submit_rejects_bad_job_id(worker_root):
    spec = simple_spec("../evil", "sess-A", [cmd_stage("build", ["true"])])
    submit_job(worker_root, spec, expect_rc=2)


def test_submit_rejects_absolute_program_with_dotdot(worker_root):
    spec = simple_spec("job-abs-attack", "sess-A", [
        cmd_stage("build", ["/bin/../..//bin/sh", "-c", "true"]),
    ])
    submit_job(worker_root, spec, expect_rc=2)


def test_submit_rejects_unknown_stage_kind(worker_root):
    spec = simple_spec("job-kind", "sess-A", [
        {"name": "build", "kind": "shell", "commands": [["true"]],
         "failure_class": "build"},
    ])
    submit_job(worker_root, spec, expect_rc=2)


def test_cleanup_is_narrow_and_refuses_running(worker_root):
    """job-cleanup 只删单个 job 目录；运行中的 job 拒删。"""
    acquire_lease(worker_root, "sess-A")
    submit_job(worker_root, simple_spec(
        "job-cleanup-me", "sess-A",
        [cmd_stage("verify", ["sleep", "20"], needs_device=True, timeout_s=60)]))
    import time
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        rc, payload, _ = run_worker(worker_root, "job-status", "--job-id", "job-cleanup-me")
        if payload["state"].get("status") == "running":
            break
        time.sleep(0.2)
    # 运行中：拒删
    rc, payload, _ = run_worker(worker_root, "job-cleanup", "--job-id", "job-cleanup-me")
    assert rc == 2
    # 取消后可删
    run_worker(worker_root, "job-cancel", "--job-id", "job-cleanup-me")
    from conftest import wait_terminal
    wait_terminal(worker_root, "job-cleanup-me", timeout=20)
    rc, payload, _ = run_worker(worker_root, "job-cleanup", "--job-id", "job-cleanup-me")
    assert rc == 0
    assert not (worker_root / "jobs" / "job-cleanup-me").exists()
    # root 下其他内容（pool、其他 job）不受影响
    assert (worker_root / "pool" / "pool_config.json").exists()


def test_cleanup_rejects_path_like_job_id(worker_root):
    rc, payload, _ = run_worker(worker_root, "job-cleanup", "--job-id", "a/b")
    assert rc == 2
    rc, payload, _ = run_worker(worker_root, "job-cleanup", "--job-id", "..")
    assert rc == 2


def test_cleanup_rejects_symlink_alias_to_another_job(worker_root):
    submit_job(worker_root, simple_spec(
        "real-job", "sess-A", [cmd_stage("build", ["true"])]))
    from conftest import wait_terminal
    wait_terminal(worker_root, "real-job")
    os.symlink("real-job", str(worker_root / "jobs" / "alias-job"))
    rc, payload, _ = run_worker(worker_root, "job-cleanup", "--job-id", "alias-job")
    assert rc == 2, payload
    assert (worker_root / "jobs" / "real-job" / "state.json").exists()
