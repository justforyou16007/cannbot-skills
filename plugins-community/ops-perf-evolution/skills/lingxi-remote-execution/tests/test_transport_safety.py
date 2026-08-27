#!/usr/bin/env python3
"""传输层与配置安全测试。

假 ssh 忠实模拟真实 ssh 的行为：把收到的参数拼成一条命令字符串、交给
远端 shell 解析执行。因此这里能真实检验：
1. 逐参数 shlex.quote 后，含空格/引号/元字符的参数能原样到达远端；
2. 配置中的 ssh_host / remote_root / ssh_args / env_script 里的危险值
   在加载阶段就被拒绝，而不是事后转义。
"""

import base64
import json
import subprocess
import sys
import time
import os
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import remote_exec

from conftest import (REMOTE_EXEC, make_fake_bin, make_fake_npu_smi, make_fake_ssh,
                      run_cli, run_worker,
                      write_host_config, write_pool_config,
                      make_fake_env_script)


def _make_transport(tmp_path):
    ssh = make_fake_ssh(tmp_path)
    make_fake_npu_smi(tmp_path, devices=(0, 1))
    for tool in ("cmake", "make", "g++", "msprof"):
        make_fake_bin(tmp_path, tool, "exit 0\n")
    env_script = make_fake_env_script(tmp_path)
    with env_script.open("a", encoding="utf-8") as fh:
        fh.write("export PATH=%s:$PATH\n" % (tmp_path / "fakebin"))
    root = tmp_path / "remote_root"
    cfg = write_host_config(tmp_path, ssh_bin=ssh, transport="ssh",
                            devices=(0, 1), remote_root=root,
                            env_script=env_script)
    return cfg, root


def test_fake_ssh_end_to_end_deploy_and_job(tmp_path):
    """deploy -> lease -> job submit/wait/fetch 全链路走假 ssh 传输。"""
    cfg, root = _make_transport(tmp_path)
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a", "deploy")
    assert rc == 0, (payload, err)
    assert (root / "lib" / "remote_worker.py").exists()

    state_file = tmp_path / "session_state.json"
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a",
                                "--state-file", str(state_file),
                                "lease", "acquire", "--session", "s1", "--owner", "op1")
    assert rc == 0, (payload, err)
    lease_id = payload["lease"]["lease_id"]
    device_id = payload["lease"]["device_id"]

    # stage-json：设备阶段回显设备号，验证 lease -> 设备阶段链路
    spec = {
        "job_id": "t-ssh-1", "session_id": "s1",
        "stages": [
            {"name": "verify", "kind": "cmd", "needs_device": True,
             "failure_class": "correctness",
             "commands": [["python3", "-c",
                           "import os; open('dev.txt','w').write("
                           "os.environ.get('ASCEND_RT_VISIBLE_DEVICES','none'))"]],
             "env": {}, "cwd": "", "timeout_s": 60},
        ],
        "artifacts": ["dev.txt"],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a",
                               "--state-file", str(state_file),
                               "job", "submit", "--job-id", "t-ssh-1",
                               "--session", "s1", "--stage-json", str(spec_path),
                               "--wait", "--poll-interval", "1", "--fetch",
                               "--fetch-dest", str(tmp_path / "fetched"))
    assert rc == 0, (payload, err)
    assert payload["state"]["status"] == "succeeded", payload
    dev_txt = (tmp_path / "fetched" / "dev.txt").read_text().strip()
    assert dev_txt == str(device_id)

    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a",
                               "--state-file", str(state_file),
                               "lease", "release", "--lease-id", lease_id)
    assert rc == 0, (payload, err)
    # 本地状态文件记录了 host/session/lease/device（断线重连依据）
    state = json.loads(state_file.read_text())
    assert state["sessions"]["s1"]["lease_id"] == lease_id
    assert state["sessions"]["s1"]["device_id"] == device_id


def test_argv_quoting_survives_ssh_roundtrip(tmp_path):
    """含空格/引号/元字符的参数经 ssh 传输后必须原样到达（不注入）。"""
    cfg, root = _make_transport(tmp_path)
    run_cli("--config", str(cfg), "--host", "npu-a", "deploy")
    # 直接构造一个带危险字符的 job_id 分隔场景：用 lease owner 携带特殊字符
    evil_owner = "op; rm -rf /tmp/pwned '$(echo hi)' `id`"
    state_file = tmp_path / "evil-session-state.json"
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a",
                                "--state-file", str(state_file),
                                "lease", "acquire", "--session", "s-evil",
                                "--owner", evil_owner)
    assert rc == 0, (payload, err)
    assert payload["lease"]["owner"] == evil_owner  # 原样到达，未被 shell 吃掉
    # 没有发生注入：不存在被 shell 展开的产物
    assert not Path("/tmp/pwned").exists()


def test_config_rejects_unsafe_ssh_host(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps({"hosts": {
        "evil": {"ssh_host": "host; rm -rf /", "remote_root": "/tmp/r"}}}), encoding="utf-8")
    rc, payload, err = run_cli("--config", str(cfg), "--host", "evil", "deploy")
    assert rc == 2
    assert "unsafe ssh host" in payload["error"]


def test_config_rejects_ssh_host_that_looks_like_an_option(tmp_path):
    cfg = tmp_path / "bad-option-host.json"
    cfg.write_text(json.dumps({"hosts": {
        "evil": {"ssh_host": "-oProxyCommand=evil", "remote_root": "/tmp/r"}}}))
    rc, payload, _ = run_cli("--config", str(cfg), "--host", "evil", "deploy")
    assert rc == 2
    assert "unsafe ssh host" in payload["error"]


def test_config_rejects_relative_remote_root(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps({"hosts": {
        "h": {"ssh_host": "host", "remote_root": "relative/path"}}}), encoding="utf-8")
    rc, payload, err = run_cli("--config", str(cfg), "--host", "h", "deploy")
    assert rc == 2
    assert "unsafe remote path" in payload["error"]


def test_config_rejects_unsafe_ssh_args(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps({"hosts": {
        "h": {"ssh_host": "host", "remote_root": "/tmp/r",
              "ssh_args": ["-o", "ProxyCommand=evil; rm -rf /"]}}}), encoding="utf-8")
    rc, payload, err = run_cli("--config", str(cfg), "--host", "h", "deploy")
    assert rc == 2
    assert "unsafe ssh_args" in payload["error"]


def test_config_rejects_relative_env_script(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps({"hosts": {
        "h": {"ssh_host": "host", "remote_root": "/tmp/r",
              "env_script": "set_env.sh"}}}), encoding="utf-8")
    rc, payload, err = run_cli("--config", str(cfg), "--host", "h", "deploy")
    assert rc == 2
    assert "env_script" in payload["error"]


def test_transport_failure_classified(tmp_path):
    """不可达的 ssh 目标必须归为 transport 失败（而不是远端 job 失败）。"""
    cfg = write_host_config(tmp_path, ssh_bin="/nonexistent/ssh-binary",
                            transport="ssh")
    rc, payload, err = run_cli("--config", str(cfg), "--host", "npu-a", "deploy")
    assert rc == 7
    assert payload["failure_class"] == "transport"


def test_fetch_rejects_remote_symlink_member(tmp_path):
    root = tmp_path / "root"
    base = root / "jobs" / "j1"
    base.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    os.symlink(str(outside), str(base / "evil"))
    session = remote_exec.RemoteSession({"transport": "local", "remote_root": str(root)})
    with pytest.raises(remote_exec.CliError, match="link/special"):
        session.fetch_dir("jobs/j1", tmp_path / "dest", patterns=["evil"])
    assert not (tmp_path / "dest" / "evil").exists()


def test_fetch_rejects_tar_option_and_reports_missing_file(tmp_path):
    root = tmp_path / "root"
    (root / "jobs" / "j1").mkdir(parents=True)
    session = remote_exec.RemoteSession({"transport": "local", "remote_root": str(root)})
    with pytest.raises(remote_exec.CliError):
        session.fetch_dir("jobs/j1", tmp_path / "dest-a", patterns=["-P"])
    with pytest.raises(remote_exec.CliError):
        session.fetch_dir("jobs/j1", tmp_path / "dest-b", patterns=["missing.txt"])
