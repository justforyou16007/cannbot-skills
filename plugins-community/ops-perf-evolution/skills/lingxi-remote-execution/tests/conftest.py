#!/usr/bin/env python3
"""lingxi-remote-execution 测试公共设施。

测试不需要真实 SSH / NPU：
- worker 测试直接在本地临时目录运行 remote_worker.py（远端行为的精确模拟，
  因为远端跑的就是同一个脚本）；
- 传输层测试用“假 ssh”（把最后一个参数交给本地 bash -c 执行，忠实模拟
  ssh 客户端“拼接参数 + 远端 shell 解析”的行为）；
- NPU 通过 pool 配置显式声明设备列表，npu-smi / cmake / make 等用假二进制。
"""

import base64
import json
import os
import subprocess
import sys
import time
import shutil
import uuid
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
WORKER = SCRIPTS_DIR / "remote_worker.py"
REMOTE_EXEC = SCRIPTS_DIR / "remote_exec.py"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# worker 调用助手
# ---------------------------------------------------------------------------

def run_worker(root, *args, timeout=120):
    """以子进程方式调用 remote_worker.py（覆盖 CLI 解析与 JSON 输出协议）。"""
    proc = subprocess.run(
        [sys.executable, str(WORKER), "--root", str(root)] + [str(a) for a in args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    out = proc.stdout.decode("utf-8", "replace").strip()
    payload = None
    if out:
        try:
            payload = json.loads(out.splitlines()[-1])
        except ValueError:
            payload = None
    return proc.returncode, payload, proc.stderr.decode("utf-8", "replace")


def write_pool_config(root, devices=(0, 1), max_build_slots=2, env_script=None):
    pool_dir = Path(root) / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    (pool_dir / "build_slots").mkdir(parents=True, exist_ok=True)
    cfg = {"devices": list(devices), "max_build_slots": max_build_slots,
           "env_script": env_script}
    (pool_dir / "pool_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return cfg


@pytest.fixture
def worker_root(tmp_path):
    root = tmp_path / "remote_root"
    root.mkdir()
    write_pool_config(root, devices=(0, 1, 2, 3), max_build_slots=2)
    return root


def acquire_lease(root, session, owner=None, ttl=600.0, timeout=10.0, device=None):
    args = ["lease-acquire", "--session", session, "--owner", owner or session,
            "--ttl", str(ttl), "--timeout", str(timeout)]
    if device is not None:
        args += ["--device", str(device)]
    rc, payload, _ = run_worker(root, *args)
    assert rc == 0, "lease acquire failed: %s" % payload
    return payload["lease"]


def submit_job(root, spec, expect_rc=0):
    b64 = base64.b64encode(json.dumps(spec).encode("utf-8")).decode("ascii")
    rc, payload, err = run_worker(root, "job-submit", "--spec-b64", b64)
    if expect_rc is not None:
        assert rc == expect_rc, "job-submit rc=%s payload=%s err=%s" % (rc, payload, err)
    return rc, payload


def stage_source(root, spec, source_dir):
    """Stage a source tree exactly as remote_exec does, for direct worker tests."""
    import remote_worker
    digest = remote_worker.tree_sha256(Path(source_dir))
    token = uuid.uuid4().hex
    rc, payload, err = run_worker(
        root, "upload-prepare", "--job-id", spec["job_id"],
        "--source-digest", digest, "--upload-token", token)
    assert rc == 0, (payload, err)
    bundle = Path(root) / payload["bundle_rel"]
    shutil.copytree(str(source_dir), str(bundle), dirs_exist_ok=True)
    spec["source_digest"] = digest
    spec["source_upload"] = payload["upload_rel"]
    return spec


def wait_terminal(root, job_id, timeout=60):
    deadline = time.monotonic() + timeout
    state = None
    while time.monotonic() < deadline:
        rc, payload, _ = run_worker(root, "job-status", "--job-id", job_id)
        assert rc == 0, payload
        state = payload["state"]
        if state.get("status") in ("succeeded", "failed", "cancelled"):
            return state
        time.sleep(0.2)
    raise AssertionError("job %s not terminal in %ss: %s" % (job_id, timeout, state))


def simple_spec(job_id, session, stages, artifacts=None):
    return {"job_id": job_id, "session_id": session, "stages": stages,
            "artifacts": artifacts or ["*.txt"]}


def cmd_stage(name, argv, needs_device=False, failure_class="build",
              env=None, cwd="", timeout_s=60):
    return {"name": name, "kind": "cmd", "commands": [argv],
            "needs_device": needs_device, "failure_class": failure_class,
            "env": env or {}, "cwd": cwd, "timeout_s": timeout_s}


# ---------------------------------------------------------------------------
# 假二进制 / 假脚本
# ---------------------------------------------------------------------------

def make_fake_bin(tmp_path, name, body):
    """创建假可执行文件；body 是 bash 脚本内容。"""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def make_fake_npu_smi(tmp_path, devices=(0, 1)):
    mapping = ["NPU ID    Chip ID    Chip Logic ID    Chip Name"]
    for d in devices:
        mapping.append("%d        0          %d                Ascend 910B3" % (d, d))
        mapping.append("%d        1          -                Mcu" % d)
    body = 'if [ "$1" = "info" ] && [ "$2" = "-m" ]; then\n'
    body += 'cat <<EOF\n%s\nEOF\n' % "\n".join(mapping)
    body += 'else\nexit 1\nfi\n'
    return make_fake_bin(tmp_path, "npu-smi", body)


def make_fake_env_script(tmp_path, ascend_home=None):
    path = tmp_path / "fake_env.sh"
    ascend_home = ascend_home or str(tmp_path / "fake_ascend")
    path.write_text("#!/usr/bin/env bash\nexport ASCEND_HOME_PATH=%s\n" % ascend_home,
                    encoding="utf-8")
    return path


def make_minimal_wheel(dest_dir, name="canarypkg", version="1.0"):
    """构造一个结构合法的最小 wheel（无第三方依赖），用于 pip --target 测试。"""
    import zipfile
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = dest_dir / ("%s-%s-py3-none-any.whl" % (name, version))
    dist_info = "%s-%s.dist-info" % (name, version)
    with zipfile.ZipFile(str(wheel_path), "w") as zf:
        zf.writestr("%s/__init__.py" % name, "VALUE = '%s- ok'\n" % name)
        zf.writestr("%s/METADATA" % dist_info,
                    "Metadata-Version: 2.1\nName: %s\nVersion: %s\n" % (name, version))
        zf.writestr("%s/WHEEL" % dist_info,
                    "Wheel-Version: 1.0\nGenerator: lingxi-test\nRoot-Is-Purelib: true\n"
                    "Tag: py3-none-any\n")
        zf.writestr("%s/RECORD" % dist_info,
                    "%s/__init__.py,,\n%s/METADATA,,\n%s/WHEEL,,\n%s/RECORD,,\n"
                    % (name, dist_info, dist_info, dist_info))
    return wheel_path


# ---------------------------------------------------------------------------
# 假 ssh：模拟“拼接参数 + 远端 shell 解析”的真实行为
# ---------------------------------------------------------------------------

def make_fake_ssh(tmp_path, log_file=None):
    """假 ssh：把最后一个参数（远端命令字符串）交给本地 bash -c 执行。

    与真实 ssh 客户端行为一致：argv 拼接成一条命令、由 shell 解析，
    因此能真实检验我们的逐参数 shlex.quote 是否安全。
    """
    body = "#!/usr/bin/env bash\n"
    if log_file:
        body += 'printf "%s\\n" "${@: -1}" >> %s\n' % log_file
    body += 'exec /usr/bin/env bash -c "${@: -1}"\n'
    return make_fake_bin(tmp_path, "fakessh", body)


def write_host_config(tmp_path, ssh_bin=None, transport="ssh", devices=(0, 1),
                      env_script=None, remote_root=None, host="npu-a"):
    cfg_path = Path(tmp_path) / "remote_hosts.json"
    entry = {
        "transport": transport,
        "ssh_host": host,
        "remote_root": str(remote_root or (Path(tmp_path) / "remote_root")),
        "devices": list(devices),
        "max_build_slots": 2,
    }
    if ssh_bin:
        entry["ssh_bin"] = str(ssh_bin)
    if env_script:
        entry["env_script"] = str(env_script)
    cfg_path.write_text(json.dumps({"hosts": {host: entry}}), encoding="utf-8")
    return cfg_path


def run_cli(*args, timeout=180, cwd=None):
    proc = subprocess.run(
        [sys.executable, str(REMOTE_EXEC)] + [str(a) for a in args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, cwd=cwd)
    out = proc.stdout.decode("utf-8", "replace").strip()
    payload = None
    if out:
        try:
            payload = json.loads(out.splitlines()[-1])
        except ValueError:
            payload = None
    return proc.returncode, payload, proc.stderr.decode("utf-8", "replace")
