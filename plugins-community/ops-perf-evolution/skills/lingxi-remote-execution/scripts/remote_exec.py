#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
"""
lingxi-remote-execution 本地 CLI。

本地负责进化编排 / 世界模型 / 代码修改 / 报告；本 CLI 只把确定性的
build、verify、profile 三个阶段提交到一台远程 NPU 主机执行，并把结果
取回到与本地执行相同的产物契约（logs / artifacts / state.json）。

安全边界：
- 配置只接收 ssh alias/host 与参数，不接收口令或私钥内容，认证依赖
  用户现有的 SSH 配置（~/.ssh/config、agent 等）。
- 所有远端命令以 argv 数组构造，经 shlex.quote 逐参数引用后交给 ssh，
  不拼接未校验的 shell 字符串；job spec 走 base64 通道。
- 文件传输用 tar 流（stdin/stdout 管道），远端解包路径经过包含性校验。
- 失败分类：transport / environment / build / correctness / profile /
  cancelled / lease_lost（本地状态文件保存 host/session/lease/device/job，
  支持断线重连后的 recover）。

典型用法（详见 SKILL.md 与 references/）：
    python3 remote_exec.py --config remote_hosts.json --host npu-a deploy
    python3 remote_exec.py --host npu-a preflight
    python3 remote_exec.py --host npu-a lease acquire --session s1 --owner FastGELU
    python3 remote_exec.py --host npu-a job submit --job-id s1_base \\
        --session s1 --src /path/to/kernel_proj \\
        --preset ascendc-eval --kernel-dir kernel --soc Ascend910B3 \\
        --verify-cmd '["python3","scripts/verify.py","."]' --wait --fetch \\
        --fetch-dest ./out/s1_base
    python3 remote_exec.py --host npu-a lease release --lease-id <id>
"""

import argparse
import base64
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "remote_worker.py"

# 允许出现在 ssh 目标 / 远端路径里的字符集（拒绝空白、引号、元字符，
# 从源头杜绝配置注入，而不是事后转义）
_SSH_HOST_RE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@._-")
_REMOTE_PATH_RE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-")


def log(msg):
    sys.stderr.write("[lingxi-remote] %s\n" % msg)
    sys.stderr.flush()


def emit(obj):
    sys.stdout.write(json.dumps(obj, sort_keys=True, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()


class CliError(Exception):
    def __init__(self, code, message):
        super(CliError, self).__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATHS = ("remote_hosts.json",)


def validate_ssh_host(host):
    if not host or host.startswith("-") or not all(c in _SSH_HOST_RE_CHARS for c in host):
        raise CliError(2, "unsafe ssh host in config: %r" % host)
    return host


def validate_remote_path(path):
    if not path or not path.startswith("/") or not all(c in _REMOTE_PATH_RE_CHARS for c in path):
        raise CliError(2, "unsafe remote path in config: %r" % path)
    if "//" in path or "/../" in path or path.endswith("/.."):
        raise CliError(2, "unsafe remote path in config: %r" % path)
    return path


def validate_remote_relpath(rel):
    """远端相对路径（限制在 remote_root 内，本地侧先做一遍包含性校验）。"""
    if not isinstance(rel, str) or not rel:
        raise CliError(2, "empty remote relative path")
    if rel.startswith(("/", "~", "-")) or "\\" in rel:
        raise CliError(2, "unsafe remote relative path: %r" % rel)
    comps = rel.split("/")
    if any(c in ("", ".", "..") for c in comps):
        raise CliError(2, "unsafe remote relative path: %r" % rel)
    if not all(all(ch in _REMOTE_PATH_RE_CHARS or ch == "*" for ch in c) for c in comps):
        raise CliError(2, "unsafe characters in remote relative path: %r" % rel)
    return rel


def _walk_source_entries(source, prefix=""):
    """Yield deterministic source entries and reject links/special files."""
    source = Path(source)
    if not source.is_dir() or source.is_symlink():
        raise CliError(2, "source must be a real directory: %s" % source)
    for current, dirs, files in os.walk(str(source), topdown=True, followlinks=False):
        dirs.sort()
        files.sort()
        current_path = Path(current)
        rel_parent = current_path.relative_to(source).as_posix()
        if rel_parent == ".":
            rel_parent = ""
        for name in dirs:
            path = current_path / name
            if path.is_symlink():
                raise CliError(2, "source contains a symlink: %s" % path)
            rel = "/".join(p for p in (prefix, rel_parent, name) if p)
            yield "dir", path, rel
        for name in files:
            path = current_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise CliError(2, "source contains a non-regular file: %s" % path)
            rel = "/".join(p for p in (prefix, rel_parent, name) if p)
            yield "file", path, rel


def source_bundle_sha256(sources):
    """Hash the exact virtual bundle layout used by repeated --src arguments."""
    digest = hashlib.sha256()
    entries = []
    for index, source in enumerate(sources):
        prefix = "" if index == 0 else "extra_%d" % (index - 1)
        if prefix:
            entries.append((prefix, "dir", Path(source)))
        for kind, path, rel in _walk_source_entries(source, prefix):
            entries.append((rel, kind, path))
    for rel, kind, path in sorted(entries, key=lambda item: (item[0], item[1])):
        if kind == "dir":
            digest.update(("D\0%s\0" % rel).encode("utf-8"))
            continue
        size = path.stat().st_size
        digest.update(("F\0%s\0%d\0" % (rel, size)).encode("utf-8"))
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def load_config(path):
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise CliError(2, "config not found: %s (see references/config-schema.md)" % path)
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise CliError(2, "config is not valid JSON: %s" % exc)
    hosts = cfg.get("hosts")
    if not isinstance(hosts, dict) or not hosts:
        raise CliError(2, "config must contain a non-empty 'hosts' object")
    for alias, entry in hosts.items():
        if not isinstance(entry, dict):
            raise CliError(2, "host entry %r must be an object" % alias)
        transport = entry.get("transport", "ssh")
        if transport not in ("ssh", "local"):
            raise CliError(2, "host %r: transport must be 'ssh' or 'local'" % alias)
        if transport == "ssh":
            validate_ssh_host(entry.get("ssh_host", ""))
        else:
            entry.setdefault("ssh_host", "local")
        validate_remote_path(entry.get("remote_root", ""))
        env_script = entry.get("env_script")
        if env_script:
            if not str(env_script).startswith("/") or "\\" in str(env_script):
                raise CliError(2, "host %r: env_script must be an absolute remote path" % alias)
        devices = entry.get("devices", "auto")
        if devices != "auto":
            if not isinstance(devices, list) or not devices or \
                    not all(isinstance(d, int) and d >= 0 for d in devices):
                raise CliError(2, "host %r: devices must be 'auto' or a list of ints" % alias)
            if len(set(devices)) != len(devices):
                raise CliError(2, "host %r: devices must not contain duplicates" % alias)
        slots = entry.get("max_build_slots", 4)
        if not isinstance(slots, int) or slots < 1:
            raise CliError(2, "host %r: max_build_slots must be a positive int" % alias)
        required_paths = entry.get("required_paths", [])
        if not isinstance(required_paths, list):
            raise CliError(2, "host %r: required_paths must be a list" % alias)
        for required_path in required_paths:
            validate_remote_path(required_path)
        ssh_args = entry.get("ssh_args", [])
        if not isinstance(ssh_args, list) or not all(isinstance(a, str) and a for a in ssh_args):
            raise CliError(2, "host %r: ssh_args must be a list of strings" % alias)
        for arg in ssh_args:
            if any(c in " \t\n\r\x00;|&<>`$'\"\\" for c in arg):
                raise CliError(2, "host %r: unsafe ssh_args entry %r" % (alias, arg))
    return cfg


def resolve_host_cfg(config, host):
    hosts = config["hosts"]
    if host not in hosts:
        raise CliError(2, "host %r not in config (available: %s)" % (host, ", ".join(sorted(hosts))))
    return hosts[host]


# ---------------------------------------------------------------------------
# 传输层
# ---------------------------------------------------------------------------

class Transport(object):
    """把 argv 命令送到“远端”执行。ssh 模式下逐参数 shlex.quote，
    ssh 客户端会把参数拼接后交给远端 shell，预引用保证 argv 原样还原。"""

    def __init__(self, entry):
        self.entry = entry
        self.transport = entry.get("transport", "ssh")

    def _prefix(self):
        if self.transport == "local":
            return []
        ssh_bin = entry_ssh_bin(self.entry)
        return [ssh_bin] + list(self.entry.get("ssh_args", [])) + [self.entry["ssh_host"]]

    def run(self, argv, timeout=120, input_bytes=None):
        if self.transport == "local":
            cmd = list(argv)
        else:
            for arg in argv:
                if "\x00" in arg:
                    raise CliError(2, "NUL byte in remote argv")
            remote_cmd = " ".join(shlex.quote(a) for a in argv)
            cmd = self._prefix() + [remote_cmd]
        try:
            result = subprocess.run(
                cmd, input=input_bytes,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=timeout)
        except FileNotFoundError:
            raise CliError(7, "transport failure: cannot execute %r" % cmd[0])
        except subprocess.TimeoutExpired:
            raise CliError(7, "transport failure: timeout after %ss" % timeout)
        return result.returncode, result.stdout, result.stderr

    def popen(self, argv, stdin_pipe=False):
        if self.transport == "local":
            cmd = list(argv)
        else:
            remote_cmd = " ".join(shlex.quote(a) for a in argv)
            cmd = self._prefix() + [remote_cmd]
        kwargs = {}
        if stdin_pipe:
            kwargs["stdin"] = subprocess.PIPE
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def entry_ssh_bin(entry):
    return str(entry.get("ssh_bin", "ssh"))


class RemoteSession(object):
    """一个 host 配置上的远端 worker 会话（无状态，每次调用都是新进程）。"""

    def __init__(self, entry):
        self.entry = entry
        self.transport = Transport(entry)
        self.root = entry["remote_root"]
        self.remote_python = str(entry.get("remote_python", "python3"))
        self.worker_path = self.root + "/lib/remote_worker.py"

    def worker_argv(self, command_args):
        return [self.remote_python, self.worker_path, "--root", self.root] + list(command_args)

    def call(self, command_args, timeout=120, input_bytes=None):
        """调用远端 worker 子命令，解析其 JSON 输出。

        返回 (ok, payload)。transport 层失败映射为 failure_class=transport。
        """
        argv = self.worker_argv(command_args)
        rc, out, err = self.transport.run(argv, timeout=timeout, input_bytes=input_bytes)
        text = out.decode("utf-8", "replace").strip()
        payload = None
        if text:
            try:
                payload = json.loads(text.splitlines()[-1])
            except ValueError:
                payload = None
        if payload is None:
            raise CliError(7, "transport failure (rc=%d): %s"
                           % (rc, err.decode("utf-8", "replace").strip()[:500]))
        return payload.get("ok", False), payload

    def deploy(self, max_build_slots, devices, env_script, required_paths=None):
        """推送 worker 脚本到 remote_root/lib/ 并写入 pool 配置。"""
        script_bytes = WORKER_SCRIPT.read_bytes()
        # 固定的远端命令模板；remote_root 已通过配置校验（绝对路径、安全字符集）
        # Multiple lingxi-evo sessions may preflight the same host together.
        # Write a per-process temporary file and rename atomically so no caller
        # can observe a half-written worker.
        remote_cmd = [
            "bash", "-c",
            'mkdir -p "$0/lib"; '
            'tmp="$0/lib/.remote_worker.py.$$.tmp"; '
            'trap \'rm -f "$tmp"\' EXIT; '
            'cat > "$tmp" && chmod 755 "$tmp" && mv -f "$tmp" "$0/lib/remote_worker.py"',
            self.root,
        ]
        rc, out, err = self.transport.run(remote_cmd, timeout=120, input_bytes=script_bytes)
        if rc != 0:
            raise CliError(7, "deploy failed (rc=%d): %s"
                           % (rc, err.decode("utf-8", "replace").strip()[:500]))
        args = ["preflight", "--write-config",
                "--max-build-slots", str(max_build_slots)]
        if devices and devices != "auto":
            args += ["--devices", ",".join(str(d) for d in devices)]
        if env_script:
            args += ["--env-script", env_script]
        for required_path in required_paths or []:
            args += ["--required-path", required_path]
        ok, payload = self.call(args, timeout=180)
        return payload

    # -- 文件传输（tar 流） --
    def push_dir(self, local_dir, remote_rel_dir):
        """把本地目录内容打包推到 remote_root/<remote_rel_dir>/ 下。"""
        validate_remote_relpath(remote_rel_dir)
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise CliError(2, "local src not found: %s" % local_dir)
        remote_dest = self.root + "/" + remote_rel_dir
        remote_cmd = ["bash", "-c", 'mkdir -p "$0" && tar -C "$0" -xf -', remote_dest]
        proc = self.transport.popen(remote_cmd, stdin_pipe=True)
        try:
            with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
                for kind, path, rel in _walk_source_entries(local_dir):
                    info = tar.gettarinfo(str(path), arcname=rel)
                    if kind == "dir":
                        tar.addfile(info)
                    else:
                        with open(path, "rb") as fh:
                            tar.addfile(info, fh)
            proc.stdin.close()
        except BrokenPipeError:
            pass
        except Exception:
            proc.kill()
            proc.wait(timeout=30)
            raise
        _out, err = proc.communicate(timeout=300)
        if proc.returncode != 0:
            raise CliError(7, "push failed (rc=%d): %s"
                           % (proc.returncode, err.decode("utf-8", "replace").strip()[:500]))
        return {"pushed": str(local_dir), "remote": remote_dest}

    def fetch_dir(self, remote_rel_dir, local_dest, patterns=None):
        """把 remote_root/<remote_rel_dir>/ 下（可按 pattern 过滤）拉回本地。"""
        validate_remote_relpath(remote_rel_dir)
        base = self.root + "/" + remote_rel_dir
        # 注意：不能同时传 "." 和 pattern——会产生重复的硬链接条目，
        # 流式解包时触发 backward seek。二选一。
        if patterns:
            for pat in patterns:
                validate_remote_relpath(pat)
            remote_cmd = ["tar", "-C", base, "-cf", "-", "--"] + list(patterns)
        else:
            remote_cmd = ["tar", "-C", base, "-cf", "-", "--", "."]
        proc = self.transport.popen(remote_cmd)
        local_dest = Path(local_dest)
        if local_dest.is_symlink():
            proc.kill()
            proc.wait(timeout=30)
            raise CliError(2, "fetch destination must not be a symlink: %s" % local_dest)
        local_dest.mkdir(parents=True, exist_ok=True)
        dest_root = local_dest.resolve()

        def safe_target(member_name):
            name = member_name
            while name.startswith("./"):
                name = name[2:]
            if name in ("", "."):
                return None
            if name.startswith(("/", "~", "-")) or "\\" in name or "\x00" in name:
                raise CliError(7, "unsafe member in remote archive: %r" % member_name)
            comps = name.split("/")
            if any(c in ("", ".", "..") for c in comps):
                raise CliError(7, "unsafe member in remote archive: %r" % member_name)
            target = dest_root.joinpath(*comps)
            # Existing symlink parents could redirect an otherwise safe member.
            cur = dest_root
            for comp in comps[:-1]:
                cur = cur / comp
                if cur.is_symlink():
                    raise CliError(7, "archive member traverses local symlink: %r" % member_name)
            if target.is_symlink():
                raise CliError(7, "archive member replaces local symlink: %r" % member_name)
            real_parent = target.parent.resolve()
            if real_parent != dest_root and dest_root not in real_parent.parents:
                raise CliError(7, "archive member escapes destination: %r" % member_name)
            return target

        extracted = []
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                for member in tar:
                    target = safe_target(member.name)
                    if target is None:
                        continue
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        source = tar.extractfile(member)
                        if source is None:
                            raise CliError(7, "cannot read remote archive member: %r" % member.name)
                        temp_target = target.with_name(
                            target.name + ".remote-tmp-" + uuid.uuid4().hex)
                        try:
                            with source, open(temp_target, "xb") as output:
                                shutil.copyfileobj(source, output, length=1024 * 1024)
                            os.replace(str(temp_target), str(target))
                        finally:
                            try:
                                if temp_target.exists():
                                    temp_target.unlink()
                            except OSError:
                                pass
                        try:
                            os.chmod(str(target), member.mode & 0o777)
                        except OSError:
                            pass
                    else:
                        raise CliError(7, "remote archive contains link/special file: %r"
                                       % member.name)
                    extracted.append(member.name)
            proc.stdout.close()
            rc = proc.wait(timeout=60)
            err = proc.stderr.read().decode("utf-8", "replace")
            if rc != 0:
                raise CliError(7, "fetch failed (rc=%d): %s" % (rc, err.strip()[:500]))
        except CliError:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            raise
        except Exception as exc:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            raise CliError(7, "invalid or incomplete remote archive: %s" % exc)
        return {"fetched_to": str(local_dest), "remote": remote_rel_dir,
                "files": extracted}


# ---------------------------------------------------------------------------
# 本地 session 状态（断线重连 / recover 的依据）
# ---------------------------------------------------------------------------

def _read_local_state_unlocked(path):
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise CliError(2, "local state file is invalid JSON; refusing to overwrite it: %s" % exc)


@contextmanager
def _local_state_lock(path):
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def load_local_state(path):
    with _local_state_lock(path):
        return _read_local_state_unlocked(path)


def _write_local_state_unlocked(path, state):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.%d.%s" % (os.getpid(), uuid.uuid4().hex))
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(tmp), str(p))


def save_local_state(path, state):
    with _local_state_lock(path):
        _write_local_state_unlocked(path, state)


def _mutate_local_state(path, mutator):
    with _local_state_lock(path):
        state = _read_local_state_unlocked(path)
        mutator(state)
        _write_local_state_unlocked(path, state)
        return state


def record_session(state_file, host, session_id, **fields):
    def _mut(state):
        sessions = state.setdefault("sessions", {})
        entry = sessions.setdefault(session_id, {"host": host})
        entry.setdefault("host", host)
        entry.update(fields)
        entry["updated_ts"] = time.time()
    _mutate_local_state(state_file, _mut)


def record_job(state_file, session_id, job_id, **fields):
    def _mut(state):
        sessions = state.setdefault("sessions", {})
        entry = sessions.setdefault(session_id, {})
        jobs = entry.setdefault("jobs", {})
        job = jobs.setdefault(job_id, {})
        job.update(fields)
        job["updated_ts"] = time.time()
    _mutate_local_state(state_file, _mut)


def mark_lease_released(state_file, lease_id):
    def _mut(state):
        for entry in (state.get("sessions", {}) or {}).values():
            if entry.get("lease_id") == lease_id:
                entry["lease_active"] = False
                entry["released_ts"] = time.time()
                entry["updated_ts"] = time.time()
    _mutate_local_state(state_file, _mut)


# ---------------------------------------------------------------------------
# job spec 构造（preset：ascendc-eval）
# ---------------------------------------------------------------------------

def parse_json_arg(text, what):
    try:
        return json.loads(text)
    except ValueError as exc:
        raise CliError(2, "--%s is not valid JSON (%s)" % (what, exc))


def validate_cmd_argv(argv, what):
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a for a in argv):
        raise CliError(2, "%s must be a non-empty JSON argv array of strings" % what)
    return argv


def build_ascendc_spec(args):
    """把原先 evaluate_ascendc.sh 的“构建+全局 pip+验证”拆成隔离的远端阶段。

    - build：cmake+make（不占设备锁，受 build slot 约束）
    - wheel + install：bdist_wheel 后 pip --target 安装到 job 私有 pylibs/
    - verify：全量精度验证（设备阶段，串行排队，PYTHONPATH 指向私有 pylibs）
    - profile：msprof 采集（设备阶段）
    """
    kernel_dir = args.kernel_dir.strip("/")
    if not kernel_dir or any(c in ("", "..", ".") for c in kernel_dir.split("/")):
        raise CliError(2, "--kernel-dir must be a relative path inside the bundle")
    if not all(c.isalnum() or c in "._-/" for c in kernel_dir):
        raise CliError(2, "--kernel-dir contains unsafe characters")
    soc = args.soc
    if not soc or not all(c.isalnum() or c in "._-" for c in soc):
        raise CliError(2, "--soc contains unsafe characters")
    if not isinstance(args.build_type, str) or not args.build_type or not all(
            c.isalnum() or c in "._-" for c in args.build_type):
        raise CliError(2, "--build-type contains unsafe characters")
    try:
        make_jobs = int(args.make_jobs)
    except (TypeError, ValueError):
        raise CliError(2, "--make-jobs must be an integer")
    if make_jobs < 1 or make_jobs > 256:
        raise CliError(2, "--make-jobs must be between 1 and 256")
    verify_cmd = None
    if args.verify_cmd is not None:
        verify_cmd = validate_cmd_argv(parse_json_arg(args.verify_cmd, "verify-cmd"), "--verify-cmd")
    stages = [
        {
            "name": "build",
            "kind": "cmd",
            "needs_device": False,
            "failure_class": "build",
            "commands": [
                ["cmake", "{bundle_dir}/%s" % kernel_dir,
                 "-DSOC_VERSION=%s" % soc,
                 "-DASCEND_CANN_PACKAGE_PATH={ascend_home_path}",
                 "-DCMAKE_BUILD_TYPE=%s" % args.build_type],
                ["make", "-j%s" % make_jobs],
            ],
            "cwd": "bundle/%s/build" % kernel_dir,
            "timeout_s": args.stage_timeout,
        },
        {
            "name": "wheel",
            "kind": "cmd",
            "needs_device": False,
            "failure_class": "build",
            "commands": [["{python}", "setup.py", "bdist_wheel"]],
            "cwd": "bundle/%s" % kernel_dir,
            "timeout_s": args.stage_timeout,
        },
        {
            "name": "install",
            "kind": "pip_install",
            "needs_device": False,
            "failure_class": "build",
            "wheel": "bundle/%s/dist/*.whl" % kernel_dir,
            "target": "pylibs",
            "timeout_s": args.stage_timeout,
        },
    ]
    if verify_cmd is not None:
        stages.append({
            "name": "verify",
            "kind": "cmd",
            "needs_device": True,
            "failure_class": "correctness",
            "commands": [verify_cmd],
            "env": {"PYTHONPATH": "{pylibs_dir}:{bundle_dir}"},
            "cwd": "bundle",
            "timeout_s": args.stage_timeout,
        })
    profile_collects_raw = False
    if args.profile_cmd:
        profile_cmd = validate_cmd_argv(parse_json_arg(args.profile_cmd, "profile-cmd"),
                                        "--profile-cmd")
        # msprof_perf_summary --compare writes PROF_GROUP data below /tmp and
        # deletes it unless --keep-prof is present.  Preserve it, then invoke
        # the worker's constrained collector so the evidence becomes a normal
        # job artifact under bundle/profiling/.
        if (any(os.path.basename(part) == "msprof_perf_summary.py" for part in profile_cmd)
                and "--compare" in profile_cmd):
            profile_collects_raw = True
            if "--keep-prof" not in profile_cmd:
                profile_cmd = profile_cmd + ["--keep-prof"]
        stages.append({
            "name": "profile",
            "kind": "cmd",
            "needs_device": True,
            "failure_class": "profile",
            "commands": [profile_cmd],
            "env": {"PYTHONPATH": "{pylibs_dir}:{bundle_dir}"},
            "cwd": "bundle",
            "timeout_s": args.stage_timeout,
        })
        if profile_collects_raw:
            # Copying potentially large profiler directories does not need the
            # card, so release the device lock before this stage.
            stages.append({
                "name": "profile_collect",
                "kind": "cmd",
                "needs_device": False,
                "failure_class": "profile",
                "commands": [[
                    "{python}", "{root}/lib/remote_worker.py", "--root", "{root}",
                    "_profile-collect", "--job-id", args.job_id,
                ]],
                "cwd": "bundle",
                "timeout_s": args.stage_timeout,
            })
    artifacts = []
    for pat in args.artifacts:
        if pat == "logs" or pat.startswith(("logs/", "bundle/")):
            artifacts.append(pat)
        else:
            artifacts.append("bundle/" + pat)
    if profile_collects_raw and "bundle/profiling" not in artifacts:
        artifacts.append("bundle/profiling")
    return {
        "job_id": args.job_id,
        "session_id": args.session,
        "stages": stages,
        "artifacts": artifacts,
    }


def build_spec_from_stages_file(args):
    spec = parse_json_arg(Path(args.stage_json).read_text(encoding="utf-8"), "--stage-json")
    if not isinstance(spec, dict):
        raise CliError(2, "--stage-json must contain a JSON object")
    spec.setdefault("job_id", args.job_id)
    spec.setdefault("session_id", args.session)
    if not isinstance(spec.get("artifacts"), list):
        spec["artifacts"] = args.artifacts
    return spec


# ---------------------------------------------------------------------------
# job 提交 / 轮询 / 拉取
# ---------------------------------------------------------------------------

class ClientLeaseHeartbeat(object):
    """Refresh a lease while this CLI process uploads, waits, and fetches."""

    def __init__(self, session, lease_id, ttl_s):
        self.session = session
        self.lease_id = lease_id
        self.interval = max(1.0, min(30.0, float(ttl_s) / 3.0))
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run,
                                       name="client-lease-heartbeat-%s" % lease_id[:8])
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                ok, payload = self.session.call(
                    ["lease-heartbeat", "--lease-id", self.lease_id], timeout=60)
                if not ok:
                    self.error = payload.get("error", "lease heartbeat failed")
                    return
            except CliError as exc:
                self.error = str(exc)
                return

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=max(2.0, self.interval * 2))


def _heartbeat_for_session(session, args):
    state = load_local_state(args.state_file)
    entry = (state.get("sessions", {}) or {}).get(args.session, {})
    lease_id = entry.get("lease_id")
    if not lease_id or entry.get("lease_active") is False:
        return None
    return ClientLeaseHeartbeat(session, lease_id, entry.get("ttl_s", 3600.0)).start()


def _job_sources(args):
    raw_src = getattr(args, "src", None)
    if isinstance(raw_src, str):
        sources = [raw_src]
    else:
        sources = list(raw_src or [])
    raw_srcs = getattr(args, "srcs", None) or []
    if isinstance(raw_srcs, str):
        raw_srcs = [raw_srcs]
    for source in raw_srcs:
        if source not in sources:
            sources.append(source)
    return sources


def prepare_job_sources(session, args, spec):
    """Upload sources to an isolated staging dir and bind their digest to spec."""
    sources = _job_sources(args)
    if not sources:
        if not args.stage_json:
            raise CliError(2, "--preset ascendc-eval requires at least one --src directory")
        return spec
    if not args.stage_json:
        kernel_path = Path(sources[0]) / args.kernel_dir
        if not kernel_path.is_dir():
            raise CliError(2, "kernel directory not found inside first --src: %s" % kernel_path)
    digest = source_bundle_sha256(sources)
    token = uuid.uuid4().hex
    ok, payload = session.call(
        ["upload-prepare", "--job-id", args.job_id,
         "--source-digest", digest, "--upload-token", token], timeout=120)
    if not ok:
        raise CliError(payload.get("code", 6),
                       "cannot prepare source upload: %s" % payload.get("error"))
    upload_rel = payload["upload_rel"]
    try:
        for index, source in enumerate(sources):
            suffix = "bundle" if index == 0 else "bundle/extra_%d" % (index - 1)
            session.push_dir(source, upload_rel + "/" + suffix)
    except Exception:
        try:
            session.call(["upload-discard", "--job-id", args.job_id,
                          "--source-digest", digest, "--upload-token", token], timeout=60)
        except CliError:
            pass
        raise
    spec["source_digest"] = digest
    spec["source_upload"] = upload_rel
    return spec

def do_submit(session, args):
    if args.stage_json:
        spec = build_spec_from_stages_file(args)
    elif args.preset == "ascendc-eval":
        spec = build_ascendc_spec(args)
    else:
        raise CliError(2, "provide --preset ascendc-eval or --stage-json <file>")
    heartbeat = _heartbeat_for_session(session, args)
    try:
        spec = prepare_job_sources(session, args, spec)
        spec_b64 = base64.b64encode(
            json.dumps(spec, sort_keys=True).encode("utf-8")).decode("ascii")

        ok, payload = session.call(["job-submit", "--spec-b64", spec_b64], timeout=120)
        if not ok:
            raise CliError(payload.get("code", 6), "job-submit failed: %s" % payload.get("error"))
        job_id = payload["job_id"]

        record_job(args.state_file, args.session, job_id,
                   status=payload.get("status"), host=args.host,
                   remote_root=session.root, source_digest=spec.get("source_digest"))
        record_session(args.state_file, args.host, args.session)

        terminal_status = None
        if args.wait:
            state = wait_terminal(session, job_id, timeout=args.wait_timeout,
                                  poll=args.poll_interval)
            payload["state"] = state
            terminal_status = state.get("status")
            record_job(args.state_file, args.session, job_id,
                       status=terminal_status, failure_class=state.get("failure_class"))
            if args.fetch:
                fetched = fetch_job(session, args, job_id)
                payload["fetched"] = fetched
        if heartbeat is not None and heartbeat.error:
            payload["lease_heartbeat_error"] = heartbeat.error
        emit(payload)
        return 0 if terminal_status in (None, "succeeded") else 1
    finally:
        if heartbeat is not None:
            heartbeat.stop()


def wait_terminal(session, job_id, timeout=7200, poll=10):
    deadline = time.monotonic() + timeout
    last = None
    while True:
        ok, payload = session.call(["job-status", "--job-id", job_id], timeout=60)
        if not ok:
            raise CliError(payload.get("code", 6),
                           "job-status failed: %s" % payload.get("error"))
        last = payload.get("state", {})
        if last.get("status") in ("succeeded", "failed", "cancelled"):
            return last
        if time.monotonic() >= deadline:
            raise CliError(7, "wait timeout after %ss (job still %s)"
                           % (timeout, last.get("status")))
        time.sleep(max(1, poll))
    return last


def fetch_job(session, args, job_id):
    """拉取 job 结果。默认拉 state/spec/logs + 该 job 产物清单里的文件。"""
    patterns = None
    if not getattr(args, "fetch_full", False):
        patterns = ["state.json", "spec.json", "logs"]
        # 产物清单是唯一权威来源；不把未展开的用户 glob 交给 tar。
        try:
            ok, payload = session.call(["job-status", "--job-id", job_id], timeout=60)
            state = payload.get("state") or {}
            manifest = state.get("artifacts_manifest") or {}
            manifest_files = manifest.get("files") or []
        except CliError:
            manifest_files = []
        for f in manifest_files:
            if f not in patterns:
                patterns.append(f)
    dest = Path(args.fetch_dest) if getattr(args, "fetch_dest", None) else \
        Path(args.fetch_dir) / job_id
    if patterns is None:
        return session.fetch_dir("jobs/" + job_id, dest, patterns=None)
    bundle_files = [p[len("bundle/"):] for p in patterns if p.startswith("bundle/")]
    root_files = [p for p in patterns if not p.startswith("bundle/")]
    results = []
    if root_files:
        results.append(session.fetch_dir("jobs/" + job_id, dest, patterns=root_files))
    if bundle_files:
        # Strip the remote bundle/ prefix so performance.json, generated
        # binaries, etc. land in the same local paths as the local backend.
        results.append(session.fetch_dir("jobs/" + job_id + "/bundle", dest,
                                         patterns=bundle_files))
    return {"fetched_to": str(dest), "parts": results}


# ---------------------------------------------------------------------------
# batch：多算子薄编排（活跃算子数受设备池约束）
# ---------------------------------------------------------------------------

def run_batch(session, args, plan):
    sessions = plan.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise CliError(2, "plan must contain a non-empty 'sessions' list")
    seen = set()
    for index, entry in enumerate(sessions):
        if not isinstance(entry, dict):
            raise CliError(2, "plan session %d must be an object" % index)
        sid = entry.get("session_id")
        if not isinstance(sid, str) or not sid or not all(
                c.isalnum() or c in "._-" for c in sid):
            raise CliError(2, "plan session %d has invalid session_id" % index)
        if sid in seen:
            raise CliError(2, "duplicate session_id in plan: %s" % sid)
        seen.add(sid)
        if entry.get("stage_json") is None and not (entry.get("src") or entry.get("srcs")):
            raise CliError(2, "plan session %s needs stage_json or source directories" % sid)
    max_active = args.max_active or len(sessions)
    if max_active < 1:
        raise CliError(2, "--max-active must be positive or zero")
    sem = threading.Semaphore(max_active)
    results = {}
    lock = threading.Lock()

    def _one(entry):
        sid = entry["session_id"]
        with sem:
            lease = None
            heartbeat = None
            try:
                # 设备池是天然上限：lease acquire 拿不到卡就阻塞在这里，
                # 不可能通过给每个进程传不同 npu 绕过全局分配
                ok, payload = session.call(
                    ["lease-acquire", "--session", sid, "--owner", entry.get("owner", sid),
                     "--ttl", str(args.ttl), "--timeout", str(args.lease_timeout)],
                    timeout=args.lease_timeout + 120)
                if not ok:
                    raise CliError(payload.get("code", 6),
                                   "lease acquire failed: %s" % payload.get("error"))
                lease = payload["lease"]
                record_session(args.state_file, args.host, sid,
                               lease_id=lease["lease_id"], device_id=lease["device_id"],
                               ttl_s=lease.get("ttl_s", args.ttl), lease_active=True,
                               owner=entry.get("owner", sid))
                heartbeat = ClientLeaseHeartbeat(
                    session, lease["lease_id"], lease.get("ttl_s", args.ttl)).start()
                # plan 条目顶层字段即 job 参数（session_id/owner 除外）
                job_fields = {k: v for k, v in entry.items()
                              if k not in ("session_id", "owner")}
                job_args = argparse.Namespace(**job_fields)
                job_args.job_id = entry.get("job_id", sid + "_job1")
                job_args.session = sid
                job_args.host = args.host
                job_args.state_file = args.state_file
                job_args.wait = True
                job_args.fetch = True
                job_args.wait_timeout = args.wait_timeout
                job_args.poll_interval = args.poll_interval
                for name, default in (
                        ("stage_json", None), ("preset", "ascendc-eval"),
                        ("kernel_dir", "kernel"), ("soc", "Ascend910B3"),
                        ("build_type", "Release"), ("make_jobs", "8"),
                        ("verify_cmd", None), ("profile_cmd", None),
                        ("stage_timeout", 3600),
                        ("artifacts", ["performance.json", "perf_report.md",
                                       "evaluation_results.json", "logs"]),
                        ("fetch_dest", None), ("fetch_dir", args.fetch_dir),
                        ("fetch_full", False), ("src", None), ("srcs", [])):
                    if not hasattr(job_args, name):
                        setattr(job_args, name, default)
                if job_args.stage_json:
                    spec = build_spec_from_stages_file(job_args)
                else:
                    spec = build_ascendc_spec(job_args)
                spec = prepare_job_sources(session, job_args, spec)
                ok2, payload2 = session.call(
                    ["job-submit", "--spec-b64", base64.b64encode(json.dumps(
                        spec, sort_keys=True).encode("utf-8")).decode("ascii")],
                    timeout=120)
                if not ok2:
                    raise CliError(payload2.get("code", 6),
                                   "job-submit failed: %s" % payload2.get("error"))
                record_job(args.state_file, sid, job_args.job_id,
                           status=payload2.get("status"))
                state = wait_terminal(session, job_args.job_id,
                                      timeout=args.wait_timeout, poll=args.poll_interval)
                fetched = fetch_job(session, job_args, job_args.job_id)
                outcome = {"session_id": sid, "status": state.get("status"),
                           "failure_class": state.get("failure_class"),
                           "device_id": lease["device_id"], "fetched": fetched}
            except CliError as exc:
                outcome = {"session_id": sid, "status": "error",
                           "failure_class": _classify_cli(exc), "error": str(exc)}
            except Exception as exc:
                outcome = {"session_id": sid, "status": "error",
                           "failure_class": "internal", "error": str(exc)}
            finally:
                if heartbeat is not None:
                    heartbeat.stop()
                if lease is not None:
                    try:
                        session.call(["lease-release", "--lease-id", lease["lease_id"]],
                                     timeout=60)
                        mark_lease_released(args.state_file, lease["lease_id"])
                    except CliError as exc:
                        log("lease release failed for %s: %s" % (sid, exc))
            with lock:
                results[sid] = outcome

    threads = [threading.Thread(target=_one, args=(e,)) for e in sessions]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    emit({"batch": results})
    ok_all = len(results) == len(sessions) and \
        all(r.get("status") == "succeeded" for r in results.values())
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _global_parser():
    parser = argparse.ArgumentParser(
        prog="remote_exec.py",
        description="Submit AscendC build/verify/profile jobs to a remote NPU host over SSH "
                    "(local execution flow stays unchanged; this is an optional backend).",
        epilog="See SKILL.md and references/ under skills/lingxi-remote-execution/ for "
               "config schema and job protocol.")
    parser.add_argument("--config", default=os.environ.get("LINGXI_REMOTE_CONFIG"),
                        help="path to remote_hosts.json (default: $LINGXI_REMOTE_CONFIG or ./remote_hosts.json)")
    parser.add_argument("--host", default=os.environ.get("LINGXI_REMOTE_HOST"),
                        help="host alias in config (default: $LINGXI_REMOTE_HOST)")
    parser.add_argument("--state-file", default=None,
                        help="local session state file (default: <cwd>/remote_session_state.json)")
    return parser


def _add_common_job_args(p):
    p.add_argument("--job-id", required=True)
    p.add_argument("--session", required=True, help="operator session id (binds the device lease)")
    p.add_argument("--src", action="append", default=None,
                   help="local dir to sync into the job bundle (repeatable)")
    p.add_argument("--preset", default="ascendc-eval", choices=["ascendc-eval"])
    p.add_argument("--stage-json", default=None,
                   help="path to a full job spec JSON (overrides preset)")
    p.add_argument("--kernel-dir", default="kernel", help="kernel dir inside bundle (preset)")
    p.add_argument("--soc", default="Ascend910B3", help="SoC version (preset)")
    p.add_argument("--build-type", default="Release")
    p.add_argument("--make-jobs", type=int, default=8)
    p.add_argument("--verify-cmd", required=False, default=None,
                   help='JSON argv, e.g. \'["python3","scripts/verify.py","."]\'')
    p.add_argument("--profile-cmd", default=None, help="JSON argv for the profile stage")
    p.add_argument("--stage-timeout", type=int, default=3600)
    p.add_argument("--artifacts", nargs="*", default=["performance.json", "perf_report.md",
                                                      "evaluation_results.json", "logs"],
                   help="artifact globs (relative to job dir) to declare in the manifest")
    p.add_argument("--wait", action="store_true", help="wait until the job reaches a terminal state")
    p.add_argument("--wait-timeout", type=float, default=7200)
    p.add_argument("--poll-interval", type=float, default=10)
    p.add_argument("--fetch", action="store_true", help="fetch results after terminal state")
    p.add_argument("--fetch-dest", default=None, help="local destination dir for fetch")
    p.add_argument("--fetch-dir", default="./remote_jobs", help="default fetch base dir")
    p.add_argument("--fetch-full", action="store_true",
                   help="fetch the whole job dir instead of state/logs/artifacts")


def _build_parser():
    parser = _global_parser()
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("deploy", help="push remote_worker.py to the host and write pool config")
    p.add_argument("--max-build-slots", type=int, default=None)

    p = sub.add_parser("preflight", help="verify ssh/python/remote_root/CANN env/npu-smi/tools/pool")
    p.add_argument("--deploy-if-missing", action="store_true", default=True)
    p.add_argument("--no-deploy", dest="deploy_if_missing", action="store_false")
    p.add_argument("--max-build-slots", type=int, default=None)

    p = sub.add_parser("lease", help="device lease operations")
    lease_sub = p.add_subparsers(dest="lease_cmd")
    lp = lease_sub.add_parser("acquire", help="acquire a host-level device lease (sticky)")
    lp.add_argument("--session", required=True)
    lp.add_argument("--owner", required=True, help="operator name for auditability")
    lp.add_argument("--ttl", type=float, default=3600.0)
    lp.add_argument("--timeout", type=float, default=60.0)
    lp.add_argument("--device", type=int, default=None)
    lp.add_argument("--no-sticky", action="store_true")
    lp = lease_sub.add_parser("heartbeat", help="refresh lease heartbeat")
    lp.add_argument("--lease-id", required=True)
    lp.add_argument("--watch", action="store_true",
                    help="keep refreshing until interrupted (use for the whole evolution session)")
    lp.add_argument("--interval", type=float, default=30.0)
    lp = lease_sub.add_parser("release", help="explicitly release a lease")
    lp.add_argument("--lease-id", required=True)
    lease_sub.add_parser("list", help="show pool and leases")

    p = sub.add_parser("job", help="job operations")
    job_sub = p.add_subparsers(dest="job_cmd")
    jp = job_sub.add_parser("submit", help="submit a job (idempotent per job_id)")
    _add_common_job_args(jp)
    jp = job_sub.add_parser("status", help="query job state")
    jp.add_argument("--job-id", required=True)
    jp.add_argument("--wait", action="store_true")
    jp.add_argument("--wait-timeout", type=float, default=7200)
    jp.add_argument("--poll-interval", type=float, default=10)
    jp = job_sub.add_parser("cancel", help="cancel a running job (kills only its process group)")
    jp.add_argument("--job-id", required=True)
    jp = job_sub.add_parser("fetch", help="fetch job results/logs/artifacts")
    jp.add_argument("--job-id", required=True)
    jp.add_argument("--dest", default=None)
    jp.add_argument("--full", action="store_true")
    jp = job_sub.add_parser("list", help="list jobs on the host")
    jp.add_argument("--session", default=None)
    jp = job_sub.add_parser("cleanup", help="remove a single job dir on the remote host")
    jp.add_argument("--job-id", required=True)
    jp.add_argument("--force", action="store_true")

    p = sub.add_parser("recover", help="re-attach after disconnect: report lease+job states")
    p.add_argument("--session", required=True)
    p.add_argument("--reattach", action="store_true",
                   help="also re-acquire (sticky) the session lease")
    p.add_argument("--ttl", type=float, default=3600.0)
    p.add_argument("--timeout", type=float, default=10.0)

    p = sub.add_parser("batch", help="run multiple operator sessions bounded by the device pool")
    p.add_argument("--plan", required=True, help="JSON plan file (see references/job-protocol.md)")
    p.add_argument("--ttl", type=float, default=3600.0)
    p.add_argument("--lease-timeout", type=float, default=600.0)
    p.add_argument("--wait-timeout", type=float, default=7200.0)
    p.add_argument("--poll-interval", type=float, default=10.0)
    p.add_argument("--max-active", type=int, default=0,
                   help="cap concurrent operators (0 = pool size)")
    p.add_argument("--fetch-dir", default="./remote_jobs")

    return parser


def _resolve_state_file(args):
    if args.state_file:
        return args.state_file
    return os.path.join(os.getcwd(), "remote_session_state.json")


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.error("a command is required")
    if args.command == "lease" and not args.lease_cmd:
        parser.error("a lease command is required")
    if args.command == "job" and not args.job_cmd:
        parser.error("a job command is required")
    args.state_file = _resolve_state_file(args)

    try:
        config_path = args.config
        if not config_path:
            for cand in DEFAULT_CONFIG_PATHS:
                if os.path.isfile(cand):
                    config_path = cand
                    break
        if not config_path:
            raise CliError(2, "no config given (--config or $LINGXI_REMOTE_CONFIG)")
        config = load_config(config_path)
        if not args.host:
            raise CliError(2, "no host given (--host or $LINGXI_REMOTE_HOST)")
        entry = resolve_host_cfg(config, args.host)
        session = RemoteSession(entry)

        if args.command == "deploy":
            build_slots = args.max_build_slots or entry.get("max_build_slots", 4)
            payload = session.deploy(build_slots, entry.get("devices", "auto"),
                                     entry.get("env_script"), entry.get("required_paths"))
            emit(payload)
            if not payload.get("ok"):
                return 1
        elif args.command == "preflight":
            def _ping():
                try:
                    ok, _payload = session.call(["ping"], timeout=60)
                    return ok
                except CliError:
                    return False

            if not _ping() and args.deploy_if_missing:
                log("worker unreachable on host, deploying first...")
                build_slots = args.max_build_slots or entry.get("max_build_slots", 4)
                session.deploy(build_slots, entry.get("devices", "auto"),
                               entry.get("env_script"), entry.get("required_paths"))
            if not _ping():
                raise CliError(7, "worker unreachable on host %r (ssh/auth/path problem)" % args.host)
            env_script = entry.get("env_script")
            devices = entry.get("devices", "auto")
            cmd = ["preflight", "--write-config",
                   "--max-build-slots", str(args.max_build_slots or
                                             entry.get("max_build_slots", 4))]
            if devices != "auto":
                cmd += ["--devices", ",".join(str(d) for d in devices)]
            if env_script:
                cmd += ["--env-script", env_script]
            for required_path in entry.get("required_paths", []):
                cmd += ["--required-path", required_path]
            ok, payload = session.call(cmd, timeout=300)
            emit(payload)
            if not ok or not payload.get("ok"):
                return 1
        elif args.command == "lease":
            if args.lease_cmd == "acquire":
                cmd = ["lease-acquire", "--session", args.session, "--owner", args.owner,
                       "--ttl", str(args.ttl), "--timeout", str(args.timeout)]
                if args.device is not None:
                    cmd += ["--device", str(args.device)]
                if args.no_sticky:
                    cmd.append("--no-sticky")
                ok, payload = session.call(cmd, timeout=args.timeout + 120)
                if not ok:
                    raise CliError(payload.get("code", 6), payload.get("error"))
                lease = payload["lease"]
                record_session(args.state_file, args.host, args.session,
                               lease_id=lease["lease_id"], device_id=lease["device_id"],
                               owner=args.owner, ttl_s=lease.get("ttl_s", args.ttl),
                               lease_active=True)
                emit({"lease": lease})
            elif args.lease_cmd == "heartbeat":
                if args.interval <= 0:
                    raise CliError(2, "--interval must be positive")
                while True:
                    ok, payload = session.call(
                        ["lease-heartbeat", "--lease-id", args.lease_id])
                    if not ok:
                        raise CliError(payload.get("code", 6), payload.get("error"))
                    emit(payload)
                    if not args.watch:
                        break
                    time.sleep(args.interval)
            elif args.lease_cmd == "release":
                ok, payload = session.call(["lease-release", "--lease-id", args.lease_id])
                if not ok:
                    raise CliError(payload.get("code", 6), payload.get("error"))
                if os.path.isfile(args.state_file):
                    mark_lease_released(args.state_file, args.lease_id)
                emit(payload)
            elif args.lease_cmd == "list":
                ok, payload = session.call(["pool-list"])
                if not ok:
                    raise CliError(payload.get("code", 6), payload.get("error"))
                emit(payload)
        elif args.command == "job":
            if args.job_cmd == "submit":
                return do_submit(session, args)
            elif args.job_cmd == "status":
                if args.wait:
                    state = wait_terminal(session, args.job_id, timeout=args.wait_timeout,
                                          poll=args.poll_interval)
                    emit({"state": state})
                    if state.get("status") != "succeeded":
                        return 1
                else:
                    ok, payload = session.call(["job-status", "--job-id", args.job_id])
                    if not ok:
                        raise CliError(payload.get("code", 6), payload.get("error"))
                    emit(payload)
            elif args.job_cmd == "cancel":
                ok, payload = session.call(["job-cancel", "--job-id", args.job_id], timeout=120)
                if not ok:
                    raise CliError(payload.get("code", 6), payload.get("error"))
                emit(payload)
            elif args.job_cmd == "fetch":
                fake = argparse.Namespace(fetch_dest=args.dest, fetch_full=args.full,
                                          fetch_dir="./remote_jobs",
                                          artifacts=["performance.json", "perf_report.md",
                                                     "evaluation_results.json", "logs"])
                emit(fetch_job(session, fake, args.job_id))
            elif args.job_cmd == "list":
                cmd = ["job-list"]
                if args.session:
                    cmd += ["--session", args.session]
                ok, payload = session.call(cmd)
                if not ok:
                    raise CliError(payload.get("code", 6), payload.get("error"))
                emit(payload)
            elif args.job_cmd == "cleanup":
                cmd = ["job-cleanup", "--job-id", args.job_id]
                if args.force:
                    cmd.append("--force")
                ok, payload = session.call(cmd)
                if not ok:
                    raise CliError(payload.get("code", 6), payload.get("error"))
                emit(payload)
        elif args.command == "recover":
            state = load_local_state(args.state_file)
            sess_state = (state.get("sessions", {}) or {}).get(args.session, {})
            report = {"session_id": args.session, "local_state": sess_state}
            ok, pool_payload = session.call(["pool-list"])
            report["pool"] = pool_payload.get("pool") if ok else None
            ok, jobs_payload = session.call(["job-list", "--session", args.session])
            report["jobs"] = jobs_payload.get("jobs", []) if ok else None
            if args.reattach:
                old_device = sess_state.get("device_id")
                ok, payload = session.call(
                    ["lease-acquire", "--session", args.session,
                     "--owner", sess_state.get("owner", args.session),
                     "--ttl", str(args.ttl), "--timeout", str(args.timeout)] +
                    (["--device", str(old_device)] if old_device is not None else []),
                    timeout=args.timeout + 120)
                if ok:
                    lease = payload["lease"]
                    if old_device is not None and int(lease["device_id"]) != int(old_device):
                        session.call(["lease-release", "--lease-id", lease["lease_id"]])
                        report["reattach_error"] = (
                            "original device %s is unavailable; a different card was not accepted. "
                            "Start a new session and rerun the baseline." % old_device)
                        report["requires_new_baseline"] = True
                    else:
                        report["reattached_lease"] = lease
                        record_session(args.state_file, args.host, args.session,
                                       lease_id=lease["lease_id"],
                                       device_id=lease["device_id"],
                                       ttl_s=lease.get("ttl_s", args.ttl), lease_active=True)
                else:
                    report["reattach_error"] = payload.get("error")
            emit(report)
            if report.get("reattach_error"):
                return 1
        elif args.command == "batch":
            try:
                plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise CliError(2, "cannot read batch plan: %s" % exc)
            return run_batch(session, args, plan)
        else:
            raise CliError(2, "unknown command: %r" % args.command)
    except CliError as exc:
        emit({"ok": False, "error": str(exc), "failure_class": _classify_cli(exc),
              "code": exc.code})
        return exc.code
    return 0


def _classify_cli(exc):
    """把本地 CLI 错误粗分为 transport / 协议错误，便于上层失败分类。"""
    if exc.code == 7:
        return "transport"
    return "protocol"


if __name__ == "__main__":
    sys.exit(main())
