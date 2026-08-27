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
lingxi-remote-execution 远端 worker（部署并运行在远端 NPU 主机上）。

设计约定（与本地 remote_exec.py 的协议）：

- 无守护进程：每次 SSH 调用是一个短生命周期 worker 进程。长任务由
  job-submit 以独立会话（start_new_session，即 setsid）派生 _job-runner，
  SSH 断开不影响远端任务；重连后用 job-status 查询即可。
- 跨进程原子设备分配：pool/device_<id>.lock 上的 flock + lease JSON。
  lease 含 lease_id / session_id / owner / 心跳时间戳 / TTL。同一 session
  sticky 复用同一张卡（baseline 到最终复测绑卡不变）。
- 每张卡的 verify/profile 通过 device_<id>.exec.lock 串行；构建不占设备锁，
  只受 build slot 信号量约束，可与设备阶段并行。
- 所有文件操作限制在 --root 内：拒绝 `..`、绝对路径逃逸、符号链接逃逸。
  删除只允许 job-cleanup 针对单个 job 目录（先做 realpath 包含性校验）。
- 环境加载：pool_config.env_script 是显式配置的可信绝对路径，worker 用
  固定的 `bash -c` 包装捕获其导出变量并合并进 stage 环境；不依赖交互式
  shell，不接受 job spec 传入的 shell 字符串。
- pip 安装只允许 --target 到 job 私有目录（job/pylibs），verify 阶段通过
  该 job 的 PYTHONPATH 使用，绝不写远端全局 Python 环境。
- 命令一律以 argv 数组执行（subprocess 不经 shell），spec 中携带
  shell 元字符的命令/路径直接校验拒绝。

输出协议：stdout 输出单个 JSON 对象，成功 {"ok": true, ...}，
失败 {"ok": false, "error": "...", "code": N}。
退出码：0 成功；2 参数/校验错误；3 job_id 冲突（已存在且 spec 不同）；
4 未找到；5 锁/租约获取超时；6 远端环境错误。
"""

import argparse
import base64
import fcntl
import glob as _glob
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

WORKER_VERSION = "1.1.0"

JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
STAGE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UPLOAD_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
# 命令程序名/路径里禁止出现的 shell 元字符与空白（命令走 argv 执行，
# 这里是防御性收紧：防止任何环节退化成 shell 拼接时被注入）
_CMD_META = set(" \t\n\r\x00;|&<>'\"`$(){}[]!*?\\")

STAGE_KINDS = ("cmd", "pip_install")
STAGE_FAILURE_CLASSES = ("build", "correctness", "profile", "environment")
# 用户 spec 禁止自行指定设备；设备唯一来源是 session lease
FORBIDDEN_ENV_KEYS = ("ASCEND_RT_VISIBLE_DEVICES",)
TERMINAL_STATES = ("succeeded", "failed", "cancelled")


class WorkerError(Exception):
    def __init__(self, code, message):
        super(WorkerError, self).__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# 基础工具：JSON 原子读写、flock、进程存活检查
# ---------------------------------------------------------------------------

def _now():
    return time.time()


def _emit(obj):
    obj.setdefault("ok", True)
    sys.stdout.write(json.dumps(obj, sort_keys=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _atomic_write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp), str(path))


def _read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


@contextmanager
def _flock_exclusive(path, timeout=60.0, poll=0.05):
    """排他 flock；超时抛 WorkerError(5)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + timeout
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise WorkerError(5, "lock timeout: %s" % path)
                time.sleep(poll)
        yield fd
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def _pid_alive(pid):
    if not pid or int(pid) <= 1:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _pgid_cmdline(pgid):
    """读取进程组首进程的 cmdline（PID 复用防护：杀进程组前必须核对身份）。"""
    try:
        with open("/proc/%d/cmdline" % int(pgid), "rb") as fh:
            raw = fh.read()
    except (OSError, ValueError):
        return None
    return raw.decode("utf-8", "replace").split("\x00")


def _proc_start_token(pid):
    """Return Linux /proc starttime, which changes when a PID is reused."""
    try:
        raw = Path("/proc/%d/stat" % int(pid)).read_text(encoding="utf-8")
        # The command name in parentheses may contain spaces.  Fields after the
        # final ')' start at field 3; starttime is field 22.
        tail = raw.rsplit(")", 1)[1].strip().split()
        return tail[19]
    except (OSError, ValueError, IndexError):
        return None


def _runner_identity_matches(state, job_id):
    """Verify both process start time and exact runner argv before signalling."""
    pid = state.get("pgid") or state.get("runner_pid")
    if not _pid_alive(pid):
        return False
    expected_token = state.get("runner_start_token")
    if expected_token and _proc_start_token(pid) != str(expected_token):
        return False
    argv = _pgid_cmdline(pid)
    if not argv or not any(os.path.basename(a) == "remote_worker.py" for a in argv):
        return False
    try:
        runner_at = argv.index("_job-runner")
        job_flag_at = argv.index("--job-id", runner_at + 1)
    except ValueError:
        return False
    return job_flag_at + 1 < len(argv) and argv[job_flag_at + 1] == job_id


# ---------------------------------------------------------------------------
# 路径安全：一切相对路径限制在 root 内
# ---------------------------------------------------------------------------

def safe_join(root, rel):
    """把 rel 限制在 root 内。拒绝空/绝对路径、`..`/`.` 分量、符号链接逃逸。"""
    if not isinstance(rel, str) or not rel:
        raise WorkerError(2, "empty relative path")
    if rel.startswith("/") or rel.startswith("~") or "\\" in rel or "\x00" in rel:
        raise WorkerError(2, "unsafe path (absolute/escape): %r" % rel)
    comps = rel.split("/")
    if any(c in ("", ".", "..") for c in comps):
        raise WorkerError(2, "unsafe path component in %r" % rel)
    cand = root.joinpath(*comps)
    real_root = root.resolve()
    real = cand.resolve()
    if real != real_root and real_root not in real.parents:
        raise WorkerError(2, "path escapes root: %r" % rel)
    return cand


def safe_glob_pattern(root, pattern):
    """校验含通配符的相对 pattern（用于 wheel/artifacts），返回绝对 pattern。"""
    if not isinstance(pattern, str) or not pattern:
        raise WorkerError(2, "empty glob pattern")
    if pattern.startswith(("/", "~", "-")) or "\x00" in pattern or "\\" in pattern:
        raise WorkerError(2, "unsafe glob pattern: %r" % pattern)
    comps = pattern.split("/")
    if any(c in ("", ".", "..") for c in comps):
        raise WorkerError(2, "unsafe glob component in %r" % pattern)
    return str(root) + "/" + pattern


def _path_has_symlink(root, path):
    root = root.resolve()
    cur = Path(path)
    while cur != root:
        if cur.is_symlink():
            return True
        if root not in cur.resolve().parents and cur.resolve() != root:
            return True
        cur = cur.parent
    return False


def safe_glob_matches(root, pattern, files_only=False):
    """Expand a glob and reject every match that escapes through a symlink."""
    root = root.resolve()
    matches = []
    for raw in sorted(_glob.glob(safe_glob_pattern(root, pattern))):
        path = Path(raw)
        real = path.resolve()
        if real != root and root not in real.parents:
            raise WorkerError(2, "glob match escapes root: %r" % pattern)
        if _path_has_symlink(root, path):
            raise WorkerError(2, "glob match traverses a symlink: %s" % path)
        if files_only and not path.is_file():
            continue
        matches.append(str(path))
    return matches


def tree_sha256(root):
    """Hash a source tree deterministically and reject links/special files."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise WorkerError(2, "source bundle is missing or is a symlink: %s" % root)
    digest = hashlib.sha256()
    entries = []
    for current, dirs, files in os.walk(str(root), topdown=True, followlinks=False):
        dirs.sort()
        files.sort()
        current_path = Path(current)
        for name in dirs:
            path = current_path / name
            if path.is_symlink():
                raise WorkerError(2, "source bundle contains symlink: %s" % path)
            rel = path.relative_to(root).as_posix()
            entries.append((rel, "dir", path))
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise WorkerError(2, "source bundle contains non-regular file: %s" % path)
            rel = path.relative_to(root).as_posix()
            entries.append((rel, "file", path))
    for rel, kind, path in sorted(entries, key=lambda item: (item[0], item[1])):
        if kind == "dir":
            digest.update(("D\0%s\0" % rel).encode("utf-8"))
            continue
        file_stat = path.stat()
        digest.update(("F\0%s\0%d\0" %
                       (rel, file_stat.st_size)).encode("utf-8"))
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_job_dir(root, job_id):
    if not JOB_ID_RE.match(job_id or ""):
        raise WorkerError(2, "invalid job_id: %r" % job_id)
    jobs_root = (root / "jobs").resolve()
    job_dir = jobs_root / job_id
    # A symlink is rejected even when it points to another job inside jobs/:
    # cleanup must never delete a different job through an alias.
    if job_dir.is_symlink():
        raise WorkerError(2, "job directory must not be a symlink: %r" % job_id)
    if job_dir.resolve() != job_dir and jobs_root not in job_dir.resolve().parents:
        raise WorkerError(2, "job_id resolves outside jobs dir: %r" % job_id)
    return job_dir


# ---------------------------------------------------------------------------
# 设备池（主机级，跨进程共享）
# ---------------------------------------------------------------------------

class DevicePool(object):
    """主机级设备租约池。

    文件结构（均在 <root>/pool/ 下）：
        pool_config.json          # 设备列表 / build slots / env_script
        device_<id>.lock          # lease 变更原子锁（flock）
        device_<id>.lease.json    # 当前租约（lease_id/session/owner/心跳/TTL）
        device_<id>.exec.lock     # 该卡 verify/profile 串行锁
        build_slots/slot_<i>.lock # 构建并发信号量
    """

    def __init__(self, root):
        self.root = root
        self.pool_dir = root / "pool"
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        (self.pool_dir / "build_slots").mkdir(parents=True, exist_ok=True)

    # --- 配置 ---
    def config(self):
        return _read_json(self.pool_dir / "pool_config.json", {}) or {}

    def write_config(self, devices, max_build_slots, env_script):
        normalized_devices = [int(d) for d in devices]
        cfg = {
            "devices": normalized_devices,
            "max_build_slots": int(max_build_slots),
            "env_script": env_script if env_script else None,
            "updated_ts": _now(),
        }
        # Concurrent operator sessions commonly run preflight together.  The
        # host-wide allocation lock serializes config publication with lease
        # allocation.  More importantly, a client with a different config may
        # not remove/relabel cards while another session still owns one.
        with _flock_exclusive(self.allocation_lock_path(), timeout=60.0):
            current = self.config()
            active = []
            for dev in current.get("devices", []):
                lease = self._read_lease(int(dev))
                if self.lease_valid(lease):
                    active.append(lease)
            semantic_current = {
                "devices": [int(d) for d in current.get("devices", [])],
                "max_build_slots": int(current.get("max_build_slots", 4)),
                "env_script": current.get("env_script"),
            }
            semantic_new = {
                "devices": normalized_devices,
                "max_build_slots": int(max_build_slots),
                "env_script": env_script if env_script else None,
            }
            if active and current and semantic_current != semantic_new:
                raise WorkerError(
                    6, "refusing to change pool config while active leases exist")
            _atomic_write_json(self.pool_dir / "pool_config.json", cfg)
        return cfg

    def devices(self):
        devs = self.config().get("devices") or []
        return [int(d) for d in devs]

    # --- 租约文件 ---
    def _lease_path(self, dev):
        return self.pool_dir / ("device_%d.lease.json" % dev)

    def _lock_path(self, dev):
        return self.pool_dir / ("device_%d.lock" % dev)

    def exec_lock_path(self, dev):
        return self.pool_dir / ("device_%d.exec.lock" % dev)

    def allocation_lock_path(self):
        return self.pool_dir / "allocation.lock"

    @staticmethod
    def lease_valid(meta, now=None):
        if not meta:
            return False
        now = now if now is not None else _now()
        try:
            hb = float(meta.get("heartbeat_ts", 0))
            ttl = float(meta.get("ttl_s", 0))
        except (TypeError, ValueError):
            return False
        return ttl > 0 and (now - hb) < ttl

    def _read_lease(self, dev):
        return _read_json(self._lease_path(dev))

    def _write_lease(self, dev, meta):
        _atomic_write_json(self._lease_path(dev), meta)

    def find_session_lease(self, session_id):
        """扫描仍然有效的 session 租约（sticky 复用入口）。"""
        for dev in self.devices():
            meta = self._read_lease(dev)
            if meta and meta.get("session_id") == session_id and self.lease_valid(meta):
                return meta
        return None

    def acquire(self, session_id, owner, ttl_s, timeout, preferred=None, sticky=True):
        """获取设备租约（跨进程原子）。

        sticky=True 时优先复用本 session 仍有效的租约（同一算子进化
        session 从 baseline 到最终复测始终绑同一张卡）。
        """
        if not JOB_ID_RE.match(session_id or ""):
            raise WorkerError(2, "invalid session id: %r" % session_id)
        if not isinstance(owner, str) or not owner or "\x00" in owner:
            raise WorkerError(2, "invalid lease owner")
        if float(ttl_s) < 1.0:
            raise WorkerError(2, "lease TTL must be at least 1 second")
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            order = list(self.devices())
            if not order:
                raise WorkerError(6, "device pool not configured (run deploy/preflight first)")
            if preferred is not None:
                preferred = int(preferred)
                if preferred not in order:
                    raise WorkerError(2, "preferred device %d not in pool %s" % (preferred, order))
                order = [preferred] + [d for d in order if d != preferred]
            # One host-wide lock covers "find sticky lease + choose free card".
            # Per-device locks alone cannot stop two concurrent acquires for the
            # same session from choosing two different free cards.
            with _flock_exclusive(self.allocation_lock_path(), timeout=60.0):
                if sticky:
                    meta = self.find_session_lease(session_id)
                    if meta:
                        dev = int(meta["device_id"])
                        with _flock_exclusive(self._lock_path(dev)):
                            cur = self._read_lease(dev)
                            if cur and cur.get("session_id") == session_id \
                                    and self.lease_valid(cur):
                                cur["heartbeat_ts"] = _now()
                                cur["reattach_ts"] = _now()
                                self._write_lease(dev, cur)
                                return cur
                for dev in order:
                    with _flock_exclusive(self._lock_path(dev)):
                        meta = self._read_lease(dev)
                        if meta and self.lease_valid(meta):
                            continue  # 被别的 session 占用
                        lease = {
                            "lease_id": uuid.uuid4().hex,
                            "session_id": session_id,
                            "owner": owner,
                            "device_id": dev,
                            "created_ts": _now(),
                            "heartbeat_ts": _now(),
                            "ttl_s": float(ttl_s),
                        }
                        self._write_lease(dev, lease)
                        return lease
            if time.monotonic() >= deadline:
                raise WorkerError(5, "no free device within %.1fs (pool=%s)" % (timeout, order))
            time.sleep(0.5)

    def _find_by_lease_id(self, lease_id):
        for dev in self.devices():
            meta = self._read_lease(dev)
            if meta and meta.get("lease_id") == lease_id:
                return dev, meta
        return None, None

    def heartbeat(self, lease_id):
        dev, meta = self._find_by_lease_id(lease_id)
        if not meta:
            raise WorkerError(4, "lease not found: %s" % lease_id)
        with _flock_exclusive(self._lock_path(dev)):
            cur = self._read_lease(dev)
            if not cur or cur.get("lease_id") != lease_id:
                raise WorkerError(4, "lease lost while heartbeating: %s" % lease_id)
            if not self.lease_valid(cur):
                raise WorkerError(4, "lease already expired: %s" % lease_id)
            cur["heartbeat_ts"] = _now()
            self._write_lease(dev, cur)
            return cur

    def require_active(self, lease_id, session_id=None, device_id=None):
        """Recheck the exact lease token while holding that device's lock."""
        dev, meta = self._find_by_lease_id(lease_id)
        if meta is None:
            raise WorkerError(4, "lease not found: %s" % lease_id)
        with _flock_exclusive(self._lock_path(dev)):
            cur = self._read_lease(dev)
            if not cur or cur.get("lease_id") != lease_id or not self.lease_valid(cur):
                raise WorkerError(4, "lease is no longer active: %s" % lease_id)
            if session_id is not None and cur.get("session_id") != session_id:
                raise WorkerError(4, "lease session changed: %s" % lease_id)
            if device_id is not None and int(cur.get("device_id")) != int(device_id):
                raise WorkerError(4, "lease device changed: %s" % lease_id)
            return cur

    def release(self, lease_id):
        dev, meta = self._find_by_lease_id(lease_id)
        if not meta:
            raise WorkerError(4, "lease not found: %s" % lease_id)
        with _flock_exclusive(self._lock_path(dev)):
            cur = self._read_lease(dev)
            if cur and cur.get("lease_id") == lease_id:
                try:
                    os.unlink(str(self._lease_path(dev)))
                except OSError:
                    pass
        return {"released": True, "lease_id": lease_id, "device_id": dev}

    def session_device(self, session_id):
        """返回 session 当前有效租约的 device_id；无则 None。"""
        meta = self.find_session_lease(session_id)
        return int(meta["device_id"]) if meta else None

    def status(self):
        out = []
        for dev in self.devices():
            meta = self._read_lease(dev)
            out.append({
                "device_id": dev,
                "lease": meta,
                "valid": self.lease_valid(meta),
            })
        return out


class BuildSlots(object):
    """构建并发信号量：flock 计数，构建不占设备锁。"""

    def __init__(self, pool_dir, max_slots):
        self.slot_dir = pool_dir / "build_slots"
        self.slot_dir.mkdir(parents=True, exist_ok=True)
        self.max_slots = max(1, int(max_slots))

    @contextmanager
    def acquire(self, stage_name="stage"):
        deadline = time.monotonic() + 86400.0
        while True:
            for i in range(self.max_slots):
                lock_path = self.slot_dir / ("slot_%d.lock" % i)
                fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    os.close(fd)
                    continue
                try:
                    yield i
                finally:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
                        os.close(fd)
                return
            if time.monotonic() >= deadline:
                raise WorkerError(5, "build slot timeout for %s" % stage_name)
            time.sleep(0.3)


# ---------------------------------------------------------------------------
# job spec 校验
# ---------------------------------------------------------------------------

def _validate_program(prog):
    if not isinstance(prog, str) or not prog:
        raise WorkerError(2, "empty program in command")
    # The preset uses this one trusted placeholder.  It is replaced with the
    # worker's own sys.executable and validated again immediately before exec.
    if prog == "{python}":
        return prog
    for ch in prog:
        if ch in _CMD_META:
            raise WorkerError(2, "program contains shell metacharacter: %r" % prog)
    if "/" in prog:
        segs = prog.split("/")
        if prog.startswith("/"):
            segs = segs[1:]
        if any(s in ("", ".", "..") for s in segs):
            raise WorkerError(2, "unsafe program path: %r" % prog)
    return prog


def _validate_env(env):
    if not isinstance(env, dict):
        raise WorkerError(2, "env must be an object")
    for key, val in env.items():
        if not isinstance(key, str) or not ENV_KEY_RE.match(key):
            raise WorkerError(2, "invalid env key: %r" % key)
        if key in FORBIDDEN_ENV_KEYS:
            raise WorkerError(2, "env key %s is reserved (device comes from lease only)" % key)
        if not isinstance(val, str) or "\x00" in val:
            raise WorkerError(2, "invalid env value for %s" % key)
    return env


def validate_spec(spec, root):
    """严格校验 job spec；返回规范化后的 dict。不做任何路径拼接执行。"""
    if not isinstance(spec, dict):
        raise WorkerError(2, "spec must be an object")
    for field in ("job_id", "session_id", "stages"):
        if field not in spec:
            raise WorkerError(2, "spec missing field: %s" % field)
    for field in ("job_id", "session_id"):
        if not JOB_ID_RE.match(str(spec[field])):
            raise WorkerError(2, "invalid %s: %r" % (field, spec[field]))
    source_digest = spec.get("source_digest")
    source_upload = spec.get("source_upload")
    if source_digest is not None:
        if not isinstance(source_digest, str) or not SHA256_RE.match(source_digest):
            raise WorkerError(2, "invalid source_digest")
        if not isinstance(source_upload, str):
            raise WorkerError(2, "source_upload is required with source_digest")
        prefix = "uploads/%s/%s-" % (spec["job_id"], source_digest)
        if not source_upload.startswith(prefix):
            raise WorkerError(2, "source_upload does not match job_id/source_digest")
        token = source_upload[len(prefix):]
        if not UPLOAD_TOKEN_RE.match(token):
            raise WorkerError(2, "invalid source_upload token")
        safe_join(root, source_upload)
    elif source_upload is not None:
        raise WorkerError(2, "source_upload requires source_digest")
    stages = spec["stages"]
    if not isinstance(stages, list) or not stages:
        raise WorkerError(2, "stages must be a non-empty list")
    seen_names = set()
    for stage in stages:
        if not isinstance(stage, dict):
            raise WorkerError(2, "stage must be an object")
        name = stage.get("name")
        if not isinstance(name, str) or not STAGE_NAME_RE.match(name):
            raise WorkerError(2, "invalid stage name: %r" % name)
        if name in seen_names:
            raise WorkerError(2, "duplicate stage name: %s" % name)
        seen_names.add(name)
        kind = stage.get("kind", "cmd")
        if kind not in STAGE_KINDS:
            raise WorkerError(2, "invalid stage kind: %r" % kind)
        failure_class = stage.get("failure_class")
        if failure_class not in STAGE_FAILURE_CLASSES:
            raise WorkerError(2, "invalid failure_class: %r" % failure_class)
        if not isinstance(stage.get("needs_device", False), bool):
            raise WorkerError(2, "needs_device must be bool")
        timeout_s = stage.get("timeout_s", 3600)
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0 or timeout_s > 172800:
            raise WorkerError(2, "invalid timeout_s for stage %s" % name)
        cwd = stage.get("cwd", "")
        if cwd:
            safe_join(root, cwd)  # 只校验，不创建
        if "env" in stage and stage["env"] is not None:
            _validate_env(stage["env"])
        if kind == "cmd":
            commands = stage.get("commands")
            if not isinstance(commands, list) or not commands:
                raise WorkerError(2, "stage %s: commands must be a non-empty list" % name)
            for argv in commands:
                if not isinstance(argv, list) or not argv:
                    raise WorkerError(2, "stage %s: command must be non-empty argv list" % name)
                for arg in argv:
                    if not isinstance(arg, str) or "\x00" in arg:
                        raise WorkerError(2, "stage %s: invalid argv element" % name)
                _validate_program(argv[0])
        elif kind == "pip_install":
            wheel = stage.get("wheel")
            target = stage.get("target")
            if not wheel or not target:
                raise WorkerError(2, "stage %s: pip_install needs wheel and target" % name)
            safe_glob_pattern(root, wheel)
            safe_join(root, target)
    artifacts = spec.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise WorkerError(2, "artifacts must be a list")
    for pat in artifacts:
        safe_glob_pattern(root, pat)
    return spec


def spec_sha256(spec):
    # Upload paths carry a per-transfer random token.  Source content is bound
    # by source_digest, so the transport token must not break job idempotency.
    canonical = dict(spec)
    canonical.pop("source_upload", None)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# job 状态
# ---------------------------------------------------------------------------

def _job_state_path(job_dir):
    return job_dir / "state.json"


def _job_state_lock(job_dir):
    return job_dir / "state.lock"


def read_state(job_dir):
    return _read_json(_job_state_path(job_dir))


def update_state(job_dir, mutator):
    """在 job 级 flock 下读取-修改-写回 state.json（runner / cancel 并发安全）。"""
    with _flock_exclusive(_job_state_lock(job_dir)):
        state = _read_json(_job_state_path(job_dir)) or {}
        mutator(state)
        _atomic_write_json(_job_state_path(job_dir), state)
        return state


def _initial_state(spec):
    return {
        "job_id": spec["job_id"],
        "session_id": spec["session_id"],
        "status": "submitted",
        "failure_class": None,
        "error": None,
        "device_id": None,
        "lease_id": None,
        "pgid": None,
        "runner_pid": None,
        "runner_start_token": None,
        "cancel_requested": False,
        "created_ts": _now(),
        "started_ts": None,
        "ended_ts": None,
        "worker_version": WORKER_VERSION,
        "stages": [
            {"name": st["name"], "status": "pending", "exit_code": None,
             "error": None, "started_ts": None, "ended_ts": None}
            for st in spec["stages"]
        ],
    }


# ---------------------------------------------------------------------------
# job 生命周期
# ---------------------------------------------------------------------------

def _upload_rel(job_id, source_digest, upload_token):
    if not JOB_ID_RE.match(job_id or ""):
        raise WorkerError(2, "invalid job_id: %r" % job_id)
    if not SHA256_RE.match(source_digest or ""):
        raise WorkerError(2, "invalid source digest")
    if not UPLOAD_TOKEN_RE.match(upload_token or ""):
        raise WorkerError(2, "invalid upload token")
    return "uploads/%s/%s-%s" % (job_id, source_digest, upload_token)


def upload_prepare(root, job_id, source_digest, upload_token):
    """Create one isolated upload staging directory under remote_root/uploads."""
    rel = _upload_rel(job_id, source_digest, upload_token)
    upload_dir = safe_join(root, rel)
    if upload_dir.exists() or upload_dir.is_symlink():
        raise WorkerError(3, "upload staging directory already exists")
    (upload_dir / "bundle").mkdir(parents=True)
    return {"upload_rel": rel, "bundle_rel": rel + "/bundle"}


def upload_discard(root, job_id, source_digest, upload_token):
    rel = _upload_rel(job_id, source_digest, upload_token)
    _cleanup_upload(root, rel)
    return {"upload_rel": rel, "removed": True}


def _cleanup_upload(root, source_upload):
    if not source_upload:
        return
    path = safe_join(root, source_upload)
    uploads_root = (root / "uploads").resolve()
    if path.is_symlink():
        raise WorkerError(2, "upload staging directory must not be a symlink")
    real = path.resolve()
    if real == uploads_root or uploads_root not in real.parents:
        raise WorkerError(2, "refusing to remove outside uploads/: %s" % path)
    if path.exists():
        shutil.rmtree(str(path))
    # Best-effort cleanup of the now-empty per-job parent.
    try:
        path.parent.rmdir()
    except OSError:
        pass

def submit_job(root, spec_b64):
    try:
        raw = base64.b64decode(spec_b64.encode("ascii"), validate=True)
        spec = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise WorkerError(2, "bad spec payload: %s" % exc)
    spec = validate_spec(spec, root)
    job_id = spec["job_id"]
    job_dir = resolve_job_dir(root, job_id)
    jobs_root = root / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    submit_lock = jobs_root / (".submit_%s.lock" % job_id)
    with _flock_exclusive(submit_lock, timeout=120.0):
        if job_dir.exists():
            old = _read_json(job_dir / "spec.json")
            if old is None:
                raise WorkerError(3, "job_id %s has an incomplete existing directory" % job_id)
            if spec_sha256(old) == spec_sha256(spec):
                _cleanup_upload(root, spec.get("source_upload"))
                state = read_state(job_dir) or {}
                return {"job_id": job_id, "status": state.get("status"),
                        "idempotent": True, "state": state}
            _cleanup_upload(root, spec.get("source_upload"))
            raise WorkerError(3, "job_id %s already exists with a different spec" % job_id)

        source_upload = spec.get("source_upload")
        source_digest = spec.get("source_digest")
        upload_bundle = None
        if source_digest:
            upload_dir = safe_join(root, source_upload)
            upload_bundle = upload_dir / "bundle"
            actual_digest = tree_sha256(upload_bundle)
            if actual_digest != source_digest:
                _cleanup_upload(root, source_upload)
                raise WorkerError(2, "uploaded source digest mismatch: expected %s, got %s"
                                  % (source_digest, actual_digest))

        creating = jobs_root / (".creating-%s-%s" % (job_id, uuid.uuid4().hex))
        creating.mkdir()
        try:
            (creating / "logs").mkdir()
            (creating / "artifacts").mkdir()
            if upload_bundle is not None:
                os.replace(str(upload_bundle), str(creating / "bundle"))
            else:
                (creating / "bundle").mkdir()
            stored_spec = dict(spec)
            stored_spec.pop("source_upload", None)
            _atomic_write_json(creating / "spec.json", stored_spec)
            _atomic_write_json(creating / "state.json", _initial_state(stored_spec))
            os.replace(str(creating), str(job_dir))
        except Exception:
            if creating.exists():
                shutil.rmtree(str(creating))
            raise
        if source_upload:
            # bundle was moved; only empty staging parents remain.
            _cleanup_upload(root, source_upload)

        # Spawn exactly once while still holding the per-job submit lock.
        runner_log = open(str(job_dir / "logs" / "runner.log"), "ab", 0)
        try:
            try:
                proc = subprocess.Popen(
                    [sys.executable, os.path.abspath(__file__), "--root", str(root),
                     "_job-runner", "--job-id", job_id],
                    stdin=subprocess.DEVNULL, stdout=runner_log, stderr=runner_log,
                    start_new_session=True, close_fds=True,
                )
            except OSError as exc:
                _finish_job(job_dir, "failed", "environment",
                            "cannot start runner: %s" % exc)
                raise WorkerError(6, "cannot start runner for %s: %s" % (job_id, exc))
        finally:
            runner_log.close()
        start_token = _proc_start_token(proc.pid)
        update_state(job_dir, lambda st: (
            st.__setitem__("runner_pid", proc.pid),
            st.__setitem__("runner_start_token", start_token)))
        state = read_state(job_dir) or {}
        return {"job_id": job_id, "status": state.get("status", "submitted"),
                "idempotent": False, "runner_pid": proc.pid}


_CANCEL_FLAG = {"requested": False}
_CURRENT_CHILD = {"proc": None}


def _signal_child_group(sig):
    child = _CURRENT_CHILD["proc"]
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, sig)
    except OSError:
        try:
            child.send_signal(sig)
        except OSError:
            pass


def _runner_sigterm(_signum, _frame):
    _CANCEL_FLAG["requested"] = True
    _signal_child_group(signal.SIGTERM)


class LeaseHeartbeat(object):
    """Keep one exact lease alive and stop the active stage if it is lost."""

    def __init__(self, pool, lease):
        self.pool = pool
        self.lease_id = lease["lease_id"]
        self.session_id = lease["session_id"]
        self.device_id = int(lease["device_id"])
        ttl = float(lease.get("ttl_s", 0))
        self.interval = max(0.1, min(30.0, ttl / 3.0))
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run,
                                       name="lease-heartbeat-%s" % self.lease_id[:8])
        self.thread.daemon = True

    def start(self):
        self.pool.heartbeat(self.lease_id)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.pool.heartbeat(self.lease_id)
            except WorkerError as exc:
                self.error = str(exc)
                _signal_child_group(signal.SIGTERM)
                return

    def ensure_active(self):
        if self.error:
            raise WorkerError(4, self.error)
        self.pool.heartbeat(self.lease_id)
        return self.pool.require_active(self.lease_id, self.session_id, self.device_id)

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=max(1.0, self.interval * 2))


def _load_env_script_exports(env_script):
    """source 可信 env_script 并捕获其导出的环境变量（一次性，合并进各 stage env）。"""
    if not env_script:
        return {}
    env_script = str(env_script)
    if not os.path.isabs(env_script):
        raise WorkerError(6, "env_script must be an absolute path: %r" % env_script)
    if not os.path.isfile(env_script):
        raise WorkerError(6, "env_script not found on remote host: %s" % env_script)
    capture = (
        "import json, os; print(json.dumps(dict(os.environ)))"
    )
    argv = ["bash", "-c", 'source "$0" >/dev/null 2>&1; exec "$1" -c "$2"',
            env_script, sys.executable, capture]
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkerError(6, "failed to source env_script %s: %s" % (env_script, exc))
    if result.returncode != 0:
        raise WorkerError(6, "env_script %s exited %d: %s"
                          % (env_script, result.returncode, result.stderr.strip()[:500]))
    try:
        exports = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise WorkerError(6, "cannot parse env exports from %s" % env_script)
    return exports if isinstance(exports, dict) else {}


def _substitute(value, ctx):
    if not isinstance(value, str):
        return value
    for key, val in ctx.items():
        value = value.replace("{" + key + "}", val)
    return value


def _run_stage_commands(stage, ctx, base_env, job_dir, log_path):
    """执行单个 stage 的命令序列，返回 (ok, exit_code, error)。"""
    name = stage["name"]
    env = dict(base_env)
    for key, val in (stage.get("env") or {}).items():
        env[key] = _substitute(val, ctx)
    cwd_rel = stage.get("cwd", "")
    cwd = job_dir if not cwd_rel else safe_join(job_dir, _substitute(cwd_rel, ctx))
    cwd.mkdir(parents=True, exist_ok=True)
    timeout_s = float(stage.get("timeout_s", 3600))

    commands = []
    if stage.get("kind") == "pip_install":
        wheel_pat = _substitute(stage["wheel"], ctx)
        matches = safe_glob_matches(job_dir, wheel_pat, files_only=True)
        if not matches:
            return False, None, "no wheel matches %s" % wheel_pat
        target = safe_join(job_dir, _substitute(stage["target"], ctx))
        target.mkdir(parents=True, exist_ok=True)
        # 关键隔离点：--target 安装到 job 私有目录，绝不触碰远端全局环境
        commands.append([sys.executable, "-m", "pip", "install",
                         "--no-warn-script-location", "--target", str(target),
                         "--upgrade"] + matches)
    else:
        for argv in stage.get("commands", []):
            expanded = [_substitute(a, ctx) for a in argv]
            _validate_program(expanded[0])
            commands.append(expanded)

    with open(str(log_path), "a", encoding="utf-8") as log:
        log.write("[%s] stage=%s begin ts=%s\n" % (_now(), name, time.strftime("%F %T")))
        for argv in commands:
            if _CANCEL_FLAG["requested"]:
                return False, None, "cancelled"
            log.write("[%s] + %s\n" % (_now(), json.dumps(argv)))
            log.flush()
            try:
                proc = subprocess.Popen(argv, cwd=str(cwd), env=env,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            except FileNotFoundError as exc:
                log.write("[%s] environment error: %s\n" % (_now(), exc))
                return False, None, "command not found: %s" % argv[0]
            _CURRENT_CHILD["proc"] = proc
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _signal_child_group(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _signal_child_group(signal.SIGKILL)
                    proc.wait()
                _CURRENT_CHILD["proc"] = None
                log.write("[%s] timeout after %.0fs\n" % (_now(), timeout_s))
                return False, None, "timeout after %.0fs" % timeout_s
            _CURRENT_CHILD["proc"] = None
            if rc != 0:
                log.write("[%s] stage=%s exit=%d\n" % (_now(), name, rc))
                return False, rc, "command exited %d" % rc
        log.write("[%s] stage=%s ok\n" % (_now(), name))
    return True, 0, None


def run_job(root, job_id):
    """_job-runner 主体：按 build -> (install) -> verify -> profile 顺序执行 spec stages。"""
    job_dir = resolve_job_dir(root, job_id)
    spec = _read_json(job_dir / "spec.json")
    if spec is None:
        raise WorkerError(4, "spec.json not found for job %s" % job_id)
    pool = DevicePool(root)
    cfg = pool.config()

    def _mark_start(state):
        if state.get("status") in TERMINAL_STATES:
            return state  # 幂等：终态 job 不再重跑
        state["status"] = "running"
        state["started_ts"] = _now()
        state["pgid"] = os.getpid()  # start_new_session 下 pgid == 本进程 pid
        state["runner_pid"] = os.getpid()
        state["runner_start_token"] = _proc_start_token(os.getpid())
        return state

    state = update_state(job_dir, _mark_start)
    if state.get("status") in TERMINAL_STATES:
        _emit({"job_id": job_id, "status": state["status"], "skipped": True})
        return

    signal.signal(signal.SIGTERM, _runner_sigterm)
    signal.signal(signal.SIGINT, _runner_sigterm)

    # 一次性 source 可信 env_script，捕获导出变量作为 stage 基础环境
    env_script = cfg.get("env_script")
    try:
        exports = _load_env_script_exports(env_script)
    except WorkerError as exc:
        _finish_job(job_dir, "failed", "environment", str(exc))
        _emit({"job_id": job_id, "status": "failed", "failure_class": "environment"})
        return
    base_env = dict(os.environ)
    base_env.update(exports)
    ascend_home = base_env.get("ASCEND_HOME_PATH", "")
    lease_keeper = None
    initial_lease = pool.find_session_lease(spec["session_id"])
    if initial_lease is not None:
        try:
            lease_keeper = LeaseHeartbeat(pool, initial_lease).start()
        except WorkerError:
            lease_keeper = None

    for stage in spec["stages"]:
        # 两条取消路径汇合：cancel 子进程写 state.cancel_requested（submit 后
        # runner 未启动的窗口），或 SIGTERM 直接置运行中标志
        cur_state = read_state(job_dir) or {}
        if _CANCEL_FLAG["requested"] or cur_state.get("cancel_requested"):
            _CANCEL_FLAG["requested"] = True
            _finish_job(job_dir, "cancelled", "cancelled", "cancelled by request")
            if lease_keeper is not None:
                lease_keeper.stop()
            _emit({"job_id": job_id, "status": "cancelled"})
            return
        name = stage["name"]
        needs_device = bool(stage.get("needs_device"))
        device_id = None
        lease_id = None
        error = None
        exit_code = None
        ok = False
        override_class = None  # lease_lost / cancelled 等不属于 stage 声明的失败类别

        def _stage_begin(st):
            for entry in st["stages"]:
                if entry["name"] == name:
                    entry["status"] = "running"
                    entry["started_ts"] = _now()

        update_state(job_dir, _stage_begin)

        try:
            if lease_keeper is not None:
                lease_keeper.ensure_active()
            if needs_device:
                # 设备唯一来自 session lease；lease 失效/被回收 => lease_lost
                lease = pool.find_session_lease(spec["session_id"])
                if lease is None:
                    override_class = "lease_lost"
                    raise WorkerError(6, "session lease lost before stage %s" % name)
                device_id = int(lease["device_id"])
                lease_id = lease["lease_id"]
                if lease_keeper is None or lease_keeper.lease_id != lease_id:
                    if lease_keeper is not None:
                        lease_keeper.stop()
                    try:
                        lease_keeper = LeaseHeartbeat(pool, lease).start()
                    except WorkerError:
                        override_class = "lease_lost"
                        raise WorkerError(6, "session lease lost before stage %s" % name)
                update_state(job_dir, lambda st: (
                    st.__setitem__("device_id", device_id),
                    st.__setitem__("lease_id", lease_id)))
                lock_ctx = _flock_exclusive(pool.exec_lock_path(device_id), timeout=86400.0)
            else:
                # 构建不占设备锁，只受 build slot 信号量约束
                lock_ctx = BuildSlots(pool.pool_dir, cfg.get("max_build_slots", 4)).acquire(name)
            with lock_ctx:
                if needs_device:
                    # The card may have waited behind another job.  Recheck the
                    # exact token only after obtaining the execution lock.
                    try:
                        lease_keeper.ensure_active()
                    except WorkerError:
                        override_class = "lease_lost"
                        raise WorkerError(6, "session lease lost while waiting for stage %s" % name)
                ctx = {
                    "job_dir": str(job_dir),
                    "bundle_dir": str(job_dir / "bundle"),
                    "pylibs_dir": str(job_dir / "pylibs"),
                    "root": str(root),
                    "python": sys.executable,
                    "ascend_home_path": ascend_home,
                    "device_id": str(device_id) if device_id is not None else "",
                }
                stage_env = dict(base_env)
                if device_id is not None:
                    stage_env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
                log_path = job_dir / "logs" / (name + ".log")
                ok, exit_code, error = _run_stage_commands(
                    stage, ctx, stage_env, job_dir, log_path)
                if lease_keeper is not None and lease_keeper.error:
                    override_class = "lease_lost"
                    ok = False
                    error = lease_keeper.error
        except WorkerError as exc:
            if needs_device or (lease_keeper is not None and lease_keeper.error):
                override_class = "lease_lost"
            error = str(exc) if error is None else error

        if _CANCEL_FLAG["requested"]:
            _finish_job(job_dir, "cancelled", "cancelled", "cancelled by request")
            if lease_keeper is not None:
                lease_keeper.stop()
            _emit({"job_id": job_id, "status": "cancelled"})
            return

        def _stage_end(st):
            for entry in st["stages"]:
                if entry["name"] == name:
                    entry["status"] = "ok" if ok else "failed"
                    entry["exit_code"] = exit_code
                    entry["error"] = error
                    entry["ended_ts"] = _now()

        update_state(job_dir, _stage_end)
        if not ok:
            if override_class is not None:
                failure_class = override_class
            elif error and error.startswith("command not found"):
                failure_class = "environment"
            elif error and error.startswith("no wheel matches"):
                failure_class = "environment"
            else:
                failure_class = stage.get("failure_class", "environment")
            _finish_job(job_dir, "failed", failure_class, "stage %s: %s" % (name, error))
            if lease_keeper is not None:
                lease_keeper.stop()
            _emit({"job_id": job_id, "status": "failed", "failure_class": failure_class,
                   "failed_stage": name})
            return

    _finish_job(job_dir, "succeeded", None, None)
    if lease_keeper is not None:
        lease_keeper.stop()
    _emit({"job_id": job_id, "status": "succeeded"})


def _finish_job(job_dir, status, failure_class, error):
    manifest = _build_manifest(job_dir)
    def _mut(st):
        st["status"] = status
        st["failure_class"] = failure_class
        st["error"] = error
        st["ended_ts"] = _now()
        st["artifacts_manifest"] = manifest
    update_state(job_dir, _mut)


def _build_manifest(job_dir):
    """按 spec 声明的 artifacts glob 生成产物清单（全部限制在 job 目录内）。"""
    spec = _read_json(job_dir / "spec.json") or {}
    files = []
    for pat in spec.get("artifacts", []):
        for match in safe_glob_matches(job_dir, pat):
            if os.path.isfile(match):
                rel = os.path.relpath(match, str(job_dir))
                if not rel.startswith(".."):
                    files.append(rel)
            elif os.path.isdir(match):
                rel = os.path.relpath(match, str(job_dir))
                if not rel.startswith(".."):
                    files.append(rel)
    files.append("logs")
    return {"files": sorted(set(files)), "generated_ts": _now()}


def collect_profile_artifacts(root, job_id):
    """Copy profiler evidence from constrained /tmp locations into one job.

    msprof_perf_summary.py records the final PROF_GROUP path for each case in
    bundle/performance.json.  Only paths created by that script's fixed
    `/tmp/msprof_{ref|asc}_bundle_cN.../PROF_GROUP_*` layout are accepted; this
    prevents a forged performance file from copying arbitrary host data.
    """
    job_dir = resolve_job_dir(root, job_id)
    performance_path = job_dir / "bundle" / "performance.json"
    performance = _read_json(performance_path)
    if not isinstance(performance, dict):
        raise WorkerError(6, "profile collector needs bundle/performance.json")
    rows = performance.get("per_case")
    if not isinstance(rows, list):
        raise WorkerError(6, "profile collector needs performance.json per_case")
    target_root = job_dir / "bundle" / "profiling"
    target_root.mkdir(parents=True, exist_ok=True)
    copied = []
    container_re = re.compile(r"^msprof_(ref|asc)_bundle_c[0-9]+(?:_[A-Za-z0-9._-]+)?$")
    group_re = re.compile(r"^PROF_GROUP_[A-Za-z0-9._-]+$")
    for case_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        for impl, key in (("ref", "ref_prof_dir"), ("asc", "asc_prof_dir")):
            raw = row.get(key)
            if not isinstance(raw, str) or not raw:
                continue
            source = Path(raw)
            try:
                resolved = source.resolve()
            except OSError:
                continue
            if (not source.is_absolute() or source.is_symlink() or not source.is_dir()
                    or resolved.parent.parent != Path("/tmp")
                    or not container_re.match(resolved.parent.name)
                    or not group_re.match(resolved.name)):
                raise WorkerError(6, "unsafe profiler artifact path: %s" % raw)
            # Reuse the source-tree validator to reject links and special files.
            tree_sha256(resolved)
            case_dir = target_root / ("case_%04d_%s" % (case_index, impl))
            case_dir.mkdir(parents=True, exist_ok=True)
            destination = case_dir / resolved.name
            if destination.exists() or destination.is_symlink():
                raise WorkerError(6, "duplicate profiler artifact: %s" % destination)
            shutil.copytree(str(resolved), str(destination))
            copied.append(str(destination.relative_to(job_dir)))
    if not copied:
        raise WorkerError(6, "no PROF_GROUP artifacts found in performance.json")
    return {"job_id": job_id, "copied": copied, "count": len(copied)}


def job_status(root, job_id):
    job_dir = resolve_job_dir(root, job_id)
    state = read_state(job_dir)
    if state is None:
        raise WorkerError(4, "job not found: %s" % job_id)
    # Crash recovery covers both the post-submit/pre-runner window and running.
    if state.get("status") in ("submitted", "running"):
        pid = state.get("pgid") or state.get("runner_pid")
        started = state.get("started_ts") or state.get("created_ts") or _now()
        if (_now() - started) > 5.0 and not _runner_identity_matches(state, job_id):
            def _mut(st):
                if st.get("status") in ("submitted", "running") \
                        and not _runner_identity_matches(st, job_id):
                    st["status"] = "failed"
                    st["failure_class"] = "environment"
                    st["error"] = "runner process died or identity changed (pid=%s)" % pid
                    st["ended_ts"] = _now()
            state = update_state(job_dir, _mut)
    return state


def job_cancel(root, job_id):
    job_dir = resolve_job_dir(root, job_id)
    state = read_state(job_dir)
    if state is None:
        raise WorkerError(4, "job not found: %s" % job_id)
    if state.get("status") in TERMINAL_STATES:
        return {"job_id": job_id, "status": state["status"], "already_terminal": True}

    update_state(job_dir, lambda st: st.__setitem__("cancel_requested", True))
    # submit 之后 runner 可能还没来得及写 pgid，短暂等待其出现
    pgid = state.get("pgid") or state.get("runner_pid")
    if pgid is None:
        for _ in range(50):
            state = read_state(job_dir) or {}
            pgid = state.get("pgid") or state.get("runner_pid")
            if pgid is not None or state.get("status") in TERMINAL_STATES:
                break
            time.sleep(0.1)
    killed = False
    pid_checked = False
    if pgid and _pid_alive(pgid):
        # PID reuse protection: require exact argv and /proc starttime token.
        if _runner_identity_matches(state, job_id):
            pid_checked = True
            try:
                os.killpg(int(pgid), signal.SIGTERM)
                killed = True
            except OSError:
                killed = False
            for _ in range(100):
                if not _pid_alive(pgid):
                    break
                time.sleep(0.1)
            if _pid_alive(pgid):
                state2 = read_state(job_dir) or state
                if _runner_identity_matches(state2, job_id):
                    try:
                        os.killpg(int(pgid), signal.SIGKILL)
                    except OSError:
                        pass
        else:
            return {"job_id": job_id, "status": "running",
                    "error": "pgid %s does not belong to job %s (PID reuse suspected), "
                             "refusing to kill" % (pgid, job_id), "killed": False}
    state = job_status(root, job_id)
    return {"job_id": job_id, "status": state.get("status"), "killed": killed,
            "pid_identity_verified": pid_checked}


def job_cleanup(root, job_id, force=False):
    """只删除单个 job 目录（先做 realpath 包含性校验）；拒绝删除运行中的 job。"""
    job_dir = resolve_job_dir(root, job_id)
    if not job_dir.exists():
        raise WorkerError(4, "job not found: %s" % job_id)
    state = read_state(job_dir)
    if state and state.get("status") not in TERMINAL_STATES:
        if not force:
            raise WorkerError(2, "job %s is not terminal; cancel it first" % job_id)
        job_cancel(root, job_id)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            state = job_status(root, job_id)
            if state.get("status") in TERMINAL_STATES:
                break
            time.sleep(0.1)
        if not state or state.get("status") not in TERMINAL_STATES:
            raise WorkerError(5, "job %s did not stop; refusing cleanup" % job_id)
    jobs_root = (root / "jobs").resolve()
    real = job_dir.resolve()
    if real == jobs_root or jobs_root not in real.parents:
        raise WorkerError(2, "refusing to delete outside jobs/: %s" % job_dir)
    shutil.rmtree(str(real))
    return {"job_id": job_id, "removed": True}


def job_list(root, session_id=None):
    jobs_root = root / "jobs"
    if not jobs_root.is_dir():
        return []
    out = []
    for entry in sorted(jobs_root.iterdir()):
        if not entry.is_dir():
            continue
        state = read_state(entry)
        if state is None:
            continue
        if session_id and state.get("session_id") != session_id:
            continue
        out.append({
            "job_id": state.get("job_id", entry.name),
            "session_id": state.get("session_id"),
            "status": state.get("status"),
            "failure_class": state.get("failure_class"),
            "device_id": state.get("device_id"),
        })
    return out


# ---------------------------------------------------------------------------
# preflight（远端环境检查）
# ---------------------------------------------------------------------------

def _detect_npu_devices(env=None):
    """Discover NPU IDs from the stable mapping table, with a table fallback.

    `npu-smi info` 的主表在不同驱动版本间列布局不同，且温度、功耗等列也包含
    小整数，逐列找数字会把这些值误当设备号。仓库现有 ascendc-env-check 的可靠
    路径是先读 `npu-smi info -m` 的映射表；这里只复制最小的设备 ID 解析逻辑，
    避免远端 worker 依赖仓库里的 Python 模块。
    """
    search_path = (env or {}).get("PATH") if env else None
    npu_smi = shutil.which("npu-smi", path=search_path)
    if not npu_smi:
        return []

    def run_npu_smi(argv):
        try:
            return subprocess.run(argv, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, universal_newlines=True,
                                  timeout=20, env=env)
        except (OSError, subprocess.TimeoutExpired):
            return None

    # Preferred format:
    # NPU ID  Chip ID  Chip Logic ID  Chip Name
    # 0       0        0              Ascend 910B3
    # 0       1        -              Mcu
    mapping = run_npu_smi([npu_smi, "info", "-m"])
    if mapping is not None and mapping.returncode == 0:
        devices = set()
        for line in mapping.stdout.splitlines():
            parts = line.split()
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            # MCU rows have no compute logic ID and must not create a card.
            if parts[2] == "-":
                continue
            devices.add(int(parts[0]))
        if devices:
            return sorted(devices)

    # Last-resort compatibility path for old npu-smi versions without -m.
    # Only accept the leading NPU field followed by a non-empty chip name;
    # never scan every numeric column in the row.
    result = run_npu_smi([npu_smi, "info"])
    if result is None or result.returncode != 0:
        return []
    devices = set()
    for line in result.stdout.splitlines():
        match = re.search(r"^\|\s*(\d+)\s+\S+.*\|", line)
        if match:
            devices.add(int(match.group(1)))
    return sorted(devices)


def preflight(root, devices_arg=None, max_build_slots=None, env_script=None,
              write_config=False, required_paths=None):
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    check("worker_version", True, WORKER_VERSION)
    check("python", sys.version_info >= (3, 6), sys.version.split()[0])

    # remote_root 可写
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / "pool" / "build_slots").mkdir(parents=True, exist_ok=True)
        (root / "jobs").mkdir(parents=True, exist_ok=True)
        probe = root / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
        check("remote_root", True, str(root))
    except OSError as exc:
        check("remote_root", False, str(exc))

    # env_script（可信绝对路径）
    exports = {}
    if env_script:
        try:
            exports = _load_env_script_exports(env_script)
            check("env_script", True, "%s (ASCEND_HOME_PATH=%s)"
                  % (env_script, exports.get("ASCEND_HOME_PATH", "")))
        except WorkerError as exc:
            check("env_script", False, str(exc))
    else:
        check("env_script", False, "not configured")

    merged_env = dict(os.environ)
    merged_env.update(exports)

    # npu-smi is always probed.  A configured list narrows the pool; it does
    # not bypass checking that those cards really exist.
    detected = _detect_npu_devices(merged_env)
    check("npu-smi", bool(detected), "detected: %s" % detected)
    if devices_arg:
        devices = [int(d) for d in devices_arg]
        missing = sorted(set(devices) - set(detected))
        check("npu_devices", bool(devices) and not missing,
              "configured: %s; missing from npu-smi: %s" % (devices, missing))
    else:
        devices = detected

    # 构建工具
    for tool in ("cmake", "make", "g++", "bash", "msprof"):
        found = shutil.which(tool, path=merged_env.get("PATH"))
        check("tool_%s" % tool, bool(found), found or "not found")
    try:
        pip_probe = subprocess.run([sys.executable, "-m", "pip", "--version"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True, timeout=30, env=merged_env)
        check("pip", pip_probe.returncode == 0,
              (pip_probe.stdout or pip_probe.stderr).strip()[:500])
    except (OSError, subprocess.TimeoutExpired) as exc:
        check("pip", False, str(exc))
    for required_path in required_paths or []:
        path = Path(required_path)
        check("required_path", path.is_file(), str(path))

    pool = DevicePool(root)
    if write_config:
        cfg = pool.write_config(devices, max_build_slots or 4, env_script)
        check("pool_config", True, json.dumps(cfg, sort_keys=True))
    else:
        cfg = pool.config()
        check("pool_config", bool(cfg.get("devices")),
              json.dumps(cfg, sort_keys=True) if cfg else "not initialized")
    ok_all = all(c["ok"] for c in checks)
    _emit({"checks": checks, "ok": ok_all, "devices": devices,
           "pool": pool.status()})
    return ok_all


def run_job_guarded(root, job_id):
    """Ensure an unexpected runner exception always becomes a terminal state."""
    try:
        run_job(root, job_id)
    except Exception as exc:
        message = "unexpected runner error: %s" % exc
        try:
            job_dir = resolve_job_dir(root, job_id)
            state = read_state(job_dir) or {}
            if state.get("status") not in TERMINAL_STATES:
                try:
                    _finish_job(job_dir, "failed", "environment", message)
                except Exception:
                    def _mut(st):
                        st["status"] = "failed"
                        st["failure_class"] = "environment"
                        st["error"] = message
                        st["ended_ts"] = _now()
                    update_state(job_dir, _mut)
        finally:
            _emit({"ok": False, "job_id": job_id, "status": "failed",
                   "failure_class": "environment", "error": message})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        prog="remote_worker.py",
        description="lingxi-remote-execution remote-side worker (stdlib only)")
    parser.add_argument("--root", required=True, help="remote root directory (absolute)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("ping", help="liveness probe")

    p = sub.add_parser("preflight", help="check remote environment")
    p.add_argument("--devices", default=None, help="comma-separated device ids (default: npu-smi auto)")
    p.add_argument("--max-build-slots", type=int, default=4)
    p.add_argument("--env-script", default=None, help="trusted absolute path to CANN env script")
    p.add_argument("--write-config", action="store_true", help="persist pool_config.json")
    p.add_argument("--required-path", action="append", default=[],
                   help="remote evaluator/profiler path that must exist (repeatable)")

    sub.add_parser("pool-list", help="list device pool and leases")

    p = sub.add_parser("lease-acquire", help="acquire device lease (atomic, sticky)")
    p.add_argument("--session", required=True)
    p.add_argument("--owner", required=True)
    p.add_argument("--ttl", type=float, default=3600.0, help="lease TTL seconds")
    p.add_argument("--timeout", type=float, default=60.0, help="max wait for a free device")
    p.add_argument("--device", type=int, default=None, help="preferred device id")
    p.add_argument("--no-sticky", action="store_true", help="do not re-attach existing session lease")

    p = sub.add_parser("lease-heartbeat", help="refresh lease heartbeat")
    p.add_argument("--lease-id", required=True)

    p = sub.add_parser("lease-release", help="release lease explicitly")
    p.add_argument("--lease-id", required=True)

    p = sub.add_parser("upload-prepare", help="create an isolated source upload staging dir")
    p.add_argument("--job-id", required=True)
    p.add_argument("--source-digest", required=True)
    p.add_argument("--upload-token", required=True)

    p = sub.add_parser("upload-discard", help="discard one isolated source upload")
    p.add_argument("--job-id", required=True)
    p.add_argument("--source-digest", required=True)
    p.add_argument("--upload-token", required=True)

    p = sub.add_parser("job-submit", help="submit job (idempotent on job_id)")
    p.add_argument("--spec-b64", required=True, help="base64-encoded JSON spec")

    p = sub.add_parser("job-status", help="query job state")
    p.add_argument("--job-id", required=True)

    p = sub.add_parser("job-cancel", help="cancel job (kills only its process group)")
    p.add_argument("--job-id", required=True)

    p = sub.add_parser("job-list", help="list jobs (optionally by session)")
    p.add_argument("--session", default=None)

    p = sub.add_parser("job-cleanup", help="remove a single job directory")
    p.add_argument("--job-id", required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("_job-runner", help="internal: detached job runner")
    p.add_argument("--job-id", required=True)

    p = sub.add_parser("_profile-collect", help="internal: collect constrained msprof artifacts")
    p.add_argument("--job-id", required=True)

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.error("a command is required")
    root_str = args.root
    if not os.path.isabs(root_str):
        _emit({"ok": False, "error": "--root must be absolute", "code": 2})
        return 2
    root = Path(root_str)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _emit({"ok": False, "error": "cannot create root: %s" % exc, "code": 6})
        return 6

    try:
        if args.command == "ping":
            _emit({"pong": True, "version": WORKER_VERSION,
                   "python": sys.version.split()[0], "root": str(root)})
        elif args.command == "preflight":
            devices = None
            if args.devices:
                devices = [d.strip() for d in args.devices.split(",") if d.strip()]
            if not preflight(root, devices, args.max_build_slots, args.env_script,
                             args.write_config, args.required_path):
                return 6
        elif args.command == "pool-list":
            _emit({"pool": DevicePool(root).status(),
                   "config": DevicePool(root).config()})
        elif args.command == "lease-acquire":
            pool = DevicePool(root)
            lease = pool.acquire(args.session, args.owner, args.ttl, args.timeout,
                                 preferred=args.device,
                                 sticky=not args.no_sticky)
            _emit({"lease": lease})
        elif args.command == "lease-heartbeat":
            _emit({"lease": DevicePool(root).heartbeat(args.lease_id)})
        elif args.command == "lease-release":
            _emit(DevicePool(root).release(args.lease_id))
        elif args.command == "upload-prepare":
            _emit(upload_prepare(root, args.job_id, args.source_digest,
                                 args.upload_token))
        elif args.command == "upload-discard":
            _emit(upload_discard(root, args.job_id, args.source_digest,
                                 args.upload_token))
        elif args.command == "job-submit":
            _emit(submit_job(root, args.spec_b64))
        elif args.command == "job-status":
            _emit({"state": job_status(root, args.job_id)})
        elif args.command == "job-cancel":
            _emit(job_cancel(root, args.job_id))
        elif args.command == "job-list":
            _emit({"jobs": job_list(root, args.session)})
        elif args.command == "job-cleanup":
            _emit(job_cleanup(root, args.job_id, args.force))
        elif args.command == "_job-runner":
            run_job_guarded(root, args.job_id)
        elif args.command == "_profile-collect":
            _emit(collect_profile_artifacts(root, args.job_id))
        else:
            _emit({"ok": False, "error": "unknown command", "code": 2})
            return 2
    except WorkerError as exc:
        _emit({"ok": False, "error": str(exc), "code": exc.code})
        return exc.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
