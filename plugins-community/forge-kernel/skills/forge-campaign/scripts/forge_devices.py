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
"""多卡实验的设备绑定与并发栅栏。

单机多卡环境下要让多个算子同时演进，有三条约束：

  一卡一 campaign   两个 campaign 共享一张卡会互相干扰计时，而论文的
                    same-job 配对计时正是要避免这个。卡不够时排队，
                    不要塞。

  代际不换卡        一个算子从 gen0 到 genN 绑定同一张卡。换卡之后
                    参考签名、噪声底、绝对耗时全部不可比，而这些
                    不可比是**静默**的 —— 数字照常产出，只是错了。

  满载后再量噪声底  这条最容易漏。N 个 campaign 同时启动时，第一个量
                    噪声底的那一刻其余几个可能还在初始化、还没压上卡，
                    量出来偏低。之后满载运行，真实噪声高于记录值，
                    于是噪声里的"改进"被当成真的收下，还会写进知识库
                    和产物库污染后续各代。

                    所以 bind 之后要过一道栅栏：等所有同批 campaign 都
                    绑定完成，再各自量噪声底。

设备租约本身复用 ops-evaluation 的 device_lease（session 级 flock +
eval 级排队锁，自动挑显存占用最低的空闲卡，能清理 stale lease）。
本脚本只补 forge-kernel 需要而那边没有的两件事：并发栅栏，以及把
「持 eval 锁跑一条命令」做成一个可直接调用的入口。

用法:
    # 绑卡（整个 campaign 生命周期占住它）
    forge_devices.py bind --pool ./device_pool --operator add_layer_norm \\
        --session-id ln_b8s4096 --concurrency 8

    # 栅栏：等同批 8 个都绑完，再往下走
    forge_devices.py wait-peers --pool ./device_pool --expected 8

    # 持 eval 锁跑评测（同一张卡上的评估串行排队）
    forge_devices.py run --pool ./device_pool -- \\
        python3 evaluate_ops_direct.py ...

    forge_devices.py status --pool ./device_pool
    forge_devices.py release --pool ./device_pool --session-id ln_b8s4096

退出码: 0 = 正常; 1 = 超时 / 栅栏未达成; 2 = 用法或环境错误
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PEERS_DIRNAME = "forge_peers"

#: 同批 campaign 等待彼此绑定的默认上限。超过说明有 campaign 起不来，
#: 继续等下去不如报出来 —— 少一个 peer 就意味着噪声底会在偏低的并发下量出。
DEFAULT_BARRIER_TIMEOUT = 900.0

#: peer 记录多久算过期。campaign 可能跑很久，所以这个值要宽松；
#: 它只用于清理崩溃进程留下的残留。
DEFAULT_STALE_AFTER = 48 * 3600.0


class DeviceError(Exception):
    """设备绑定失败。"""


# ---------------------------------------------------------------------------
# 复用 ops-evaluation 的租约实现
# ---------------------------------------------------------------------------

def _candidate_lease_paths() -> List[Path]:
    """device_lease.py 可能在的位置。

    安装后所有 skill 被摊平到同一个 skills/ 目录；开发时它在
    ops-perf-evolution 的插件目录下。两种布局都要认 —— 只写一种的话，
    换个环境就会以「找不到模块」收场，而那个报错不会说清该装什么。
    """
    here = Path(__file__).resolve().parent          # .../forge-campaign/scripts
    skills_root = here.parent.parent                 # .../skills
    return [
        skills_root / "ops-evaluation" / "scripts",
        skills_root.parent.parent / "ops-perf-evolution" / "skills"
        / "ops-evaluation" / "scripts",
    ]


def load_device_lease():
    """导入 device_lease 模块，找不到时说清楚缺的是什么。"""
    for path in _candidate_lease_paths():
        if (path / "device_lease.py").is_file():
            sys.path.insert(0, str(path))
            import device_lease  # noqa: PLC0415
            return device_lease
    searched = "\n  ".join(str(p) for p in _candidate_lease_paths())
    raise DeviceError(
        "找不到 device_lease.py。它属于 ops-evaluation skill，"
        "forge-kernel 通过 marketplace 的 forge-kernel-borrowed-skills 依赖它。\n"
        f"已查找:\n  {searched}\n"
        "若是手工部署，确认 ops-evaluation 与本 skill 装在同一个 skills/ 下。"
    )


# ---------------------------------------------------------------------------
# 并发栅栏
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def peers_dir(pool: Path) -> Path:
    return pool / PEERS_DIRNAME


def register_peer(pool: Path, session_id: str, operator: str,
                  device_id: int, concurrency: Optional[int]) -> Path:
    """登记自己已绑卡。栅栏靠数这些记录判断同批有几个就位了。"""
    directory = peers_dir(pool)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session_id}.json"
    path.write_text(json.dumps({
        "session_id": session_id, "operator": operator,
        "device_id": device_id, "concurrency": concurrency,
        "pid": os.getpid(), "bound_at": _now(), "bound_ts": time.time(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def unregister_peer(pool: Path, session_id: str) -> None:
    try:
        (peers_dir(pool) / f"{session_id}.json").unlink(missing_ok=True)
    except OSError:
        pass


def list_peers(pool: Path,
               stale_after: float = DEFAULT_STALE_AFTER) -> List[Dict[str, Any]]:
    """列出当前已绑卡的同批 campaign。过期记录会被清掉。"""
    directory = peers_dir(pool)
    if not directory.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    now = time.time()
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        bound_ts = record.get("bound_ts")
        if isinstance(bound_ts, (int, float)) and now - bound_ts > stale_after:
            # 崩溃进程留下的残留。留着会让栅栏误判为已满员。
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        out.append(record)
    return out


def wait_for_peers(pool: Path, expected: int,
                   timeout: float = DEFAULT_BARRIER_TIMEOUT,
                   poll: float = 2.0) -> List[Dict[str, Any]]:
    """等同批 campaign 都绑定完成。

    这道栅栏唯一的作用是让噪声底在**真实并发水平**下量出来。少等一个，
    噪声底就偏低，而偏低的噪声底会让噪声里的改动被当成真的收下 ——
    那种错误是静默的，之后污染知识库与产物库。
    """
    deadline = time.time() + timeout
    while True:
        peers = list_peers(pool)
        if len(peers) >= expected:
            return peers
        if time.time() >= deadline:
            bound = [f"{p['operator']}@dev{p['device_id']}" for p in peers]
            raise DeviceError(
                f"等待同批 campaign 超时：期望 {expected} 个，"
                f"{timeout:.0f}s 内只等到 {len(peers)} 个。\n"
                f"  已就位: {bound or '（无）'}\n"
                "  少一个 peer 意味着噪声底会在偏低的并发下量出，"
                "之后满载运行时真实噪声更高，噪声里的「改进」会被收下。\n"
                "  要么把起不来的 campaign 查清楚，要么把 --expected 调成"
                "实际并发数后重来。"
            )
        time.sleep(poll)


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------

def cmd_bind(args: argparse.Namespace) -> int:
    pool = Path(args.pool)
    try:
        device_lease = load_device_lease()
    except DeviceError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    pool.mkdir(parents=True, exist_ok=True)
    lease = device_lease.SessionDeviceLease(pool_dir=str(pool))
    device_ids = ([int(x) for x in args.device_ids.split(",")]
                  if args.device_ids else None)
    try:
        device_id = lease.acquire_session(
            op_name=args.operator, session_id=args.session_id,
            device_ids=device_ids, timeout=args.timeout,
            preferred_device=args.device_id,
        )
    except Exception as exc:  # noqa: BLE001 - 上游抛什么都要说清
        print(f"错误: 取卡失败: {exc}\n"
              "  一卡一 campaign，卡被占满时会等待。"
              "共享一张卡会互相干扰计时，所以这里排队而不是塞。",
              file=sys.stderr)
        return 1

    register_peer(pool, args.session_id, args.operator, device_id,
                  args.concurrency)

    print(f"已绑定 device {device_id}")
    print(f"  算子    {args.operator}")
    print(f"  session {args.session_id}")
    if args.concurrency:
        print(f"  并发    计划 {args.concurrency} 个")
        print()
        print("下一步：等同批都绑完再量噪声底，否则会在偏低的并发下量出")
        print(f"  forge_devices.py wait-peers --pool {args.pool} "
              f"--expected {args.concurrency}")
    if args.as_json:
        print(json.dumps({"device_id": device_id,
                          "session_id": args.session_id}, ensure_ascii=False))
    return 0


def cmd_wait_peers(args: argparse.Namespace) -> int:
    try:
        peers = wait_for_peers(Path(args.pool), args.expected,
                               timeout=args.timeout)
    except DeviceError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    print(f"同批 {len(peers)} 个 campaign 已全部绑定：")
    for peer in sorted(peers, key=lambda p: p.get("device_id", -1)):
        print(f"  device {peer['device_id']:<3} {peer['operator']}")
    print("\n现在可以各自量噪声底 —— 此刻的并发水平就是之后运行时的水平。")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """持 eval 锁跑一条命令。

    同一张卡上的评估必须串行：两次评测并发会互相干扰计时，而配对计时的
    全部意义就在于两个 kernel 处在相同条件下。
    """
    if not args.argv:
        print("错误: 需要在 -- 之后给出要执行的命令", file=sys.stderr)
        return 2
    try:
        device_lease = load_device_lease()
    except DeviceError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    lease = device_lease.SessionDeviceLease(pool_dir=str(Path(args.pool)))
    try:
        with lease.eval_lock(timeout=args.timeout):
            proc = subprocess.run(args.argv, check=False)
            return proc.returncode
    except Exception as exc:  # noqa: BLE001
        print(f"错误: 取 eval 锁失败: {exc}\n"
              "  同一张卡上的评估要串行 —— 并发评测会互相干扰计时。",
              file=sys.stderr)
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    pool = Path(args.pool)
    peers = list_peers(pool)
    try:
        device_lease = load_device_lease()
        lease = device_lease.SessionDeviceLease(pool_dir=str(pool))
        available = lease.list_available()
    except Exception:  # noqa: BLE001 - 环境问题不该让 status 挂掉
        available = None

    if args.as_json:
        print(json.dumps({"peers": peers, "available": available},
                         ensure_ascii=False, indent=2))
        return 0

    print(f"已绑定 {len(peers)} 个 campaign")
    for peer in sorted(peers, key=lambda p: p.get("device_id", -1)):
        print(f"  device {peer['device_id']:<3} {peer['operator']:<24} "
              f"session {peer['session_id']}")
    if available is not None:
        print(f"\n空闲设备: {available or '（无）'}")
    else:
        print("\n空闲设备: 探测失败（npu-smi 不可用？）")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    pool = Path(args.pool)
    try:
        device_lease = load_device_lease()
    except DeviceError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    lease = device_lease.SessionDeviceLease(pool_dir=str(pool))
    lease.adopt_session(args.session_id, args.device_id)
    lease.release_session()
    unregister_peer(pool, args.session_id)
    print(f"已释放 session {args.session_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pool", default=argparse.SUPPRESS)
    common.add_argument("--json", action="store_true", dest="as_json",
                        default=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(
        description="多卡实验的设备绑定与并发栅栏")
    parser.add_argument("--pool", default="./device_pool",
                        help="设备池目录（也可写在子命令后面）")
    parser.add_argument("--json", action="store_true", dest="as_json")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bind = sub.add_parser("bind", help="为一个 campaign 绑定一张卡",
                            parents=[common])
    p_bind.add_argument("--operator", required=True)
    p_bind.add_argument("--session-id", required=True,
                        help="整个 campaign 生命周期用同一个，代际之间不要换")
    p_bind.add_argument("--device-id", type=int, default=None,
                        help="指定卡号；不给则自动挑显存占用最低的空闲卡")
    p_bind.add_argument("--device-ids", default=None,
                        help="可用卡范围，逗号分隔，如 0,1,2,3")
    p_bind.add_argument("--concurrency", type=int, default=None,
                        help="同批计划并发数。给了会提示接下来要过栅栏")
    p_bind.add_argument("--timeout", type=float, default=60.0)
    p_bind.set_defaults(func=cmd_bind)

    p_wait = sub.add_parser("wait-peers", help="等同批 campaign 都绑定完成",
                            parents=[common])
    p_wait.add_argument("--expected", type=int, required=True)
    p_wait.add_argument("--timeout", type=float,
                        default=DEFAULT_BARRIER_TIMEOUT)
    p_wait.set_defaults(func=cmd_wait_peers)

    p_run = sub.add_parser("run", help="持 eval 锁跑一条命令",
                           parents=[common])
    p_run.add_argument("--timeout", type=float, default=600.0)
    p_run.add_argument("argv", nargs=argparse.REMAINDER,
                       help="-- 之后是要执行的命令")
    p_run.set_defaults(func=cmd_run)

    p_sta = sub.add_parser("status", help="看当前绑定情况", parents=[common])
    p_sta.set_defaults(func=cmd_status)

    p_rel = sub.add_parser("release", help="释放设备租约", parents=[common])
    p_rel.add_argument("--session-id", required=True)
    p_rel.add_argument("--device-id", type=int, default=None)
    p_rel.set_defaults(func=cmd_release)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    # argparse.REMAINDER 会把 '--' 本身也收进来
    argv = getattr(args, "argv", None)
    if isinstance(argv, list) and argv and argv[0] == "--":
        args.argv = argv[1:]
    try:
        return args.func(args)
    except DeviceError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
