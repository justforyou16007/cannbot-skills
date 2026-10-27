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
"""参考实现适配：把三种来源归一化成评测脚本能吃的一个路径。

ForgeKernel 对参考实现只有一个要求：**能测，不能读**。满足这条的来源在
昇腾上有三种，但下游的 evaluate_ops_direct.py 只认一种输入形态 ——
一个 OPP 安装路径。本脚本负责这层转换。

  builtin_aclnn   CANN 算子库里的现成实现。它天然是封闭的：我们通过
                  aclnn 接口调它、计时、采集，但从不接触它的源码。
                  转换方式是给出一个不注册任何自定义算子的 OPP 目录，
                  让派发落到厂商内置实现上。

  opp_install     已构建好的自定义算子安装路径（ops 仓算子）。
                  它有源码，所以「封闭」要靠 seal_reference.py 登记
                  禁读清单来人为维持。

  sealed_kernel   ForgeKernel 自己上一代的产出。论文 §3.5 的自我提升
                  就建立在这上面 —— 参考既然只被当作可测量的二进制，
                  那它是谁写的就无关紧要。

关于 builtin_aclnn 的一个重要说明
---------------------------------
「把 ASCEND_CUSTOM_OPP_PATH 指向一个空的 OPP 目录，aclnn 调用就会落到
内置实现」这个行为，**本脚本的作者没有在真机上验证过**。它是依据
ASCEND_CUSTOM_OPP_PATH 的用途做出的推断，不是查证过的结论。

所以本脚本不假设它成立，而是提供 `verify` 子命令把它变成一次显式检查：
在目标机器上跑一次，拿到确切结论。verify 不通过时请改用 opp_install
来源，或把 verify 的输出回贴以便调整。

用法:
    python3 reference_adapter.py prepare --kind builtin_aclnn \\
        --op-name add_layer_norm --work-dir ./forge_work
    python3 reference_adapter.py prepare --kind opp_install \\
        --op-name add_layer_norm --path /abs/path/to/install --work-dir ./forge_work
    python3 reference_adapter.py verify --spec ./forge_work/reference_spec.json

退出码: 0 = 通过; 1 = 预检未通过; 2 = 用法或 IO 错误
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SPEC_FILENAME = "reference_spec.json"

KIND_BUILTIN = "builtin_aclnn"
KIND_OPP = "opp_install"
KIND_SEALED = "sealed_kernel"
ALL_KINDS = (KIND_BUILTIN, KIND_OPP, KIND_SEALED)

#: 空 OPP 骨架的目录名。放在 work-dir 下，不污染用户的其他目录。
NULL_OPP_DIRNAME = "null_opp"

#: evaluate_ops_direct.py 经由 detect_vendor_subdir 寻找的 vendors 子目录名。
#: 这里用它的默认回退值，保证目录结构能被识别。
NULL_VENDOR_SUBDIR = "custom_nn"


class AdapterError(Exception):
    """适配失败，消息应当直接可操作。"""


# ---------------------------------------------------------------------------
# 诊断输出
# ---------------------------------------------------------------------------

def _diag(title: str, lines: List[str]) -> str:
    """拼一段便于直接回贴的诊断信息。

    排障时最耗时的是「现场信息不全」，所以这里把判断依据一并打出来：
    看的是哪个变量、哪个路径、存不存在。
    """
    body = "\n".join(f"  {line}" for line in lines)
    return f"{title}\n{body}"


def _env_snapshot() -> List[str]:
    keys = ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH",
            "ASCEND_CUSTOM_OPP_PATH", "ASCEND_RT_VISIBLE_DEVICES",
            "ASCEND_DEVICE_ID", "LD_LIBRARY_PATH")
    out = []
    for key in keys:
        value = os.environ.get(key)
        out.append(f"{key}={value if value else '<未设置>'}")
    return out


# ---------------------------------------------------------------------------
# 空 OPP 骨架
# ---------------------------------------------------------------------------

def build_null_opp(work_dir: Path) -> Path:
    """搭一个不注册任何自定义算子的 OPP 目录。

    目的是让 evaluate_ops_direct.py 在「baseline」一侧什么自定义算子也
    加载不到，从而由 CANN 内置实现接管这次调用。

    结构照着 detect_vendor_subdir 的识别规则搭：它找 <path>/vendors/ 下
    以 custom 开头的子目录，所以这里建一个同名空目录。
    """
    root = work_dir / NULL_OPP_DIRNAME
    vendor = root / "vendors" / NULL_VENDOR_SUBDIR
    vendor.mkdir(parents=True, exist_ok=True)

    marker = root / "README.md"
    marker.write_text(
        "# 空 OPP 目录（ForgeKernel 自动生成）\n\n"
        "本目录**故意不包含任何自定义算子**。\n\n"
        "它的用途是把 `ASCEND_CUSTOM_OPP_PATH` 指到一个注册为空的位置，\n"
        "使 aclnn 调用落到 CANN 算子库的内置实现上 —— 那个实现就是本次\n"
        "campaign 的封闭参考：只测，不读源码。\n\n"
        "请勿在此目录下放置任何算子产物，那会让参考不再是内置实现，\n"
        "后续所有加速比都会失去意义。\n",
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def prepare(kind: str, op_name: str, work_dir: Path,
            path: Optional[str] = None,
            reference_id: Optional[str] = None) -> Dict[str, Any]:
    """产出 reference_spec.json，供后续所有步骤引用同一个参考定义。"""
    if kind not in ALL_KINDS:
        raise AdapterError(f"未知参考来源 '{kind}'；可选: {list(ALL_KINDS)}")
    work_dir.mkdir(parents=True, exist_ok=True)

    if kind == KIND_BUILTIN:
        baseline_path = build_null_opp(work_dir)
        verified = False
        note = (
            "内置 aclnn 实现。本来源依赖「空 OPP 目录使派发落到内置实现」"
            "这一行为，必须先跑 verify 在目标机器上确认，不可想当然。"
        )
    else:
        if not path:
            raise AdapterError(f"来源 '{kind}' 必须提供 --path")
        baseline_path = Path(path)
        if not baseline_path.is_absolute():
            raise AdapterError(
                f"--path 必须是绝对路径（收到 '{path}'）；"
                "相对路径在子进程切换工作目录后会解析错位"
            )
        if not baseline_path.is_dir():
            raise AdapterError(f"参考路径不存在或不是目录: {baseline_path}")
        vendors = baseline_path / "vendors"
        if not vendors.is_dir():
            raise AdapterError(_diag(
                f"参考路径下缺少 vendors/ 目录: {baseline_path}",
                [
                    "OPP 安装路径应当形如 <install>/vendors/<custom_*>/",
                    f"实际内容: {sorted(p.name for p in baseline_path.iterdir())[:10]}",
                    "若这是 ops 仓构建产物，请确认 .run 包已用 --install-path 安装完成",
                ],
            ))
        verified = True
        note = "已构建的算子安装路径。源码对我们可见，封闭性靠禁读清单维持。"

    spec = {
        "schema_version": "1.0",
        "kind": kind,
        "op_name": op_name,
        "reference_id": reference_id or op_name,
        "baseline_path": str(baseline_path.resolve()),
        "sealed": True,
        "dispatch_verified": verified,
        "note": note,
    }
    spec_path = work_dir / SPEC_FILENAME
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return spec


def cmd_prepare(args: argparse.Namespace) -> int:
    try:
        spec = prepare(
            kind=args.kind, op_name=args.op_name,
            work_dir=Path(args.work_dir), path=args.path,
            reference_id=args.reference_id,
        )
    except AdapterError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    print(f"参考定义已写入 {Path(args.work_dir) / SPEC_FILENAME}")
    print(f"  来源        {spec['kind']}")
    print(f"  算子        {spec['op_name']}")
    print(f"  baseline    {spec['baseline_path']}")
    print(f"  派发已验证  {'是' if spec['dispatch_verified'] else '否'}")
    if not spec["dispatch_verified"]:
        print()
        print("下一步必须先验证派发行为：")
        print(f"  python3 {Path(__file__).name} verify "
              f"--spec {Path(args.work_dir) / SPEC_FILENAME}")
        print("未验证就开跑，可能把「其实没调到内置实现」当成了参考基线。")
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def check_cann_env() -> List[Tuple[bool, str]]:
    """逐项检查 CANN 环境。返回 (是否通过, 说明)。

    每项都给出「看的是什么」，这样失败信息可以直接用来排障。
    """
    checks: List[Tuple[bool, str]] = []

    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if home and Path(home).is_dir():
        checks.append((True, f"CANN 安装路径: {home}"))
    else:
        fallback = Path("/usr/local/Ascend/ascend-toolkit/latest")
        if fallback.is_dir():
            checks.append((
                True,
                f"CANN 安装路径: {fallback}（环境变量未设置，用默认路径探到）；"
                "建议 source <CANN>/set_env.sh 后再跑"
            ))
        else:
            checks.append((
                False,
                "未找到 CANN 安装：ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME 都未指向有效目录，"
                f"默认路径 {fallback} 也不存在。请先 source <CANN>/set_env.sh"
            ))

    npu_smi = shutil.which("npu-smi")
    if npu_smi:
        checks.append((True, f"npu-smi: {npu_smi}"))
    else:
        checks.append((
            False,
            "未找到 npu-smi，无法确认设备可用。它通常在 /usr/local/Ascend/driver/tools/"
        ))

    try:
        import torch_npu  # noqa: F401
        checks.append((True, "torch_npu 可导入"))
    except ImportError as exc:
        checks.append((
            False,
            f"torch_npu 不可导入（{exc}）。评测脚本依赖它调用算子与计时"
        ))

    return checks


def check_device() -> Tuple[bool, str]:
    """确认至少有一张卡可见。"""
    npu_smi = shutil.which("npu-smi")
    if not npu_smi:
        return False, "npu-smi 不可用，跳过设备探测"
    try:
        proc = subprocess.run(
            [npu_smi, "info"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"执行 npu-smi info 失败: {exc}"
    if proc.returncode != 0:
        return False, (
            f"npu-smi info 返回码 {proc.returncode}\n"
            f"  stdout: {proc.stdout.strip()[:300]}\n"
            f"  stderr: {proc.stderr.strip()[:300]}"
        )
    return True, "npu-smi info 执行成功，设备可见"


def cmd_verify(args: argparse.Namespace) -> int:
    """在目标机器上把「参考能不能用」变成一次有确切结论的检查。"""
    spec_path = Path(args.spec)
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"错误: 无法读取 {spec_path}: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"错误: {spec_path} 不是合法 JSON: {exc}", file=sys.stderr)
        return 2

    print(f"验证参考定义: {spec_path}")
    print(f"  来源 {spec.get('kind')}  算子 {spec.get('op_name')}\n")

    failures: List[str] = []

    print("— CANN 环境 —")
    for ok, message in check_cann_env():
        print(f"  [{'通过' if ok else '失败'}] {message}")
        if not ok:
            failures.append(message)

    print("\n— 设备 —")
    ok, message = check_device()
    print(f"  [{'通过' if ok else '失败'}] {message}")
    if not ok:
        failures.append(message)

    print("\n— 参考路径 —")
    baseline = Path(spec.get("baseline_path", ""))
    if baseline.is_dir():
        print(f"  [通过] {baseline} 存在")
        vendors = baseline / "vendors"
        subdirs = sorted(p.name for p in vendors.iterdir()) if vendors.is_dir() else []
        print(f"  [信息] vendors/ 子目录: {subdirs or '（空）'}")
        if spec.get("kind") == KIND_BUILTIN and any(
                (vendors / d).is_dir() and any((vendors / d).iterdir())
                for d in subdirs):
            msg = (
                f"参考声明为内置实现，但 {vendors} 下的子目录非空。"
                "这会让派发落到自定义算子而非内置实现，参考基线将失去意义。"
                "请清空该目录，或改用 opp_install 来源"
            )
            print(f"  [失败] {msg}")
            failures.append(msg)
    else:
        msg = f"参考路径不存在: {baseline}"
        print(f"  [失败] {msg}")
        failures.append(msg)

    if spec.get("kind") == KIND_BUILTIN:
        print("\n— 派发行为 —")
        print("  [待确认] 「空 OPP 目录使 aclnn 落到内置实现」这一行为尚未在本机证实。")
        print("           确认方式：跑一次 evaluate_ops_direct.py，令 baseline 与")
        print("           evolved 都指向本参考路径，再用 msprof 采集，查看")
        print("           kernel_details 中出现的算子名是否为厂商内置实现的名字。")
        print("           若出现的是自定义算子名，说明派发没有按预期落下去。")

    print()
    if failures:
        # 先把上面的明细刷出去，否则摘要会插在明细前面，回贴时顺序混乱
        sys.stdout.flush()
        print(_diag(
            f"预检未通过：{len(failures)} 项失败",
            ["环境快照："] + _env_snapshot(),
        ), file=sys.stderr)
        print("\n把以上整段输出回贴即可定位问题。", file=sys.stderr)
        return 1

    print("预检通过。")
    if spec.get("kind") == KIND_BUILTIN:
        print("注意：派发行为仍需按上面的方式确认一次，确认后"
              "把 reference_spec.json 的 dispatch_verified 置为 true。")
    return 0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把三种参考来源归一化成评测脚本能吃的 baseline 路径",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="生成 reference_spec.json")
    p_prep.add_argument("--kind", required=True, choices=list(ALL_KINDS))
    p_prep.add_argument("--op-name", required=True)
    p_prep.add_argument("--work-dir", required=True)
    p_prep.add_argument("--path", default=None,
                        help="opp_install / sealed_kernel 的绝对安装路径")
    p_prep.add_argument("--reference-id", default=None,
                        help="参考标识，默认取算子名")
    p_prep.set_defaults(func=cmd_prepare)

    p_ver = sub.add_parser("verify", help="在目标机器上预检参考是否可用")
    p_ver.add_argument("--spec", required=True)
    p_ver.set_defaults(func=cmd_verify)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except AdapterError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
