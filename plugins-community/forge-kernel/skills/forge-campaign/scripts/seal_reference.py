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
"""封存参考：把「不能读」从一句口号变成两份可执行的清单。

论文有一条铁律：no code crosses the boundary —— 参考实现的源码不得进入
agent 的上下文，而且这条规则要由确定性的 driver 在提示词之外强制执行。
原因很直接：如果 agent 读了参考的源码，那它就不是在「从硬件行为反推架构」，
而是在抄。抄来的结论无法迁移到没有开源实现的算子上，整套方法的价值就没了。

光在提示词里写「不要读参考源码」是不够的 —— 提示词会被压缩、被改写、
被遗忘，而且无法阻止一次无心的 grep。所以封存动作产出两份清单，交给
两个独立于提示词的执行点：

  sealed_paths.txt   禁读路径。交给 PreToolUse hook，在工具调用层面拦截
                     对这些路径的读取。

  denylist.txt       专有名词。交给知识库 lint，阻止参考特有的符号名
                     混进可复用经验里 —— 那会让经验绑死在这个参考上。

此外记录一份 manifest（含校验和），用于事后核查封存内容有没有被动过。

用法:
    python3 seal_reference.py seal --artifact /path/to/ref_impl \\
        --op-name add_layer_norm --work-dir ./forge_work
    python3 seal_reference.py manifest --work-dir ./forge_work
    python3 seal_reference.py check --work-dir ./forge_work --path /some/file

退出码: 0 = 正常 / 未越界; 1 = 越界或校验失败; 2 = 用法或 IO 错误
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

MANIFEST_FILENAME = "sealed_manifest.json"
SEALED_PATHS_FILENAME = "sealed_paths.txt"
DENYLIST_FILENAME = "denylist.txt"

#: 会被当作源码的扩展名。封存时这些文件的内容要被禁读。
SOURCE_SUFFIXES = {
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx",
    ".cu", ".cuh", ".py", ".s", ".asm", ".inc", ".tiling",
}

#: 从源码里抽标识符用的模式。只取足够有辨识度的：类名、函数名、宏名。
RE_IDENTIFIER = re.compile(
    r"\b(?:class|struct)\s+([A-Z][A-Za-z0-9_]{4,})"
    r"|#define\s+([A-Z][A-Z0-9_]{5,})"
    r"|\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+){2,})\b"
)

#: 太通用的词即使形态符合也不该进禁用名单，否则会把正常文字全误伤。
COMMON_WORDS = {
    "DataCopy", "LocalTensor", "GlobalTensor", "TPipe", "TQue", "TBuf",
    "AscendC", "PlatformAscendC", "TilingData", "GetBlockIdx", "SetGlobalBuffer",
    "DeQueue", "EnQueue", "AllocTensor", "FreeTensor", "InitBuffer",
}

#: 单个文件读取上限，避免把超大文件整个读进内存。
MAX_SCAN_BYTES = 2 * 1024 * 1024


class SealError(Exception):
    """封存失败。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_identifiers(text: str) -> Set[str]:
    """从源码文本里抽出有辨识度的标识符。

    这份名单的作用是让 lint 能发现「经验卡片里抄了参考的内部命名」。
    宁可少抽一些也不要把通用 API 名抽进来 —— 误伤会让 lint 变成噪声，
    最后被人忽略，那比没有 lint 更糟。
    """
    found: Set[str] = set()
    for match in RE_IDENTIFIER.finditer(text):
        for group in match.groups():
            if not group:
                continue
            if group in COMMON_WORDS:
                continue
            if len(group) < 6:
                continue
            found.add(group)
    return found


def seal(artifact: Path, op_name: str, work_dir: Path) -> Dict[str, Any]:
    """封存一份参考产物，产出禁读清单、专有名词清单与 manifest。"""
    if not artifact.exists():
        raise SealError(f"待封存路径不存在: {artifact}")
    work_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    if artifact.is_file():
        files = [artifact]
    else:
        files = [p for p in sorted(artifact.rglob("*")) if p.is_file()]

    if not files:
        raise SealError(f"待封存路径下没有任何文件: {artifact}")

    entries: List[Dict[str, Any]] = []
    identifiers: Set[str] = set()
    source_count = 0

    for path in files:
        is_source = path.suffix.lower() in SOURCE_SUFFIXES
        entry: Dict[str, Any] = {
            "path": str(path.resolve()),
            "relative": str(path.relative_to(artifact)) if artifact.is_dir() else path.name,
            "bytes": path.stat().st_size,
            "is_source": is_source,
        }
        try:
            entry["sha256"] = _sha256(path)
        except OSError as exc:
            entry["sha256"] = None
            entry["error"] = str(exc)

        if is_source:
            source_count += 1
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")[:MAX_SCAN_BYTES]
                identifiers |= extract_identifiers(text)
            except OSError:
                pass
        entries.append(entry)

    manifest = {
        "schema_version": "1.0",
        "op_name": op_name,
        "artifact": str(artifact.resolve()),
        "file_count": len(entries),
        "source_file_count": source_count,
        "identifier_count": len(identifiers),
        "files": entries,
    }

    (work_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 禁读清单：源码文件的绝对路径，外加产物根目录本身
    sealed_paths = [str(artifact.resolve())]
    sealed_paths += [e["path"] for e in entries if e["is_source"]]
    (work_dir / SEALED_PATHS_FILENAME).write_text(
        "# ForgeKernel 禁读清单（自动生成，勿手改）\n"
        "# PreToolUse hook 依据本清单拦截对参考实现源码的读取。\n"
        "# 读了参考源码，就不是从硬件行为反推架构，而是抄。\n"
        + "\n".join(sealed_paths) + "\n",
        encoding="utf-8",
    )

    # 专有名词清单：追加而非覆盖，多个参考的名词要并存
    denylist_path = work_dir / DENYLIST_FILENAME
    existing: Set[str] = set()
    if denylist_path.exists():
        for line in denylist_path.read_text(encoding="utf-8").splitlines():
            token = line.split("#", 1)[0].strip()
            if token:
                existing.add(token)
    merged = sorted(existing | identifiers)
    denylist_path.write_text(
        "# ForgeKernel 参考实现专有名词（自动生成 + 可手工追加）\n"
        "# 知识库 lint 依据本清单阻止这些名词进入可复用经验卡片。\n"
        "# 经验要能被只共享硬件的其他 campaign 读懂，就不能留参考的痕迹。\n"
        + "\n".join(merged) + "\n",
        encoding="utf-8",
    )

    return manifest


def cmd_seal(args: argparse.Namespace) -> int:
    try:
        manifest = seal(Path(args.artifact), args.op_name, Path(args.work_dir))
    except SealError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    work_dir = Path(args.work_dir)
    print(f"已封存 {manifest['op_name']}")
    print(f"  产物      {manifest['artifact']}")
    print(f"  文件数    {manifest['file_count']}（其中源码 {manifest['source_file_count']}）")
    print(f"  专有名词  {manifest['identifier_count']} 个")
    print(f"  禁读清单  {work_dir / SEALED_PATHS_FILENAME}")
    print(f"  名词清单  {work_dir / DENYLIST_FILENAME}")
    if manifest["source_file_count"] == 0:
        print()
        print("提示: 未发现源码文件。若参考本就是二进制（如 CANN 内置实现），")
        print("      这是正常的 —— 它天然满足封闭要求，无需额外防护。")
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    path = Path(args.work_dir) / MANIFEST_FILENAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"错误: 无法读取 {path}: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"错误: {path} 不是合法 JSON: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    print(f"算子 {manifest.get('op_name')}")
    print(f"产物 {manifest.get('artifact')}")
    print(f"文件 {manifest.get('file_count')} 个，源码 {manifest.get('source_file_count')} 个\n")

    drifted = []
    for entry in manifest.get("files", []):
        path_obj = Path(entry["path"])
        if not path_obj.exists():
            drifted.append((entry["relative"], "已删除"))
            continue
        if entry.get("sha256") and _sha256(path_obj) != entry["sha256"]:
            drifted.append((entry["relative"], "内容已变更"))

    if drifted:
        print("封存内容发生变化：", file=sys.stderr)
        for name, reason in drifted:
            print(f"  {name}: {reason}", file=sys.stderr)
        print("\n参考变了，之前所有相对它的加速比都不再可比。", file=sys.stderr)
        return 1
    print("校验通过：封存内容与记录一致。")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """查某个路径是否在禁读清单内。供 hook 和人工排查共用同一套判定。"""
    sealed_file = Path(args.work_dir) / SEALED_PATHS_FILENAME
    if not sealed_file.exists():
        print(f"未找到禁读清单 {sealed_file}；尚未封存任何参考", file=sys.stderr)
        return 0

    sealed: List[str] = []
    for line in sealed_file.read_text(encoding="utf-8").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            sealed.append(token)

    target = str(Path(args.path).resolve())
    for entry in sealed:
        if target == entry or target.startswith(entry.rstrip("/") + "/"):
            print(f"越界：{args.path} 位于封存的参考实现内（匹配 {entry}）",
                  file=sys.stderr)
            print("参考只能测，不能读。读了它的源码，架构结论就不再是从硬件"
                  "行为反推出来的，也就无法迁移到没有开源实现的算子上。",
                  file=sys.stderr)
            return 1
    print(f"未越界：{args.path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="封存参考实现，产出禁读清单与专有名词清单",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    sub = parser.add_subparsers(dest="command", required=True)

    p_seal = sub.add_parser("seal", help="封存一份参考产物")
    p_seal.add_argument("--artifact", required=True, help="参考实现路径")
    p_seal.add_argument("--op-name", required=True)
    p_seal.add_argument("--work-dir", required=True)
    p_seal.set_defaults(func=cmd_seal)

    p_man = sub.add_parser("manifest", help="查看封存清单并校验是否被改动")
    p_man.add_argument("--work-dir", required=True)
    p_man.set_defaults(func=cmd_manifest)

    p_check = sub.add_parser("check", help="查某路径是否越过封存边界")
    p_check.add_argument("--work-dir", required=True)
    p_check.add_argument("--path", required=True)
    p_check.set_defaults(func=cmd_check)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
