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

封什么、不封什么
----------------
禁读清单收两类文件：**源码**和**配置**。配置（算子的 .json/.yaml 等）必须一起
封 —— 它把参考实现了哪些分支、每个分支用什么 tiling key、支持哪些输入形态
组合全列了出来，那正是本该从硬件行为反推的结论，而且照抄它连代码都不用读。

专有名词清单只从源码里抽，而且抽完要减掉**公开接口名**：抽名词的正则分不清
`DataCopyPad`（谁都该用的公开接口）和 `KernelXxxSpecialCase`（参考的内部命名），
而前者一旦进了禁用名单，围栏会拦住所有正常候选。判据不靠手写排除表，靠扫
机器上的 `ascendc/include` —— 那里就是「哪些名字是公开接口」的权威答案。

封存范围要按**单个算子**挑，不要封整棵安装树。整棵树封进来有两个后果：别的
算子里的 `class Kernel` 之类会被当成本参考的专有名词（围栏全是误伤），而且
禁读清单长到几千条后，PreToolUse hook 的比对开销压在每一次工具调用上。
做法是先挑出本算子相关的文件软链到一个中转目录，再封这个中转目录（封存会对
软链做 resolve，清单里落的仍是真实路径，照样拦得住）。

用法:
    python3 seal_reference.py seal --artifact /path/to/ref_impl \\
        --op-name add_layer_norm --work-dir ./forge_work
    python3 seal_reference.py seal --artifact /path/to/ref_impl \\
        --op-name add_layer_norm --work-dir ./forge_work \\
        --public-headers /usr/local/Ascend/cann-8.5.0/aarch64-linux/ascendc/include
    python3 seal_reference.py manifest --work-dir ./forge_work
    python3 seal_reference.py check --work-dir ./forge_work --path /some/file

退出码: 0 = 正常 / 未越界; 1 = 越界或校验失败; 2 = 用法或 IO 错误
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

MANIFEST_FILENAME = "sealed_manifest.json"
SEALED_PATHS_FILENAME = "sealed_paths.txt"
DENYLIST_FILENAME = "denylist.txt"

#: 算子名清单的文件名。和 hwkb.py 的同名常量必须一致 —— 一处是写方，
#: 一处是读方，拼错不会报错，只会让 L-NARRATIVE 的算子名那半条静默失效。
#: tests/test_hwkb.py 有一条用例盯住两个常量相等。
OPERATOR_NAMES_FILENAME = "operator_names.txt"

#: 会被当作源码的扩展名：既禁读，也要从里面抽专有名词。
SOURCE_SUFFIXES = {
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx",
    ".cu", ".cuh", ".py", ".s", ".asm", ".inc", ".tiling",
}

#: 配置类扩展名：同样禁读，但**不**从里面抽名词。
#:
#: 为什么必须封：算子配置表（如 <op>.json）把参考实现了哪些分支、每个分支
#: 用什么 tiling key、支持哪些输入形态组合全列了出来。那正是本该从硬件行为
#: 反推的结论，而且它比源码更现成 —— 照着抄一遍不需要读懂任何代码。
#: 早先只封源码后缀，这类文件是完全敞开的。
#:
#: 为什么不抽名词：配置里的键名大多是框架通用字段，抽进禁用名单只会误伤。
CONFIG_SUFFIXES = {
    ".json", ".yaml", ".yml", ".ini", ".cfg", ".toml", ".cmake",
}

#: 从源码里抽标识符用的模式。只取足够有辨识度的：类名、函数名、宏名。
RE_IDENTIFIER = re.compile(
    r"\b(?:class|struct)\s+([A-Z][A-Za-z0-9_]{4,})"
    r"|#define\s+([A-Z][A-Z0-9_]{5,})"
    r"|\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+){2,})\b"
)

#: 只有一个英文单词的名字（Kernel / Buffer / Tiling 这种）不算专有名词。
#:
#: `class Kernel {` 这类写法在任何代码里都有，把 Kernel 收进禁用名单意味着
#: 任何正常候选都会被拦。专有名词的前提是「这个名字只属于那一份参考」，
#: 单个普通英文单词不满足这个前提。确实需要盯某个单词时，手工往
#: denylist.txt 追加即可 —— 封存是合并而非覆盖。
RE_SINGLE_PLAIN_WORD = re.compile(r"^[A-Z][a-z]+$")

#: 太通用的词即使形态符合也不该进禁用名单，否则会把正常文字全误伤。
COMMON_WORDS = {
    "DataCopy", "LocalTensor", "GlobalTensor", "TPipe", "TQue", "TBuf",
    "AscendC", "PlatformAscendC", "TilingData", "GetBlockIdx", "SetGlobalBuffer",
    "DeQueue", "EnQueue", "AllocTensor", "FreeTensor", "InitBuffer",
    # 每份 AscendC 教程都这么写的宏，不属于任何一份参考
    "BUFFER_NUM", "TILE_LENGTH", "TOTAL_LENGTH",
}

#: 公开 API 白名单缓存文件名。放在工作目录里，便于人工过目与复用。
PUBLIC_API_FILENAME = "public_api_allowlist.txt"

#: 去哪里找 AscendC 公开头文件。按顺序试，全都找不到就降级（见下）。
#: 这里只收 kernel 侧编程接口（ascendc/include），不收 host 侧 aclnn 头文件 ——
#: 调用 aclnn 本身就是「转发到厂商实现」，那由 fence_check 的 F-ACLNN 规则管，
#: 不该靠白名单放行。
ASCEND_ROOT_PATTERNS = (
    "/usr/local/Ascend/ascend-toolkit/latest",
    "/usr/local/Ascend/cann-*",
    "/usr/local/Ascend/latest",
)
PUBLIC_HEADER_SUBGLOBS = ("*/ascendc/include", "ascendc/include")
PUBLIC_HEADER_SUFFIXES = {".h", ".hpp", ".hxx", ".inc"}

#: 公开头文件里认什么算一个「名字」。比抽专有名词的模式宽：白名单宁可多收，
#: 少收一个就会多一次误伤。
RE_ANY_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")

#: 注释里提到的名字不算声明。去掉注释再抽，避免头文件的说明文字污染白名单。
RE_C_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)

#: 单个文件读取上限，避免把超大文件整个读进内存。
MAX_SCAN_BYTES = 2 * 1024 * 1024

#: 注释。比对特征行之前要去掉 —— 注释是说明，不是实现，而且两份独立实现
#: 完全可以有同样的说明（都抄了同一份公开文档的措辞）。
RE_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
RE_LINE_COMMENT = re.compile(r"//[^\n]*")

#: 扫候选时跳过的目录。构建产物里当然会有和参考相同的东西（它们由同一套
#: 头文件生成），拿它们报搬运是纯误报。
#:
#: 注意 third_party **不在**这里 —— 理由同 fence_check：把一份搬运放进
#: 被跳过的目录，是围栏与本检查共同漏过的那条路。
SKIP_DIRS = {
    "build", "build_out", "output", "__pycache__", ".git",
    "node_modules", ".pytest_cache", "dist",
}


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

    这一步只做「形态上够不够特殊」的判断。「是不是公开 API」由调用方拿
    公开头文件白名单再减一遍，见 load_public_api_allowlist。
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
            if RE_SINGLE_PLAIN_WORD.match(group):
                continue
            found.add(group)
    return found


# ---------------------------------------------------------------------------
# 公开 API 白名单
#
# 要解决的问题：抽名词的正则会从参考源码里抽出 DataCopyPad、WholeReduceSum
# 这类**公开**接口名。它们一进禁用名单，任何正常写的候选都会被围栏拦住 ——
# 候选本来就该用公开接口。手写排除表治不了这个：公开接口有上千个，手写
# 十几个必然漏。
#
# 可靠的判据不在我们手里，在机器上：`ascendc/include` 下的头文件就是
# 「哪些名字是公开接口」的权威答案。扫一遍，凡是在那里出现过的名字都不算
# 参考的专有名词。
# ---------------------------------------------------------------------------

def discover_public_header_dirs(extra: Optional[Iterable[str]] = None) -> List[Path]:
    """找出本机的 AscendC 公开头文件目录。"""
    dirs: List[Path] = []
    seen: Set[str] = set()

    def _add(path: Path) -> None:
        if not path.is_dir():
            return
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            dirs.append(path)

    for raw in (extra or ()):
        candidate = Path(raw).expanduser()
        if not candidate.is_dir():
            raise SealError(f"--public-headers 指向的目录不存在: {candidate}")
        _add(candidate)

    roots: List[Path] = []
    env_root = os.environ.get("ASCEND_HOME_PATH")
    if env_root:
        roots.append(Path(env_root))
    for pattern in ASCEND_ROOT_PATTERNS:
        base = Path(pattern)
        if "*" in base.name:
            if base.parent.is_dir():
                roots.extend(sorted(base.parent.glob(base.name)))
        else:
            roots.append(base)

    for root in roots:
        if not root.is_dir():
            continue
        for subglob in PUBLIC_HEADER_SUBGLOBS:
            for hit in sorted(root.glob(subglob)):
                _add(hit)
    return dirs


def collect_public_api_tokens(dirs: Iterable[Path]) -> Set[str]:
    """把公开头文件里出现过的名字全收进来。"""
    tokens: Set[str] = set()
    for directory in dirs:
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in PUBLIC_HEADER_SUFFIXES or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")[:MAX_SCAN_BYTES]
            except OSError:
                continue
            tokens.update(RE_ANY_IDENTIFIER.findall(RE_C_COMMENT.sub(" ", text)))
    return tokens


def _read_token_file(path: Path) -> Set[str]:
    out: Set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            out.add(token)
    return out


def load_public_api_allowlist(work_dir: Path,
                              extra_dirs: Optional[Iterable[str]] = None,
                              refresh: bool = False) -> Dict[str, Any]:
    """拿到公开 API 白名单。

    扫一遍头文件要几秒，而同一个工作目录会反复封存，所以结果缓存到
    work_dir/public_api_allowlist.txt。缓存也让这份名单可以被人工过目和
    手工补充 —— 它决定了围栏会不会误伤，不该是个看不见的内部状态。
    """
    cache = work_dir / PUBLIC_API_FILENAME
    if cache.is_file() and not refresh and not extra_dirs:
        return {
            "enabled": True, "source": "cache", "dirs": [str(cache)],
            "tokens": _read_token_file(cache),
        }

    dirs = discover_public_header_dirs(extra_dirs)
    if not dirs:
        return {"enabled": False, "source": "not_found", "dirs": [], "tokens": set()}

    tokens = collect_public_api_tokens(dirs)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        "# AscendC 公开接口名（自动扫描 + 可手工追加）\n"
        "# 用途：凡在此出现的名字都不算参考实现的专有名词，封存时不进 denylist。\n"
        "# 理由：候选本来就该用公开接口，把公开接口名当专有名词会让围栏\n"
        "#       拦住所有正常实现，最后结果是围栏被关掉。\n"
        + "".join(f"# 扫描来源: {d}\n" for d in dirs)
        + "\n".join(sorted(tokens)) + "\n",
        encoding="utf-8",
    )
    return {
        "enabled": True, "source": "scan",
        "dirs": [str(d) for d in dirs], "tokens": tokens,
    }


#: 知识库根目录的默认位置（相对本脚本）。封存时把名词清单一并写到那里。
#:
#: 为什么要写两份：`fence_check` 读工作目录里那份（候选围栏），而知识库
#: lint 读 `<kb_root>/denylist.txt`。早先只写工作目录那份，于是 hwkb 侧
#: `load_denylist` 永远返回空表、`L-DENY` 一条都不报 —— 一张整段抄了参考
#: 内部符号名的卡片可以入库，而 lint 打印「通过」。
#: 两个消费方、两个路径，接线必须由封存这一步做掉，不能指望有人记得拷。
DEFAULT_KB_ROOT = (Path(__file__).resolve().parent.parent.parent
                   / "forge-hwkb" / "references")


def seal(artifact: Path, op_name: str, work_dir: Path,
         public_header_dirs: Optional[Iterable[str]] = None,
         use_public_api_filter: bool = True,
         refresh_public_api: bool = False,
         kb_root: Optional[Path] = None) -> Dict[str, Any]:
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

    allow: Dict[str, Any] = {
        "enabled": False, "source": "disabled", "dirs": [], "tokens": set()}
    if use_public_api_filter:
        allow = load_public_api_allowlist(
            work_dir, public_header_dirs, refresh_public_api)
    allowlist: Set[str] = allow.pop("tokens")

    entries: List[Dict[str, Any]] = []
    candidates: Set[str] = set()
    source_count = 0
    config_count = 0

    for path in files:
        suffix = path.suffix.lower()
        is_source = suffix in SOURCE_SUFFIXES
        is_config = suffix in CONFIG_SUFFIXES
        entry: Dict[str, Any] = {
            "path": str(path.resolve()),
            "relative": str(path.relative_to(artifact)) if artifact.is_dir() else path.name,
            "bytes": path.stat().st_size,
            "is_source": is_source,
            "is_config": is_config,
            # 禁读与否只看这一个字段。源码和配置都禁读，理由不同：源码泄露
            # 实现手法，配置泄露「实现了哪些分支、用什么 tiling key」。
            "sealed": is_source or is_config,
        }
        try:
            entry["sha256"] = _sha256(path)
        except OSError as exc:
            entry["sha256"] = None
            entry["error"] = str(exc)

        if is_config:
            config_count += 1
        if is_source:
            source_count += 1
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")[:MAX_SCAN_BYTES]
                candidates |= extract_identifiers(text)
            except OSError:
                pass
        entries.append(entry)

    identifiers = sorted(t for t in candidates if t not in allowlist)
    dropped = sorted(t for t in candidates if t in allowlist)

    denylist_path = work_dir / DENYLIST_FILENAME
    existing: Set[str] = set()
    if denylist_path.exists():
        existing = _read_token_file(denylist_path)
    # 公开接口名要从**已有**条目里也剔掉一遍。不这么做的话，上一次封存
    # （比如没带 --public-headers 那次）写进去的 DataCopyPad 会永远留在清单里，
    # 「重新封存一次」这个补救动作对它完全无效 —— 人只能去手工删文件。
    # 公开接口名不可能是某一份参考的专有标记，无论它是怎么进来的。
    stale = sorted(t for t in existing if t in allowlist)
    merged = sorted((existing - allowlist) | set(identifiers))

    manifest = {
        "schema_version": "1.1",
        "op_name": op_name,
        "artifact": str(artifact.resolve()),
        "file_count": len(entries),
        "source_file_count": source_count,
        "config_file_count": config_count,
        "sealed_file_count": sum(1 for e in entries if e["sealed"]),
        "identifier_count": len(identifiers),
        "public_api_filter": {
            "enabled": allow["enabled"],
            "source": allow["source"],
            "dirs": allow["dirs"],
            "allowlist_size": len(allowlist),
            "dropped_tokens": dropped,
            "removed_stale_tokens": stale,
        },
        "files": entries,
    }

    (work_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 禁读清单：源码与配置文件的绝对路径，外加产物根目录本身
    sealed_paths = [str(artifact.resolve())]
    sealed_paths += [e["path"] for e in entries if e["sealed"]]

    # 再加上每个被封文件的**真实父目录**。
    #
    # 为什么需要：清单里原来只有「产物根目录」和「逐个文件」两种条目，而
    # hook 对 Grep / Glob 这类递归工具的判据是「请求路径是不是某个被封条目
    # 的祖先」。产物根目录是软链、或者源码散在产物之外（`opp_install` 那种
    # 来源很常见）时，真实父目录就既不等于产物根、也不在逐文件清单里 ——
    # 于是 `grep -rn class /real/parent/` 从两边都漏掉。
    #
    # 为什么不止加「直接父目录」而是一路加到产物根：漏掉中间任何一层，
    # 对那一层的递归请求就会放行，而它底下照样有被封的文件。
    parents: Set[str] = set()
    artifact_real = artifact.resolve()
    for entry in entries:
        if not entry["sealed"]:
            continue
        current = Path(entry["path"]).resolve().parent
        while True:
            text = str(current)
            if text in parents:
                break   # 这一支已经加过，上面的层级也必然加过
            parents.add(text)
            if current == current.parent:
                break   # 到根了
            # 产物目录自己已经在清单里；再往上就是产物之外的公共目录
            # （比如 /usr/local/Ascend），封它们会把无关路径一起拦住。
            if current == artifact_real:
                break
            current = current.parent
    sealed_paths += sorted(parents - set(sealed_paths))

    (work_dir / SEALED_PATHS_FILENAME).write_text(
        "# ForgeKernel 禁读清单（自动生成，勿手改）\n"
        "# PreToolUse hook 依据本清单拦截对参考实现源码与配置的读取。\n"
        "# 读了参考源码，就不是从硬件行为反推架构，而是抄。\n"
        "#\n"
        "# 三类条目：产物根目录、逐个被封文件、每个被封文件的真实父目录。\n"
        "# 第三类是给 Grep / Glob 这类递归工具用的 —— 它们的判据是\n"
        "# 「请求路径是不是某个被封条目的祖先」，而产物根是软链时，\n"
        "# 真实父目录既不等于产物根、也不在逐文件清单里。\n"
        + "\n".join(sealed_paths) + "\n",
        encoding="utf-8",
    )

    denylist_body = (
        "# ForgeKernel 参考实现专有名词（自动生成 + 可手工追加）\n"
        "# 知识库 lint 依据本清单阻止这些名词进入可复用经验卡片。\n"
        "# 经验要能被只共享硬件的其他 campaign 读懂，就不能留参考的痕迹。\n"
        + "\n".join(merged) + "\n")
    denylist_path.write_text(denylist_body, encoding="utf-8")

    # 同一份清单写进知识库根。两个消费方各读一个路径，接线在这里做掉。
    kb_target = Path(kb_root) if kb_root is not None else DEFAULT_KB_ROOT
    kb_written: Optional[str] = None
    kb_error: Optional[str] = None
    kb_operators: Optional[str] = None
    try:
        kb_target.mkdir(parents=True, exist_ok=True)
        kb_denylist = kb_target / DENYLIST_FILENAME
        # 知识库那份也是合并写入：多个 campaign、多个参考的名词要并存，
        # 而公开接口名同样要从旧条目里剔掉（理由见上面那段）。
        kb_existing = (_read_token_file(kb_denylist)
                       if kb_denylist.exists() else set())
        kb_merged = sorted((kb_existing - allowlist) | set(identifiers))
        kb_denylist.write_text(
            "# ForgeKernel 参考实现专有名词（由 seal_reference.py 写入）\n"
            "# hwkb.py 的 L-DENY 检查读本文件。为空或缺失时 lint 会报错而不是放行。\n"
            + "\n".join(kb_merged) + "\n",
            encoding="utf-8")
        kb_written = str(kb_denylist)

        # 算子名清单，和 denylist 同一次写入。
        #
        # 为什么不放在 denylist 里：两者要的结论不同。denylist 装参考的
        # **私有符号名**，命中报 L-DENY（泄漏了不该读的东西）；算子名是
        # **公开的**，命中报 L-NARRATIVE（把一条硬件机制绑死在一个算子上，
        # 而跨算子复用正是知识库存在的理由）。混在一起，前一类的措辞会把
        # 后一类说成泄漏，人会据此去改豁免而不是去改写卡片。
        #
        # 为什么和 denylist 同一次写入而不另开一处：L-DENY 失效过一次，
        # 原因就是 seal 写一处、hwkb 读另一处，中间没人搬运。多一个独立的
        # 写入点就是多一条可能断掉的线。
        kb_operator_file = kb_target / OPERATOR_NAMES_FILENAME
        op_existing = (_read_token_file(kb_operator_file)
                       if kb_operator_file.exists() else set())
        op_merged = sorted(op_existing | {op_name})
        kb_operator_file.write_text(
            "# 已封存过的算子名（由 seal_reference.py 写入）\n"
            "# hwkb.py 的 L-NARRATIVE 用它阻止卡片把硬件机制绑死在某个算子上。\n"
            + "\n".join(op_merged) + "\n",
            encoding="utf-8")
        kb_operators = str(kb_operator_file)
    except OSError as exc:
        kb_error = str(exc)

    manifest["denylist"] = {
        "work_dir_copy": str(denylist_path),
        "kb_copy": kb_written,
        "kb_operator_names": kb_operators,
        "kb_error": kb_error,
        "token_count": len(merged),
    }
    (work_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    return manifest


def cmd_seal(args: argparse.Namespace) -> int:
    try:
        manifest = seal(
            Path(args.artifact), args.op_name, Path(args.work_dir),
            public_header_dirs=args.public_headers or None,
            use_public_api_filter=not args.no_public_api_filter,
            refresh_public_api=args.refresh_public_api,
            kb_root=Path(args.kb_root) if args.kb_root else None,
        )
    except SealError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    work_dir = Path(args.work_dir)
    api = manifest["public_api_filter"]
    print(f"已封存 {manifest['op_name']}")
    print(f"  产物      {manifest['artifact']}")
    print(f"  文件数    {manifest['file_count']}"
          f"（源码 {manifest['source_file_count']}，配置 {manifest['config_file_count']}）")
    print(f"  禁读路径  {manifest['sealed_file_count']} 个文件 + 产物根目录")
    print(f"  专有名词  {manifest['identifier_count']} 个"
          + (f"（已排除 {len(api['dropped_tokens'])} 个公开接口名）"
             if api["dropped_tokens"] else ""))
    if api.get("removed_stale_tokens"):
        print(f"  已清理    旧清单里的 {len(api['removed_stale_tokens'])} 个"
              "公开接口名（上次封存留下的误伤源）")
    print(f"  禁读清单  {work_dir / SEALED_PATHS_FILENAME}")
    print(f"  名词清单  {work_dir / DENYLIST_FILENAME}")
    deny = manifest.get("denylist") or {}
    if deny.get("kb_copy"):
        print(f"  已同步至  {deny['kb_copy']}（hwkb 的 L-DENY 读这一份）")
    if deny.get("kb_operator_names"):
        print(f"  算子名表  {deny['kb_operator_names']}"
              "（hwkb 的 L-NARRATIVE 读这一份）")
    if deny.get("kb_error"):
        print(f"\n警告: 清单没能写进知识库根：{deny['kb_error']}",
              file=sys.stderr)
        print("      hwkb 的 L-DENY 读 <kb_root>/denylist.txt、"
              "L-NARRATIVE 读 <kb_root>/operator_names.txt，",
              file=sys.stderr)
        print("      写不进去这两道闸就是空的。", file=sys.stderr)
    if manifest["source_file_count"] == 0:
        print()
        print("提示: 未发现源码文件。若参考本就是二进制（如 CANN 内置实现），")
        print("      这是正常的 —— 它天然满足封闭要求，无需额外防护。")
    if not api["enabled"] and manifest["source_file_count"] > 0:
        print()
        print("警告: 没找到 AscendC 公开头文件，专有名词未按公开接口过滤。",
              file=sys.stderr)
        print("      后果是 DataCopyPad 这类公开接口名可能被当成参考的专有符号，",
              file=sys.stderr)
        print("      围栏会拦住正常候选。请用 --public-headers 指到",
              file=sys.stderr)
        print("      <CANN 安装路径>/<arch>-linux/ascendc/include 后重新封存。",
              file=sys.stderr)
    if manifest["file_count"] > 500:
        print()
        print(f"警告: 本次封存了 {manifest['file_count']} 个文件。",
              file=sys.stderr)
        print("      封存范围应当只含**本算子**的产物。整棵安装树封进来会有两个后果：",
              file=sys.stderr)
        print("      一是别的算子里的 class/宏 被当成本参考的专有名词，围栏全是误伤；",
              file=sys.stderr)
        print("      二是禁读清单过长，PreToolUse hook 每次比对的开销压在每个工具调用上。",
              file=sys.stderr)
        print("      做法：先挑出本算子相关的文件，软链到一个中转目录，再封中转目录。",
              file=sys.stderr)
        print("      （封存会对软链做 resolve，清单里落的仍是真实路径，照样拦得住。）",
              file=sys.stderr)

    _warn_if_inside_work_tree(Path(args.artifact), work_dir)
    return 0


def _warn_if_inside_work_tree(artifact: Path, work_dir: Path) -> None:
    """中转目录放在工作区里会让针对自己代码的搜索被连带拦住。

    边界 hook 对 Grep/Glob 要判两个方向：目标在封存目录**内**，以及目标是
    封存目录的**上层** —— 后者同样能递归读到封存内容，所以必须拦。

    代价是：封存目录一旦落在工作目录下，`Grep path=.` 这种最常用的调用就
    成了「上层目录」，会被一并拦住。那是个很烦人的误拦，而误拦的真实结局
    是有人把 hook 关掉 —— 关掉之后真正的越界就无人把守了。

    所以位置本身是个要当场说清的决定，不能等到第一次被拦时才让人去猜。
    """
    try:
        artifact_real = artifact.resolve()
        cwd = Path.cwd().resolve()
        work_real = work_dir.resolve()
    except OSError:
        return

    inside_cwd = artifact_real == cwd or cwd in artifact_real.parents
    inside_work = artifact_real == work_real or work_real in artifact_real.parents
    if not (inside_cwd or inside_work):
        return

    # 先把 stdout 刷出去，否则警告会插在摘要前面，回贴时顺序混乱
    sys.stdout.flush()
    where = "工作目录" if inside_work else "当前目录"
    print(file=sys.stderr)
    print(f"警告: 封存目标在{where}之内（{artifact_real}）。", file=sys.stderr)
    print("      边界 hook 必须拦住「以封存目录的上层为起点的递归搜索」——",
          file=sys.stderr)
    print("      那样同样能读到参考源码。于是封存目录落在工作区里时，",
          file=sys.stderr)
    print("      针对你自己代码的 `Grep path=.` 会被一并拦住。", file=sys.stderr)
    print("      建议把中转目录放到工作区之外，例如：", file=sys.stderr)
    print(f"        {_suggest_stage_dir()}", file=sys.stderr)
    print("      然后重新封存（禁读清单落的是软链指向的真实路径，不受位置影响）。",
          file=sys.stderr)


def _suggest_stage_dir() -> str:
    """给一个本机确实可写、且在工作区之外的中转目录位置。"""
    for base in ("/cache", os.environ.get("XDG_CACHE_HOME"),
                 str(Path.home() / ".cache")):
        if base and Path(base).is_dir() and os.access(base, os.W_OK):
            return str(Path(base) / "forge-kernel" / "ref_stage" / "<算子名>")
    return "/tmp/forge-kernel/ref_stage/<算子名>"


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
    print(f"文件 {manifest.get('file_count')} 个，"
          f"源码 {manifest.get('source_file_count')} 个，"
          f"配置 {manifest.get('config_file_count', 0)} 个\n")

    drifted = verify_manifest(manifest)
    if drifted:
        print("封存内容发生变化：", file=sys.stderr)
        for item in drifted:
            print(f"  {item['relative']}: {item['reason']}", file=sys.stderr)
        print("\n参考变了，之前所有相对它的加速比都不再可比。", file=sys.stderr)
        return 1
    print("校验通过：封存内容与记录一致。")
    return 0


def verify_manifest(manifest: Dict[str, Any]) -> List[Dict[str, str]]:
    """封存内容还和记录一致吗。返回漂移项，空表示一致。

    三个方向，第三个是后补的：

      已删除     参考少了文件
      内容变更   参考的某个文件被改过
      **新增**   参考多了文件，而它**不在 manifest 里**

    第三个方向要紧，因为它不是「记录对不上」这么轻 —— 新文件没进 manifest，
    就没进禁读清单，于是 hook **不会拦**对它的读取。也就是说参考树上多出
    一个 agent 可以随便读的源文件，而所有校验（比对已记录的那些）全绿。

    这件事在 `opp_install` 那类来源上很容易发生：那是一个活的安装目录，
    另一个任务重装一次算子包就会往里加文件。
    """
    drifted: List[Dict[str, str]] = []
    recorded: Set[str] = set()
    for entry in manifest.get("files", []):
        if not isinstance(entry, dict) or "path" not in entry:
            continue
        recorded.add(str(Path(entry["path"]).resolve()))
        path_obj = Path(entry["path"])
        if not path_obj.exists():
            drifted.append({"relative": entry.get("relative", entry["path"]),
                            "reason": "已删除"})
            continue
        if entry.get("sha256") and _sha256(path_obj) != entry["sha256"]:
            drifted.append({"relative": entry.get("relative", entry["path"]),
                            "reason": "内容已变更"})

    raw_artifact = manifest.get("artifact")
    if raw_artifact:
        artifact = Path(raw_artifact)
        if artifact.is_dir():
            for path in sorted(artifact.rglob("*")):
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix not in SOURCE_SUFFIXES and suffix not in CONFIG_SUFFIXES:
                    continue
                if str(path.resolve()) in recorded:
                    continue
                drifted.append({
                    "relative": str(path.relative_to(artifact)),
                    "reason": ("**新增**，且不在封存清单里 —— "
                               "也就是说 hook 不会拦对它的读取。"
                               "重新跑一次 seal"),
                })
    return drifted


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


#: 一行要有多少个「有信息的字符」才算得上一条可用于比对的特征行。
#:
#: 太短的行在任何两份 C++ 里都会撞：`}`、`};`、`return;`、`#include <cstdint>`。
#: 拿它们比对会得到一堆误报，而误报的代价是整条检查被豁免掉。
DISTINCTIVE_MIN_CHARS = 28

#: 同一条特征行在参考里出现过多少次之后就不再算「特征」。
#:
#: 一份参考里重复十几次的行（常见的宏展开、模板样板）说明它是这份代码的
#: 习惯写法，而习惯写法会被任何读过同类文档的人写出来。
DISTINCTIVE_MAX_REPEATS = 3


def distinctive_lines(text: str) -> Set[str]:
    """从一段源码里抽出可用于比对的特征行。

    归一化：去注释、压空白、丢掉太短的行。保留标识符与结构，所以「同一段
    代码重排了缩进」仍然能对上，而「两份代码恰好都有一个 for 循环」对不上。
    """
    out: List[str] = []
    text = RE_BLOCK_COMMENT.sub(" ", text)
    for raw in text.splitlines():
        line = RE_LINE_COMMENT.sub("", raw)
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < DISTINCTIVE_MIN_CHARS:
            continue
        out.append(line)
    counts: Dict[str, int] = {}
    for line in out:
        counts[line] = counts.get(line, 0) + 1
    return {line for line, n in counts.items()
            if n <= DISTINCTIVE_MAX_REPEATS}


def check_tree(candidate: Path, work_dir: Path,
               artifact: Optional[Path] = None) -> Dict[str, Any]:
    """候选里有没有参考源码的明文搬运。

    两级判据，由强到弱：

      整文件相同   候选某个文件的 sha256 等于参考某个文件的 sha256。
                   manifest 里已经逐文件记了 sha256，所以这一级不需要再读
                   参考 —— 比对的是两个摘要。
      特征行命中   候选里出现了参考的特征行。这一级**需要读参考**，所以
                   只有在参考路径仍可访问时才做，而且它跑在封存动作这一侧
                   （不是 agent 那一侧）—— 工具读参考是允许的，agent 不行。

    为什么要有这个检查：封存 hook 拦的是「读」。它拦不住一次成功的读之后
    发生的事 —— 比如封存漏挑了一个文件（后缀不在 SOURCE_SUFFIXES 里）、
    或者在封存之前就已经读过。那之后源码进了候选，而围栏（比对符号名）
    只在名字没改时才抓得到。

    返回的 `findings` 为空表示没发现搬运。**这不等于「没有搬运」** ——
    改过名字、换过结构的搬运这两级都抓不到。结论里写明这一点。
    """
    manifest_path = work_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise SealError(
            f"找不到封存清单 {manifest_path}。check-tree 要拿它里面的逐文件 "
            "sha256 做比对 —— 先跑一次 seal。")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealError(f"封存清单读不出来: {exc}") from exc

    sealed_entries = [e for e in manifest.get("files", [])
                      if isinstance(e, dict) and e.get("sealed")]
    by_hash = {e["sha256"]: e["path"] for e in sealed_entries if e.get("sha256")}

    findings: List[Dict[str, Any]] = []
    scanned = 0

    # 第一级：整文件摘要相同。
    candidate_files = ([candidate] if candidate.is_file()
                       else [p for p in sorted(candidate.rglob("*"))
                             if p.is_file()])
    for path in candidate_files:
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        scanned += 1
        digest = _sha256(path)
        if digest in by_hash:
            findings.append({
                "rule": "T-SAME-FILE",
                "candidate_file": str(path),
                "reference_file": by_hash[digest],
                "detail": ("候选这个文件与参考的某个文件逐字节相同。"
                           "这不是「写得像」，是同一份文件"),
            })

    # 第二级：特征行命中。需要读参考，所以参考不可访问时跳过并说明。
    reference_root = artifact
    if reference_root is None:
        raw = manifest.get("artifact")
        reference_root = Path(raw) if raw else None
    line_check: Dict[str, Any]
    if reference_root is None or not reference_root.exists():
        line_check = {
            "performed": False,
            "reason": (f"参考路径 {reference_root} 已不可访问，特征行这一级"
                       "跳过。整文件摘要那一级不受影响（它只用 manifest）"),
        }
    else:
        reference_lines: Set[str] = set()
        for entry in sealed_entries:
            ref_path = Path(entry["path"])
            if not ref_path.is_file():
                continue
            if ref_path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            try:
                reference_lines |= distinctive_lines(
                    ref_path.read_text(encoding="utf-8", errors="ignore")
                    [:MAX_SCAN_BYTES])
            except OSError:
                continue
        hits_by_file: Dict[str, List[str]] = {}
        for path in candidate_files:
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            hits = sorted(distinctive_lines(text) & reference_lines)
            if hits:
                hits_by_file[str(path)] = hits[:5]
        for file_path, hits in sorted(hits_by_file.items()):
            findings.append({
                "rule": "T-SAME-LINES",
                "candidate_file": file_path,
                "hit_count": len(hits),
                "samples": hits,
                "detail": ("候选里出现了参考的特征行（已去注释、压空白、"
                           "丢掉过短与高重复的行）。\n"
                           "    这些行在两份独立实现里同时出现的可能性很低"),
            })
        line_check = {"performed": True,
                      "reference_line_count": len(reference_lines)}

    return {
        "schema_version": "1.0",
        "candidate": str(candidate),
        "files_scanned": scanned,
        "sealed_file_count": len(sealed_entries),
        "line_check": line_check,
        "findings": findings,
        "passed": not findings,
        # 这句话要进报告，不能只写在文档里。
        "coverage_note": ("两级判据都只抓**明文搬运**：逐字节相同的文件、"
                          "以及未改动的特征行。\n"
                          "改过名字、重排过结构的搬运两级都抓不到 —— "
                          "`findings` 为空不等于「没有搬运」。\n"
                          "这道检查补的是封存 hook 的一个具体缺口："
                          "hook 拦「读」，拦不住一次成功的读之后发生的事"
                          "（封存漏挑了文件、或在封存之前就已经读过）。"),
    }


def cmd_check_tree(args: argparse.Namespace) -> int:
    try:
        report = check_tree(
            Path(args.candidate), Path(args.work_dir),
            Path(args.artifact) if args.artifact else None)
    except SealError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1

    print(f"候选      {report['candidate']}")
    print(f"扫描      {report['files_scanned']} 个源文件 vs "
          f"{report['sealed_file_count']} 个被封文件")
    if not report["line_check"]["performed"]:
        print(f"特征行    跳过：{report['line_check']['reason']}")
    else:
        print(f"特征行    参考侧 {report['line_check']['reference_line_count']} "
              "条可比对的行")
    print()
    if report["passed"]:
        print("未发现明文搬运。")
        print()
        print(f"注意: {report['coverage_note']}")
        return 0
    print(f"发现 {len(report['findings'])} 处疑似明文搬运：\n", file=sys.stderr)
    for item in report["findings"]:
        print(f"[{item['rule']}] {item['candidate_file']}", file=sys.stderr)
        if item.get("reference_file"):
            print(f"    与参考的 {item['reference_file']} 逐字节相同",
                  file=sys.stderr)
        for sample in item.get("samples", []):
            print(f"    · {sample[:120]}", file=sys.stderr)
        print(f"    {item['detail']}\n", file=sys.stderr)
    print("参考只能测，不能读。源码进了候选，架构结论就不再是从硬件行为反推"
          "出来的，", file=sys.stderr)
    print("也就无法迁移到没有开源实现的算子上。", file=sys.stderr)
    return 1


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
    p_seal.add_argument("--public-headers", action="append", default=[],
                        metavar="DIR",
                        help="AscendC 公开头文件目录，可重复。"
                             "在这些头文件里出现过的名字不算参考专有名词。"
                             "不给则自动在 ASCEND_HOME_PATH 等位置查找")
    p_seal.add_argument("--refresh-public-api", action="store_true",
                        help="重新扫描公开头文件，覆盖已缓存的白名单")
    p_seal.add_argument("--kb-root", default=None,
                        help="知识库根目录，名词清单会一并写进去供 hwkb 的 "
                             "L-DENY 使用。默认是本插件的 "
                             "skills/forge-hwkb/references")
    p_seal.add_argument("--no-public-api-filter", action="store_true",
                        help="不按公开接口过滤专有名词。"
                             "会让 DataCopyPad 这类公开接口名进入禁用名单，"
                             "围栏将拦住正常候选，仅排查时使用")
    p_seal.set_defaults(func=cmd_seal)

    p_man = sub.add_parser("manifest", help="查看封存清单并校验是否被改动")
    p_man.add_argument("--work-dir", required=True)
    p_man.set_defaults(func=cmd_manifest)

    p_tree = sub.add_parser(
        "check-tree", help="查候选里有没有参考源码的明文搬运")
    p_tree.add_argument("--candidate", required=True, help="候选目录或文件")
    p_tree.add_argument("--work-dir", required=True,
                        help="封存工作目录（里面有 sealed_manifest.json）")
    p_tree.add_argument("--artifact", default=None,
                        help="参考路径；不给则用 manifest 里记的那个。"
                             "参考不可访问时只做整文件摘要比对，"
                             "并在报告里说明特征行这一级跳过了")
    p_tree.set_defaults(func=cmd_check_tree)

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
