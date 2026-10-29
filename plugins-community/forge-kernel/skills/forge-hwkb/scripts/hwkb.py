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
"""硬件知识库：准入、检索、边界 lint、死路表。

这是 ForgeKernel 里唯一跨 campaign 存活的东西。它之所以能跨算子复用，
靠的是两条规矩，本脚本把这两条变成程序判定：

  准入要测量   一条经验必须有目标架构上的探针实测给它背书才能入库。
               agent 自己说「这样更快」不算证据 —— 论文把 agent 自评
               的探针单列为一类 bug 来源。

  边界要干净   卡片正文不得留下它被总结出来时所针对的那个参考实现的
               痕迹。出处只能引用公开文档、公开 API 符号、探针判定。
               做到这点，一条在 attention 反向 campaign 里总结出的经验，
               才能被一个只共享同款硬件的 GEMM campaign 读懂并复用。

卡片按芯片架构分区存放，因为硬件机制本就是架构相关的 —— DAV_2201 的
set/wait 强配对经验搬到 DAV_3510 的 BufferID 同步上是错的。

死路表（no-go）以症状为 key，记录试过且无效的方向，作为重试前置条件。
它和正向经验一样重要：论文说死路同样在积累。

用法:
    python3 hwkb.py validate --card draft.md
    python3 hwkb.py add --card draft.md
    python3 hwkb.py query --symptom "mte2 占比高" --arch DAV_2201
    python3 hwkb.py lint --all
    python3 hwkb.py nogo add --symptom "..." --tried "..." --evidence "[MEASURED, job 9]"
    python3 hwkb.py nogo check --symptom "..."

退出码: 0 = 通过 / 命中; 1 = 不通过 / 未命中; 2 = 用法或 IO 错误
"""

import argparse
import contextlib
import fcntl
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - 环境问题，给出可操作的提示
    print(
        "错误: 需要 PyYAML 才能解析卡片 frontmatter。\n"
        "  安装: pip install 'PyYAML>=6.0'\n"
        "  本仓 tests/system/scripts/requirements.txt 已声明该依赖。",
        file=sys.stderr,
    )
    sys.exit(2)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SKILL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB_ROOT = SKILL_ROOT / "references"

#: 架构 -> 存放目录。新增架构时在这里加一行，不要散落在各处。
ARCH_DIRS = {
    "DAV_2201": "dav2201",
    "DAV_3510": "dav3510",
}

#: 卡片正文必须有的四个小节（论文 §3.4：机制、触发症状、最小可运行例、已知坑）。
REQUIRED_SECTIONS = ("机制", "触发症状", "最小可运行例", "已知坑")

#: 出处只允许这三类来源。其他一律视为越过 no-code 边界。
ALLOWED_PROVENANCE_KINDS = {"probe_verdict", "public_doc", "public_api"}

NOGO_FILENAME = "nogo.json"
DENYLIST_FILENAME = "denylist.txt"

#: lint 用：出处里出现源码文件路径，说明引用了不该读的东西。
RE_SOURCE_PATH = re.compile(
    r"[\w./-]+\.(?:cpp|cc|cxx|h|hpp|cu|cuh|py|s|asm)\b", re.IGNORECASE
)

RE_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


class Issue:
    """一条问题。code 用于测试断言，message 给人看。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "message": self.message}

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


# ---------------------------------------------------------------------------
# 跨任务互斥
# ---------------------------------------------------------------------------

#: 取锁超时。知识库的写操作都很短（追加一条 JSON、拷一个卡片），
#: 等这么久还拿不到就说明有东西卡住了，该报出来而不是继续等。
LOCK_TIMEOUT = 30.0


@contextlib.contextmanager
def exclusive(kb_root: Path, timeout: float = LOCK_TIMEOUT):
    """对知识库加排他锁，覆盖「读 → 改 → 写回」的全过程。

    多个演进任务并行跑时共用同一个知识库。死路表的写入是

        load_nogo  →  append  →  write

    不加锁时两个任务同时登记死路，后写的会把先写的那条覆盖掉 ——
    **而且两边都报「已登记」**。死路表的全部价值在于跨代存活，
    丢一条就意味着下一代会把那个方向原封不动再试一遍，
    这正是它要消除的浪费。

    卡片入库也在锁内：两个任务同时 add 同一个 id 时，没有锁的话
    「已存在」检查和复制之间有窗口，双方都会判定不存在然后依次写入。
    """
    kb_root.mkdir(parents=True, exist_ok=True)
    lock_path = kb_root / ".lock"
    deadline = time.time() + timeout
    handle = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() >= deadline:
                    raise TimeoutError(
                        f"取知识库锁超时（{timeout:.0f}s）：{lock_path}。"
                        "若确认没有别的任务在跑，删掉这个锁文件即可。")
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(handle)


# ---------------------------------------------------------------------------
# 卡片解析
# ---------------------------------------------------------------------------

def parse_card(text: str) -> Tuple[Dict[str, Any], str]:
    """拆出 frontmatter 与正文。

    解析失败时抛 ValueError，调用方负责转成 Issue —— 这样解析逻辑本身
    不需要知道上层怎么报错。
    """
    match = RE_FRONTMATTER.match(text)
    if not match:
        raise ValueError(
            "卡片缺少 YAML frontmatter；格式应为 '---' 包裹的头部，随后是正文"
        )
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ValueError(f"frontmatter 不是合法 YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError("frontmatter 必须是键值映射")
    return meta, match.group(2)


def load_card(path: Path) -> Tuple[Dict[str, Any], str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"无法读取 {path}: {exc}") from exc
    return parse_card(text)


def _sections(body: str) -> Dict[str, str]:
    """按 markdown 二级标题切出小节，返回 {标题: 内容}。"""
    out: Dict[str, str] = {}
    current: Optional[str] = None
    buf: List[str] = []
    for line in body.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            if current is not None:
                out[current] = "\n".join(buf).strip()
            current = heading.group(1).strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf).strip()
    return out


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return not value
    return False


# ---------------------------------------------------------------------------
# 准入校验
# ---------------------------------------------------------------------------

def check_admission(meta: Dict[str, Any], body: str) -> List[Issue]:
    """准入校验：没有探针实测背书的经验一律拒收。

    论文 §3.4：提案必须经过目标架构上的探针认证才能入库；认证把真正起
    作用的机制和只是伴随发生的分开。
    """
    out: List[Issue] = []

    for field in ("id", "arch", "mechanism", "symptom", "symptom_channel"):
        if _is_blank(meta.get(field)):
            out.append(Issue("A-FIELD", f"frontmatter 缺少必填字段 '{field}'"))

    arch = meta.get("arch")
    if arch and arch not in ARCH_DIRS:
        out.append(Issue(
            "A-ARCH",
            f"未知架构 '{arch}'；已知: {sorted(ARCH_DIRS)}。"
            "硬件机制与架构绑定，不能跨架构套用",
        ))

    # 核心：探针认证
    cert = meta.get("certified_by")
    if not isinstance(cert, dict) or _is_blank(cert):
        out.append(Issue(
            "A-CERT",
            "缺少 certified_by：没有探针实测背书的经验不得入库",
        ))
    else:
        if _is_blank(cert.get("job")):
            out.append(Issue(
                "A-CERT-JOB", "certified_by 必须指明是哪次 job 认证的，否则无法复核",
            ))
        cert_arch = cert.get("arch")
        if _is_blank(cert_arch):
            out.append(Issue(
                "A-CERT-ARCH", "certified_by 必须指明认证所在架构",
            ))
        elif arch and cert_arch != arch:
            out.append(Issue(
                "A-CERT-ARCH",
                f"认证架构 '{cert_arch}' 与卡片架构 '{arch}' 不一致；"
                "必须在目标架构上认证",
            ))
        if _is_blank(cert.get("delta")):
            out.append(Issue(
                "A-CERT-DELTA",
                "certified_by 必须记录实测变化量，用以区分真机制与巧合加速",
            ))

    # 出处
    prov = meta.get("provenance")
    if not isinstance(prov, list) or not prov:
        out.append(Issue("A-PROV", "缺少 provenance：至少给出一条可核验的出处"))
    else:
        for idx, entry in enumerate(prov):
            if not isinstance(entry, dict):
                out.append(Issue(
                    "A-PROV", f"provenance[{idx}] 必须是含 kind 与 ref 的对象",
                ))
                continue
            kind = entry.get("kind")
            if kind not in ALLOWED_PROVENANCE_KINDS:
                out.append(Issue(
                    "A-PROV-KIND",
                    f"provenance[{idx}] 的 kind '{kind}' 不被允许；"
                    f"只接受 {sorted(ALLOWED_PROVENANCE_KINDS)}",
                ))
            if _is_blank(entry.get("ref")):
                out.append(Issue("A-PROV-REF", f"provenance[{idx}] 缺少 ref"))

    # 正文四小节
    found = _sections(body)
    for section in REQUIRED_SECTIONS:
        if section not in found:
            out.append(Issue(
                "A-SECTION",
                f"正文缺少 '## {section}' 小节；四小节缺一，这条经验就不足以被别的 campaign 复用",
            ))
        elif _is_blank(found[section]):
            out.append(Issue("A-SECTION", f"'## {section}' 小节为空"))

    return out


# ---------------------------------------------------------------------------
# no-code 边界 lint
# ---------------------------------------------------------------------------

def load_denylist(kb_root: Path) -> List[str]:
    """读取参考实现专有名词清单。

    这份清单由 campaign driver 维护：开一个 campaign 时把该参考的内部
    符号名登记进来，之后任何卡片带上这些词都会被挡下。
    """
    path = kb_root / DENYLIST_FILENAME
    if not path.exists():
        return []
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            out.append(token)
    return out


def check_no_code_boundary(meta: Dict[str, Any], body: str,
                           denylist: List[str]) -> List[Issue]:
    """边界检查：卡片不得留下参考实现的痕迹。

    论文 §3.4：参考特有的名字留在人类专用归档里，由确定性 lint 保持分离；
    因此一条经验不保留它被总结时所针对的那个参考的痕迹 —— 这正是它能被
    另一个只共享硬件的 campaign 读懂的原因。
    """
    out: List[Issue] = []
    haystack = yaml.safe_dump(meta, allow_unicode=True) + "\n" + body
    lowered = haystack.lower()

    for token in denylist:
        if token.lower() in lowered:
            out.append(Issue(
                "L-DENY",
                f"出现参考实现专有名词 '{token}'；"
                "该名词属人类专用归档，不得进入可复用经验",
            ))

    prov = meta.get("provenance")
    if isinstance(prov, list):
        for idx, entry in enumerate(prov):
            if not isinstance(entry, dict):
                continue
            ref = str(entry.get("ref", ""))
            match = RE_SOURCE_PATH.search(ref)
            if match and entry.get("kind") != "public_doc":
                out.append(Issue(
                    "L-SRC",
                    f"provenance[{idx}] 引用了源码路径 '{match.group(0)}'；"
                    "出处只能是公开文档、公开 API 符号或探针判定",
                ))

    return out


# ---------------------------------------------------------------------------
# 知识库读写
# ---------------------------------------------------------------------------

def arch_dir(kb_root: Path, arch: str) -> Path:
    sub = ARCH_DIRS.get(arch)
    if sub is None:
        raise ValueError(f"未知架构 '{arch}'；已知: {sorted(ARCH_DIRS)}")
    return kb_root / sub


def iter_cards(kb_root: Path, arch: Optional[str] = None):
    """遍历知识库里的卡片，产出 (路径, meta, body)。解析失败的跳过并记录。"""
    arches = [arch] if arch else list(ARCH_DIRS)
    for name in arches:
        directory = kb_root / ARCH_DIRS.get(name, "")
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            if path.name.upper().startswith("INDEX"):
                continue
            try:
                meta, body = load_card(path)
            except ValueError:
                continue
            yield path, meta, body


def cmd_validate(args: argparse.Namespace) -> int:
    """只校验不入库，供 agent 起草时自检。"""
    kb_root = Path(args.kb_root)
    try:
        meta, body = load_card(Path(args.card))
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    issues = check_admission(meta, body)
    issues += check_no_code_boundary(meta, body, load_denylist(kb_root))
    return _report(issues, args.as_json, f"卡片 {args.card} 满足准入与边界要求")


def cmd_add(args: argparse.Namespace) -> int:
    """校验通过才写入知识库。不通过就原样拒绝，不做任何「尽力而为」的降级。"""
    kb_root = Path(args.kb_root)
    card_path = Path(args.card)
    try:
        meta, body = load_card(card_path)
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    issues = check_admission(meta, body)
    issues += check_no_code_boundary(meta, body, load_denylist(kb_root))
    if issues:
        print(f"拒绝入库：{card_path} 有 {len(issues)} 项问题", file=sys.stderr)
        for issue in issues:
            print(f"  {issue}", file=sys.stderr)
        return 1

    try:
        target_dir = arch_dir(kb_root, str(meta["arch"]))
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    try:
        with exclusive(kb_root):
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{meta['id']}.md"
            if target.exists() and not args.force:
                print(f"错误: {target} 已存在；如确需覆盖请加 --force",
                      file=sys.stderr)
                return 2
            shutil.copyfile(card_path, target)
    except TimeoutError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    print(f"已入库: {target}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """按症状/架构/算子族检索。

    检索以症状为主键，因为 arch agent 的使用场景是「我观测到了这个，
    有人遇到过吗」。
    """
    kb_root = Path(args.kb_root)
    needle = (args.symptom or "").strip().lower()
    family = (args.family or "").strip().lower()

    hits = []
    for path, meta, body in iter_cards(kb_root, args.arch):
        if meta.get("status") == "retired" and not args.include_retired:
            continue
        if family:
            families = meta.get("operator_families") or []
            families = [str(f).lower() for f in families] if isinstance(families, list) else []
            # 空的 operator_families 表示算子无关，对任何族都适用
            if families and family not in families:
                continue
        if needle:
            text = " ".join(str(meta.get(k, "")) for k in ("mechanism", "symptom")).lower()
            if needle not in text and needle not in body.lower():
                continue
        hits.append((path, meta))

    if args.as_json:
        print(json.dumps({
            "count": len(hits),
            "cards": [
                {
                    "id": m.get("id"),
                    "arch": m.get("arch"),
                    "mechanism": m.get("mechanism"),
                    "symptom": m.get("symptom"),
                    "path": str(p),
                }
                for p, m in hits
            ],
        }, ensure_ascii=False, indent=2))
    elif not hits:
        print("未命中任何经验")
    else:
        for path, meta in hits:
            print(f"{meta.get('id')}  [{meta.get('arch')}]  {meta.get('mechanism')}")
            print(f"    症状: {meta.get('symptom')}")
            print(f"    路径: {path}")
    return 0 if hits else 1


def cmd_lint(args: argparse.Namespace) -> int:
    """对单张卡片或整库做边界检查。"""
    kb_root = Path(args.kb_root)
    denylist = load_denylist(kb_root)
    all_issues: List[Issue] = []

    if args.card:
        try:
            meta, body = load_card(Path(args.card))
        except ValueError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 2
        all_issues = check_no_code_boundary(meta, body, denylist)
        scope = args.card
    else:
        count = 0
        for path, meta, body in iter_cards(kb_root, args.arch):
            count += 1
            for issue in check_no_code_boundary(meta, body, denylist):
                all_issues.append(Issue(issue.code, f"{path.name}: {issue.message}"))
        scope = f"{count} 张卡片"

    return _report(all_issues, args.as_json, f"{scope} 未越过 no-code 边界")


# ---------------------------------------------------------------------------
# 死路表
# ---------------------------------------------------------------------------

def _nogo_path(kb_root: Path) -> Path:
    return kb_root / NOGO_FILENAME


def load_nogo(kb_root: Path) -> List[Dict[str, Any]]:
    path = _nogo_path(kb_root)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data.get("entries", []) if isinstance(data, dict) else []


def cmd_nogo_add(args: argparse.Namespace) -> int:
    """登记一条死路。死路和正向经验一样要带证据 —— 没测过就不算死路。"""
    kb_root = Path(args.kb_root)
    if not args.evidence.strip():
        print("错误: 死路同样需要证据；未经测量的放弃不构成死路", file=sys.stderr)
        return 2

    try:
        with exclusive(kb_root):
            entries = load_nogo(kb_root)
            entries.append({
                "symptom": args.symptom,
                "tried": args.tried,
                "evidence": args.evidence,
                "arch": args.arch,
                "operator": args.operator,
            })
            _nogo_path(kb_root).write_text(
                json.dumps({"entries": entries}, ensure_ascii=False,
                           indent=2) + "\n",
                encoding="utf-8",
            )
    except TimeoutError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    print(f"已登记死路（当前 {len(entries)} 条）: {args.tried}")
    return 0


def cmd_nogo_check(args: argparse.Namespace) -> int:
    """重试前置条件：这个症状下有没有已经试过且无效的方向。

    退出码 1 表示命中死路 —— 调用方应当据此避开，而不是再试一遍。
    """
    kb_root = Path(args.kb_root)
    needle = args.symptom.strip().lower()
    hits = [
        e for e in load_nogo(kb_root)
        if needle in str(e.get("symptom", "")).lower()
        and (not args.arch or e.get("arch") == args.arch)
    ]

    if args.as_json:
        print(json.dumps({"hit": bool(hits), "entries": hits},
                         ensure_ascii=False, indent=2))
    elif hits:
        print(f"命中 {len(hits)} 条死路，以下方向已试过且无效：")
        for entry in hits:
            print(f"  - {entry.get('tried')}")
            print(f"    证据: {entry.get('evidence')}")
    else:
        print("未命中死路，该方向可以尝试")
    return 1 if hits else 0


# ---------------------------------------------------------------------------
# 输出与入口
# ---------------------------------------------------------------------------

def _report(issues: List[Issue], as_json: bool, ok_message: str) -> int:
    if as_json:
        print(json.dumps({
            "passed": not issues,
            "issue_count": len(issues),
            "issues": [i.as_dict() for i in issues],
        }, ensure_ascii=False, indent=2))
        return 1 if issues else 0
    if not issues:
        print(f"通过：{ok_message}")
        return 0
    print(f"未通过：{len(issues)} 项问题", file=sys.stderr)
    for issue in issues:
        print(f"  {issue}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    """构造命令行。

    `--kb-root` 与 `--json` 同时挂顶层和每个子命令，两种写法都能跑：

        hwkb.py --kb-root ./kb nogo add ...
        hwkb.py nogo add --kb-root ./kb ...

    只挂顶层是 argparse 的默认做法，但它要求全局参数写在子命令**之前**，
    而人和 agent 的直觉都是写在后面。写错时只报
    「unrecognized arguments」，不说该放哪里 —— agent 拿到这个报错往往
    不会去调参数位置，而是判定脚本坏了、转而自己判断。

    子命令那份用 SUPPRESS 作默认值：不传时不写入 namespace，顶层取值
    得以保留；传了才覆盖。用普通默认值的话，`--kb-root X nogo add` 会被
    子命令的默认值悄悄覆盖回去。
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--kb-root", default=argparse.SUPPRESS,
                        help=f"知识库根目录（默认 {DEFAULT_KB_ROOT}）")
    common.add_argument("--json", action="store_true", dest="as_json",
                        default=argparse.SUPPRESS,
                        help="以 JSON 输出，供脚本消费")

    parser = argparse.ArgumentParser(
        description="ForgeKernel 硬件知识库：准入、检索、边界 lint、死路表",
    )
    parser.add_argument("--kb-root", default=str(DEFAULT_KB_ROOT),
                        help="知识库根目录（也可写在子命令后面）")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="以 JSON 输出（也可写在子命令后面）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="校验草稿卡片，不入库", parents=[common])
    p_val.add_argument("--card", required=True)
    p_val.set_defaults(func=cmd_validate)

    p_add = sub.add_parser("add", help="校验通过后写入知识库", parents=[common])
    p_add.add_argument("--card", required=True)
    p_add.add_argument("--force", action="store_true", help="允许覆盖同 id 卡片")
    p_add.set_defaults(func=cmd_add)

    p_q = sub.add_parser("query", help="按症状/架构/算子族检索", parents=[common])
    p_q.add_argument("--symptom", default="")
    p_q.add_argument("--arch", default=None, choices=sorted(ARCH_DIRS))
    p_q.add_argument("--family", default="")
    p_q.add_argument("--include-retired", action="store_true")
    p_q.set_defaults(func=cmd_query)

    p_lint = sub.add_parser("lint", help="no-code 边界检查", parents=[common])
    p_lint.add_argument("--card", default=None, help="不给则检查整库")
    p_lint.add_argument("--all", action="store_true", help="检查整库（等价于不给 --card）")
    p_lint.add_argument("--arch", default=None, choices=sorted(ARCH_DIRS))
    p_lint.set_defaults(func=cmd_lint)

    p_nogo = sub.add_parser("nogo", help="死路表", parents=[common])
    nogo_sub = p_nogo.add_subparsers(dest="nogo_command", required=True)

    p_na = nogo_sub.add_parser("add", help="登记一条已验证无效的方向", parents=[common])
    p_na.add_argument("--symptom", required=True)
    p_na.add_argument("--tried", required=True, help="试过的方向")
    p_na.add_argument("--evidence", required=True, help="证明它无效的实测证据")
    p_na.add_argument("--arch", default=None, choices=sorted(ARCH_DIRS))
    p_na.add_argument("--operator", default="")
    p_na.set_defaults(func=cmd_nogo_add)

    p_nc = nogo_sub.add_parser("check", help="重试前置检查", parents=[common])
    p_nc.add_argument("--symptom", required=True)
    p_nc.add_argument("--arch", default=None, choices=sorted(ARCH_DIRS))
    p_nc.set_defaults(func=cmd_nogo_check)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
