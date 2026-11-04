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

#: 算子名清单。由 `seal_reference.py seal` 与 denylist 一同写入知识库根。
#:
#: 它和 denylist 是两件事：denylist 装的是参考实现的**内部符号名**（私有的，
#: 不该出现在任何地方）；这里装的是**算子名**（公开的，出现在卡片正文里
#: 不是泄漏，而是把一条硬件机制绑死在一个算子上）。两者的处理也不同 ——
#: 前者报 L-DENY，后者报 L-NARRATIVE。
OPERATOR_NAMES_FILENAME = "operator_names.txt"

#: 人类专用归档目录。参考实现的文件名、函数名、行号、转述内容去这里。
#:
#: 为什么要有这么一个地方：真从参考行为反推出结论的 agent 手上确实握着
#: 参考特有的细节。不给它一个合法去处，它只有两条路 —— 写进卡片（越过
#: no-code 边界），或者丢掉（结论从此不可复核）。它会选后者，因为那是免费的。
#:
#: 这个目录不在 `iter_cards` 的遍历范围内（它只扫 ARCH_DIRS 下的 *.md），
#: 所以 `query` 不会把它喂给 agent。要让 agent **读不到**它，还需要
#: hooks.json 里那条 `Read(//**/forge-hwkb/references/provenance/**)`
#: 的 deny —— 目录布局只保证「不主动递给它」，deny 才保证「它拿不到」。
PROVENANCE_DIRNAME = "provenance"

#: 隔离区。lint 不过的卡片移到这里，移出 agent 可读的那棵树。
#:
#: 为什么不是直接删：lint 不过的卡片里可能有真经验，只是写法越界。
#: 删掉等于连经验一起丢，而留在原处等于继续供给 —— 隔离是第三条路。
QUARANTINE_DIRNAME = "quarantine"

#: 隔离之前先给几次返修机会。
#:
#: 上游的做法是「不过先返修两轮，仍不过才移出学生可读的树」。直接隔离的
#: 问题不在于严格，而在于**第一次失败往往是形式问题**：少了一个小节、
#: provenance 的 kind 写错、归档指针拼错。这些改一行就好，而隔离会让这张
#: 卡片离开 agent 的视野 —— 于是那条真经验也跟着离开，直到有人想起来去
#: quarantine 目录里翻。
#:
#: 计数记在卡片的 frontmatter 里（`lint_failures`），所以它跨进程存活 ——
#: 记在内存里的话，每次 lint 都是第一次，这条规则等于不存在。
QUARANTINE_AFTER_FAILURES = 2

#: 返修计数字段名。
LINT_FAILURES_FIELD = "lint_failures"

#: 禁用词。这些词把「这条经验是从哪来的」带进了可复用经验里。
#:
#: 前一半是上游明确列出的三个（`oracle` / `teacher` / `reference
#: implementation`）。后一半是它们在本仓的中文对应 —— 我们的卡片正文
#: **强制中文**（AGENTS.md 知识沉淀约定），所以中文那一半才是主要形态。
BANNED_TERMS = (
    "oracle",
    "teacher",
    "reference implementation",
    "reference impl",
    "reference kernel",
    "reference source",
    "golden kernel",
    "参考实现",
    "参考源码",
    "参考代码",
    "参考侧源码",
    "标杆实现",
    "标杆源码",
    "教师模型",
)

#: 禁用词与叙事规则的词边界**只认 ASCII**。
#:
#: 理由是 Python 正则的一个具体行为：`\w`（因而 `\b`）把 CJK 当词字符。
#: 于是 `\boracle\b` 对「oracle的实现」**不匹配** —— `e` 是词字符、`的`
#: 也是词字符，两者之间没有边界。而卡片正文强制中文，所以「英文词紧跟
#: 一个汉字」不是偶发情形，是默认情形：用 `\b` 等于这条规则基本不生效。
#:
#: 换成 ASCII 字符类的前后否定环视之后，`的` 不在 `[A-Za-z0-9_]` 里，
#: 右边界成立，匹配照常。同时仍然挡住 `DataCopyEx` 吃掉
#: `DataCopyExtParams` 那类误报（见 compile_denylist）。
ASCII_LEFT = r"(?<![A-Za-z0-9_])"
ASCII_RIGHT = r"(?![A-Za-z0-9_])"

#: 叙事规则一：运行日期。一条带日期的经验读起来是 campaign 日记，
#: 而日记对别的 campaign 没有用 —— 它说的是「那天发生了什么」，
#: 不是「这个硬件在什么条件下会这样」。
RE_NARR_DATE = re.compile(
    r"(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)"
    r"|\d{4}\s*年\s*\d{1,2}\s*月")

#: 叙事规则二：轮次与代际。同上，而且它还会把读者引向一份它读不到的台账。
RE_NARR_ROUND = re.compile(
    r"第\s*\d+\s*[轮代次]"
    r"|" + ASCII_LEFT + r"(?:iter|iteration|round|gen|generation|epoch)"
    r"\s*#?\s*\d+" + ASCII_RIGHT,
    re.IGNORECASE)

#: 叙事规则三：commit sha。指向一个仓库状态，而卡片要跨仓库活下去。
#:
#: 形状先粗筛（7~40 位十六进制且两侧无 ASCII 词字符），再由
#: `_looks_like_sha` 做两道排除。写成两步而不是一条巨型正则，
#: 是因为那两道排除讲清理由比塞进环视更重要。
RE_HEXISH = re.compile(ASCII_LEFT + r"[0-9a-f]{7,40}" + ASCII_RIGHT)

#: 卡片 id 的合法形状：R-<架构号>-<序号>，如 R-2201-001。
#:
#: 为什么要校验：id 会被直接拼成入库文件名（<kb_root>/<arch>/<id>.md）。
#: 不校验的话，`id: ../../../../tmp/x` 这种写法会把文件写到知识库之外，
#: 而且「目标已存在就要 --force」那道保护也一起失效 —— 判存在性用的是
#: 拼好的路径，指向哪里取决于 id 本身。
#: 顺带还挡住两类更常见的事故：id 里带空格或斜杠导致检索对不上、
#: 大小写不一致导致同一条经验入库两次。
RE_CARD_ID = re.compile(r"^R-\d{3,5}-\d{3,4}$")

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

def check_certification_against_ledger(
        meta: Dict[str, Any]) -> Tuple[List[Issue], List[str]]:
    """把「这条经验的认证」从一句散文变成一个可机器核对的指针。

    A-CERT 那几条只查 job / arch / delta 三个字段**非空** —— 于是
    `job: "测过了"`、`delta: "更快"` 全部通过。而真正的判据
    campaign 早就算出来了：`kb_admissible`（增益超噪声底 **且** 机制已核实），
    只是从来没人读它。

    所以这里接上：`certified_by` 里给 `state` + `decision` 两个指针时，
    回到那份 campaign 台账里核对该决策确实拿到了 `kb_admissible`。
    不给指针时不拒收（老卡片、手工整理的经验都没有指针），但要说清这条
    认证是自述的、没有被机器核对过 —— 两种卡片的可信度不该看起来一样。

    返回 (阻塞问题, 提示)。没有指针只进「提示」：拒收会把老卡片一起挡住，
    而指针**在而对不上**才说明有人在伪造认证，那是阻塞问题。
    """
    out: List[Issue] = []
    notes: List[str] = []
    cert = meta.get("certified_by")
    if not isinstance(cert, dict):
        return out, notes

    state_path = cert.get("state")
    decision = cert.get("decision")
    if not state_path and not decision:
        notes.append(
            "certified_by 没有 state/decision 指针，这条认证是**自述**的 —— "
            "job 和 delta 两个字段只被查了非空，没有任何机器核对。"
            "由 campaign 产出的经验请补上 state（campaign_state.json 路径）"
            "与 decision（决策 id），入库时会回台账核对 kb_admissible"
            "（增益超噪声底且机制已核实）。")
        return out, notes

    if not state_path or not decision:
        out.append(Issue(
            "A-CERT-LINK",
            "certified_by 的 state 与 decision 必须成对出现，"
            "只有一个无法回台账定位到那一轮裁决",
        ))
        return out, notes

    path = Path(str(state_path))
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        out.append(Issue("A-CERT-LINK", f"读不到 certified_by.state 指向的 {path}: {exc}"))
        return out, notes
    except json.JSONDecodeError as exc:
        out.append(Issue("A-CERT-LINK", f"{path} 不是合法 JSON: {exc}"))
        return out, notes
    if not isinstance(state, dict):
        out.append(Issue("A-CERT-LINK", f"{path} 的顶层不是对象"))
        return out, notes

    ledger = state.get("ledger")
    if not isinstance(ledger, list):
        out.append(Issue("A-CERT-LINK", f"{path} 里没有台账（ledger）"))
        return out, notes

    hits = [e for e in ledger if isinstance(e, dict)
            and e.get("decision") == decision]
    if not hits:
        out.append(Issue(
            "A-CERT-LINK",
            f"台账 {path} 里没有决策 {decision!r} 的记录 —— "
            "认证指针指向了一轮不存在的裁决"))
        return out, notes

    admissible = [e for e in hits
                  if e.get("kb_admissible") and not e.get("invalidated")]
    if not admissible:
        verdicts = sorted({str(e.get("verdict")) for e in hits})
        out.append(Issue(
            "A-CERT-NOT-ADMISSIBLE",
            f"决策 {decision} 在台账里的裁决是 {verdicts}，"
            "没有一条标了 kb_admissible。\n"
            "    kb_admissible 的含义是「增益超过噪声底**且**机制已核实」——"
            "只有这种改动才有资格变成可复用经验。\n"
            "    一次 accept_unverified（变快了但机制没核实）或 revert "
            "都不构成经验：那是一个巧合，\n"
            "    写进知识库之后会被其他算子照着做决策。"))
        return out, notes

    entry = admissible[0]
    arch_in_cert = cert.get("arch")
    arch_in_state = state.get("arch")
    if arch_in_cert and arch_in_state and arch_in_cert != arch_in_state:
        out.append(Issue(
            "A-CERT-ARCH",
            f"certified_by.arch 是 {arch_in_cert}，而那次 campaign 跑在 "
            f"{arch_in_state} 上。硬件机制与架构绑定，不能跨架构套用"))

    gain = entry.get("gain")
    if isinstance(gain, (int, float)) and gain <= 1.0:
        out.append(Issue(
            "A-CERT-DELTA",
            f"台账里该轮的增益是 {gain}（没有变快），"
            "不足以支撑一条正向经验"))
    return out, notes


def check_provenance_archive(meta: Dict[str, Any],
                             kb_root: Optional[Path] = None) -> List[Issue]:
    """人类专用归档指针：参考侧的细节有没有被安放，而不是被丢掉。

    这条规则补的是一个漏洞，而不是加一道形式要求。agent 真从参考行为反推
    出结论时，手上握着参考特有的细节（哪个文件、哪个函数、第几行、转述的
    片段）。卡片里不许写（L-DENY / L-BANNED 会拦），于是只剩两条路：

      写进卡片   被 lint 拦住 → 它会改写成模糊表述绕过去，细节丢了
      直接丢掉   零成本、零报错 → 结论从此不可复核

    两条路都通向「结论不可复核」。给一个合法去处之后才有第三条路。

    `kb_root` 给了就连文件是否存在一起查。不给只查形状 —— 这是为了让
    「起草时自检」这种没有库上下文的调用也能用，三个真实命令都会传。
    """
    out: List[Issue] = []
    pointer = meta.get("provenance_archive")
    if _is_blank(pointer):
        out.append(Issue(
            "A-ARCHIVE",
            "缺少 provenance_archive：参考侧的细节（文件名、函数名、行号、"
            "转述内容）要有一个合法去处。\n"
            f"    写成 '{PROVENANCE_DIRNAME}/<卡片 id>.md'，"
            "内容放到知识库根的同名文件里。\n"
            "    该目录不进 query 的供给范围，也被 hooks.json 的 deny 挡住 ——"
            "它是给人看的。\n"
            "    确实没有参考侧细节（结论纯粹来自公开文档与自家探针）时，"
            "归档文件里就写这一句。"))
        return out

    text = str(pointer).strip()
    expected_prefix = PROVENANCE_DIRNAME + "/"
    if not text.startswith(expected_prefix):
        out.append(Issue(
            "A-ARCHIVE",
            f"provenance_archive '{text}' 不在 {PROVENANCE_DIRNAME}/ 下。"
            "归档必须落在那个目录里 —— 别处没有 deny 规则保护，"
            "等于把参考细节放回 agent 的读集"))
        return out

    relative = text[len(expected_prefix):]
    if not relative or "/" in relative or ".." in relative:
        out.append(Issue(
            "A-ARCHIVE",
            f"provenance_archive '{text}' 的文件名部分不合规；"
            f"要求形如 '{PROVENANCE_DIRNAME}/R-2201-001.md'（不含子目录）"))
        return out

    card_id = str(meta.get("id", "")).strip()
    if card_id and relative != f"{card_id}.md":
        out.append(Issue(
            "A-ARCHIVE",
            f"provenance_archive 指向 '{relative}'，但卡片 id 是 "
            f"'{card_id}'。一卡一档，否则两张卡片共用一份归档时，"
            "退役其中一张就会连另一张的证据一起带走"))

    if kb_root is not None and not (kb_root / text).is_file():
        out.append(Issue(
            "A-ARCHIVE",
            f"归档文件 {kb_root / text} 不存在。指针指向空处等于没有归档 ——"
            "先把参考侧细节写进那个文件"))
    return out


def check_admission(meta: Dict[str, Any], body: str,
                    kb_root: Optional[Path] = None) -> List[Issue]:
    """准入校验：没有探针实测背书的经验一律拒收。

    论文 §3.4：提案必须经过目标架构上的探针认证才能入库；认证把真正起
    作用的机制和只是伴随发生的分开。
    """
    out: List[Issue] = []

    for field in ("id", "arch", "mechanism", "symptom", "symptom_channel"):
        if _is_blank(meta.get(field)):
            out.append(Issue("A-FIELD", f"frontmatter 缺少必填字段 '{field}'"))

    card_id = meta.get("id")
    if card_id is not None and not _is_blank(card_id):
        if not RE_CARD_ID.match(str(card_id).strip()):
            out.append(Issue(
                "A-ID",
                f"卡片 id '{card_id}' 不合规；要求形如 R-2201-001"
                "（R-<架构号>-<序号>）。id 会被直接拼成入库文件名，"
                "带路径分隔符或其他字符会把文件写到知识库之外，"
                "「已存在需 --force」那道保护也会一起失效",
            ))

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

    out += check_provenance_archive(meta, kb_root)

    return out


# ---------------------------------------------------------------------------
# no-code 边界 lint
# ---------------------------------------------------------------------------

def load_denylist(kb_root: Path) -> List[str]:
    """读取参考实现专有名词清单。

    这份清单由 `seal_reference.py seal` 写入（它同时写工作目录和这里）：
    封存一个参考时把它的内部符号名登记进来，之后任何卡片带上这些词都会
    被 L-DENY 挡下。

    清单缺失或为空时返回空表，但**调用方必须据此报告 L-DENY 未生效**——
    见 denylist_status。早先这里静默返回空表，而 seal 又只写工作目录，
    于是 L-DENY 在真实使用中一条都不报，`lint` 照样打印「通过」。
    一道关不上的门比没有门更糟：它会让人以为查过了。
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


def denylist_status(kb_root: Path, denylist: List[str]) -> Optional[str]:
    """L-DENY 这道闸到底有没有生效。返回要报告的问题，正常则 None。"""
    path = kb_root / DENYLIST_FILENAME
    if not path.exists():
        return (f"没有 {path} —— L-DENY（参考专有名词）这道检查**未生效**。\n"
                "  这份清单由封存那一步写入：\n"
                "    python3 <forge-campaign>/scripts/seal_reference.py seal \\\n"
                "        --artifact <参考> --op-name <算子> --work-dir <工作目录>\n"
                "  未封存任何参考时这是正常的；已经在跑 campaign 就说明接线断了。")
    if not denylist:
        return (f"{path} 里没有任何名词 —— L-DENY 这道检查形同不存在。\n"
                "  参考若确实是二进制（如 CANN 内置实现），没有符号名可抽，"
                "这是正常的；\n"
                "  否则请检查封存是否真的抽到了名词（看 sealed_manifest.json "
                "的 identifier_count）。")
    return None


def compile_denylist(denylist: List[str]) -> Optional[re.Pattern]:
    """把专有名词清单编成一条带词边界的正则。

    为什么要词边界：裸子串匹配会让清单里的短名字吃掉候选里更长的名字 ——
    `DataCopyEx` 命中 `DataCopyExtParams`，而后者是另一个符号。这种误报会
    让人把整条检查豁免掉，真越界也跟着放过去。

    为什么合成一条：逐个名词扫全文是「名词数 × 文本长度」，清单上千条时
    每张卡片的 lint 都要等。合成一条正则，全文只过一遍。
    """
    tokens = sorted({t for t in denylist if t}, key=lambda t: (-len(t), t))
    if not tokens:
        return None
    body = "|".join(re.escape(token) for token in tokens)
    # 边界取严，与 fence_check 的 compile_denylist 保持同一判据（理由见那边）：
    # 放宽右边界就分不开「参考类名加后缀」和「另一个公开接口名」，而误伤的
    # 代价是整条规则被豁免掉。
    return re.compile(rf"(?<![A-Za-z0-9_])({body})(?![A-Za-z0-9_])",
                      re.IGNORECASE)


def _looks_like_sha(token: str) -> bool:
    """这串十六进制看起来是 commit sha 吗。

    两道排除，各挡一类误报：
      全是数字   `8388608`（理论最小搬运字节数）落在 [0-9a-f] 里，但它是个数
      全是字母   `defaced` 这类英文词正好只用 a-f，但它是个词
    真 sha 几乎必然两者兼有。宁可漏掉少数纯数字/纯字母的 sha ——
    误报的代价是整条规则被豁免掉，那样连真 sha 也不拦了。
    """
    return (any(c.isdigit() for c in token)
            and any(c in "abcdef" for c in token))


def load_operator_names(kb_root: Path) -> List[str]:
    """读算子名清单。和 denylist 同一套写入/读取约定。"""
    path = kb_root / OPERATOR_NAMES_FILENAME
    if not path.exists():
        return []
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            out.append(token)
    return out


def operator_names_status(kb_root: Path, names: List[str]) -> Optional[str]:
    """算子名这半条规则有没有生效。返回要报告的话，正常则 None。

    和 denylist_status 同样的纪律：这半条规则靠一份外部文件驱动，文件不在
    时它是个空操作。空操作本身不要紧，**空操作加一句「通过」**才要紧 ——
    那正是 L-DENY 曾经的样子（seal 写工作目录、hwkb 读知识库根，中间没人
    搬运，于是这道检查一条都不报，而 lint 照样打印「通过」）。
    """
    if names:
        return None
    path = kb_root / OPERATOR_NAMES_FILENAME
    return (f"没有 {path}（或它是空的）—— L-NARRATIVE 里的「算子名」"
            "那半条**未生效**。\n"
            "  这份清单由封存那一步写入（与 denylist.txt 同一次写入）。\n"
            "  日期 / 轮次 / commit sha 三条不依赖任何文件，始终生效。")


def check_banned_terms(meta: Dict[str, Any], body: str) -> List[Issue]:
    """禁用词：卡片不得提到「有个参考实现」这件事本身。

    denylist 拦的是参考的**私有符号名**，这里拦的是另一类词：`oracle`、
    `teacher`、`参考实现`。它们不是任何一份源码里的标识符，所以 denylist
    永远抓不到 —— 但它们一样会把卡片钉死在一次具体的逆向上：读到
    「比参考实现慢 20%」的人无从判断那个参考是什么，这句话对他没有信息量。
    """
    out: List[Issue] = []
    haystack = yaml.safe_dump(meta, allow_unicode=True) + "\n" + body
    for term in BANNED_TERMS:
        pattern = re.compile(ASCII_LEFT + re.escape(term) + ASCII_RIGHT,
                             re.IGNORECASE)
        if pattern.search(haystack):
            out.append(Issue(
                "L-BANNED",
                f"出现禁用词 '{term}'。卡片要能被一个**没有参考实现**的 "
                "campaign 读懂：\n"
                "    把结论改写成硬件条件（「MTE2 占比 > 50% 且复用次数 ≥ 4 时」），"
                "而不是相对关系（「比参考慢」）。\n"
                f"    参考侧的具体细节放 {PROVENANCE_DIRNAME}/<id>.md"))
    return out


def check_narrative(meta: Dict[str, Any], body: str,
                    operator_names: Optional[List[str]] = None) -> List[Issue]:
    """叙事规则：卡片不是 campaign 日记。

    四条，前三条零接线、始终生效；第四条要一份算子名清单。

      日期      「2026-11-03 那次」对别的 campaign 没有意义
      轮次代际  「第 3 轮发现」指向一份读者拿不到的台账
      commit    指向一个仓库状态，而卡片要跨仓库活下去
      算子名    把一条**硬件**机制绑死在一个算子上，而跨算子复用正是
                这个知识库存在的全部理由；算子适用范围有
                `operator_families` 这个结构化字段承载，正文里重提
                只剩下「这条是从哪个算子的逆向里来的」这一层含义

    扫描范围刻意不含 `certified_by` 与 `provenance`：那两处的职责恰恰是
    **指向一次具体的测量**，日期和 job 号写得越具体越好。叙事规则管的是
    结论的表述，不是证据的指针。
    """
    out: List[Issue] = []
    narrative_fields = ("mechanism", "symptom")
    haystack = "\n".join(
        [str(meta.get(f, "")) for f in narrative_fields] + [body])

    hit = RE_NARR_DATE.search(haystack)
    if hit:
        out.append(Issue(
            "L-NARRATIVE",
            f"正文里有运行日期 '{hit.group(0)}'。卡片讲的是硬件在什么条件下"
            "会这样，不是那天发生了什么 —— 日期去 certified_by.job"))

    hit = RE_NARR_ROUND.search(haystack)
    if hit:
        out.append(Issue(
            "L-NARRATIVE",
            f"正文里有轮次/代际 '{hit.group(0).strip()}'。它指向一份读者"
            "拿不到的台账；要表达的条件请直接写成条件"))

    for match in RE_HEXISH.finditer(haystack):
        if _looks_like_sha(match.group(0)):
            out.append(Issue(
                "L-NARRATIVE",
                f"正文里有 commit sha '{match.group(0)}'。它指向一个仓库"
                "状态，而这张卡片要跨仓库、跨代际活下去"))
            break

    for name in operator_names or []:
        pattern = re.compile(ASCII_LEFT + re.escape(name) + ASCII_RIGHT,
                             re.IGNORECASE)
        if pattern.search(haystack):
            out.append(Issue(
                "L-NARRATIVE",
                f"正文里有算子名 '{name}'。硬件机制跨算子成立，写上具体算子"
                "会让读者按「这条是别的算子的」跳过它；\n"
                "    适用范围写进 operator_families，正文只留硬件条件"))
    return out


def check_no_code_boundary(meta: Dict[str, Any], body: str,
                           denylist: List[str],
                           operator_names: Optional[List[str]] = None
                           ) -> List[Issue]:
    """边界检查：卡片不得留下参考实现的痕迹。

    论文 §3.4：参考特有的名字留在人类专用归档里，由确定性 lint 保持分离；
    因此一条经验不保留它被总结时所针对的那个参考的痕迹 —— 这正是它能被
    另一个只共享硬件的 campaign 读懂的原因。

    四类痕迹，判据各不相同，所以分成四段：

      L-DENY       参考的私有符号名（靠封存抽出的清单）
      L-SRC        出处里的源码路径
      L-BANNED     「参考实现」「oracle」这类词 —— 它们不是任何源码里的
                   标识符，所以 denylist 永远抓不到
      L-NARRATIVE  日期 / 轮次 / commit sha / 算子名

    后两类挂在这个函数里而不是另起一个入口，是因为 validate、add、lint
    三个命令都调它 —— 新规则自动对三者生效。另起一个入口就要记得在三处
    各接一次，而漏掉一处的后果是那条命令上的规则静默失效。
    """
    out: List[Issue] = []
    haystack = yaml.safe_dump(meta, allow_unicode=True) + "\n" + body

    pattern = compile_denylist(denylist)
    if pattern is not None:
        # 报错信息里用清单上的原始拼写，方便按原文去清单里核对
        canonical = {t.lower(): t for t in denylist if t}
        reported: set = set()
        for match in pattern.finditer(haystack):
            hit = match.group(1)
            token = canonical.get(hit.lower(), hit)
            if token in reported:
                continue
            reported.add(token)
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

    out += check_banned_terms(meta, body)
    out += check_narrative(meta, body, operator_names)

    return out


# ---------------------------------------------------------------------------
# 知识库读写
# ---------------------------------------------------------------------------

def arch_dir(kb_root: Path, arch: str) -> Path:
    sub = ARCH_DIRS.get(arch)
    if sub is None:
        raise ValueError(f"未知架构 '{arch}'；已知: {sorted(ARCH_DIRS)}")
    return kb_root / sub


def iter_cards(kb_root: Path, arch: Optional[str] = None,
               unreadable: Optional[List[Tuple[Path, str]]] = None):
    """遍历知识库里的卡片，产出 (路径, meta, body)。

    解析失败的卡片不会被产出，但**会被记到 unreadable 里**（调用方给了
    这个列表的话）。为什么不能一声不响地跳过：`lint --all` 就是靠这个函数
    遍历的，一个解析不动的文件被跳过之后，lint 会报「全部通过」——
    而解析不动的最常见原因恰恰是有人把参考源码整段粘进了卡片，
    把 frontmatter 撑坏了。最该被 lint 拦住的东西，成了唯一躲过 lint 的。
    """
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
            except (ValueError, OSError) as exc:
                if unreadable is not None:
                    unreadable.append((path, str(exc)))
                continue
            yield path, meta, body


#: 索引里被自动重写的那一段的起止标记。
#:
#: 为什么只重写一段而不是整个文件：INDEX.md 里手写的那部分（为什么按架构
#: 分目录、怎么加卡片、怎么检索）是人和 agent 都要读的导读，自动生成的
#: 那张表替代不了它。整份重写会把它删掉，而「我的生成器需要这个位置」
#: 从来不是删掉一段有用文档的理由。
INDEX_BEGIN = "<!-- hwkb:index:begin 以下由 hwkb.py 自动重写，勿手改 -->"
INDEX_END = "<!-- hwkb:index:end -->"


def render_index_table(kb_root: Path, arch: str) -> str:
    """把一个架构下的卡片渲染成一张「症状 → 机制 → 卡片」密集表。

    为什么要这张表：上游每一轮**无条件**把它塞进架构 agent 的读集，理由是
    检索要先想到去检索。`query` 是拉取式的 —— agent 得先怀疑「这个现象
    别人遇到过」才会去查，而它通常不会怀疑。一张随输入一起到达的密集表
    把这一步从「想起来」变成「看见」。

    所以这张表按症状排，而不是按 id —— agent 的实际问句是「我观测到了
    这个」，不是「第 1 号经验是什么」。
    """
    rows: List[Tuple[str, str, str]] = []
    for path, meta, _ in iter_cards(kb_root, arch):
        if meta.get("status") == "retired":
            continue
        rows.append((
            str(meta.get("symptom", "")).replace("|", "\\|"),
            str(meta.get("mechanism", "")).replace("|", "\\|"),
            path.name,
        ))
    rows.sort()

    lines = [INDEX_BEGIN,
             "",
             "## 现有卡片（按症状）",
             "",
             "先在这里找「我观测到的现象」，命中就去读那张卡片；",
             "都不命中再考虑它是一条新经验。",
             "",
             "| 触发症状 | 机制 | 卡片 |",
             "|---|---|---|"]
    if not rows:
        lines.append("| （暂无经验） | | |")
    for symptom, mechanism, name in rows:
        lines.append(f"| {symptom} | {mechanism} | [{name}]({name}) |")
    lines += ["", INDEX_END]
    return "\n".join(lines)


def render_index(kb_root: Path, arch: str) -> str:
    """生成索引全文：手写部分原样保留，只换掉标记之间那一段。

    没有标记时把生成段**追加**在文件末尾（并保留原有全文）；文件不存在
    时就只有生成段。两种情况都不会丢掉已有内容。
    """
    table = render_index_table(kb_root, arch)
    path = index_path(kb_root, arch)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""

    begin = existing.find(INDEX_BEGIN)
    end = existing.find(INDEX_END)
    if begin >= 0 and end > begin:
        head = existing[:begin]
        tail = existing[end + len(INDEX_END):]
        return head + table + tail
    if existing.strip():
        return existing.rstrip("\n") + "\n\n" + table + "\n"
    return table + "\n"


def index_path(kb_root: Path, arch: str) -> Path:
    return arch_dir(kb_root, arch) / "INDEX.md"


def write_index(kb_root: Path, arch: str) -> Path:
    path = index_path(kb_root, arch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_index(kb_root, arch), encoding="utf-8")
    return path


def check_index_fresh(kb_root: Path, arch: str) -> List[Issue]:
    """索引里的生成段和目录内容对不对得上。

    `add` 会顺手重写它，所以正常路径下这条永远通过。它盯的是另一条路径：
    有人直接往 `dav2201/` 丢了个 .md（或手工删了一张）。那样生成段里就少
    一行 —— 而那张卡片从此只能靠 `query` 的关键词碰巧命中，架构 agent
    读到的那张密集表里它不存在。

    只比对生成段。手写导读怎么改都不算「对不上」—— 把手写部分也纳入
    比对，等于规定那段文字不许动，而那段文字本来就是给人改的。
    """
    path = index_path(kb_root, arch)
    if not any(True for _ in iter_cards(kb_root, arch)):
        # 一张卡片都没有的架构目录不需要索引。这是知识库的正常起点 ——
        # 在这里报问题，第一次 lint 就会红，而红在一个没有任何人做错事的
        # 地方。规则在这种地方响一次，下一次就会被人整条关掉。
        return []
    if not path.exists():
        return [Issue("L-INDEX", f"{path} 不存在；跑一次 `hwkb.py reindex` 生成")]
    existing = path.read_text(encoding="utf-8")
    begin = existing.find(INDEX_BEGIN)
    end = existing.find(INDEX_END)
    if begin < 0 or end <= begin:
        return [Issue(
            "L-INDEX",
            f"{path} 里没有 `{INDEX_BEGIN}` / `{INDEX_END}` 标记，"
            "自动索引段不存在。\n"
            "    跑 `hwkb.py reindex` 把它追加上去（手写内容会原样保留）")]
    if existing[begin:end + len(INDEX_END)] != render_index_table(kb_root, arch):
        return [Issue(
            "L-INDEX",
            f"{path} 的自动索引段与目录内容对不上（多半是直接往目录里"
            "丢了卡片，没走 add）。\n"
            "    索引是架构 agent 每轮无条件读到的那张表，漏一行就等于"
            "那条经验只能靠关键词碰巧命中。\n"
            "    跑 `hwkb.py reindex` 重建")]
    return []


def bump_lint_failures(card_path: Path) -> Optional[int]:
    """给卡片的不过检计数加一，返回加完之后的值。

    计数写进 frontmatter 而不是记在内存里：记在内存里的话，每次 lint 都是
    第一次，「返修两轮再隔离」这条规则等于不存在。

    frontmatter 读不动时返回 None —— 调用方据此直接隔离。那类卡片最常见的
    成因恰恰是正文里粘了大段源码把 frontmatter 撑坏了，而它也无处记计数。
    """
    try:
        meta, body = load_card(card_path)
    except (ValueError, OSError):
        return None
    count = int(meta.get(LINT_FAILURES_FIELD) or 0) + 1
    meta[LINT_FAILURES_FIELD] = count
    try:
        card_path.write_text(
            "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False)
            + "---\n" + body,
            encoding="utf-8")
    except OSError:
        return None
    return count


def clear_lint_failures(card_path: Path) -> None:
    """卡片过检之后清掉计数。

    不清的话，一张修好又改坏的卡片会带着历史计数直接被隔离 —— 而它这次
    只错了一处。计数要衡量的是「连续不过检」，不是「一辈子不过检过几次」。
    """
    try:
        meta, body = load_card(card_path)
    except (ValueError, OSError):
        return
    if LINT_FAILURES_FIELD not in meta:
        return
    meta.pop(LINT_FAILURES_FIELD, None)
    try:
        card_path.write_text(
            "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False)
            + "---\n" + body,
            encoding="utf-8")
    except OSError:
        pass


def quarantine(kb_root: Path, card_path: Path, arch: str) -> Path:
    """把一张不过 lint 的卡片移出 agent 可读的那棵树。

    不是删除：越界的卡片里往往有真经验，只是写法不合规。留在原处则会
    继续被 query 供给出去 —— 它越界的那部分恰恰是最可能被照抄的部分。
    """
    target_dir = kb_root / QUARANTINE_DIRNAME / ARCH_DIRS.get(arch, "unknown")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / card_path.name
    shutil.move(str(card_path), str(target))
    return target


def _gate_status(kb_root: Path) -> Tuple[List[str], List[str],
                                         List[Tuple[str, str]]]:
    """把两份外部清单读齐，并把「哪道闸没生效」一并带出来。

    返回 (denylist, operator_names, [(闸名, 说明), ...])。

    为什么要专门有这个函数：L-DENY 和 L-NARRATIVE 的算子名那半条都靠
    外部文件驱动，文件不在时它们是空操作。曾经犯过的错是「空操作 + 一句
    通过」，所以这里把状态说明和清单一起返回 —— 调用方拿到清单的同时
    必然也拿到了状态，不存在只取前者的写法。

    闸名要跟着说明一起回去，而不是只回一句「有 N 道闸未生效，见上」：
    读结论的人得当场知道**哪一项**没被查过，而不是去上面的 stderr 里
    自己对。
    """
    denylist = load_denylist(kb_root)
    operator_names = load_operator_names(kb_root)
    notes: List[Tuple[str, str]] = []
    detail = denylist_status(kb_root, denylist)
    if detail:
        notes.append(("L-DENY", detail))
    detail = operator_names_status(kb_root, operator_names)
    if detail:
        notes.append(("L-NARRATIVE 的算子名", detail))
    return denylist, operator_names, notes


def _coverage_suffix(notes: List[Tuple[str, str]]) -> str:
    """结论后面那句「这句话不覆盖什么」。"""
    if not notes:
        return ""
    names = "、".join(name for name, _ in notes)
    return f"（但 {names} 未生效，这句话不覆盖它们）"


def cmd_validate(args: argparse.Namespace) -> int:
    """只校验不入库，供 agent 起草时自检。"""
    kb_root = Path(args.kb_root)
    try:
        meta, body = load_card(Path(args.card))
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    denylist, operator_names, notes = _gate_status(kb_root)
    for _, detail in notes:
        print(f"注意: {detail}", file=sys.stderr)
    issues = check_admission(meta, body, kb_root)
    cert_issues, cert_notes = check_certification_against_ledger(meta)
    issues += cert_issues + check_no_code_boundary(
        meta, body, denylist, operator_names)
    for note in cert_notes:
        print(f"注意: {note}", file=sys.stderr)
    label = f"卡片 {args.card} 满足准入与边界要求" + _coverage_suffix(notes)
    return _report(issues, args.as_json, label)


def cmd_add(args: argparse.Namespace) -> int:
    """校验通过才写入知识库。不通过就原样拒绝，不做任何「尽力而为」的降级。"""
    kb_root = Path(args.kb_root)
    card_path = Path(args.card)
    try:
        meta, body = load_card(card_path)
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    denylist, operator_names, notes = _gate_status(kb_root)
    issues = check_admission(meta, body, kb_root)
    cert_issues, cert_notes = check_certification_against_ledger(meta)
    issues += cert_issues + check_no_code_boundary(
        meta, body, denylist, operator_names)
    for note in cert_notes:
        print(f"注意: {note}", file=sys.stderr)
    for _, detail in notes:
        # 不硬拒。封存之前、或参考本就是二进制（没有符号名可抽）时，
        # 清单缺失是正常状态，拒绝会把正当用法一起卡住 —— 而人会反射性地
        # 去加豁免参数，那和静默放行是同一件事。
        #
        # 真正的缺陷是「这道闸没生效，而输出说通过」。所以这里只做一件事：
        # 把它说出来，并在入库成功的那句话里标明覆盖范围。
        print(f"注意: {detail}", file=sys.stderr)
    if issues:
        print(f"拒绝入库：{card_path} 有 {len(issues)} 项问题", file=sys.stderr)
        for issue in issues:
            print(f"  {issue}", file=sys.stderr)
        return 1

    arch = str(meta["arch"])
    try:
        target_dir = arch_dir(kb_root, arch)
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    card_id = str(meta["id"]).strip()
    if not RE_CARD_ID.match(card_id):
        # 准入校验已经查过一遍。这里再查是因为「校验」和「拼路径」之间
        # 隔着好几十行，任何一次重构把前者绕过去，后者就会往任意位置写文件。
        # 拼路径之前自己确认一次，不依赖调用链上游。
        print(f"错误: 卡片 id '{card_id}' 不合规，拒绝据此拼接入库路径",
              file=sys.stderr)
        return 2

    try:
        with exclusive(kb_root):
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{card_id}.md"
            if target.exists() and not args.force:
                print(f"错误: {target} 已存在；如确需覆盖请加 --force",
                      file=sys.stderr)
                return 2
            shutil.copyfile(card_path, target)
            # 索引在锁内重写。放到锁外的话，两个任务同时入库时后一次重写
            # 可能基于前一次复制之前的目录快照 —— 卡片进了库而索引里没有它。
            written_index = write_index(kb_root, arch)
    except TimeoutError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    print(f"已入库: {target}")
    print(f"已重建索引: {written_index}")
    for name, _ in notes:
        print(f"  注意: 入库时 {name} 未生效 —— 这张卡片的该项没有被查过。")
    if notes:
        print("        把清单补齐之后用 lint --all 重查一遍。")
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
    unreadable: List[Tuple[Path, str]] = []
    for path, meta, body in iter_cards(kb_root, args.arch, unreadable):
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

    # 检索不到和「有卡片没读进来」是两件事。不说的话，一次「未命中任何经验」
    # 会被当成「库里确实没有」，于是一条已有的经验被白白重走一遍。
    for path, reason in unreadable:
        print(f"提示: {path.name} 无法解析（{reason}），未纳入本次检索。"
              "若它可能含相关经验，请先修好这张卡片。", file=sys.stderr)

    return 0 if hits else 1


def cmd_lint(args: argparse.Namespace) -> int:
    """对单张卡片或整库做检查。

    整库模式下跑的是**准入 + 边界**两套，不是只有边界。理由很具体：准入
    只在 `add` 那一刻执行过一次，而卡片此后会被手工编辑、会被 git 合并、
    会被直接丢进目录。只查边界的 lint 对这些情况会说「通过」，而那句话
    并不包含「这张卡片仍然带着可核验的认证」。
    """
    kb_root = Path(args.kb_root)
    denylist, operator_names, notes = _gate_status(kb_root)
    all_issues: List[Issue] = []
    for _, detail in notes:
        print(f"注意: {detail}", file=sys.stderr)

    if args.card:
        try:
            meta, body = load_card(Path(args.card))
        except ValueError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 2
        all_issues = check_no_code_boundary(
            meta, body, denylist, operator_names)
        scope = args.card
    else:
        count = 0
        unreadable: List[Tuple[Path, str]] = []
        # 要隔离的卡片先攒着：边遍历边移动会改变 glob 的结果。
        offenders: List[Tuple[Path, str]] = []
        for path, meta, body in iter_cards(kb_root, args.arch, unreadable):
            count += 1
            card_issues = check_no_code_boundary(
                meta, body, denylist, operator_names)
            card_issues += check_admission(meta, body, kb_root)
            card_issues += check_certification_against_ledger(meta)[0]
            for issue in card_issues:
                all_issues.append(
                    Issue(issue.code, f"{path.name}: {issue.message}"))
            if card_issues:
                offenders.append((path, str(meta.get("arch", ""))))
            elif args.quarantine and meta.get(LINT_FAILURES_FIELD):
                # 过检了就清掉计数。
                #
                # 不清的话，一张修好又改坏的卡片会带着历史计数直接被隔离 ——
                # 而它这次只错了一处。计数衡量的是「连续不过检」，
                # 不是「一辈子不过检过几次」。
                clear_lint_failures(path)
                print(f"已清除不过检计数: {path.name}（本次过检）",
                      file=sys.stderr)
        # 解析不动的卡片要算成问题，不能跳过。
        #
        # lint 的职责是「库里没有参考实现的痕迹」。一个解析不动的文件没被
        # 检查过，所以这句话对它并不成立 —— 而解析不动最常见的原因恰恰是
        # 有人把参考源码整段粘进了卡片，把 frontmatter 撑坏了。
        # 跳过它等于：最该被 lint 拦住的东西，成了唯一躲过 lint 的。
        for path, reason in unreadable:
            all_issues.append(Issue(
                "L-PARSE",
                f"{path.name}: 无法解析（{reason}）。"
                "未经检查的卡片不能算「未越界」—— "
                "frontmatter 坏掉的常见原因是正文里粘了大段源码"))
            offenders.append((path, "unknown"))

        for arch in ([args.arch] if args.arch else list(ARCH_DIRS)):
            if arch_dir(kb_root, arch).is_dir():
                for issue in check_index_fresh(kb_root, arch):
                    all_issues.append(issue)

        if getattr(args, "quarantine", False) and offenders:
            moved_any: List[Tuple[str, str]] = []
            bumped: List[Tuple[str, int]] = []
            for path, arch in offenders:
                count = bump_lint_failures(path)
                if count is None:
                    # 连 frontmatter 都读不动（L-PARSE 那一类）。计数写不进去，
                    # 所以也没法给返修机会 —— 直接隔离。这类卡片最常见的成因
                    # 恰恰是正文里粘了大段源码，把 frontmatter 撑坏了。
                    target = quarantine(kb_root, path, arch)
                    moved_any.append((path.name, str(target)))
                    continue
                if count >= QUARANTINE_AFTER_FAILURES:
                    target = quarantine(kb_root, path, arch)
                    moved_any.append((path.name, str(target)))
                else:
                    bumped.append((path.name, count))

            for name, target in moved_any:
                print(f"已隔离: {name} -> {target}", file=sys.stderr)
            for name, count in bumped:
                print(f"记一次不过检: {name}"
                      f"（第 {count}/{QUARANTINE_AFTER_FAILURES} 次，"
                      "再不过就隔离）", file=sys.stderr)
            if moved_any:
                for arch in {a for _, a in offenders if a in ARCH_DIRS}:
                    write_index(kb_root, arch)
                print(f"隔离了 {len(moved_any)} 张卡片。它们已移出 query 的"
                      "供给范围，但没有被删除 —— 修好之后用 add 重新入库。",
                      file=sys.stderr)
            if bumped:
                print(f"另有 {len(bumped)} 张记了一次不过检但**留在原处** ——"
                      "第一次失败往往是形式问题（少个小节、kind 写错、"
                      "归档指针拼错），", file=sys.stderr)
                print("      改一行就好。直接隔离会让那条真经验也跟着离开"
                      "agent 的视野。", file=sys.stderr)
        scope = f"{count} 张卡片"

    label = f"{scope} 满足边界与准入要求" + _coverage_suffix(notes)
    return _report(all_issues, args.as_json, label)


def cmd_reindex(args: argparse.Namespace) -> int:
    """重建索引。手工动过目录之后用它，不要手改 INDEX.md。"""
    kb_root = Path(args.kb_root)
    arches = [args.arch] if args.arch else list(ARCH_DIRS)
    try:
        with exclusive(kb_root):
            for arch in arches:
                print(f"已重建: {write_index(kb_root, arch)}")
    except TimeoutError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------------------
# 死路表
# ---------------------------------------------------------------------------

def _nogo_path(kb_root: Path) -> Path:
    return kb_root / NOGO_FILENAME


class NogoError(Exception):
    """死路表读不出来。"""


def load_nogo(kb_root: Path) -> List[Dict[str, Any]]:
    """读死路表。读不动就报错，不要当成空表。

    为什么不能退回空表：`nogo add` 的流程是「读出全部条目 → 追加一条 →
    整份写回」。文件损坏时读成空表，这一写就把整张死路表清空了，只剩刚
    加的那一条。而死路表的作用是防止后续 campaign 重走已经证伪的路 ——
    它被悄悄清空之后，那些路会被一条条重试，每条都要花掉真实的评测预算。

    损坏本身不罕见：写入过程被 Ctrl-C 打断就会留下一个截断的 JSON。
    """
    path = _nogo_path(kb_root)
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise NogoError(f"无法读取死路表 {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NogoError(
            f"死路表 {path} 不是合法 JSON: {exc}\n"
            "  多半是上一次写入被打断留下了截断的文件。\n"
            "  这里不会退回空表继续 —— 那样下一次 nogo add 会把整张表\n"
            "  覆盖成只有一条，之前所有证伪过的方向会被重新尝试一遍。\n"
            f"  请先人工确认 {path} 的内容（必要时从 git 历史恢复）。"
        ) from exc
    if not isinstance(data, dict):
        raise NogoError(
            f"死路表 {path} 的顶层不是对象，无法解析出 entries")
    entries = data.get("entries")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise NogoError(f"死路表 {path} 里的 entries 不是数组")
    return entries


def cmd_nogo_add(args: argparse.Namespace) -> int:
    """登记一条死路。死路和正向经验一样要带证据 —— 没测过就不算死路。"""
    kb_root = Path(args.kb_root)
    if not args.evidence.strip():
        print("错误: 死路同样需要证据；未经测量的放弃不构成死路", file=sys.stderr)
        return 2
    if not args.retry_precondition.strip():
        print("错误: 缺少 --retry-precondition。\n"
              "  这张表是**重试前置条件表**，不是禁令表。一条只说"
              "「别试了」的记录会把一个条件变了就该重试的方向永久封死；\n"
              "  而条件确实会变 —— 换形状、换 tiling、上游决策改了，"
              "同一个方向可能就成立了。\n"
              "  写清楚「什么变了就值得再试一次」，例如："
              "\"UB 余量 > 64KB 时重试\"。\n"
              "  确实在任何条件下都不成立时，写 \"无（机制上不可能）\" "
              "并在证据里说明为什么。",
              file=sys.stderr)
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
                "input_shape": args.input_shape,
                "retry_precondition": args.retry_precondition,
            })
            _nogo_path(kb_root).write_text(
                json.dumps({"entries": entries}, ensure_ascii=False,
                           indent=2) + "\n",
                encoding="utf-8",
            )
    except (TimeoutError, NogoError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    print(f"已登记死路（当前 {len(entries)} 条）: {args.tried}")
    return 0


def nogo_binding(entry: Dict[str, Any], operator: str,
                 input_shape: str) -> Tuple[bool, str]:
    """这条命中对当前这次询问有没有约束力。

    返回 (是否约束, 理由)。约束 = 「别再试了」；不约束 = 「有人在别的
    条件下试过且无效，先读一遍，再自己决定」。

    为什么要分这两档：一条死路记的是「在那个算子、那个形状下无效」。
    搬到另一个形状上，同一个方向可能完全成立 —— 双缓冲在 H=1024 下挤爆
    UB，在 H=256 下不会。一律当禁令用，会把一堆本该成立的方向永久封死，
    而且封得毫无痕迹（调用方只看到 exit 1）。

    反过来，条件完全相同的那次命中必须有约束力，否则这张表只是一份
    读物 —— 而它存在的理由正是省下重走一遍的预算。
    """
    recorded_op = str(entry.get("operator") or "").strip().lower()
    recorded_shape = str(entry.get("input_shape") or "").strip().lower()
    want_op = (operator or "").strip().lower()
    want_shape = (input_shape or "").strip().lower()

    # 条目没记算子 = 算子无关的硬件结论，对谁都成立。
    if recorded_op and want_op and recorded_op != want_op:
        return False, (f"那次是在 {entry.get('operator')} 上测的，"
                       f"这次是 {operator}")
    if recorded_shape and want_shape and recorded_shape != want_shape:
        return False, (f"那次的形状是 {entry.get('input_shape')}，"
                       f"这次是 {input_shape}")
    if recorded_shape and not want_shape:
        return False, (f"那次的形状是 {entry.get('input_shape')}，"
                       "而这次没给 --input-shape，无从判断是不是同一个条件")
    return True, "算子与形状都对得上"


def cmd_nogo_check(args: argparse.Namespace) -> int:
    """重试前置条件：这个症状下有没有已经试过且无效的方向。

    退出码 1 只给**有约束力**的命中（同算子、同形状）。条件不同的命中
    照样打印出来，但退出码是 0 —— 那是一条该读的提示，不是一道禁令。
    """
    kb_root = Path(args.kb_root)
    needle = args.symptom.strip().lower()
    try:
        all_entries = load_nogo(kb_root)
    except NogoError as exc:
        # 不能退回「未命中死路，该方向可以尝试」—— 那是在表读不出来的情况下
        # 给出一个肯定的放行结论，而调用方正是据此决定要不要花预算重试。
        print(f"错误: {exc}", file=sys.stderr)
        print("  死路表读不出来时无法回答「这个方向试过没有」，"
              "不要当成「没试过」。", file=sys.stderr)
        return 2
    hits = [
        e for e in all_entries
        if needle in str(e.get("symptom", "")).lower()
        and (not args.arch or e.get("arch") == args.arch)
    ]
    binding: List[Tuple[Dict[str, Any], str]] = []
    advisory: List[Tuple[Dict[str, Any], str]] = []
    for entry in hits:
        is_binding, why = nogo_binding(entry, args.operator, args.input_shape)
        (binding if is_binding else advisory).append((entry, why))

    if args.as_json:
        print(json.dumps({
            "hit": bool(hits),
            "binding": [e for e, _ in binding],
            "advisory": [{"entry": e, "why_not_binding": w}
                         for e, w in advisory],
        }, ensure_ascii=False, indent=2))
        return 1 if binding else 0

    if binding:
        print(f"命中 {len(binding)} 条死路，以下方向在相同条件下已试过且无效：")
        for entry, _ in binding:
            print(f"  - {entry.get('tried')}")
            print(f"    证据: {entry.get('evidence')}")
            print(f"    什么变了才值得再试: "
                  f"{entry.get('retry_precondition') or '（未记录）'}")
    if advisory:
        print(f"另有 {len(advisory)} 条条件不同的记录，供参考（不构成禁令）：")
        for entry, why in advisory:
            print(f"  - {entry.get('tried')}（{why}）")
            print(f"    证据: {entry.get('evidence')}")
            print(f"    什么变了才值得再试: "
                  f"{entry.get('retry_precondition') or '（未记录）'}")
    if not hits:
        print("未命中死路，该方向可以尝试")
    return 1 if binding else 0


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

    p_lint = sub.add_parser("lint", help="边界 + 准入检查", parents=[common])
    p_lint.add_argument("--card", default=None, help="不给则检查整库")
    p_lint.add_argument("--all", action="store_true", help="检查整库（等价于不给 --card）")
    p_lint.add_argument("--arch", default=None, choices=sorted(ARCH_DIRS))
    p_lint.add_argument("--quarantine", action="store_true",
                        help=f"把不过检的卡片移到 {QUARANTINE_DIRNAME}/，"
                             "移出 query 的供给范围（不删除）")
    p_lint.set_defaults(func=cmd_lint)

    p_ri = sub.add_parser("reindex", help="重建 INDEX.md", parents=[common])
    p_ri.add_argument("--arch", default=None, choices=sorted(ARCH_DIRS))
    p_ri.set_defaults(func=cmd_reindex)

    p_nogo = sub.add_parser("nogo", help="死路表", parents=[common])
    nogo_sub = p_nogo.add_subparsers(dest="nogo_command", required=True)

    p_na = nogo_sub.add_parser("add", help="登记一条已验证无效的方向", parents=[common])
    p_na.add_argument("--symptom", required=True)
    p_na.add_argument("--tried", required=True, help="试过的方向")
    p_na.add_argument("--evidence", required=True, help="证明它无效的实测证据")
    # --arch 必填：硬件结论与架构绑定，而「不填 = 对所有架构成立」是这张表
    # 上最贵的默认值 —— 一条在 DAV_2201 上测出来的死路会把 DAV_3510 上同一个
    # 方向一并封死，而那两颗芯片的 UB 容量和流水结构都不一样。
    p_na.add_argument("--arch", required=True, choices=sorted(ARCH_DIRS),
                      help="实测所在架构（必填：死路不跨架构成立）")
    p_na.add_argument("--operator", default="",
                      help="实测所在算子；留空表示这是算子无关的硬件结论")
    p_na.add_argument("--input-shape", default="",
                      help="实测所在输入形状，如 'B=8,S=4096,H=1024,fp16'；"
                           "留空表示与形状无关")
    p_na.add_argument("--retry-precondition", required=True,
                      help="什么变了才值得再试一次。这张表是重试前置条件表，"
                           "不是禁令表")
    p_na.set_defaults(func=cmd_nogo_add)

    p_nc = nogo_sub.add_parser("check", help="重试前置检查", parents=[common])
    p_nc.add_argument("--symptom", required=True)
    p_nc.add_argument("--arch", default=None, choices=sorted(ARCH_DIRS))
    p_nc.add_argument("--operator", default="",
                      help="本次的算子。给了才能把「同条件命中」和"
                           "「别的算子上试过」分开")
    p_nc.add_argument("--input-shape", default="", help="本次的输入形状")
    p_nc.set_defaults(func=cmd_nogo_check)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
