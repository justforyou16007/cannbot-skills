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
"""结构核验：声称的结构到底有没有被构建出来。

这个探针回答的问题和性能无关，而且**在任何性能数出来之前**就能答。

为什么需要它
------------
机制核验此前只能被**一个停顿比例**反驳。于是有两种错判，方向相反：

  假通过   一条 `SUPPLY_KEEPUP` 机制行锚定 `aiv_mte2_wait_ratio < 4.2`。
           实现写的其实是单 buffer 回退，但本轮因为别的改动（减小 tile、
           改了对齐）使 mte2_wait 恰好降到 4.0 且降幅够大 → 判「机制已核实」
           → 这个解释被封进产物库和知识库，而双缓冲从未存在。

  假失败   机制确实构建了，但当前 shape 下它不是瓶颈，停顿指标不动
           → 判「平凡通过」→ 一个正确的架构决策被当作「增益另有来源」。

论文实现用静态通道解决这件事，而且给了一条硬规则：**未落地的结构优先于
改验收措辞** —— 把措辞改宽不会让一个「不在」变成「在」。它记录的翻车模式是：

  > 设计判定矩阵乘的 A 操作数走寄存器源，而验收只查寄存器用量。于是一个
  > 「预留了寄存器预算、但操作数仍走共享内存往返」的退化实现通过了验收
  > 并被当成设计落地。预算留出来了，路径没建起来。

两层证据
--------
按代价从低到高，两层都查，各自独立给判定：

  源码层   候选自己的源码**完全可读** —— 封存 hook 只拦参考实现。
           `TQue<..., 2>` 的缓冲深度、`SetFlag/WaitFlag` 是否成对、
           L1 buffer 有没有真的分配，这些从源码直接可判，
           **不需要任何工具链，不需要硬件**。

  IR 层    `ccec -emit-llvm -c` + `llvm-dis` 把毕昇编译器停在 LLVM IR，
           而 IR 里**地址空间直接编码了操作数所在的存储**：
           `addrspace(1)` = GM，`addrspace(6)` = UB。
           所以「从 L1 直送」与「走 GM 往返」是二值可判的。

           边界：IR 在指令选择与寄存器分配**之前**，所以证不了
           「寄存器没溢出」「调度深度是多少」。那些要走仿真通道，
           见 ops/ops-simulator 的 structural-verification.md。

最终助记符反汇编在昇腾上不可用（厂商三处同时关闭：llvm-objdump 缺指令表、
设备侧 -S 被拒、编译管线无 .s 阶段），所以这里不追求它 —— 追求的是
「结构在不在」这个二值事实。

失败方向
--------
工具链不在、源码读不到、断言写错形状，一律**报错而不是放行**。
「没查」不等于「查过且通过」—— 一条查不了的结构断言如果被当成通过，
那它比没有这条断言更糟：它会让人以为核验过了。

用法:
    # 只查源码层（不需要 CANN 工具链）
    python3 structure_probe.py --candidate ./cand \
        --assert "buffer_depth:2" --assert "flag_paired" --json

    # 连 IR 层一起查
    python3 structure_probe.py --candidate ./cand \
        --assert "operand_space:addrspace(6)" \
        --arch dav-c220-vec --json

    # 列出支持的断言
    python3 structure_probe.py --list-assertions

退出码: 0 = 全部成立; 1 = 有断言不成立; 2 = 用法错误或查不了
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: 候选源码里要扫的后缀。
SOURCE_SUFFIXES = {".cpp", ".cc", ".cxx", ".h", ".hpp", ".cce"}

#: 单个文件的读取上限。
MAX_FILE_BYTES = 2 * 1024 * 1024

#: 不扫的目录。构建产物与第三方代码里的匹配说明不了候选自己的结构。
#:
#: 注意 third_party **不在**这个列表里：从被跳过的目录 import 一份转发实现
#: 是围栏漏过的那条路，结构核验不该重复同一个错。
SKIP_DIRS = {".git", "build", "build_out", "__pycache__", "cmake-build-debug"}

#: AscendC 的缓冲队列声明。深度是第二个模板参数。
RE_TQUE = re.compile(
    r"\bT(?:Que|Buf|QueBind)\s*<\s*(?P<args>[^<>]*(?:<[^<>]*>[^<>]*)*)>")

#: 同步原语。成对性是 SetFlag/WaitFlag 的核心结构事实。
RE_SET_FLAG = re.compile(r"\bSetFlag\s*<")
RE_WAIT_FLAG = re.compile(r"\bWaitFlag\s*<")

#: InitBuffer 的第二个实参是块数（double buffer 时为 2）。
RE_INIT_BUFFER = re.compile(
    r"\bInitBuffer\s*\(\s*(?P<queue>\w+)\s*,\s*(?P<blocks>\d+)")

#: IR 里的地址空间。
RE_ADDRSPACE = re.compile(r"addrspace\((?P<n>\d+)\)")

#: 地址空间 → 存储层级。来源见 ops-simulator 的 structural-verification.md。
ADDRSPACE_NAMES = {"1": "GM", "6": "UB"}


class StructureError(Exception):
    """结构核验无法进行。"""


# ---------------------------------------------------------------------------
# 读候选源码
# ---------------------------------------------------------------------------

def iter_sources(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SOURCE_SUFFIXES else []
    out: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        out.append(path)
    return out


def read_sources(root: Path) -> Tuple[str, List[Path]]:
    """把候选源码读成一整段文本，顺带返回读了哪些文件。

    拼成一段是因为结构断言问的是「整个候选里有没有」，而不是「某个文件里
    有没有」—— 一个 kernel 的缓冲声明和同步调用经常分在头文件与实现里。
    """
    files = iter_sources(root)
    if not files:
        raise StructureError(
            f"{root} 下没有可扫的源文件（后缀: {sorted(SOURCE_SUFFIXES)}）。\n"
            "  结构核验查的是候选**自己**的源码 —— 它完全可读，"
            "封存 hook 只拦参考实现。\n"
            "  若候选只有构建产物没有源码，这条通道用不了，"
            "要走 ops-simulator 的仿真指令表。")
    chunks: List[str] = []
    for path in files:
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks), files


def strip_comments(text: str) -> str:
    """去注释。注释里写「这里是 double buffer」不算证据。"""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


# ---------------------------------------------------------------------------
# 源码层断言
# ---------------------------------------------------------------------------

def check_buffer_depth(code: str, expected: int) -> Dict[str, Any]:
    """缓冲深度。double buffer 的结构事实就是「深度 ≥ 2」。

    两处证据：模板参数里的深度，以及 InitBuffer 的块数。两者都看，
    因为不同写法把深度放在不同位置。
    """
    template_depths: List[int] = []
    for match in RE_TQUE.finditer(code):
        args = [a.strip() for a in match.group("args").split(",")]
        for arg in args[1:]:
            if arg.isdigit():
                template_depths.append(int(arg))
                break
    init_blocks = [int(m.group("blocks")) for m in RE_INIT_BUFFER.finditer(code)]

    observed = template_depths + init_blocks
    if not observed:
        return {
            "assertion": f"buffer_depth:{expected}",
            "verdict": "unknown",
            "detail": ("源码里没找到缓冲深度的声明（TQue/TBuf 的模板参数，"
                       "或 InitBuffer 的块数）。"
                       "换了写法时这条查不了 —— 不要当成通过"),
            "observed": [],
        }
    peak = max(observed)
    return {
        "assertion": f"buffer_depth:{expected}",
        "verdict": "pass" if peak >= expected else "fail",
        "detail": (f"声明的缓冲深度最大为 {peak}"
                   + ("" if peak >= expected else
                      f"，低于要求的 {expected} —— "
                      "这就是「声明了多缓冲、实际是单缓冲回退」那种退化。"
                      "该修实现，不是把验收线改宽")),
        "observed": {"template_depths": template_depths,
                     "init_blocks": init_blocks},
    }


def check_flag_paired(code: str) -> Dict[str, Any]:
    """同步成对。少一侧意味着同步缺失 —— 可能这个 case 恰好不出错，
    但不是设计要的结构。"""
    sets = len(RE_SET_FLAG.findall(code))
    waits = len(RE_WAIT_FLAG.findall(code))
    if sets == 0 and waits == 0:
        return {
            "assertion": "flag_paired",
            "verdict": "fail",
            "detail": ("源码里没有任何 SetFlag/WaitFlag —— 声称有跨流水同步，"
                       "但一条同步原语都没有"),
            "observed": {"set_flag": 0, "wait_flag": 0},
        }
    return {
        "assertion": "flag_paired",
        "verdict": "pass" if sets == waits else "fail",
        "detail": (f"SetFlag {sets} 条、WaitFlag {waits} 条"
                   + ("，成对" if sets == waits else
                      "，**不成对** —— 少的那一侧说明同步链断了。"
                      "注意条数相等只是必要条件，配对关系要人工过目")),
        "observed": {"set_flag": sets, "wait_flag": waits},
    }


def check_symbol(code: str, symbol: str, want: bool) -> Dict[str, Any]:
    """某个构造在不在。用于「声称用了 X」这类断言。"""
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(symbol)
                         + r"(?![A-Za-z0-9_])")
    hits = len(pattern.findall(code))
    present = hits > 0
    label = f"{'has' if want else 'absent'}:{symbol}"
    return {
        "assertion": label,
        "verdict": "pass" if present == want else "fail",
        "detail": (f"{symbol} 出现 {hits} 次"
                   + ("" if present == want else
                      f"，而断言要求它{'出现' if want else '不出现'}")),
        "observed": {"hits": hits},
    }


# ---------------------------------------------------------------------------
# IR 层断言
# ---------------------------------------------------------------------------

def emit_ir(source: Path, arch: str, include_dirs: List[str],
            work_dir: Path) -> Tuple[Optional[str], str]:
    """把一个源文件编到 LLVM IR 并反汇编成可读文本。

    返回 (IR 文本, 说明)。拿不到就返回 (None, 原因)。
    """
    ccec = shutil.which("ccec")
    llvm_dis = shutil.which("llvm-dis")
    if not ccec:
        return None, ("PATH 里没有 ccec（毕昇编译器）。"
                      "IR 层查不了 —— source CANN 的 set_env.sh 之后重试，"
                      "或只跑源码层断言")
    if not llvm_dis:
        return None, "PATH 里没有 llvm-dis，无法把 .bc 反汇编成可读 IR"

    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [ccec, f"--cce-aicore-arch={arch}", "-xcce", "-std=c++17",
           "-emit-llvm", "-c", str(source)]
    for item in include_dirs:
        cmd += ["-I", item]
    try:
        proc = subprocess.run(cmd, cwd=str(work_dir), capture_output=True,
                              text=True, timeout=600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"执行 ccec 失败: {exc}"
    if proc.returncode != 0:
        return None, (f"ccec 返回 {proc.returncode}。"
                      f"stderr: {(proc.stderr or '').strip()[:400]}")

    # ccec 不带 -o 时是多产物，device 侧那份 .bc 名字随架构变化，这里全找。
    bitcode = sorted(work_dir.glob("*.bc"))
    if not bitcode:
        return None, ("ccec 成功了但没有产出 .bc。"
                      "产物名随 CANN 版本与架构变化，"
                      f"工作目录里现有: {[p.name for p in work_dir.iterdir()][:10]}")
    texts: List[str] = []
    for item in bitcode:
        try:
            proc = subprocess.run([llvm_dis, str(item), "-o", "-"],
                                  capture_output=True, text=True,
                                  timeout=300, check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0 and proc.stdout:
            texts.append(proc.stdout)
    if not texts:
        return None, "llvm-dis 没能从 .bc 反汇编出 IR"
    return "\n".join(texts), f"从 {len(bitcode)} 份 .bc 取到 IR"


def check_operand_space(ir_text: str, expected: str) -> Dict[str, Any]:
    """操作数所在的存储层级。

    这是上游那条翻车模式的直接判据：IR 的地址空间编码了操作数在哪一级
    存储，所以「预留了寄存器预算但操作数仍走 GM 往返」在这里是二值可判的。
    """
    found = sorted({m.group("n") for m in RE_ADDRSPACE.finditer(ir_text)})
    want = RE_ADDRSPACE.search(expected)
    if not want:
        return {
            "assertion": f"operand_space:{expected}",
            "verdict": "unknown",
            "detail": (f"断言 {expected!r} 不是 `addrspace(N)` 的形状。"
                       f"已知: {ADDRSPACE_NAMES}"),
            "observed": found,
        }
    target = want.group("n")
    present = target in found
    named = [f"{n}({ADDRSPACE_NAMES.get(n, '?')})" for n in found]
    return {
        "assertion": f"operand_space:addrspace({target})",
        "verdict": "pass" if present else "fail",
        "detail": (f"IR 里出现的地址空间: {named}"
                   + ("" if present else
                      f"，不含 addrspace({target})"
                      f"（{ADDRSPACE_NAMES.get(target, '?')}）—— "
                      "声称的存储层级没有被用上")),
        "observed": found,
    }


# ---------------------------------------------------------------------------
# 断言解析与执行
# ---------------------------------------------------------------------------

ASSERTION_HELP = {
    "buffer_depth:N": "缓冲深度 ≥ N（TQue/TBuf 模板参数或 InitBuffer 块数）",
    "flag_paired": "SetFlag 与 WaitFlag 条数相等且都非零",
    "has:SYMBOL": "候选源码里出现 SYMBOL（带词边界）",
    "absent:SYMBOL": "候选源码里不出现 SYMBOL",
    "operand_space:addrspace(N)": "IR 里出现地址空间 N（1=GM、6=UB）— 需要 ccec",
}


def needs_ir(assertion: str) -> bool:
    return assertion.startswith("operand_space:")


def run_source_assertion(code: str, assertion: str) -> Dict[str, Any]:
    if assertion == "flag_paired":
        return check_flag_paired(code)
    if assertion.startswith("buffer_depth:"):
        raw = assertion.split(":", 1)[1]
        if not raw.isdigit():
            raise StructureError(f"buffer_depth 要一个整数，收到 {raw!r}")
        return check_buffer_depth(code, int(raw))
    if assertion.startswith("has:"):
        return check_symbol(code, assertion.split(":", 1)[1], want=True)
    if assertion.startswith("absent:"):
        return check_symbol(code, assertion.split(":", 1)[1], want=False)
    raise StructureError(
        f"不认识的断言 {assertion!r}。支持的形状:\n  "
        + "\n  ".join(f"{k} — {v}" for k, v in ASSERTION_HELP.items()))


def probe(candidate: Path, assertions: List[str], arch: str,
          include_dirs: List[str], work_dir: Path,
          allow_missing_toolchain: bool = False) -> Dict[str, Any]:
    code_raw, files = read_sources(candidate)
    code = strip_comments(code_raw)

    results: List[Dict[str, Any]] = []
    ir_text: Optional[str] = None
    ir_note = ""

    if any(needs_ir(a) for a in assertions):
        # 挑一个源文件编 IR。多文件工程下这是个近似 —— 真要全覆盖应当
        # 对每个 kernel 源各编一次，这里先用最大那个（通常是 kernel 本体）。
        target = max(files, key=lambda p: p.stat().st_size)
        ir_text, ir_note = emit_ir(target, arch, include_dirs, work_dir)
        if ir_text is None and not allow_missing_toolchain:
            raise StructureError(
                f"IR 层断言需要工具链，但拿不到 IR：{ir_note}\n"
                "  「查不了」不等于「通过」—— 一条查不了的结构断言被当成通过，"
                "比没有这条断言更糟：\n"
                "  它会让人以为核验过了。\n"
                "  只想跑源码层时把 operand_space 断言去掉，"
                "或加 --allow-missing-toolchain 显式接受这个缺口。")

    for assertion in assertions:
        if needs_ir(assertion):
            if ir_text is None:
                results.append({
                    "assertion": assertion, "verdict": "unknown",
                    "detail": f"拿不到 IR：{ir_note}", "observed": None,
                })
                continue
            expected = assertion.split(":", 1)[1]
            results.append(check_operand_space(ir_text, expected))
        else:
            results.append(run_source_assertion(code, assertion))

    failed = [r for r in results if r["verdict"] == "fail"]
    unknown = [r for r in results if r["verdict"] == "unknown"]
    return {
        "schema_version": "1.0",
        "candidate": str(candidate),
        "files_scanned": len(files),
        "ir": {"available": ir_text is not None, "note": ir_note or None,
               "arch": arch},
        "assertions": results,
        # 「有不成立的」与「有查不了的」要分开：前者是实现没做到，
        # 后者是我们没查成 —— 两者该做的事完全不同。
        "passed": not failed and not unknown,
        "failed_count": len(failed),
        "unknown_count": len(unknown),
    }


def render(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"候选      {report['candidate']}")
    lines.append(f"源文件    {report['files_scanned']} 个")
    ir = report["ir"]
    lines.append("IR 层     " + ("可用" if ir["available"]
                                 else f"不可用（{ir['note']}）"))
    lines.append("")
    marks = {"pass": "成立", "fail": "不成立", "unknown": "查不了"}
    for item in report["assertions"]:
        lines.append(f"  [{marks[item['verdict']]}] {item['assertion']}")
        lines.append(f"          {item['detail']}")
    lines.append("")
    if report["passed"]:
        lines.append("全部结构断言成立。")
    else:
        if report["failed_count"]:
            lines.append(f"{report['failed_count']} 条断言不成立 —— "
                         "声称的结构没有被构建出来。")
            lines.append("这是实现问题：该修实现（或给实现补一张经验卡片），")
            lines.append("**不是**把验收线改宽 —— 改措辞不会让一个「不在」"
                         "变成「在」。")
        if report["unknown_count"]:
            lines.append(f"{report['unknown_count']} 条断言查不了。"
                         "这不等于通过，先把通道补上。")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行。独立出来是为了让文档命令门禁能拿到它。"""
    parser = argparse.ArgumentParser(
        description="结构核验：声称的结构到底有没有被构建出来")
    parser.add_argument("--candidate", type=Path,
                        help="候选目录或单个源文件")
    parser.add_argument("--assert", dest="assertions", action="append",
                        default=[], metavar="ASSERTION",
                        help="结构断言，可重复。用 --list-assertions 看支持的形状")
    parser.add_argument("--arch", default="dav-c220-vec",
                        help="IR 层用的 --cce-aicore-arch 值")
    parser.add_argument("-I", dest="includes", action="append", default=[],
                        metavar="DIR", help="传给 ccec 的头文件目录，可重复")
    parser.add_argument("--work-dir", type=Path, default=Path("./.structure_probe"),
                        help="IR 中间产物的落盘目录")
    parser.add_argument("--allow-missing-toolchain", action="store_true",
                        help="拿不到 IR 时仍然输出报告（相关断言标为查不了）。"
                             "「查不了」不等于「通过」")
    parser.add_argument("--list-assertions", action="store_true",
                        help="列出支持的断言形状")
    parser.add_argument("--output", type=Path, default=None,
                        help="把报告写到文件，供 campaign.py record 引用")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.list_assertions:
        print("支持的结构断言：\n")
        for shape, note in ASSERTION_HELP.items():
            print(f"  {shape:34} {note}")
        print("\n地址空间对照：" + ", ".join(
            f"{n}={name}" for n, name in ADDRSPACE_NAMES.items()))
        return 0

    if args.candidate is None or not args.assertions:
        print("错误: 需要 --candidate 与至少一个 --assert。"
              "用 --list-assertions 看支持的形状", file=sys.stderr)
        return 2

    try:
        report = probe(args.candidate, args.assertions, args.arch,
                       args.includes, args.work_dir,
                       args.allow_missing_toolchain)
    except StructureError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.output:
        try:
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        except OSError as exc:
            print(f"错误: 无法写入 {args.output}: {exc}", file=sys.stderr)
            return 2

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
