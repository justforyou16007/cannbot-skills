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
"""转发围栏：拒绝「其实是在调参考实现」的候选。

论文 Appendix G 把这道围栏放在**任何计时之前**：拒绝任何调用或转发到
参考 kernel 的候选。

为什么必须有它
--------------
没有围栏，整个度量体系是可以被绕过的，而且绕过的方式看起来完全正常：

  候选直接调厂商的 aclnn 实现 → 加速比 1.0，精度满分，所有门禁都过。
  看起来是「追平了参考」，实际一行 kernel 都没写。

自演进场景下更隐蔽：第 n 代只要转发给第 n-1 代，就能稳定产出「与上代
持平」的结果，链条可以这样空转很多轮，每轮都「没有退化」。

难点：不能误伤
--------------
候选**必须**定义一个同名算子 —— 那是它的本职工作。所以不能简单地禁用
算子名。要区分的是两件事：

  定义它    写 kernel、注册 op、提供自己的入口      ← 正当
  转发给它  调用厂商 aclnn 接口、调框架的同名实现    ← 越界

昇腾上这个区分是清晰的：aclnn 的两段式调用（aclnnXxxGetWorkspaceSize
加 aclnnXxx）是调用厂商实现的标志，候选自己的 kernel 不会长这样。

误伤的代价比漏判更高：一次误伤会让人去关掉围栏，关掉之后真正的转发
就无人把守了。所以每条规则都给出 file:line 证据，让人能当场判断，
并且支持按规则豁免。

用法:
    python3 fence_check.py --candidate ./my_kernel --op-name add_layer_norm
    python3 fence_check.py --candidate ./my_kernel --spec ./forge_work/reference_spec.json
    python3 fence_check.py --candidate ./my_kernel --op-name add_layer_norm \\
        --denylist ./forge_work/denylist.txt --json

退出码: 0 = 通过，可以计时; 1 = 发现转发，拒绝计时; 2 = 用法或 IO 错误
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: 要扫描的文件后缀。候选的实现与它的 Python 入口都要看。
SCAN_SUFFIXES = {
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx",
    ".py", ".cmake", ".txt", ".json", ".sh",
}

#: 单文件扫描上限，避免被超大文件拖住。
MAX_FILE_BYTES = 4 * 1024 * 1024

#: 跳过的目录：构建产物与第三方代码不是候选的实现。
SKIP_DIRS = {
    "build", "build_out", "output", "__pycache__", ".git",
    "third_party", "node_modules", ".pytest_cache", "dist",
}


class FenceError(Exception):
    """围栏无法完成检查。"""


class Finding:
    """一处越界。file:line 让人能当场判断是不是误伤。"""

    def __init__(self, rule: str, path: Path, line_no: int,
                 line: str, reason: str) -> None:
        self.rule = rule
        self.path = path
        self.line_no = line_no
        self.line = line.strip()[:200]
        self.reason = reason

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule, "file": str(self.path), "line": self.line_no,
            "text": self.line, "reason": self.reason,
        }

    def __str__(self) -> str:
        return f"[{self.rule}] {self.path}:{self.line_no}\n    {self.line}\n    {self.reason}"


# ---------------------------------------------------------------------------
# 算子名的各种书写形态
# ---------------------------------------------------------------------------

def name_variants(op_name: str) -> Dict[str, str]:
    """同一个算子在不同层面的写法。

    add_layer_norm 在 aclnn 接口上是 AddLayerNorm，在 torch_npu 上可能是
    npu_add_layer_norm。围栏要认全这些形态，否则换个写法就绕过去了。
    """
    snake = re.sub(r"[-\s]+", "_", op_name.strip()).lower()
    camel = "".join(part.capitalize() for part in snake.split("_") if part)
    return {"snake": snake, "camel": camel, "raw": op_name.strip()}


def build_rules(op_name: str,
                reference_paths: List[str]) -> List[Tuple[str, re.Pattern, str]]:
    """构造围栏规则。每条都给出「为什么这是转发而不是实现」。"""
    names = name_variants(op_name)
    camel = re.escape(names["camel"])
    snake = re.escape(names["snake"])

    rules: List[Tuple[str, re.Pattern, str]] = [
        (
            "F-ACLNN",
            re.compile(rf"\baclnn{camel}(?:GetWorkspaceSize)?\s*\("),
            ("调用了厂商 aclnn 接口。aclnn 的两段式调用是调用 CANN 算子库"
             "内置实现的标志 —— 这是转发，不是自己实现"),
        ),
        (
            "F-TORCH-NPU",
            re.compile(rf"\btorch_npu\s*\.\s*npu_{snake}\b"),
            "调用了 torch_npu 提供的同名算子，它会派发到厂商实现",
        ),
        (
            "F-TORCH-FUNC",
            re.compile(rf"\btorch\s*\.\s*(?:nn\s*\.\s*functional\s*\.\s*)?{snake}\s*\("),
            "调用了框架的同名实现，它最终派发到厂商 kernel",
        ),
        (
            "F-ATEN",
            re.compile(rf"\btorch\s*\.\s*ops\s*\.\s*aten\s*\.\s*{snake}\b"),
            "调用了 aten 的同名算子，等同于使用框架内置实现",
        ),
        (
            "F-OPP-SWITCH",
            # 只抓「写」不抓「读」：os.environ.get(...) 做日志是正当的，
            # 真正危险的是候选自己改派发路径。覆盖三种写法：
            #   os.environ['X'] = ...   变量名后隔着引号与方括号才到等号
            #   export X=...            变量名后直接是等号
            #   setenv("X", ...)        没有等号，单独匹配
            re.compile(
                r"ASCEND_CUSTOM_OPP_PATH[\"'\]\s]{0,4}=(?!=)"
                r"|(?:setenv|putenv)\s*\([^)]*ASCEND_CUSTOM_OPP_PATH"
            ),
            ("在候选代码里改写 ASCEND_CUSTOM_OPP_PATH。派发路径应由评测"
             "驱动统一控制；候选自己改它可以把调用导向参考实现"),
        ),
    ]

    for raw in reference_paths:
        cleaned = raw.strip().rstrip("/")
        if not cleaned:
            continue
        rules.append((
            "F-REF-PATH",
            re.compile(re.escape(cleaned)),
            f"引用了封存的参考路径 {cleaned}；候选不应依赖参考的任何产物",
        ))

    return rules


# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------

def iter_source_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SCAN_SUFFIXES:
            yield path


def strip_comment(line: str, suffix: str) -> str:
    """去掉行内注释。

    注释里提到参考的名字是正当的 —— 解释「我们没有用 aclnnXxx，因为……」
    不该被判越界。只看可执行的部分。
    """
    if suffix == ".py":
        marker = line.find("#")
        return line[:marker] if marker >= 0 else line
    if suffix in (".sh", ".cmake", ".txt"):
        marker = line.find("#")
        return line[:marker] if marker >= 0 else line
    marker = line.find("//")
    return line[:marker] if marker >= 0 else line


def scan_file(path: Path,
              rules: List[Tuple[str, re.Pattern, str]],
              denylist: List[str]) -> List[Finding]:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return []
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    findings: List[Finding] = []
    suffix = path.suffix.lower()
    in_block_comment = False

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw

        # C 系列的块注释：同样属于「说明」而非「执行」
        if suffix not in (".py", ".sh"):
            if in_block_comment:
                end = line.find("*/")
                if end < 0:
                    continue
                line = line[end + 2:]
                in_block_comment = False
            start = line.find("/*")
            if start >= 0:
                end = line.find("*/", start + 2)
                if end < 0:
                    in_block_comment = True
                    line = line[:start]
                else:
                    line = line[:start] + line[end + 2:]

        code = strip_comment(line, suffix)
        if not code.strip():
            continue

        for rule, pattern, reason in rules:
            if pattern.search(code):
                findings.append(Finding(rule, path, line_no, raw, reason))

        for token in denylist:
            if token and token in code:
                findings.append(Finding(
                    "F-DENY", path, line_no, raw,
                    f"出现参考实现的专有符号 {token}；候选不应引用参考的内部命名",
                ))

    return findings


def load_denylist(path: Optional[Path]) -> List[str]:
    if path is None or not path.exists():
        return []
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            out.append(token)
    return out


def load_reference_paths(spec_path: Optional[Path]) -> Tuple[Optional[str], List[str]]:
    """从 reference_spec.json 取算子名与参考路径。"""
    if spec_path is None:
        return None, []
    if not spec_path.exists():
        raise FenceError(
            f"找不到 {spec_path}。\n"
            "reference_spec.json 由参考适配步骤生成，先跑：\n"
            "  python3 <forge-campaign>/scripts/reference_adapter.py prepare \\\n"
            "      --kind builtin_aclnn --op-name <算子> --work-dir <工作目录>\n"
            "或者不用 --spec，直接给 --op-name。"
        )
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FenceError(f"无法读取 {spec_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FenceError(f"{spec_path} 不是合法 JSON: {exc}") from exc

    paths: List[str] = []
    for key in ("baseline_path",):
        value = spec.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())
    return spec.get("op_name"), paths


def check(candidate: Path, op_name: str, reference_paths: List[str],
          denylist: List[str],
          exempt: Iterable[str] = ()) -> Dict[str, Any]:
    if not candidate.exists():
        raise FenceError(f"候选路径不存在: {candidate}")

    rules = build_rules(op_name, reference_paths)
    exempt_set = {r.strip().upper() for r in exempt if r.strip()}

    findings: List[Finding] = []
    scanned = 0
    for path in iter_source_files(candidate):
        scanned += 1
        findings.extend(scan_file(path, rules, denylist))

    kept = [f for f in findings if f.rule not in exempt_set]
    waived = [f for f in findings if f.rule in exempt_set]

    if scanned == 0:
        raise FenceError(
            f"{candidate} 下没有可扫描的源文件。"
            f"支持的后缀: {sorted(SCAN_SUFFIXES)}"
        )

    return {
        "passed": not kept,
        "op_name": op_name,
        "candidate": str(candidate),
        "files_scanned": scanned,
        "findings": [f.as_dict() for f in kept],
        "waived": [f.as_dict() for f in waived],
        "exempted_rules": sorted(exempt_set),
        "_objects": kept,
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="转发围栏：在计时之前拒绝调用或转发到参考实现的候选",
    )
    parser.add_argument("--candidate", required=True, type=Path,
                        help="候选实现的目录或单个文件")
    parser.add_argument("--op-name", default=None,
                        help="算子名；不给则从 --spec 读")
    parser.add_argument("--spec", type=Path, default=None,
                        help="reference_spec.json，用于取算子名与参考路径")
    parser.add_argument("--denylist", type=Path, default=None,
                        help="参考专有名词清单（seal_reference.py 生成）")
    parser.add_argument("--exempt", default="",
                        help="豁免的规则号，逗号分隔，如 F-TORCH-FUNC")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    try:
        spec_op, reference_paths = load_reference_paths(args.spec)
        op_name = args.op_name or spec_op
        if not op_name:
            print("错误: 需要 --op-name 或含 op_name 的 --spec", file=sys.stderr)
            return 2
        result = check(
            candidate=args.candidate, op_name=op_name,
            reference_paths=reference_paths,
            denylist=load_denylist(args.denylist),
            exempt=args.exempt.split(","),
        )
    except FenceError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    objects: List[Finding] = result.pop("_objects")

    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1

    if result["passed"]:
        print(f"围栏通过：扫描 {result['files_scanned']} 个文件，未发现转发到参考实现。")
        if result["waived"]:
            print(f"（{len(result['waived'])} 处因豁免规则 "
                  f"{result['exempted_rules']} 未计入，请确认豁免是必要的）")
        return 0

    print(f"围栏拒绝：发现 {len(objects)} 处疑似转发到参考实现\n", file=sys.stderr)
    for finding in objects:
        print(f"{finding}\n", file=sys.stderr)
    print(
        "这些调用会让加速比失去意义：候选若转发给参考，测出来的就是参考自己\n"
        "的性能，而不是候选的。自演进场景下还会让链条空转 —— 每代都「与上代\n"
        "持平」，但一行新代码也没有。\n"
        "\n"
        "若确属误判（例如这是候选自己的算子注册而非调用厂商实现），\n"
        "可用 --exempt 豁免对应规则，但请在 milestone 里记录豁免理由。",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
