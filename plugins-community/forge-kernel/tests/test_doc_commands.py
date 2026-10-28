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
"""文档里的命令必须能跑。

这个测试的由来：`--state` 曾经只挂在顶层 parser 上，于是 argparse 要求它
写在子命令**之前**，而全部 15 处文档（包括程序自己打印的两条恢复提示）
都写在后面。照抄执行一律报 `unrecognized arguments`。

这类 bug 的代价不在于报错本身，而在于**谁来读这个报错**。coding agent
拿到它往往不会去调参数位置，而是判定「这个脚本坏了，我自己判断算了」——
可整套设计的要害恰恰是不让 agent 自己裁决。一个参数位置的问题能把裁决
回路整个旁路掉。

所以这里把「文档与 --help 不漂移」变成常驻门禁：从 SKILL.md、quickstart、
agent 定义、以及脚本自身的 docstring 里抠出所有命令行示例，拿真实的
parser 去解析一遍。

运行: python3 -m pytest plugins-community/forge-kernel/tests/ -v
"""

import contextlib
import io
import re
import shlex
import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PLUGIN_ROOT / "skills" / "forge-campaign" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import campaign  # noqa: E402

#: 需要数值的参数：占位符要换成数字，换成字符串会假性失败。
NUMERIC_OPTIONS = {
    "--value", "--noise-floor", "--target-gain", "--budget",
    "--max-records", "--stagnation-threshold", "--phase-target-gain",
}

#: 省略号表示「其余参数同上」，不是完整命令。
ELLIPSES = {"...", "…"}

RE_INVOCATION = re.compile(r"campaign\.py\s+(.+)")


def _join_continuations(text: str) -> str:
    """把 shell 续行接起来。

    脚本 docstring 里的反斜杠是转义过的（`\\\\`），文档里是单个，两种都要认。
    """
    text = re.sub(r"\\\\\s*\n\s*", " ", text)
    return re.sub(r"\\\s*\n\s*", " ", text)


def _looks_like_prose(argline: str) -> bool:
    """排除散文。

    判据是「引号外出现中文」。流程图里的 `campaign.py reopen 按依赖图作废
    下游` 是说明，不是命令；而 `--reason "拓扑决策有缺陷"` 里的中文在引号内，
    是正当的参数值。只看首个词不够 —— 散文的首词往往正好是子命令名。
    """
    stripped = re.sub(r'"[^"]*"|\'[^\']*\'', "", argline)   # 引号内的中文是参数值
    stripped = re.sub(r"<[^>]*>", "", stripped)              # <实测值> 这类占位符同理
    return bool(re.search(r"[\u4e00-\u9fff（）：]", stripped))


def _fenced_blocks(text: str):
    """只取围栏代码块里的内容。

    区分标准是意图：```bash 块里的命令是给人照抄执行的，句子里的
    `campaign.py ledger` 只是在指代这条命令。把后者也拿去解析会制造
    大量假失败，而假失败会让人关掉这个测试 —— 那就白做了。
    """
    inside = False
    buf = []
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            if inside:
                yield "\n".join(buf)
                buf = []
            inside = not inside
            continue
        if inside:
            buf.append(line)
    if inside and buf:
        yield "\n".join(buf)


def _docstring_usage(text: str):
    """脚本模块 docstring 里的用法块。

    出事的两条恢复提示就在脚本里，所以这里必须覆盖到。只取缩进的行 ——
    那是用法块的形态，散文不会缩进。
    """
    match = re.match(r'\A\s*(?:#[^\n]*\n)*\s*"""(.*?)"""', text, re.DOTALL)
    if not match:
        return
    for line in match.group(1).splitlines():
        if line.startswith(("    ", "\t")) and "campaign.py" in line:
            yield line


def _iter_candidate_lines():
    """产出 (来源, 单行命令文本)。"""
    for path in sorted(PLUGIN_ROOT.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for block in _fenced_blocks(text):
            for line in _join_continuations(block).splitlines():
                if "campaign.py" in line:
                    yield path, line

    for path in sorted(SCRIPTS.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        joined = _join_continuations(text)
        for line in _docstring_usage(joined):
            yield path, line
        # 程序运行时打印给用户的恢复提示 —— 照抄必须能跑。
        # 从模块 docstring 之后开始找，否则会把 docstring 里的多行用法
        # 在引号处截断，产出一条残缺的重复条目。
        body = joined
        end = re.match(r'\A\s*(?:#[^\n]*\n)*\s*""".*?"""', joined, re.DOTALL)
        if end:
            body = joined[end.end():]
        for match in re.finditer(
                r'"\s*(campaign\.py\s+[a-z-]+[^"\\]*)', body):
            yield path, match.group(1)


def extract_commands():
    """抠出所有 campaign.py 调用，产出 (来源, 原文, argv)。"""
    out = []
    seen = set()
    for path, raw_line in _iter_candidate_lines():
        match = RE_INVOCATION.search(raw_line)
        if not match:
            continue
        # campaign.py 之前出现中文，说明它嵌在流程描述里（如
        # 「12. 代际提升后  重测噪声底 → campaign.py set-noise-floor」），
        # 是对命令的指代而非可照抄的命令本身。
        if re.search(r"[\u4e00-\u9fff]", raw_line[: match.start()]):
            continue
        argline = match.group(1)
        argline = argline.split("#", 1)[0].strip().rstrip("`\"')")
        if not argline or _looks_like_prose(argline):
            continue
        try:
            tokens = shlex.split(argline)
        except ValueError:
            continue
        if not tokens or tokens[0].startswith("-"):
            continue
        if any(t in ELLIPSES for t in tokens):
            continue

        # 占位符按前一个选项的类型替换
        argv = []
        for idx, token in enumerate(tokens):
            if token.startswith("<") or token.startswith("{"):
                previous = tokens[idx - 1] if idx else ""
                token = "1.0" if previous in NUMERIC_OPTIONS else "placeholder"
            argv.append(token)

        key = (str(path), tuple(argv))
        if key in seen:
            continue
        seen.add(key)
        out.append((path, argline, argv))
    return out


DOC_COMMANDS = extract_commands()


def test_documented_commands_were_found():
    """抽取器本身要有效 —— 一条都没抠到说明正则失配，测试会假性通过。"""
    assert len(DOC_COMMANDS) >= 10, (
        f"只抠到 {len(DOC_COMMANDS)} 条命令，抽取规则可能失配")


@pytest.mark.parametrize(
    "source,argline,argv",
    DOC_COMMANDS,
    ids=[f"{p.name}:{a[:40]}" for p, a, _ in DOC_COMMANDS],
)
def test_documented_command_parses(source, argline, argv):
    """文档里写的命令，拿真实 parser 解析必须通过。"""
    parser = campaign.build_parser()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code not in (0, None):
            message = buf.getvalue().strip().splitlines()
            detail = message[-1] if message else "（无输出）"
            pytest.fail(
                f"{source.relative_to(PLUGIN_ROOT)} 里的命令解析失败\n"
                f"  原文: campaign.py {argline}\n"
                f"  报错: {detail}\n"
                f"  照抄这条命令的人（或 agent）会直接撞上这个错误")


def test_state_works_in_both_positions():
    """--state 写在子命令前后都要能跑。

    只支持一种位置时，写错的人拿到的是 'unrecognized arguments'，
    它不说该放哪里 —— 这正是 agent 会开始自己猜的地方。
    """
    parser = campaign.build_parser()
    before = parser.parse_args(["--state", "a.json", "status"])
    after = parser.parse_args(["status", "--state", "a.json"])
    assert before.state == "a.json"
    assert after.state == "a.json"
