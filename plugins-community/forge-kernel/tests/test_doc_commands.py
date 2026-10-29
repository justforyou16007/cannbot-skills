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

import argparse
import contextlib
import io
import re
import shlex
import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PLUGIN_ROOT / "skills" / "forge-campaign" / "scripts"

#: 所有带 build_parser() 的脚本都要进门禁。
#:
#: 只盯一个脚本的话，另外几个的文档示例照样会漂 —— 而漂移的代价不是
#: 报错本身，是 agent 拿到 "unrecognized arguments" 之后会判定脚本坏了、
#: 转而自己判断，把裁决回路整个旁路掉。
SCRIPT_DIRS = [
    PLUGIN_ROOT / "skills" / "forge-campaign" / "scripts",
    PLUGIN_ROOT / "skills" / "forge-hwkb" / "scripts",
    PLUGIN_ROOT / "skills" / "forge-probes" / "scripts",
    PLUGIN_ROOT / "skills" / "forge-milestone" / "scripts",
]
for _directory in SCRIPT_DIRS:
    sys.path.insert(0, str(_directory))

import campaign  # noqa: E402
import fence_check  # noqa: E402
import hwkb  # noqa: E402
import lever_probe  # noqa: E402
import milestone_validate  # noqa: E402
import noise_floor  # noqa: E402
import reference_adapter  # noqa: E402
import reference_library  # noqa: E402
import seal_reference  # noqa: E402
import signature_probe  # noqa: E402

#: 脚本名 → 它的 parser 工厂。
PARSERS = {
    "campaign.py": campaign.build_parser,
    "reference_library.py": reference_library.build_parser,
    "reference_adapter.py": reference_adapter.build_parser,
    "seal_reference.py": seal_reference.build_parser,
    "hwkb.py": hwkb.build_parser,
    "fence_check.py": fence_check.build_parser,
    "lever_probe.py": lever_probe.build_parser,
    "signature_probe.py": signature_probe.build_parser,
    "noise_floor.py": noise_floor.build_parser,
    "milestone_validate.py": milestone_validate.build_parser,
}

#: 需要数值的参数：占位符要换成数字，换成字符串会假性失败。
#: 清单不手维护 —— 见下。
def _numeric_options() -> set:
    """从 parser 里推出哪些选项收数值，不手维护清单。

    手维护过一版，漏了 8 个（--baseline-us / --time-us / --gain 等），
    于是那些示例里的占位符被换成 "placeholder" 再喂给 type=float，
    解析必然失败 —— 而失败的是门禁自己，不是文档。这类假失败最后的
    结局是有人把门禁关掉，那就全白做了。

    parser 自己知道每个选项收什么类型，问它即可。
    """
    found = set()

    def walk(parser):
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for sub in action.choices.values():
                    walk(sub)
            elif action.type in (float, int):
                found.update(action.option_strings)

    for factory in PARSERS.values():
        walk(factory())
    return found


NUMERIC_OPTIONS = _numeric_options()

#: 省略号表示「其余参数同上」，不是完整命令。
ELLIPSES = {"...", "…"}

RE_INVOCATION = re.compile(
    r"(" + "|".join(re.escape(name) for name in PARSERS) + r")\s+(.+)")


def _mentions_a_gated_script(line: str) -> bool:
    return any(name in line for name in PARSERS)


def _has_subcommands(script: str) -> bool:
    """这个脚本是不是子命令式的。

    子命令式的脚本（campaign.py 等）第一个词必须是子命令名；扁平的
    （fence_check.py、noise_floor.py 等）第一个词就是选项。拿前者的
    规则去筛后者，会把后者的全部示例当成「不是命令」悄悄跳过 ——
    于是它们零覆盖，而门禁看起来一切正常。
    """
    parser = PARSERS[script]()
    return any(isinstance(a, argparse._SubParsersAction)  # noqa: SLF001
               for a in parser._actions)                  # noqa: SLF001


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
        if line.startswith(("    ", "\t")) and _mentions_a_gated_script(line):
            yield line


def _iter_candidate_lines():
    """产出 (来源, 单行命令文本)。"""
    for path in sorted(PLUGIN_ROOT.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for block in _fenced_blocks(text):
            for line in _join_continuations(block).splitlines():
                if _mentions_a_gated_script(line):
                    yield path, line

    for directory in SCRIPT_DIRS:
        for path in sorted(directory.glob("*.py")):
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
        names = "|".join(re.escape(n) for n in PARSERS)
        for match in re.finditer(
                r'"\s*((?:' + names + r')\s+[a-z-]+[^"\\]*)', body):
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
        script = match.group(1)
        argline = match.group(2)
        argline = argline.split("#", 1)[0].strip().rstrip("`\"')")
        if not argline or _looks_like_prose(argline):
            continue
        try:
            tokens = shlex.split(argline)
        except ValueError:
            continue
        if not tokens:
            continue
        if _has_subcommands(script) and tokens[0].startswith("-"):
            continue   # 子命令式脚本的首词必须是子命令名
        if any(t in ELLIPSES for t in tokens):
            continue

        # 占位符按前一个选项的类型替换
        argv = []
        for idx, token in enumerate(tokens):
            if token.startswith("<") or token.startswith("{"):
                previous = tokens[idx - 1] if idx else ""
                token = "1.0" if previous in NUMERIC_OPTIONS else "placeholder"
            argv.append(token)

        key = (str(path), script, tuple(argv))
        if key in seen:
            continue
        seen.add(key)
        out.append((path, script, argline, argv))
    return out


DOC_COMMANDS = extract_commands()


def test_documented_commands_were_found():
    """抽取器本身要有效 —— 一条都没抠到说明正则失配，测试会假性通过。"""
    assert len(DOC_COMMANDS) >= 10, (
        f"只抠到 {len(DOC_COMMANDS)} 条命令，抽取规则可能失配")


@pytest.mark.parametrize(
    "source,script,argline,argv",
    DOC_COMMANDS,
    ids=[f"{p.name}:{s}:{a[:30]}" for p, s, a, _ in DOC_COMMANDS],
)
def test_documented_command_parses(source, script, argline, argv):
    """文档里写的命令，拿真实 parser 解析必须通过。"""
    parser = PARSERS[script]()
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
                f"  原文: {script} {argline}\n"
                f"  报错: {detail}\n"
                f"  照抄这条命令的人（或 agent）会直接撞上这个错误")


#: 脚本 → (全局参数, 一个无必填项的子命令路径)
GLOBAL_OPTIONS = [
    ("campaign.py", "--state", ["status"], "state"),
    ("reference_library.py", "--library", ["list"], "library"),
    ("hwkb.py", "--kb-root", ["nogo", "check", "--symptom", "x"], "kb_root"),
]


@pytest.mark.parametrize("script,option,subcommand,dest", GLOBAL_OPTIONS)
def test_global_option_works_in_both_positions(script, option, subcommand, dest):
    """全局参数写在子命令前后都要能跑。

    只支持一种位置时，写错的人拿到的是 'unrecognized arguments'，
    它不说该放哪里 —— 这正是 agent 会开始自己猜的地方，而它猜的方向
    往往是「这个脚本坏了，我自己判断算了」。
    """
    parser = PARSERS[script]()
    before = parser.parse_args([option, "X"] + subcommand)
    after = parser.parse_args(subcommand + [option, "X"])
    assert getattr(before, dest) == "X"
    assert getattr(after, dest) == "X"


def test_every_script_with_a_parser_is_gated():
    """新增带 build_parser 的脚本时，别忘了挂进门禁。"""
    found = set()
    for directory in SCRIPT_DIRS:
        for path in directory.glob("*.py"):
            if "def build_parser" in path.read_text(encoding="utf-8"):
                found.add(path.name)
    missing = found - set(PARSERS)
    assert not missing, f"这些脚本有 build_parser 但没进门禁: {sorted(missing)}"


def test_every_gated_script_has_documented_commands():
    """每个进门禁的脚本都要有被扫到的示例。

    零覆盖意味着门禁对它形同虚设 —— 而那正是之前只盯 campaign.py 时的
    状态：其余九个脚本的文档示例怎么漂都没人管。
    """
    covered = {script for _p, script, _a, _v in DOC_COMMANDS}
    missing = set(PARSERS) - covered
    assert not missing, (
        f"这些脚本在门禁里零覆盖: {sorted(missing)}。"
        "要么文档里没有可执行示例，要么被抽取规则误过滤了")


@pytest.mark.parametrize("source", ["markdown", "docstring"])
def test_every_gated_script_is_covered_from_each_source(source):
    """两个来源要分别够得着每个脚本，不能只看合计。

    上一条只问「这个脚本有没有被扫到」。但示例有两个来源 —— markdown
    文档和脚本自己的 docstring（含运行时打印的恢复提示）—— 合计非零
    并不说明两边都扫到了。

    这不是假想：markdown 分支曾经一直硬编码着 `"campaign.py" in line`，
    于是九个脚本的 61 条 markdown 示例全程无人看管。而因为它们各自都有
    docstring 示例，合计不为零，上一条守卫一路放行。**少覆盖一个来源和
    少覆盖一个脚本是同一种失效，只是前者看不出来。**

    所以按来源分别断言。一侧断掉时当场点名是哪一侧，而不是等到某条
    示例漂了之后才发现根本没人在看。
    """
    covered = {
        script for path, script, _a, _v in DOC_COMMANDS
        if (str(path).endswith(".md")) == (source == "markdown")
    }
    missing = set(PARSERS) - covered
    assert not missing, (
        f"这些脚本在「{source}」这一侧零覆盖: {sorted(missing)}。"
        "合计不为零不代表两边都扫到了 —— 抽取规则可能只对另一侧生效")
