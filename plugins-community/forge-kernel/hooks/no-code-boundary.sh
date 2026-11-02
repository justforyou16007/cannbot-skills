#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# no-code-boundary.sh — PreToolUse hook，拦截对封存参考实现源码的读取。
#
# 为什么需要它
# ------------
# ForgeKernel 的方法建立在一个前提上：架构结论是从硬件可观测行为反推出来的，
# 不是从参考源码里看来的。这个前提一旦破了，方法就退化成「抄一份开源实现」，
# 而抄来的结论无法迁移到没有开源实现的算子上 —— 那恰恰是这套方法最有价值的
# 场景。
#
# 论文明确要求这条规则由确定性的 driver 在提示词之外强制执行。道理很实际：
# 提示词会被上下文压缩截断、会被改写、会被遗忘，而且挡不住一次无心的 grep。
# 工具调用层的拦截不受这些影响。
#
# 工作方式
# --------
# 读 Claude Code 传入的 hook JSON，取出本次工具调用要碰的路径，比对封存清单
# （由 seal_reference.py 生成）。命中就 exit 2，Claude Code 会把 stderr 的内容
# 回显给模型并阻止这次调用。
#
# 清单位置按以下顺序查找：
#   1. $FORGE_SEALED_PATHS            显式指定清单文件
#   2. $FORGE_WORK_DIR/sealed_paths.txt
#   3. ./forge_work/sealed_paths.txt  默认工作目录
# 三处都没有就放行 —— 尚未封存任何参考时不应妨碍正常工作。
#
# 逃生舱：FORGE_BOUNDARY_HOOK_DISABLE=1 降级为仅告警。调试用，不要常开。

set -uo pipefail

HOOK_NAME="forge-no-code-boundary"

# ---------------------------------------------------------------------------
# 定位封存清单
# ---------------------------------------------------------------------------
resolve_sealed_list() {
    if [[ -n "${FORGE_SEALED_PATHS:-}" && -f "${FORGE_SEALED_PATHS}" ]]; then
        echo "${FORGE_SEALED_PATHS}"
        return 0
    fi
    if [[ -n "${FORGE_WORK_DIR:-}" && -f "${FORGE_WORK_DIR}/sealed_paths.txt" ]]; then
        echo "${FORGE_WORK_DIR}/sealed_paths.txt"
        return 0
    fi
    if [[ -f "./forge_work/sealed_paths.txt" ]]; then
        echo "./forge_work/sealed_paths.txt"
        return 0
    fi
    return 1
}

SEALED_LIST="$(resolve_sealed_list)" || {
    # 还没封存任何参考，放行
    exit 0
}

# ---------------------------------------------------------------------------
# 解析输入并比对封存清单
#
# 用 python3 而不是 jq：jq 不一定装，而 python3 是本插件所有脚本的共同前提，
# 它不在就有更大的问题要先解决。
#
# 为什么解析和比对放在同一个 python 进程里：
# 这个 hook 挂在 Read/Grep/Glob/Bash 上，**每一次**工具调用都要等它返回，
# 所以它的耗时直接变成 agent 每一步的固定开销。早先的版本是两层 shell
# 循环（清单 N 条 × 本次目标 M 个），而且每比一次就起一个子 shell 做
# cd+pwd 归一化。封存整棵 opp/built-in（7583 条）时实测单次调用要 90 秒，
# 远超 hooks.json 里的 10 秒上限 —— 结果是要么被杀、要么白等，agent 循环
# 根本推不动。
#
# 同样的活在一个 python 进程里是「M 次路径归一化 + N 次字符串比较」，
# 没有 fork，万条清单也是毫秒级。反正输入本来就要起 python 来解析 JSON，
# 比对顺手一起做完，进程数从 N×M+1 降到 1。
# ---------------------------------------------------------------------------
VIOLATION="$(SEALED_LIST="${SEALED_LIST}" python3 -I -c '
import json, os, sys

try:
    payload = json.load(sys.stdin)
except (json.JSONDecodeError, ValueError):
    sys.exit(0)

tool = payload.get("tool_name", "")
data = payload.get("tool_input") or {}
if not isinstance(data, dict):
    sys.exit(0)

# --- 本次工具调用要碰的东西 ---------------------------------------------
# 文件路径类：路径就在参数里，归一化后比。
# 每个路径留两种写法：realpath 跟到软链真身（封存清单里存的是真身，
# 读软链同样算越界），abspath 保留字面形态（清单项本身是软链时才匹配得上）。
targets = []
for key in ("file_path", "path", "notebook_path"):
    value = data.get(key)
    if isinstance(value, str) and value:
        forms = {os.path.abspath(value), os.path.realpath(value)}
        targets.append((value, forms))

# Grep/Glob 不给 path 时默认搜当前目录，也要算进来 —— 否则「不写 path
# 的一次 grep」就是条绕过去的路。
if tool in ("Grep", "Glob") and not targets:
    cwd = os.getcwd()
    targets.append((cwd + "（未指定 path，默认搜这里）",
                    {os.path.abspath(cwd), os.path.realpath(cwd)}))

# Bash：命令文本里可能直接写了封存路径，只能做子串匹配。
command = None
if tool == "Bash":
    value = data.get("command")
    if isinstance(value, str) and value:
        command = value

if not targets and command is None:
    sys.exit(0)

# Grep/Glob 会递归向下搜。所以「目标是封存目录的上层」同样能读到封存内容，
# 和「目标在封存目录内」是一回事。Read 读一个具体文件不会递归，不适用。
recursive = tool in ("Grep", "Glob")

# --- 封存清单 -----------------------------------------------------------
try:
    with open(os.environ["SEALED_LIST"], "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()
except OSError:
    sys.exit(0)

for raw in lines:
    sealed = raw.split("#", 1)[0].strip()
    if not sealed:
        continue
    prefix = sealed.rstrip("/") + "/"

    for shown, forms in targets:
        for form in forms:
            if form == sealed or form.startswith(prefix):
                print("目标路径位于封存的参考实现内: " + shown)
                sys.exit(0)
            if recursive and prefix.startswith(form.rstrip("/") + "/"):
                print("递归搜索的起点是封存参考的上层目录: " + shown
                      + "\n它会读到: " + sealed)
                sys.exit(0)

    if command is not None and sealed in command:
        print("命令中引用了封存路径: " + sealed)
        sys.exit(0)
' 2>/dev/null)" || exit 0

[[ -z "${VIOLATION}" ]] && exit 0

# ---------------------------------------------------------------------------
# 拦截
# ---------------------------------------------------------------------------
MESSAGE="[${HOOK_NAME}] ${VIOLATION}

参考实现是封闭的：只能测，不能读。

这不是形式要求。ForgeKernel 的全部价值在于「架构决策是从 profiler 可观测的
硬件行为反推出来的」—— 一旦读了源码，结论就变成了抄来的，而抄来的结论无法
迁移到没有开源实现的算子上。

需要了解这个参考的架构，正确做法是测它：
  - 跑一次配对评测，看 pipeline 各段占比与瓶颈判定
  - 用 msprof 采 kernel_details，读 block_dim、各流水线占用
  - 把观测到的症状交给 forge-reference-analysis 翻译成架构决策假设

封存清单: ${SEALED_LIST}

若是递归搜索被拦：把搜索起点收窄到你自己的候选目录即可。被拦说明封存的
参考就在这个起点之下 —— 中转目录应当放在工作区之外（见 quickstart 步骤 1），
放在工作区里会让针对自己代码的 grep 一并被拦。

如确属误判（例如这个路径并非参考实现），请修正清单后重试。"

if [[ "${FORGE_BOUNDARY_HOOK_DISABLE:-0}" == "1" ]]; then
    echo "${MESSAGE}" >&2
    echo "[${HOOK_NAME}] 逃生舱已开启（FORGE_BOUNDARY_HOOK_DISABLE=1），仅告警不拦截。" >&2
    exit 0
fi

echo "${MESSAGE}" >&2
exit 2
