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
# 解析 hook 输入
#
# 用 python3 而不是 jq：jq 不一定装，而 python3 是本插件所有脚本的共同前提，
# 它不在就有更大的问题要先解决。
# ---------------------------------------------------------------------------
INPUT="$(cat)"

TARGETS="$(printf '%s' "${INPUT}" | python3 -I -c '
import json, sys

try:
    payload = json.load(sys.stdin)
except (json.JSONDecodeError, ValueError):
    sys.exit(0)

tool = payload.get("tool_name", "")
data = payload.get("tool_input") or {}
if not isinstance(data, dict):
    sys.exit(0)

out = []

# 直接读文件的工具：路径就在参数里
for key in ("file_path", "path", "notebook_path"):
    value = data.get(key)
    if isinstance(value, str) and value:
        out.append(value)

# Bash：命令行里可能含有封存路径，整条命令交给下一步做子串匹配。
# 用 \x01 作标记而不是 \x00 —— bash 的字符串存不住 null 字节，命令替换
# 会把它吞掉，标记就失效了。
if tool == "Bash":
    command = data.get("command")
    if isinstance(command, str) and command:
        out.append("\x01" + command)

print("\n".join(out))
' 2>/dev/null)" || exit 0

[[ -z "${TARGETS}" ]] && exit 0

# ---------------------------------------------------------------------------
# 比对封存清单
# ---------------------------------------------------------------------------
VIOLATION=""

while IFS= read -r sealed; do
    # 去注释与空白
    sealed="${sealed%%#*}"
    sealed="$(printf '%s' "${sealed}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "${sealed}" ]] && continue

    while IFS= read -r target; do
        [[ -z "${target}" ]] && continue

        if [[ "${target}" == $'\x01'* ]]; then
            # Bash 命令：看命令文本里有没有提到封存路径
            cmd="${target#$'\x01'}"
            if [[ "${cmd}" == *"${sealed}"* ]]; then
                VIOLATION="命令中引用了封存路径: ${sealed}"
                break 2
            fi
            continue
        fi

        # 文件路径：归一化后比对前缀
        abs="$(cd "$(dirname "${target}")" 2>/dev/null && pwd)/$(basename "${target}")" \
            || abs="${target}"
        if [[ "${abs}" == "${sealed}" || "${abs}" == "${sealed%/}/"* ]]; then
            VIOLATION="目标路径位于封存的参考实现内: ${target}"
            break 2
        fi
    done <<< "${TARGETS}"
done < "${SEALED_LIST}"

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
如确属误判（例如这个路径并非参考实现），请修正清单后重试。"

if [[ "${FORGE_BOUNDARY_HOOK_DISABLE:-0}" == "1" ]]; then
    echo "${MESSAGE}" >&2
    echo "[${HOOK_NAME}] 逃生舱已开启（FORGE_BOUNDARY_HOOK_DISABLE=1），仅告警不拦截。" >&2
    exit 0
fi

echo "${MESSAGE}" >&2
exit 2
