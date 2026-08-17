#!/bin/bash
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
# msprof runner - 统一性能采集入口
#
# 支持三种模式：
#   1. 标准模式 (默认): 对单个可执行文件采集 7 组 aic-metrics + sample-based
#   2. --compare 模式:  对算子目录下的 model.py vs model_new_ascendc.py 做对比测试
#   3. --quick 模式:    单卡单进程批量对比（每 case 一个 torch.npu.Stream，一次 msprof 采全部）
#
# Usage (标准模式):
#   bash msprof_profile_run.sh [--warm-up=N] [--output=<dir>] -- <executable> [args...]
#
# Usage (--compare 模式):
#   bash msprof_profile_run.sh --compare --output-dir=/path/to/op_dir [--warm-up=N] [--device=N]
#
# Usage (--quick 模式):
#   bash msprof_profile_run.sh --quick --output-dir=/path/to/op_dir [--warm-up=N] [--device=N]
#

# Example (标准):
#   bash msprof_profile_run.sh --warm-up=3 --output=./msprof_output -- \
#        ./matmul_tutorial_mxfp4_pingpong 8448 4096 4096
#
# Example (对比):
#   bash msprof_profile_run.sh --compare --output-dir=./output/GELU --warm-up=3 --device=0
#
# Example (快速):
#   bash msprof_profile_run.sh --quick --output-dir=./output/GELU --warm-up=3 --device=0
#

# 产出:
#   标准模式: <output_dir>/PROF_GROUP_<timestamp>/PROF_<Metric>/ (7 个子目录 + 1 个 Sample)
#   对比模式: <output_dir>/performance.json + performance.log + perf_report.md
#   快速模式: <output_dir>/performance.json + performance.log + perf_report.md
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_START_TIME=$(date +%s%N)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 格式化耗时输出（纳秒精度入参）
format_elapsed() {
    local ns=$1
    local ms=$(( ns / 1000000 ))
    local s=$(( ms / 1000 ))
    local remain_ms=$(( ms % 1000 ))
    if [[ $s -gt 0 ]]; then
        printf "%d.%03ds" $s $remain_ms
    else
        printf "%dms" $ms
    fi
}

print_elapsed() {
    local now_ns
    now_ns=$(date +%s%N)
    local elapsed_ns=$(( now_ns - SCRIPT_START_TIME ))
    local msg="[INFO] Total elapsed time: $(format_elapsed $elapsed_ns)"
    echo ""
    echo "$msg"
    # 如果指定了输出目录，同时写入 performance.log
    if [[ -n "${1:-}" ]]; then
        echo "" >> "$1/performance.log"
        echo "$msg" >> "$1/performance.log"
    fi
}

# 默认参数
WARM_UP=1
OUTPUT_DIR="./msprof_output"
APP_CMD=()

# 模式开关
MODE="standard"   # standard | compare | quick
OUTPUT_DIR_ARG=""  # --compare/--quick 模式用的算子目录
DEVICE_ID=""
REFERENCE_CACHE="${LINGXI_REFERENCE_CACHE:-}"

usage() {
    cat <<'EOF'
Usage: bash msprof_profile_run.sh [OPTIONS] -- <executable> [args...]
       bash msprof_profile_run.sh --compare --output-dir=<dir> [OPTIONS]
       bash msprof_profile_run.sh --quick --output-dir=<dir> [OPTIONS]

通用选项:
  --warm-up=N     在正式采集之前，先跑 N 次可执行文件（不采集）预热 DVFS。默认 3。
  --output=<dir>  msprof 结果根目录。默认 ./msprof_output。
  -h, --help      Show this help.

标准模式 (默认):
  bash msprof_profile_run.sh [通用选项] -- <executable> [args...]

对比模式 (--compare):
  --compare              启用对比模式：测试 model.py vs model_new_ascendc.py
  --output-dir=<dir>     算子输出目录（包含 model.py, model_new_ascendc.py, .jsonl）
  --device=N             指定 NPU 设备 ID；未指定时自动选择空闲卡
  --repeats=N            重复采集次数（默认 1）
  --retry=N              单 case 解析失败重试次数（默认 2）
  --reference-cache=PATH 会话级 reference 计时缓存（quick 模式相同 model/case 只跑一次）
  --keep-prof            保留 msprof 原始 PROF 目录（用于深度分析）

快速模式 (--quick):
  --quick                启用快速模式：只采集 1 轮（只获取 kernel 时间，不采集 7 个 aic-metrics）
  --output-dir=<dir>     算子输出目录（包含 model.py, model_new_ascendc.py, .jsonl）
  --device=N             指定 NPU 设备 ID；未指定时自动选择空闲卡
  --repeats=N            重复采集次数（默认 1）
  --retry=N              单 case 解析失败重试次数（默认 2）

脚本会按顺序用 msprof 采集 7 个 aic-metrics:
  PipeUtilization, ArithmeticUtilization, Memory, MemoryL0, MemoryUB,
  L2Cache, ResourceConflictRatio
所有 PROF 目录放在同一个 PROF_GROUP_<timestamp>/ 下，便于 msprof_perf_summary.py 汇总。

快速模式 (--quick) 只采集 1 轮，不采集 7 个 aic-metrics，只获取 kernel 时间。
EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --warm-up=*) WARM_UP="${1#*=}"; shift ;;
        --output=*)  OUTPUT_DIR="${1#*=}"; shift ;;
        --output-dir=*) OUTPUT_DIR_ARG="${1#*=}"; shift ;;
        --device=*)  DEVICE_ID="${1#*=}"; shift ;;
        --repeats=*) REPEATS="${1#*=}"; shift ;;
        --retry=*)   RETRY="${1#*=}"; shift ;;
        --reference-cache=*) REFERENCE_CACHE="${1#*=}"; shift ;;
        --keep-prof) KEEP_PROF=1; shift ;;
        --compare)   MODE="compare"; shift ;;
        --quick)     MODE="quick"; shift ;;
        -h|--help)   usage; exit 0 ;;
        --)          shift; APP_CMD=("$@"); break ;;
        -*)          echo "ERROR: unknown option $1" >&2; usage; exit 1 ;;
        *)           APP_CMD=("$@"); break ;;
    esac
done

# 检查 msprof
if ! command -v msprof >/dev/null 2>&1; then
    echo "ERROR: 未找到 msprof，请先 source CANN set_env.sh / setenv.bash" >&2
    exit 1
fi

# ============================================================================
# 标准模式：对单个可执行文件采集
# ============================================================================
run_standard() {
    if [[ ${#APP_CMD[@]} -eq 0 ]]; then
        echo "ERROR: 缺少可执行文件参数 (使用 -- 分隔 msprof 选项与 app 命令)" >&2
        usage
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"
    OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
    TS="$(date +%Y%m%d_%H%M%S)"
    GROUP_DIR="${OUTPUT_DIR}/PROF_GROUP_${TS}"
    mkdir -p "$GROUP_DIR"

    echo "=== msprof profiling ==="
    echo "App      : ${APP_CMD[*]}"
    echo "Output   : ${GROUP_DIR}"
    echo "Warm-up  : ${WARM_UP}"

    if [[ "$WARM_UP" -gt 0 ]]; then
        echo "--- Warm-up x${WARM_UP} ---"
        for i in $(seq 1 "$WARM_UP"); do
            echo "  warm-up ${i}/${WARM_UP}"
            "${APP_CMD[@]}" >/dev/null 2>&1 || true
        done
    fi

    METRICS=(PipeUtilization ArithmeticUtilization Memory MemoryL0 MemoryUB L2Cache ResourceConflictRatio)

    for M in "${METRICS[@]}"; do
        SUB="${GROUP_DIR}/PROF_${M}"
        mkdir -p "$SUB"
        echo "--- aic-metrics=${M} -> ${SUB} ---"
        msprof --output="$SUB" \
               --ai-core=on \
               --aic-metrics="$M" \
               --task-time=on \
               --ascendcl=on \
               "${APP_CMD[@]}" >"${SUB}/msprof.log" 2>&1 || {
            echo "ERROR: msprof run failed for ${M}. See ${SUB}/msprof.log" >&2
            tail -20 "${SUB}/msprof.log" >&2 || true
            exit 1
        }
    done

    # 追加 sample-based 采集一份，用于逐核负载均衡分析
    SAMPLE_SUB="${GROUP_DIR}/PROF_Sample"
    mkdir -p "$SAMPLE_SUB"
    echo "--- sample-based (per-core load balance) -> ${SAMPLE_SUB} ---"
    msprof --output="$SAMPLE_SUB" \
           --ai-core=on \
           --aic-metrics=PipeUtilization \
           --aic-mode=sample-based \
           --aic-freq=100 \
           --task-time=on \
           --ascendcl=on \
           "${APP_CMD[@]}" >"${SAMPLE_SUB}/msprof.log" 2>&1 || {
        echo "WARN: sample-based run failed, per-core analysis will be skipped. See ${SAMPLE_SUB}/msprof.log" >&2
        tail -20 "${SAMPLE_SUB}/msprof.log" >&2 || true
    }

    print_elapsed
    echo ""
    echo "=== Done. PROF group = ${GROUP_DIR} ==="
    echo "Next:"
    echo "  python3 ${SCRIPT_DIR}/msprof_perf_summary.py ${GROUP_DIR} <ops_dir>"
}

# ============================================================================
# 对比模式：model.py vs model_new_ascendc.py
# ============================================================================
run_compare() {
    if [[ -z "$OUTPUT_DIR_ARG" ]]; then
        echo "ERROR: --compare 模式必须指定 --output-dir=<dir>" >&2
        usage
        exit 1
    fi

    OUT_DIR="$(cd "$OUTPUT_DIR_ARG" && pwd)"

    # 调用 msprof_perf_summary.py --compare（已融合 kernel_perf.py 功能）
    local extra_args=()
    [[ -n "$DEVICE_ID" ]] && extra_args+=("--device" "$DEVICE_ID")
    [[ -n "${REPEATS:-}" ]] && extra_args+=("--repeats" "$REPEATS")
    [[ -n "${RETRY:-}" ]] && extra_args+=("--retry" "$RETRY")
    [[ "${KEEP_PROF:-0}" == "1" ]] && extra_args+=("--keep-prof")
    [[ -n "$REFERENCE_CACHE" ]] && extra_args+=("--reference-cache" "$REFERENCE_CACHE")

    python3 "${SCRIPT_DIR}/msprof_perf_summary.py" --compare \
        --output-dir "$OUT_DIR" \
        --warmup "$WARM_UP" \
        "${extra_args[@]}"

    print_elapsed "$OUT_DIR"
}

# ============================================================================
# 快速模式：只采集 1 轮（只获取 kernel 时间，不采集 7 个 aic-metrics）
# ============================================================================
run_quick() {
    if [[ -z "$OUTPUT_DIR_ARG" ]]; then
        echo "ERROR: --quick 模式必须指定 --output-dir=<dir>" >&2
        usage
        exit 1
    fi

    OUT_DIR="$(cd "$OUTPUT_DIR_ARG" && pwd)"

    # 调用 msprof_perf_summary.py --quick（只测时间，不采集 7 个 metrics）
    local extra_args=()
    [[ -n "$DEVICE_ID" ]] && extra_args+=("--device" "$DEVICE_ID")
    [[ -n "${REPEATS:-}" ]] && extra_args+=("--repeats" "$REPEATS")
    [[ -n "${RETRY:-}" ]] && extra_args+=("--retry" "$RETRY")
    [[ -n "$REFERENCE_CACHE" ]] && extra_args+=("--reference-cache" "$REFERENCE_CACHE")

    python3 "${SCRIPT_DIR}/msprof_perf_summary.py" --quick \
        --output-dir "$OUT_DIR" \
        --warmup "$WARM_UP" \
        "${extra_args[@]}"

    print_elapsed "$OUT_DIR"
}

# ============================================================================
# ============================================================================
# 主入口
# ============================================================================
case "$MODE" in
    standard)
        run_standard
        ;;
    compare)
        run_compare
        ;;
    quick)
        run_quick
        ;;
    *)
        echo "ERROR: 未知模式: $MODE" >&2
        usage
        exit 1
        ;;
esac
