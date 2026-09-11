#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Block-level TileLang design for GroupedMatmul（slice-M / MoE 分组矩阵乘）。
#
# 语义：y[rows_g] = x[rows_g] @ weight[g] (+ bias[g])，g ∈ [0, E)
#   x [M, K]（沿 M 轴合并所有 expert 的 token 行）
#   weight [E, K, N]（transpose_weight=false）或 [E, N, K]（true）
#   bias [E, N] 可选（全程 fp32）
#   group_list：cumsum，长度 E，末值 = M；相邻相等 = 空组（跳过）
#   y [M, N]；split_item 仅影响 host 层输出包装（kernel 恒算 [M, N]）
#
# 本文件只确定 block 级任务划分、流水骨架、metadata 与同步关系；
# 具体计算（L1/L0 双缓冲 mma 流水、bias BT 数据流、尾块钳位）在 tile_level 补全。
#
# ── 任务划分（block 间分工）────────────────────────────────────────────────
# 三维 Kernel 不支持 → 一维 grid + block_metadata 预计算表（grouped_gemm 模式）：
#   host 把 group_list 展开成 block_metadata[total_blocks, 4]（int32）：
#     [g, m_start_global, valid_m, n_start]
#   每个 block 负责一个 (g 的某个 m 砖, 某个 n 砖)：
#     total_blocks = Σ_g ceildiv(m_g, block_M) × n_num   （空组 m_g=0 → 0 块，天然跳过）
#   kernel 内 bx = cid // n_num 索引 metadata，by = cid % n_num。
#
# ── 核间分工（Fixed Core）──────────────────────────────────────────────────
#   launch = min(total_blocks, CORE_NUM)（CORE_NUM = get_core_num("cube")，编译期读目标芯片）
#   核内 T.serial 遍历 task_idx = i*CORE_NUM + cid，use_swizzle 做 L2 局部性映射，
#   task_idx < total_blocks 守卫（matmul 蓝本 idiom）。
#
# ── 流水骨架（纯 cube，无 CV）──────────────────────────────────────────────
#   GM → L1（MTE2，S1=2 双缓冲，K_L1 chunk 预取与 mma 重叠）
#   L1 → L0A/L0B（MTE1，S2=2 双缓冲，block_K 乒乓）
#   L0A×L0B → L0C（M，mma 累加；bias 时首个 mma 带 BT bias 操作数预置 L0C）
#   L0C → GM（FIX，fixpipe 钳位 M/N 尾块）
#   全手写 flag（mte1/mte2、m/mte1、fix/m），不开 AUTO_SYNC（intrinsic 版）。
#
# ── metadata 与同步关系 ────────────────────────────────────────────────────
#   block_metadata 为 GM int32 tensor，kernel 内标量读取（每 block 一次，开销可忽略）。
#   m_start/valid_m 为运行期值（per-shape 编译时是 host 预计算的常量结果；
#   Phase 4 泛化时由 host 从运行期 group_list 计算后传入）。
#   bias[g] 的 GM 偏移 = g * N（运行期 g 来自 metadata）。

from dataclasses import dataclass

# tile 配置（per-shape 编译期常量；含义见 PERF_DESIGN.md §5）
# fp16/bf16: block_M=128, block_N=256, block_K=64, K_L1=256
# fp32:      block_M=128, block_N=128, block_K=64, K_L1=128


@dataclass(frozen=True)  # frozen: tilelang.jit 需要对 cfg 取 hash 做编译缓存键
class GroupedMatmulConfig:
    m: int                 # 总 M（= group_list[-1]）
    n: int                 # N
    k: int                 # K
    e: int                 # expert 数
    dtype: str = "float16"
    has_bias: bool = False
    transpose_weight: bool = False
    total_blocks: int = 0  # host 预计算的 Σ_g ceildiv(m_g, block_M) × n_num


# TODO(tile-level): 具体 kernel 实现见 design/tile_level/grouped_matmul.py
