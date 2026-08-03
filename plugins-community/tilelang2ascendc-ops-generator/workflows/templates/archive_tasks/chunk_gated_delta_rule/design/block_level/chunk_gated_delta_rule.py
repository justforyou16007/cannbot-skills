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

"""Block-level design of Chunked Gated Delta Rule — 入口 / 任务划分 + 流水编排。

对应 AscendC `CGDR::Process`（kernel/op_kernel/chunk_gated_delta_rule.h）。本文件回答
BlockLevelDesign 的四件事中的「任务划分」和「流水编排」；C/V 分工的细粒度计算
在 stage1/2/3 中展开。

## 任务划分（block = (batch, value-head) 单元）

输出 `out[T, Nv, Dv]`。状态 S `(Dv, Dk)` 沿 chunk 递推，因此 (batch, value-head)
之间完全独立、chunk 之间串行。取：

    block 数     = batch * nv          # 每个 (b, nv) 一个 block，无写冲突、无跨 block 同步
    block 输出区 = out[该 (b,nv) 的全部 chunk 行, nv, :]
    block 遍历   = 该 (b,nv) 的 chunk 序列（ci=0..n_chunks(b)-1，S 在块内 UB 携带）
    key head     = value head nv 读 key head nv*Nk//Nv

各 block 的 chunk 数由 actual_seq_lengths[b] 决定；不同 block 并行（persistent-kernel
循环 `unit = cid*tasksPerCore + localIdx`）。

## 块内流水（每 chunk 三阶段，对应 AscendC Stage1/2/3）

    for ci in 0..n_chunks(b):
        stage1: within-chunk 预计算（C1  kbt=kb@k^T；V1  A/X 逆矩阵 + kb/vb/kbe/qPrime/kg；C2  v_inner/k_cumdecay）
        stage2: 跨 chunk 递推    （C3  v_prime=kcumdecay@S、attn_inter=qPrime@S；V2  v_new；V3  S 更新）
        stage3: chunk 内累加     （V4  attn_c=mask(qkt*decay)*scale；C4  out += attn_c@v_new）

C 侧（cube matmul）与 V 侧（向量）通过 workspace（GM，bf16）交接，阶段边界做全核同步，
对应 AscendC 的 AIC/AIV 分工与 `SyncAll`。

## 为什么这样划分

- 优先无写冲突、无额外同步的切分：按 (batch, value-head) 切，块间完全独立。
- 状态递推决定 chunk 串行：S 是跨 chunk 的 loop-carried 依赖，必须块内串行，无法像
  FlashAttention 那样把不同 chunk 的 C/V 交错重叠（除非引入 chunk-group + GM 状态携带
  + SyncAll 的二次并行，AscendC 即如此；本设计保留块内串行为主路径）。
- workspace 存 stage1 产物（bf16），与 AscendC 一致：gCumExp/kCumDecay/vInner/qPrime/qkt/kg。

## 同步设计（对应 AscendC 的 `SyncAll<false>()` 与 AIC↔AIV 事件）

AscendC 用两套同步，block-level 里一一对应：

1. **阶段间 / 组间全核 barrier —— `SyncAll<false>()` ↔ `T.barrier_all()`**：
   在 `RunStage1 → RunStage2 → RunStage3` 之间（以及 AscendC 的 chunk-group 之间）各插一次
   全核同步，保证上一阶段所有核的产物（workspace/out/S）已就绪、下一阶段才读。
   本设计的 main() 在 stage1→2→3 边界显式调用 `T.barrier_all()`（见下方代码）。
2. **阶段内 AIC↔AIV 交接 —— workspace + cross/pipe flag**：
   Stage1 内部 C（cube matmul）与 V（向量）通过 workspace 交接，用
   `T.set_cross_flag`/`T.wait_cross_flag`（AIC/AIV 间）与 `T.set_flag`/`T.wait_flag`
   （MTE2→V 等 pipe 级）同步；对应 AscendC 的 `SetFlag`/`WaitFlag`/`CrossCoreSetFlag`。
   这一层在 stage1/2/3 各自内部实现。

> 说明：AscendC 的 `SyncAll` 同步的是"一组核共同处理一个 chunk-group"；本设计块=（batch,head）
> 独立、每核跑自己的单元，`T.barrier_all()` 同步的是启动的全部 (cid, vid) 线程——语义都是
> "阶段边界全核就绪"，是本设计对应 `SyncAll` 的落点。
"""

import tilelang
import tilelang.language as T

from design.chunk_gated_delta_rule_common import CHUNK_SIZE, NUM_PHYSICAL_CORES, pass_configs
from design.block_level.stage1 import run_stage1
from design.block_level.stage2 import run_stage2
from design.block_level.stage3 import run_stage3

# workspace 索引常量（与 AscendC tiling 对齐）
WS_GCUMEXP = 0    # (block_num, C)      fp32   g_cum 的 exp
WS_KCUMDECAY = 1  # (block_num, C, Dk)  bf16
WS_VINNER = 2     # (block_num, C, Dv)  bf16
WS_QPRIME = 3     # (block_num, C, Dk)  bf16
WS_QKT = 4        # (block_num, C, C)   bf16
WS_KG = 5         # (block_num, C, Dk)  bf16


@tilelang.jit(out_idx=[7, 8], workspace_idx=[9, 10, 11, 12, 13, 14], pass_configs=pass_configs)
def chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv, chunk_size=CHUNK_SIZE, scale=1.0):
    C = chunk_size
    total_units = batch * nv                 # block 数
    used_core_num = min(NUM_PHYSICAL_CORES, max(1, total_units))
    tasks_per_core = T.ceildiv(total_units, used_core_num)
    max_chunks = (seq_len + C - 1) // C      # 单 (b,nv) 最大 chunk 数（workspace 上界）

    @T.prim_func
    def main(
        query: T.Tensor((seq_len, nk, dk), "bfloat16"),
        key: T.Tensor((seq_len, nk, dk), "bfloat16"),
        value: T.Tensor((seq_len, nv, dv), "bfloat16"),
        beta: T.Tensor((seq_len, nv), "bfloat16"),
        g: T.Tensor((seq_len, nv), "float32"),
        initial_state: T.Tensor((batch, nv, dv, dk), "bfloat16"),
        actual_seq_lengths: T.Tensor((batch,), "int32"),
        out: T.Tensor((seq_len, nv, dv), "bfloat16"),
        final_state: T.Tensor((batch, nv, dv, dk), "bfloat16"),
        # workspace：stage1 产物（bf16/fp32），块内 C/V 交接
        ws_gcumexp: T.Tensor((total_units, C), "float32"),
        ws_kcumdecay: T.Tensor((total_units, C, dk), "bfloat16"),
        ws_vinner: T.Tensor((total_units, C, dv), "bfloat16"),
        ws_qprime: T.Tensor((total_units, C, dk), "bfloat16"),
        ws_qkt: T.Tensor((total_units, C, C), "bfloat16"),
        ws_kg: T.Tensor((total_units, C, dk), "bfloat16"),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # ---- 任务划分：persistent-kernel 风格，把 block=(b,nv) 分给 AI Core ----
            for local_idx in T.serial(tasks_per_core):
                unit = cid * tasks_per_core + local_idx
                if unit < total_units:
                    b = unit // nv
                    h = unit % nv
                    kh = (h * nk) // nv      # 头部分组：value head h 读 key head kh

                    # 该 (b,nv) 的序列窗口：base=sum(seq[0..b))，Lb=seq[b]
                    base = T.alloc_ub((1,), "int32")
                    base[0] = 0
                    for bb in T.serial(b):
                        base[0] += actual_seq_lengths[bb]
                    Lb = actual_seq_lengths[b]

                    # ---- 状态 S（块内 UB，跨 chunk 携带），初值 = initial_state[b,h] ----
                    with T.Scope("V"):
                        state = T.alloc_ub((dv, dk), "float32")
                        T.copy(initial_state[b, h, :, :], state)   # bf16 -> fp32 升型
                        T.barrier_all()

                    # ---- 块内 chunk 串行循环：每 chunk 跑 stage1 -> stage2 -> stage3 ----
                    for ci in T.serial(max_chunks):
                        if ci * C < Lb:
                            s0 = base[0] + ci * C
                            valid = T.if_then_else(Lb - ci * C < C, Lb - ci * C, C)

                            # ---- Stage1 -> SyncAll -> Stage2 -> SyncAll -> Stage3 ----
                            # 边界全核同步对应 AscendC `SyncAll<false>()`（见 docstring §同步设计）。
                            # stage1：within-chunk 预计算 -> workspace（C1/V1/C2）
                            run_stage1(
                                query, key, value, beta, g,
                                s0, valid, kh, h, scale, unit, C, dk, dv,
                                ws_gcumexp[unit], ws_kcumdecay[unit],
                                ws_vinner[unit], ws_qprime[unit], ws_qkt[unit], ws_kg[unit],
                            )
                            T.barrier_all()   # SyncAll：Stage1 workspace 全部就绪

                            # stage2：跨 chunk 递推（状态依赖），out 写 attn_inter，更新 S
                            run_stage2(
                                ws_gcumexp[unit], ws_kcumdecay[unit], ws_vinner[unit],
                                ws_qprime[unit], ws_kg[unit], state, valid, C, dk, dv,
                                out[s0:s0 + valid, h, :], final_state[b, h],
                            )
                            T.barrier_all()   # SyncAll：out 的 attn_inter 已写、S 已更新

                            # stage3：attn_c = mask(qkt*decay)*scale；out += attn_c@v_new
                            run_stage3(
                                ws_qkt[unit], ws_gcumexp[unit], ws_vinner[unit],
                                scale, valid, C, dk, dv, out[s0:s0 + valid, h, :],
                            )
                            T.barrier_all()   # SyncAll：本 chunk 的 out 累加完成，进入下一 chunk

    return main
