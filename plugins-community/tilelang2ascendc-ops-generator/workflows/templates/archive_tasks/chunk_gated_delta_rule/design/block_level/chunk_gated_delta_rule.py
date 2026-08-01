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

"""Block-level design of Chunked Gated Delta Rule — **入口 / 编排器**.

对应 AscendC 实现：`kernel/op_kernel/chunk_gated_delta_rule.h` 中的 `CGDR` 类。
本文件只描述「批次 → chunk-group → 三阶段」的编排骨架与状态载体，各阶段的细粒度
计算见 `stage1.py` / `stage2.py` / `stage3.py`。

算子（Chunked Gated Delta Rule，chunk 版门控 Delta 规则，线性注意力 prefill 阶段）：
- 输入：query/key `(T,Nk,Dk)`、value `(T,Nv,Dv)`、beta `(T,Nv)`、g `(T,Nv)`、
  initial_state `(B,Nv,Dv,Dk)`、actual_seq_lengths `(B,)`；scale 属性。
- 输出：out `(T,Nv,Dv)`、final_state `(B,Nv,Dv,Dk)`。
- 语义：对每个 (batch, value-head) 单元，状态 S `(Dv,Dk)` 沿 chunk 递推；
  头部分组 value head nv 读 key head `nv*Nk//Nv`。

AscendC 编排（`CGDR::Process`）：
```
for bid in 0..B:                      # 每个 batch
    seqStart, seqEnd = actual_seq_lengths[bid] 累计
    for pos in seqStart..seqEnd step maxGroupLength:   # 每个 chunk-group
        cg = ChunkGroup(pos, length)
        curState = (pos==seqStart) ? initial_state[bid] : final_state[bid]
        RunStage1(cg)      # 组内所有 chunk 的 within-chunk 预计算（写 workspace）
        SyncAll()
        RunStage2(cg, curState, final_state[bid])   # 用状态算 out 的一部分 + 状态更新
        SyncAll()
        RunStage3(cg)      # mask qkt -> attn_c，累加到 out
        SyncAll()
```
组间通过 `final_state[bid]`（GM 上 bf16）携带状态；组内跨核用 `SyncAll` 强同步
（AIC 算 matmul、AIV 算向量，阶段间全核 barrier）。
"""

import tilelang
import tilelang.language as T

from design.chunk_gated_delta_rule_common import (
    CHUNK_SIZE,
    NUM_PHYSICAL_CORES,
    pass_configs,
)

# chunk-group 长度 = p(每核并行 chunk 数) * aiCoreNum * chunkSize（与 AscendC 一致）。
CHUNKS_PER_CORE = 2


@tilelang.jit(out_idx=[7, 8], workspace_idx=[], pass_configs=pass_configs)
def chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv, chunk_size=CHUNK_SIZE, scale=1.0):
    C = chunk_size
    total_chunks = (seq_len + C - 1) // C
    used_core_num = min(NUM_PHYSICAL_CORES, max(1, total_chunks))
    tasks_per_core = T.ceildiv(total_chunks, used_core_num)

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
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # 与 AscendC CGDR::Process 对应：batch -> chunk-group -> stage1/2/3。
            # 每个核心负责一组连续的 chunk；chunk 之间按组串行，组间状态经
            # final_state 携带。三阶段间用全核同步（TileLang 里由
            # TL_ASCEND_AUTO_SYNC / T.barrier_all 体现，对应 AscendC SyncAll）。
            #
            # Stage1（within-chunk 预计算，见 stage1.py）：
            #   gCumExp / kCumDecay / vInner / qPrime / qkt / kg 写 workspace
            # Stage2（跨 chunk 递推，见 stage2.py）：
            #   v_prime = kCumdecay @ S；v_new = vInner - v_prime
            #   attn_inter = qPrime @ S  -> out；S = exp(g_last)*S + kg^T @ v_new -> final_state
            # Stage3（见 stage3.py）：
            #   attn_c = mask(qkt * decay)；out += attn_c @ v_new
            for local_idx in T.serial(tasks_per_core):
                chunk = cid * tasks_per_core + local_idx
                if chunk < total_chunks:
                    with T.Scope("V"):
                        pass  # 编排骨架：state 初始化 + chunk 循环 + 阶段调用见 stage1/2/3

    return main
