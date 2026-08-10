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

"""Block-level 设计：Chunked Gated Delta Rule — 单 kernel 入口（chunk-group 协作模型）。

对应 AscendC `CGDR::Process`（单 kernel；batch → chunk-group → RunStage1/2/3 + SyncAll）。
细算落地在 `design/tile_level/`（stage1/2/3.py，TODO(tile-level)）。满足
`BlockLevelDesign-quality-checklist.md`。

## 任务划分（checklist #1，对齐 AscendC Process）

状态 S `(Dv, Dk)` 沿 chunk 递推，因此 chunk-group 之间串行、group 内 (nv, chunk) 任务按核并行：

    blockDim = aiCoreNum ≈ NUM_PHYSICAL_CORES
    maxGroupLength = P_NUM(2) × aiCoreNum × chunkSize(64)      # 一组覆盖行数
    Process()（每核同 body）:
        for bid in 0..b-1:
            seqStart/seqEnd 由 actual_seq_lengths 累计
            for pos in seqStart..seqEnd step maxGroupLength:    # chunk-group 循环
                cg = {startPos:pos, length:min(maxGroupLength, seqEnd-pos)}
                curState = (pos==seqStart) ? initState[b] : finalState[b]
                RunStage1(cg)   # totalChunk=nv*numChunk 按 coreIdx 均分（stage1.h former/tail）
                SyncAll
                RunStage2(cg, curState, finalState[b])  # nvPerCore=ceil(Nv/core) 切核，核内串行 cId
                SyncAll
                RunStage3(cg)   # totalChunks=nv*chunkNum 融合均分（stage3.h）
                SyncAll

## 布局整理（checklist #3）

TND 已按 (T, N, D) 连续排布，无需额外 pack。头分组 `kh=(nv*Nk)//Nv` 在 stage1 内按运行时
head 索引取用。workspace 布局 `(Nv, maxGroupLength, ...)`，行偏移
`chunkRowBase = nvId*maxGroupLength + cgId*C`（stage1.h `SetChunkOffset`）。

## C/V 分工与同步（checklist #4, #5）

- **阶段间**：`T.sync_all()` ↔ AscendC `SyncAll<false>()`（每 chunk-group 阶段边界各一次）。
- **入口**：`T.sync_all()` ↔ AscendC `SyncAll<false>()`（Process 开头一次，所有核就绪）。
- **阶段内 AIC↔AIV**：stage1/2/3 用 `T.Scope("C")`（cube）/`T.Scope("V")`（向量）分工，
  GM workspace 交接 + `T.set_cross_flag`/`T.wait_cross_flag`（对应 AscendC `SetFlag`/`WaitFlag`）。
  信号编号 0..7 与 AscendC stage 内 event id 一一对应（见 ascendc_tilelang_diff.md §7）。

## workspace（checklist #6，GM，AIC↔AIV 交接；形状 `(Nv, maxGroupLength, ...)`）

| 张量 | shape | dtype | 谁写 | 谁读 |
|---|---|---|---|---|
| `q/k/kbe(gBK)/qprime/kg/k_cumdecay` | `(Nv, G, Dk)` | bf16 | stage1 | stage1/2 |
| `vb/v_inner` | `(Nv, G, Dv)` | bf16 | stage1 | stage2 更新 / stage3 |
| `kbt/qkt/x/attn_c` | `(Nv, G, C)` | bf16 | stage1/3 | stage1/3 |
| `g_exp/g_exp_neg` | `(Nv, G)` | fp32 | stage1 | stage1/2/3 |
| 状态 S | `initial_state/final_state (B, Nv, Dv, Dk)` | bf16 | stage2 | stage2（curState） |
| `out` | `(T, Nv, Dv)` | bf16 | stage2（attn_inter）+ stage3（累加） | — |

已精简（照抄 AscendC）：无 `kb / v_prime / attn_inter / v_new / dstate / out_cont` workspace——
kkt 用纯 k（beta 在 V1a 广播乘）、`kbe` 存 gBK（`-β·g_exp·k`，带负号）、attn_inter 直写 out、
v_new 即 C3 原子加更新后的 `v_inner`、dstate 由 C3'' 原子加进 final_state、out_cont 由 C4 原子加进 out。

## GEMM 约束（checklist #10）

- block_M = chunk_size = 64 ≥ 16；block_N（dk/dv ≤ 128）≥ 16，满足 NPU 分形。
- 最大 matmul `X@vb`：`64 × 128 × sizeof(float) = 32KB ≤ L0C(128KB)` ✓。

## 技术约束（checklist #9）

- 尾 chunk/尾 group：`valid = min(C, cg_len - cgId*C)`、`cg_len = min(maxGroupLength, ...)` 在
  kernel 内用 `T.if_then_else` 由运行时 `Lb`/`gpos` 算出。
- 动态 batch 长度：kernel 内 `base` 累计 `actual_seq_lengths` 得 `Lb` 驱动 group 循环守卫。

## 细算落地（checklist #7）

各 stage 的具体计算标 `TODO(tile-level)`，在 `design/tile_level/stageN.py` 落地。
"""

import tilelang
import tilelang.language as T

from design.chunk_gated_delta_rule_common import (
    CHUNK_SIZE,
    NUM_PHYSICAL_CORES,
    P_NUM,
    pass_configs,
)
from design.block_level.stage1 import run_stage1
from design.block_level.stage2 import run_stage2
from design.block_level.stage3 import run_stage3


@tilelang.jit(
    out_idx=[7, 8],
    workspace_idx=[9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                   19, 20, 21, 22],
    pass_configs=pass_configs,
)
def chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv,
                           chunk_size=CHUNK_SIZE, scale=1.0):
    """编排器：build 单 kernel（chunk-group 协作骨架），host 单 launch。

    返回 callable `(query, key, value, beta, g, initial_state, actual_seq_lengths) -> (out, final_state)`。
    细算 TODO(tile-level) 在 tile_level 落地。
    """
    C = chunk_size
    used_core_num = NUM_PHYSICAL_CORES
    max_group_length = P_NUM * used_core_num * C
    group_chunks = max_group_length // C
    max_groups = (seq_len + max_group_length - 1) // max_group_length

    @T.prim_func
    def main(
        query: T.Tensor((seq_len, nk, dk), "bfloat16"),            # 0
        key: T.Tensor((seq_len, nk, dk), "bfloat16"),              # 1
        value: T.Tensor((seq_len, nv, dv), "bfloat16"),            # 2
        beta: T.Tensor((seq_len, nv), "bfloat16"),                 # 3
        g: T.Tensor((seq_len, nv), "float32"),                     # 4
        initial_state: T.Tensor((batch, nv, dv, dk), "bfloat16"),  # 5
        actual_seq_lengths: T.Tensor((batch,), "int32"),           # 6
        out: T.Tensor((seq_len, nv, dv), "bfloat16"),              # 7
        final_state: T.Tensor((batch, nv, dv, dk), "bfloat16"),    # 8
        # ---- per-(Nv, maxGroupLength) GM workspace（AscendC 布局，复用于所有组）----
        ws_q: T.Tensor((nv, max_group_length, dk), "bfloat16"),        # 9
        ws_k: T.Tensor((nv, max_group_length, dk), "bfloat16"),        # 10
        ws_kbe: T.Tensor((nv, max_group_length, dk), "bfloat16"),      # 11
        ws_qprime: T.Tensor((nv, max_group_length, dk), "bfloat16"),   # 12
        ws_kg: T.Tensor((nv, max_group_length, dk), "bfloat16"),       # 13
        ws_vb: T.Tensor((nv, max_group_length, dv), "bfloat16"),       # 14
        ws_kbt: T.Tensor((nv, max_group_length, C), "bfloat16"),       # 15
        ws_qkt: T.Tensor((nv, max_group_length, C), "bfloat16"),       # 16
        ws_x: T.Tensor((nv, max_group_length, C), "bfloat16"),         # 17
        ws_v_inner: T.Tensor((nv, max_group_length, dv), "bfloat16"),  # 18
        ws_k_cumdecay: T.Tensor((nv, max_group_length, dk), "bfloat16"),  # 19
        ws_g_exp: T.Tensor((nv, max_group_length), "float32"),        # 20
        ws_g_exp_neg: T.Tensor((nv, max_group_length), "float32"),    # 21
        ws_attn_c: T.Tensor((nv, max_group_length, C), "bfloat16"),   # 22
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # ---- 入口全核同步（对应 AscendC Process 开头的 SyncAll<false>()）----
            # 所有核就绪后才开始 stage1 的跨核 GM 读写，避免首轮竞态。
            # 必须用 sync_all（核间 SyncAll），不能用 barrier_all（那只是本核 PipeBarrier）。
            T.sync_all()
            # ---- 所有核协同跑 batch → chunk-group 循环（AscendC Process）----
            # 循环层次与 AscendC 完全对应：for bid → for pos(step=maxGroupLength) →
            # 直接 RunStage1(cg) → SyncAll → RunStage2(cg, curState, finalState) →
            # SyncAll → RunStage3(cg) → SyncAll。任务分片（stage 内按 coreIdx 均分）
            # 封装在各 run_stageN 内部（对齐 stageN.h 的 Process）。
            for bid in T.serial(batch):
                # seq offset of this batch (start row in TND) = AscendC seqStart
                base = T.alloc_var("int32", init=0)
                base = 0
                for bbz in T.serial(batch):
                    if bbz < bid:
                        base = base + actual_seq_lengths[bbz]
                Lb = actual_seq_lengths[bid]          # AscendC length → seqEnd = base + Lb

                # ---- chunk-group 循环（step = maxGroupLength）----
                for gi in T.serial(max_groups):
                    gpos = base + gi * max_group_length
                    if gpos < base + Lb:               # AscendC `pos < seqEnd`
                        cg_len = T.if_then_else(base + Lb - gpos < max_group_length,
                                                base + Lb - gpos, max_group_length)
                        num_chunk = (cg_len + C - 1) // C

                        # AscendC: RunStage1(cg)
                        run_stage1(
                            cid, used_core_num, group_chunks,
                            nk, nv, dk, dv, C, scale,
                            gpos, cg_len, num_chunk,
                            query, key, value, beta, g,
                            ws_q, ws_k, ws_kbe, ws_qprime, ws_kg, ws_vb,
                            ws_kbt, ws_qkt, ws_x, ws_v_inner, ws_k_cumdecay,
                            ws_g_exp, ws_g_exp_neg,
                        )
                        T.sync_all()                   # SyncAll

                        # AscendC: RunStage2(cg, curState, finalState[b])
                        run_stage2(
                            cid, used_core_num, group_chunks,
                            nk, nv, dk, dv, C,
                            bid, gi, gpos, cg_len, num_chunk,
                            ws_k_cumdecay, ws_qprime, ws_v_inner, ws_kg,
                            initial_state, final_state, out,
                            ws_g_exp,
                        )
                        T.sync_all()                   # SyncAll

                        # AscendC: RunStage3(cg)
                        run_stage3(
                            cid, used_core_num, group_chunks,
                            nk, nv, dk, dv, C, scale,
                            gpos, cg_len, num_chunk,
                            out, ws_qkt, ws_v_inner, ws_g_exp, ws_g_exp_neg,
                            ws_attn_c,
                        )
                        T.sync_all()                   # SyncAll

    return main


def _host_run(query, key, value, beta, g, initial_state, actual_seq_lengths,
              kernel):
    """Host 侧：只传 7 个真实输入；out/final_state + 14 个 workspace 由
    tilelang 经 out_idx/workspace_idx 自动分配（零初始化），kernel 返回
    (out, final_state)。"""
    return kernel(
        query, key, value, beta, g, initial_state, actual_seq_lengths,
    )


def chunk_gated_delta_rule_host(batch, seq_len, nk, nv, dk, dv,
                                chunk_size=CHUNK_SIZE, scale=1.0):
    """Host wrapper：build 单 kernel 并 launch（骨架，细算 TODO(tile-level) 未落地）。"""
    C = int(chunk_size)
    kernel = chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv,
                                    chunk_size=C, scale=scale)

    def run(query, key, value, beta, g, initial_state, actual_seq_lengths):
        return _host_run(query, key, value, beta, g, initial_state,
                         actual_seq_lengths, kernel)

    return run
