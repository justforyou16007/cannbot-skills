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

"""Tile-level TileLang design — SINGLE kernel for the Chunked Gated Delta Rule,
mirrored on `design/block_level/` and the AscendC `CGDR::Process` (one kernel;
batch → chunk-group → RunStage1/2/3 + SyncAll).  **chunk-group 协作模型**（对齐
AscendC，不再用 unit 私有链）。

    chunk_gated_delta_rule.py   entry / orchestration   <- @tilelang.jit, ONE prim_func
    stage1.py                   within-chunk precompute <- run_stage1 (inlined function)
    stage2.py                   cross-chunk recurrence  <- run_stage2 (inlined function)
    stage3.py                   within-chunk accumulate <- run_stage3 (inlined function)

There is exactly ONE `@tilelang.jit` kernel (here) and ONE `T.Kernel(NUM_PHYSICAL_CORES)`.
stage1/2/3 are plain Python functions inlined into the single prim_func.  Inside the
batch → chunk-group loop, each valid group runs stage1 → barrier → stage2 → barrier →
stage3 → barrier once.  **任务分片（stage 内按 coreIdx 均分）封装在各 run_stageN 内部**，
对齐 stageN.h 的 Process（stage1/3 融合均分，stage2 按 nv 切核 + 核内 cId 串行）。

## 并行模型（对齐 AscendC Process，不再 unit 私有链）

    blockDim = aiCoreNum ≈ NUM_PHYSICAL_CORES
    maxGroupLength = P_NUM(2) × aiCoreNum × chunkSize(64)
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

## 状态与 workspace

Workspace 为 AscendC 布局 `(Nv, maxGroupLength, ...)`，本 chunk 行偏移
`row = cgId * C`（stage1.h `SetChunkOffset` 的 `chunkRowBase`）。状态 S 直接走 GM
`initial_state`/`final_state`（不再有 per-unit ws_S），组间经 `final_state` 串行；
Stage2 内 `curState = (gi==0 && cId==0) ? initial_state[b,nvId] : final_state[b,nvId]`。

## C/V 分工与同步

    - 阶段间：`T.sync_all()` ↔ AscendC `SyncAll<false>()`（每 chunk-group 阶段边界各一次）。
    - 阶段内 AIC↔AIV：stage1/2/3 用 `T.Scope("C")`/`T.Scope("V")` 分工，GM workspace
      交接 + `T.set_cross_flag`/`T.wait_cross_flag`（信号编号 0..7 与 AscendC stage 内
      event id 一一对应）。每 chunk 迭代尾部由阶段间 sync_all 排干所有 pipeline，
      cross-flag 干净 re-arm。

## 数值语义（与 unit 版本一致）

所有 matmul 为 cube `T.gemm_v0`；向量 op 在 UB；每个 bf16 边界走两段式 `T.copy`；
beta/g 逐标量加载；尾 chunk `valid = min(C, cg_len - cgId*C)`、尾 group
`cg_len = min(maxGroupLength, base+Lb-gpos)`。只改并行/布局/状态来源，不改数学。
"""

import tilelang
import tilelang.language as T
import torch

from design.chunk_gated_delta_rule_common import (
    CHUNK_SIZE,
    NUM_PHYSICAL_CORES,
    P_NUM,
    pass_configs,
)
from design.tile_level.stage1 import run_stage1
from design.tile_level.stage2 import run_stage2
from design.tile_level.stage3 import run_stage3


@tilelang.jit(
    # 手动传参：out/final_state + 全部 ws_xxx 由 host 分配并显式传入。
    # 不用 out_idx/workspace_idx：tilelang 的自动分配路径实测有首次运行竞态
    # （同一 prim_func，out_idx=[7,8] 自动分配 → trial0 错、trial1+ 对；手动传参稳定）。
    out_idx=[],
    workspace_idx=[],
    pass_configs=pass_configs,
)
def chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv,
                           chunk_size=CHUNK_SIZE, scale=1.0):
    """Build the SINGLE-kernel runner.  Mirrors `CGDR::Process` (chunk-group 协作)."""
    C = chunk_size
    # F = C//2 作为 Python 常量传进 stage1 宏（宏体内写 `F = C // 2` 会被 TVM script
    # 解析成自由 Var，导致 (F,F) buffer shape 非常量 → LowerTileOp segfault）。
    F = C // 2
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
            # 注意必须用 sync_all（核间 SyncAll），不能用 barrier_all（那只是本核 PipeBarrier）。
            T.sync_all()

            # ---- 所有核协同跑 batch → chunk-group 循环（AscendC Process）----
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
                    if gpos < base + Lb:              # AscendC `pos < seqEnd`
                        cg_len = T.if_then_else(base + Lb - gpos < max_group_length,
                                                base + Lb - gpos, max_group_length)
                        num_chunk = (cg_len + C - 1) // C

                        # AscendC: RunStage1(cg)
                        run_stage1(
                            cid, used_core_num, group_chunks,
                            nk, nv, dk, dv, C, F, scale,
                            gpos, cg_len, num_chunk,
                            query, key, value, beta, g,
                            ws_q, ws_k, ws_kbe, ws_qprime, ws_kg, ws_vb,
                            ws_kbt, ws_qkt, ws_x, ws_v_inner, ws_k_cumdecay,
                            ws_g_exp, ws_g_exp_neg,
                        )
                        T.sync_all()                   # 核间 SyncAll（对应 AscendC；barrier_all 只是本核 PipeBarrier）

                        # AscendC: RunStage2(cg, curState, finalState[b])
                        run_stage2(
                            cid, used_core_num, group_chunks,
                            nk, nv, dk, dv, C,
                            bid, gi, gpos, cg_len, num_chunk,
                            ws_k_cumdecay, ws_qprime, ws_v_inner, ws_kg,
                            initial_state, final_state, out,
                            ws_g_exp,
                        )
                        T.sync_all()                   # 核间 SyncAll（对应 AscendC；barrier_all 只是本核 PipeBarrier）

                        # AscendC: RunStage3(cg)
                        run_stage3(
                            cid, used_core_num, group_chunks,
                            nk, nv, dk, dv, C, scale,
                            gpos, cg_len, num_chunk,
                            out, ws_qkt, ws_v_inner, ws_g_exp, ws_g_exp_neg,
                            ws_attn_c,
                        )
                        T.sync_all()                   # 核间 SyncAll（对应 AscendC；barrier_all 只是本核 PipeBarrier）

    return main


def _host_run(query, key, value, beta, g, initial_state, actual_seq_lengths,
              kernel):
    """Host 侧手动分配全部 workspace/out/final_state 并传参（不用 tilelang 自动分配）：
    tilelang 的 out_idx/workspace_idx 自动分配路径实测有首次运行竞态（trial0 错、
    trial1+ 对），手动传参则稳定。参数顺序与 main() 声明一一对应。"""
    dev = query.device
    batch = initial_state.shape[0]
    seq_len = query.shape[0]
    nv = value.shape[1]
    dk = query.shape[2]
    dv = value.shape[2]
    C = CHUNK_SIZE
    mgl = P_NUM * NUM_PHYSICAL_CORES * C
    out = torch.zeros((seq_len, nv, dv), dtype=torch.bfloat16, device=dev)
    final_state = torch.zeros((batch, nv, dv, dk), dtype=torch.bfloat16, device=dev)

    def ws(sh, dt): return torch.zeros(sh, dtype=dt, device=dev)

    # 9-14
    w_q = ws((nv, mgl, dk), torch.bfloat16); w_k = ws((nv, mgl, dk), torch.bfloat16)
    w_kbe = ws((nv, mgl, dk), torch.bfloat16); w_qprime = ws((nv, mgl, dk), torch.bfloat16)
    w_kg = ws((nv, mgl, dk), torch.bfloat16); w_vb = ws((nv, mgl, dv), torch.bfloat16)
    # 15-19
    w_kbt = ws((nv, mgl, C), torch.bfloat16); w_qkt = ws((nv, mgl, C), torch.bfloat16)
    w_x = ws((nv, mgl, C), torch.bfloat16)
    w_v_inner = ws((nv, mgl, dv), torch.bfloat16); w_k_cumdecay = ws((nv, mgl, dk), torch.bfloat16)
    # 20-22
    w_g_exp = ws((nv, mgl), torch.float32); w_g_exp_neg = ws((nv, mgl), torch.float32)
    w_attn_c = ws((nv, mgl, C), torch.bfloat16)

    kernel(query, key, value, beta, g, initial_state, actual_seq_lengths,
           out, final_state,
           w_q, w_k, w_kbe, w_qprime, w_kg, w_vb,
           w_kbt, w_qkt, w_x, w_v_inner, w_k_cumdecay,
           w_g_exp, w_g_exp_neg, w_attn_c)
    return out, final_state


def chunk_gated_delta_rule_host(batch, seq_len, nk, nv, dk, dv,
                                chunk_size=CHUNK_SIZE, scale=1.0):
    """Host wrapper：build 单 kernel 并 launch（chunk-group 协作细算）。"""
    C = int(chunk_size)
    kernel = chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv,
                                    chunk_size=C, scale=scale)

    def run(query, key, value, beta, g, initial_state, actual_seq_lengths):
        return _host_run(query, key, value, beta, g, initial_state,
                         actual_seq_lengths, kernel)

    return run
