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

"""Block-level 设计：Stage2 — 跨 chunk 递推（独立 jit kernel，对应 tile-level `stage2_ker`）。

对应 AscendC `Stage2`（stage2.h）。消费 Stage1 workspace（kCumDecay/vInner/qPrime/gCum/kg）
与当前状态 S，对当前 chunk 计算依赖状态的部分，更新 S。满足 `BlockLevelDesign-quality-checklist.md`。

## C/V 子阶段（checklist #4）

    V0'  g_last = exp(g_cum[C-1])；S ← g_last·S            （先衰减，向量）
    C3   v_prime  = k_cumdecay @ S^T                        （cube，T.gemm_v0）
         attn_inter = qPrime @ S^T    → 写 out 的历史贡献   （cube）
    V2   v_new = v_inner - v_prime                          （逐元素）
    V3   S ← S + v_new^T @ kg   → 写 S/final_state          （外积累加，可 cube）

## workspace 读/写（checklist #6）

    读：ws_gcumexp / ws_kcumdecay / ws_vinner / ws_qprime / ws_kg（stage1 产）
    写：out（attn_inter 部分）、S / final_state

## 块内遍历（checklist #2）

    with T.Kernel(used_core_num):
        for unit in 分给本 kernel 的 (b, head) 集合:
            if b == 当前 batch 且 属于本 chunk:
                # 每单元：V0'(衰减 S) -> C3 -> V2 -> V3
"""

import tilelang
import tilelang.language as T

from design.chunk_gated_delta_rule_common import CHUNK_SIZE, pass_configs


@tilelang.jit(out_idx=[], workspace_idx=[], pass_configs=pass_configs)
def stage2_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=CHUNK_SIZE):
    C = chunk_size
    total_units = batch * nv
    used_core_num = min(20, total_units)

    @T.prim_func
    def main(
        ws_gcumexp: T.Tensor((C,), "float32"),
        ws_kcumdecay: T.Tensor((C, dk), "bfloat16"),
        ws_vinner: T.Tensor((C, dv), "bfloat16"),
        ws_qprime: T.Tensor((C, dk), "bfloat16"),
        ws_kg: T.Tensor((C, dk), "bfloat16"),
        S: T.Tensor((nv, dv, dk), "float32"),
        ci: T.Tensor((1,), "int32"),
        bb: T.Tensor((1,), "int32"),
        Lb: T.Tensor((1,), "int32"),
        out: T.Tensor((seq_len, nv, dv), "bfloat16"),
        final_state: T.Tensor((nv, dv, dk), "bfloat16"),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # TODO(tile-level): unit 分配；`h=unit%nv, kh=(h*nk)//nv, s0=base[b]+ci[0]*C`
            #                   `valid=T.if_then_else(...)`；`if b==bb[0]:`；curState=S[h] 或初值

            # V0'：g_last = exp(ws_gcumexp[C-1])；S[h] ← g_last·S[h]
            with T.Scope("V"):
                # TODO(tile-level): T.tile.mul(S[h], S[h], g_last)
                pass

            # C3：v_prime = k_cumdecay @ S[h]^T；attn_inter = qPrime @ S[h]^T（写 out）
            with T.Scope("C"):
                # TODO(tile-level):
                #   T.gemm_v0(ws_kcumdecay, S[h], v_prime, transpose_B=True)
                #   T.gemm_v0(ws_qprime, S[h], attn_inter, transpose_B=True)
                #   T.copy 降型(attn_inter) -> out[s0:s0+valid, h, 0:dv]
                pass

            # V2：v_new = v_inner - v_prime
            with T.Scope("V"):
                # TODO(tile-level): T.tile.sub(v_new, ws_vinner, v_prime)
                pass

            # V3：S[h] ← S[h] + v_new^T @ kg（外积更新，可放 C 侧 T.gemm_v0 或 V 侧外积）
            with T.Scope("V"):
                # TODO(tile-level):
                #   dstate = v_new^T @ ws_kg（(Dv,C)@(C,Dk)->(Dv,Dk)）
                #   S[h] = T.tile.add(S[h], dstate)；写 final_state[h]（降型）
                pass

    return main
