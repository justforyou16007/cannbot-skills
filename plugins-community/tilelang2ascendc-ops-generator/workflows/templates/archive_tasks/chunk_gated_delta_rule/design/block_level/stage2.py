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

"""Block-level 设计：Stage2 — 跨 chunk 递推（C3 / V2 / V3）。

对应 AscendC `Stage2`（stage2.h）与 **tile-level 的 `stage2_ker`**（`design/tile_level/stage2.py`，
独立 jit kernel）。消费 Stage1 workspace（kCumDecay/vInner/qPrime/gCum/kg）
与当前状态 S（首组=initial_state，后续=上一 chunk 更新后的 S），对当前 chunk：

    C3  v_prime    = k_cumdecay @ S            （C×Dk @ Dk×Dv -> C×Dv）
        attn_inter = qPrime @ S               （C×Dk @ Dk×Dv -> C×Dv）-> 写 out
    V2  v_new      = v_inner - v_prime        （逐元素，v_inner 来自 Stage1）
    V3  S'         = g_last*S + kg^T @ v_new  （g_last=exp(g_cum[last])；kg 来自 Stage1）
                  -> 写 final_state[b,h]（块内 UB 状态更新）

说明：
- out 本阶段只写 attn_inter（历史状态对本 chunk 的贡献）；chunk 内贡献 attn_c@v_new
  由 Stage3 累加。
- 状态 S 更新为 `g_last*S + kg^T@v_new`，写回 final_state，并保留在块内 UB 供下一 chunk。
- C3 的 matmul 在 C 侧；V2/V3 的逐元素与状态更新在 V 侧；C/V 经 workspace/UB 交接。

## C/V 子阶段（块内）

    C3  v_prime = k_cumdecay @ S；attn_inter = qPrime @ S
    V2  v_new = v_inner - v_prime
    V3  S = g_last*S + kg^T @ v_new（kg^T @ v_new 是 (Dk,C)@(C,Dv)，也可放 C 侧 matmul）
"""

import tilelang.language as T


def stage2_ker(
    ws_gcumexp, ws_kcumdecay, ws_vinner, ws_qprime, ws_kg,
    state,            # (Dv, Dk) fp32 块内 UB，跨 chunk 携带
    valid, C, dk, dv,
    out_slice,        # out[s0:s0+valid, h, :]，写 attn_inter
    final_state_slice,  # final_state[b, h, :, :]，写更新后状态
):
    """当前 chunk 的跨 chunk 递推（依赖状态 S），更新 S 并写 out 的状态部分。"""
    # ---- C3：v_prime / attn_inter ----
    with T.Scope("C"):
        # TODO(tile-level): T.gemm(v_prime, k_cumdecay, state)
        #                    T.gemm(attn_inter, qPrime, state) -> out_slice
        pass

    # ---- V2：v_new = v_inner - v_prime ----
    with T.Scope("V"):
        # TODO(tile-level): v_new = v_inner - v_prime（逐元素）
        pass

    # ---- V3：S = g_last*S + kg^T @ v_new ----
    with T.Scope("V"):
        # TODO(tile-level): g_last = exp(ws_gcumexp[C-1])
        #                    state = g_last*state + kg^T @ v_new
        #                    T.copy(state 降型, final_state_slice)
        pass

    T.barrier_all()
