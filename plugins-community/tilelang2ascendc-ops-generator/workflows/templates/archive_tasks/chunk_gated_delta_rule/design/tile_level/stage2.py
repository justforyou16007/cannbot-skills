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

"""Tile-level Stage2 - 跨 chunk 递推（单 kernel 内的函数，非独立 jit）。

Mirrors block_level/stage2.py / AscendC `Stage2`, but *inlined* into the single
`chunk_gated_delta_rule` prim_func (no separate kernel).  For one `unit` and one
chunk it consumes the Stage1 workspace frag + the carried state `S` (GM
`ws_S[unit]`) and computes:

    C3   v_prime    = k_cumdecay @ S^T            -> ws_v_prime   (AIC / cube)
         attn_inter = qPrime @ S^T                -> ws_attn_inter (AIC / cube)
    V2   v_new      = v_inner - v_prime           -> ws_v_new     (AIV / vector)
    C3'  dstate     = v_new^T @ kg                -> ws_dstate     (AIC / cube)
    V3   g_last     = exp(g_cum[C-1])             (AIV / scalar)
         S = g_last*S + dstate                    -> ws_S[unit] (carried)   (AIV / vector)
         out        <- attn_inter ; final_state[b,h] <- S

Correspondence to AscendC `Stage2` (chunk_gated_delta_rule_stage2.h):
    AIC (cube, T.Scope("C")) : CalVPrime / CalAttnInter (C3), CalStateNew (C3') --
        v_prime/attn_inter/dstate 均为 T.gemm_v0; rhs 是 S 转置 read, trans = 真 S 布局 (Dv,Dk).
    AIV (vector, T.Scope("V")) : CalGCumExp 取 g_last(Muls: S←g_last·S) + V2 减法 + V3 累加.
        S = g_last*S + dstate 拆为 T.tile.mul(S, S, gl) + T.tile.add(S, S, dstate)
        (AscendC 把 g_last·S 放 AIV、+dstate 由 AIC IterateAll accumulate 折入 finalState).
    同步: C3 产物经 GM + cross_flag 3 (=AIC CrossCoreSetFlag(0x2)/AIV WaitFlag(0x2)),
          V2 产物经 cross_flag 4 (=AIV SetFlag(0x5)/AIC WaitFlag(0x5)),
          C3' 产物经 cross_flag 5 (=AIC SetFlag(0x4)/AIV WaitFlag(0x4)).

State transport (学 AscendC `curState`): S lives in the persistent GM tensor
`ws_S`, initialised once per unit from `initial_state[b,h]` before the chunk
loop and read/written here across chunks.  Cross-chunk dependency is ordered by
the enclosing kernel's chunk loop; a `T.barrier_all()` before/after guarantees
stage-1 product completeness and that the updated S isn't read early.

UB reuse (学 AscendC `tmpBuff_`): every `T.alloc_ub` temporal is declared inside
this function, so the planner can overlap them with stage1/3 temporaries on the
same physical UB (per-stage peak, not the union).

C<->V handoff per chunk via GM + cross-flag IDs 3,4,5, re-armed each iteration.
"""


import tilelang.language as T


@T.macro(hygienic=False)
def run_stage2(nk, nv, dk, dv, C,
               unit, b, h, s0, valid,
               ws_k_cumdecay, ws_qprime, ws_v_inner, ws_kg,
               ws_S, out, final_state,
               ws_v_prime, ws_attn_inter, ws_v_new, ws_dstate, ws_g_exp):
    # ---- C3 : v_prime / attn_inter (cube) ----
    with T.Scope("C"):
        kcd_l1 = T.alloc_L1((C, dk), "bfloat16")
        sp_s_l1 = T.alloc_L1((dv, dk), "bfloat16")
        T.copy(ws_k_cumdecay[unit, 0, 0], kcd_l1)
        T.copy(ws_S[unit, 0, 0], sp_s_l1)
        vp_l0 = T.alloc_L0C((C, dv), "float32")
        T.gemm_v0(kcd_l1, sp_s_l1, vp_l0, transpose_B=True, init=True)
        T.copy(vp_l0, ws_v_prime[unit, 0, 0])
        qprime_l1 = T.alloc_L1((C, dk), "bfloat16")
        ai_l0 = T.alloc_L0C((C, dv), "float32")
        T.copy(ws_qprime[unit, 0, 0], qprime_l1)
        T.gemm_v0(qprime_l1, sp_s_l1, ai_l0, transpose_B=True, init=True)
        T.copy(ai_l0, ws_attn_inter[unit, 0, 0])
        T.set_cross_flag("FIX", 3)          # C -> V

    # ---- V2 : v_new = v_inner - v_prime (AIV vector, 学 AscendC FIX 子块) ----
    with T.Scope("V"):
        T.wait_cross_flag(3)
        vinner_bf = T.alloc_ub((C, dv), "bfloat16")
        vprime_bf = T.alloc_ub((C, dv), "bfloat16")
        vinner_ub = T.alloc_ub((C, dv), "float32")
        vprime_ub = T.alloc_ub((C, dv), "float32")
        vnew_ub = T.alloc_ub((C, dv), "float32")
        # v_inner (bf16 via two-stage T.copy)
        T.copy(ws_v_inner[unit, 0, 0], vinner_bf)
        T.copy(vinner_bf, vinner_ub)
        # v_prime (bf16 via two-stage T.copy)
        T.copy(ws_v_prime[unit, 0, 0], vprime_bf)
        T.copy(vprime_bf, vprime_ub)
        # v_new = v_inner - v_prime  (vectorized, no scalar loop)
        T.tile.sub(vnew_ub, vinner_ub, vprime_ub)
        vnew_bf = T.alloc_ub((C, dv), "bfloat16")
        T.copy(vnew_ub, vnew_bf)
        T.copy(vnew_bf, ws_v_new[unit, 0, 0])
        T.set_cross_flag("MTE3", 4)         # V -> C

    # ---- C3' : dstate = v_new^T @ kg ----
    with T.Scope("C"):
        T.wait_cross_flag(4)
        vn_l1_s2 = T.alloc_L1((C, dv), "bfloat16")
        kg_l1 = T.alloc_L1((C, dk), "bfloat16")
        T.copy(ws_v_new[unit, 0, 0], vn_l1_s2)
        T.copy(ws_kg[unit, 0, 0], kg_l1)
        # dstate = v_new^T @ kg  ->  (dv, dk), i.e. state layout directly
        # (was (dk,dv) requiring a transpose read in the update below).
        ds_l0 = T.alloc_L0C((dv, dk), "float32")
        T.gemm_v0(vn_l1_s2, kg_l1, ds_l0, transpose_A=True, init=True)
        T.copy(ds_l0, ws_dstate[unit, 0, 0])
        T.set_cross_flag("FIX", 5)         # C -> V

    # ---- V3a : out <- attn_inter ----
    with T.Scope("V"):
        T.wait_cross_flag(5)
        ai_bf = T.alloc_ub((C, dv), "bfloat16")
        out_ub_s2 = T.alloc_ub((C, dv), "float32")
        out_bf_s2 = T.alloc_ub((C, dv), "bfloat16")
        T.copy(ws_attn_inter[unit, 0, 0], ai_bf)
        T.copy(ai_bf, out_ub_s2)
        T.copy(out_ub_s2, out_bf_s2)
        T.copy(out_bf_s2[0:valid, 0:dv], out[s0:s0 + valid, h, 0:dv])
    T.barrier_all()
    # ---- V3b : S = g_last*S + dstate (separate scope so the planner frees the
    # out-write buffers before aliasing these) ----
    with T.Scope("V"):
        # g_last = exp(g_cum[C-1]) (学 AscendC CalGCumExp 取最后一位)
        g_exp_s2 = T.alloc_ub((C,), "float32")
        T.copy(ws_g_exp[unit, 0], g_exp_s2)
        gl = g_exp_s2[C - 1]
        state_bf_s2 = T.alloc_ub((dv, dk), "bfloat16")
        state_s2 = T.alloc_ub((dv, dk), "float32")
        T.copy(ws_S[unit, 0, 0], state_bf_s2)
        T.copy(state_bf_s2, state_s2)
        # S = g_last * S   (vector scale, 学 AscendC CalGCumExp Muls)
        T.tile.mul(state_s2, state_s2, gl)
        ds_bf = T.alloc_ub((dv, dk), "bfloat16")
        ds_ub = T.alloc_ub((dv, dk), "float32")
        T.copy(ws_dstate[unit, 0, 0], ds_bf)
        T.copy(ds_bf, ds_ub)
        # S = S + dstate   (vector accumulate, 学 AscendC CalStateNew 累加)
        T.tile.add(state_s2, state_s2, ds_ub)
        T.copy(state_s2, state_bf_s2)
        T.copy(state_bf_s2, ws_S[unit, 0, 0])
        T.copy(state_bf_s2, final_state[b, h, :, :])
