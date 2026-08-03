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

"""Tile-level 设计：Stage2 — 跨 chunk 递推（独立 jit kernel）。

Mirrors block_level/stage2.py / AscendC `Stage2`.  One standalone kernel that,
for chunk column `ci`, consumes the Stage1 workspace frag + the current state S
(total_units, Dv, Dk) for every (batch, value-head) unit in parallel:

    C3  v_prime    = k_cumdecay @ S^T            -> ws_v_prime
        attn_inter = qPrime @ S^T                -> out (historical part)
    V2  v_new      = v_inner - v_prime           -> ws_v_new
    C3' dstate     = v_new^T @ kg
    V3  out        <- attn_inter ; S = g_last*S + dstate
                      -> S (in place) + final_state[b,h]

A standalone kernel launch is a hard host-level barrier: S read here is the
updated S from the previous chunk (host delivers it), so no cross-chunk
cross-flag desync is possible.  All matmuls cube (T.gemm_v0) in T.Scope("C"),
vector ops in T.Scope("V"); C<->V handoff via GM workspace + cross-flag (IDs 3..5).
"""

import tilelang
import tilelang.language as T

from design.chunk_gated_delta_rule_common import (
    CHUNK_SIZE,
    NUM_PHYSICAL_CORES,
    pass_configs,
)


def key_head(h, nk, nv):
    return (h * nk) // nv


@tilelang.jit(
    out_idx=[],
    workspace_idx=[],
    pass_configs=pass_configs,
)
def stage2_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=CHUNK_SIZE):
    C = chunk_size
    total_units = batch * nv
    used_core_num = min(NUM_PHYSICAL_CORES, max(1, total_units))
    tasks_per_core = T.ceildiv(total_units, used_core_num)

    @T.prim_func
    def main(
        ws_k_cumdecay: T.Tensor((total_units, C, dk), "bfloat16"),   # 0
        ws_qprime: T.Tensor((total_units, C, dk), "bfloat16"),       # 1
        ws_v_inner: T.Tensor((total_units, C, dv), "bfloat16"),      # 2
        ws_kg: T.Tensor((total_units, C, dk), "bfloat16"),           # 3
        S: T.Tensor((total_units, dv, dk), "bfloat16"),              # 4
        out: T.Tensor((seq_len, nv, dv), "bfloat16"),                # 5
        final_state: T.Tensor((batch, nv, dv, dk), "bfloat16"),      # 6
        ws_v_prime: T.Tensor((total_units, C, dv), "bfloat16"),      # 7
        ws_attn_inter: T.Tensor((total_units, C, dv), "bfloat16"),   # 8
        ws_v_new: T.Tensor((total_units, C, dv), "bfloat16"),        # 9
        ws_dstate: T.Tensor((total_units, dk, dv), "bfloat16"),      # 10
        ws_g_exp: T.Tensor((total_units, C), "float32"),             # 11
        actual_seq_lengths: T.Tensor((batch,), "int32"),             # 12
        ci: T.Tensor((1,), "int32"),                                 # 13  current chunk column
        bb: T.Tensor((1,), "int32"),                                 # 14  current batch
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            for local_idx in T.serial(tasks_per_core):
                unit = cid * tasks_per_core + local_idx
                if unit < total_units:
                    b = unit // nv
                    h = unit % nv
                    base = T.alloc_ub((1,), "int32")
                    base[0] = 0
                    for bbz in T.serial(b):
                        base[0] += actual_seq_lengths[bbz]
                    Lb = actual_seq_lengths[b]
                    if b == bb[0] and ci[0] * C < Lb:
                        s0 = base[0] + ci[0] * C
                        valid = T.if_then_else(Lb - ci[0] * C < C, Lb - ci[0] * C, C)

                        # ---- C3 : v_prime / attn_inter (cube) ----
                        with T.Scope("C"):
                            kcd_l1 = T.alloc_L1((C, dk), "bfloat16")
                            sp_s_l1 = T.alloc_L1((dv, dk), "bfloat16")
                            T.copy(ws_k_cumdecay[unit, 0, 0], kcd_l1)
                            T.copy(S[unit, 0, 0], sp_s_l1)
                            vp_l0 = T.alloc_L0C((C, dv), "float32")
                            T.gemm_v0(kcd_l1, sp_s_l1, vp_l0, transpose_B=True, init=True)
                            T.copy(vp_l0, ws_v_prime[unit, 0, 0])
                            q_l1 = T.alloc_L1((C, dk), "bfloat16")
                            ai_l0 = T.alloc_L0C((C, dv), "float32")
                            T.copy(ws_qprime[unit, 0, 0], q_l1)
                            T.gemm_v0(q_l1, sp_s_l1, ai_l0, transpose_B=True, init=True)
                            T.copy(ai_l0, ws_attn_inter[unit, 0, 0])
                            T.set_cross_flag("FIX", 3)          # C -> V

                        # ---- V2 : v_new = v_inner - v_prime ----
                        with T.Scope("V"):
                            T.wait_cross_flag(3)
                            vinner_bf = T.alloc_ub((C, dv), "bfloat16")
                            vinner_ub = T.alloc_ub((C, dv), "float32")
                            vprime_bf = T.alloc_ub((C, dv), "bfloat16")
                            vprime_ub = T.alloc_ub((C, dv), "float32")
                            vnew_ub = T.alloc_ub((C, dv), "float32")
                            T.copy(ws_v_inner[unit, 0, 0], vinner_bf)
                            T.copy(vinner_bf, vinner_ub)
                            T.copy(ws_v_prime[unit, 0, 0], vprime_bf)
                            T.copy(vprime_bf, vprime_ub)
                            for t in T.serial(C):
                                for d in T.serial(dv):
                                    vnew_ub[t, d] = vinner_ub[t, d] - vprime_ub[t, d]
                            vnew_bf = T.alloc_ub((C, dv), "bfloat16")
                            T.copy(vnew_ub, vnew_bf)
                            T.copy(vnew_bf, ws_v_new[unit, 0, 0])
                            T.set_cross_flag("MTE3", 4)         # V -> C

                        # ---- C3' : dstate = v_new^T @ kg ----
                        with T.Scope("C"):
                            T.wait_cross_flag(4)
                            vn_l1 = T.alloc_L1((C, dv), "bfloat16")
                            kg_l1 = T.alloc_L1((C, dk), "bfloat16")
                            ds_l0 = T.alloc_L0C((dk, dv), "float32")
                            T.copy(ws_v_new[unit, 0, 0], vn_l1)
                            T.copy(ws_kg[unit, 0, 0], kg_l1)
                            T.gemm_v0(kg_l1, vn_l1, ds_l0, transpose_A=True, init=True)
                            T.copy(ds_l0, ws_dstate[unit, 0, 0])
                            T.set_cross_flag("FIX", 5)         # C -> V

                        # ---- V3 : out <- attn_inter ; S = g_last*S + dstate ----
                        with T.Scope("V"):
                            T.wait_cross_flag(5)
                            ai_bf = T.alloc_ub((C, dv), "bfloat16")
                            out_ub = T.alloc_ub((C, dv), "float32")
                            out_bf = T.alloc_ub((C, dv), "bfloat16")
                            T.copy(ws_attn_inter[unit, 0, 0], ai_bf)
                            T.copy(ai_bf, out_ub)
                            T.copy(out_ub, out_bf)
                            T.copy(out_bf[0:valid, 0:dv], out[s0:s0 + valid, h, 0:dv])
                            ds_bf = T.alloc_ub((dk, dv), "bfloat16")
                            ds_ub = T.alloc_ub((dk, dv), "float32")
                            T.copy(ws_dstate[unit, 0, 0], ds_bf)
                            T.copy(ds_bf, ds_ub)
                            g_exp = T.alloc_ub((C,), "float32")
                            T.copy(ws_g_exp[unit, 0], g_exp)
                            state_bf = T.alloc_ub((dv, dk), "bfloat16")
                            state = T.alloc_ub((dv, dk), "float32")
                            T.copy(S[unit, 0, 0], state_bf)
                            T.copy(state_bf, state)
                            gl = g_exp[C - 1]
                            for d in T.serial(dv):
                                for k in T.serial(dk):
                                    state[d, k] = gl * state[d, k] + ds_ub[k, d]
                            T.copy(state, state_bf)
                            T.copy(state_bf, S[unit, 0, 0])
                            T.copy(state_bf, final_state[b, h, :, :])

    return main
