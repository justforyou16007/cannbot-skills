#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See License in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tile-level TileLang design (expert mode + cube) for Chunked Gated Delta Rule.

Cube/vector rework built on the gdn (Gated Delta Net) TileLang examples
`tilelang-ascend/examples/linear_attention_and_rnn/gdn/`:
  - gdn_chunk_h.py            single-kernel, per-chunk state-carrying recurrence;
                              T.gemm_v0 (cube) in T.Scope("C"), vector ops in
                              T.Scope("V"), GM-workspace handoff + set/wait_cross_flag.
  - gdn_chunk_scaled_dot_kkt  kb@k^T / q@k^T cube + decay/mask.
  - gdn_chunk_o.py            q@S (attn_inter) and attn_c@v cube.
  - gdn_solve_tril.py         (I-A)^{-1} forward substitution, reused for X.
  - gdn_chunk_cumsum.py       chunked prefix sum, reused for g_cum.

Layout TND (T,Nk,Dk)|(T,Nv,Dv)|(T,Nv).  Each (batch, value-head) unit independent;
state S (Dv,Dk) carried across the unit's chunks; value head h reads key head
(h*Nk)//Nv.  Chunk size C = 64.  All cube operands bfloat16 in L1.

Math (golden == model.py):
  within-chunk
    g_cum = cumsum(g);  decay[i][j] = exp(g_cum[i]-g_cum[j]) (j<=i)
    kb = k*beta ; vb = v*beta
    A   = -(kb@k^T)*decay   (strictly lower)
    X   = (I-A)^{-1}        (forward substitution, gdn_solve_tril style)
    v_inner = X@vb ;  k_cumdecay = X@(kb*exp(g_cum))
    qPrime = q*scale*exp(g_cum);  qkt = q@k^T
    attn_c = (qkt*decay).tril()*scale ;  kg = k*exp(g_last-g_cum)
  inter-chunk
    v_prime  = k_cumdecay@S^T ;  v_new = v_inner - v_prime
    attn_inter = qPrime@S^T
    out  = attn_inter + attn_c @ v_new
    S'   = exp(g_last)*S + kg^T @ v_new

Expert mode: only low-level primitives.  No T.parallel / T.Parallel /
T.pipeline / T.Pipelined.  All bf16 I/O via T.copy (two-stage up/down cast);
arithmetic in fp32 on UB; cube gemm operands bfloat16 in L1.
"""

import tilelang
import tilelang.language as T

from design.chunk_gated_delta_rule_common import (
    CHUNK_SIZE,
    NUM_PHYSICAL_CORES,
    pass_configs,
)


def key_head(h, nk, nv):
    """The value head *h* reads key head (h*Nk)//Nv (head grouping)."""
    return (h * nk) // nv


@tilelang.jit(out_idx=[7, 8], workspace_idx=[9, 10, 11, 12, 13, 14, 15, 16, 17,
                                            18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                                            28, 29],
              pass_configs=pass_configs)
def chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv,
                           chunk_size=CHUNK_SIZE, scale=1.0):
    C = chunk_size
    total_units = batch * nv
    used_core_num = min(NUM_PHYSICAL_CORES, max(1, total_units))
    tasks_per_core = T.ceildiv(total_units, used_core_num)
    max_chunks = (seq_len + C - 1) // C

    @T.prim_func
    def main(
        query: T.Tensor((seq_len, nk, dk), "bfloat16"),          # 0
        key: T.Tensor((seq_len, nk, dk), "bfloat16"),            # 1
        value: T.Tensor((seq_len, nv, dv), "bfloat16"),          # 2
        beta: T.Tensor((seq_len, nv), "bfloat16"),               # 3
        g: T.Tensor((seq_len, nv), "float32"),                   # 4
        initial_state: T.Tensor((batch, nv, dv, dk), "bfloat16"),  # 5
        actual_seq_lengths: T.Tensor((batch,), "int32"),         # 6
        out: T.Tensor((seq_len, nv, dv), "bfloat16"),            # 7
        final_state: T.Tensor((batch, nv, dv, dk), "bfloat16"),  # 8
        # ---- plain q (scaled) and k, written by V0 ----
        ws_q: T.Tensor((total_units, C, dk), "bfloat16"),        # 9
        ws_k: T.Tensor((total_units, C, dk), "bfloat16"),        # 10
        ws_kb: T.Tensor((total_units, C, dk), "bfloat16"),       # 11
        ws_kbe: T.Tensor((total_units, C, dk), "bfloat16"),      # 12
        ws_qprime: T.Tensor((total_units, C, dk), "bfloat16"),   # 13
        ws_kg: T.Tensor((total_units, C, dk), "bfloat16"),       # 14
        ws_vb: T.Tensor((total_units, C, dv), "bfloat16"),       # 15
        ws_kbt: T.Tensor((total_units, C, C), "bfloat16"),       # 16
        ws_qkt: T.Tensor((total_units, C, C), "bfloat16"),       # 17
        ws_x: T.Tensor((total_units, C, C), "bfloat16"),         # 18
        ws_attn_c: T.Tensor((total_units, C, C), "bfloat16"),    # 19
        ws_v_inner: T.Tensor((total_units, C, dv), "bfloat16"),  # 20
        ws_k_cumdecay: T.Tensor((total_units, C, dk), "bfloat16"),  # 21
        ws_attn_inter: T.Tensor((total_units, C, dv), "bfloat16"),  # 22
        ws_v_prime: T.Tensor((total_units, C, dv), "bfloat16"),  # 23
        ws_v_new: T.Tensor((total_units, C, dv), "bfloat16"),    # 24
        ws_out_cont: T.Tensor((total_units, C, dv), "bfloat16"),  # 25
        ws_dstate: T.Tensor((total_units, dk, dv), "bfloat16"),  # 26
        ws_state: T.Tensor((total_units, dv, dk), "bfloat16"),   # 27
        ws_g_exp: T.Tensor((total_units, C), "float32"),         # 28
        ws_g_exp_neg: T.Tensor((total_units, C), "float32"),     # 29
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # ---------------- allocations (declared once, reused) ----------------
            # L1 cube operands
            kb_l1 = T.alloc_L1((C, dk), "bfloat16")
            k_l1 = T.alloc_L1((C, dk), "bfloat16")
            q_l1 = T.alloc_L1((C, dk), "bfloat16")
            s_l1 = T.alloc_L1((dv, dk), "bfloat16")
            x_l1 = T.alloc_L1((C, C), "bfloat16")
            vb_l1 = T.alloc_L1((C, dv), "bfloat16")
            kbe_l1 = T.alloc_L1((C, dk), "bfloat16")
            kcd_l1 = T.alloc_L1((C, dk), "bfloat16")
            vp_s_l1 = T.alloc_L1((dv, dk), "bfloat16")
            ac_l1 = T.alloc_L1((C, C), "bfloat16")
            vn_l1 = T.alloc_L1((C, dv), "bfloat16")
            kg_l1 = T.alloc_L1((C, dk), "bfloat16")
            # L0C cube accumulators
            kbt_l0 = T.alloc_L0C((C, C), "float32")
            qkt_l0 = T.alloc_L0C((C, C), "float32")
            ai_l0 = T.alloc_L0C((C, dv), "float32")
            vi_l0 = T.alloc_L0C((C, dv), "float32")
            kd_l0 = T.alloc_L0C((C, dk), "float32")
            vp_l0 = T.alloc_L0C((C, dv), "float32")
            oc_l0 = T.alloc_L0C((C, dv), "float32")
            ds_l0 = T.alloc_L0C((dk, dv), "float32")

            for local_idx in T.serial(tasks_per_core):
                unit = cid * tasks_per_core + local_idx
                if unit < total_units:
                    b = unit // nv
                    h = unit % nv
                    kh = key_head(h, nk, nv)

                    base = T.alloc_ub((1,), "int32")
                    base[0] = 0
                    for bb in T.serial(b):
                        base[0] += actual_seq_lengths[bb]
                    Lb = actual_seq_lengths[b]

                    # V-side fp32 compute buffers (reused across chunks/phases)
                    q_buf = T.alloc_ub((C, dk), "bfloat16")
                    k_buf = T.alloc_ub((C, dk), "bfloat16")
                    v_buf = T.alloc_ub((C, dv), "bfloat16")
                    qc = T.alloc_ub((C, dk), "float32")
                    kc = T.alloc_ub((C, dk), "float32")
                    vc = T.alloc_ub((C, dv), "float32")
                    cb = T.alloc_ub((C, dk), "float32")
                    cbv = T.alloc_ub((C, dv), "float32")
                    temp_k = T.alloc_ub((C, dk), "bfloat16")
                    temp_v = T.alloc_ub((C, dv), "bfloat16")
                    kbe_f = T.alloc_ub((C, dk), "float32")
                    qprime_f = T.alloc_ub((C, dk), "float32")
                    kg_f = T.alloc_ub((C, dk), "float32")
                    bc = T.alloc_ub((C,), "float32")
                    gc = T.alloc_ub((C,), "float32")
                    g_cum = T.alloc_ub((C,), "float32")
                    g_exp = T.alloc_ub((C,), "float32")
                    g_neg = T.alloc_ub((C,), "float32")
                    g_exp_neg = T.alloc_ub((C,), "float32")
                    g_last_l = T.alloc_ub((1,), "float32")
                    beta_b16 = T.alloc_ub((1,), "bfloat16")
                    beta_f32 = T.alloc_ub((1,), "float32")
                    g_f32 = T.alloc_ub((1,), "float32")
                    # A / X / attn_c (C,C) fp32 + bf16 staging
                    a_ub = T.alloc_ub((C, C), "float32")
                    attn_ub = T.alloc_ub((C, C), "float32")
                    kbt_ub = T.alloc_ub((C, C), "float32")
                    qkt_ub = T.alloc_ub((C, C), "float32")
                    x_bf = T.alloc_ub((C, C), "bfloat16")
                    attn_bf = T.alloc_ub((C, C), "bfloat16")
                    kbt_bf = T.alloc_ub((C, C), "bfloat16")
                    qkt_bf = T.alloc_ub((C, C), "bfloat16")
                    acc1 = T.alloc_ub((1,), "float32")
                    # (C,Dv) fp32 + bf16 staging
                    vinner_ub = T.alloc_ub((C, dv), "float32")
                    vprime_ub = T.alloc_ub((C, dv), "float32")
                    vnew_ub = T.alloc_ub((C, dv), "float32")
                    ai_ub = T.alloc_ub((C, dv), "float32")
                    ac_ub = T.alloc_ub((C, dv), "float32")
                    out_ub = T.alloc_ub((C, dv), "float32")
                    vinner_bf = T.alloc_ub((C, dv), "bfloat16")
                    vprime_bf = T.alloc_ub((C, dv), "bfloat16")
                    vnew_bf = T.alloc_ub((C, dv), "bfloat16")
                    ai_bf = T.alloc_ub((C, dv), "bfloat16")
                    ac_bf = T.alloc_ub((C, dv), "bfloat16")
                    out_bf = T.alloc_ub((C, dv), "bfloat16")
                    ds_ub = T.alloc_ub((dk, dv), "float32")
                    ds_bf = T.alloc_ub((dk, dv), "bfloat16")
                    state = T.alloc_ub((dv, dk), "float32")
                    state_bf = T.alloc_ub((dv, dk), "bfloat16")

                    # init state from initial_state (bf16 -> fp32)
                    T.set_flag("mte2", "v", 1)
                    T.wait_flag("mte2", "v", 1)
                    T.tile.fill(state_bf, 0.0)
                    T.copy(initial_state[b, h, :, :], state_bf)
                    T.tile.fill(state, 0.0)
                    T.copy(state_bf, state)
                    T.barrier_all()

                    # ---------------- chunk loop ----------------
                    for ci in T.serial(max_chunks):
                        if ci * C < Lb:
                            s0 = base[0] + ci * C
                            valid = T.if_then_else(Lb - ci * C < C, Lb - ci * C, C)

                            # ============ V0 : elementwise prep -> workspace ============
                            with T.Scope("V"):
                                T.tile.fill(q_buf, 0.0)
                                T.copy(query[s0:s0 + valid, kh, 0:dk], q_buf)
                                T.tile.fill(k_buf, 0.0)
                                T.copy(key[s0:s0 + valid, kh, 0:dk], k_buf)
                                T.tile.fill(v_buf, 0.0)
                                T.copy(value[s0:s0 + valid, h, 0:dv], v_buf)
                                T.copy(q_buf, qc)
                                T.copy(k_buf, kc)
                                T.copy(v_buf, vc)
                                T.barrier_all()
                                # per-scalar beta/g
                                T.tile.fill(bc, 0.0)
                                T.tile.fill(gc, 0.0)
                                for t in T.serial(C):
                                    if t < valid:
                                        T.copy(beta[s0 + t, h:h + 1], beta_b16)
                                        T.copy(g[s0 + t, h:h + 1], g_f32)
                                        T.set_flag("mte2", "v", 1)
                                        T.wait_flag("mte2", "v", 1)
                                        T.copy(beta_b16, beta_f32)
                                        bc[t] = beta_f32[0]
                                        gc[t] = g_f32[0]
                                    else:
                                        bc[t] = 0.0
                                        gc[t] = 0.0
                                T.barrier_all()
                                # g_cum cumsum
                                g_cum[0] = gc[0]
                                for t in T.serial(1, C):
                                    g_cum[t] = g_cum[t - 1] + gc[t]
                                for t in T.serial(C):
                                    g_neg[t] = 0.0 - g_cum[t]
                                T.tile.exp(g_exp, g_cum)
                                T.tile.exp(g_exp_neg, g_neg)
                                # ws_q = q*scale ; ws_k = plain k
                                for t in T.serial(C):
                                    for d in T.serial(dk):
                                        cb[t, d] = qc[t, d] * scale
                                T.copy(cb, temp_k)
                                T.copy(temp_k, ws_q[unit, 0, 0])
                                for t in T.serial(C):
                                    for d in T.serial(dk):
                                        cb[t, d] = kc[t, d]
                                T.copy(cb, temp_k)
                                T.copy(temp_k, ws_k[unit, 0, 0])
                                # kb = k*beta
                                for t in T.serial(C):
                                    for d in T.serial(dk):
                                        cb[t, d] = kc[t, d] * bc[t]
                                T.copy(cb, temp_k)
                                T.copy(temp_k, ws_kb[unit, 0, 0])
                                # vb = v*beta
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        cbv[t, d] = vc[t, d] * bc[t]
                                T.copy(cbv, temp_v)
                                T.copy(temp_v, ws_vb[unit, 0, 0])
                                # kbe = kb*exp(g_cum)
                                for t in T.serial(C):
                                    for d in T.serial(dk):
                                        kbe_f[t, d] = kc[t, d] * bc[t] * g_exp[t]
                                T.copy(kbe_f, temp_k)
                                T.copy(temp_k, ws_kbe[unit, 0, 0])
                                # qPrime = q*scale*exp(g_cum)
                                for t in T.serial(C):
                                    for d in T.serial(dk):
                                        qprime_f[t, d] = qc[t, d] * scale * g_exp[t]
                                T.copy(qprime_f, temp_k)
                                T.copy(temp_k, ws_qprime[unit, 0, 0])
                                # kg = k*exp(g_last-g_cum)  (g_last = g_cum[C-1])
                                for t in T.serial(C):
                                    for d in T.serial(dk):
                                        kg_f[t, d] = kc[t, d] * g_exp[C - 1] * g_exp_neg[t]
                                T.copy(kg_f, temp_k)
                                T.copy(temp_k, ws_kg[unit, 0, 0])
                                T.copy(g_exp, ws_g_exp[unit, 0])
                                T.copy(g_exp_neg, ws_g_exp_neg[unit, 0])
                                # mirror state -> ws_state (bf16)
                                T.copy(state, state_bf)
                                T.copy(state_bf, ws_state[unit, 0, 0])
                                T.set_cross_flag("MTE3", 0)   # V0 -> C1

                            # ============ C1 : kbt / qkt / attn_inter ============
                            with T.Scope("C"):
                                T.wait_cross_flag(0)
                                T.copy(ws_kb[unit, 0, 0], kb_l1)
                                T.copy(ws_k[unit, 0, 0], k_l1)
                                T.gemm_v0(kb_l1, k_l1, kbt_l0, transpose_B=True, init=True)
                                T.copy(kbt_l0, ws_kbt[unit, 0, 0])
                                T.copy(ws_q[unit, 0, 0], q_l1)
                                T.gemm_v0(q_l1, k_l1, qkt_l0, transpose_B=True, init=True)
                                T.copy(qkt_l0, ws_qkt[unit, 0, 0])
                                T.copy(ws_qprime[unit, 0, 0], q_l1)
                                T.copy(ws_state[unit, 0, 0], s_l1)
                                T.gemm_v0(q_l1, s_l1, ai_l0, transpose_B=True, init=True)
                                T.copy(ai_l0, ws_attn_inter[unit, 0, 0])
                                T.set_cross_flag("FIX", 3)   # C1 -> V1

                            # ============ V1 : A / X solve / attn_c ============
                            with T.Scope("V"):
                                T.wait_cross_flag(3)
                                T.copy(ws_kbt[unit, 0, 0], kbt_bf)
                                T.copy(kbt_bf, kbt_ub)
                                T.copy(ws_qkt[unit, 0, 0], qkt_bf)
                                T.copy(qkt_bf, qkt_ub)
                                T.copy(ws_g_exp[unit, 0], g_exp)
                                T.copy(ws_g_exp_neg[unit, 0], g_exp_neg)
                                # A = strictly lower
                                for i in T.serial(C):
                                    for j in T.serial(C):
                                        if j < i:
                                            a_ub[i, j] = -(kbt_ub[i, j] * g_exp[i] * g_exp_neg[j])
                                        else:
                                            a_ub[i, j] = 0.0
                                # X forward substitution
                                for i in T.serial(C):
                                    for j in T.serial(C):
                                        attn_ub[i, j] = a_ub[i, j]
                                for i in range(1, C):
                                    for j in T.serial(i):
                                        acc1[0] = 0.0
                                        for jp in T.serial(i):
                                            acc1[0] += a_ub[i, jp] * attn_ub[jp, j]
                                        attn_ub[i, j] = a_ub[i, j] + acc1[0]
                                for i in T.serial(C):
                                    attn_ub[i, i] = 1.0
                                T.copy(attn_ub, x_bf)
                                T.copy(x_bf, ws_x[unit, 0, 0])
                                # attn_c = (qkt*decay).tril()  (q already scaled in ws_q)
                                for i in T.serial(C):
                                    for j in T.serial(C):
                                        if j <= i:
                                            attn_ub[i, j] = qkt_ub[i, j] * g_exp[i] * g_exp_neg[j]
                                        else:
                                            attn_ub[i, j] = 0.0
                                T.copy(attn_ub, attn_bf)
                                T.copy(attn_bf, ws_attn_c[unit, 0, 0])
                                T.set_cross_flag("MTE3", 6)   # V1 -> C2

                            # ============ C2 : v_inner / k_cumdecay / v_prime ============
                            with T.Scope("C"):
                                T.wait_cross_flag(6)
                                T.copy(ws_x[unit, 0, 0], x_l1)
                                T.copy(ws_vb[unit, 0, 0], vb_l1)
                                T.gemm_v0(x_l1, vb_l1, vi_l0, init=True)
                                T.copy(vi_l0, ws_v_inner[unit, 0, 0])
                                T.copy(ws_kbe[unit, 0, 0], kbe_l1)
                                T.gemm_v0(x_l1, kbe_l1, kd_l0, init=True)
                                T.copy(kd_l0, ws_k_cumdecay[unit, 0, 0])
                                # v_prime = k_cumdecay @ S^T
                                T.copy(ws_k_cumdecay[unit, 0, 0], kcd_l1)
                                T.copy(ws_state[unit, 0, 0], vp_s_l1)
                                T.gemm_v0(kcd_l1, vp_s_l1, vp_l0, transpose_B=True, init=True)
                                T.copy(vp_l0, ws_v_prime[unit, 0, 0])
                                T.set_cross_flag("FIX", 9)   # C2 -> V2

                            # ============ V2 : v_new = v_inner - v_prime ============
                            with T.Scope("V"):
                                T.wait_cross_flag(9)
                                T.copy(ws_v_inner[unit, 0, 0], vinner_bf)
                                T.copy(vinner_bf, vinner_ub)
                                T.copy(ws_v_prime[unit, 0, 0], vprime_bf)
                                T.copy(vprime_bf, vprime_ub)
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        vnew_ub[t, d] = vinner_ub[t, d] - vprime_ub[t, d]
                                T.copy(vnew_ub, vnew_bf)
                                T.copy(vnew_bf, ws_v_new[unit, 0, 0])
                                T.set_cross_flag("MTE3", 12)   # V2 -> C3

                            # ============ C3 : out_cont / dstate ============
                            with T.Scope("C"):
                                T.wait_cross_flag(12)
                                T.copy(ws_attn_c[unit, 0, 0], ac_l1)
                                T.copy(ws_v_new[unit, 0, 0], vn_l1)
                                T.gemm_v0(ac_l1, vn_l1, oc_l0, init=True)
                                T.copy(oc_l0, ws_out_cont[unit, 0, 0])
                                T.copy(ws_kg[unit, 0, 0], kg_l1)
                                T.gemm_v0(kg_l1, vn_l1, ds_l0, transpose_A=True, init=True)
                                T.copy(ds_l0, ws_dstate[unit, 0, 0])
                                T.set_cross_flag("FIX", 15)   # C3 -> V3

                            # ============ V3 : out / state update ============
                            with T.Scope("V"):
                                T.wait_cross_flag(15)
                                T.copy(ws_attn_inter[unit, 0, 0], ai_bf)
                                T.copy(ai_bf, ai_ub)
                                T.copy(ws_out_cont[unit, 0, 0], ac_bf)
                                T.copy(ac_bf, ac_ub)
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        out_ub[t, d] = ai_ub[t, d] + ac_ub[t, d]
                                T.copy(out_ub, out_bf)
                                T.copy(out_bf[0:valid, 0:dv], out[s0:s0 + valid, h, 0:dv])
                                # dstate is (dk,dv) ; add dstate^T into (dv,dk) state
                                T.copy(ws_dstate[unit, 0, 0], ds_bf)
                                T.copy(ds_bf, ds_ub)
                                for d in T.serial(dv):
                                    for k in T.serial(dk):
                                        state[d, k] = g_exp[C - 1] * state[d, k] + ds_ub[k, d]

                    # ---- write final_state for this (batch, value-head) unit ----
                    with T.Scope("V"):
                        T.copy(state, state_bf)
                        T.copy(state_bf, final_state[b, h, :, :])

    return main
