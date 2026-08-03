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

"""Tile-level TileLang design (expert mode) for Chunked Gated Delta Rule.

This is the *expert-mode* rework of the earlier scalar design.  It expresses
the per-(batch, value-head) chunk recurrence over the low-level primitives only:

    T.Kernel / T.alloc_ub / T.copy / T.tile.* / T.reduce_sum / T.serial /
    T.Scope("V") / T.set_flag / T.wait_flag

No developer-mode primitives (T.parallel / T.Parallel / T.pipeline /
T.Pipelined) are used anywhere.

Layout & maths
--------------
* TND tensors: query T(T,Nk,Dk), key T(T,Nk,Dk), value T(T,Nv,Dv),
  beta T(T,Nv) bf16, g T(T,Nv) fp32, initial/final_state (B,Nv,Dv,Dk),
  actual_seq_lengths (B,), out (T,Nv,Dv).
* Each (batch, value-head) unit is independent; value head *h* reads key head
  (h*Nk)//Nv (head grouping).  State S (Dv,Dk) is carried across the chunks of
  one (batch,head) unit.
* Chunking (C = chunk_size): within-chunk precomputation (gCum/decay/kb/vb/
  A/(I-A)^{-1}/vInner/kCumDecay/qkt/attn_c/qPrime/kg) + inter-chunk state
  feed (vPrime/vNew/attnInter, S update).  Symbol relations follow
  `GDR-block-level-AscendC-对应.md` and `design/block_level/`.

bf16 handling
-------------
The Ascend backend (dav-2201) rejects element-wise bf16 <-> fp32 scalar casts.
Therefore *all* bf16 I/O goes through T.copy (MTE/vector copy), which lowers to
copy instructions that carry the type conversion:
    bf16 global -> bf16 UB (T.copy, zero-padded via T.tile.fill)
    bf16 UB     -> fp32 UB (T.copy upcast)
    fp32 UB     -> bf16 UB (T.copy downcast)
    bf16 UB     -> bf16 global (T.copy)
Arithmetic runs in fp32 on UB.

nv<16 strided column load workaround
------------------------------------
beta/g are (T,Nv) column vectors taken at value-head *h*.  A blocked
`T.copy(beta[s0:s0+C, h], bc)` compiles to maskShapeN=1 which only copies the
leading element on this backend ("nv < 16 β/g column load" issue).  We therefore
load beta/g **per scalar** for every row of the chunk, exactly as
`fused_sigmoid_gating_delta_rule_varlen.py` does for per-token β/g:

    T.copy(beta[t, h:h+1], b16[1])     # bf16 scalar global->UB
    T.copy(g[t, h:h+1],  g[1])         # fp32 scalar global->UB
    T.set_flag("mte2","v",n); T.wait_flag("mte2","v",n)
    T.copy(b16[1], f32[1])             # upcast in UB (vector)
    bc[t] = f32[0]; gc[t] = g[0]

A width-1 (1-element) scalar load is independent of nv and works for any nv.
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


@tilelang.jit(out_idx=[7, 8], workspace_idx=[], pass_configs=pass_configs)
def chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv,
                           chunk_size=CHUNK_SIZE, scale=1.0):
    C = chunk_size
    total_units = batch * nv
    used_core_num = min(NUM_PHYSICAL_CORES, max(1, total_units))
    tasks_per_core = T.ceildiv(total_units, used_core_num)
    # Upper bound on the number of chunks any batch may need.
    max_chunks = (seq_len + C - 1) // C

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
            for local_idx in T.serial(tasks_per_core):
                unit = cid * tasks_per_core + local_idx
                if unit < total_units:
                    b = unit // nv
                    h = unit % nv
                    kh = key_head(h, nk, nv)
                    with T.Scope("V"):
                        # ---- state (Dv, Dk) fp32, matches golden layout ----
                        state = T.alloc_ub((dv, dk), "float32")
                        is_b16 = T.alloc_ub((dv, dk), "bfloat16")
                        os_b16 = T.alloc_ub((dv, dk), "bfloat16")

                        # ---- per-chunk row/vector buffers (fp32 compute) ----
                        qc = T.alloc_ub((C, dk), "float32")
                        kc = T.alloc_ub((C, dk), "float32")
                        vc = T.alloc_ub((C, dv), "float32")
                        bc = T.alloc_ub((C,), "float32")
                        gc = T.alloc_ub((C,), "float32")
                        g_cum = T.alloc_ub((C,), "float32")
                        g_cum_neg = T.alloc_ub((C,), "float32")
                        g_exp = T.alloc_ub((C,), "float32")
                        g_exp_neg = T.alloc_ub((C,), "float32")
                        # kb = k*beta is folded inline into A and k_cumdecay (dead
                        # buffer dropped to fit dk=dv=64 in UB).  decay/kbt/qkt/
                        # vb/qg/k_aug are likewise folded inline.  We keep only the
                        # (C,C) `a` (strictly-lower A) and `attn` ((I-A)^{-1}) plus
                        # `attn_c` result buffers.
                        a = T.alloc_ub((C, C), "float32")
                        attn = T.alloc_ub((C, C), "float32")
                        v_inner = T.alloc_ub((C, dv), "float32")
                        k_cumdecay = T.alloc_ub((C, dk), "float32")
                        attn_c = T.alloc_ub((C, C), "float32")
                        v_new = T.alloc_ub((C, dv), "float32")
                        attn_inter = T.alloc_ub((C, dv), "float32")

                        # ---- bf16 staging for block input/output ----
                        qb16 = T.alloc_ub((C, dk), "bfloat16")
                        kb16 = T.alloc_ub((C, dk), "bfloat16")
                        vb16 = T.alloc_ub((C, dv), "bfloat16")
                        ob16 = T.alloc_ub((C, dv), "bfloat16")

                        # ---- per-scalar staging for beta (bf16) / g (fp32) ----
                        beta_b16 = T.alloc_ub((1,), "bfloat16")
                        beta_f32 = T.alloc_ub((1,), "float32")
                        g_f32 = T.alloc_ub((1,), "float32")

                        # ---- per-batch init: base = sum(seq[0..b)), Lb = seq[b] ----
                        base = T.alloc_ub((1,), "int32")
                        base[0] = 0
                        for bb in T.serial(b):
                            base[0] += actual_seq_lengths[bb]
                        Lb = actual_seq_lengths[b]

                        # initial_state -> state (bf16 upcast to fp32)
                        T.tile.fill(is_b16, 0.0)
                        T.copy(initial_state[b, h, :, :], is_b16)
                        T.tile.fill(state, 0.0)
                        T.copy(is_b16, state)
                        T.barrier_all()

                        for ci in T.serial(max_chunks):
                            if ci * C < Lb:
                                s0 = base[0] + ci * C
                                valid = T.if_then_else(Lb - ci * C < C,
                                                       Lb - ci * C, C)

                                # ---- load q,k,v (bf16 block copies, full slices) ----
                                # maskShapeN = dk/dv >= 16, so these block copies
                                # are safe.  Rows t >= valid are zero-padded.
                                T.tile.fill(qb16, 0.0)
                                T.copy(query[s0:s0 + valid, kh, 0:dk], qb16)
                                T.tile.fill(kb16, 0.0)
                                T.copy(key[s0:s0 + valid, kh, 0:dk], kb16)
                                T.tile.fill(vb16, 0.0)
                                T.copy(value[s0:s0 + valid, h, 0:dv], vb16)
                                T.copy(qb16, qc)
                                T.copy(kb16, kc)
                                T.copy(vb16, vc)
                                T.barrier_all()

                                # ---- beta / g: per-scalar column load (nv<16 safe) ----
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

                                # ---- within-chunk ----
                                # g_cum = cumsum(gc)
                                g_cum[0] = gc[0]
                                for _i in T.serial(1, C):
                                    g_cum[_i] = g_cum[_i - 1] + gc[_i]
                                # exponentials of +-g_cum (fp32 buffers)
                                for t in T.serial(C):
                                    g_cum_neg[t] = 0.0 - g_cum[t]
                                T.tile.exp(g_exp, g_cum)          # exp(g_cum)
                                T.tile.exp(g_exp_neg, g_cum_neg)  # exp(-g_cum)
                                # A = -decay*(kb@kc^T) strictly lower, with both the
                                # beta (kb=kc*beta, folded as bc[i]) and the decay
                                # factor decay[i][j]=exp(g_cum[i])*exp(-g_cum[j])
                                # folded into the dot product (kb/decay/kbt dead).
                                for i in T.serial(C):
                                    for j in T.serial(C):
                                        if j < i:
                                            a[i, j] = 0.0
                                            for kx in T.serial(dk):
                                                a[i, j] += kc[i, kx] * bc[i] * kc[j, kx]
                                            a[i, j] = -a[i, j] * g_exp[i] * g_exp_neg[j]
                                        else:
                                            a[i, j] = 0.0

                                # attn = (I-A)^{-1}  via forward substitution
                                for i in T.serial(C):
                                    for j in T.serial(C):
                                        attn[i, j] = a[i, j]
                                for i in T.serial(1, C):
                                    for j in T.serial(i):
                                        for jp in T.serial(i):
                                            attn[i, j] += a[i, jp] * attn[jp, j]
                                for i in T.serial(C):
                                    attn[i, i] = 1.0

                                # v_inner = attn @ (v*beta), beta folded inline
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        v_inner[t, d] = 0.0
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        for kx in T.serial(C):
                                            v_inner[t, d] += attn[t, kx] * vc[kx, d] * bc[kx]

                                # k_cumdecay = attn @ (kb * exp(g_cum)),
                                # with kb=kc*beta folded inline
                                for t in T.serial(C):
                                    for d in T.serial(dk):
                                        k_cumdecay[t, d] = 0.0
                                for t in T.serial(C):
                                    for d in T.serial(dk):
                                        for kx in T.serial(C):
                                            k_cumdecay[t, d] += attn[t, kx] * kc[kx, d] * bc[kx] * g_exp[kx]

                                # ---- inter-chunk ----
                                # attn_c = (qc@kc^T * decay).tril(), decay factor
                                # (exp(g_cum[i])*exp(-g_cum[j])) folded into the dot
                                # product (qkt buffer dropped).
                                for i in T.serial(C):
                                    for j in T.serial(C):
                                        if j <= i:
                                            attn_c[i, j] = 0.0
                                            for kx in T.serial(dk):
                                                attn_c[i, j] += qc[i, kx] * kc[j, kx]
                                            attn_c[i, j] = attn_c[i, j] * g_exp[i] * g_exp_neg[j]
                                        else:
                                            attn_c[i, j] = 0.0

                                # v_new = v_inner - k_cumdecay@state (v_prime folded inline)
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        v_new[t, d] = v_inner[t, d]
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        for kx in T.serial(dk):
                                            v_new[t, d] -= k_cumdecay[t, kx] * state[d, kx]

                                # attn_inter = (q*exp(g_cum)) @ state, exp(g_cum[t])
                                # folded inline (qg buffer dropped)
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        attn_inter[t, d] = 0.0
                                        for kx in T.serial(dk):
                                            attn_inter[t, d] += qc[t, kx] * state[d, kx]
                                        attn_inter[t, d] = attn_inter[t, d] * g_exp[t]

                                # out = attn_inter + attn_c @ v_new
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        for j in T.serial(C):
                                            attn_inter[t, d] += attn_c[t, j] * v_new[j, d]

                                # fp32 -> bf16 UB, write valid rows (full slices)
                                T.tile.fill(ob16, 0.0)
                                T.copy(attn_inter[0:C, 0:dv], ob16)
                                T.copy(ob16[0:valid, 0:dv], out[s0:s0 + valid, h, 0:dv])
                                T.barrier_all()

                                # ---- state update ----
                                # S = exp(g_last)S + (k*exp(g_last-g_cum))^T @ v_new,
                                # the k_aug factor kc[t,k]*exp(g_last-g_cum[t]) folded
                                # inline (k_aug buffer dropped).
                                decay_last = g_exp[C - 1]
                                for d in T.serial(dv):
                                    for k in T.serial(dk):
                                        state[d, k] = decay_last * state[d, k]
                                        for t in T.serial(C):
                                            state[d, k] += kc[t, k] * decay_last * g_exp_neg[t] * v_new[t, d]

                        # after the last chunk: state -> final_state (downcast)
                        T.tile.fill(os_b16, 0.0)
                        T.copy(state, os_b16)
                        T.copy(os_b16, final_state[b, h, :, :])

    return main
