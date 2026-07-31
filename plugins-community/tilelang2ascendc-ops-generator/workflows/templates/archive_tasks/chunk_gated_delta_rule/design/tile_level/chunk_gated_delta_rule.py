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

"""Tile-level TileLang design for Chunked Gated Delta Rule.

Expresses the per-(batch, head) chunk recurrence with TileLang primitives.
The within-chunk solve ((I - A)^{-1}) is a serial forward substitution over
the strictly-lower matrix; all other matmuls are expressed with T.gemm.

NOTE: this file documents the tile-level algorithm the AscendC kernel
follows.  TileLang is used for design expression in this workflow — the
production kernel is the AscendC implementation under kernel/.
"""

import tilelang
import tilelang.language as T

from design.chunk_gated_delta_rule_common import (
    CHUNK_SIZE,
    NUM_PHYSICAL_CORES,
    pass_configs,
)

# The value head `h` reads key head `h * nk // nv` (head grouping).
def key_head(h, nk, nv):
    return (h * nk) // nv


# ---------------------------------------------------------------------------
#  Decay helper: decay[i][j] = exp(g_cum[i]) * exp(-g_cum[j])  (j <= i only)
# ---------------------------------------------------------------------------

def _decay_matrix(g_exp, g_exp_neg, C):
    decay = T.alloc_ub((C, C), "float32")
    for i in T.serial(C):
        for j in T.serial(C):
            decay[i, j] = g_exp[i] * g_exp_neg[j]
            if j > i:
                decay[i, j] = 0.0
    return decay


def _cumsum(dst, src, count):
    acc = T.alloc_ub((1,), "float32")
    for i in T.serial(count):
        if i == 0:
            acc[0] = src[0]
        else:
            acc[0] = acc[0] + src[i]
        dst[i] = acc[0]


# ---------------------------------------------------------------------------
#  Within-chunk solve:  X = (I - A)^{-1}, A = -(kb @ k^T) * decay (strictly lower)
# ---------------------------------------------------------------------------

def _solve_within_chunk(kb, kT, decay, C):
    kbt = T.alloc_ub((C, C), "float32")
    T.gemm(kbt, kb, kT)                       # kbt = kb @ k^T
    attn = T.alloc_ub((C, C), "float32")
    for i in T.serial(C):
        for j in T.serial(C):
            attn[i, j] = -kbt[i, j] * decay[i, j]   # A, strictly lower (decay masks)
    # forward substitution: X[i][:i] = A[i][:i] + A[i][:i] @ X[:i][:i]
    for i in T.serial(1, C):
        for j in T.serial(i):
            s = attn[i, j]
            for jp in T.serial(i):
                s = s + attn[i, jp] * attn[jp, j]
            attn[i, j] = s
    for i in T.serial(C):
        attn[i, i] = 1.0                      # eye
    return attn


# ---------------------------------------------------------------------------
#  Inter-chunk stage
# ---------------------------------------------------------------------------

def _inter_chunk(qc, kc, attn, vb, kb, decay, g_exp, state, C, dk, dv):
    v_inner = T.alloc_ub((C, dv), "float32")
    T.gemm(v_inner, attn, vb)                 # X @ vb

    kbe = T.alloc_ub((C, dk), "float32")
    for t in T.serial(C):
        for d in T.serial(dk):
            kbe[t, d] = kb[t, d] * g_exp[t]
    k_cumdecay = T.alloc_ub((C, dk), "float32")
    T.gemm(k_cumdecay, attn, kbe)             # X @ (kb*exp(g_cum))

    qkt = T.alloc_ub((C, C), "float32")
    T.gemm(qkt, qc, kT_of(kc, C, dk))         # q @ k^T
    attn_c = T.alloc_ub((C, C), "float32")
    for i in T.serial(C):
        for j in T.serial(C):
            attn_c[i, j] = qkt[i, j] * decay[i, j]  # causal, keeps j <= i

    v_prime = T.alloc_ub((C, dv), "float32")
    T.gemm(v_prime, k_cumdecay, state)        # k_cumdecay @ S  (S in Dk, Dv)
    v_new = T.alloc_ub((C, dv), "float32")
    for t in T.serial(C):
        for d in T.serial(dv):
            v_new[t, d] = v_inner[t, d] - v_prime[t, d]

    qg = T.alloc_ub((C, dk), "float32")
    for t in T.serial(C):
        for d in T.serial(dk):
            qg[t, d] = qc[t, d] * g_exp[t]
    attn_inter = T.alloc_ub((C, dv), "float32")
    T.gemm(attn_inter, qg, state)             # (q*exp(g_cum)) @ S
    return attn_c, v_new, attn_inter


# k^T for a (C, dk) tile, used by T.gemm(q @ k^T).
def kT_of(k, C, dk):
    kt = T.alloc_ub((dk, C), "float32")
    for d in T.serial(dk):
        for t in T.serial(C):
            kt[d, t] = k[t, d]
    return kt


# ---------------------------------------------------------------------------
#  State recurrence:  S = exp(g_last)*S + (k*exp(g_last-g_cum))^T @ v_new
# ---------------------------------------------------------------------------

def _state_update(kc, g_exp, g_exp_neg, v_new, state, C, dk, dv):
    g_last = g_exp[C - 1]
    for dk_i in T.serial(dk):
        for dv_i in T.serial(dv):
            state[dk_i, dv_i] = g_last * state[dk_i, dv_i]
    k_aug = T.alloc_ub((C, dk), "float32")
    for t in T.serial(C):
        for d in T.serial(dk):
            k_aug[t, d] = kc[t, d] * g_last * g_exp_neg[t]
    k_aug_t = kT_of(k_aug, C, dk)             # (dk, C)
    for dk_i in T.serial(dk):
        for dv_i in T.serial(dv):
            s = state[dk_i, dv_i]
            for t in T.serial(C):
                s = s + k_aug_t[dk_i, t] * v_new[t, dv_i]
            state[dk_i, dv_i] = s


# ---------------------------------------------------------------------------
#  JIT-compiled entry point
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[7], workspace_idx=[8], pass_configs=pass_configs)
def chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv, chunk_size=CHUNK_SIZE, scale=1.0):
    C = chunk_size
    n_chunks = (seq_len + C - 1) // C
    total_units = batch * nv
    used_core_num = min(NUM_PHYSICAL_CORES, max(1, total_units))
    tasks_per_core = T.ceildiv(total_units, used_core_num)

    @T.prim_func
    def main(
        query: T.Tensor((seq_len, nk, dk), "bfloat16"),
        key: T.Tensor((seq_len, nk, dk), "bfloat16"),
        value: T.Tensor((seq_len, nv, dv), "bfloat16"),
        beta: T.Tensor((seq_len, nv), "bfloat16"),
        g: T.Tensor((seq_len, nv), "float32"),
        initial_state: T.Tensor((batch, nv, dv, dk), "float32"),
        actual_seq_lengths: T.Tensor((batch,), "int32"),
        out: T.Tensor((seq_len, nv, dv), "bfloat16"),
        workspaces: tuple,
        final_state: T.Tensor((batch, nv, dv, dk), "float32"),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            for local_idx in T.serial(tasks_per_core):
                unit = cid * tasks_per_core + local_idx
                if unit < total_units:
                    b = unit // nv
                    h = unit % nv
                    kh = key_head(h, nk, nv)
                    with T.Scope("V"):
                        # state in transposed (Dk, Dv) layout
                        state = T.alloc_ub((dk, dv), "float32")
                        for dk_i in T.serial(dk):
                            for dv_i in T.serial(dv):
                                state[dk_i, dv_i] = initial_state[b, h, dv_i, dk_i]

                        for ci in T.serial(n_chunks):
                            s0 = ci * C
                            valid = T.if_then_else(seq_len - s0 < C, seq_len - s0, C)

                            qc = T.alloc_ub((C, dk), "float32")
                            kc = T.alloc_ub((C, dk), "float32")
                            vc = T.alloc_ub((C, dv), "float32")
                            bc = T.alloc_ub((C,), "float32")
                            gc = T.alloc_ub((C,), "float32")
                            for t in T.serial(C):
                                for d in T.serial(dk):
                                    qc[t, d] = T.if_then_else(
                                        t < valid, query[s0 + t, kh, d] * scale, 0.0)
                                    kc[t, d] = T.if_then_else(
                                        t < valid, key[s0 + t, kh, d], 0.0)
                                for d in T.serial(dv):
                                    vc[t, d] = T.if_then_else(
                                        t < valid, value[s0 + t, h, d], 0.0)
                                bc[t] = T.if_then_else(t < valid, beta[s0 + t, h], 0.0)
                                gc[t] = T.if_then_else(t < valid, g[s0 + t, h], 0.0)

                            g_cum = T.alloc_ub((C,), "float32")
                            _cumsum(g_cum, gc, C)
                            g_exp = T.alloc_ub((C,), "float32")
                            g_exp_neg = T.alloc_ub((C,), "float32")
                            for t in T.serial(C):
                                g_exp[t] = T.exp(g_cum[t])
                                g_exp_neg[t] = T.exp(-g_cum[t])

                            kb = T.alloc_ub((C, dk), "float32")
                            vb = T.alloc_ub((C, dv), "float32")
                            for t in T.serial(C):
                                for d in T.serial(dk):
                                    kb[t, d] = kc[t, d] * bc[t]
                                for d in T.serial(dv):
                                    vb[t, d] = vc[t, d] * bc[t]

                            decay = _decay_matrix(g_exp, g_exp_neg, C)
                            kT = kT_of(kc, C, dk)
                            attn = _solve_within_chunk(kb, kT, decay, C)
                            attn_c, v_new, attn_inter = _inter_chunk(
                                qc, kc, attn, vb, kb, decay, g_exp, state, C, dk, dv)

                            for t in T.serial(C):
                                for d in T.serial(dv):
                                    o = attn_inter[t, d]
                                    for j in T.serial(C):
                                        o = o + attn_c[t, j] * v_new[j, d]
                                    if t < valid:
                                        out[s0 + t, h, d] = T.cast(o, "bfloat16")

                            _state_update(kc, g_exp, g_exp_neg, v_new, state, C, dk, dv)

                        for dk_i in T.serial(dk):
                            for dv_i in T.serial(dv):
                                final_state[b, h, dv_i, dk_i] = state[dk_i, dv_i]

    return main
