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
The value head `h` reads key head `h * nk // nv` (head grouping); the state
S (Dv, Dk) is carried across chunks of one (batch, value-head) unit and reset
per batch.

Ascend bf16 constraint (TL_ASCEND on a dav-2201 target): scalar (element-wise)
bf16 <-> fp32 casts are not supported by the backend ("not support bf16 type
cast").  All bf16 I/O therefore goes through `T.copy` (MTE/vector copy), which
lowers to supported copy instructions:
    bf16 global -> bf16 UB (T.copy, zero-padding via T.tile.fill)
    bf16 UB     -> fp32 UB (T.copy upcast)
    fp32 UB     -> bf16 UB (T.copy downcast)
    bf16 UB     -> bf16 global (T.copy)
All arithmetic is carried out in fp32 on UB buffers using explicit serial
element-wise loops; the within-chunk solve ((I - A)^{-1}) is a serial forward
substitution over the strictly-lower A.
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


@tilelang.jit(out_idx=[7, 8], workspace_idx=[], pass_configs=pass_configs)
def chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv, chunk_size=CHUNK_SIZE, scale=1.0):
    C = chunk_size
    total_units = batch * nv
    used_core_num = min(NUM_PHYSICAL_CORES, max(1, total_units))
    tasks_per_core = T.ceildiv(total_units, used_core_num)
    max_chunks = (seq_len + C - 1) // C  # upper bound on per-batch chunk count

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

                        # ---- per-chunk row buffers (fp32 compute) ----
                        qc = T.alloc_ub((C, dk), "float32")
                        kc = T.alloc_ub((C, dk), "float32")
                        vc = T.alloc_ub((C, dv), "float32")
                        bc = T.alloc_ub((C,), "float32")
                        gc = T.alloc_ub((C,), "float32")
                        g_cum = T.alloc_ub((C,), "float32")
                        g_cum_neg = T.alloc_ub((C,), "float32")
                        g_exp = T.alloc_ub((C,), "float32")
                        g_exp_neg = T.alloc_ub((C,), "float32")
                        kb = T.alloc_ub((C, dk), "float32")
                        vb = T.alloc_ub((C, dv), "float32")
                        decay = T.alloc_ub((C, C), "float32")
                        kbt = T.alloc_ub((C, C), "float32")
                        a = T.alloc_ub((C, C), "float32")
                        attn = T.alloc_ub((C, C), "float32")
                        v_inner = T.alloc_ub((C, dv), "float32")
                        kbe = T.alloc_ub((C, dk), "float32")
                        k_cumdecay = T.alloc_ub((C, dk), "float32")
                        qkt = T.alloc_ub((C, C), "float32")
                        attn_c = T.alloc_ub((C, C), "float32")
                        v_prime = T.alloc_ub((C, dv), "float32")
                        v_new = T.alloc_ub((C, dv), "float32")
                        qg = T.alloc_ub((C, dk), "float32")
                        attn_inter = T.alloc_ub((C, dv), "float32")
                        k_aug = T.alloc_ub((C, dk), "float32")

                        # bf16 staging for input loading / output writing
                        qb16 = T.alloc_ub((C, dk), "bfloat16")
                        kb16 = T.alloc_ub((C, dk), "bfloat16")
                        vb16 = T.alloc_ub((C, dv), "bfloat16")
                        bb16 = T.alloc_ub((C,), "bfloat16")
                        ob16 = T.alloc_ub((C, dv), "bfloat16")

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
                                valid = T.if_then_else(Lb - ci * C < C, Lb - ci * C, C)

                                # load q,k,v,beta (bf16) with zero-padding, upcast to fp32
                                T.tile.fill(qb16, 0.0)
                                T.copy(query[s0:s0 + valid, kh, 0:dk], qb16)
                                T.copy(qb16, qc)
                                T.tile.fill(kb16, 0.0)
                                T.copy(key[s0:s0 + valid, kh, 0:dk], kb16)
                                T.copy(kb16, kc)
                                T.tile.fill(vb16, 0.0)
                                T.copy(value[s0:s0 + valid, h, 0:dv], vb16)
                                T.copy(vb16, vc)
                                T.tile.fill(bb16, 0.0)
                                T.copy(beta[s0:s0 + valid, h], bb16)
                                T.copy(bb16, bc)
                                T.tile.fill(gc, 0.0)
                                T.copy(g[s0:s0 + valid, h], gc[0:valid])
                                T.barrier_all()

                                # --- within-chunk ---
                                # g_cum = cumsum(gc); decay[i][j]=exp(g_cum[i])*exp(-g_cum[j]) j<=i
                                g_cum[0] = gc[0]
                                for _i in T.serial(1, C):
                                    g_cum[_i] = g_cum[_i - 1] + gc[_i]
                                for t in T.Parallel(C):
                                    g_cum_neg[t] = 0.0 - g_cum[t]
                                for t in T.Parallel(C):
                                    g_exp[t] = T.exp(g_cum[t])
                                for t in T.Parallel(C):
                                    g_exp_neg[t] = T.exp(g_cum_neg[t])
                                T.barrier_all()
                                for i in T.serial(C):
                                    for j in T.serial(C):
                                        if j <= i:
                                            decay[i, j] = g_exp[i] * g_exp_neg[j]
                                        else:
                                            decay[i, j] = 0.0

                                for t in T.serial(C):
                                    for d in T.serial(dk):
                                        kb[t, d] = kc[t, d] * bc[t]
                                    for d in T.serial(dv):
                                        vb[t, d] = vc[t, d] * bc[t]

                                # kbt = kb @ kc^T
                                for i in T.serial(C):
                                    for j in T.serial(C):
                                        kbt[i, j] = 0.0
                                for i in T.serial(C):
                                    for j in T.serial(C):
                                        for kx in T.serial(dk):
                                            kbt[i, j] += kb[i, kx] * kc[j, kx]

                                # A = -kbt * decay, strictly lower
                                for i in T.serial(C):
                                    for j in T.serial(C):
                                        if j < i:
                                            a[i, j] = -kbt[i, j] * decay[i, j]
                                        else:
                                            a[i, j] = 0.0
                                # attn = (I-A)^{-1}: forward substitution
                                for i in T.serial(C):
                                    for j in T.serial(C):
                                        attn[i, j] = a[i, j]
                                for i in T.serial(1, C):
                                    for j in T.serial(i):
                                        for jp in T.serial(i):
                                            attn[i, j] += a[i, jp] * attn[jp, j]
                                for i in T.serial(C):
                                    attn[i, i] = 1.0

                                # v_inner = attn @ vb
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        v_inner[t, d] = 0.0
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        for kx in T.serial(C):
                                            v_inner[t, d] += attn[t, kx] * vb[kx, d]

                                # kbe = kb * exp(g_cum); k_cumdecay = attn @ kbe
                                for t in T.serial(C):
                                    for d in T.serial(dk):
                                        kbe[t, d] = kb[t, d] * g_exp[t]
                                for t in T.serial(C):
                                    for d in T.serial(dk):
                                        k_cumdecay[t, d] = 0.0
                                for t in T.serial(C):
                                    for d in T.serial(dk):
                                        for kx in T.serial(C):
                                            k_cumdecay[t, d] += attn[t, kx] * kbe[kx, d]

                                # --- inter-chunk ---
                                # qkt = qc @ kc^T ; attn_c = (qkt*decay).tril()
                                for i in T.serial(C):
                                    for j in T.serial(C):
                                        qkt[i, j] = 0.0
                                for i in T.serial(C):
                                    for j in T.serial(C):
                                        for kx in T.serial(dk):
                                            qkt[i, j] += qc[i, kx] * kc[j, kx]
                                for i in T.serial(C):
                                    for j in T.serial(C):
                                        if j <= i:
                                            attn_c[i, j] = qkt[i, j] * decay[i, j]
                                        else:
                                            attn_c[i, j] = 0.0

                                # v_prime = k_cumdecay @ state ; v_new = v_inner - v_prime
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        v_prime[t, d] = 0.0
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        for kx in T.serial(dk):
                                            v_prime[t, d] += k_cumdecay[t, kx] * state[d, kx]
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        v_new[t, d] = v_inner[t, d] - v_prime[t, d]

                                # attn_inter = (q*exp(g_cum)) @ state
                                for t in T.serial(C):
                                    for d in T.serial(dk):
                                        qg[t, d] = qc[t, d] * g_exp[t]
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        attn_inter[t, d] = 0.0
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        for kx in T.serial(dk):
                                            attn_inter[t, d] += qg[t, kx] * state[d, kx]

                                # out = attn_inter + attn_c @ v_new
                                for t in T.serial(C):
                                    for d in T.serial(dv):
                                        for j in T.serial(C):
                                            attn_inter[t, d] += attn_c[t, j] * v_new[j, d]
                                # downcast fp32 -> bf16 UB, then write valid rows to global
                                T.tile.fill(ob16, 0.0)
                                T.copy(attn_inter[0:C, 0:dv], ob16)
                                T.copy(ob16[0:valid, 0:dv], out[s0:s0 + valid, h, 0:dv])
                                T.barrier_all()

                                # --- state update: S = exp(g_last)S + (k*exp(g_last-g_cum))^T @ v_new
                                decay_last = g_exp[C - 1]
                                for t in T.serial(C):
                                    for d in T.serial(dk):
                                        k_aug[t, d] = kc[t, d] * decay_last * g_exp_neg[t]
                                for d in T.serial(dv):
                                    for k in T.serial(dk):
                                        state[d, k] = decay_last * state[d, k]
                                for d in T.serial(dv):
                                    for k in T.serial(dk):
                                        for t in T.serial(C):
                                            state[d, k] += k_aug[t, k] * v_new[t, d]

                        # after the last chunk: state -> final_state (downcast bf16)
                        T.tile.fill(os_b16, 0.0)
                        T.copy(state, os_b16)
                        T.copy(os_b16, final_state[b, h, :, :])

    return main
