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

"""Tile-level 设计：Stage1 — within-chunk 预计算（独立 jit kernel）。

Mirrors block_level/stage1.py / AscendC `Stage1`.  One standalone TileLang
kernel that, for a given chunk column `ci`, computes every within-chunk quantity
for all (batch, value-head) units in parallel and writes the current chunk's
workspace frag (bf16 via two-stage T.copy; g_exp/g_exp_neg fp32).

Running each stage as its own kernel (stage1 / stage2 / stage3, launched in
stream order from the entry) gives a hard host-level barrier between stages:
no C<->V cross-flag can leak across chunk iterations (fixes the desync that
broke multi-full-chunk cases), and each kernel only holds its own stage's UB
footprint (fits dk=dv=64, the case that overflowed the old single kernel).

All matmuls cube (T.gemm_v0) in T.Scope("C"), vector ops in T.Scope("V");
C<->V handoff via GM workspace + cross-flag (IDs 0..2).
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


@tilelang.jit(
    out_idx=[],
    workspace_idx=[],
    pass_configs=pass_configs,
)
def stage1_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=CHUNK_SIZE, scale=1.0):
    C = chunk_size
    total_units = batch * nv
    used_core_num = min(NUM_PHYSICAL_CORES, max(1, total_units))
    tasks_per_core = T.ceildiv(total_units, used_core_num)

    @T.prim_func
    def main(
        query: T.Tensor((seq_len, nk, dk), "bfloat16"),            # 0
        key: T.Tensor((seq_len, nk, dk), "bfloat16"),              # 1
        value: T.Tensor((seq_len, nv, dv), "bfloat16"),            # 2
        beta: T.Tensor((seq_len, nv), "bfloat16"),                 # 3
        g: T.Tensor((seq_len, nv), "float32"),                     # 4
        actual_seq_lengths: T.Tensor((batch,), "int32"),           # 5
        ci: T.Tensor((1,), "int32"),                               # 6  current chunk column
        bb: T.Tensor((1,), "int32"),                               # 7  current batch
        # ---- stage workspace (one chunk frag per unit) ----
        ws_q: T.Tensor((total_units, C, dk), "bfloat16"),          # 8
        ws_k: T.Tensor((total_units, C, dk), "bfloat16"),          # 9
        ws_kb: T.Tensor((total_units, C, dk), "bfloat16"),         # 10
        ws_kbe: T.Tensor((total_units, C, dk), "bfloat16"),        # 11
        ws_qprime: T.Tensor((total_units, C, dk), "bfloat16"),     # 12
        ws_kg: T.Tensor((total_units, C, dk), "bfloat16"),         # 13
        ws_vb: T.Tensor((total_units, C, dv), "bfloat16"),         # 14
        ws_kbt: T.Tensor((total_units, C, C), "bfloat16"),         # 15
        ws_qkt: T.Tensor((total_units, C, C), "bfloat16"),         # 16
        ws_x: T.Tensor((total_units, C, C), "bfloat16"),           # 17
        ws_v_inner: T.Tensor((total_units, C, dv), "bfloat16"),    # 18
        ws_k_cumdecay: T.Tensor((total_units, C, dk), "bfloat16"),  # 19
        ws_g_exp: T.Tensor((total_units, C), "float32"),           # 20
        ws_g_exp_neg: T.Tensor((total_units, C), "float32"),       # 21
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            for local_idx in T.serial(tasks_per_core):
                unit = cid * tasks_per_core + local_idx
                if unit < total_units:
                    b = unit // nv
                    h = unit % nv
                    kh = key_head(h, nk, nv)
                    base = T.alloc_ub((1,), "int32")
                    base[0] = 0
                    for bbz in T.serial(b):
                        base[0] += actual_seq_lengths[bbz]
                    Lb = actual_seq_lengths[b]
                    if b == bb[0] and ci[0] * C < Lb:
                        s0 = base[0] + ci[0] * C
                        valid = T.if_then_else(Lb - ci[0] * C < C, Lb - ci[0] * C, C)

                        with T.Scope("V"):
                            q_buf = T.alloc_ub((C, dk), "bfloat16")
                            T.tile.fill(q_buf, 0.0)
                            T.copy(query[s0:s0 + valid, kh, 0:dk], q_buf)
                            k_buf = T.alloc_ub((C, dk), "bfloat16")
                            T.tile.fill(k_buf, 0.0)
                            T.copy(key[s0:s0 + valid, kh, 0:dk], k_buf)
                            v_buf = T.alloc_ub((C, dv), "bfloat16")
                            T.tile.fill(v_buf, 0.0)
                            T.copy(value[s0:s0 + valid, h, 0:dv], v_buf)
                            f = T.alloc_ub((C, dk), "float32")
                            fv = T.alloc_ub((C, dv), "float32")
                            cb = T.alloc_ub((C, dk), "float32")
                            temp_k = T.alloc_ub((C, dk), "bfloat16")
                            temp_v = T.alloc_ub((C, dv), "bfloat16")
                            T.barrier_all()
                            bc = T.alloc_ub((C,), "float32")
                            gc = T.alloc_ub((C,), "float32")
                            T.tile.fill(bc, 0.0)
                            T.tile.fill(gc, 0.0)
                            beta_b16 = T.alloc_ub((1,), "bfloat16")
                            beta_f32 = T.alloc_ub((1,), "float32")
                            g_f32 = T.alloc_ub((1,), "float32")
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
                            g_cum = T.alloc_ub((C,), "float32")
                            g_exp = T.alloc_ub((C,), "float32")
                            g_neg = T.alloc_ub((C,), "float32")
                            g_exp_neg = T.alloc_ub((C,), "float32")
                            g_cum[0] = gc[0]
                            for t in T.serial(1, C):
                                g_cum[t] = g_cum[t - 1] + gc[t]
                            for t in T.serial(C):
                                g_neg[t] = 0.0 - g_cum[t]
                            T.tile.exp(g_exp, g_cum)
                            T.tile.exp(g_exp_neg, g_neg)
                            # ws_q = q*scale ; qPrime
                            T.copy(q_buf, f)
                            for t in T.serial(C):
                                for d in T.serial(dk):
                                    cb[t, d] = f[t, d] * scale
                            T.copy(cb, temp_k)
                            T.copy(temp_k, ws_q[unit, 0, 0])
                            T.copy(q_buf, f)
                            for t in T.serial(C):
                                for d in T.serial(dk):
                                    cb[t, d] = f[t, d] * scale * g_exp[t]
                            T.copy(cb, temp_k)
                            T.copy(temp_k, ws_qprime[unit, 0, 0])
                            # k family
                            T.copy(k_buf, f)
                            for t in T.serial(C):
                                for d in T.serial(dk):
                                    cb[t, d] = f[t, d]
                            T.copy(cb, temp_k)
                            T.copy(temp_k, ws_k[unit, 0, 0])
                            for t in T.serial(C):
                                for d in T.serial(dk):
                                    cb[t, d] = f[t, d] * bc[t]
                            T.copy(cb, temp_k)
                            T.copy(temp_k, ws_kb[unit, 0, 0])
                            for t in T.serial(C):
                                for d in T.serial(dk):
                                    cb[t, d] = f[t, d] * bc[t] * g_exp[t]
                            T.copy(cb, temp_k)
                            T.copy(temp_k, ws_kbe[unit, 0, 0])
                            for t in T.serial(C):
                                for d in T.serial(dk):
                                    cb[t, d] = f[t, d] * g_exp[C - 1] * g_exp_neg[t]
                            T.copy(cb, temp_k)
                            T.copy(temp_k, ws_kg[unit, 0, 0])
                            # vb = v*beta
                            T.copy(v_buf, fv)
                            for t in T.serial(C):
                                for d in T.serial(dv):
                                    fv[t, d] = fv[t, d] * bc[t]
                            T.copy(fv, temp_v)
                            T.copy(temp_v, ws_vb[unit, 0, 0])
                            T.copy(g_exp, ws_g_exp[unit, 0])
                            T.copy(g_exp_neg, ws_g_exp_neg[unit, 0])
                            T.set_cross_flag("MTE3", 0)          # V0 -> C1

                        # ---- C1 : kbt / qkt ----
                        with T.Scope("C"):
                            T.wait_cross_flag(0)
                            kb_l1 = T.alloc_L1((C, dk), "bfloat16")
                            k_l1 = T.alloc_L1((C, dk), "bfloat16")
                            kbt_l0 = T.alloc_L0C((C, C), "float32")
                            T.copy(ws_kb[unit, 0, 0], kb_l1)
                            T.copy(ws_k[unit, 0, 0], k_l1)
                            T.gemm_v0(kb_l1, k_l1, kbt_l0, transpose_B=True, init=True)
                            T.copy(kbt_l0, ws_kbt[unit, 0, 0])
                            q_l1 = T.alloc_L1((C, dk), "bfloat16")
                            qkt_l0 = T.alloc_L0C((C, C), "float32")
                            T.copy(ws_q[unit, 0, 0], q_l1)
                            T.gemm_v0(q_l1, k_l1, qkt_l0, transpose_B=True, init=True)
                            T.copy(qkt_l0, ws_qkt[unit, 0, 0])
                            T.set_cross_flag("FIX", 1)          # C1 -> V1

                        # ---- V1 : A / X = (I-A)^{-1} ----
                        with T.Scope("V"):
                            T.wait_cross_flag(1)
                            kbt_bf = T.alloc_ub((C, C), "bfloat16")
                            kbt_ub = T.alloc_ub((C, C), "float32")
                            T.copy(ws_kbt[unit, 0, 0], kbt_bf)
                            T.copy(kbt_bf, kbt_ub)
                            g_expb = T.alloc_ub((C,), "float32")
                            g_exp_negb = T.alloc_ub((C,), "float32")
                            T.copy(ws_g_exp[unit, 0], g_expb)
                            T.copy(ws_g_exp_neg[unit, 0], g_exp_negb)
                            a_ub = T.alloc_ub((C, C), "float32")
                            for i in T.serial(C):
                                for j in T.serial(C):
                                    if j < i:
                                        a_ub[i, j] = -(kbt_ub[i, j] * g_expb[i] * g_exp_negb[j])
                                    else:
                                        a_ub[i, j] = 0.0
                            # golden quantizes A to bf16 before the forward-sub;
                            # round A through a bf16 staging to match numerics.
                            a_bf = T.alloc_ub((C, C), "bfloat16")
                            T.copy(a_ub, a_bf)
                            T.copy(a_bf, a_ub)
                            attn_ub = T.alloc_ub((C, C), "float32")
                            for i in T.serial(C):
                                for j in T.serial(C):
                                    attn_ub[i, j] = a_ub[i, j]
                            acc1 = T.alloc_ub((1,), "float32")
                            for i in T.serial(1, C):
                                for j in T.serial(i):
                                    acc1[0] = 0.0
                                    for jp in T.serial(i):
                                        acc1[0] += a_ub[i, jp] * attn_ub[jp, j]
                                    attn_ub[i, j] = a_ub[i, j] + acc1[0]
                            for i in T.serial(C):
                                attn_ub[i, i] = 1.0
                            x_bf = T.alloc_ub((C, C), "bfloat16")
                            T.copy(attn_ub, x_bf)
                            T.copy(x_bf, ws_x[unit, 0, 0])
                            T.set_cross_flag("MTE3", 2)         # V1 -> C2

                        # ---- C2 : v_inner / k_cumdecay ----
                        with T.Scope("C"):
                            T.wait_cross_flag(2)
                            x_l1 = T.alloc_L1((C, C), "bfloat16")
                            T.copy(ws_x[unit, 0, 0], x_l1)
                            vb_l1 = T.alloc_L1((C, dv), "bfloat16")
                            vi_l0 = T.alloc_L0C((C, dv), "float32")
                            T.copy(ws_vb[unit, 0, 0], vb_l1)
                            T.gemm_v0(x_l1, vb_l1, vi_l0, init=True)
                            T.copy(vi_l0, ws_v_inner[unit, 0, 0])
                            kbe_l1 = T.alloc_L1((C, dk), "bfloat16")
                            kd_l0 = T.alloc_L0C((C, dk), "float32")
                            T.copy(ws_kbe[unit, 0, 0], kbe_l1)
                            T.gemm_v0(x_l1, kbe_l1, kd_l0, init=True)
                            T.copy(kd_l0, ws_k_cumdecay[unit, 0, 0])

    return main
