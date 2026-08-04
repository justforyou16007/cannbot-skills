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

"""Tile-level Stage1 - within-chunk 预计算（单 kernel 内的函数，非独立 jit）。

Mirrors block_level/stage1.py / AscendC `Stage1`, but unlike the archived
3-kernel version it is *not* its own kernel: `run_stage1` is a plain Python
function inlined into the single `chunk_gated_delta_rule` prim_func, invoked for
every valid chunk inside the unit loop.  It computes every within-chunk quantity
for one (batch, value-head) `unit` and writes the current chunk's GM workspace
frag (bf16 via two-stage T.copy; g_exp/g_exp_neg fp32).

UB reuse (学 AscendC `tmpBuff_` 一次申请): every `T.alloc_ub` temporal is
declared *inside* this function so its liveness is confined to the stage-1
segment of one chunk iteration.  The TileLang memory planner
(`tl.ascend_memory_planning`) then reuses the same physical UB for stage2/3
temporaries (they run strictly after stage1, separated by `T.barrier_all()`),
so the on-chip footprint is only the per-stage peak, not the union — dk=dv=64
now fits where the old hoisted-allocation single kernel (384KB) did not.

C<->V handoff per chunk via GM workspace + cross-flag IDs 0,1,2 (V->C1, C1->V1,
V1->C2), re-armed every chunk iteration (each chunk's V/C re-set the flag before
the partner waits; stage boundaries additionally `barrier_all`).
"""


import tilelang
import tilelang.language as T


def key_head(h, nk, nv):
    """The value head *h* reads key head (h*Nk)//Nv (head grouping)."""
    return (h * nk) // nv


@T.macro(hygienic=False)
def run_stage1(nk, nv, dk, dv, C, scale,
               unit, b, h, kh, s0, valid,
               query, key, value, beta, g,
               ws_q, ws_k, ws_kb, ws_kbe, ws_qprime, ws_kg, ws_vb,
               ws_kbt, ws_qkt, ws_x, ws_v_inner, ws_k_cumdecay,
               ws_g_exp, ws_g_exp_neg):
    # 求逆分块边长: diagonal blocks O00/O11 用 V 前代, off-diagonal O10 用 cube gemm.
    F = C // 2
    # ---- V0 : elementwise prep -> workspace ----
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
        T.copy(ws_kb[unit, 0, 0], kb_l1)
        T.copy(ws_k[unit, 0, 0], k_l1)
        kbt_l0 = T.alloc_L0C((C, C), "float32")
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
        # golden quantizes A to bf16 before the forward-sub; round A through a
        # bf16 staging to match numerics.
        a_bf = T.alloc_ub((C, C), "bfloat16")
        T.copy(a_ub, a_bf)
        T.copy(a_bf, a_ub)
        attn_ub = T.alloc_ub((C, C), "float32")
        for i in T.serial(C):
            for j in T.serial(C):
                attn_ub[i, j] = a_ub[i, j]
        acc1 = T.alloc_ub((1,), "float32")
        # AIC/AIV 协同求逆 ((I-A)^{-1}), 仿 solve_tril_64_ker:
        #   A = [[M00,0],[M10,M11]] (对角线以下), O = [[O00,0],[O10,O11]].
        #   - AIV (V pipe): 对角子块 O00 / O11 用 V 向量前代 (对应 AscendC InverseAIV).
        #   - AIC (C pipe):  非对角子块 O10 = O11 @ (+M10) @ O00 用 cube gemm, 在 C2 完成
        #                     (对应 AscendC InverseAICProcess / AttnInverseMMCompute).
        # 这里 V 只算对角子块, 并把 off-diag 的 +M10(round 后) 暂存到 ws_x[F:C,0:F],
        # 留给 C2 的 gemm 覆盖成 O10.  sign: M10 = a_ub[F:C,0:F] (负), +M10 = kbt*decay.
        # -- O00 = (I + M00)^-1, 行 0..F --
        for i in T.serial(1, F):
            for j in T.serial(i):
                acc1[0] = 0.0
                for jp in T.serial(i):
                    acc1[0] += a_ub[i, jp] * attn_ub[jp, j]
                attn_ub[i, j] = a_ub[i, j] + acc1[0]
        for i in T.serial(F):
            attn_ub[i, i] = 1.0
        # -- O11 = (I + M11)^-1, 行 F..C --
        for ii in T.serial(1, F):
            for jj in T.serial(ii):
                acc1[0] = 0.0
                for jp in T.serial(F, ii + F):
                    acc1[0] += a_ub[ii + F, jp] * attn_ub[jp, jj + F]
                attn_ub[ii + F, jj + F] = a_ub[ii + F, jj + F] + acc1[0]
        for i in T.serial(F, C):
            attn_ub[i, i] = 1.0
        # -- off-diag: 暂存 +M10 (round 后取正) 供 C2 gemm 用 --
        for i in T.serial(F, C):
            for j in T.serial(F):
                attn_ub[i, j] = 0.0 - a_ub[i, j]
        x_bf = T.alloc_ub((C, C), "bfloat16")
        T.copy(attn_ub, x_bf)
        T.copy(x_bf, ws_x[unit, 0, 0])
        T.set_cross_flag("MTE3", 2)         # V1 -> C2

    # ---- C2 : X off-diagonal (AIC gemm) + v_inner / k_cumdecay ----
    with T.Scope("C"):
        T.wait_cross_flag(2)
        # AIC 完成求逆 off-diag: O10 = (O11 @ +M10) @ O00   (仿 AscendC InverseAICProcess / solve_tril_64_ker)
        #   AIV(V) 已把 O00/O11(对角) 与 +M10(off-diag 取正) 暂存到 ws_x[F:C,0:F].
        #   独立 C 子阶段, 让小块(F,F) cube buffer 不与全尺寸(X@vb/X@kbe) L1/L0C 重叠生命周期.
        o11_l1 = T.alloc_L1((F, F), "bfloat16")
        p10_l1 = T.alloc_L1((F, F), "bfloat16")
        t10_l0 = T.alloc_L0C((F, F), "float32")
        o00_l1 = T.alloc_L1((F, F), "bfloat16")
        p10b_l1 = T.alloc_L1((F, F), "bfloat16")
        o10_l0 = T.alloc_L0C((F, F), "float32")
        T.copy(ws_x[unit, F:C, F:C], o11_l1)              # O11
        T.copy(ws_x[unit, F:C, 0:F], p10_l1)              # +M10
        T.gemm_v0(o11_l1, p10_l1, t10_l0, init=True)      # O11 @ (+M10)
        T.copy(t10_l0, ws_x[unit, F:C, 0:F])
        T.copy(ws_x[unit, 0:F, 0:F], o00_l1)              # O00
        T.copy(ws_x[unit, F:C, 0:F], p10b_l1)             # (O11 @ +M10) round 到 bf16
        T.gemm_v0(p10b_l1, o00_l1, o10_l0, init=True)     # (O11@+M10) @ O00
        T.copy(o10_l0, ws_x[unit, F:C, 0:F])              # 覆盖为最终 O10; ws_x 现为完整 X

    # ---- C2b : X @ vb / X @ kbe (原 C2 gemm) ----
    with T.Scope("C"):
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
