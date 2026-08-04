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

"""Tile-level Stage3 - chunk 内贡献累加（单 kernel 内的函数，非独立 jit）。

Mirrors block_level/stage3.py / AscendC `Stage3`, *inlined* into the single
`chunk_gated_delta_rule` prim_func (no separate kernel).  For one `unit` and one
chunk it adds the within-chunk contribution onto `out` (which Stage2 already
seeded with attn_inter) for every unit:

    V4  attn_c = mask(qkt * decay) * scale     # decay=g_exp[i]*g_exp_neg[j], j<=i
    C4  out_cont = attn_c @ v_new              # (C,C)@(C,Dv) cube
    V5  out  += out_cont

UB reuse (学 AscendC `tmpBuff_`): `T.alloc_ub` temporaries are declared inside
this function so the planner overlaps them with stage1/2 temporaries on the same
physical UB.

C<->V handoff per chunk via GM + cross-flag IDs 6,7, re-armed each iteration.
"""

import tilelang.language as T


@T.macro(hygienic=False)
def run_stage3(nk, nv, dk, dv, C, scale,
               unit, b, h, s0, valid,
               out, ws_qkt, ws_v_new, ws_g_exp, ws_g_exp_neg,
               ws_attn_c, ws_out_cont):
    # ---- V4 : attn_c = mask(qkt*decay)*scale ----
    with T.Scope("V"):
        qkt_bf_s3 = T.alloc_ub((C, C), "bfloat16")
        qkt_ub_s3 = T.alloc_ub((C, C), "float32")
        T.copy(ws_qkt[unit, 0, 0], qkt_bf_s3)
        T.copy(qkt_bf_s3, qkt_ub_s3)
        g_exp_s3 = T.alloc_ub((C,), "float32")
        g_exp_neg_s3 = T.alloc_ub((C,), "float32")
        T.copy(ws_g_exp[unit, 0], g_exp_s3)
        T.copy(ws_g_exp_neg[unit, 0], g_exp_neg_s3)
        # attn_c computed in-place over qkt_ub_s3 (masked + decayed), drops a
        # separate (C,C) fp32 working buffer.
        for i in T.serial(C):
            for j in T.serial(C):
                if j <= i:
                    qkt_ub_s3[i, j] = qkt_ub_s3[i, j] * g_exp_s3[i] * g_exp_neg_s3[j] * scale
                else:
                    qkt_ub_s3[i, j] = 0.0
        attn_bf_s3 = T.alloc_ub((C, C), "bfloat16")
        T.copy(qkt_ub_s3, attn_bf_s3)
        T.copy(attn_bf_s3, ws_attn_c[unit, 0, 0])
        T.set_cross_flag("MTE3", 6)         # V -> C

    # ---- C4 : out_cont = attn_c @ v_new ----
    with T.Scope("C"):
        T.wait_cross_flag(6)
        ac_l1 = T.alloc_L1((C, C), "bfloat16")
        vn_l1_s3 = T.alloc_L1((C, dv), "bfloat16")
        T.copy(ws_attn_c[unit, 0, 0], ac_l1)
        T.copy(ws_v_new[unit, 0, 0], vn_l1_s3)
        oc_l0 = T.alloc_L0C((C, dv), "float32")
        T.gemm_v0(ac_l1, vn_l1_s3, oc_l0, init=True)
        T.copy(oc_l0, ws_out_cont[unit, 0, 0])
        T.set_cross_flag("FIX", 7)         # C -> V

    # ---- V5 : out += attn_c @ v_new ----
    with T.Scope("V"):
        T.wait_cross_flag(7)
        out_bf_s3 = T.alloc_ub((C, dv), "bfloat16")
        # tail pad: stage2 seeded `out` only as far as `valid` rows; zero-fill
        # the (C-valid) rows so the fp32 accumulate below never reads garbage UB
        # (对应 AscendC CalMaskedQKT 的 "chunkSize 自动补齐" 语义).
        T.tile.fill(out_bf_s3, 0.0)
        T.copy(out[s0:s0 + valid, h, 0:dv], out_bf_s3)
        out_ub_s3 = T.alloc_ub((C, dv), "float32")
        T.copy(out_bf_s3, out_ub_s3)
        ac_bf_s3 = T.alloc_ub((C, dv), "bfloat16")
        ac_ub_s3 = T.alloc_ub((C, dv), "float32")
        T.copy(ws_out_cont[unit, 0, 0], ac_bf_s3)
        T.copy(ac_bf_s3, ac_ub_s3)
        for t in T.serial(C):
            for d in T.serial(dv):
                out_ub_s3[t, d] = out_ub_s3[t, d] + ac_ub_s3[t, d]
        T.copy(out_ub_s3, out_bf_s3)
        T.copy(out_bf_s3[0:valid, 0:dv], out[s0:s0 + valid, h, 0:dv])
