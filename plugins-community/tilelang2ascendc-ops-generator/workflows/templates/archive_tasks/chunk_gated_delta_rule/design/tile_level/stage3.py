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

"""Tile-level 设计：Stage3 — chunk 内贡献累加（独立 jit kernel）。

Mirrors block_level/stage3.py / AscendC `Stage3`.  One standalone kernel that,
for chunk column `ci`, adds the within-chunk part onto out (which Stage2 already
seeded with attn_inter), for every unit in parallel:

    V4  attn_c = mask(qkt * decay) * scale     # decay=g_exp[i]*g_exp_neg[j], j<=i
    C4  out   += attn_c @ v_new                # (C,C)@(C,Dv) cube, accumulate

A standalone kernel launch is a hard host-level barrier.  All matmuls cube
(T.gemm_v0), vector ops in T.Scope("V"); C<->V handoff via GM + cross-flag (6..7).
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
def stage3_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=CHUNK_SIZE, scale=1.0):
    C = chunk_size
    total_units = batch * nv
    used_core_num = min(NUM_PHYSICAL_CORES, max(1, total_units))
    tasks_per_core = T.ceildiv(total_units, used_core_num)

    @T.prim_func
    def main(
        out: T.Tensor((seq_len, nv, dv), "bfloat16"),              # 0
        ws_qkt: T.Tensor((total_units, C, C), "bfloat16"),         # 1
        ws_v_new: T.Tensor((total_units, C, dv), "bfloat16"),      # 2
        ws_g_exp: T.Tensor((total_units, C), "float32"),           # 3
        ws_g_exp_neg: T.Tensor((total_units, C), "float32"),       # 4
        ws_attn_c: T.Tensor((total_units, C, C), "bfloat16"),      # 5
        ws_out_cont: T.Tensor((total_units, C, dv), "bfloat16"),   # 6
        actual_seq_lengths: T.Tensor((batch,), "int32"),           # 7
        ci: T.Tensor((1,), "int32"),                               # 8  current chunk column
        bb: T.Tensor((1,), "int32"),                               # 9  current batch
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

                        # ---- V4 : attn_c = mask(qkt*decay)*scale ----
                        with T.Scope("V"):
                            qkt_bf = T.alloc_ub((C, C), "bfloat16")
                            qkt_ub = T.alloc_ub((C, C), "float32")
                            T.copy(ws_qkt[unit, 0, 0], qkt_bf)
                            T.copy(qkt_bf, qkt_ub)
                            g_exp = T.alloc_ub((C,), "float32")
                            g_exp_neg = T.alloc_ub((C,), "float32")
                            T.copy(ws_g_exp[unit, 0], g_exp)
                            T.copy(ws_g_exp_neg[unit, 0], g_exp_neg)
                            attn_ub = T.alloc_ub((C, C), "float32")
                            for i in T.serial(C):
                                for j in T.serial(C):
                                    if j <= i:
                                        attn_ub[i, j] = qkt_ub[i, j] * g_exp[i] * g_exp_neg[j] * scale
                                    else:
                                        attn_ub[i, j] = 0.0
                            attn_bf = T.alloc_ub((C, C), "bfloat16")
                            T.copy(attn_ub, attn_bf)
                            T.copy(attn_bf, ws_attn_c[unit, 0, 0])
                            T.set_cross_flag("MTE3", 6)         # V -> C

                        # ---- C4 : out_cont = attn_c @ v_new ----
                        with T.Scope("C"):
                            T.wait_cross_flag(6)
                            ac_l1 = T.alloc_L1((C, C), "bfloat16")
                            vn_l1 = T.alloc_L1((C, dv), "bfloat16")
                            oc_l0 = T.alloc_L0C((C, dv), "float32")
                            T.copy(ws_attn_c[unit, 0, 0], ac_l1)
                            T.copy(ws_v_new[unit, 0, 0], vn_l1)
                            T.gemm_v0(ac_l1, vn_l1, oc_l0, init=True)
                            T.copy(oc_l0, ws_out_cont[unit, 0, 0])
                            T.set_cross_flag("FIX", 7)         # C -> V

                        # ---- V5 : out += attn_c @ v_new ----
                        with T.Scope("V"):
                            T.wait_cross_flag(7)
                            out_bf = T.alloc_ub((C, dv), "bfloat16")
                            out_ub = T.alloc_ub((C, dv), "float32")
                            T.copy(out[s0:s0 + valid, h, 0:dv], out_bf)
                            T.copy(out_bf, out_ub)
                            ac_bf = T.alloc_ub((C, dv), "bfloat16")
                            ac_ub = T.alloc_ub((C, dv), "float32")
                            T.copy(ws_out_cont[unit, 0, 0], ac_bf)
                            T.copy(ac_bf, ac_ub)
                            for t in T.serial(C):
                                for d in T.serial(dv):
                                    out_ub[t, d] = out_ub[t, d] + ac_ub[t, d]
                            T.copy(out_ub, out_bf)
                            T.copy(out_bf[0:valid, 0:dv], out[s0:s0 + valid, h, 0:dv])

    return main
