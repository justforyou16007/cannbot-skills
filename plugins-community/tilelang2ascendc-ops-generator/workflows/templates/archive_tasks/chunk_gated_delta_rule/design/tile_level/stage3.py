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

Mirrors block_level/stage3.py / AscendC `Stage3`（stage3.h），但为单 kernel 内联函数。
**C/V 分工与同步完全照抄 AscendC（AIC/AIV 并行，2 flag）**。
**任务分片封装在此函数内部**（对应 stage3.h `Stage3::Process`）：融合任务
`totalChunks = Nv × chunkNum` 均分（`chunksPerCore = ceil`；`idx → nvId = idx/chunkNum,
cgId = idx%chunkNum`）。**不依赖状态，无串行**。workspace 行偏移 `row = cgId * C`。

    V4  attn_c = mask(qkt · decay) · scale   → ws_attn_c        （AIV 向量，↔ CalMaskedQKT 写 tmpGM）
    C4  out += attn_c @ v_new（gemm→L0C → atomic_add 到 out）    （cube，↔ AICProcess enAtomic=1）

v_new = `ws_v_inner`（Stage2 已把 v_inner 原子加更新为 v_new）。C4 把 `attn_c @ v_new`
**原子加直写 out**（out 已有 Stage2 的 attn_inter）。无 V5 / ws_out_cont。

同步 2 flag：9↔0x4（V4 写完 attn_c → C4 读）、7↔0x3（C4 累加完 → V4 可覆盖 ws_attn_c）。
V4 内 set 9 后 wait 7（照抄 AscendC AIV 分支：set 0x4 后 wait 0x3）。

**buffer 分配纪律**：本 stage 全部 on-chip buffer 在函数顶部一次分配（静态），任务分片循环
（`for nvId / for cgId`）内只复用，不再 alloc。
"""


import tilelang.language as T


@T.macro(hygienic=False)
def run_stage3(cid, used_core_num, group_chunks,
               nk, nv, dk, dv, C, scale,
               gpos, cg_len, num_chunk,
               out, ws_qkt, ws_v_inner, ws_g_exp, ws_g_exp_neg,
               ws_attn_c):
    # ---- 统一分配本 stage 全部 on-chip buffer（函数顶部，分片循环外静态复用）----
    # V4 attn_c = mask(qkt*decay)*scale
    qkt_bf_s3 = T.alloc_ub((C, C), "bfloat16")
    qkt_ub_s3 = T.alloc_ub((C, C), "float32")
    g_exp_s3 = T.alloc_ub((C,), "float32")
    g_exp_neg_s3 = T.alloc_ub((C,), "float32")
    # 衰减矩阵构造：decay[i,j] = g_exp[i]/g_exp[j]（T.tile 外积，避免嵌套 for 编译 bug）
    ge_row_s3 = T.alloc_ub((C, C), "float32")
    ge_col_s3 = T.alloc_ub((C, C), "float32")
    decay_s3 = T.alloc_ub((C, C), "float32")
    attn_bf_s3 = T.alloc_ub((C, C), "bfloat16")
    # C4 out += attn_c @ v_new
    ac_l1 = T.alloc_L1((C, C), "bfloat16")
    vn_l1_s3 = T.alloc_L1((C, dv), "bfloat16")
    oc_l0 = T.alloc_L0C((C, dv), "float32")

    # ---- AscendC Stage3::Process：totalChunks = nv*num_chunk 融合均分（chunksPerCore=ceil）----
    total_chunks = nv * num_chunk
    chunks_per_core = (total_chunks + used_core_num - 1) // used_core_num
    start = cid * chunks_per_core
    end = T.if_then_else(start + chunks_per_core < total_chunks,
                         start + chunks_per_core, total_chunks)

    for nvId in T.serial(nv):
        for cgId in T.serial(group_chunks):
            idx = nvId * num_chunk + cgId
            # 守卫必须含 cgId < num_chunk：idx = nvId*num_chunk + cgId 反解不唯一，
            # 当 nvId 小、cgId >= num_chunk 时 idx 仍可能 < totalChunks 而误通过，
            # 导致 s0/workspace/out 偏移越界（dv 小时尤其明显，MTE 写越界）。
            if idx >= start and idx < end and cgId < num_chunk:
                s0 = gpos + cgId * C
                valid = T.if_then_else(cg_len - cgId * C < C, cg_len - cgId * C, C)
                row = cgId * C

                # ---- V4（V scope / AIV）：写 attn_c → set 0x4 → wait 0x3（↔ CalMaskedQKT 写 tmpGM）----
                with T.Scope("V"):
                    T.copy(ws_qkt[nvId, row, 0], qkt_bf_s3)
                    T.copy(qkt_bf_s3, qkt_ub_s3)
                    T.copy(ws_g_exp[nvId, row], g_exp_s3)
                    T.copy(ws_g_exp_neg[nvId, row], g_exp_neg_s3)
                    # attn_c = mask(qkt·decay)·scale，衰减 decay[i,j]=g_exp[i]/g_exp[j]
                    # （= exp(g_cum[i]-g_cum[j])，对应 AscendC CalMaskedQKT 一次 exp）。
                    # 用 T.tile 构造外积 + 逐元素乘，避免嵌套 for 的标量赋值编译 bug。
                    T.tile.broadcast(ge_row_s3, g_exp_s3, axis=1)   # 每行 = g_exp[i]
                    T.tile.broadcast(ge_col_s3, g_exp_s3, axis=0)   # 每列 = g_exp[j]
                    T.tile.div(decay_s3, ge_row_s3, ge_col_s3)
                    T.tile.mul(qkt_ub_s3, qkt_ub_s3, decay_s3)
                    # 上三角清零（因果 mask，保留对角）
                    for i in T.serial(C):
                        for j in T.serial(C):
                            if j > i:
                                qkt_ub_s3[i, j] = 0.0
                    T.copy(qkt_ub_s3, attn_bf_s3)
                    T.copy(attn_bf_s3, ws_attn_c[nvId, row, 0])
                    # set/wait 放 V scope 内（AIV 流），照抄 AscendC AIV 分支：set 0x4 后 wait 0x3。
                    T.set_cross_flag("MTE3", 9)         # ↔ 0x4：V4 写完 attn_c -> C4 可读
                    T.wait_cross_flag(7)                # ↔ 0x3：等本轮 C4 累加完 -> 下 idx V4 可覆盖 ws_attn_c

                # ---- C4（C scope / AIC）：wait 0x4 → 累加 out → set 0x3（↔ AICProcess enAtomic=1）----
                with T.Scope("C"):
                    T.wait_cross_flag(9)                # ↔ 0x4：等 V4 写完 attn_c
                    T.copy(ws_attn_c[nvId, row, 0], ac_l1)
                    T.copy(ws_v_inner[nvId, row, 0], vn_l1_s3)
                    T.gemm_v0(ac_l1, vn_l1_s3, oc_l0, init=True)     # (C, dv)
                    T.tile.atomic_add(out[s0:s0 + valid, nvId, 0:dv], oc_l0)   # out += attn_c @ v_new（L0C→GM 原子加，↔ AICProcess enAtomic=1）
                    T.set_cross_flag("FIX", 7)          # ↔ 0x3：C4 累加完 -> V4 可覆盖 ws_attn_c
