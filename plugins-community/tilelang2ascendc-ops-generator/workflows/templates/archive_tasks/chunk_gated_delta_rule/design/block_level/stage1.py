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

"""Block-level 设计：Stage1 — within-chunk 预计算（单 kernel 内的 T.macro 骨架）。

对应 AscendC `Stage1`（stage1.h），**C/V 分工与同步完全照抄（AIC/AIV 并行流水）**。细算在
`design/tile_level/stage1.py` 落地（TODO(tile-level)）。
**任务分片封装在 stage 内部**（照抄 stage1.h `Stage1::Process`）：`totalChunk = nv × num_chunk`
按 `coreIdx` 均分（former/tail），`taskId` 反解 `(nvId, cgId)`；workspace 为 AscendC 布局
`(Nv, maxGroupLength, ...)`，本 chunk 行偏移 `row = cgId * C`（stage1.h `SetChunkOffset` 的
`chunkRowBase`）。满足 checklist #4/#5/#6/#7/#10。

## C/V 子阶段（照抄 AscendC 分工，checklist #4）

    V0a  prep 连续化：ws_q = q*scale；ws_k = k                            （向量，↔ QKPreProcess）
    C1   kkt = k @ k^T → ws_kbt；qkt = q @ k^T → ws_qkt                    （cube，↔ kkt + qkt）
    V0b  g_cum = cumsum(g)；g_exp/g_exp_neg = exp(±g_cum)                  （向量，↔ GCumExp，与 C1 并行）
    V1a  attn_1 = ws_kbt · β · decay（decay=g_exp[i]·g_exp_neg[j], j<=i）；A=-(attn_1).tril(-1)；
         X 对角子块前代 + off-diag M10 暂存到 ws_x                        （向量，↔ KKBeta + Gamma + InverseCompute）
    C2a  off-diag 求逆 X10 = X11·M10·X00（覆盖 ws_x 为完整 X）              （cube，↔ AttnInverseMMCompute）
    V2   ws_kbe = -β·g_exp·k（gBK 带负号）；ws_vb = v·β；ws_kg             （向量，↔ GBK + VBeta）
    C2b  k_cumdecay = X @ ws_kbe；v_inner = X @ ws_vb                      （cube，↔ k_cumdecay + v_inner）
    V3   ws_qprime = q·scale·g_exp                                        （向量，↔ QPrimeCompute）

注：kkt 用**纯 k**（beta 在 V1a 广播乘，照抄 AscendC KKBetaCompute），故无 ws_kb；
ws_kbe 存 **gBK（-β·g_exp·k，带负号）**，C2b 直接 gemm 得 k_cumdecay（对应 AscendC `attn@gBK`）。

## 同步（checklist #5，AIC↔AIV 并行，6 flag 与 AscendC 一一对应）

    - flag 0 ↔ 0x9：V0a prep 写完 → C1 可算 kkt/qkt
    - flag 1 ↔ 0x8：C1 kkt 写完 → V1a 可算 KKBeta/求逆
    - flag 5 ↔ 0xA：C1 qkt 写完 → V3 wait（stage1 完成）
    - flag 2 ↔ 0x7：V1a 求逆对角写完 → C2a 可算 off-diag
    - flag 3 ↔ 0x6：V2 gBK 写完 → C2b 可算 k_cumdecay
    - flag 4 ↔ 0x5：V2 vb 写完 → C2b 可算 v_inner

阶段边界由入口 `T.sync_all()` 兜底。编号避开 stage2（6/8/10）、stage3（9/7）。
"""

import tilelang.language as T


@T.macro(hygienic=False)
def run_stage1(cid, used_core_num, group_chunks,
               nk, nv, dk, dv, C, scale,
               gpos, cg_len, num_chunk,
               query, key, value, beta, g,
               ws_q, ws_k, ws_kbe, ws_qprime, ws_kg, ws_vb,
               ws_kbt, ws_qkt, ws_x, ws_v_inner, ws_k_cumdecay,
               ws_g_exp, ws_g_exp_neg):
    """within-chunk 预计算（骨架；任务分片照抄 stage1.h Process，细算见 tile_level/stage1.py）。"""
    # ---- AscendC Stage1::Process：totalChunk = nv*num_chunk 按 coreIdx 均分 ----
    #   `for taskId = start; taskId < end; taskId += paraNum(4)`，taskId 反解 (nvId, cgId)
    total_chunk = nv * num_chunk
    tail_chunk_num = total_chunk // used_core_num
    former_core_num = total_chunk % used_core_num
    former_chunk_num = tail_chunk_num + 1
    start = T.if_then_else(cid < former_core_num,
                           cid * former_chunk_num,
                           former_core_num * former_chunk_num
                           + (cid - former_core_num) * tail_chunk_num)
    end = start + T.if_then_else(cid < former_core_num, former_chunk_num, tail_chunk_num)

    for taskId in T.serial(nv * group_chunks):
        if taskId >= start and taskId < end:
            nvId = taskId % nv
            cgId = taskId // nv
            s0 = gpos + cgId * C
            valid = T.if_then_else(cg_len - cgId * C < C, cg_len - cgId * C, C)
            row = cgId * C
            kh = (nvId * nk) // nv      # 该 value head 读的 key head（head 分组）

            # ---- V0a : prep 连续化（↔ QKPreProcess）----
            with T.Scope("V"):
                # TODO(tile-level): 加载 chunk q/k（s0..s0+valid, kh）+ v（nvId）
                #                   ws_q = q*scale；ws_k = k（连续化落 GM）
                pass
            T.set_cross_flag("MTE3", 0)          # ↔ 0x9：prep 写完 -> C1 可算 kkt/qkt

            # ---- C1 : kkt / qkt（↔ AIC kkt + qkt，与 V0b 的 g_cum 并行）----
            with T.Scope("C"):
                T.wait_cross_flag(0)
                # TODO(tile-level): T.gemm_v0(ws_k, ws_k, ws_kbt, transpose_B=True)   # kkt = k@k^T（纯 k）
                #                   T.gemm_v0(ws_q, ws_k, ws_qkt, transpose_B=True)   # qkt = q@k^T
                pass
            T.set_cross_flag("FIX", 1)          # ↔ 0x8：kkt 写完 -> V1a 可算 KKBeta/求逆
            T.set_cross_flag("FIX", 5)          # ↔ 0xA：qkt 写完 -> V3 wait（stage1 完成）

            # ---- V0b : g_cum / g_exp / g_exp_neg（↔ GCumExp，与 C1 并行）----
            with T.Scope("V"):
                # TODO(tile-level): 加载 β/g（逐标量）；g_cum=cumsum(g)
                #                   ws_g_exp = T.tile.exp(g_cum)；ws_g_exp_neg = T.tile.exp(-g_cum)
                pass

            # ---- V1a : KKBeta + Gamma + Inverse 对角（↔ KKBeta + InverseCompute）----
            with T.Scope("V"):
                T.wait_cross_flag(1)
                # TODO(tile-level): decay[i][j] = ws_g_exp[i]*ws_g_exp_neg[j] (j<=i)
                #                   attn_1 = ws_kbt * β * decay（β 广播，↔ KKBeta+Gamma）
                #                   A = -(attn_1).tril(-1)；X 对角子块 AIV 向量前代
                #                   + off-diag M10（负值）暂存 -> ws_x
                pass
            T.set_cross_flag("MTE3", 2)         # ↔ 0x7：求逆对角写完 -> C2a 可算 off-diag

            # ---- C2a : off-diag 求逆（↔ AttnInverseMMCompute）----
            with T.Scope("C"):
                T.wait_cross_flag(2)
                # TODO(tile-level): X10 = X11 @ M10 @ X00（gemm）覆盖 ws_x 的 off-diag，ws_x 变完整 X
                pass

            # ---- V2 : gBK / vb / kg（↔ GBK + VBeta，与 C2a 并行）----
            with T.Scope("V"):
                # TODO(tile-level): ws_kbe = -β·g_exp·k（gBK 带负号，供 C2b 算 k_cumdecay）
                #                   ws_vb = v·β
                #                   ws_kg = k·g_exp[C-1]·g_exp_neg
                pass
            T.set_cross_flag("MTE3", 3)         # ↔ 0x6：gBK 写完 -> C2b 可算 k_cumdecay
            T.set_cross_flag("MTE3", 4)         # ↔ 0x5：vb 写完 -> C2b 可算 v_inner

            # ---- C2b : k_cumdecay / v_inner（↔ AIC k_cumdecay + v_inner）----
            with T.Scope("C"):
                T.wait_cross_flag(3)
                T.wait_cross_flag(4)
                # TODO(tile-level): T.gemm_v0(ws_x, ws_kbe, ws_k_cumdecay)   # X @ gBK（gBK 带负号）
                #                   T.gemm_v0(ws_x, ws_vb, ws_v_inner)      # X @ vb
                pass

            # ---- V3 : qPrime（↔ QPrimeCompute）----
            with T.Scope("V"):
                T.wait_cross_flag(5)            # ↔ 0xA：等 C1 写完 qkt（stage1 完成）
                # TODO(tile-level): ws_qprime = q·scale·g_exp（读 GM q 或 ws_q）
                pass
