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

Mirrors block_level/stage1.py / AscendC `Stage1`（stage1.h），但为单 kernel 内联函数。
**C/V 分工与同步完全照抄 AscendC（AIC/AIV 并行流水，6 flag）**：
**任务分片封装在此函数内部**（对应 stage1.h `Stage1::Process`）：`totalChunk = nv × num_chunk`
按 `coreIdx` 均分（former/tail），`taskId` 反解 `(nvId, cgId)`；workspace 为 AscendC 布局
`(Nv, maxGroupLength, ...)`，本 chunk 行偏移 `row = cgId * C`（stage1.h `SetChunkOffset`）。

    V0a  prep：ws_q = q*scale；ws_k = k；β 存 bc                          （↔ QKPreProcess）
    C1   kkt = k @ k^T（纯 k）；qkt = q @ k^T                             （↔ kkt + qkt，与 V0b 并行）
    V0b  g_cum=cumsum(g)；g_exp/g_exp_neg=exp(±g_cum)                     （↔ GCumExp，与 C1 并行）
    V1a  attn_1 = kkt·β·decay（β 行乘 bc[i]，↔ KKBeta+Gamma）；
         A=-(attn_1).tril(-1)；X 对角 AIV 前代 + off-diag M10 暂存         （↔ InverseCompute）
    C2a  off-diag X10 = X11·M10·X00（覆盖 ws_x）                          （↔ AttnInverseMMCompute）
    V2   ws_kbe = -β·g_exp·k（gBK 带负号）；ws_vb = v·β；ws_kg             （↔ GBK + VBeta，与 C2a 并行）
    C2b  k_cumdecay = X @ gBK；v_inner = X @ vb                           （↔ k_cumdecay + v_inner）
    V3   ws_qprime = q·scale·g_exp                                        （↔ QPrimeCompute）

同步 6 flag：0↔0x9（prep→kkt）、1↔0x8（kkt→KKBeta）、5↔0xA（qkt→stage1 完成）、
2↔0x7（求逆对角→off-diag）、3↔0x6（gBK→k_cumdecay）、4↔0x5（vb→v_inner）。
**set/wait_cross_flag 必须放在对应 scope 内**（set 放发出方 scope、wait 放接收方 scope），
否则 tilelang 把它生成到主流程（AIC+AIV 都执行），flag 语义失效（实测 UUB 崩溃）。

**buffer 分配纪律**：本 stage 全部 on-chip buffer 在函数顶部一次分配（静态），任务分片循环
（`for cgId / for nvId`）内只复用，不再 alloc。
"""


import tilelang.language as T


def key_head(h, nk, nv):
    """The value head *h* reads key head (h*Nk)//Nv (head grouping)."""
    return (h * nk) // nv


@T.macro(hygienic=False)
def run_stage1(cid, used_core_num, group_chunks,
               nk, nv, dk, dv, C, F, scale,
               gpos, cg_len, num_chunk,
               query, key, value, beta, g,
               ws_q, ws_k, ws_kbe, ws_qprime, ws_kg, ws_vb,
               ws_kbt, ws_qkt, ws_x, ws_v_inner, ws_k_cumdecay,
               ws_g_exp, ws_g_exp_neg):
    # 求逆分块边长 F 由入口按 Python 常量传入（宏体内 `F = C // 2` 会被 TVM script
    # 解析成自由 Var，使 (F,F) buffer shape 非常量 → LowerTileOp makeBufferWithLayout 崩）。

    # ---- 统一分配本 stage 全部 on-chip buffer（函数顶部，分片循环外静态复用）----
    # V0a/V2/V3 elementwise prep
    q_buf = T.alloc_ub((C, dk), "bfloat16")
    k_buf = T.alloc_ub((C, dk), "bfloat16")
    v_buf = T.alloc_ub((C, dv), "bfloat16")
    f = T.alloc_ub((C, dk), "float32")
    fv = T.alloc_ub((C, dv), "float32")
    temp_k = T.alloc_ub((C, dk), "bfloat16")
    temp_v = T.alloc_ub((C, dv), "bfloat16")
    bc = T.alloc_ub((C,), "float32")          # β（V0a 填，V1a/V2 读）
    gc = T.alloc_ub((C,), "float32")
    beta_b16 = T.alloc_ub((1,), "bfloat16")
    beta_f32 = T.alloc_ub((1,), "float32")
    g_f32 = T.alloc_ub((1,), "float32")
    g_cum = T.alloc_ub((C,), "float32")
    g_exp = T.alloc_ub((C,), "float32")
    g_neg = T.alloc_ub((C,), "float32")
    g_exp_neg = T.alloc_ub((C,), "float32")
    # C1 kkt/qkt（纯 k）
    k_l1 = T.alloc_L1((C, dk), "bfloat16")
    k_l1b = T.alloc_L1((C, dk), "bfloat16")   # kkt = k@k^T 的两个操作数
    kkt_l0 = T.alloc_L0C((C, C), "float32")
    q_l1 = T.alloc_L1((C, dk), "bfloat16")
    qkt_l0 = T.alloc_L0C((C, C), "float32")
    # V1a A / X = (I-A)^{-1}
    kkt_bf = T.alloc_ub((C, C), "bfloat16")
    kkt_ub = T.alloc_ub((C, C), "float32")
    g_expb = T.alloc_ub((C,), "float32")
    g_exp_negb = T.alloc_ub((C,), "float32")
    attn_ub = T.alloc_ub((C, C), "float32")
    acc1 = T.alloc_ub((1,), "float32")
    x_bf = T.alloc_ub((C, C), "bfloat16")
    # C2a off-diag 求逆（小块 (F,F) cube buffer）
    o11_l1 = T.alloc_L1((F, F), "bfloat16")
    p10_l1 = T.alloc_L1((F, F), "bfloat16")
    t10_l0 = T.alloc_L0C((F, F), "float32")
    o00_l1 = T.alloc_L1((F, F), "bfloat16")
    p10b_l1 = T.alloc_L1((F, F), "bfloat16")
    o10_l0 = T.alloc_L0C((F, F), "float32")
    # C2b X @ vb / X @ gBK
    x_l1 = T.alloc_L1((C, C), "bfloat16")
    vb_l1 = T.alloc_L1((C, dv), "bfloat16")
    vi_l0 = T.alloc_L0C((C, dv), "float32")
    kbe_l1 = T.alloc_L1((C, dk), "bfloat16")
    kd_l0 = T.alloc_L0C((C, dk), "float32")

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

    for cgId in T.serial(group_chunks):
        for nvId in T.serial(nv):
            taskId = nvId + cgId * nv
            if taskId >= start and taskId < end:
                s0 = gpos + cgId * C
                valid = T.if_then_else(cg_len - cgId * C < C, cg_len - cgId * C, C)
                row = cgId * C
                kh = (nvId * nk) // nv      # 该 value head 读的 key head（head 分组）

                # ---- V0a : prep（ws_q = q*scale；ws_k = k；β 存 bc）----
                with T.Scope("V"):
                    T.tile.fill(q_buf, 0.0)
                    T.copy(query[s0:s0 + valid, kh, 0:dk], q_buf)
                    T.tile.fill(k_buf, 0.0)
                    T.copy(key[s0:s0 + valid, kh, 0:dk], k_buf)
                    T.tile.fill(v_buf, 0.0)
                    T.copy(value[s0:s0 + valid, nvId, 0:dv], v_buf)
                    T.tile.fill(bc, 0.0)
                    for t in T.serial(C):
                        if t < valid:
                            T.copy(beta[s0 + t, nvId:nvId + 1], beta_b16)
                            T.set_flag("mte2", "v", 1)
                            T.wait_flag("mte2", "v", 1)
                            T.copy(beta_b16, beta_f32)
                            bc[t] = beta_f32[0]
                        else:
                            bc[t] = 0.0
                    # ws_q = q*scale ; ws_k = k（f 就地乘，省 cb 中转）
                    T.copy(q_buf, f)
                    for t in T.serial(C):
                        for d in T.serial(dk):
                            f[t, d] = f[t, d] * scale
                    T.copy(f, temp_k)
                    T.copy(temp_k, ws_q[nvId, row, 0])
                    T.copy(k_buf, f)
                    T.copy(f, temp_k)
                    T.copy(temp_k, ws_k[nvId, row, 0])
                    T.set_cross_flag("MTE3", 0)          # ↔ 0x9：prep 写完 -> C1 可算 kkt/qkt（V scope 内 = AIV 侧）

                # ---- C1 : kkt = k@k^T（纯 k）；qkt = q@k^T（与 V0b 并行）----
                with T.Scope("C"):
                    T.wait_cross_flag(0)
                    T.copy(ws_k[nvId, row, 0], k_l1)
                    T.copy(ws_k[nvId, row, 0], k_l1b)
                    T.gemm_v0(k_l1, k_l1b, kkt_l0, transpose_B=True, init=True)
                    T.copy(kkt_l0, ws_kbt[nvId, row, 0])
                    T.copy(ws_q[nvId, row, 0], q_l1)
                    T.gemm_v0(q_l1, k_l1, qkt_l0, transpose_B=True, init=True)
                    T.copy(qkt_l0, ws_qkt[nvId, row, 0])
                    T.set_cross_flag("FIX", 1)          # ↔ 0x8：kkt 写完 -> V1a 可算（C scope 内 = AIC 侧）
                    T.set_cross_flag("FIX", 5)          # ↔ 0xA：qkt 写完 -> V3 wait（C scope 内 = AIC 侧）

                # ---- V0b : g_cum/g_exp/g_exp_neg（与 C1 并行）----
                with T.Scope("V"):
                    T.tile.fill(gc, 0.0)
                    for t in T.serial(C):
                        if t < valid:
                            T.copy(g[s0 + t, nvId:nvId + 1], g_f32)
                            gc[t] = g_f32[0]
                        else:
                            gc[t] = 0.0
                    g_cum[0] = gc[0]
                    for t in T.serial(1, C):
                        g_cum[t] = g_cum[t - 1] + gc[t]
                    for t in T.serial(C):
                        g_neg[t] = 0.0 - g_cum[t]
                    T.tile.exp(g_exp, g_cum)
                    T.tile.exp(g_exp_neg, g_neg)
                    T.copy(g_exp, ws_g_exp[nvId, row])
                    T.copy(g_exp_neg, ws_g_exp_neg[nvId, row])

                # ---- V1a : attn_1 = kkt*β*decay；A；X 对角前代 + off-diag 暂存 ----
                with T.Scope("V"):
                    T.wait_cross_flag(1)
                    T.copy(ws_kbt[nvId, row, 0], kkt_bf)
                    T.copy(kkt_bf, kkt_ub)
                    T.copy(ws_g_exp[nvId, row], g_expb)
                    T.copy(ws_g_exp_neg[nvId, row], g_exp_negb)
                    for i in T.serial(C):
                        for j in T.serial(C):
                            if j < i:
                                # kbt[i,j] = β[i]·kkt[i,j]（β 行乘，照抄 KKBetaCompute）
                                # A 就地覆盖 kkt_ub（省 a_ub 中转）
                                kkt_ub[i, j] = -(kkt_ub[i, j] * bc[i] * g_expb[i] * g_exp_negb[j])
                            else:
                                kkt_ub[i, j] = 0.0
                    # A 保持 fp32 直接进求逆（照抄 KKBetaCompute+InverseCompute：
                    # attnUbFloat 乘完 beta/exp(差) 后不进 bf16，直接 InverseAIV）。
                    for i in T.serial(C):
                        for j in T.serial(C):
                            attn_ub[i, j] = kkt_ub[i, j]
                    # AIC/AIV 协同求逆 ((I-A)^{-1}), 对应 solve_tril_64_ker:
                    #   A = [[M00,0],[M10,M11]] (对角线以下), O = [[O00,0],[O10,O11]].
                    #   - AIV (V pipe): 对角子块 O00 / O11 用 V 向量前代 (对应 AscendC InverseAIV).
                    #   - AIC (C pipe):  非对角子块 O10 = O11 @ M10 @ O00 用 cube gemm, 在 C2a 完成.
                    # -- O00 = (I + M00)^-1, 行 0..F --（A 已就地存在 kkt_ub）
                    for i in T.serial(1, F):
                        for j in T.serial(i):
                            acc1[0] = 0.0
                            for jp in T.serial(i):
                                acc1[0] += kkt_ub[i, jp] * attn_ub[jp, j]
                            attn_ub[i, j] = kkt_ub[i, j] + acc1[0]
                    for i in T.serial(F):
                        attn_ub[i, i] = 1.0
                    # -- O11 = (I + M11)^-1, 行 F..C --
                    for ii in T.serial(1, F):
                        for jj in T.serial(ii):
                            acc1[0] = 0.0
                            for jp in T.serial(F, ii + F):
                                acc1[0] += kkt_ub[ii + F, jp] * attn_ub[jp, jj + F]
                            attn_ub[ii + F, jj + F] = kkt_ub[ii + F, jj + F] + acc1[0]
                    for i in T.serial(F, C):
                        attn_ub[i, i] = 1.0
                    # -- off-diag: 暂存 M10 = A[F:C,0:F] (负值) 供 C2a gemm 用 --
                    for i in T.serial(F, C):
                        for j in T.serial(F):
                            attn_ub[i, j] = kkt_ub[i, j]
                    T.copy(attn_ub, x_bf)
                    T.copy(x_bf, ws_x[nvId, row, 0])
                    T.set_cross_flag("MTE3", 2)         # ↔ 0x7：求逆对角写完 -> C2a 可算 off-diag（V scope 内 = AIV 侧）

                # ---- C2a : off-diag 求逆（X10 = X11·M10·X00，覆盖 ws_x）----
                with T.Scope("C"):
                    T.wait_cross_flag(2)
                    T.copy(ws_x[nvId, row + F:row + C, F:C], o11_l1)          # O11
                    T.copy(ws_x[nvId, row + F:row + C, 0:F], p10_l1)          # M10 (负)
                    T.gemm_v0(o11_l1, p10_l1, t10_l0, init=True)      # O11 @ M10
                    T.copy(t10_l0, ws_x[nvId, row + F:row + C, 0:F])
                    T.copy(ws_x[nvId, row:row + F, 0:F], o00_l1)             # O00
                    T.copy(ws_x[nvId, row + F:row + C, 0:F], p10b_l1)        # (O11 @ M10) round 到 bf16
                    T.gemm_v0(p10b_l1, o00_l1, o10_l0, init=True)     # (O11@M10) @ O00
                    T.copy(o10_l0, ws_x[nvId, row + F:row + C, 0:F])         # ws_x 现为完整 X

                # ---- V2 : gBK / vb / kg（与 C2a 并行）----
                with T.Scope("V"):
                    # gBK = -β·g_exp·k -> ws_kbe（带负号，供 C2b 算 k_cumdecay；f 就地）
                    T.copy(k_buf, f)
                    for t in T.serial(C):
                        for d in T.serial(dk):
                            f[t, d] = 0.0 - (f[t, d] * bc[t] * g_exp[t])
                    T.copy(f, temp_k)
                    T.copy(temp_k, ws_kbe[nvId, row, 0])
                    # vb = v*β
                    T.copy(v_buf, fv)
                    for t in T.serial(C):
                        for d in T.serial(dv):
                            fv[t, d] = fv[t, d] * bc[t]
                    T.copy(fv, temp_v)
                    T.copy(temp_v, ws_vb[nvId, row, 0])
                    # kg = k·g_exp[C-1]·g_exp_neg（f 就地）
                    T.copy(k_buf, f)
                    for t in T.serial(C):
                        for d in T.serial(dk):
                            f[t, d] = f[t, d] * g_exp[C - 1] * g_exp_neg[t]
                    T.copy(f, temp_k)
                    T.copy(temp_k, ws_kg[nvId, row, 0])
                    T.set_cross_flag("MTE3", 3)         # ↔ 0x6：gBK 写完 -> C2b 可算 k_cumdecay（V scope 内 = AIV 侧）
                    T.set_cross_flag("MTE3", 4)         # ↔ 0x5：vb 写完 -> C2b 可算 v_inner（V scope 内 = AIV 侧）

                # ---- C2b : k_cumdecay = X@gBK；v_inner = X@vb ----
                with T.Scope("C"):
                    T.wait_cross_flag(3)
                    T.wait_cross_flag(4)
                    T.copy(ws_x[nvId, row, 0], x_l1)
                    T.copy(ws_kbe[nvId, row, 0], kbe_l1)
                    T.gemm_v0(x_l1, kbe_l1, kd_l0, init=True)          # X @ gBK（gBK 带负号）
                    T.copy(kd_l0, ws_k_cumdecay[nvId, row, 0])
                    T.copy(ws_vb[nvId, row, 0], vb_l1)
                    T.gemm_v0(x_l1, vb_l1, vi_l0, init=True)           # X @ vb
                    T.copy(vi_l0, ws_v_inner[nvId, row, 0])

                # ---- V3 : qPrime = q*scale*g_exp（等 qkt 写完，stage1 收尾）----
                with T.Scope("V"):
                    T.wait_cross_flag(5)            # ↔ 0xA：等 C1 写完 qkt（stage1 完成）
                    T.copy(q_buf, f)
                    for t in T.serial(C):
                        for d in T.serial(dk):
                            f[t, d] = f[t, d] * scale * g_exp[t]
                    T.copy(f, temp_k)
                    T.copy(temp_k, ws_qprime[nvId, row, 0])
