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

"""Block-level 设计：Stage1 — within-chunk 预计算（C1 / V1 / C2）。

对应 AscendC `Stage1`（stage1.h）。对当前 chunk（行窗口 [s0, s0+valid)）计算所有
「只依赖本 chunk、不依赖历史状态」的量，写 workspace（bf16/fp32）供 Stage2/3 用。

## C/V 子阶段（块内）

    C1  kbt = kb @ k^T                     （C×C，kb=k*β；cube matmul）
    V1  g_cum=cumsum(g)；g_exp/g_exp_neg=exp(±g_cum)
        decay[i][j] = g_exp[i]*g_exp_neg[j] (j<=i，否则 0)
        A = -(kbt * decay).tril(-1)        （严格下三角）
        X = (I - A)^{-1}                    （forward substitution，对角=1）
        kb/vb/kbe = k*β, v*β, kb*exp(g_cum)；qPrime = q*scale*exp(g_cum)；kg = k*exp(g_last-g_cum)
    C2  v_inner    = X @ vb                （C×Dv）
        k_cumdecay = X @ kbe               （C×Dk）

## 输入/输出

输入：query/key/value/beta/g + 行窗口 s0/valid + 头映射 (kh, h) + scale
输出（写 workspace，逐 chunk 复用）：
    ws_gcumexp(C)  fp32 | ws_kcumdecay(C,Dk) ws_vinner(C,Dv) ws_qprime(C,Dk) ws_qkt(C,C) ws_kg(C,Dk)  bf16

C/V 交接：C1 的 kbt 写 UB 给 V1；V1 的 X/kb/vb/kbe 给 C2；C2 产物经 workspace 落 GM，
阶段内同步（对应 AscendC AIC/AIV 事件）。
"""

import tilelang.language as T


def run_stage1(
    query, key, value, beta, g,
    s0, valid, kh, h, scale, unit, C, dk, dv,
    ws_gcumexp, ws_kcumdecay, ws_vinner, ws_qprime, ws_qkt, ws_kg,
):
    """当前 chunk 的 within-chunk 预计算，产物写 workspace。"""
    # ---- 输入加载（bf16 -> fp32 升型，尾部补零）----
    # TODO(tile-level): qc/kc/vc = T.copy(query[s0:s0+valid, kh, 0:dk], ...) 升型；
    #                    bc = T.copy(beta[s0:s0+valid, h])；gc = T.copy(g[s0:s0+valid, h])

    # ---- C1：kbt = kb @ k^T ----
    with T.Scope("C"):
        # TODO(tile-level): kb = kc*bc；T.gemm(kbt, kb, kT)   # kT = k^T (dk, C)
        pass

    # ---- V1：decay / A / X 逆矩阵 / 各逐元素量 ----
    with T.Scope("V"):
        # TODO(tile-level):
        #   g_cum = cumsum(gc)；g_exp, g_exp_neg = exp(±g_cum)
        #   decay[i][j] = g_exp[i]*g_exp_neg[j] (j<=i)
        #   A = -(kbt*decay).tril(-1)
        #   X = (I-A)^{-1}  forward substitution（对角=1）
        #   kbe = kb*g_exp；qPrime = qc*scale*g_exp；kg = kc*exp(g_last-g_cum)
        pass

    # ---- C2：v_inner / k_cumdecay ----
    with T.Scope("C"):
        # TODO(tile-level): T.gemm(v_inner, X, vb)；T.gemm(k_cumdecay, X, kbe)
        pass

    # ---- 产物写 workspace（bf16），供 Stage2/3 ----
    # TODO(tile-level): T.copy(g_exp, ws_gcumexp)；T.copy(降型, ws_kcumdecay/ws_vinner/...)
    T.barrier_all()
