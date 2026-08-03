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

"""Block-level 设计：Stage1 — within-chunk 预计算（独立 jit kernel，对应 tile-level `stage1_ker`）。

对应 AscendC `Stage1`（stage1.h）。对当前 chunk（行窗口 `[s0, s0+valid)`）计算所有
「只依赖本 chunk、不依赖历史状态」的量，写 workspace（bf16/fp32）供 Stage2/3 用。
满足 `BlockLevelDesign-quality-checklist.md`（C/V 分工 #4、workspace #6、TODO #7、GEMM #10）。

## C/V 子阶段（checklist #4）

    V0  g_cum=cumsum(g)；g_exp/g_exp_neg=exp(±g_cum)      （向量）
    C1  kbt = kb @ k^T   （kb=k·β，cube matmul，T.gemm_v0）
    V1  decay[i][j]=g_exp[i]·g_exp_neg[j] (j<=i)
        A = -(kbt·decay) 下三角；X = (I-A)^{-1}（前代）；kb/vb/kbe/qPrime/kg（逐元素）
    C2  v_inner = X @ vb；k_cumdecay = X @ kbe            （cube matmul）

## 块内遍历（checklist #2）

    with T.Kernel(used_core_num):
        for unit in 分给本 kernel 的 (b, head) 集合:
            if b == 当前 batch 且 属于本 chunk:
                # 每单元：加载 chunk -> V0/C1/V1/C2 -> 写 workspace

## workspace 写（checklist #6）

    ws_gcumexp(C) fp32 | ws_kcumdecay(C,Dk) ws_vinner(C,Dv) ws_qprime(C,Dk) ws_qkt(C,C) ws_kg(C,Dk) bf16

## GEMM（checklist #10）

    kbt: (C,Dk)@(Dk,C)；v_inner: (C,C)@(C,Dv)；k_cumdecay: (C,C)@(C,Dk)——block 维 64/128 ≥ 16。
"""

import tilelang
import tilelang.language as T

from design.chunk_gated_delta_rule_common import CHUNK_SIZE, pass_configs


@tilelang.jit(out_idx=[], workspace_idx=[], pass_configs=pass_configs)
def stage1_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=CHUNK_SIZE, scale=1.0):
    C = chunk_size
    total_units = batch * nv
    used_core_num = min(20, total_units)   # 可换平台核数

    @T.prim_func
    def main(
        query: T.Tensor((seq_len, nk, dk), "bfloat16"),
        key: T.Tensor((seq_len, nk, dk), "bfloat16"),
        value: T.Tensor((seq_len, nv, dv), "bfloat16"),
        beta: T.Tensor((seq_len, nv), "bfloat16"),
        g: T.Tensor((seq_len, nv), "float32"),
        ci: T.Tensor((1,), "int32"),
        bb: T.Tensor((1,), "int32"),
        Lb: T.Tensor((1,), "int32"),
        ws_gcumexp: T.Tensor((C,), "float32"),
        ws_kcumdecay: T.Tensor((C, dk), "bfloat16"),
        ws_vinner: T.Tensor((C, dv), "bfloat16"),
        ws_qprime: T.Tensor((C, dk), "bfloat16"),
        ws_qkt: T.Tensor((C, C), "bfloat16"),
        ws_kg: T.Tensor((C, dk), "bfloat16"),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # TODO(tile-level): persistent 分配 unit，`b=unit//nv, h=unit%nv, kh=(h*nk)//nv`
            #                   `s0=base[b]+ci[0]*C, valid=T.if_then_else(Lb[0]-ci[0]*C<C, ...)`
            #                   `if b==bb[0]:` 只处理当前 batch

            # V0：加载 chunk 的 q/k/v/β/g（bf16→fp32 升型，逐标量 β/g，切片写全最后一维）
            with T.Scope("V"):
                # TODO(tile-level):
                #   qc/kc/vc = T.copy(query[s0:s0+valid, kh, 0:dk], ...) 升型
                #   bc/gc = 逐标量 beta[s0+t,h:h+1] / g[s0+t,h:h+1] -> [1] -> 升型
                #   g_cum=cumsum(gc)；g_exp/g_exp_neg=T.tile.exp(±g_cum)
                #   kb=kc*bc；vb=vc*bc；kbe=kb*g_exp；qPrime=qc*scale*g_exp；kg=kc*exp(g_last-g_cum)
                pass
            # C↔V 同步（checklist #5）：V0 产物（kb/vb）就绪 -> C1
            T.set_cross_flag("MTE3", 0)          # V0 -> C1

            # C1：kbt = kb @ k^T
            with T.Scope("C"):
                T.wait_cross_flag(0)             # 等 V0 的 kb 就绪
                # TODO(tile-level): T.gemm_v0(kb, kT, kbt)  # kT = k^T (dk, C)
                pass
            T.set_cross_flag("FIX", 1)          # C1 -> V1：kbt 就绪

            # V1：A / X 前代 / 各逐元素量
            with T.Scope("V"):
                T.wait_cross_flag(1)             # 等 C1 的 kbt 就绪
                # TODO(tile-level):
                #   decay[i][j]=g_exp[i]*g_exp_neg[j] (j<=i)
                #   A = -(kbt*decay).tril(-1)
                #   X = (I-A)^{-1} forward substitution（对角=1）
                pass
            T.set_cross_flag("MTE3", 2)          # V1 -> C2：X 就绪

            # C2：v_inner / k_cumdecay（写 workspace，bf16 降型）
            with T.Scope("C"):
                T.wait_cross_flag(2)             # 等 V1 的 X 就绪
                # TODO(tile-level):
                #   T.gemm_v0(X, vb, v_inner)；T.gemm_v0(X, kbe, k_cumdecay)
                #   T.copy 降型 -> ws_kcumdecay / ws_vinner / ws_qprime / ws_qkt / ws_kg
                pass

    return main
