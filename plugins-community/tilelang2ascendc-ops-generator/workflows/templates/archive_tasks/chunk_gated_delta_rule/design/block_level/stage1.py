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

"""Block-level 设计：Stage1 — within-chunk 预计算。

对应 AscendC：`kernel/op_kernel/chunk_gated_delta_rule_stage1.h` 的 `Stage1` 类。
输入整个 chunk-group（长度 ≤ maxGroupLength），输出到 GM workspace（bf16 / fp32）。

对组内每个 chunk（chunkSize=64），本阶段完成 chunk 内部所有与「历史状态无关」的量：

  g_cum    = cumsum(g)                                   # fp32（GCumExpCompute）
  g_cum_exp = exp(g_cum)                                 # fp32
  kb       = k * beta                                    # KKBetaCompute
  vb       = v * beta
  kbt      = kb @ k^T                                    # cube matmul
  A        = -(kbt * decay).tril(-1)                     # decay[i][j]=exp(g_cum[i]-g_cum[j])，严格下三角
  X        = (I - A)^{-1}                                # InverseAIV：列向 Gather 前代，对角=1
  v_inner  = X @ vb                                      # bf16 存储
  k_cumdecay = X @ (kb * exp(g_cum))                     # bf16 存储（GBKCompute）
  qPrime   = q * scale * exp(g_cum)                      # bf16 存储（QKPreProcess + scale）
  qkt      = q @ k^T                                     # bf16 存储（原始，未乘 decay）
  kg       = k * exp(g_last - g_cum)                     # bf16 存储（QKPreProcess 顺带产出）

Stage1 不接触状态 S；所有产物写回 workspace，供 Stage2/Stage3 使用。
AIC 负责 cube matmul（kbt、v_inner、k_cumdecay、qkt），AIV 负责向量部分
（cumsum/exp/decay/逆矩阵前代/逐元素乘积），经 workspace + 全核同步交给后续阶段。

注意：v_inner 这里指「chunk 内部反演后的值」（X @ vβ），不含历史状态贡献；
历史状态贡献（- k_cumdecay @ S）由 Stage2 叠加。数值上各产物均为 bf16，
逆矩阵 X 用 fp32 算完再转 bf16 落 workspace（与 AscendC 一致）。
"""


def stage1(
    query,
    key,
    value,
    beta,
    g,
    chunk_group,      # (startPos, length, chunkSize)
    g_cum_exp_out,    # (Nv, group_len)      fp32
    k_cumdecay_out,   # (Nv, group_len, Dk)  bf16
    v_inner_out,      # (Nv, group_len, Dv)  bf16
    q_prime_out,      # (Nv, group_len, Dk)  bf16
    qkt_out,          # (Nv, group_len, C)   bf16
    kg_out,           # (Nv, group_len, Dk)  bf16
):
    """对 chunk-group 内每个 chunk 计算 within-chunk 量（见顶部公式）。"""
    # AIC：kbt = kb @ k^T；v_inner = X @ vb；k_cumdecay = X @ (kb*exp(g_cum))；qkt = q @ k^T
    # AIV：g_cum/cumsum/exp；A = -(kbt*decay).tril(-1)；X = (I-A)^{-1}（InverseAIV 列向前代）
    #      qPrime = q*scale*exp(g_cum)；kg = k*exp(g_last-g_cum)
    # 全部写 workspace，完成后全核同步，交 Stage2。
    raise NotImplementedError("Stage1 计算细节见 tile_level 设计与 golden（model.py）")
