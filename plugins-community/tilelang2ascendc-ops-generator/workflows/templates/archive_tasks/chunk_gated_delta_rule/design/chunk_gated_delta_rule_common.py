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

"""Shared config and helpers for the chunk_gated_delta_rule designs.

The algorithm is the chunked form of the Gated Delta Rule (linear attention).
Within a chunk of size C the computation is:

    g_cum    = cumsum(g)                       # (C,)
    decay    = exp(g_cum[i] - g_cum[j])        # lower-triangular (C, C)
    kb       = k * beta ; vb = v * beta
    A        = -(kb @ k^T) * decay             # strictly lower
    X        = (I - A)^{-1}                    # forward substitution
    v_inner  = X @ vb
    k_cumdecay = X @ (kb * exp(g_cum))
    attn_c   = (q @ k^T) * decay               # causal, keeps j <= i
    v_prime  = k_cumdecay @ S ;  v_new = v_inner - v_prime
    attn_inter = (q * exp(g_cum)) @ S
    out      = attn_inter + attn_c @ v_new
    S       := exp(g_cum[-1]) * S + (k*exp(g_cum[-1]-g_cum))^T @ v_new

The state S (Dk, Dv) is carried across chunks; different (batch, value-head)
units are independent and therefore parallelized over AIV cores.
"""

from collections import namedtuple

import tilelang
import tilelang.language as T

CHUNK_SIZE = 64
NUM_PHYSICAL_CORES = 20

# ---- chunk-group 协作模型常量（对齐 AscendC tiling，chunk_gated_delta_rule.cpp）----
P_NUM = 2                              # AscendC `P_NUM`：group 长度因子，maxGroupLength = P_NUM*aiCoreNum*chunkSize
STAGE_ONE_PARA_NUM = 4                 # AscendC `STAGE_ONE_PARA_NUM`：stage1 任务循环步进 paraNum（一次流水 4 个 chunk）
MAX_GROUP_LENGTH = P_NUM * NUM_PHYSICAL_CORES * CHUNK_SIZE   # maxGroupLength：一组覆盖行数
GROUP_CHUNKS = MAX_GROUP_LENGTH // CHUNK_SIZE                # 组内最大 chunk 数

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# Per-unit task context: which (batch, value head) and chunk this tile covers.
TaskContext = namedtuple("TaskContext", ["batch", "head", "chunk", "task_idx", "vid"])


def seq_slices(seq_lens, chunk_size=CHUNK_SIZE):
    """Return per (batch, chunk) (start, end, valid) row windows in the TND layout."""
    windows = []
    seq_start = 0
    for length in seq_lens:
        n_chunks = max(1, (int(length) + chunk_size - 1) // chunk_size)
        for ci in range(n_chunks):
            s0 = seq_start + ci * chunk_size
            s1 = min(seq_start + int(length), s0 + chunk_size)
            windows.append((s0, s1, s1 - s0))
        seq_start += int(length)
    return windows
