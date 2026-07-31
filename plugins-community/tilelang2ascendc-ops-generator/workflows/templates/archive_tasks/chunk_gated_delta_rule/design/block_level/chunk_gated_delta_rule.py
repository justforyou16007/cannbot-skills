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

"""Block-level TileLang design for Chunked Gated Delta Rule.

This file captures the task decomposition, the chunk pipeline boundaries and
the state-carrier ownership.  Fine-grained math is left to the tile-level
design (see design/tile_level/chunk_gated_delta_rule.py).
"""

import tilelang
import tilelang.language as T

from design.chunk_gated_delta_rule_common import (
    CHUNK_SIZE,
    NUM_PHYSICAL_CORES,
    pass_configs,
)


@tilelang.jit(out_idx=[7], workspace_idx=[8], pass_configs=pass_configs)
def chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv, chunk_size=CHUNK_SIZE, scale=1.0):
    C = chunk_size
    n_chunks = (seq_len + C - 1) // C
    total_units = batch * nv
    used_core_num = min(NUM_PHYSICAL_CORES, max(1, total_units))
    tasks_per_core = T.ceildiv(total_units, used_core_num)

    @T.prim_func
    def main(
        query: T.Tensor((seq_len, nk, dk), "bfloat16"),
        key: T.Tensor((seq_len, nk, dk), "bfloat16"),
        value: T.Tensor((seq_len, nv, dv), "bfloat16"),
        beta: T.Tensor((seq_len, nv), "bfloat16"),
        g: T.Tensor((seq_len, nv), "float32"),
        initial_state: T.Tensor((batch, nv, dv, dk), "float32"),
        actual_seq_lengths: T.Tensor((batch,), "int32"),
        out: T.Tensor((seq_len, nv, dv), "bfloat16"),
        workspaces: tuple,
        final_state: T.Tensor((batch, nv, dv, dk), "float32"),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # Task partition: every (batch, value head) unit is independent.
            # Each core walks its share of units; the state carrier S is owned
            # by the unit and carried across its chunks.
            for local_idx in T.serial(tasks_per_core):
                unit = cid * tasks_per_core + local_idx
                if unit < total_units:
                    b = unit // nv
                    h = unit % nv
                    # State carrier: S (Dk, Dv), initialized from initial_state
                    # (transposed) at chunk 0 and written back to final_state
                    # after the last chunk.
                    #
                    # Pipeline boundaries (per chunk, executed in order):
                    #   1. within-chunk precompute
                    #        g_cum, decay, kb, vb          (tile-level: T.exp / T.cumsum / broadcast)
                    #   2. within-chunk solve
                    #        A -> (I - A)^{-1} -> v_inner, k_cumdecay
                    #        (tile-level: T.gemm + forward-substitution loop)
                    #   3. inter-chunk
                    #        v_prime, v_new, attn_inter, out
                    #        (tile-level: T.gemm)
                    #   4. state recurrence
                    #        S := exp(g_last)*S + (k*exp(g_last-g_cum))^T @ v_new
                    #   5. write out rows [chunk_start, chunk_end)
                    for ci in T.serial(n_chunks):
                        pass

    return main
