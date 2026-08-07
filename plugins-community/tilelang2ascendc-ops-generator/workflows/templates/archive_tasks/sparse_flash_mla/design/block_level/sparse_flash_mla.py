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

"""Block-level TileLang design for sparse_flash_mla (SWA / HCA).

This file captures task decomposition, the two-segment KV walk, window-mask
strategy, and the flash-softmax state machine. Fine-grained math is filled in
by the tile-level design.

Coverage: SWA and HCA for any query sequence length (the decode row S1 == 1
is the headline scenario; prefill rows fall out of the same kernel through
runtime window bounds). CSA (per-query topk gather of ``cmp_kv`` rows) is an
extension point: implement it with ``T.tile.gather`` (element gather inside a
UB tile) or per-row dynamic GM copies of ``cmp_kv[b, idx, 0, :]``.

CSA index semantics (verified against the arch35 packaged-op kernel,
``ComputeS2LoopInfo``/``GetRealCmpS2Idx``, and the pytest golden): for query
row ``s`` only the first ``min(topk, threshold(s))`` entries of its
``cmp_sparse_indices`` row are attended — the per-query cmp-loop length is
capped by the causal threshold, not by topk. A ``-1`` entry terminates the
scan of that row; entries ``>= threshold(s)`` (causal overflow) are skipped,
the scan continues. ``threshold(s) = floor(causal(s) / cmp_ratio)`` as in
``model.py:cmp_threshold``. A gather-based extension must reproduce exactly
this prefix + stop + skip rule, because a bare ``topk``-length gather would
attend keys the packaged op never reads.
"""

import tilelang
import tilelang.language as T

from sparse_flash_mla_common import (
    MASK_NO,
    MASK_CAUSAL,
    MASK_BAND,
    UNLIMITED,
    BLOCK_N,
    ori_window_blocks,
    pass_configs,
)


# ---------------------------------------------------------------------------
#  Design summary (kept as docstrings below each section)
# ---------------------------------------------------------------------------

# Task partition:
#   Each kernel instance processes one decode query row:
#       q[b, 0, :, :]  ->  output[b, 0, :, :]
#   grid = batch. The full head set N1 (up to 128) is processed by the single
#   instance; KV_N == 1 and D == 512 are operator constraints, so no GQA
#   mapping or K-dim splitting is needed at block level.

# KV walk:
#   The attended key stream is the concatenation of two segments:
#       ori segment: ori_kv[b, win_start : win_end, 0, :]  (band/causal window)
#       cmp segment: cmp_kv[b, 0 : cmp_thr, 0, :]          (compressed prefix)
#   The kernel walks *all* KV blocks of both segments and masks out columns
#   outside the window / above the threshold. Masking instead of skipping
#   keeps the loop body branch-free for every block (the mask interval may be
#   a TIR expression of the query row, so no per-row branches either).

# Window masking strategy:
#   - block fully outside [start, end)  -> mask extent is zero, every score
#     column gets MASK_FILL (-2^30), so exp() underflows to zero
#   - block partially overlapping       -> mask leading/trailing score
#     columns with MASK_FILL
#   - cmp block columns >= cmp_thr      -> masked the same way

# Flash-softmax state (per query head row):
#   m_i     initialised to max(sinks)   (attention sinks act as virtual keys)
#   sumexp  initialised to sum(exp(sinks - m_i))
#   each block: m_old = m_i; m_i = max(m_i, rowmax(score));
#               alpha = exp(m_old - m_i)
#               sumexp = alpha * sumexp + rowsum(exp(score - m_i))
#               acc_o  = alpha * acc_o  + P @ V
#   epilogue: out = acc_o / sumexp
#   This matches model.py's softmax_with_sinks exactly (sinks normalise but
#   contribute no value).

# Synchronisation / buffering:
#   Single-kernel design, no cross-core communication: each instance owns
#   one query row, so no workspace or cross-scope flags are required.
#   C/V pipelining is left to the tile-level (double-buffered KV loads).


@tilelang.jit(pass_configs=pass_configs)
def sparse_flash_mla_fwd(
    batch,
    heads,
    dim,
    ori_seq,
    cmp_seq,
    cmp_ratio,
    ori_mask_mode,
    cmp_mask_mode,
    ori_win_left,
    ori_win_right,
):
    """Block-level kernel entry: decode-row task partition + KV walk.

    The signature mirrors the tile-level config; the body delegates the
    per-block math to the tile-level helpers, keeping the block level a
    structural description (T.Kernel -> KV loop -> mask/skip -> softmax
    state update).
    """
    dtype = "bfloat16"
    accum_dtype = "float"

    q_shape = [batch, heads, dim]
    kv_shape = [batch, ori_seq + cmp_seq, dim]  # logical concatenated KV
    out_shape = [batch, heads, dim]

    @T.prim_func
    def main(
        q: T.Tensor(q_shape, dtype),
        ori_kv: T.Tensor([batch, ori_seq, dim], dtype),
        cmp_kv: T.Tensor([batch, cmp_seq, dim], dtype),
        sinks: T.Tensor([heads], "float"),
        output: T.Tensor(out_shape, dtype),
    ):
        with T.Kernel(batch, is_npu=True) as (bz,):
            # block-level pseudo-body; real math lives in tile_level/
            # (kept as a compilable skeleton so the design stays executable)
            for t in T.serial(T.ceildiv(ori_seq + cmp_seq, BLOCK_N)):
                if t < T.ceildiv(ori_seq, BLOCK_N):
                    # ori segment block t: apply window mask then softmax step
                    pass
                else:
                    # cmp segment block: skip if index >= cmp_thr
                    pass

    return main
