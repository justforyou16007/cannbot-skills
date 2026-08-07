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

"""Tile-level TileLang design for sparse_flash_mla (SWA / HCA).

One kernel instance per batch row (grid = batch). Inside, the kernel walks
``q`` rows ``s`` and query-head blocks sequentially; the attended KV stream of
row ``s`` is the concatenation of the ori window ``[start(s), end(s))`` and
the compressed prefix ``[0, cmp_thr(s))`` of ``cmp_kv``. The wrapper lays
both segments into a single ``kv_all`` tensor (``[ori_pad | cmp_pad]``) and
the kernel walks it block by block (``BLOCK_N``) in ONE loop; out-of-window
columns are masked with ``MASK_FILL`` (-2^30, safe under exp) instead of
being skipped, which keeps the loop body branch-free. ``start/end/cmp_thr``
are TIR expressions of ``s``, so prefill rows (q_pos > 0) fall out of the
same kernel as the decode row (q_pos == 0); the decode case simply makes
them compile-time constants.

Flash-softmax state carries the attention sinks as virtual keys::

    m_i     := sinks                       (init)
    sumexp  := 1                           (init, = exp(sinks - m_i))
    per block: m_old = m_i
               m_i    = max(m_i, rowmax(score))
               alpha  = exp(m_old - m_i)
               sumexp = alpha * sumexp + rowsum(exp(score - m_i))
               acc_o  = alpha * acc_o  + softmax(score) @ V
    epilogue: out = acc_o / sumexp

This matches ``model.py:softmax_with_sinks`` exactly.

Implementation notes (all verified on 910B hardware):

* The kernel body is written straight-line: tilelang-ascend silently drops
  the statements of any helper function called from the traced body, so no
  helper is called for statement emission (scalar-bound helpers that return
  TIR expressions are fine).
* L0C lives on the cube (AIC) pipe, UB on the vector (AIV) pipe; the two AIVs
  of a MIX kernel each own half of the BLOCK_H rows (``vid``). Cross-pipe
  copies are staged through GM workspaces by the TL_ASCEND_AUTO_CV_SYNC
  pass. Statement order must avoid a circular cross-core wait: the AIV's P
  write (read by the AIC PV gemm) must not wait on the AIC's acc_o staging,
  so the P cast / UB->L1 copy precedes the PV gemm, and the O accumulation
  (``acc_o = alpha * acc_o + P @ V``) happens in UB after reading back
  ``P @ V`` from L0C. The PV gemm therefore re-inits every block.
* All three cross-pipe copies must live in ONE serial loop: the sync pass
  requires per-workspace cube/vec sync-point counts to match, and a second
  loop reusing the same workspaces breaks that invariant.
* ``idx`` is a [BLOCK_N] f32 host-side constant (arange) used to build the
  per-column window mask arithmetically; it is a kernel parameter so no
  device-side constant buffer is needed.
* The caller passes a pre-allocated output tensor (the out_idx return-value
  path is not used).
"""

from dataclasses import dataclass

import tilelang
import tilelang.language as T

from sparse_flash_mla_common import (
    MASK_NO,
    MASK_CAUSAL,
    MASK_BAND,
    UNLIMITED,
    MASK_FILL,
    BLOCK_N,
)

# query-head block size: 32 rows keep one QK tile [BH, BLOCK_N] fp32 in L0C
BLOCK_H = 32

# per-AIV row half of a head block (2 vector cores per MIX kernel)
BLOCK_H_HALF = BLOCK_H // 2

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@dataclass(frozen=True)
class SparseFlashMlaConfig:
    """Compile-time kernel configuration."""
    batch: int
    q_seq: int
    heads: int
    dim: int
    ori_seq: int
    cmp_seq: int = 0
    cmp_ratio: int = 1
    ori_mask_mode: int = MASK_BAND
    cmp_mask_mode: int = MASK_CAUSAL
    ori_win_left: int = 127
    ori_win_right: int = 0
    softmax_scale: float = -1.0   # <= 0 -> default 1 / sqrt(dim)
    dtype: str = "bfloat16"


def _ori_window_start(causal, ori_seq, mask_mode, wl):
    """First valid token of ori_kv for query row ``s`` (TIR expr or int)."""
    if mask_mode == MASK_NO or wl == UNLIMITED:
        return 0
    if mask_mode == MASK_CAUSAL:
        return 0
    return T.max(causal - wl - 1, 0)


def _ori_window_end(causal, ori_seq, mask_mode, wr):
    """Exclusive end of the valid ori_kv window for query row ``s``."""
    if mask_mode == MASK_NO:
        return ori_seq
    if mask_mode == MASK_CAUSAL:
        return T.max(causal, 0)
    if wr == UNLIMITED:
        return ori_seq
    return T.max(T.min(causal + wr, ori_seq), 0)


def _cmp_thr(causal, ori_seq, cmp_seq, cmp_ratio, mask_mode):
    """Number of leading valid cmp_kv tokens for query row ``s``."""
    if cmp_seq <= 0:
        return 0
    if mask_mode == MASK_NO:
        return T.min(ori_seq // cmp_ratio, cmp_seq)
    return T.max(T.min(causal // cmp_ratio, cmp_seq), 0)


def sparse_flash_mla_fwd(cfg: SparseFlashMlaConfig):
    """Build and compile the sparse_flash_mla kernel for ``cfg``.

    Returns a compiled kernel callable as ``kernel(q, kv_all, sinks, idx,
    output)`` with ``kv_all`` ([B, ori_pad + cmp_pad, D], both S dimensions
    padded to BLOCK_N) built by the wrapper, ``sinks`` ([N1] f32),
    ``idx`` ([BLOCK_N] f32 arange, mask helper constant) and a pre-allocated
    ``output`` ([B, S1, N1, D] in the config dtype).
    """
    batch, q_seq, heads, dim = cfg.batch, cfg.q_seq, cfg.heads, cfg.dim
    ori_seq, cmp_seq = cfg.ori_seq, cfg.cmp_seq
    cmp_ratio = cfg.cmp_ratio
    dtype = cfg.dtype
    accum_dtype = "float"
    sm_scale = cfg.softmax_scale if cfg.softmax_scale > 0 else dim ** -0.5
    wl, wr = cfg.ori_win_left, cfg.ori_win_right

    # S dimensions are padded to block alignment so every KV copy is a static
    # full-block copy; padded columns fall outside every window and are
    # masked out below.
    ori_pad = ((ori_seq + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    cmp_pad = max(((cmp_seq + BLOCK_N - 1) // BLOCK_N) * BLOCK_N, 1)
    head_blocks = (heads + BLOCK_H - 1) // BLOCK_H
    hb = BLOCK_H_HALF

    q_shape = [batch, q_seq, heads, dim]
    kv_shape = [batch, ori_pad + cmp_pad, dim]
    out_shape = [batch, q_seq, heads, dim]

    @T.prim_func
    def main(
        q: T.Tensor(q_shape, dtype),
        kv_all: T.Tensor(kv_shape, dtype),
        sinks: T.Tensor([heads], "float"),
        idx: T.Tensor([BLOCK_N], "float"),
        output: T.Tensor(out_shape, dtype),
    ):
        with T.Kernel(batch, is_npu=True) as (cid, vid):
            # ---- cube-side (AIC) buffers -----------------------------------
            q_l1 = T.alloc_L1([BLOCK_H, dim], dtype)
            kv_l1 = T.alloc_L1([BLOCK_N, dim], dtype)
            p_l1 = T.alloc_L1([BLOCK_H, BLOCK_N], dtype)
            scores_l0c = T.alloc_L0C([BLOCK_H, BLOCK_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([BLOCK_H, dim], accum_dtype)
            # ---- vector-side (AIV) buffers; half of the head rows each -----
            scores_ub = T.alloc_ub([hb, BLOCK_N], accum_dtype)
            acc_o_ub = T.alloc_ub([hb, dim], accum_dtype)
            acc_o_new = T.alloc_ub([hb, dim], accum_dtype)
            acc_o_half = T.alloc_ub([hb, dim], dtype)
            p_ub = T.alloc_ub([hb, BLOCK_N], dtype)
            p_ub_nz = T.alloc_ub([hb, BLOCK_N], dtype)
            m_i = T.alloc_ub([hb], accum_dtype)
            m_old = T.alloc_ub([hb], accum_dtype)
            sumexp = T.alloc_ub([hb], accum_dtype)
            rowsum = T.alloc_ub([hb], accum_dtype)
            idx_ub = T.alloc_ub([BLOCK_N], accum_dtype)
            mask_lo = T.alloc_ub([BLOCK_N], accum_dtype)
            mask_hi = T.alloc_ub([BLOCK_N], accum_dtype)
            mask_col = T.alloc_ub([BLOCK_N], accum_dtype)

            T.copy(idx, idx_ub)

            for s in T.serial(q_seq):
                # per-row window bounds (TIR expressions of s; the decode row
                # q_pos == 0 makes causal == ori_seq and folds to constants)
                causal = ori_seq - q_seq + s + 1
                o_start = _ori_window_start(causal, ori_seq,
                                            cfg.ori_mask_mode, wl)
                o_end = _ori_window_end(causal, ori_seq,
                                        cfg.ori_mask_mode, wr)
                cmp_thr = _cmp_thr(causal, ori_seq, cmp_seq, cmp_ratio,
                                   cfg.cmp_mask_mode)
                ori_blocks = T.ceildiv(o_end, BLOCK_N)
                cmp_blocks = T.ceildiv(cmp_thr, BLOCK_N)
                cmp_base = ori_blocks * BLOCK_N  # logical index of cmp start
                total_blocks = ori_blocks + cmp_blocks
                kv_shift = ori_pad - cmp_base   # GM offset of cmp segment

                for h in T.serial(head_blocks):
                    # heads % BLOCK_H == 0 is a wrapper contract, so every
                    # head block is full and all slices stay static
                    h_lo = h * BLOCK_H

                    # load this (s, hb) query tile + init flash state
                    T.copy(q[cid, s, h_lo:h_lo + BLOCK_H, :], q_l1)
                    T.copy(sinks[h_lo + vid * hb:h_lo + vid * hb + hb], m_i)
                    T.tile.fill(sumexp, T.float32(1.0))
                    T.tile.fill(acc_o_ub, T.float32(0.0))

                    # fused ori+cmp block loop (single loop; see module doc)
                    for t in T.serial(total_blocks):
                        # AIC: KV tile + QK
                        kv_off = t * BLOCK_N + T.if_then_else(
                            t < ori_blocks, 0, kv_shift)
                        T.copy(kv_all[cid, kv_off:kv_off + BLOCK_N, :], kv_l1)
                        T.gemm_v0(q_l1, kv_l1, scores_l0c,
                                  transpose_B=True, init=True)
                        T.copy(scores_l0c, scores_ub)
                        # AIV: scale + per-column window mask (branch-free)
                        T.tile.mul(scores_ub, scores_ub, sm_scale)
                        lo = T.if_then_else(t < ori_blocks, o_start, cmp_base)
                        hi = T.if_then_else(t < ori_blocks, o_end,
                                            cmp_base + cmp_thr)
                        lo2 = T.min(T.max(lo - t * BLOCK_N, 0), BLOCK_N)
                        hi2 = T.min(T.max(hi - t * BLOCK_N, 0), BLOCK_N)
                        lo2f = lo2.astype("float32")
                        hi2f = hi2.astype("float32")
                        T.tile.add(mask_lo, idx_ub, 1.0 - lo2f)
                        T.tile.max(mask_lo, mask_lo, 0.0)
                        T.tile.min(mask_lo, mask_lo, 1.0)
                        T.tile.mul(mask_hi, idx_ub, -1.0)
                        T.tile.add(mask_hi, mask_hi, hi2f)
                        T.tile.max(mask_hi, mask_hi, 0.0)
                        T.tile.min(mask_hi, mask_hi, 1.0)
                        T.tile.mul(mask_lo, mask_lo, mask_hi)  # valid ∈ {0,1}
                        # scores = scores * valid + MASK_FILL * (1 - valid);
                        # a multiplicative mask avoids the fp32 catastrophic
                        # cancellation of adding/subtracting MASK_FILL.
                        T.tile.mul(mask_hi, mask_lo, -1.0)
                        T.tile.add(mask_hi, mask_hi, 1.0)     # 1 - valid
                        T.tile.mul(mask_col, mask_hi, T.float32(MASK_FILL))
                        for hi in range(hb):
                            T.tile.mul(scores_ub[hi, :], scores_ub[hi, :],
                                       mask_lo)
                            T.tile.add(scores_ub[hi, :], scores_ub[hi, :],
                                       mask_col)
                        # AIV: flash-softmax update
                        T.copy(m_i, m_old)
                        T.reduce_max(scores_ub, m_i, dim=-1)
                        T.tile.max(m_i, m_i, m_old)
                        T.tile.sub(m_old, m_old, m_i)
                        T.tile.exp(m_old, m_old)
                        for hi in range(hb):
                            T.tile.sub(scores_ub[hi, :], scores_ub[hi, :],
                                       m_i[hi])
                        T.tile.exp(scores_ub, scores_ub)
                        T.reduce_sum(scores_ub, rowsum, dim=-1)
                        T.tile.mul(sumexp, sumexp, m_old)
                        T.tile.add(sumexp, sumexp, rowsum)
                        # AIV: P cast + UB -> L1 (before PV, no circular wait)
                        T.copy(scores_ub, p_ub)
                        T.copy(p_ub, p_l1, tmp=p_ub_nz)
                        # AIC: PV fresh per block, staged back to UB
                        T.gemm_v0(p_l1, kv_l1, acc_o_l0c, init=True)
                        T.copy(acc_o_l0c, acc_o_new)
                        # AIV: acc_o = alpha * acc_o + P @ V
                        for hi in range(hb):
                            T.tile.mul(acc_o_ub[hi, :], acc_o_ub[hi, :],
                                       m_old[hi])
                            T.tile.add(acc_o_ub[hi, :], acc_o_ub[hi, :],
                                       acc_o_new[hi, :])

                    # epilogue: normalize, cast, store
                    for hi in range(hb):
                        T.tile.div(acc_o_ub[hi, :], acc_o_ub[hi, :],
                                   sumexp[hi])
                    T.copy(acc_o_ub, acc_o_half)
                    T.copy(acc_o_half,
                           output[cid, s,
                                  h_lo + vid * hb:h_lo + vid * hb + hb, :])

    return tilelang.compile(main, pass_configs=pass_configs, target="ascendc")
