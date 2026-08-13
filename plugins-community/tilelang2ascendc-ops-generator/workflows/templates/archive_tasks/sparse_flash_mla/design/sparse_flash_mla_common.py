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

"""Shared config and helpers for the sparse_flash_mla block/tile-level designs.

The TileLang design covers the SWA and HCA attention modes for any query
sequence length; the decode row (S1 == 1) is the headline scenario and folds
the per-row window bounds to compile-time constants. CSA requires per-query
topk gather on chip and is intentionally left as a design extension point
(see ``design/block_level/sparse_flash_mla.py``).
"""

import tilelang
import tilelang.language as T

# mask mode values (identical to the operator spec / reference model.py)
MASK_NO = 0
MASK_CAUSAL = 3
MASK_BAND = 4

UNLIMITED = -1

# a large negative fill value used for masked score columns (safe under exp)
MASK_FILL = -2.0 ** 30

# KV-block size for the decode kernel
BLOCK_N = 128

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def ori_window_blocks(q_pos, q_len, ori_len, mask_mode, win_left, win_right,
                      block_n=BLOCK_N):
    """First/last KV-block of ori_kv attended by query q_pos.

    Returns (block_lo, block_hi) such that only blocks
    ``[block_lo, block_hi)`` may contain valid tokens. Matches the slice
    semantics of ``model.py:ori_window`` rounded out to block granularity;
    per-column masking inside boundary blocks is applied by the kernel.
    """
    if mask_mode == MASK_NO:
        start, end = 0, ori_len
    else:
        causal = ori_len - q_len + q_pos + 1
        if mask_mode == MASK_CAUSAL:
            start, end = 0, max(causal, 0)
        elif mask_mode == MASK_BAND:
            end = ori_len if win_right == UNLIMITED else min(max(causal + win_right, 0), ori_len)
            start = 0 if win_left == UNLIMITED else max(causal - win_left - 1, 0)
        else:
            raise ValueError(f"unsupported ori_mask_mode: {mask_mode}")
    return start // block_n, (end + block_n - 1) // block_n
