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

import sys
from pathlib import Path

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))
if str(_TASK_DIR / "design") not in sys.path:
    sys.path.insert(0, str(_TASK_DIR / "design"))

from design.tile_level.flash_attention_gqa import (  # noqa: E402
    flash_attention_gqa_fwd as tl_flash_attention_gqa_fwd,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q/k/v must be 4D [B, H, S, D]")
        if q.shape[1] % k.shape[1] != 0:
            raise ValueError("q heads must be a multiple of kv heads")
        if not (q.dtype == k.dtype == v.dtype):
            raise ValueError("q/k/v must have the same dtype")
        dtype_map = {torch.float16: "float16", torch.bfloat16: "bfloat16"}
        if q.dtype not in dtype_map:
            raise ValueError(
                f"tilelang GQA design supports float16/bfloat16 only, got {q.dtype}"
            )

        batch, n_heads, q_seq_len, dim = q.shape
        if dim % 64 != 0:
            raise ValueError("head dim (d_k) must be divisible by 64")
        kv_heads = k.shape[1]
        n_kv = batch * kv_heads
        group = n_heads // kv_heads

        # GQA pack: turn the MHA kernel into a "heads = 1" problem by folding the
        # group dimension into the query sequence length.
        #
        #   q: [B, n_heads, Sq, D] -> [B, n_kv, group, Sq, D] -> [B*n_kv, group*Sq, D]
        packed_q = q.reshape(batch, kv_heads, group, q_seq_len, dim).reshape(
            n_kv, group * q_seq_len, dim
        )
        #   k/v: [B, n_kv, Skv, D] -> [B*n_kv, Skv, D]
        flat_k = k.reshape(n_kv, k.shape[2], dim)
        flat_v = v.reshape(n_kv, v.shape[2], dim)

        # The migrated kernel declares K/V as [batch, heads, kv_seq_len_padded, dim]
        # (kv_seq_len padded up to a whole 64-wide block).  When Skv is not a
        # multiple of 64, the input must be padded to that length so the kernel's
        # tail-block reads stay in bounds (zeros), letting the tail mask work.
        kv_seq_len = int(k.shape[2])
        kv_padded = ((kv_seq_len + 63) // 64) * 64
        if kv_padded != kv_seq_len:
            pad = kv_padded - kv_seq_len
            flat_k = torch.nn.functional.pad(flat_k, (0, 0, 0, pad))
            flat_v = torch.nn.functional.pad(flat_v, (0, 0, 0, pad))

        dtype_str = dtype_map[q.dtype]
        kernel = tl_flash_attention_gqa_fwd(
            n_kv,
            1,
            group * q_seq_len,
            kv_seq_len,
            dim,
            dtype=dtype_str,
        )
        packed_o = kernel(packed_q, flat_k, flat_v)

        # [B*n_kv, 1, group*Sq, D] -> [B*n_kv, group*Sq, D] -> [B, n_kv, group, Sq, D]
        # -> [B, n_heads, Sq, D]
        out = packed_o.reshape(n_kv, 1, group * q_seq_len, dim)
        out = out.reshape(batch, kv_heads, group, q_seq_len, dim).reshape(
            batch, n_heads, q_seq_len, dim
        )
        return out


def get_init_inputs():
    return []
