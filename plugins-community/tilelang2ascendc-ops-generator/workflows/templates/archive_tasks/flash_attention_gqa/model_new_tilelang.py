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
            raise ValueError("q/k/v must be 4D [B, Hnq, S, D]")
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
        kv_heads = k.shape[1]
        kv_seq_len = int(k.shape[2])
        dtype_str = dtype_map[q.dtype]

        if dim % 64 != 0:
            raise ValueError("head dim (d_k) must be divisible by 64")

        # ---- 原生 GQA 调用（host 零 pad）----
        # kernel 直接用原始长度。q/k/v 不 pad，直接传原始 [B,nq,Sq,D] / [B,nkv,Skv,D]。
        # 尾块（K/V 尾块、Q 尾块越界子块）由 kernel 内自动处理：
        #   - K/V 尾块越界读靠 tilelang LegalizeSafeMemoryAccess 安全兜底（返回垃圾），
        #     tail_valid mask 排除越界列（softmax 概率 0）；
        #   - Q 尾块整块越界子块显式跳过，部分越界子块靠安全读 + 逐 half-row 行守卫。
        # kernel 输出即为原始 q_seq_len 长度，无需切片回原长。
        kernel = tl_flash_attention_gqa_fwd(
            batch,
            q_seq_len,
            kv_seq_len,
            n_heads,
            kv_heads,
            dim,
            dtype=dtype_str,
        )
        out = kernel(q, k, v)
        # 返回形状应为 [batch, n_heads, q_seq_len, dim]，与 ref 一致。
        return out


def get_init_inputs():
    return []
