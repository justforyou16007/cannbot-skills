#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# ----------------------------------------------------------------------------------------------------------

# TileLang flash attention 入口（ascendc 后端）：handsync
# （AUTO_SYNC 关闭手写同步 + dim<=512 分档 + V2 压缩免零初始化/整幅
#  broadcast-mul + V1 brcb-sub dim<=128 门控）。

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from design.tile_level.flash_attention_handsync import (
    MAX_DIM,
    FlashAttentionConfig,
    flash_attention_handsync_fwd as tl_flash_attention_fwd,
)

# (dtype, batch, heads, q_seq_len, kv_seq_len, dim, causal) -> 已编译 kernel 的进程级缓存
_KERNEL_CACHE: dict = {}


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                is_causal: bool = False):
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                f"Expected 4D tensors, got q.ndim={q.ndim}, k.ndim={k.ndim}, v.ndim={v.ndim}"
            )
        if not (q.dtype == k.dtype == v.dtype):
            raise ValueError(
                f"All input dtypes must match, got q={q.dtype}, k={k.dtype}, v={v.dtype}"
            )

        dtype_str = str(q.dtype).split(".")[-1]
        if dtype_str not in ("float16", "bfloat16", "float32"):
            raise ValueError(
                f"Unsupported dtype: {dtype_str}, expected float16, bfloat16 or float32"
            )

        batch, heads, q_seq_len, dim = q.shape
        kv_seq_len = k.shape[2]

        if dim > MAX_DIM:
            raise ValueError(
                f"head dim {dim} exceeds MAX_DIM={MAX_DIM}"
            )

        cache_key = (dtype_str, batch, heads, q_seq_len, kv_seq_len, dim, is_causal)
        kernel = _KERNEL_CACHE.get(cache_key)
        if kernel is None:
            cfg = FlashAttentionConfig(
                batch=batch,
                heads=heads,
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len,
                dim=dim,
                dtype=dtype_str,
                causal=is_causal,
            )
            kernel = tl_flash_attention_fwd(cfg)
            _KERNEL_CACHE[cache_key] = kernel

        return kernel(q, k, v, 1.0 / (dim ** 0.5))
