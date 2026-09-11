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

# TileLang matmul 入口（ascendc 后端）。设计：design/tile_level/matmul.py
# （per-shape 编译；最终交付物为手写泛化 AscendC kernel，运行期 M/N/K）。

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design.tile_level.matmul import MatmulConfig, matmul_fwd as tl_matmul_fwd

# (dtype, M, N, K) -> 已编译 kernel 的进程级缓存
_KERNEL_CACHE: dict = {}


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Expected 2D tensors, got {a.ndim}D and {b.ndim}D")
        if a.dtype != b.dtype:
            raise ValueError(f"dtype mismatch: {a.dtype} vs {b.dtype}")
        dtype_str = str(a.dtype).split(".")[-1]
        if dtype_str not in ("float16", "bfloat16", "float32"):
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        M, K = a.shape
        K2, N = b.shape
        if K2 != K:
            raise ValueError(f"K mismatch: {K} vs {K2}")

        key = (dtype_str, M, N, K)
        kernel = _KERNEL_CACHE.get(key)
        if kernel is None:
            cfg = MatmulConfig(m=M, n=N, k=K, dtype=dtype_str)
            kernel = tl_matmul_fwd(cfg)
            _KERNEL_CACHE[key] = kernel
        return kernel(a, b)


def get_init_inputs():
    return []
