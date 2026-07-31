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

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_LIB_PATTERN = str(_KERNEL_BUILD / "flash_attention_gqa_ext*")

# Prefer whl import (triggers TORCH_LIBRARY registration in register.cpp)
try:
    import flash_attention_gqa_ext  # noqa: F401
except ImportError:
    # Fallback: direct .so loading from kernel/build/
    if _LIB_PATTERN not in "".join(sys.path):
        import glob as _glob

        _libs = _glob.glob(_LIB_PATTERN)
        if _libs:
            torch.ops.load_library(_libs[0])


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q/k/v must be 4D [B, H, S, D]")
        if q.shape[1] % k.shape[1] != 0:
            raise ValueError("q heads must be a multiple of kv heads")
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("q/k/v must have the same dtype")
        if not all(t.is_contiguous() for t in (q, k, v)):
            raise ValueError("q/k/v must be contiguous")

        return torch.ops.npu.flash_attention_gqa(q, k, v)


def get_init_inputs():
    return []
