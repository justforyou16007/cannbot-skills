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

import glob
import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_LIB_PATTERN = str(_KERNEL_BUILD / "chunk_gated_delta_rule_ext*")

# Prefer whl import (triggers TORCH_LIBRARY registration in register.cpp)
try:
    import chunk_gated_delta_rule_ext  # noqa: F401
except ImportError:
    # Fallback: direct .so loading from kernel/build/
    if _LIB_PATTERN not in "".join(sys.path):
        _libs = glob.glob(_LIB_PATTERN)
        if _libs:
            torch.ops.load_library(_libs[0])


class ModelNew(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                beta: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor,
                actual_seq_lengths: torch.Tensor):
        if query.dtype not in (torch.bfloat16,):
            raise ValueError(f"query must be bfloat16, got {query.dtype}")
        if initial_state.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"initial_state must be bfloat16 or float32, got {initial_state.dtype}")
        if not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous() and
                beta.is_contiguous() and initial_state.is_contiguous() and
                actual_seq_lengths.is_contiguous()):
            raise ValueError("inputs must be contiguous")

        if g is None:
            g = torch.zeros_like(beta, dtype=torch.float32)
        g = g.contiguous()

        out, final_state = torch.ops.npu.chunk_gated_delta_rule(
            query, key, value, beta, initial_state, actual_seq_lengths, g, self.scale)
        return out, final_state
