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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design.tile_level.chunk_gated_delta_rule import chunk_gated_delta_rule_host


def _build_kernel(query, key, value, actual_seq_lengths):
    seq_len = query.shape[0]
    nk = query.shape[1]
    dk = query.shape[2]
    nv = value.shape[1]
    dv = value.shape[2]
    batch = int(actual_seq_lengths.shape[0])
    # 手动传参 wrapper：host 分配全部 workspace/out/final_state 并显式传入
    # （tilelang 自动分配路径实测有首次运行竞态）。
    return chunk_gated_delta_rule_host(
        batch, seq_len, nk, nv, dk, dv,
        chunk_size=64,
        scale=1.0,
    )


class ModelNew(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                beta: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor,
                actual_seq_lengths: torch.Tensor):
        if query.ndim != 3:
            raise ValueError(f"query must be 3D (T, Nk, Dk), got {query.ndim}D")
        if value.ndim != 3:
            raise ValueError(f"value must be 3D (T, Nv, Dv), got {value.ndim}D")

        kernel = _build_kernel(query, key, value, actual_seq_lengths)
        out, final_state = kernel(query, key, value, beta, g, initial_state,
                                  actual_seq_lengths)
        return out, final_state


def get_init_inputs():
    return []
