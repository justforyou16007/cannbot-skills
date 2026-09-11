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

# Matmul 参考实现（torch）：C[M, N] = A[M, K] @ B[K, N]

import json
import os
from pathlib import Path

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return a @ b


def get_input_groups():
    # MM_CASE_FILE selects the case list.
    case_name = os.environ.get("MM_CASE_FILE", "general_case.jsonl")
    case_path = Path(__file__).resolve().parent / case_name
    input_groups = []
    with case_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inputs = json.loads(line)["inputs"]
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[inputs[0]["dtype"]]
            a = torch.rand(*inputs[0]["shape"], dtype=dtype)
            b = torch.rand(*inputs[1]["shape"], dtype=dtype)
            input_groups.append([a, b])
    return input_groups


def get_init_inputs():
    return []
