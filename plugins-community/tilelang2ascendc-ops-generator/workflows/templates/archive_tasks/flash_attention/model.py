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

import json
import os
from pathlib import Path

import torch
import torch.nn as nn


def case_is_causal(case):
    """用例的 is_causal：支持顶层键或 inputs 内 attr 项（GQA 约定）；缺省 False。"""
    if case.get("is_causal") is not None:
        return bool(case["is_causal"])
    for inp in case.get("inputs", []):
        if inp.get("type") == "attr" and inp.get("name") == "is_causal":
            return bool(inp.get("value"))
    return False


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                is_causal: bool = False):
        acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * (1.0 / q.shape[-1])**0.5
        if is_causal:
            # 右下对齐 causal（GQA golden 同语义）：行 i 可见列 j <= i + diff
            s_q, s_kv = q.shape[2], k.shape[2]
            diff = s_kv - s_q
            i = torch.arange(s_q, device=acc.device).view(s_q, 1)
            j = torch.arange(s_kv, device=acc.device).view(1, s_kv)
            acc = acc.masked_fill(j > (i + diff), float("-inf"))
        acc = acc.softmax(dim=-1)
        o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
        return o


def get_input_groups():
    # FA_CASE_FILE selects the case list; TileLang verification passes
    # general_tilelang_case.jsonl (TileLang does not support float32).
    # Case format matches the ops-profiling jsonl: {"inputs": [{type, shape, dtype}]}.
    case_name = os.environ.get("FA_CASE_FILE", "general_case.jsonl")
    case_path = Path(__file__).resolve().parent / case_name
    input_groups = []
    with case_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            inputs = case["inputs"]
            batch_size, n_heads, seq_len_q, d_k = inputs[0]["shape"]
            seq_len_kv = inputs[1]["shape"][2]
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[inputs[0]["dtype"]]
            q = torch.rand(batch_size, n_heads, seq_len_q, d_k, dtype=dtype)
            k = torch.rand(batch_size, n_heads, seq_len_kv, d_k, dtype=dtype)
            v = torch.rand(batch_size, n_heads, seq_len_kv, d_k, dtype=dtype)
            group = [q, k, v]
            if case_is_causal(case):
                group.append(True)
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
