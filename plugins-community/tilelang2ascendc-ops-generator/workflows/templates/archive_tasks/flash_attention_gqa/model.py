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

# GQA (Grouped Query Attention) flash attention golden reference.
# Logic taken from the new-source model.py:
#   q.view(B, nkv, group, Sq, D) + einsum + softmax -> [B, nq, Sq, D].
import csv
from pathlib import Path

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q_heads = q.shape[1]
        kv_heads = k.shape[1]
        if q_heads % kv_heads != 0:
            raise ValueError("q heads must be a multiple of kv heads")

        group_num = q_heads // kv_heads
        q_grouped = q.view(q.shape[0], kv_heads, group_num, q.shape[2], q.shape[3])
        scale = (1.0 / q.shape[-1]) ** 0.5
        acc = torch.einsum("bngsd,bnkd->bngsk", q_grouped, k) * scale
        acc = acc.softmax(dim=-1)
        o = torch.einsum("bngsk,bnkd->bngsd", acc, v)
        return o.reshape(q.shape[0], q_heads, q.shape[2], q.shape[3])


def _read_case_csvs():
    """Read basic_case.csv + general_case.csv into a list of param dicts.

    Both files sit next to model.py and share the header:
    batch_size,n_heads_q,n_heads_kv,seq_len_q,seq_len_kv,d_k,dtype
    """
    task_dir = Path(__file__).resolve().parent
    rows = []
    for name in ("basic_case.csv", "general_case.csv"):
        csv_path = task_dir / name
        if not csv_path.is_file():
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def get_input_groups():
    """Return a list of [q, k, v] input groups for NPU verification.

    The cases are loaded from basic_case.csv + general_case.csv and built by
    get_inputs_from_case(). verification_ascendc.py runs every returned group
    on NPU and compares it against the golden Model.
    """
    input_groups = []
    for params in _read_case_csvs():
        input_groups.append(get_inputs_from_case(params))
    return input_groups


def get_init_inputs():
    return []


def get_inputs_from_case(params: dict) -> list:
    """根据CSV测试用例参数构造输入张量"""
    batch_size = int(params['batch_size'])
    n_heads_q = int(params['n_heads_q'])
    n_heads_kv = int(params['n_heads_kv'])
    seq_len_q = int(params['seq_len_q'])
    seq_len_kv = int(params['seq_len_kv'])
    d_k = int(params['d_k'])
    dtype_str = params['dtype']
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)
    q = torch.rand(batch_size, n_heads_q, seq_len_q, d_k, dtype=dtype)
    k = torch.rand(batch_size, n_heads_kv, seq_len_kv, d_k, dtype=dtype)
    v = torch.rand(batch_size, n_heads_kv, seq_len_kv, d_k, dtype=dtype)
    return [q, k, v]


def get_init_inputs_from_case(params: dict) -> list:
    """根据CSV测试用例参数构造模型初始化参数"""
    return []
