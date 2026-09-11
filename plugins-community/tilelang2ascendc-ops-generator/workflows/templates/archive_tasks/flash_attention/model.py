#!/usr/bin/env python3
"""MHA 参考模型（cann-bench level4/mha）。

golden 直调官方参考实现（task/golden.py 同构内联）：标准 MHA，BSND 布局，
causal 右下角对齐，scaleValue<=0 时自动 1/sqrt(D)。
用例来自 task/cases.csv。
"""
import csv
import json
import os

import torch
import torch.nn as nn


def mha(query, key, value, scaleValue=-1.0, is_causal=False):
    """官方 golden（task/golden.py 逐字内联）。"""
    B, S, N, D = query.shape
    S_kv = key.shape[1]

    if scaleValue <= 0:
        scaleValue = 1.0 / (D ** 0.5)

    # BSND -> BNSD
    q = query.permute(0, 2, 1, 3)  # [B, N, S, D]
    k = key.permute(0, 2, 1, 3)    # [B, N, S_kv, D]
    v = value.permute(0, 2, 1, 3)  # [B, N, S_kv, D]

    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue  # [B, N, S, S_kv]
    if is_causal:
        i = torch.arange(S, device=scores.device).unsqueeze(-1)
        j = torch.arange(S_kv, device=scores.device).unsqueeze(0)
        causal_mask = j > (i + (S_kv - S))
        scores = scores.masked_fill(causal_mask, float('-inf'))
    # F217 全 mask 行保护（golden.py L69-74 同构）：全 -inf 行 softmax 后置 0
    scores_max = scores.max(dim=-1, keepdim=True).values
    all_masked = torch.isinf(scores_max) & (scores_max < 0)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    attn_weights = torch.where(all_masked, torch.zeros_like(attn_weights), attn_weights)
    attn_output = torch.matmul(attn_weights, v)  # [B, N, S, D]

    return attn_output.permute(0, 2, 1, 3).contiguous()  # [B, S, N, D]


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, scaleValue=-1.0, is_causal=False):
        return mha(query, key, value, scaleValue=scaleValue, is_causal=is_causal)


def get_input_groups():
    torch.manual_seed(42)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "task", "cases.csv")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    input_groups = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            shapes = json.loads(row["input_shape"])
            dtypes = json.loads(row["dtype"])
            attrs = json.loads(row["attrs"])
            dt = dtype_map[dtypes[0]]
            # value_range [-1, 1]：均匀分布
            q = (torch.rand(shapes[0], dtype=torch.float32) * 2 - 1).to(dt)
            k = (torch.rand(shapes[1], dtype=torch.float32) * 2 - 1).to(dt)
            v = (torch.rand(shapes[2], dtype=torch.float32) * 2 - 1).to(dt)
            scale = float(attrs.get("scaleValue", -1.0))
            causal = bool(attrs.get("is_causal", False))
            input_groups.append([q, k, v, scale, causal])
    return input_groups


if __name__ == "__main__":
    groups = get_input_groups()
    print(f"loaded {len(groups)} cases")
    for i, g in enumerate(groups):
        q, k, v, s, c = g
        print(i, tuple(q.shape), tuple(k.shape), q.dtype, s, c)
