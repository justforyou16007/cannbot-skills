#!/usr/bin/env python3
"""GQA 参考模型（cann-bench level4/gqa）。

golden 直调官方参考实现（task/golden.py 同构内联，见 desc.md §5）：
G = N_q // N_kv 折叠进 matmul 的 M 维，K/V 按 N_kv 头复用不物化。
用例来自 task/cases.csv → gqa.json（JSONL）。
"""
import json
import math
import os

import torch
import torch.nn as nn


def gqa(query, key, value, scaleValue=-1.0, is_causal=False):
    """官方 golden（task/golden.py 逐字内联）。"""
    B, S, N_q, D = query.shape
    S_kv = key.shape[1]
    N_kv = key.shape[2]

    if scaleValue <= 0:
        scaleValue = 1.0 / (D ** 0.5)

    G = N_q // N_kv
    q = query.reshape(B, S, N_kv, G, D).permute(0, 2, 3, 1, 4).reshape(B, N_kv, G * S, D)
    k = key.permute(0, 2, 1, 3)    # [B, N_kv, S_kv, D]
    v = value.permute(0, 2, 1, 3)  # [B, N_kv, S_kv, D]

    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    scores = scores.reshape(B, N_kv, G, S, S_kv)
    if is_causal:
        i = torch.arange(S, device=scores.device).unsqueeze(-1)
        j = torch.arange(S_kv, device=scores.device).unsqueeze(0)
        causal_mask = j > (i + (S_kv - S))
        scores = scores.masked_fill(causal_mask, float('-inf'))
    scores_max = scores.max(dim=-1, keepdim=True).values
    all_masked = torch.isinf(scores_max) & (scores_max < 0)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    attn_weights = torch.where(all_masked, torch.zeros_like(attn_weights), attn_weights)
    attn_weights = attn_weights.reshape(B, N_kv, G * S, S_kv)
    attn_output = torch.matmul(attn_weights, v)

    return attn_output.reshape(B, N_kv, G, S, D).permute(0, 3, 1, 2, 4).reshape(B, S, N_q, D).contiguous()


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, scaleValue=-1.0, is_causal=False):
        return gqa(query, key, value, scaleValue=scaleValue, is_causal=is_causal)


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gqa.json")
    cases = []
    with open(json_path) as f:
        decoder = json.JSONDecoder()
        buf = f.read()
    # 兼容 JSONL（逐行）与 pretty 多行 JSON：流式 raw_decode
    idx = 0
    n = len(buf)
    while idx < n:
        while idx < n and buf[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(buf, idx)
        cases.append(obj)
        idx = end

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    input_groups = []
    for case in cases:
        inputs = {i["name"]: i for i in case["inputs"]}
        dt = dtype_map[inputs["query"]["dtype"]]
        # value_range [-1, 1]：均匀分布
        q = torch.rand(inputs["query"]["shape"], dtype=torch.float32) * 2 - 1
        k = torch.rand(inputs["key"]["shape"], dtype=torch.float32) * 2 - 1
        v = torch.rand(inputs["value"]["shape"], dtype=torch.float32) * 2 - 1
        q = q.to(dt)
        k = k.to(dt)
        v = v.to(dt)
        scale = inputs.get("scaleValue", {}).get("value", -1.0)
        causal = inputs.get("is_causal", {}).get("value", False)
        input_groups.append([q, k, v, float(scale), bool(causal)])
    return input_groups


if __name__ == "__main__":
    groups = get_input_groups()
    print(f"loaded {len(groups)} cases")
    for i, g in enumerate(groups):
        q, k, v, s, c = g
        print(i, tuple(q.shape), tuple(k.shape), q.dtype, s, c)
