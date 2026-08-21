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

import csv
import math
from pathlib import Path

import torch
import torch.nn as nn

NEG_INF_FILL = -40000.0
SOFTMAX_STATS_LAST_DIM = 8


class Model(nn.Module):
    """FlashAttentionScoreGrad 的 torch 参考实现（fp32 计算）。

    与 kernel/op_host 中 FlashAttentionScoreGrad 算子的输入语义对齐：
    输入为 BNSD 布局，keep_prob=1.0（无 dropout）、无 pse；
    atten_mask 为 bool 张量，True 表示该位置被屏蔽（与算子语义一致）。
    softmax_max / softmax_sum / attention_in 为前向统计量与输出，由调用方提供。
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dy: torch.Tensor,
        softmax_max: torch.Tensor,
        softmax_sum: torch.Tensor,
        attention_in: torch.Tensor,
        atten_mask: torch.Tensor = None,
        scale: float = None,
    ):
        in_dtype = q.dtype
        qf, kf, vf, dyf = q.float(), k.float(), v.float(), dy.float()
        batch, n_heads_q, seq_q, d_qk = qf.shape
        n_heads_kv = kf.shape[1]
        if scale is None:
            scale = 1.0 / math.sqrt(d_qk)

        group = n_heads_q // n_heads_kv
        if group > 1:
            kf = kf.repeat_interleave(group, dim=1)
            vf = vf.repeat_interleave(group, dim=1)

        x_max = softmax_max[..., :1].float()
        x_sum = softmax_sum[..., :1].float()

        # 由前向统计量重建 P = exp(S - max) / sum，掩码位置显式置零
        scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
        if atten_mask is not None:
            scores = scores.masked_fill(atten_mask.bool(), NEG_INF_FILL)
        p = torch.exp(scores - x_max) / x_sum
        if atten_mask is not None:
            p = p.masked_fill(atten_mask.bool(), 0.0)

        # 反向公式（keep_prob=1.0，无 pse）：
        #   dv = P^T @ dy
        #   dp = dy @ V^T
        #   d  = rowsum(dy ∘ attention_in)
        #   ds = P ∘ (dp - d) * scale
        #   dq = ds @ K,  dk = ds^T @ Q
        dp = torch.matmul(dyf, vf.transpose(-1, -2))
        d_rowsum = (dyf * attention_in.float()).sum(dim=-1, keepdim=True)
        ds = p * (dp - d_rowsum) * scale
        dq = torch.matmul(ds, kf)
        dk = torch.matmul(ds.transpose(-1, -2), qf)
        dv = torch.matmul(p.transpose(-1, -2), dyf)

        if group > 1:
            # GQA：kernel 对 kv head 做组内归约，golden 对齐该语义
            b, _, s_kv, _ = kf.shape
            dk = dk.view(b, n_heads_kv, group, s_kv, d_qk).sum(dim=2)
            dv = dv.view(b, n_heads_kv, group, s_kv, vf.shape[-1]).sum(dim=2)

        return dq.to(in_dtype), dk.to(in_dtype), dv.to(in_dtype)


def _forward_stats(q, k, v, scale, atten_mask):
    """fp32 前向，生成 kernel 所需的 softmax_max / softmax_sum / attention_in。"""
    qf, kf, vf = q.float(), k.float(), v.float()
    n_heads_q = qf.shape[1]
    n_heads_kv = kf.shape[1]
    group = n_heads_q // n_heads_kv
    if group > 1:
        kf = kf.repeat_interleave(group, dim=1)
        vf = vf.repeat_interleave(group, dim=1)

    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    if atten_mask is not None:
        scores = scores.masked_fill(atten_mask.bool(), NEG_INF_FILL)
    x_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - x_max)
    x_sum = exp_scores.sum(dim=-1, keepdim=True)
    p = exp_scores / x_sum
    if atten_mask is not None:
        p = p.masked_fill(atten_mask.bool(), 0.0)
    attention_in = torch.matmul(p, vf)

    stats_shape = list(x_max.shape[:-1]) + [SOFTMAX_STATS_LAST_DIM]
    softmax_max = x_max.expand(stats_shape).contiguous()
    softmax_sum = x_sum.expand(stats_shape).contiguous()
    return softmax_max, softmax_sum, attention_in


def get_input_groups():
    case_path = Path(__file__).resolve().parent / "general_case.csv"
    input_groups = []
    with case_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            batch_size = int(row["batch_size"])
            n_heads_q = int(row["n_heads_q"])
            n_heads_kv = int(row["n_heads_kv"])
            seq_len_q = int(row["seq_len_q"])
            seq_len_kv = int(row["seq_len_kv"])
            d_qk = int(row["d_qk"])
            d_v = int(row["d_v"])
            causal = row["causal"].strip().lower() == "true"
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[row["dtype"]]

            scale = 1.0 / math.sqrt(d_qk)
            torch.manual_seed(0)
            q = torch.rand(batch_size, n_heads_q, seq_len_q, d_qk, dtype=dtype)
            k = torch.rand(batch_size, n_heads_kv, seq_len_kv, d_qk, dtype=dtype)
            v = torch.rand(batch_size, n_heads_kv, seq_len_kv, d_v, dtype=dtype)
            dy = torch.rand(batch_size, n_heads_q, seq_len_q, d_v, dtype=dtype)

            atten_mask = None
            if causal:
                atten_mask = torch.triu(
                    torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool), diagonal=1
                ).view(1, 1, seq_len_q, seq_len_kv)

            softmax_max, softmax_sum, attention_in = _forward_stats(q, k, v, scale, atten_mask)
            input_groups.append(
                [q, k, v, dy, softmax_max, softmax_sum, attention_in.to(dtype), atten_mask, scale]
            )
    return input_groups


def get_init_inputs():
    return []
