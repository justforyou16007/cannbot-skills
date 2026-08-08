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

"""bf16 三阶段参考实现（语义照抄官方 ops-transformer benchmark.py 的
chunk_gdn_benchmark_opt：solve_triangular 求逆 + 中间量按 bf16 量化）。

接口保持与验证脚本一致的 TND 布局：
    query            (T, Nk, Dk)  bfloat16
    key              (T, Nk, Dk)  bfloat16
    value            (T, Nv, Dv)  bfloat16
    beta             (T, Nv)      bfloat16
    g                (T, Nv)      float32   (alpha = exp(g))
    initial_state    (B, Nv, Dv, Dk)  bfloat16 / float32
    actual_seq_lengths (B,)       int32
    out              (T, Nv, Dv)  bfloat16
    final_state      (B, Nv, Dv, Dk)

计算按三阶段（对应算子 stage1/2/3 的数据流），中间量 v_beta/q_prime/attn_1/
k_cumdecay/kg/v_new/attn_inter/final_state/out 全部 bf16 量化，与官方 benchmark.py
语义一致。scale 只乘在 q_prime（stage1）与 masked_qkt（stage3），与官方一致。

每个 value head nv 读自己的 key head nv * Nk // Nv（head grouping，等价于官方
repeat_interleave，Nv % Nk == 0）。state 矩阵为 (Dv, Dk) 布局。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


CHUNK_SIZE = 64


def _stage1(query, key, value, g, beta, scale, C):
    """stage1：chunk 内 kkt/qkt/g_cum/求逆/v_inner/k_cumdecay/q_prime/kg。

    照抄官方 benchmark stage1_opt。返回 (g_cum, k_cumdecay, v_inner, q_prime, kg, qkt)。
    """
    S, Nk, Dk = query.shape
    _, Nv, Dv = value.shape
    device = query.device

    if Nv // Nk > 1:
        query = query.repeat_interleave(Nv // Nk, dim=1)
        key = key.repeat_interleave(Nv // Nk, dim=1)

    padded = (S + C - 1) // C * C
    n_chunks = padded // C
    pad = padded - S

    def pad_seq(t):
        if pad > 0:
            return F.pad(t, [0, 0] * (t.dim() - 1) + [0, pad])
        return t

    q_pad = pad_seq(query)  # (Sp, Nv, Dk)
    k_pad = pad_seq(key)
    v_pad = pad_seq(value)
    g_pad = pad_seq(g)
    b_pad = pad_seq(beta)

    # Batch over all heads & chunks
    q_all = q_pad.permute(1, 0, 2).reshape(Nv * n_chunks, C, Dk)
    k_all = k_pad.permute(1, 0, 2).reshape(Nv * n_chunks, C, Dk)
    v_all = v_pad.permute(1, 0, 2).reshape(Nv * n_chunks, C, Dv)
    g_all = g_pad.permute(1, 0).reshape(Nv * n_chunks, C)
    b_all = b_pad.permute(1, 0).reshape(Nv * n_chunks, C)

    k_f = k_all.float()
    b_f = b_all.float().unsqueeze(-1)

    kkt = (k_all @ k_all.transpose(-1, -2)).float() * b_f  # (Nv*nc, C, C)
    qkt = q_all @ k_all.transpose(-1, -2)  # (Nv*nc, C, C)

    g_cum = g_all.cumsum(dim=-1)  # (Nv*nc, C)
    g_cum_exp = g_cum.exp()  # (Nv*nc, C)

    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=-1)
    diff = g_cum[:, :, None] - g_cum[:, None, :]
    attn = (diff * lower).exp() * lower  # (Nv*nc, C, C)

    eye = torch.eye(C, dtype=torch.float32, device=device)
    attn_1 = torch.linalg.solve_triangular(eye + kkt * attn, eye, upper=False)
    attn_1 = attn_1.to(torch.bfloat16)  # (I-A)^{-1}，bf16

    kg = (k_f * (g_cum[:, -1, None] - g_cum).exp()[..., None]).to(torch.bfloat16)
    k_cumdecay = (b_f * g_cum_exp[:, :, None] * (-1) * k_f).to(torch.bfloat16)

    v_beta = (v_all.float() * b_f).to(torch.bfloat16)
    q_prime = (q_all.float() * scale * g_cum_exp[:, :, None]).to(torch.bfloat16)

    v_inner = attn_1 @ v_beta  # (Nv*nc, C, Dv)
    k_cumdecay = attn_1 @ k_cumdecay  # (Nv*nc, C, Dk)

    g_cum = g_cum.reshape(Nv, padded)
    k_cumdecay = k_cumdecay.reshape(Nv, padded, Dk)
    v_inner = v_inner.reshape(Nv, padded, Dv)
    q_prime = q_prime.reshape(Nv, padded, Dk)
    kg = kg.reshape(Nv, padded, Dk)
    qkt = qkt.reshape(Nv, padded, C)
    return g_cum, k_cumdecay, v_inner, q_prime, kg, qkt


def _stage2(q_prime, v_inner, g_cum, k_cumdecay, state, kg, C):
    """stage2：inter-chunk，逐 chunk 串行更新 state。

    照抄官方 benchmark stage2_opt。返回 (final_state, attn_inter, v_new)。
    """
    Nv, Sp, Dv = v_inner.shape
    _, _, Dk = q_prime.shape
    n_chunks = Sp // C
    device = q_prime.device

    attn_inter = torch.zeros((Nv, Sp, Dv), dtype=torch.bfloat16, device=device)
    v_new = torch.zeros((Nv, Sp, Dv), dtype=torch.bfloat16, device=device)

    cur_state = state  # (Nv, Dv, Dk) bf16
    for idx in range(n_chunks):
        s = idx * C
        e = s + C

        qg = q_prime[:, s:e, :]  # (Nv, C, Dk)
        vi = v_inner[:, s:e, :]  # (Nv, C, Dv)
        gc = g_cum[:, s:e]  # (Nv, C)
        kcd = k_cumdecay[:, s:e, :]  # (Nv, C, Dk)
        kgc = kg[:, s:e, :]  # (Nv, C, Dk)

        state_f = cur_state.float()  # (Nv, Dv, Dk)
        state_fT = state_f.transpose(-1, -2)  # (Nv, Dk, Dv)

        attn_inter_chunk = (qg.float() @ state_fT).to(torch.bfloat16)  # (Nv, C, Dv)
        v_prime = (kcd.float() @ state_fT).to(torch.bfloat16)  # (Nv, C, Dv)
        v_new_chunk = vi + v_prime  # (Nv, C, Dv) bf16

        state_out = v_new_chunk.transpose(-1, -2) @ kgc  # (Nv, Dv, Dk)
        decay = gc.exp()[:, -1][:, None, None]  # (Nv, 1, 1)
        state_old = (state_f * decay).to(torch.bfloat16)  # (Nv, Dv, Dk)
        cur_state = state_old + state_out  # (Nv, Dv, Dk) bf16

        attn_inter[:, s:e, :] = attn_inter_chunk
        v_new[:, s:e, :] = v_new_chunk

    final_state = cur_state.to(torch.bfloat16)
    return final_state, attn_inter, v_new


def _stage3(qkt, scale, g_cum, attn_inter, v_new, C):
    """stage3：out = attn_inter + masked_qkt @ v_new。

    照抄官方 benchmark stage3_opt。
    """
    Nv, Sp, Dv = attn_inter.shape
    device = qkt.device
    n_chunks = Sp // C

    qkt_all = qkt.reshape(Nv * n_chunks, C, C)
    g_all = g_cum.reshape(Nv * n_chunks, C)
    ai_all = attn_inter.reshape(Nv * n_chunks, C, Dv)
    vn_all = v_new.reshape(Nv * n_chunks, C, Dv)

    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=0)
    diff = g_all[:, :, None] - g_all[:, None, :]
    decay = (diff * lower).exp() * lower.float()  # (Nv*nc, C, C)
    masked_qkt = qkt_all.float() * scale * decay  # (Nv*nc, C, C)
    attn_inner = masked_qkt.to(torch.bfloat16) @ vn_all  # (Nv*nc, C, Dv)
    core_attn_out = (ai_all + attn_inner).to(torch.bfloat16)  # (Nv*nc, C, Dv)

    attn_out = core_attn_out.reshape(Nv, Sp, Dv).permute(1, 0, 2)  # (Sp, Nv, Dv)
    return attn_out


def chunk_gated_delta_rule(query, key, value, beta, g, initial_state,
                           actual_seq_lengths, scale=1.0, chunk_size=CHUNK_SIZE):
    """bf16 三阶段参考实现（语义照抄官方 benchmark.py）。

    interface 与验证脚本一致；state 沿 batch 逐条串行更新。
    """
    T, Nk, Dk = query.shape
    B, Nv, Dv, _ = initial_state.shape
    device, dtype = query.device, query.dtype
    C = int(chunk_size)

    attn_out = torch.empty((T, Nv, Dv), dtype=dtype, device=device)
    final_state = torch.empty(B, Nv, Dv, Dk, dtype=initial_state.dtype, device=device)

    start = 0
    for bid in range(B):
        L = int(actual_seq_lengths[bid].item())
        end = start + L
        cur_state = initial_state[bid].to(torch.bfloat16).clone()  # (Nv, Dv, Dk)

        g_cum, k_cum_decay, v_inner, q_prime, kg, qkt = _stage1(
            query[start:end], key[start:end], value[start:end],
            g[start:end], beta[start:end], scale, C)

        cur_state, attn_inter, v_new = _stage2(
            q_prime, v_inner, g_cum, k_cum_decay, cur_state, kg, C)

        final_state[bid] = cur_state.to(initial_state.dtype)
        attn_out_padded = _stage3(qkt, scale, g_cum, attn_inter, v_new, C)
        attn_out[start:end, ...] = attn_out_padded[:L]
        start = end

    return attn_out, final_state


class Model(nn.Module):
    def __init__(self, scale: float = 1.0, chunk_size: int = CHUNK_SIZE) -> None:
        super().__init__()
        self.scale = float(scale)
        self.chunk_size = int(chunk_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                beta: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor,
                actual_seq_lengths: torch.Tensor):
        return chunk_gated_delta_rule(
            query, key, value, beta, g, initial_state, actual_seq_lengths,
            scale=self.scale, chunk_size=self.chunk_size)


def _make_inputs(batch, n_heads, n_value_heads, d_key, d_value,
                 state_dtype, seed, seq_lens=None):
    """Build one test input group satisfying the operator's value constraints.

    seq_lens: per-batch sequence lengths (defaults to all equal to seq_len).
    """
    if seq_lens is None:
        seq_lens = [64] * batch
    seq_lens = list(seq_lens)
    total = int(sum(seq_lens))
    generator = torch.Generator().manual_seed(seed)
    q = (torch.randn(total, n_heads, d_key, generator=generator) * 0.5).to(torch.bfloat16)
    k = (torch.randn(total, n_heads, d_key, generator=generator) * 0.5).to(torch.bfloat16)
    v = (torch.randn(total, n_value_heads, d_value, generator=generator) * 0.5).to(torch.bfloat16)
    beta = (torch.rand(total, n_value_heads, generator=generator) * 0.5 + 0.25).to(torch.bfloat16)
    g = (-torch.rand(total, n_value_heads, generator=generator)).to(torch.float32)
    initial_state = (torch.randn(batch, n_value_heads, d_value, d_key,
                                 generator=generator) * 0.5).to(state_dtype)
    actual_seq_lengths = torch.tensor(seq_lens, dtype=torch.int32)
    return [q, k, v, beta, g, initial_state, actual_seq_lengths]


def get_input_groups():
    # NOTE: initial_state is forced to bfloat16 for all cases.  ASCEND910B3
    # does not support the FP32-state path (tiling rejects stateIsFp32 unless
    # the SoC is ASCEND950), so every case must use bf16 state.
    bf16 = torch.bfloat16
    cases = [
        {"batch": 1, "n_heads": 1, "n_value_heads": 1, "d_key": 32, "d_value": 32, "state_dtype": bf16, "seed": 1001, "seq_lens": [64]},
        {"batch": 1, "n_heads": 2, "n_value_heads": 4, "d_key": 32, "d_value": 32, "state_dtype": bf16, "seed": 1002, "seq_lens": [128]},
        {"batch": 2, "n_heads": 2, "n_value_heads": 2, "d_key": 64, "d_value": 64, "state_dtype": bf16, "seed": 1003, "seq_lens": [64, 64]},
        {"batch": 2, "n_heads": 1, "n_value_heads": 4, "d_key": 16, "d_value": 16, "state_dtype": bf16, "seed": 1004, "seq_lens": [96, 96]},
        {"batch": 2, "n_heads": 2, "n_value_heads": 4, "d_key": 32, "d_value": 64, "state_dtype": bf16, "seed": 1005, "seq_lens": [64, 96]},
    ]
    input_groups = []
    for case in cases:
        input_groups.append(_make_inputs(
            case["batch"], case["n_heads"], case["n_value_heads"],
            case["d_key"], case["d_value"], case["state_dtype"], case["seed"],
            seq_lens=case["seq_lens"]))
    return input_groups


def get_init_inputs():
    return []
