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

"""Golden (reference) implementation of Chunked Gated Delta Rule.

Interface matches the CANN ChunkGatedDeltaRule operator (TND layout):
    query            (T, Nk, Dk)  bfloat16
    key              (T, Nk, Dk)  bfloat16
    value            (T, Nv, Dv)  bfloat16
    beta             (T, Nv)      bfloat16
    g                (T, Nv)      float32   (alpha = exp(g))
    initial_state    (B, Nv, Dv, Dk)  bfloat16 / float32
    actual_seq_lengths (B,)       int32
    out              (T, Nv, Dv)  bfloat16
    final_state      (B, Nv, Dv, Dk)

Each value head nv reads its own key head nv * Nk // Nv (head grouping,
Nv % Nk == 0).  The state matrix lives in (Dv, Dk) layout, matching the
kernel.  See README/design docs for the math.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


CHUNK_SIZE = 64


def _b16(x):
    """Quantize to bfloat16 (round to nearest) and keep fp32 for arithmetic.

    Emulates the AscendC kernel's bf16 intermediate storage: the cube matmul
    C output, stage-1/2/3 workspace tensors (vInner / kCumDecay / qPrime / kg /
    qkt / attn inverse / state) are all bf16 in the CANN implementation.  The
    golden follows the same numerics so the NPU verification gate matches.
    """
    return x.to(torch.bfloat16).to(torch.float32)


def chunk_gated_delta_rule(query, key, value, beta, g, initial_state,
                           actual_seq_lengths, scale=1.0, chunk_size=CHUNK_SIZE):
    """Reference (float32 + bf16-intermediate quantization) golden target."""
    T, Nk, Dk = query.shape
    Nv, Dv = value.shape[1], value.shape[2]
    B = initial_state.shape[0]
    device, dtype = query.device, query.dtype
    C = int(chunk_size)

    out = torch.zeros(T, Nv, Dv, dtype=dtype, device=device)
    final_state = torch.empty(B, Nv, Dv, Dk, dtype=initial_state.dtype, device=device)

    q = query.float() * scale
    k = key.float()
    v = value.float()
    beta_f = beta.float()
    g_f = g.float()

    seq_start = 0
    for b in range(B):
        L = int(actual_seq_lengths[b].item())
        seq_end = seq_start + L
        S = initial_state[b].float()  # (Nv, Dv, Dk)
        n_chunks = (L + C - 1) // C
        for ci in range(n_chunks):
            s0 = seq_start + ci * C
            s1 = min(seq_end, s0 + C)
            cur = s1 - s0
            for nv in range(Nv):
                nk_h = (nv * Nk) // Nv
                qc = q[s0:s1, nk_h, :].contiguous()          # (cur, Dk)
                kc = k[s0:s1, nk_h, :].contiguous()          # (cur, Dk)
                vc = v[s0:s1, nv, :].contiguous()            # (cur, Dv)
                bc = beta_f[s0:s1, nv].contiguous()          # (cur,)
                gc = g_f[s0:s1, nv].contiguous()             # (cur,)
                Snv = S[nv]                                  # (Dv, Dk)
                if cur < C:
                    qc = F.pad(qc, (0, 0, 0, C - cur))
                    kc = F.pad(kc, (0, 0, 0, C - cur))
                    vc = F.pad(vc, (0, 0, 0, C - cur))
                    bc = F.pad(bc, (0, C - cur))
                    gc = F.pad(gc, (0, C - cur))

                # --- within-chunk ---
                g_cum = torch.cumsum(gc, dim=0)                        # (C,)   fp32 (gCum fp32)
                g_cum_exp = torch.exp(g_cum)                           # (C,)   fp32 (gCumExp fp32)
                decay = torch.exp(g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2))
                decay_mask = torch.tril(decay)                         # (C, C) fp32
                kb = _b16(kc * bc.unsqueeze(-1))                       # k*beta, stored bf16
                vb = _b16(vc * bc.unsqueeze(-1))                       # v*beta (vBeta), stored bf16
                kbt = _b16(kb @ kc.t())                                # kk matmul C, bf16
                A = _b16((-kbt * decay_mask).tril(-1))                 # strictly lower, attnWs bf16
                attn = A.clone()
                for i in range(1, C):                                  # (I-A)^{-1}
                    attn[i, :i] = A[i, :i] + A[i, :i] @ attn[:i, :i]
                attn = _b16(attn + torch.eye(C, dtype=torch.float32, device=device))  # inverseRes bf16
                v_inner = _b16(attn @ vb)                              # matmul C, bf16
                kbe = _b16(kb * g_cum_exp.unsqueeze(-1))               # gBK stored bf16
                k_cumdecay = _b16(attn @ kbe)                          # matmul C, bf16

                # --- inter-chunk ---
                qkt = _b16(qc @ kc.t())                                # qkt matmul C, bf16 (qK)
                attn_c = _b16((qkt * decay_mask).tril())               # keep j <= i, bf16
                v_prime = _b16(k_cumdecay @ Snv.t())                   # matmul C, bf16
                v_new = _b16(v_inner - v_prime)                        # matmul B operand, bf16
                qg = _b16(qc * g_cum_exp.unsqueeze(-1))                # qPrime stored bf16
                attn_inter = _b16(qg @ Snv.t())                        # matmul C, bf16
                out_c = attn_inter + _b16(attn_c @ v_new)              # matmul C, bf16
                out[s0:s1, nv, :] = out_c[:cur].to(dtype)

                # --- state update: S = exp(g_last)*S + (k*exp(g_last-g_cum))^T @ v_new ---
                g_last = g_cum[-1]
                k_aug = _b16(kc * torch.exp(g_last - g_cum).unsqueeze(-1))   # kg stored bf16
                Snv = _b16(torch.exp(g_last) * Snv + (v_new.t() @ k_aug))    # state bf16 (curStateBf16)
                S[nv] = Snv
        final_state[b] = S.to(initial_state.dtype)
        seq_start = seq_end

    return out, final_state


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
