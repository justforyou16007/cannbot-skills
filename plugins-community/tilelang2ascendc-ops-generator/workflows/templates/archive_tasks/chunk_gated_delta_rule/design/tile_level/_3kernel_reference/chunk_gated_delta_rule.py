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

"""Tile-level TileLang design (expert mode + cube) for Chunked Gated Delta Rule.

Refactor of the prior single-file implementation into AscendC-style
stage1/2/3 + host orchestration, mirrored on `design/block_level/`:

    chunk_gated_delta_rule.py   entry / orchestration   <-> block_level/chunk_gated_delta_rule.py
    stage1.py                   within-chunk precompute <-> block_level/stage1.py
    stage2.py                   cross-chunk recurrence  <-> block_level/stage2.py
    stage3.py                   within-chunk accumulate <-> block_level/stage3.py

Orchestration choice — THREE standalone jit kernels (stage1/2/3) launched in
stream order from this entry (host does only the batch/chunk group loop + state
transport in torch):
    for b in range(batch):
      seqStart..seqEnd by actual_seq_lengths
      for ci in 0..n_chunks(b):
        curState = initial_state[b] if ci==0 else state (carried via GM S tensor)
        stage1_ker(ci) -> workspace
        stage2_ker(ci, curState) -> out(attn_inter), S' (GM S + final_state)
        stage3_ker(ci)           -> out += attn_c @ v_new

Why THREE kernels (the sanctioned fallback when single-kernel barrier fails):
1. The old single kernel's C<->V cross-flag pipeline was reused across chunk
   iterations with no stage-boundary synchronisation, so it desynchronised for
   any unit with >=2 full chunks (case1/2/4 ~97% wrong).  Each kernel launch is a
   hard stream-ordered host barrier and re-arms the cross-flag handoff, so no
   transaction leaks into the next chunk (case1 now matches).
2. The single kernel also accumulated every stage's UB buffer allocations into
   one kernel; at dk=dv=64 (case2) they exceeded UB and produced garbage.  Each
   kernel here only holds its own stage's buffers, so dk=dv=64 fits (case2/case4).

All matmuls are cube `T.gemm_v0`, vector ops in UB, every bf16 boundary goes
through two-stage `T.copy`; expert mode (no T.parallel / T.pipeline).  state
(Dv,Dk); final_state/initial_state (B,Nv,Dv,Dk); value head h reads key head
(h*Nk)//Nv.
"""

import torch

from design.tile_level.stage1 import stage1_ker
from design.tile_level.stage2 import stage2_ker
from design.tile_level.stage3 import stage3_ker


def chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv,
                           chunk_size=64, scale=1.0):
    """Return a callable runner (mirrors block_level entry) that, given the
    operator tensors, builds workspace, runs the batch/chunk-group pipeline of
    stage1/2/3 kernels and returns (out, final_state)."""
    def run(query, key, value, beta, g, initial_state, actual_seq_lengths):
        device = query.device
        total_units = batch * nv
        C = int(chunk_size)

        # per-chunk workspace (one frag per unit, reused across chunks)
        def w3(specs, dt2):
            return {n: torch.zeros(sh, dtype=dt2, device=device)
                    for (n, sh) in specs}
        ws_u_dk = w3([("q", (total_units, C, dk)), ("k", (total_units, C, dk)),
                      ("kb", (total_units, C, dk)), ("kbe", (total_units, C, dk)),
                      ("qprime", (total_units, C, dk)), ("kg", (total_units, C, dk)),
                      ("v_inner", (total_units, C, dv)), ("k_cumdecay", (total_units, C, dk))],
                     torch.bfloat16)
        ws_u_dv = w3([("vb", (total_units, C, dv)), ("v_prime", (total_units, C, dv)),
                      ("attn_inter", (total_units, C, dv)), ("v_new", (total_units, C, dv)),
                      ("out_cont", (total_units, C, dv))], torch.bfloat16)
        ws_u_cc = w3([("kbt", (total_units, C, C)), ("qkt", (total_units, C, C)),
                      ("x", (total_units, C, C)), ("attn_c", (total_units, C, C))],
                     torch.bfloat16)
        ws_u_g = w3([("g_exp", (total_units, C)), ("g_exp_neg", (total_units, C))],
                    torch.float32)
        ws_u_dstate = w3([("dstate", (total_units, dk, dv))], torch.bfloat16)
        ws_u_ds = w3([("S", (total_units, dv, dk))], torch.bfloat16)

        out = torch.zeros((seq_len, nv, dv), dtype=query.dtype, device=device)
        final_state = torch.empty((batch, nv, dv, dk), dtype=initial_state.dtype,
                                  device=device)

        # One kernel per stage; ci and bb are runtime tensors so no per-chunk
        # repackaging is needed (build is fast, ~3 kernels).
        k1 = stage1_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=C, scale=scale)
        k2 = stage2_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=C)
        k3 = stage3_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=C, scale=scale)

        ci = torch.tensor([0], dtype=torch.int32, device=device)
        bb = torch.tensor([0], dtype=torch.int32, device=device)
        seq_start = 0
        for b in range(batch):
            length = int(actual_seq_lengths[b].item())
            seq_end = seq_start + length
            n_chunks = (length + C - 1) // C
            bb[0] = b
            for chunk_i in range(n_chunks):
                ci[0] = chunk_i
                # curState for chunk_i==0 is initial_state[b]; else the carried S
                cur_state = ws_u_ds["S"]
                if chunk_i == 0:
                    # copy initial_state[b] -> S[unit] for this batch's units
                    for h in range(nv):
                        u = b * nv + h
                        cur_state[u].copy_(initial_state[b, h])
                # stage1
                k1(
                    query, key, value, beta, g, actual_seq_lengths, ci, bb,
                    ws_u_dk["q"], ws_u_dk["k"], ws_u_dk["kb"], ws_u_dk["kbe"],
                    ws_u_dk["qprime"], ws_u_dk["kg"], ws_u_dv["vb"],
                    ws_u_cc["kbt"], ws_u_cc["qkt"], ws_u_cc["x"],
                    ws_u_dk["v_inner"], ws_u_dk["k_cumdecay"],
                    ws_u_g["g_exp"], ws_u_g["g_exp_neg"],
                )
                # stage2
                k2(
                    ws_u_dk["k_cumdecay"], ws_u_dk["qprime"], ws_u_dk["v_inner"],
                    ws_u_dk["kg"], cur_state, out, final_state,
                    ws_u_dv["v_prime"], ws_u_dv["attn_inter"], ws_u_dv["v_new"],
                    ws_u_dstate["dstate"], ws_u_g["g_exp"], actual_seq_lengths,
                    ci, bb,
                )
                # stage3
                k3(
                    out, ws_u_cc["qkt"], ws_u_dv["v_new"],
                    ws_u_g["g_exp"], ws_u_g["g_exp_neg"],
                    ws_u_cc["attn_c"], ws_u_dv["out_cont"], actual_seq_lengths,
                    ci, bb,
                )
            # synchronize the platform after each batch's stage pipeline
            torch.npu.synchronize()
            seq_start = seq_end
        return out, final_state

    return run
