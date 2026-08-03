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

"""Tile-level TileLang design — SINGLE kernel (expert mode + cube) for the
Chunked Gated Delta Rule, mirrored on `design/block_level/` and the AscendC
`CGDR::Process` (one kernel; batch -> unit -> chunk -> RunStage1/2/3 + SyncAll).

    chunk_gated_delta_rule.py   entry / orchestration   <- @tilelang.jit, ONE prim_func
    stage1.py                   within-chunk precompute <- run_stage1 (inlined function)
    stage2.py                   cross-chunk recurrence  <- run_stage2 (inlined function)
    stage3.py                   within-chunk accumulate <- run_stage3 (inlined function)

There is exactly ONE `@tilelang.jit` kernel (here) and ONE `T.Kernel`.  stage1/2/3
are plain Python functions inlined into the single prim_func.  Inside the unit
loop, each valid chunk runs:

    for ci in T.serial(max_chunks):
        if ci*C < Lb:
            run_stage1(...)     # -> GM workspace
            T.barrier_all()     # SyncAll (学 AscendC SyncAll<false>)
            run_stage2(...)     # cross-chunk state recurrence
            T.barrier_all()
            run_stage3(...)     # within-chunk accumulate
            T.barrier_all()

Why ONE kernel (per user requirement, matches AscendC): AscendC runs stage1/2/3
inside a single kernel body with `SyncAll<false>()` between; the archived
3-kernel was only a workaround for two backend issues that are both now solved:

1. Cross-chunk C<->V desync: solved by the `T.barrier_all()` (full pipe barrier)
   between stages — every stage-boundary SyncAll drains all pipelines, so the
   per-chunk cross-flag handoff (set/wait_cross_flag IDs 0..7) re-arms cleanly
   and never leaks a transaction into the next chunk.

2. UB overflow at dk=dv=64: solved by AscendC-style UB reuse.  stage1/2/3
   declare their `T.alloc_ub` temporaries *inside* their own function segment,
   so each stage's on-chip footprint is confined to its own chunk-iteration
   segment (stages don't overlap in time — separated by barrier_all).  The
   TileLang memory planner (`tl.ascend_memory_planning`) then reuses the same
   physical UB across stages, giving peak = per-stage max, not the union
   (the old hoisted-allocation single kernel needed ~384KB and overflowed).

All matmuls are cube `T.gemm_v0`; vector ops in UB; every bf16 boundary goes
through two-stage `T.copy`; expert mode (no T.parallel / T.pipeline); beta/g
loaded per-scalar; sliced writes cover the full last dim.  State (Dv,Dk);
final_state/initial_state (B,Nv,Dv,Dk); value head h reads key head (h*Nk)//Nv.
"""

import tilelang
import tilelang.language as T

from design.chunk_gated_delta_rule_common import (
    CHUNK_SIZE,
    NUM_PHYSICAL_CORES,
    pass_configs,
)
from design.tile_level.stage1 import run_stage1, key_head as kh_fn
from design.tile_level.stage2 import run_stage2
from design.tile_level.stage3 import run_stage3


@tilelang.jit(
    out_idx=[],
    workspace_idx=[],
    pass_configs=pass_configs,
)
def _gdr_impl(batch, seq_len, nk, nv, dk, dv,
              chunk_size=CHUNK_SIZE, scale=1.0):
    """Build the SINGLE-kernel runner.  Mirrors `CGDR::Process`."""
    C = chunk_size
    total_units = batch * nv
    used_core_num = min(NUM_PHYSICAL_CORES, max(1, total_units))
    tasks_per_core = T.ceildiv(total_units, used_core_num)
    max_chunks = (seq_len + C - 1) // C

    @T.prim_func
    def main(
        query: T.Tensor((seq_len, nk, dk), "bfloat16"),            # 0
        key: T.Tensor((seq_len, nk, dk), "bfloat16"),              # 1
        value: T.Tensor((seq_len, nv, dv), "bfloat16"),            # 2
        beta: T.Tensor((seq_len, nv), "bfloat16"),                 # 3
        g: T.Tensor((seq_len, nv), "float32"),                     # 4
        initial_state: T.Tensor((batch, nv, dv, dk), "bfloat16"),  # 5
        actual_seq_lengths: T.Tensor((batch,), "int32"),           # 6
        out: T.Tensor((seq_len, nv, dv), "bfloat16"),              # 7
        final_state: T.Tensor((batch, nv, dv, dk), "bfloat16"),    # 8
        # ---- per-unit GM workspace (one chunk frag per unit, reused) ----
        ws_q: T.Tensor((total_units, C, dk), "bfloat16"),          # 9
        ws_k: T.Tensor((total_units, C, dk), "bfloat16"),          # 10
        ws_kb: T.Tensor((total_units, C, dk), "bfloat16"),         # 11
        ws_kbe: T.Tensor((total_units, C, dk), "bfloat16"),        # 12
        ws_qprime: T.Tensor((total_units, C, dk), "bfloat16"),     # 13
        ws_kg: T.Tensor((total_units, C, dk), "bfloat16"),         # 14
        ws_vb: T.Tensor((total_units, C, dv), "bfloat16"),         # 15
        ws_kbt: T.Tensor((total_units, C, C), "bfloat16"),         # 16
        ws_qkt: T.Tensor((total_units, C, C), "bfloat16"),         # 17
        ws_x: T.Tensor((total_units, C, C), "bfloat16"),           # 18
        ws_v_inner: T.Tensor((total_units, C, dv), "bfloat16"),    # 19
        ws_k_cumdecay: T.Tensor((total_units, C, dk), "bfloat16"),  # 20
        ws_v_prime: T.Tensor((total_units, C, dv), "bfloat16"),    # 21
        ws_attn_inter: T.Tensor((total_units, C, dv), "bfloat16"),  # 22
        ws_v_new: T.Tensor((total_units, C, dv), "bfloat16"),      # 23
        ws_out_cont: T.Tensor((total_units, C, dv), "bfloat16"),   # 24
        ws_dstate: T.Tensor((total_units, dv, dk), "bfloat16"),    # 25  (dv,dk) state layout
        ws_S: T.Tensor((total_units, dv, dk), "bfloat16"),         # 26  carried state
        ws_attn_c: T.Tensor((total_units, C, C), "bfloat16"),      # 27
        ws_g_exp: T.Tensor((total_units, C), "float32"),           # 28
        ws_g_exp_neg: T.Tensor((total_units, C), "float32"),       # 29
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # ---- per-core grid over (batch, value-head) units ----
            for local_idx in T.serial(tasks_per_core):
                unit = cid * tasks_per_core + local_idx
                if unit < total_units:
                    b = unit // nv
                    h = unit % nv
                    kh = kh_fn(h, nk, nv)

                    # seq offset of this batch (start row in TND)
                    base = T.alloc_ub((1,), "int32")
                    base[0] = 0
                    for bbz in T.serial(b):
                        base[0] += actual_seq_lengths[bbz]
                    Lb = actual_seq_lengths[b]

                    # initialise carried state from initial_state[b,h]
                    # (stage through UB: direct GM->GM copy is not supported)
                    with T.Scope("V"):
                        init_bf = T.alloc_ub((dv, dk), "bfloat16")
                        T.tile.fill(init_bf, 0.0)
                        T.copy(initial_state[b, h, :, :], init_bf)
                        T.copy(init_bf, ws_S[unit, 0, 0])

                    # ---- per-unit chunk loop ----
                    for ci in T.serial(max_chunks):
                        if ci * C < Lb:
                            s0 = base[0] + ci * C
                            valid = T.if_then_else(Lb - ci * C < C,
                                                   Lb - ci * C, C)
                            run_stage1(
                                nk, nv, dk, dv, C, scale,
                                unit, b, h, kh, s0, valid,
                                query, key, value, beta, g,
                                ws_q, ws_k, ws_kb, ws_kbe, ws_qprime, ws_kg,
                                ws_vb, ws_kbt, ws_qkt, ws_x,
                                ws_v_inner, ws_k_cumdecay,
                                ws_g_exp, ws_g_exp_neg,
                            )
                            T.barrier_all()

                            run_stage2(
                                nk, nv, dk, dv, C,
                                unit, b, h, s0, valid,
                                ws_k_cumdecay, ws_qprime, ws_v_inner, ws_kg,
                                ws_S, out, final_state,
                                ws_v_prime, ws_attn_inter, ws_v_new,
                                ws_dstate, ws_g_exp,
                            )
                            T.barrier_all()

                            run_stage3(
                                nk, nv, dk, dv, C, scale,
                                unit, b, h, s0, valid,
                                out, ws_qkt, ws_v_new,
                                ws_g_exp, ws_g_exp_neg,
                                ws_attn_c, ws_out_cont,
                            )
                            T.barrier_all()

    return main


def chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv,
                           chunk_size=CHUNK_SIZE, scale=1.0):
    """Host wrapper: build the SINGLE kernel once and launch it.

    Mirrors the 3-kernel entry's signature `(batch, seq_len, nk, nv, dk, dv,
    chunk_size, scale) -> callable`.  The chunk loop lives *inside* the kernel
    (per-unit), so this host only allocates the per-unit GM workspace once and
    issues a single launch covering all batches/chunks.
    """
    import torch

    C = int(chunk_size)
    kernel = _gdr_impl(batch, seq_len, nk, nv, dk, dv,
                       chunk_size=C, scale=scale)

    def run(query, key, value, beta, g, initial_state, actual_seq_lengths):
        device = query.device
        total_units = batch * nv

        # per-unit GM workspace (one chunk frag per unit, reused across chunks)
        def w3(specs, dt2):
            return {n: torch.zeros(sh, dtype=dt2, device=device)
                    for (n, sh) in specs}

        ws_u_dk = w3([("q", (total_units, C, dk)), ("k", (total_units, C, dk)),
                      ("kb", (total_units, C, dk)), ("kbe", (total_units, C, dk)),
                      ("qprime", (total_units, C, dk)), ("kg", (total_units, C, dk)),
                      ("v_inner", (total_units, C, dv)),
                      ("k_cumdecay", (total_units, C, dk))], torch.bfloat16)
        ws_u_dv = w3([("vb", (total_units, C, dv)),
                      ("v_prime", (total_units, C, dv)),
                      ("attn_inter", (total_units, C, dv)),
                      ("v_new", (total_units, C, dv)),
                      ("out_cont", (total_units, C, dv))], torch.bfloat16)
        ws_u_cc = w3([("kbt", (total_units, C, C)), ("qkt", (total_units, C, C)),
                      ("x", (total_units, C, C)),
                      ("attn_c", (total_units, C, C))], torch.bfloat16)
        ws_u_g = w3([("g_exp", (total_units, C)),
                     ("g_exp_neg", (total_units, C))], torch.float32)
        ws_u_dstate = w3([("dstate", (total_units, dv, dk))], torch.bfloat16)
        ws_u_ds = w3([("S", (total_units, dv, dk))], torch.bfloat16)  # carried state

        out = torch.zeros((seq_len, nv, dv), dtype=query.dtype, device=device)
        final_state = torch.empty((batch, nv, dv, dk),
                                  dtype=initial_state.dtype, device=device)

        # single launch: kernel internally loops over all batches/chunks/units
        kernel(
            query, key, value, beta, g, initial_state, actual_seq_lengths,
            out, final_state,
            ws_u_dk["q"], ws_u_dk["k"], ws_u_dk["kb"], ws_u_dk["kbe"],
            ws_u_dk["qprime"], ws_u_dk["kg"], ws_u_dv["vb"],
            ws_u_cc["kbt"], ws_u_cc["qkt"], ws_u_cc["x"],
            ws_u_dk["v_inner"], ws_u_dk["k_cumdecay"],
            ws_u_dv["v_prime"], ws_u_dv["attn_inter"], ws_u_dv["v_new"],
            ws_u_dv["out_cont"], ws_u_dstate["dstate"], ws_u_ds["S"],
            ws_u_cc["attn_c"],
            ws_u_g["g_exp"], ws_u_g["g_exp_neg"],
        )
        return out, final_state

    return run
