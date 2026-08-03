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

"""Block-level design of Chunked Gated Delta Rule — 入口 / 任务划分 + 3-kernel 编排。

对应 AscendC `CGDR::Process`，并与 tile-level（`design/tile_level/`）的 3 个独立 jit kernel
（`stage1_ker`/`stage2_ker`/`stage3_ker`）一一对应。**编排代码与 tile-level 同名实现一致**
（此处是 block-level 的落地编排；stage kernel 内部细算在 tile-level 落地）。
满足 `BlockLevelDesign-quality-checklist.md`。

## 任务划分（checklist #1）

状态 S `(Dv, Dk)` 沿 chunk 递推，因此 (batch, value-head) 之间完全独立、chunk 之间串行：

    block 数     = batch * nv          # 每个 (b, nv) 一个 block，无写冲突、无跨 block 同步
    block 输出区 = out[seqStart_b + ci*C : seqStart_b + ci*C + valid, nv, :]   # 具体切片
    key head     = value head nv 读 key head (nv*Nk)//Nv

## 布局整理（checklist #3）

TND 已按 (T, N, D) 连续排布，无需额外 pack。头分组 `kh=(nv*Nk)//Nv` 在 stage kernel 内按
运行时 head 索引取用。

## 块内遍历循环（checklist #2，本文件下方 `run()` 中的 host 循环）

    for b in range(batch):                                    # 遍历 batch
        for ci in range((Lb + C - 1) // C):                   # 遍历该 batch 的 chunk
            curState = initial_state[b] 或 上一 chunk 的 S
            stage1_ker(ci, b)   # within-chunk -> workspace（独立 jit kernel）
            stage2_ker(ci, b, curState)  # 跨 chunk 递推 -> out(attn_inter), S'
            stage3_ker(ci, b)   # chunk 内累加 -> out += attn_c@v_new

每个 stage kernel 内部再遍历其职责的 (batch×value-head) 单元（见各 stage 文件）。

## C/V 分工与同步（checklist #4, #5）

- **阶段间同步 = 3-kernel 启动顺序**：`stage1_ker → stage2_ker → stage3_ker` 同一 stream 顺序
  启动，kernel N 完成后 N+1 才读其产物 —— 对应 AscendC `SyncAll<false>()`。
  若未来合成单 kernel，阶段边界改用 `T.barrier_all()`；当前 3-kernel 是已验证编排
  （避免单 kernel 的 dk=64 UB 溢出与 C↔V 跨迭代失步）。
- **阶段内 AIC↔AIV**：各 stage kernel 内部用 `T.Scope("C")`（cube matmul）/`T.Scope("V")`
  （向量）分工，经 GM workspace 交接 + `T.set_cross_flag`/`T.wait_cross_flag`（或 pipe 级
  `T.set_flag`/`T.wait_flag`）同步 —— 对应 AscendC `SetFlag`/`WaitFlag`。

## workspace（checklist #6，GM，AIC↔AIV 交接）

| 张量 | shape | dtype | 谁写 | 谁读 |
|---|---|---|---|---|
| `S`（状态） | `(B*Nv, Dv, Dk)` | bf16 | stage2 末 | stage2 开头（初值/上一 chunk） |
| `g_exp`/`g_exp_neg` | `(B*Nv, C)` | fp32 | stage1 | stage1/2/3 |
| `kb/kbe/qprime/kg/k/k_cumdecay` | `(B*Nv, C, Dk)` | bf16 | stage1 | stage2 |
| `vb/v_prime/attn_inter/v_new/out_cont` | `(B*Nv, C, Dv)` | bf16 | stage1/2 | stage2/3 |
| `kbt/qkt/x/attn_c` | `(B*Nv, C, C)` | bf16 | stage1/3 | stage1/2/3 |
| `dstate` | `(B*Nv, Dk, Dv)` | bf16 | stage2 | stage2 |
| `out` | `(T, Nv, Dv)` | bf16 | stage2（attn_inter）+ stage3（累加） | — |

## GEMM 约束（checklist #10）

- block_M = chunk_size = 64 ≥ 16；block_N（dk/dv ≤ 128）≥ 16，满足 NPU 分形。
- 最大 matmul `X@vb`：`64 × 128 × sizeof(float) = 32KB ≤ L0C(128KB)` ✓。

## 技术约束（checklist #9）

- 尾 chunk：`valid = min(C, Lb - ci*C)` 在 stage kernel 内用 `T.if_then_else` 由运行时
  `Lb`/`ci` 算出（见 stage1/2/3）。
- 动态 batch 长度：host 读 `actual_seq_lengths` 得 Lb 驱动 chunk 循环。

## 细算落地（checklist #7）

stage kernel 内部的具体计算标 `TODO(tile-level)`，在 `design/tile_level/stageN.py` 落地。
"""

import torch

from design.block_level.stage1 import stage1_ker
from design.block_level.stage2 import stage2_ker
from design.block_level.stage3 import stage3_ker


def chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv,
                           chunk_size=64, scale=1.0):
    """编排器：build 3 个 stage kernel，host 按 batch×chunk 顺序调用。

    返回 callable `(query, key, value, beta, g, initial_state, actual_seq_lengths) -> (out, final_state)`。
    与 tile-level 入口同名同结构（细算 TODO(tile-level) 在 tile-level 落地）。
    """
    def run(query, key, value, beta, g, initial_state, actual_seq_lengths):
        device = query.device
        total_units = batch * nv
        C = int(chunk_size)

        # ---- per-unit workspace（跨 chunk 复用），checklist #6 ----
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

        # ---- build 3 stage kernels（每 chunk 复用）----
        k1 = stage1_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=C, scale=scale)
        k2 = stage2_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=C)
        k3 = stage3_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=C, scale=scale)

        # ---- host 块内遍历（checklist #2）：batch → chunk → stage1/2/3 ----
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
                cur_state = ws_u_ds["S"]
                if chunk_i == 0:
                    for h in range(nv):
                        u = b * nv + h
                        cur_state[u].copy_(initial_state[b, h])   # 初值
                # stage1: within-chunk 预计算 -> workspace
                k1(query, key, value, beta, g, actual_seq_lengths, ci, bb,
                   ws_u_dk["q"], ws_u_dk["k"], ws_u_dk["kb"], ws_u_dk["kbe"],
                   ws_u_dk["qprime"], ws_u_dk["kg"], ws_u_dv["vb"],
                   ws_u_cc["kbt"], ws_u_cc["qkt"], ws_u_cc["x"],
                   ws_u_dk["v_inner"], ws_u_dk["k_cumdecay"],
                   ws_u_g["g_exp"], ws_u_g["g_exp_neg"])
                # stage2: 跨 chunk 递推（状态依赖）-> out(attn_inter), S'
                k2(ws_u_dk["k_cumdecay"], ws_u_dk["qprime"], ws_u_dk["v_inner"],
                   ws_u_dk["kg"], cur_state, out, final_state,
                   ws_u_dv["v_prime"], ws_u_dv["attn_inter"], ws_u_dv["v_new"],
                   ws_u_dstate["dstate"], ws_u_g["g_exp"], actual_seq_lengths,
                   ci, bb)
                # stage3: chunk 内累加 -> out += attn_c@v_new
                k3(out, ws_u_cc["qkt"], ws_u_dv["v_new"],
                   ws_u_g["g_exp"], ws_u_g["g_exp_neg"],
                   ws_u_cc["attn_c"], ws_u_dv["out_cont"], actual_seq_lengths,
                   ci, bb)
            torch.npu.synchronize()          # batch 边界同步
            seq_start = seq_end
        return out, final_state

    return run
