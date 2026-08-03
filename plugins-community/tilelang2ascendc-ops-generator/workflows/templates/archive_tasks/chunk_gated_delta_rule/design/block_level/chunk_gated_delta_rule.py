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
（`stage1_ker`/`stage2_ker`/`stage3_ker`）一一对应。细粒度计算标 `TODO(tile-level)`，
在 tile-level 落地。本文件 + stage1/2/3 满足 `BlockLevelDesign-quality-checklist.md`。

## 任务划分（checklist #1）

状态 S `(Dv, Dk)` 沿 chunk 递推，因此 (batch, value-head) 之间完全独立、chunk 之间串行：

    block 数     = batch * nv          # 每个 (b, nv) 一个 block，无写冲突、无跨 block 同步
    block 输出区 = out[seqStart_b + ci*C : seqStart_b + ci*C + valid, nv, :]   # 具体切片
    key head     = value head nv 读 key head (nv*Nk)//Nv

## 布局整理（checklist #3）

TND 已按 (T, N, D) 连续排布；无需额外 pack。头分组 `kh=(nv*Nk)//Nv` 在 stage kernel 内按
运行时 head 索引取用（见 stage1/2/3）。

## 块内遍历循环（checklist #2，host 层代码）

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
| `S`（状态） | `(B, Nv, Dv, Dk)` | fp32 | stage2 末 | stage2 开头（初值/上一 chunk） |
| `ws_gcumexp` | `(C,)` | fp32 | stage1 | stage1/2/3 |
| `ws_kcumdecay` | `(C, Dk)` | bf16 | stage1 | stage2 |
| `ws_vinner` | `(C, Dv)` | bf16 | stage1 | stage2/3 |
| `ws_qprime` | `(C, Dk)` | bf16 | stage1 | stage2 |
| `ws_qkt` | `(C, C)` | bf16 | stage1 | stage3 |
| `ws_kg` | `(C, Dk)` | bf16 | stage1 | stage2 |
| `out` | `(T, Nv, Dv)` | bf16 | stage2（attn_inter）+ stage3（累加） | — |

## GEMM 约束（checklist #10）

- block_M = chunk_size = 64 ≥ 16，block_N（=dk/dv ≤ 128）≥ 16，满足 NPU 分形。
- 最大 matmul `X@vb`：`64 × 128 × sizeof(float) = 32KB ≤ L0C(128KB)` ✓。

## 技术约束（checklist #9）

- 尾 chunk：`valid = min(C, Lb - ci*C)` 在 stage kernel 内用 `T.if_then_else` 由运行时
  `Lb`/`ci` 算出（见 stage1/2/3）。
- 动态 batch 长度：host 读 `actual_seq_lengths` 得 Lb 驱动 chunk 循环。

## 细算落地（checklist #7）

每个 stage 的具体计算标 `TODO(tile-level)`，在 `design/tile_level/stageN.py` 落地。
"""

import tilelang

from design.chunk_gated_delta_rule_common import CHUNK_SIZE, pass_configs
from design.block_level.stage1 import stage1_ker
from design.block_level.stage2 import stage2_ker
from design.block_level.stage3 import stage3_ker


def chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv,
                           chunk_size=CHUNK_SIZE, scale=1.0):
    """编排器：build 3 个 stage kernel，host 按 batch×chunk 顺序调用。

    返回 callable `(query, key, value, beta, g, initial_state, actual_seq_lengths) -> (out, final_state)`。
    """
    # TODO(tile-level): 与 tile_level/chunk_gated_delta_rule.py 一致——build k1/k2/k3，
    #                   分配 S/out/workspace 持久 buffer，按下方循环调用。
    # k1 = stage1_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=chunk_size, scale=scale)
    # k2 = stage2_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=chunk_size)
    # k3 = stage3_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=chunk_size, scale=scale)
    #
    # for b in range(batch):                     # 块内遍历（checklist #2）
    #     for ci in range(n_chunks(b)):
    #         k1(ci, b) -> k2(ci, b, curState) -> k3(ci, b)
    return None  # 占位；实际编排见 tile-level 同名实现
