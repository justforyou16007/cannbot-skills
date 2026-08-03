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

对应 AscendC `CGDR::Process`，并与 **tile-level 的实际实现**（`design/tile_level/`）一一对应：
tile-level 采用 **3 个独立 jit kernel**（`stage1_ker`/`stage2_ker`/`stage3_ker`），由 host
按 batch×chunk 顺序依次启动；本文件描述同一套编排（任务划分 + 阶段边界 + 同步），
不再使用单 kernel + `T.serial(max_chunks)` 的写法。

## 任务划分（block = (batch, value-head) 单元）

输出 `out[T, Nv, Dv]`。状态 S `(Dv, Dk)` 沿 chunk 递推，因此 (batch, value-head)
之间完全独立、chunk 之间串行。取：

    block 数     = batch * nv          # 每个 (b, nv) 一个 block，无写冲突、无跨 block 同步
    block 输出区 = out[该 (b,nv) 的全部 chunk 行, nv, :]
    block 遍历   = 该 (b,nv) 的 chunk 序列（S 在 GM `S`/`final_state` 中携带）
    key head     = value head nv 读 key head nv*Nk//Nv

## 编排（host 3-kernel，对应 tile-level 实现）

    for b in 0..B:                                  # host 层（model_new_tilelang.py）
        for ci in 0..n_chunks(b):                    # 每个 chunk
            curState = initial_state[b] 或 上一 chunk 更新后的 S
            stage1_ker(ci, b)          # within-chunk 预计算 -> workspace（独立 jit kernel）
            stage2_ker(ci, b, curState) # 跨 chunk 递推：v_prime/attn_inter/out + S'（独立 kernel）
            stage3_ker(ci, b)          # chunk 内累加：out += attn_c@v_new（独立 kernel）

## 同步（对应 AscendC `SyncAll<false>()` 与 AIC↔AIV 事件）

1. **阶段间全核同步 —— 3-kernel 启动顺序**：`stage1_ker → stage2_ker → stage3_ker` 在同一
   stream 上顺序启动，kernel N 完成后 N+1 才读它的产物——语义对应 `SyncAll<false>()`
   （阶段边界全核就绪）。若未来合成单 kernel（省 launch 开销），阶段边界改用
   `T.barrier_all()`；当前 3-kernel 是已实现/验证的编排（避免单 kernel 的 dk=64 UB 溢出
   与 C↔V 跨迭代失步）。
2. **阶段内 AIC↔AIV 交接 —— workspace + cross/pipe flag**：每个 stage kernel 内部，C（cube
   matmul）与 V（向量）经 GM workspace 交接，用 `T.set_cross_flag`/`T.wait_cross_flag`
   （AIC/AIV 间）与 `T.set_flag`/`T.wait_flag`（pipe 级）同步；对应 AscendC 的
   `SetFlag`/`WaitFlag`/`CrossCoreSetFlag`。这一层在 stage1/2/3 各自 kernel 内部实现。

> 为什么 3-kernel 而不是单 kernel：单 kernel + `T.barrier_all()` 阶段边界确实能修好多 chunk
> 失步，但 3 个阶段的全部 UB buffer 塞进一个 kernel 会在 dk=dv=64 时溢出（~384KB）；
> 拆 3 个独立 kernel 后每个只持自己阶段的 buffer，dk=64 也放得下，且 kernel 间天然有序，
> cross-flag 不会跨 chunk 泄漏。
"""

import tilelang

from design.chunk_gated_delta_rule_common import CHUNK_SIZE, pass_configs
from design.block_level.stage1 import stage1_ker
from design.block_level.stage2 import stage2_ker
from design.block_level.stage3 import stage3_ker


def chunk_gated_delta_rule(batch, seq_len, nk, nv, dk, dv,
                           chunk_size=CHUNK_SIZE, scale=1.0):
    """编排器（与 tile-level 同名同结构）：build 3 个 stage kernel，按 batch×chunk 顺序调用。

    返回 callable `(query, key, value, beta, g, initial_state, actual_seq_lengths) -> (out, final_state)`。
    具体 buffer/workspace 分配与 launch 见 tile-level 实现；此处只描述阶段边界与数据流。
    """
    # k1 = stage1_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=chunk_size, scale=scale)
    # k2 = stage2_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=chunk_size)
    # k3 = stage3_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=chunk_size, scale=scale)
    #
    # 每个 chunk：k1(ci,b) -> k2(ci,b,curState) -> k3(ci,b)
    # 状态经 GM S / final_state 携带；细节（workspace 分配、S 搬运）见 tile-level。
    raise NotImplementedError("编排细节见 design/tile_level/chunk_gated_delta_rule.py")
