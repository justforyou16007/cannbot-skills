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

"""Block-level TileLang design for GQA flash attention（任务划分 + C/V 流水 + 同步）。

GQA（Grouped Query Attention）：$O = \mathrm{softmax}(Q K^{\top}/\sqrt{D})\,V$，
Q 为 `[B, Hq, Sq, D]`，K/V 为 `[B, Hkv, Skv, D]`，每个 KV 头被 $Hq/Hkv$ 个 query 头共享。

## 任务划分（block = (batch, kv_head, q_block)）

按 BlockLevelDesign 的 GQA 例子，先做布局整理（pack）再切 block：
    Q:  [B, Hq, Sq, D]  ->  [B*Hkv, group*Sq, D]   （group = Hq/Hkv，packed q_seq）
    K/V:[B, Hkv, Skv, D] ->  [B*Hkv, Skv, D]
之后每个 block 固定一个 `(bz, by, bx)` 输出区：
    block 数      = q_blocks * Hkv * B      （q_blocks = Sq/block_M，block_M=64）
    block 输出区  = Output[bz, by, bx*block_M:(bx+1)*block_M, :]
    block 遍历    = 该 (batch, kv_head) 的全部 KV 块（kv_loops 次，ring 流水）
packed q_seq 不再整块整除时用 ceildiv 出 q_blocks，写回有 `row_start < q_seq` 守卫。

## 块内 C/V 流水（对应 FlashAttention 固定分解，prelaunch=2 / ring_slots=3）

    C0  预载 Q 块到 L1，贯穿整个 C1/C2 流水
    C1  S = Q @ K_t^T   （cube matmul，L0A/L0B/L0C）
    V1  online softmax（scale / 尾块 mask / reduce_max / exp / reduce_sum）+ merge 元数据
    C2  O_tmp = P_t @ V_t（cube matmul）
    V2  用 V1 的 alpha/sumexp rescale 并 merge 到运行累加器

时间线（prelaunch=2）：
    t=0: C1(0)/V1(0)；t=1: C1(1)/V1(1)；t=2: C1(2)+C2(0)/V1(2)+V2(0)；… tail: C2/V2

## 同步策略（expert mode 手动同步）

- **C/V 交接**：AIC 写 workspace（S/P/O）、AIV 读，用 `T.set_cross_flag`/`T.wait_cross_flag`
  （编号见下）；对应 AscendC 的 `SetFlag`/`WaitFlag`（AIC↔AIV 事件）。
- **ring buffer**：`slot = t % ring_slots` 轮转，prelaunch=2 保证 AIC/AIV 不必等 free 信号。

## workspace（GM，AIC↔AIV 交接）

| 张量 | shape | dtype | 语义 |
|---|---|---|---|
| `workspace_s` | [block_num, ring_slots, block_M, block_N] | fp32 | C1 的 S = QK^T |
| `workspace_p` | [block_num, ring_slots, block_M, block_N] | fp16 | V1 的 softmax 后 P |
| `workspace_o` | [block_num, ring_slots, block_M, D] | fp32 | C2 的 O_tmp = P@V |
| `workspace_meta` | [block_num, ring_slots, block_M, 2] | fp32 | V1 的 (m_i, sumexp_i) 合并元数据 |

> 细粒度计算见 tile-level（`design/tile_level/flash_attention_gqa.py`，17 例验证过）。
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# AIC<->AIV workspace 交接信号编号（与 tile-level / AscendC 事件对应）
SIG_S_READY = 0   # C1 写完 workspace_s -> V1 可读
SIG_P_READY = 1   # V1 写完 workspace_p -> C2 可读
SIG_O_READY = 2   # C2 写完 workspace_o -> V2 可读


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6, 7], pass_configs=pass_configs)
def flash_attention_gqa_fwd(
    batch,
    seq_len,
    heads,
    dim,
):
    block_M, block_N = 64, 64
    cube_K = 64
    prelaunch = 2
    ring_slots = prelaunch + 1

    dtype = "float16"
    accum_dtype = "float"

    shape = [batch, heads, seq_len, dim]
    q_blocks = seq_len // block_M
    block_num = q_blocks * heads * batch
    kv_loops = T.ceildiv(seq_len, block_N)
    qk_k_loops = T.ceildiv(dim, cube_K)
    tail_valid = seq_len % block_N

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
        workspace_s: T.Tensor([block_num, ring_slots, block_M, block_N], accum_dtype),
        workspace_p: T.Tensor([block_num, ring_slots, block_M, block_N], dtype),
        workspace_o: T.Tensor([block_num, ring_slots, block_M, dim], accum_dtype),
        workspace_meta: T.Tensor([block_num, ring_slots, block_M, 2], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            # ---- 任务分解：cid -> (bx, by, bz) ----
            bx = cid % q_blocks
            by = cid // q_blocks % heads
            bz = cid // q_blocks // heads % batch

            # ---- C0：预载 Q 块到 Cube 侧 L1，贯穿流水 ----
            with T.Scope("C"):
                # TODO(tile-level, C0): T.copy(Q[bz, by, bx*block_M:(bx+1)*block_M, :], q_l1)
                #       q_l1 = T.alloc_L1([block_M, dim])，annotate_layout(make_zn_layout)
                pass

            # ---- V0：初始化 vid 拥有的半 tile 的运行累加器 ----
            with T.Scope("V"):
                # TODO(tile-level, V0): acc_o=0, sumexp=0, m_i=-inf（T.tile.fill）
                pass

            # ---- ring 流水主循环 ----
            for t in T.serial(kv_loops + prelaunch):
                # ---------- 生产者：C1(t) / V1(t) ----------
                if t < kv_loops:
                    slot_prod = t % ring_slots
                    with T.Scope("C"):
                        # TODO(tile-level, C1): 计算 S = Q @ K_t^T 并写 workspace_s[cid, slot_prod]
                        #   - T.copy(K[bz, by, t*block_N:(t+1)*block_N, :], k_l1)
                        #   - for kk in qk_k_loops:
                        #       T.copy(q_l1[0, kk*cube_K], lhs_l0)
                        #       T.copy(k_l1[0, kk*cube_K], rhs_l0, transpose=True)
                        #       T.mma(lhs_l0, rhs_l0, acc_s_l0c, init=(kk==0))
                        #   - T.copy(acc_s_l0c, workspace_s[cid, slot_prod, :, :])
                        T.set_cross_flag("FIX", SIG_S_READY)   # C1(t) 完成 -> V1(t)

                    with T.Scope("V"):
                        # TODO(tile-level, V1): online softmax 更新并写 workspace_p/meta
                        #   - T.wait_cross_flag(SIG_S_READY)
                        #   - 取 workspace_s[cid, slot_prod] 的 vid 半 tile；乘 sm_scale=1/sqrt(D)
                        #   - 尾块(t==kv_loops-1)用 mask_col 把 [tail_valid, block_N) 置 -inf
                        #   - T.reduce_max -> m_i；T.tile.max(m_i, m_i, m_i_prev)
                        #   - exp(S - m_i)；T.reduce_sum -> sumexp_i
                        #   - 写 P 到 workspace_p[cid, slot_prod]；写 (m_i_prev, sumexp_i) 到
                        #     workspace_meta[cid, slot_prod, vid 行, 0/1]
                        T.wait_cross_flag(SIG_S_READY)
                        T.set_cross_flag("MTE3", SIG_P_READY)  # V1(t) 完成 -> C2(t)

                # ---------- 消费者：C2(now_k) / V2(now_k) ----------
                if t >= prelaunch:
                    now_k = t - prelaunch
                    slot_cons = now_k % ring_slots
                    with T.Scope("C"):
                        # TODO(tile-level, C2): O_tmp = P @ V_t 并写 workspace_o[cid, slot_cons]
                        #   - T.wait_cross_flag(SIG_P_READY)
                        #   - T.copy(workspace_p[cid, slot_cons], acc_s_l1)
                        #   - T.copy(V[bz, by, now_k*block_N:(now_k+1)*block_N, :], v_l1)
                        #   - for dd in qk_k_loops:
                        #       T.copy(acc_s_l1, lhs_l0); T.copy(v_l1[0, dd*cube_K], rhs_l0)
                        #       T.mma(lhs_l0, rhs_l0, acc_o_l0c, init=True)
                        #   - T.copy(acc_o_l0c, workspace_o[cid, slot_cons, :, :])
                        T.wait_cross_flag(SIG_P_READY)
                        T.set_cross_flag("FIX", SIG_O_READY)   # C2(now_k) 完成 -> V2(now_k)

                    with T.Scope("V"):
                        # TODO(tile-level, V2): rescale + merge O_tmp
                        #   - T.wait_cross_flag(SIG_O_READY)
                        #   - 从 workspace_meta[cid, slot_cons, vid 行, 0/1] 取 alpha/meta_sumexp
                        #   - acc_o *= alpha；acc_o += workspace_o 的 vid 半 tile
                        #   - sumexp = sumexp*alpha + meta_sumexp
                        T.wait_cross_flag(SIG_O_READY)

            # ---- V2-epilogue：归一化并写回 ----
            with T.Scope("V"):
                # TODO(tile-level, V2-epilogue):
                #   - acc_o /= sumexp（逐 vid 半 tile）
                #   - T.copy(acc_o, acc_o_half)；写 Output[bz, by, row_start:row_start+block_M//2, :]
                #     （row_start = bx*block_M + vid*block_M//2，需 < seq_len 守卫）
                pass

    return main
