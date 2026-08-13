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

# Tile-level TileLang design for flash attention（MHA，kv_heads == heads）。
#
# 关键尺寸：BLOCK_M=64 / BLOCK_N=64 / cube_k=64 / prelaunch=2 / ring_slots=3。
#   L1 的 dim 维按 cube_k 补齐（dim_pad），尾列由编译器零填充，cube_k 无需整除 dim。
#
# 任务内 KV ring 流水（每 task = 一个 (bz, by, bx) 的 64 行 Q 块，整 KV 扫描在同核）：
#   for t in kv_loops + prelaunch:
#     if t < kv_loops:      C1(t)/V1(t)     # 生产：S=Q@K^T -> ws_s；online softmax -> ws_p + meta
#     if t >= prelaunch:    C2(t-2)/V2(t-2) # 消费：O=P@V -> ws_o；alpha rescale merge
#   prelaunch=2 / ring_slots=3，slot = t % ring_slots 轮转。
#
# running 状态：max/sum 经 workspace_meta（0=alpha, 1=sumexp）在 V1->V2 间传递；
#   running O 驻留 UB（acc_o），逐 kv 块 alpha 缩放累加，末块归一化写出。
#
# Tail：K/V/Q 的 GM 越界读、Output 越界写由编译器自动处理（安全读 / 跳过越界写）；
#   S 尾列（kv 尾块）在 V1 内显式 mask 成 -inf（越界列读 0 会被 exp(0)=1
#   当成有效分数污染分母）。

from dataclasses import dataclass

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout
from tvm.tir import BufferRegion
from tvm.ir import Range


@dataclass(frozen=True)
class FlashAttentionConfig:
    """Configuration encapsulating FlashAttention kernel shapes and data type."""
    batch: int
    heads: int
    q_seq_len: int
    kv_seq_len: int
    dim: int
    dtype: str = "float16"


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# AIC<->AIV workspace 交接信号
SIG_S_READY = 0   # C1 写完 workspace_s -> V1 可读
SIG_P_READY = 1   # V1 写完 workspace_p -> C2 可读
SIG_O_READY = 2   # C2 写完 workspace_o -> V2 可读

MAX_CORES = 20   # used_core_num = min(block_num, MAX_CORES)


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6, 7], pass_configs=pass_configs)
def flash_attention_fwd(cfg: FlashAttentionConfig):
    batch = cfg.batch
    heads = cfg.heads
    q_seq_len = cfg.q_seq_len
    kv_seq_len = cfg.kv_seq_len
    dim = cfg.dim
    dtype = cfg.dtype

    block_m, block_n = 64, 64
    cube_k = 64
    prelaunch = 2
    ring_slots = prelaunch + 1

    accum_dtype = "float"

    sm_scale = (1.0 / dim) ** 0.5

    q_blocks = T.ceildiv(q_seq_len, block_m)
    block_num = q_blocks * heads * batch
    used_core_num = min(MAX_CORES, block_num)
    tasks_per_core = T.ceildiv(block_num, used_core_num)
    kv_loops = T.ceildiv(kv_seq_len, block_n)
    qk_k_loops = T.ceildiv(dim, cube_k)
    # L1 的 dim 维按 cube_k 补齐：GM->L1 拷贝越界部分编译器自动零填充，
    # 对 S 的 K 维求和 / O 的无效列都是 0 贡献；L0 始终读满 tile。
    dim_pad = qk_k_loops * cube_k

    # Host 零 pad：Q/K/V/Out 均用原始序列长度，尾块由 kernel 内处理。
    q_shape = [batch, heads, q_seq_len, dim]
    kv_shape = [batch, heads, kv_seq_len, dim]
    out_shape = [batch, heads, q_seq_len, dim]

    if block_num <= 0:
        raise ValueError(f"block_num ({block_num}) must be > 0")

    tail_valid = kv_seq_len % block_n

    @T.prim_func
    def main(
        q: T.Tensor(q_shape, dtype),
        k: T.Tensor(kv_shape, dtype),
        v: T.Tensor(kv_shape, dtype),
        output: T.Tensor(out_shape, dtype),
        workspace_s: T.Tensor([block_num, ring_slots, block_m, block_n], accum_dtype),
        workspace_p: T.Tensor([block_num, ring_slots, block_m, block_n], dtype),
        workspace_o: T.Tensor([block_num, ring_slots, block_m, dim], accum_dtype),
        workspace_meta: T.Tensor([block_num, ring_slots, block_m, 2], accum_dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # ---- L1 缓冲（Q 常驻 task 级；K/V 每 kv 块缓存；dim 维按 dim_pad 补齐）----
            q_l1 = T.alloc_L1([block_m, dim_pad], dtype)
            k_l1 = T.alloc_L1([block_n, dim_pad], dtype)
            v_l1 = T.alloc_L1([block_n, dim_pad], dtype)
            acc_s_l1 = T.alloc_L1([block_m, block_n], dtype)
            T.annotate_layout({
                q_l1: make_zn_layout(q_l1),
                k_l1: make_zn_layout(k_l1),
                v_l1: make_zn_layout(v_l1),
                acc_s_l1: make_zn_layout(acc_s_l1),
            })

            lhs_l0 = T.alloc_L0A([block_m, cube_k], dtype)
            rhs_l0 = T.alloc_L0B([cube_k, block_n], dtype)
            # C2 专用 L0：A=P [block_m, block_n]，B=V 切片 [block_n, cube_k]
            p_l0 = T.alloc_L0A([block_m, block_n], dtype)
            v_l0 = T.alloc_L0B([block_n, cube_k], dtype)
            acc_s_l0c = T.alloc_L0C([block_m, block_n], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_m, cube_k], accum_dtype)

            # ---- V 侧 UB：每个 AIV 子核处理 block_m//2 行 ----
            acc_o = T.alloc_ub([block_m // 2, dim], accum_dtype)      # running O（驻留）
            acc_o_ub = T.alloc_ub([block_m // 2, dim], accum_dtype)   # 当前块 O 贡献
            acc_o_half = T.alloc_ub([block_m // 2, dim], dtype)
            sumexp = T.alloc_ub([block_m // 2], accum_dtype)          # running sum（驻留）
            m_i = T.alloc_ub([block_m // 2], accum_dtype)             # running max（驻留）
            m_i_prev = T.alloc_ub([block_m // 2], accum_dtype)        # prev max / alpha
            sumexp_i_ub = T.alloc_ub([block_m // 2], accum_dtype)     # 当前块 sum
            alpha_ub = T.alloc_ub([block_m // 2], accum_dtype)
            sumexp_meta_ub = T.alloc_ub([block_m // 2], accum_dtype)
            acc_s_ub = T.alloc_ub([block_m // 2, block_n], accum_dtype)
            acc_s_half = T.alloc_ub([block_m // 2, block_n], dtype)
            mask_col = T.alloc_ub([block_n], accum_dtype)

            for local_idx in T.serial(tasks_per_core):
                task_idx = cid * tasks_per_core + local_idx
                # TIR 无 continue：用 if 守卫整个 task body
                if task_idx < block_num:
                    bx = task_idx % q_blocks
                    by = task_idx // q_blocks % heads
                    bz = task_idx // q_blocks // heads % batch

                    # ---- C0：Q 常驻 L1（尾块越界读由编译器零填充）----
                    with T.Scope("C"):
                        T.copy(q[bz, by, bx * block_m: (bx + 1) * block_m, 0:dim_pad], q_l1)

                    # ---- V0：running 状态初始化 ----
                    with T.Scope("V"):
                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(sumexp, 0.0)
                        T.tile.fill(m_i, -2 ** 30)

                    # ---- 任务内 KV ring 流水 ----
                    for t in T.serial(kv_loops + prelaunch):
                        # ---------- 生产者：C1(t) / V1(t) ----------
                        if t < kv_loops:
                            slot_prod = t % ring_slots

                            with T.Scope("C"):
                                # C1: S = Q @ K^T -> workspace_s[task_idx, slot_prod]
                                T.copy(k[bz, by, t * block_n: (t + 1) * block_n, 0:dim_pad],
                                       k_l1)
                                for kk in T.serial(qk_k_loops):
                                    T.copy(q_l1[0, kk * cube_k], lhs_l0)
                                    T.copy(k_l1[0, kk * cube_k], rhs_l0, transpose=True)
                                    T.mma(lhs_l0, rhs_l0, acc_s_l0c, init=(kk == 0))
                                T.copy(acc_s_l0c, workspace_s[task_idx, slot_prod, :, :])
                                T.set_cross_flag("FIX", SIG_S_READY)   # C1(t) 完成 -> V1(t)

                            with T.Scope("V"):
                                # V1: scale/mask/online softmax -> workspace_p + meta
                                T.wait_cross_flag(SIG_S_READY)
                                h_start = vid * (block_m // 2)
                                T.copy(m_i, m_i_prev)   # running max（上一 kv 块）
                                T.copy(
                                    workspace_s[task_idx, slot_prod,
                                                h_start: h_start + block_m // 2, :],
                                    acc_s_ub,
                                )
                                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                                # kv 尾块列 mask：越界列加 -inf
                                if tail_valid != 0 and t == kv_loops - 1:
                                    T.tile.fill(mask_col, T.float32(-2 ** 30))
                                    T.tile.fill(
                                        BufferRegion(mask_col, [Range(0, tail_valid)]),
                                        T.float32(0.0),
                                    )
                                    for h_i in range(block_m // 2):
                                        T.tile.add(acc_s_ub[h_i, :], acc_s_ub[h_i, :],
                                                   mask_col)
                                # online softmax：new_max = max(chunk_max, running_max)
                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                T.tile.max(m_i, m_i, m_i_prev)
                                T.tile.sub(m_i_prev, m_i_prev, m_i)
                                T.tile.exp(m_i_prev, m_i_prev)          # alpha
                                for h_i in range(block_m // 2):
                                    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :],
                                               m_i[h_i])
                                T.tile.exp(acc_s_ub, acc_s_ub)
                                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                                # P 写 workspace_p（dtype），meta: 0=alpha, 1=sumexp_i
                                T.copy(acc_s_ub, acc_s_half)
                                T.copy(
                                    acc_s_half,
                                    workspace_p[task_idx, slot_prod,
                                                h_start: h_start + block_m // 2, :],
                                )
                                for h_i in range(block_m // 2):
                                    row = h_start + h_i
                                    workspace_meta[task_idx, slot_prod, row, 0] = m_i_prev[h_i]
                                    workspace_meta[task_idx, slot_prod, row, 1] = sumexp_i_ub[h_i]
                                T.set_cross_flag("MTE3", SIG_P_READY)  # V1(t) 完成 -> C2(t)

                        # ---------- 消费者：C2(t-2) / V2(t-2) ----------
                        if t >= prelaunch:
                            now_k = t - prelaunch
                            slot_cons = now_k % ring_slots

                            with T.Scope("C"):
                                # C2: O = P @ V -> workspace_o[task_idx, slot_cons]
                                T.wait_cross_flag(SIG_P_READY)
                                T.copy(workspace_p[task_idx, slot_cons, :, :], acc_s_l1)
                                T.copy(v[bz, by, now_k * block_n: (now_k + 1) * block_n,
                                         0:dim_pad], v_l1)
                                for dd in T.serial(qk_k_loops):
                                    T.copy(acc_s_l1, p_l0)
                                    T.copy(v_l1[0, dd * cube_k], v_l0)
                                    T.mma(p_l0, v_l0, acc_o_l0c, init=True)
                                    T.copy(
                                        acc_o_l0c,
                                        workspace_o[task_idx, slot_cons, :,
                                                    dd * cube_k: (dd + 1) * cube_k],
                                    )
                                T.set_cross_flag("FIX", SIG_O_READY)  # C2(t-2) 完成 -> V2(t-2)

                            with T.Scope("V"):
                                # V2: 读 meta（alpha/sumexp_i），merge 当前 O 贡献到 running 状态
                                T.wait_cross_flag(SIG_O_READY)
                                h_start = vid * (block_m // 2)
                                for h_i in range(block_m // 2):
                                    row = h_start + h_i
                                    alpha_ub[h_i] = workspace_meta[task_idx, slot_cons, row, 0]
                                    sumexp_meta_ub[h_i] = workspace_meta[task_idx, slot_cons, row, 1]
                                T.copy(
                                    workspace_o[task_idx, slot_cons,
                                                h_start: h_start + block_m // 2, :],
                                    acc_o_ub,
                                )
                                for h_i in range(block_m // 2):
                                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], alpha_ub[h_i])
                                T.tile.add(acc_o, acc_o, acc_o_ub)
                                T.tile.mul(sumexp, sumexp, alpha_ub)
                                T.tile.add(sumexp, sumexp, sumexp_meta_ub)

                    # ---- 收尾：归一化并写出（越界写由编译器自动跳过）----
                    with T.Scope("V"):
                        for h_i in range(block_m // 2):
                            T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])
                        T.copy(acc_o, acc_o_half)
                        row_start = bx * block_m + vid * (block_m // 2)
                        T.copy(
                            acc_o_half,
                            output[bz, by, row_start: row_start + block_m // 2, :],
                        )

    return main
