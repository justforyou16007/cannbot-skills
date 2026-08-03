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

# Tile-level TileLang design for GQA flash attention.
#
# GQA is expressed as an MHA problem with heads = 1 by packing the group
# dimension into the query sequence (see model_new_tilelang.py):
#   * Q: [B, Hq, Sq, D] -> [B*Hkv, group*Sq, D]
#   * K/V: [B, Hkv, Skv, D] -> [B*Hkv, Skv, D]
# The kernel is launched with batch = B*Hkv, heads = 1, q_seq = group*Sq,
# kv_seq = Skv and runs the standard flash-attention ring pipeline
# (S = QK^T on cube, online softmax on vector, P@V on cube).

import tilelang
from tilelang import DataType, language as T
from tilelang.intrinsics import make_zn_layout
from tvm.tir import BufferRegion
from tvm.ir import Range


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6, 7], pass_configs=pass_configs)
def flash_attention_gqa_fwd(
    batch,
    heads,
    q_seq_len,
    kv_seq_len,
    dim,
    dtype="float16",
):
    block_M, block_N = 64, 64
    cube_K = 64
    num_physical_cores = 20
    prelaunch = 2
    ring_slots = prelaunch + 1

    accum_dtype = "float"

    sm_scale = (1.0 / dim) ** 0.5

    # Pad kv_seq_len to block_N alignment for tensor shapes
    kv_seq_len_padded = ((kv_seq_len + block_N - 1) // block_N) * block_N

    q_shape = [batch, heads, q_seq_len, dim]
    kv_shape = [batch, heads, kv_seq_len_padded, dim]
    out_shape = [batch, heads, q_seq_len, dim]

    q_blocks = T.ceildiv(q_seq_len, block_M)
    block_num = q_blocks * heads * batch
    used_core_num = min(num_physical_cores, block_num)
    tasks_per_core = T.ceildiv(block_num, used_core_num)
    kv_loops = T.ceildiv(kv_seq_len, block_N)
    qk_k_loops = T.ceildiv(dim, cube_K)

    assert dim % cube_K == 0
    assert block_num > 0

    tail_valid = kv_seq_len % block_N

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(kv_shape, dtype),  # type: ignore
        V: T.Tensor(kv_shape, dtype),  # type: ignore
        Output: T.Tensor(out_shape, dtype),  # type: ignore
        workspace_1: T.Tensor([block_num, ring_slots, block_M, block_N], accum_dtype),
        workspace_2: T.Tensor([block_num, ring_slots, block_M, block_N], dtype),
        workspace_3: T.Tensor([block_num, ring_slots, block_M, dim], accum_dtype),
        workspace_meta: T.Tensor([block_num, ring_slots, block_M, 2], accum_dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)
            acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)
            T.annotate_layout({
                q_l1: make_zn_layout(q_l1),
                k_l1: make_zn_layout(k_l1),
                v_l1: make_zn_layout(v_l1),
                acc_s_l1: make_zn_layout(acc_s_l1),
            })

            lhs_l0 = T.alloc_L0A([block_M, cube_K], dtype)
            rhs_l0 = T.alloc_L0B([cube_K, block_N], dtype)
            acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)

            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)
            sumexp = T.alloc_ub([block_M // 2], accum_dtype)
            m_i = T.alloc_ub([block_M // 2], accum_dtype)

            acc_s_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            m_i_prev = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_ub_ = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            sumexp_i_ub = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)
            acc_o_ub = T.alloc_ub([block_M // 2, dim], accum_dtype)
            acc_o_half = T.alloc_ub([block_M // 2, dim], dtype)
            alpha_ub = T.alloc_ub([block_M // 2], accum_dtype)
            sumexp_meta_ub = T.alloc_ub([block_M // 2], accum_dtype)
            mask_col = T.alloc_ub([block_N], accum_dtype)

            for local_idx in T.serial(tasks_per_core):
                task_idx = cid * tasks_per_core + local_idx
                if task_idx < block_num:
                    bx = task_idx % q_blocks
                    by = task_idx // q_blocks % heads
                    bz = task_idx // q_blocks // heads % batch

                    with T.Scope("C"):
                        T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], q_l1)
                    with T.Scope("V"):
                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(sumexp, 0.0)
                        T.tile.fill(m_i, -2**30)

                    for t in T.serial(kv_loops + prelaunch):
                        if t < kv_loops:
                            slot_prod = t % ring_slots
                            with T.Scope("C"):
                                T.copy(K[bz, by, t * block_N:(t + 1) * block_N, :], k_l1)
                                for kk in T.serial(qk_k_loops):
                                    T.copy(q_l1[0, kk * cube_K], lhs_l0)
                                    T.copy(k_l1[0, kk * cube_K], rhs_l0, transpose=True)
                                    T.mma(lhs_l0, rhs_l0, acc_s_l0c, init=(kk == 0))
                                T.copy(acc_s_l0c, workspace_1[task_idx, slot_prod, :, :])
                                T.set_cross_flag("FIX", 0)

                            with T.Scope("V"):
                                T.tile.fill(acc_s_ub, 0.0)
                                T.copy(m_i, m_i_prev)
                                T.wait_cross_flag(0)
                                T.copy(
                                    workspace_1[
                                        task_idx,
                                        slot_prod,
                                        vid * block_M // 2:vid * block_M // 2 + block_M // 2,
                                        :,
                                    ],
                                    acc_s_ub_,
                                )
                                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)

                                # mask 超出 kv_seq_len 的列（K/V 尾部块的无效列）
                                if tail_valid != 0:
                                    if t == kv_loops - 1:
                                        T.tile.fill(mask_col, T.float32(-2**30))
                                        T.tile.fill(
                                            BufferRegion(mask_col, [Range(0, tail_valid)]),
                                            T.float32(0.0),
                                        )
                                        for h_i in range(block_M // 2):
                                            T.tile.add(acc_s_ub[h_i, :], acc_s_ub[h_i, :], mask_col)

                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                T.tile.max(m_i, m_i, m_i_prev)
                                T.tile.sub(m_i_prev, m_i_prev, m_i)
                                T.tile.exp(m_i_prev, m_i_prev)
                                for h_i in range(block_M // 2):
                                    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                                T.tile.exp(acc_s_ub, acc_s_ub)
                                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)

                                T.copy(acc_s_ub, acc_s_half)
                                T.copy(
                                    acc_s_half,
                                    workspace_2[
                                        task_idx,
                                        slot_prod,
                                        vid * block_M // 2:vid * block_M // 2 + block_M // 2,
                                        :,
                                    ],
                                )
                                for h_i in range(block_M // 2):
                                    workspace_meta[
                                        task_idx,
                                        slot_prod,
                                        vid * block_M // 2 + h_i,
                                        0,
                                    ] = m_i_prev[h_i]
                                    workspace_meta[
                                        task_idx,
                                        slot_prod,
                                        vid * block_M // 2 + h_i,
                                        1,
                                    ] = sumexp_i_ub[h_i]
                                T.set_cross_flag("MTE3", 1)

                        if t >= prelaunch:
                            now_k = t - prelaunch
                            slot_cons = now_k % ring_slots
                            with T.Scope("C"):
                                T.wait_cross_flag(1)
                                T.copy(workspace_2[task_idx, slot_cons, :, :], acc_s_l1)
                                T.copy(V[bz, by, now_k * block_N:(now_k + 1) * block_N, :], v_l1)
                                for dd in T.serial(qk_k_loops):
                                    T.copy(acc_s_l1, lhs_l0)
                                    T.copy(v_l1[0, dd * cube_K], rhs_l0)
                                    T.mma(lhs_l0, rhs_l0, acc_o_l0c, init=True)
                                    T.copy(
                                        acc_o_l0c,
                                        workspace_3[
                                            task_idx,
                                            slot_cons,
                                            :,
                                            dd * cube_K:(dd + 1) * cube_K,
                                        ],
                                    )
                                T.set_cross_flag("FIX", 2)

                            with T.Scope("V"):
                                T.wait_cross_flag(2)
                                for h_i in range(block_M // 2):
                                    alpha_ub[h_i] = workspace_meta[
                                        task_idx,
                                        slot_cons,
                                        vid * block_M // 2 + h_i,
                                        0,
                                    ]
                                    sumexp_meta_ub[h_i] = workspace_meta[
                                        task_idx,
                                        slot_cons,
                                        vid * block_M // 2 + h_i,
                                        1,
                                    ]
                                T.copy(
                                    workspace_3[
                                        task_idx,
                                        slot_cons,
                                        vid * block_M // 2:vid * block_M // 2 + block_M // 2,
                                        :,
                                    ],
                                    acc_o_ub,
                                )
                                for h_i in range(block_M // 2):
                                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], alpha_ub[h_i])
                                T.tile.add(acc_o, acc_o, acc_o_ub)
                                T.tile.mul(sumexp, sumexp, alpha_ub)
                                T.tile.add(sumexp, sumexp, sumexp_meta_ub)

                    with T.Scope("V"):
                        for h_i in range(block_M // 2):
                            T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

                        T.copy(acc_o, acc_o_half)
                        # 只写回 q_seq_len 范围内的有效行
                        row_start = bx * block_M + vid * block_M // 2
                        if row_start < q_seq_len:
                            T.copy(
                                acc_o_half,
                                Output[
                                    bz,
                                    by,
                                    row_start:row_start + block_M // 2,
                                    :,
                                ],
                            )

    return main
