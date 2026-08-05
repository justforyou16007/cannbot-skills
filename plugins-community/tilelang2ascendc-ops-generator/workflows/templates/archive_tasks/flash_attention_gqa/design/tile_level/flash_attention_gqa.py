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

# Tile-level TileLang design for GQA flash attention（原生 GQA）。
#
# 布局（host 零 pad）：Q[B,nq,Sq,D]、K/V[B,nkv,Skv,D]、Out[B,nq,Sq,D]，均用原始序列长度。
# K/V 尾块与 Q 尾块由 kernel 内自动处理（见 C1/V1/V2 的 tail 处理）。
#
# 2D 线性化核网格 (bNkv × gSq)，s2 不拆核：
#   本核负责 q 区间 [start_q, end_q)，每 task 内跑完整 KV 扫描（所有 s2 块都在同一 core），
#   running 状态跨 s2 自然串联。
#
# 任务内 s2 ring 流水：
#   for t in s2_outer_blocks + prelaunch:
#     if t < s2_outer_blocks:   C1(t)/V1(t)   # 生产：S=Q@K[s2]^T、softmax
#     if t >= prelaunch:        C2(t-2)/V2(t-2)  # 消费：O=P@V[s2]、merge
#   prelaunch=2 / ring_slots=3，slot = t % ring_slots 轮转。
#
# L1 缓冲：Q 常驻 q_l1[m_base_size, dim]（task 级加载一次，跨 s2 复用；
#   K/V 每 s2 预载 k_l1/v_l1[s2_base_size, dim]，s2 内 4 个 M 子块共享。
#   L0 为 block_M × block_N（128×128）；L0C double buffer（M 子块级交替）。
# running softmax 状态（max/sum）按行存 workspace_meta，跨 s2 读 prev_slot；
#   running O 走 workspace_acc_o GM 流水。

import tilelang
from tilelang import DataType, language as T
from tilelang.intrinsics import make_zn_layout


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# AIC<->AIV workspace 交接信号
SIG_S_READY = 0   # C1 写完 workspace_s -> V1 可读
SIG_P_READY = 1   # V1 写完 workspace_p -> C2 可读
SIG_O_READY = 2   # C2 写完 workspace_o -> V2 可读

MAX_CORES = 20   # used_core_num = min(q_work, MAX_CORES)


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6, 7, 8], pass_configs=pass_configs)
def flash_attention_gqa_fwd(
    batch,
    q_seq_len,
    kv_seq_len,
    heads,      # nq = query 头数
    kv_heads,   # nkv = key/value 头数
    dim,
    dtype="float16",
):
    block_M, block_N = 128, 128
    cube_K = 128
    prelaunch = 2
    ring_slots = prelaunch + 1
    m_base_size = 512          # 任务粒度（每任务 Q 行数）
    s2_base_size = 512         # 任务粒度（KV 外层步）

    accum_dtype = "float"

    group_num = heads // kv_heads
    sm_scale = (1.0 / dim) ** 0.5

    # ---- 布局（原生 GQA；host 零 pad，均用原始长度）----
    q_shape = [batch, heads, q_seq_len, dim]
    kv_shape = [batch, kv_heads, kv_seq_len, dim]
    out_shape = [batch, heads, q_seq_len, dim]

    # ---- tiling：任务瓦片（m_base/s2_base = 512）----
    q_outer_blocks = T.ceildiv(q_seq_len, m_base_size)
    s2_outer_blocks = T.ceildiv(kv_seq_len, s2_base_size)   # KV 512 行外层块数
    total_bnkv = batch * kv_heads
    total_gsq = group_num * q_outer_blocks
    q_work = total_bnkv * total_gsq                          # 切核维度（s2 不拆核）
    used_core_num = min(MAX_CORES, q_work)
    tasks_per_core = T.ceildiv(q_work, used_core_num)

    m_sub_num = m_base_size // block_M                       # 4 个 Q 128 子块
    s2_sub_per = s2_base_size // block_N                     # 每 s2 内 4 个 KV 128 子块
    chunk_rows = 16                                          # V1 softmax 行 chunk

    qk_k_loops = T.ceildiv(dim, cube_K)
    assert dim % cube_K == 0
    assert q_work > 0

    # 最后 s2 块的有效列数（K/V 尾块；kv_seq_len 是 s2_base 倍数时 = s2_base_size，无尾）
    tail_valid_s2 = kv_seq_len - (s2_outer_blocks - 1) * s2_base_size
    dim_align = ((dim + 15) // 16) * 16

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(kv_shape, dtype),  # type: ignore
        V: T.Tensor(kv_shape, dtype),  # type: ignore
        Output: T.Tensor(out_shape, dtype),  # type: ignore
        workspace_s: T.Tensor([used_core_num, ring_slots, m_base_size, s2_base_size], accum_dtype),
        workspace_p: T.Tensor([used_core_num, ring_slots, m_base_size, s2_base_size], dtype),
        workspace_o: T.Tensor([used_core_num, ring_slots, m_base_size, dim_align], accum_dtype),
        workspace_meta: T.Tensor([used_core_num, ring_slots, m_base_size, 3], accum_dtype),
        workspace_acc_o: T.Tensor([used_core_num, ring_slots, m_base_size, dim_align], accum_dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # ---- L1 缓冲（Q 常驻 + K/V 每 s2 缓存 512 行）----
            q_l1 = T.alloc_L1([m_base_size, dim], dtype)     # Q 常驻
            k_l1 = T.alloc_L1([s2_base_size, dim], dtype)    # K 缓存
            v_l1 = T.alloc_L1([s2_base_size, dim], dtype)    # V 缓存
            acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)
            T.annotate_layout({
                q_l1: make_zn_layout(q_l1),
                k_l1: make_zn_layout(k_l1),
                v_l1: make_zn_layout(v_l1),
                acc_s_l1: make_zn_layout(acc_s_l1),
            })

            lhs_l0 = T.alloc_L0A([block_M, cube_K], dtype)
            rhs_l0 = T.alloc_L0B([cube_K, block_N], dtype)
            # L0C double buffer（M 子块级交替 ）
            acc_s_l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

            # ---- V1（online softmax）buffer：S 按 16 行 × 512 列 chunk ----
            acc_s_ub = T.alloc_ub([chunk_rows, s2_base_size], accum_dtype)
            acc_s_half = T.alloc_ub([chunk_rows, s2_base_size], dtype)
            m_i = T.alloc_ub([chunk_rows], accum_dtype)        # 当前 chunk 新 max
            m_i_prev = T.alloc_ub([chunk_rows], accum_dtype)   # prev max / 输出 exp 系数 alpha
            sumexp = T.alloc_ub([chunk_rows], accum_dtype)     # prev running sum
            sumexp_i = T.alloc_ub([chunk_rows], accum_dtype)   # 当前 chunk running sum

            # ---- V2（merge）buffer ----
            acc_o_ub = T.alloc_ub([block_M // 2, dim], accum_dtype)
            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)
            acc_o_half = T.alloc_ub([block_M // 2, dim], dtype)
            alpha_ub = T.alloc_ub([block_M // 2], accum_dtype)
            sumexp_ub = T.alloc_ub([block_M // 2], accum_dtype)

            # 尾块列 mask：createvecindex + compare 生成按位掩码，select 把越界列填 -1e30
            col_idx = T.alloc_ub([s2_base_size], "int32")
            col_idx_f = T.alloc_ub([s2_base_size], "float")
            cmp_mask = T.alloc_ub([s2_base_size // 8], "uint8")

            # 本核线性 Q 区间（s2 不拆核）
            start_q = (q_work * cid) // used_core_num
            end_q = (q_work * (cid + 1)) // used_core_num

            for task_local in T.serial(tasks_per_core):
                cur_q = start_q + task_local
                if cur_q < end_q:
                    # ---- 任务反解 (bNkv, gSq) -> (bz, kvHead, qHead, qRowStart) ----
                    cur_bnkv = cur_q // total_gsq
                    cur_gsq = cur_q % total_gsq
                    bz = cur_bnkv // kv_heads
                    kv_head = cur_bnkv % kv_heads
                    group_id = cur_gsq // q_outer_blocks
                    q_outer_idx = cur_gsq % q_outer_blocks
                    q_head = kv_head * group_num + group_id
                    q_row_start = q_outer_idx * m_base_size

                    # ---- C0：Q 常驻 L1（；越界行安全读兜底，m_sub 守卫跳过）----
                    with T.Scope("C"):
                        T.copy(
                            Q[bz, q_head, q_row_start:q_row_start + m_base_size, :],
                            q_l1,
                        )

                    # ---- 任务内 s2 ring 流水----
                    #   if t < s2_outer_blocks:   C1(t)/V1(t) 生产（S=Q@K[s2]^T、softmax）
                    #   if t >= prelaunch:        C2(t-2)/V2(t-2) 消费（O=P@V[s2]、merge）
                    #   M 子块在 C1/C2 内部遍历（。
                    for t in T.serial(s2_outer_blocks + prelaunch):
                        # ---------- 生产者：C1(t) / V1(t) ----------
                        if t < s2_outer_blocks:
                            cur_s2 = t
                            slot_prod = t % ring_slots
                            is_first_s2 = (t == 0)
                            is_last_s2 = (t == s2_outer_blocks - 1)

                            with T.Scope("C"):
                                # 预载 K[s2] 512 行到 L1（M 子块共享）
                                T.copy(
                                    K[bz, kv_head, cur_s2 * s2_base_size:(cur_s2 + 1) * s2_base_size, :],
                                    k_l1,
                                )
                                # C1: S = Q_l1 @ K[s2]^T -> workspace_s[cid, slot_prod]
                                #   M 子块外层、KV 128 子块内层。
                                for m_sub in T.serial(m_sub_num):
                                    if q_row_start + m_sub * block_M < q_seq_len:   # Q 尾块守卫
                                        for n_blk in T.serial(s2_sub_per):
                                            for kk in T.serial(qk_k_loops):
                                                T.copy(q_l1[m_sub * block_M, kk * cube_K], lhs_l0)
                                                T.copy(k_l1[n_blk * block_N, kk * cube_K], rhs_l0, transpose=True)
                                                T.mma(lhs_l0, rhs_l0, acc_s_l0c[m_sub % 2, :, :], init=(kk == 0))
                                            T.copy(
                                                acc_s_l0c[m_sub % 2, :, :],
                                                workspace_s[
                                                    cid, slot_prod,
                                                    m_sub * block_M:m_sub * block_M + block_M,
                                                    n_blk * block_N:n_blk * block_N + block_N,
                                                ],
                                            )
                                T.set_cross_flag("FIX", SIG_S_READY)   # C1(t) 完成 -> V1(t)

                            with T.Scope("V"):
                                # V1: online softmax（整 s2 的 S 按 16 行 chunk）-> workspace_p + meta
                                #   running max/sum 跨 s2：isFirstS2 用默认，否则读 prev_slot meta
                                #   。
                                T.wait_cross_flag(SIG_S_READY)
                                if tail_valid_s2 != s2_base_size and is_last_s2:
                                    T.tile.createvecindex(col_idx, 0)
                                    T.copy(col_idx, col_idx_f)
                                for m_sub in T.serial(m_sub_num):
                                    if q_row_start + m_sub * block_M < q_seq_len:
                                        base_row = m_sub * block_M + vid * (block_M // 2)
                                        for c in T.serial((block_M // 2) // chunk_rows):
                                            base = base_row + c * chunk_rows
                                            # 读 prev running 状态（跨 s2）
                                            if is_first_s2:
                                                T.tile.fill(m_i_prev, -2**30)
                                                T.tile.fill(sumexp, 0.0)
                                            else:
                                                prev_slot = (slot_prod + ring_slots - 1) % ring_slots
                                                for r in range(chunk_rows):
                                                    m_i_prev[r] = workspace_meta[cid, prev_slot, base + r, 2]
                                                    sumexp[r] = workspace_meta[cid, prev_slot, base + r, 1]
                                            # 读 S chunk（当前 s2 全 512 列）
                                            T.copy(
                                                workspace_s[cid, slot_prod, base:base + chunk_rows, 0:s2_base_size],
                                                acc_s_ub,
                                            )
                                            T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                                            # 尾块列 mask（最后 s2 块）：列索引 > tail_valid_s2-1 -> -1e30
                                            if tail_valid_s2 != s2_base_size and is_last_s2:
                                                for r in range(chunk_rows):
                                                    T.tile.compare(cmp_mask, col_idx_f, T.float32(tail_valid_s2 - 1), "LE")
                                                    T.tile.select(
                                                        acc_s_ub[r, :], cmp_mask, acc_s_ub[r, :],
                                                        T.float32(-1e30), "VSEL_TENSOR_SCALAR_MODE",
                                                    )
                                            # online softmax
                                            T.reduce_max(acc_s_ub, m_i, dim=-1)
                                            T.tile.max(m_i, m_i, m_i_prev)     # 新 max = max(块 max, prev)
                                            T.tile.sub(m_i_prev, m_i_prev, m_i)  # alpha = prev_max - new_max
                                            T.tile.exp(m_i_prev, m_i_prev)
                                            for r in range(chunk_rows):
                                                T.tile.sub(acc_s_ub[r, :], acc_s_ub[r, :], m_i[r])
                                            T.tile.exp(acc_s_ub, acc_s_ub)
                                            T.reduce_sum(acc_s_ub, sumexp_i, dim=-1)
                                            # running sum = sum_i + alpha·prev_sum（跨 s2）
                                            T.tile.mul(sumexp, sumexp, m_i_prev)
                                            T.tile.add(sumexp_i, sumexp_i, sumexp)
                                            # 写 P（fp16）-> workspace_p
                                            T.copy(acc_s_ub, acc_s_half)
                                            T.copy(
                                                acc_s_half,
                                                workspace_p[cid, slot_prod, base:base + chunk_rows, 0:s2_base_size],
                                            )
                                            # 写 meta[slot]：0=exp 系数 alpha、1=running sum、2=新 max
                                            for r in range(chunk_rows):
                                                workspace_meta[cid, slot_prod, base + r, 0] = m_i_prev[r]
                                                workspace_meta[cid, slot_prod, base + r, 1] = sumexp_i[r]
                                                workspace_meta[cid, slot_prod, base + r, 2] = m_i[r]
                                T.set_cross_flag("MTE3", SIG_P_READY)  # V1(t) 完成 -> C2(t)

                        # ---------- 消费者：C2(t-2) / V2(t-2) ----------
                        if t >= prelaunch:
                            k = t - prelaunch
                            cur_s2 = k
                            slot_cons = k % ring_slots
                            is_first_s2 = (k == 0)
                            is_last_s2 = (k == s2_outer_blocks - 1)

                            with T.Scope("C"):
                                # C2: O = P @ V -> workspace_o[cid, slot_cons]
                                #   M 子块外层、KV 128 子块内层 L0C 累加；
                                #   workspace_o 在 n_blk 累加区间外每 m_sub 写一次（Fixpipe 语义）。
                                T.wait_cross_flag(SIG_P_READY)
                                # 预载 V[s2] 512 行到 L1（M 子块共享）
                                T.copy(
                                    V[bz, kv_head, cur_s2 * s2_base_size:(cur_s2 + 1) * s2_base_size, :],
                                    v_l1,
                                )
                                for m_sub in T.serial(m_sub_num):
                                    if q_row_start + m_sub * block_M < q_seq_len:
                                        for n_blk in T.serial(s2_sub_per):
                                            T.copy(
                                                workspace_p[
                                                    cid, slot_cons,
                                                    m_sub * block_M:m_sub * block_M + block_M,
                                                    n_blk * block_N:n_blk * block_N + block_N,
                                                ],
                                                acc_s_l1,
                                            )
                                            for dd in T.serial(qk_k_loops):
                                                T.copy(acc_s_l1, lhs_l0)
                                                T.copy(v_l1[n_blk * block_N, dd * cube_K], rhs_l0)
                                                T.mma(lhs_l0, rhs_l0, acc_o_l0c[m_sub % 2, :, :], init=(n_blk == 0 and dd == 0))
                                        # 累加完成，写 workspace_o（每 m_sub 一次，n_blk 循环外）
                                        for dd in T.serial(qk_k_loops):
                                            T.copy(
                                                acc_o_l0c[m_sub % 2, :, :],
                                                workspace_o[
                                                    cid, slot_cons,
                                                    m_sub * block_M:m_sub * block_M + block_M,
                                                    dd * cube_K:(dd + 1) * cube_K,
                                                ],
                                            )
                                T.set_cross_flag("FIX", SIG_O_READY)  # C2(t-2) 完成 -> V2(t-2)

                            with T.Scope("V"):
                                # V2: running O GM 流水（跨 s2）
                                T.wait_cross_flag(SIG_O_READY)
                                for m_sub in T.serial(m_sub_num):
                                    if q_row_start + m_sub * block_M < q_seq_len:
                                        base_row = m_sub * block_M + vid * (block_M // 2)
                                        # 读当前 slot meta（alpha、running sum）
                                        for r in range(block_M // 2):
                                            alpha_ub[r] = workspace_meta[cid, slot_cons, base_row + r, 0]
                                            sumexp_ub[r] = workspace_meta[cid, slot_cons, base_row + r, 1]
                                        # 当前 s2 块 O 贡献
                                        T.copy(
                                            workspace_o[cid, slot_cons, base_row:base_row + block_M // 2, :],
                                            acc_o_ub,
                                        )
                                        # prev running O（isFirstS2 视为 0）并 rescale
                                        if is_first_s2:
                                            T.tile.fill(acc_o, 0.0)
                                        else:
                                            prev_slot = (slot_cons + ring_slots - 1) % ring_slots
                                            T.copy(
                                                workspace_acc_o[cid, prev_slot, base_row:base_row + block_M // 2, :],
                                                acc_o,
                                            )
                                            for r in range(block_M // 2):
                                                T.tile.mul(acc_o[r, :], acc_o[r, :], alpha_ub[r])
                                        T.tile.add(acc_o, acc_o, acc_o_ub)          # merge 当前块
                                        if is_last_s2:
                                            # 末 s2 块：归一化并写回 Output（q 尾块行守卫）
                                            for r in range(block_M // 2):
                                                T.tile.div(acc_o[r, :], acc_o[r, :], sumexp_ub[r])
                                            T.copy(acc_o, acc_o_half)
                                            row_start = q_row_start + base_row
                                            if row_start < q_seq_len:
                                                T.copy(
                                                    acc_o_half,
                                                    Output[bz, q_head, row_start:row_start + block_M // 2, :],
                                                )
                                        else:
                                            # 非末 s2 块：写回当前 slot（running O 在 GM 流转）
                                            T.copy(
                                                acc_o,
                                                workspace_acc_o[cid, slot_cons, base_row:base_row + block_M // 2, :],
                                            )

    return main
