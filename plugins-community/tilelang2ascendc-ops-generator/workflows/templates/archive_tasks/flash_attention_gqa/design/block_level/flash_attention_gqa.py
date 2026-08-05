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

"""Block-level TileLang design for flash_attention_gqa（GQA flash attention）。

原生 GQA、host 零 pad：Q[B,nq,Sq,D]、K/V[B,nkv,Skv,D]、Out[B,nq,Sq,D]，group_num = nq/nkv。

任务划分（2D 线性化核网格 (bNkv × gSq)），**s2 不拆核**：同一 (bNkv, gSq) 的完整
KV 扫描（所有 s2 块）都在同一 core，running max/sum/AccO 状态跨 s2 自然串联（s2 拆核会让 core 1
读不到前 core 状态）。任务反解、输出/输入切片见代码。

任务内 s2 ring 流水：每 task 一个 t 循环，
if t < s2_outer_blocks 生产 C1(t)/V1(t)（S=Q@K[s2]^T、softmax），if t >= prelaunch
消费 C2(t-2)/V2(t-2)（O=P@V[s2]、merge）；M 子块在 C1/C2 内部遍历。循环骨架见代码。

L1 缓冲：Q 常驻 q_l1[m_base_size, dim]（isFirstS2 加载）、K/V 每 s2 预载
k_l1/v_l1[s2_base_size, dim]；L0 为 block_M × block_N（128×128），L0C double buffer（M 子块交替）。

技术约束：
- K/V 尾块：越界读安全兜底；tail_valid 驱动 T.tile.select 按位掩码填 -1e30
- q 尾块：整块越界 m_sub 跳过（写 Output 越界行由编译器自动跳过）
- GEMM 容量：128×128×4 = 64KB ≤ L0C 128KB ✓

细算见各段 `# TODO(tile-level, ...)` 注释。
"""

import math

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# AIC<->AIV workspace 交接信号编号
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
    s2_base_size = 512         # 任务粒度（KV 外层步；内部按 block_N 扫描）

    accum_dtype = "float"

    group_num = heads // kv_heads
    sm_scale = (1.0 / dim) ** 0.5

    # ---- 布局（原生 GQA；host 零 pad，均用原始序列长度）----
    q_shape = [batch, heads, q_seq_len, dim]
    kv_shape = [batch, kv_heads, kv_seq_len, dim]
    out_shape = [batch, heads, q_seq_len, dim]

    # ---- tiling：任务瓦片 ----
    # 切核维度 = qWork（total_bnkv × total_gsq），s2 不拆核。
    # 任务内：Q 常驻（m_base_size 行）+ KV 按 s2_base_size(512) 外层预载 + block_N 子块流水。
    q_outer_blocks = math.ceil(q_seq_len / m_base_size)
    s2_outer_blocks = math.ceil(kv_seq_len / s2_base_size)   # KV 512 行外层块数
    total_bnkv = batch * kv_heads
    total_gsq = group_num * q_outer_blocks
    q_work = total_bnkv * total_gsq                          # 切核维度（s2 不拆核）
    used_core_num = min(MAX_CORES, q_work)
    tasks_per_core = math.ceil(q_work / used_core_num)
    m_sub_num = m_base_size // block_M                       # 4 个 Q 128 子块
    s2_sub_per = s2_base_size // block_N                     # 4 个 KV 128 子块/s2

    tail_valid = kv_seq_len % block_N    # 尾块有效列数
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
            # ---- L1 缓冲（Q 常驻 + KP/V 缓存 512 行粒度）----
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

            # V1（online softmax）buffer
            sumexp = T.alloc_ub([block_M // 2], accum_dtype)
            m_i = T.alloc_ub([block_M // 2], accum_dtype)
            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)
            acc_s_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            m_i_prev = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_ub_ = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            sumexp_i_ub = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)
            acc_o_ub = T.alloc_ub([block_M // 2, dim], accum_dtype)
            acc_o_half = T.alloc_ub([block_M // 2, dim], dtype)
            alpha_ub = T.alloc_ub([block_M // 2], accum_dtype)
            sumexp_meta_ub = T.alloc_ub([block_M // 2], accum_dtype)
            # 尾块列 mask
            col_idx = T.alloc_ub([block_N], "int32")
            col_idx_f = T.alloc_ub([block_N], "float")
            cmp_mask = T.alloc_ub([block_N // 8], "uint8")

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
                        # TODO(tile-level, C0): Q 常驻加载细算
                        T.copy(
                            Q[bz, q_head, q_row_start:q_row_start + m_base_size, :],
                            q_l1,
                        )

                    # ---- 任务内 s2 ring 流水----
                    #   if t < s2_outer_blocks:   C1(t)/V1(t) 生产（S=Q@K[s2]^T、softmax）
                    #   if t >= prelaunch:        C2(t-2)/V2(t-2) 消费（O=P@V[s2]、merge）
                    #   M 子块在 C1/C2 内部遍历（。
                    #   slot = t % ring_slots 轮转。
                    for t in T.serial(s2_outer_blocks + prelaunch):
                        # ---------- 生产者：C1(t) / V1(t) ----------
                        if t < s2_outer_blocks:
                            cur_s2 = t
                            slot_prod = t % ring_slots
                            is_first_s2 = (t == 0)
                            is_last_s2 = (t == s2_outer_blocks - 1)
                            with T.Scope("C"):
                                # 预载 K[s2] 512 行到 L1（M 子块共享）
                                # TODO(tile-level, C1): K 预载细算
                                T.copy(
                                    K[bz, kv_head, cur_s2 * s2_base_size:(cur_s2 + 1) * s2_base_size, :],
                                    k_l1,
                                )
                                # C1: S = Q_l1 @ K[s2]^T -> workspace_s[cid, slot_prod]
                                #   TODO(tile-level, C1): M 子块 × KV 子块的 gemm 细算
                                T.set_cross_flag("FIX", SIG_S_READY)   # C1(t) 完成 -> V1(t)
                            with T.Scope("V"):
                                # V1: online softmax（整 s2 的 S 按 16 行 chunk）-> workspace_p + meta
                                #   TODO(tile-level, V1): softmax 细算（prev running、tail mask、reduce）
                                T.wait_cross_flag(SIG_S_READY)
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
                                #   TODO(tile-level, C2): V 预载 + M 子块 × KV 子块累加的 gemm 细算
                                T.wait_cross_flag(SIG_P_READY)
                                T.set_cross_flag("FIX", SIG_O_READY)  # C2(t-2) 完成 -> V2(t-2)
                            with T.Scope("V"):
                                # V2: running O GM 流水（跨 s2）
                                #   TODO(tile-level, V2): merge 细算（prev rescale、归一化写 Output/写回）
                                T.wait_cross_flag(SIG_O_READY)

    return main
