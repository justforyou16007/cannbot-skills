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

"""Block-level TileLang design for flash_attention（MHA，kv_heads == heads）。

捕捉任务划分、阶段边界、ring buffer 归属与跨 scope 同步；细粒度数学
（softmax / merge / 归一化）留作 TODO，由 tile-level 设计填充。
可运行实现见 design/tile_level/flash_attention.py，本文件即其结构骨架。

任务划分：
  每 task = 一个 (bz, by, bx) 的 block_m 行 Q 块，整 KV 扫描在同一核；
  used_core_num = min(20, block_num)，task_idx = cid * tasks_per_core + local_idx，
  越界 task 用 if 守卫（TIR 无 continue）。

任务内 KV ring 流水（prelaunch=2 / ring_slots=3，slot = t % ring_slots）：
  for t in kv_loops + prelaunch:
    if t < kv_loops:      C1(t)/V1(t)     # 生产：S=Q@K^T -> ws_s；softmax -> ws_p + meta
    if t >= prelaunch:    C2(t-2)/V2(t-2) # 消费：O=P@V -> ws_o；rescale merge

跨 scope 同步（AIC<->AIV 握手）：
  SIG_S_READY=0：C1 写完 ws_s -> V1 可读（FIX set / V wait）
  SIG_P_READY=1：V1 写完 ws_p + meta -> C2 可读（MTE3 set / C wait）
  SIG_O_READY=2：C2 写完 ws_o -> V2 可读（FIX set / V wait）

Tail：K/V/Q 的 GM 越界读、Output 越界写由编译器自动处理（安全读 / 跳过越界写）；
  唯一例外是 softmax 的 kv 尾列——越界列读 0 会被 exp(0)=1 当成有效分数污染分母，
  需在 V1 内显式 mask 成大负值（仅最后一个 kv 块）。
"""

import tilelang
import tilelang.language as T


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
def flash_attention_fwd(batch, heads, q_seq_len, kv_seq_len, dim, dtype="float16"):
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
    # cube_k 无需整除 dim。
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
        # C<->V 交接全部走 GM workspace，按 task 粒度持有（task_idx 索引）：
        workspace_s: T.Tensor([block_num, ring_slots, block_m, block_n], accum_dtype),
        workspace_p: T.Tensor([block_num, ring_slots, block_m, block_n], dtype),
        workspace_o: T.Tensor([block_num, ring_slots, block_m, dim], accum_dtype),
        # meta 行布局：0=alpha（exp(prev_max - new_max)），1=当前块 sumexp
        workspace_meta: T.Tensor([block_num, ring_slots, block_m, 2], accum_dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # ---- L1 缓冲（Q 常驻 task 级；K/V 每 kv 块缓存；P 经 acc_s_l1 过路；
            #       dim 维按 dim_pad 补齐）----
            q_l1 = T.alloc_L1([block_m, dim_pad], dtype)
            k_l1 = T.alloc_L1([block_n, dim_pad], dtype)
            v_l1 = T.alloc_L1([block_n, dim_pad], dtype)
            acc_s_l1 = T.alloc_L1([block_m, block_n], dtype)

            # ---- L0：C1 用 lhs/rhs（K 维 cube_k 切片）；C2 用 p_l0/v_l0 ----
            lhs_l0 = T.alloc_L0A([block_m, cube_k], dtype)
            rhs_l0 = T.alloc_L0B([cube_k, block_n], dtype)
            p_l0 = T.alloc_L0A([block_m, block_n], dtype)
            v_l0 = T.alloc_L0B([block_n, cube_k], dtype)
            acc_s_l0c = T.alloc_L0C([block_m, block_n], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_m, cube_k], accum_dtype)

            # ---- V 侧 UB：每个 AIV 子核处理 block_m//2 行 ----
            # running O / sumexp / max 驻留 UB，逐 kv 块更新
            acc_o = T.alloc_ub([block_m // 2, dim], accum_dtype)
            sumexp = T.alloc_ub([block_m // 2], accum_dtype)
            m_i = T.alloc_ub([block_m // 2], accum_dtype)
            acc_s_ub = T.alloc_ub([block_m // 2, block_n], accum_dtype)
            acc_s_half = T.alloc_ub([block_m // 2, block_n], dtype)

            for local_idx in T.serial(tasks_per_core):
                task_idx = cid * tasks_per_core + local_idx
                if task_idx < block_num:
                    bx = task_idx % q_blocks
                    by = task_idx // q_blocks % heads
                    bz = task_idx // q_blocks // heads % batch

                    # ---- C0：Q 常驻 L1（尾块越界读编译器零填充）----
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
                                    # TODO(tile-level, C1)：Q/K 的 K 维切片进 L0 并 mma
                                    # （init=(kk == 0)），K^T 走 transpose copy
                                    pass
                                # TODO(tile-level, C1)：acc_s_l0c 写 workspace_s
                                T.set_cross_flag("FIX", SIG_S_READY)   # C1(t) 完成 -> V1(t)

                            with T.Scope("V"):
                                # V1: online softmax -> workspace_p + meta
                                T.wait_cross_flag(SIG_S_READY)
                                # TODO(tile-level, V1)：
                                # - 读 ws_s 半 tile（vid * block_m//2 行起），乘 sm_scale
                                # - kv 尾块（tail_valid != 0 且 t == kv_loops-1）越界列
                                #   mask 成大负值（softmax 特例：越界 0 会污染分母）
                                # - running max/sumexp 在线更新，alpha = exp(prev - new)
                                # - P cast 成 dtype 写 ws_p；alpha/sumexp_i 写 ws_meta
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
                                    # TODO(tile-level, C2)：P 进 p_l0、V 的 dim 切片进
                                    # v_l0，mma（init=True）后按 dim 切片写 ws_o
                                    pass
                                T.set_cross_flag("FIX", SIG_O_READY)  # C2(t-2) 完成 -> V2(t-2)

                            with T.Scope("V"):
                                # V2: merge 当前 O 贡献到 running 状态
                                T.wait_cross_flag(SIG_O_READY)
                                # TODO(tile-level, V2)：
                                # - 读 ws_meta 的 alpha/sumexp_i（当前 slot）
                                # - 读 ws_o 半 tile；acc_o 按行乘 alpha 后累加
                                # - sumexp = sumexp * alpha + sumexp_i
                                pass

                    # ---- 收尾：归一化并写出（越界写编译器自动跳过）----
                    with T.Scope("V"):
                        # TODO(tile-level, V2-epilogue)：
                        # - acc_o 按行除以 sumexp，cast 成 dtype
                        # - 写 output[bz, by, bx*block_m + vid*block_m//2 起半 tile]
                        pass

    return main
