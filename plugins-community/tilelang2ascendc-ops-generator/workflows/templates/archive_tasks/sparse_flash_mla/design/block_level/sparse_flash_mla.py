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

"""Block-level TileLang design for sparse_flash_mla (SWA / HCA / CSA).

The task partition mirrors the packaged operator's metadata-driven 2D grid
(``FA_METADATA`` core ranges in ``op_kernel/sparse_flash_mla_metadata.h``);
the fine-grained math is filled in by the tile-level design.

任务划分（2D 线性化核网格 (bN2 × gSq)，**s2 不拆核**）：
- 任务空间 = bN2 × gSq，bN2 = batch × n2（KV 元组），
  gSq = group × q_outer_blocks（q 外层块，m_base_size 行粒度）。
- 同一 (bN2, gSq) 的完整 KV 扫描（ori 段窗口 + cmp 段前缀）都在同一 core，
  running max/sumexp/AccO 状态跨 s2 自然串联（s2 拆核会让 core 读不到前核状态）。
- 核数为 used_core_num = min(MAX_CORES, q_work)，每核线性区间反解
  (bz, kvHead, qHead, qRowStart)，与打包算子 FA/FD metadata 的任务区间同构
  （打包算子把区间写成每核 [bn2_start, m_start, s2_start, ...] 的 metadata 表，
  这里用算术反解等价表达）。

KV 双段 walk：
- 每任务内被注意的 KV 流 = ori 段窗口 [o_start(s), o_end(s)) ∪ cmp 段前缀
  [0, cmp_thr(s))，两段在**同一个 serial 循环**里按块拼接（ori 块在前、
  cmp 块在后），共享同一 flash-softmax 状态机。
- 掩码而不是跳过：块完全越界 -> mask 范围 0；部分越界 -> 逐列
  MASK_FILL (-2^30) 掩码，exp() 下溢为 0。循环体分支无关，
  o_start/o_end/cmp_thr 是查询行 s 的 TIR 表达式（decode 行 s==0 折为常量）。

MIX AIC/AIV ring 流水（与打包算子的 AIC/AIV 分工一致）：
- 每任务一个 t 循环（总块数 + prelaunch 拍）：
  if t < total_blocks 生产 C1(t)/V1(t)（S=Q@K[seg]^T、softmax，写 GM workspace）；
  if t >= prelaunch 消费 C2(t-2)/V2(t-2)（O=P@V、merge）。
- AIC 做 QK/PV gemm，AIV 做 softmax/merge（两个 AIV 各持 BLOCK_H 的一半行，
  vid 分工）；workspace ring 的 slot = t % ring_slots 轮转，跨核交接用
  set/wait_cross_flag 信号（tilelang-ascend 的 AUTO_CV_SYNC 也可自动推导，
  归档 tile_level 即用自动路径）。
- 缓冲：Q 常驻 L1（m_base_size 行）、KV 块预载 L1、L0C 128×128（QK）
  与 128×D（PV）double buffer。注意 D=512 时 PV 的 L0C 容量
  128×512×4 = 256KB > 128KB，因此 head 块取 BLOCK_H=32（32×512×4 = 64KB），
  head 维以 32 行分块循环。

Flash-softmax 状态机（sinks 为虚拟 key，与 model.py:softmax_with_sinks 一致）：
    m_i     := sinks                        （初始 = 虚拟 key 的 max）
    sumexp  := 1                            （= exp(sinks - m_i)）
    per block: m_old = m_i;  m_i = max(m_i, rowmax(score));
               alpha = exp(m_old - m_i)
               sumexp = alpha * sumexp + rowsum(exp(score - m_i))
               acc_o  = alpha * acc_o + P @ V
    epilogue: out = acc_o / sumexp
sinks 只进分母不进分子（贡献零值）。

CSA 扩展点（当前 tile_level 未覆盖，见 model_new_tilelang.py 的
NotImplementedError）：每查询行按 cmp_sparse_indices 的 topk 前缀 gather
cmp_kv 行——用 T.tile.gather（UB tile 内元素 gather）或逐行动态 GM 拷贝
cmp_kv[b, idx, :]；gather 实现必须复现打包算子的前缀+停止+跳过规则
（仅前 min(K, threshold) 项、-1 停止、>= threshold 跳过），
见 model.py:gather_cmp_indices 与 kernel/ 源码 ComputeS2LoopInfo。

归档说明：design/tile_level/ 是简化直通实现（单核单任务 1D 网格 + 无
workspace ring，已在本机 910B 全量验证）；本文件是与打包算子调度的对齐
描述，ring/双缓冲细算见各段 TODO(tile-level) 标注，供生成 MLA 类算子时
直接参考。
"""

import math

import tilelang
import tilelang.language as T

from sparse_flash_mla_common import (
    MASK_NO,
    MASK_CAUSAL,
    MASK_BAND,
    UNLIMITED,
    MASK_FILL,
    BLOCK_N,
    pass_configs,
)

# query-head block size: 32 rows keep one PV tile [32, 512] fp32 (64KB) in L0C
BLOCK_H = 32

# KV 外层预载粒度（与打包算子 s2BaseSize 同量级）
S2_BASE = 512

# AIC<->AIV workspace 交接信号编号
SIG_S_READY = 0   # C1 写完 workspace_s -> V1 可读
SIG_P_READY = 1   # V1 写完 workspace_p -> C2 可读
SIG_O_READY = 2   # C2 写完 workspace_o -> V2 可读

MAX_CORES = 20    # used_core_num = min(q_work, MAX_CORES)


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6, 7, 8], pass_configs=pass_configs)
def sparse_flash_mla_fwd(
    batch,
    q_seq,
    heads,
    dim,
    ori_seq,
    cmp_seq,
    cmp_ratio,
    ori_mask_mode,
    cmp_mask_mode,
    ori_win_left,
    ori_win_right,
):
    """Block-level kernel entry: 2D task partition + two-segment KV walk.

    The signature mirrors the tile-level config; the body keeps the structural
    skeleton (T.Kernel -> task unroll -> two-segment KV loop -> softmax state
    update) with the fine-grained math marked ``TODO(tile-level, ...)``.
    """
    dtype = "bfloat16"
    accum_dtype = "float"

    group = heads  # KV_N == 1 is an operator constraint, so group = heads
    m_base_size = 512          # 任务粒度（每任务 Q 行数）
    q_outer_blocks = math.ceil(q_seq / m_base_size)
    total_bnkv = batch * 1     # n2 == 1
    total_gsq = group * q_outer_blocks
    q_work = total_bnkv * total_gsq
    used_core_num = min(MAX_CORES, q_work)
    tasks_per_core = math.ceil(q_work / used_core_num)

    ori_blocks = math.ceil(ori_seq / BLOCK_N)
    cmp_blocks = math.ceil(cmp_seq / BLOCK_N)
    total_blocks = ori_blocks + cmp_blocks
    prelaunch = 2
    ring_slots = prelaunch + 1

    q_shape = [batch, q_seq, heads, dim]
    kv_shape = [batch, ori_seq + cmp_seq, dim]   # 逻辑拼接 KV 流
    out_shape = [batch, q_seq, heads, dim]

    @T.prim_func
    def main(
        q: T.Tensor(q_shape, dtype),
        ori_kv: T.Tensor([batch, ori_seq, dim], dtype),
        cmp_kv: T.Tensor([batch, cmp_seq, dim], dtype),
        sinks: T.Tensor([heads], "float"),
        output: T.Tensor(out_shape, dtype),
        workspace_s: T.Tensor([used_core_num, ring_slots, m_base_size, BLOCK_N], accum_dtype),
        workspace_p: T.Tensor([used_core_num, ring_slots, m_base_size, BLOCK_N], dtype),
        workspace_o: T.Tensor([used_core_num, ring_slots, m_base_size, dim], accum_dtype),
        workspace_meta: T.Tensor([used_core_num, ring_slots, m_base_size, 3], accum_dtype),
        workspace_acc_o: T.Tensor([used_core_num, ring_slots, m_base_size, dim], accum_dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # ---- L1 缓冲（Q 常驻 + KV 预载）----
            q_l1 = T.alloc_L1([m_base_size, dim], dtype)
            kv_l1 = T.alloc_L1([S2_BASE, dim], dtype)
            # ---- L0C（QK 128×128 与 PV 128×D 交替；D=512 时 PV 用
            #      BLOCK_H=32 行块，见模块 docstring）----
            scores_l0c = T.alloc_L0C([BLOCK_H, BLOCK_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([BLOCK_H, dim], accum_dtype)
            # ---- UB 状态（flash-softmax，每 AIV 持 BLOCK_H/2 行）----
            m_i = T.alloc_ub([BLOCK_H // 2], accum_dtype)
            sumexp = T.alloc_ub([BLOCK_H // 2], accum_dtype)
            acc_o_ub = T.alloc_ub([BLOCK_H // 2, dim], accum_dtype)
            scores_ub = T.alloc_ub([BLOCK_H // 2, BLOCK_N], accum_dtype)
            mask_col = T.alloc_ub([BLOCK_N], accum_dtype)
            # idx = [BLOCK_N] f32 arange 常量由调用方传入（见 tile_level 实现）

            # 本核线性任务区间（s2 不拆核）
            start_q = (q_work * cid) // used_core_num
            end_q = (q_work * (cid + 1)) // used_core_num

            for task_local in T.serial(tasks_per_core):
                cur_q = start_q + task_local
                if cur_q < end_q:
                    # ---- 任务反解 (bN2, gSq) -> (bz, kvHead, qHead, qRowStart) ----
                    cur_bnkv = cur_q // total_gsq
                    cur_gsq = cur_q % total_gsq
                    bz = cur_bnkv // 1
                    group_id = cur_gsq // q_outer_blocks
                    q_outer_idx = cur_gsq % q_outer_blocks
                    q_head = group_id            # n2 == 1: q_head == group_id
                    q_row_start = q_outer_idx * m_base_size

                    # ---- C0：Q 常驻 L1（m_base_size 行；越界行安全读兜底，
                    #      m_sub 守卫跳过）----
                    with T.Scope("C"):
                        # TODO(tile-level, C0): Q 常驻加载细算
                        T.copy(q[bz, q_row_start:q_row_start + m_base_size, q_head, :], q_l1)

                    # ---- 任务内双段 KV ring 流水 ----
                    #   两段（ori 窗口 + cmp 前缀）拼接为一个 t 循环：
                    #   t < ori_blocks  -> ori 段块（窗口掩码）
                    #   否则             -> cmp 段块（前缀阈值掩码）
                    #   if t < total_blocks: 生产 C1(t)/V1(t)
                    #   if t >= prelaunch:   消费 C2(t-2)/V2(t-2)
                    for t in T.serial(total_blocks + prelaunch):
                        # ---------- 生产者：C1(t) / V1(t) ----------
                        if t < total_blocks:
                            is_ori = t < ori_blocks
                            cur_seg = t - T.if_then_else(is_ori, 0, ori_blocks)
                            slot_prod = t % ring_slots
                            with T.Scope("C"):
                                # 预载 KV 块（ori: kv_all[ori_base + seg*BLOCK_N]；
                                #   cmp: kv_all[ori_base + ori_blocks*BLOCK_N + ...]）
                                # TODO(tile-level, C1): KV 预载 + QK gemm 细算
                                #   （ori 段每列按窗口掩码、cmp 段按阈值掩码）
                                T.set_cross_flag("FIX", SIG_S_READY)   # C1(t) 完成
                            with T.Scope("V"):
                                # V1: online softmax（MASK_FILL 掩码、tail）-> workspace_p
                                T.wait_cross_flag(SIG_S_READY)
                                T.set_cross_flag("MTE3", SIG_P_READY)  # V1(t) 完成
                        # ---------- 消费者：C2(t-2) / V2(t-2) ----------
                        if t >= prelaunch:
                            k = t - prelaunch
                            slot_cons = k % ring_slots
                            with T.Scope("C"):
                                # C2: O = P @ V -> workspace_o（PV 以 BLOCK_H=32
                                #   行块累加，因 D=512 时 128 行整块的 L0C 超限）
                                T.wait_cross_flag(SIG_P_READY)
                                T.set_cross_flag("FIX", SIG_O_READY)  # C2(t-2) 完成
                            with T.Scope("V"):
                                # V2: running O GM 流水（跨 s2 merge、归一化）
                                T.wait_cross_flag(SIG_O_READY)

    return main
