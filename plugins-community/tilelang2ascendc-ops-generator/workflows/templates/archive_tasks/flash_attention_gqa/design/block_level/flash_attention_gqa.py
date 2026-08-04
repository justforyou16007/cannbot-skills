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

"""Block-level TileLang design for GQA flash attention，按 AscendC 算子级调度 2D 线性化核网格。

对应 AscendC `flash_attention_gqa`（op_host/flash_attention_gqa_tiling.cpp、
op_kernel/flash_attention_kernel.h 等）。共同点：原生 GQA 布局、`(bNkv × gSq)` 2D 线性化
核网格、group 显式、`qHead` 映射、C/V 4 阶段 ring。满足 `BlockLevelDesign-quality-checklist.md`
（C/V 分工 #4、workspace #6、TODO #7、GEMM #10）。

## 原生 GQA 布局（AscendC LoadQInternal / ComputeMM1 一致）

    Q:   [B, nq,  Sq, D]      nq = heads（query 头数）
    K/V: [B, nkv, Skv, D]     nkv = kv_heads（key 头数），kv_seq_len 按 block_N pad
    Out: [B, nq,  Sq, D]
    group_num = nq / nkv；每个 KV 头被 group_num 个 query 头共享。

Q/out 偏移用 `bz*nq*qSeqLen + qHead*qSeqLen`，K/V 偏移用 `bz*nkv*kvSeqLen + kvHead*kvSeqLen`
（与 AscendC LoadQInternal / ComputeMM1 逐字对齐）。

## 任务划分：`(bNkv, gSq)` 2D 线性化（tiling + FillCoreRangeEnds，对照 tiling.cpp）

    totalBNkv = batch * nkv
    totalGSq  = groupNum * qOuterBlocks     （qOuterBlocks = CeilDiv(qSeqLenAlign, mBaseSize)）
    totalWork = totalBNkv * totalGSq
    usedCoreNum = min(totalWork, MAX_CORES=20)
    核网格：每核负责线性区间 `[startLinear, endLinear)`，linear = bNkv*totalGSq + gSq。
    start/endLinear 由 FillCoreRangeEnds 反解（bNkvEnd/gSqEnd 端点）；

## 任务反解（对照 kernel.h ring 分支，逐字一致）

    bz       = curBNkv     // nkv
    kvHead   = curBNkv     %  nkv
    groupId  = curGSq      // qOuterBlocks
    qOuterIdx= curGSq      %  qOuterBlocks
    qHead    = kvHead * groupNum + groupId
    qRowStart= qOuterIdx * mBaseSize
    KV 按 `[bz, kvHead, s2 块]` 取；Q/out 按 `[bz, qHead, q 块]` 取。

## C/V 子阶段（checklist #4，对应 AscendC C0/C1/V1/C2/V2）

    C0  预载 Q 块到 Cube 侧 L1（PrepareQ），贯穿本 qOuter 的所有 s2 块；isLastS2 -> ReleaseQ
    C1  S = Q @ K_t^T   （cube，L0A/L0B/L0C matmul，写 workspace_s）
    V1  online softmax（scale / 尾块 mask / reduce_max / exp / reduce_sum）+ 写 workspace_p + meta
    C2  O_tmp = P @ V_t （cube，写 workspace_o）
    V2  用 V1 的 alpha/sumexp rescale 并 merge 到运行累加器 + 归一化写 Output

时间线（kernel.h，PRELAUNCH=2 / RING_SLOTS=3）：
    cycle 0: C1(0)/V1(0)
    cycle 1: C1(1)+C2(0) / V1(1)+V2(0)
    cycle 2: C1(2)+C2(1) / V1(2)+V2(1) ... tail: C2/V2 收尾

## 同步策略（expert mode，对照 AscendC SetFlag/WaitFlag 事件编号）

    SIG_S_READY=0  C1 写完 workspace_s -> V1 读     （kernel.h sQueue ProducerReleaseFix）
    SIG_P_READY=1  V1 写完 workspace_p -> C2 读     （pQueue）
    SIG_O_READY=2  C2 写完 workspace_o -> V2 读     （oQueue）
    ring buffer：slot = seq % RING_SLOTS 轮转；PRELAUNCH=2 保证 AIC/AIV 不必等 free 信号。

## workspace（GM，AIC↔AIV 交接；对照 kernel.h wsS/wsP/wsO/wsMeta 布局）

| 张量 | shape | dtype | 语义 |
|---|---|---|---|
| `workspace_s`  | [usedCoreNum, RING_SLOTS, mBaseSize, s2BaseSize] | fp32 | C1 的 S = QK^T |
| `workspace_p`  | [usedCoreNum, RING_SLOTS, mBaseSize, s2BaseSize] | fp16 | V1 的 softmax 后 P |
| `workspace_o`  | [usedCoreNum, RING_SLOTS, mBaseSize, dimAlign]   | fp32 | C2 的 O_tmp = P@V |
| `workspace_meta`| [usedCoreNum, RING_SLOTS, mBaseSize, 3]          | fp32 | V1 的 (m_i, sumexp_i) 合并元数据 |

> 每核独立分片（perCoreBytes），dim 维照 AscendC dimAlign（AlignUp(dim,16)）。
> 细粒度计算见 tile-level（`design/tile_level/flash_attention_gqa.py`）。
"""

import math

import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# AIC<->AIV workspace 交接信号编号（与 AscendC workspace_queue.h SIG_* 事件对应）
SIG_S_READY = 0   # C1 写完 workspace_s -> V1 可读
SIG_P_READY = 1   # V1 写完 workspace_p -> C2 可读
SIG_O_READY = 2   # C2 写完 workspace_o -> V2 可读

MAX_CORES = 20   # AscendC MAX_CORES；usedCoreNum = min(totalWork, MAX_CORES)


def _fill_core_range_ends(total_bnkv, total_gsq, used_core_num, b_nkv_end, g_sq_end):
    """AscendC tiling.cpp FillCoreRangeEnds：把 totalWork*(core+1)/usedCoreNum 反解成
    (bNkvEnd, gSqEnd) 线性区间端点。tilelang kernel 内用同款 cid 数学等价推导，
    这里保留 Python 数组仅供对照 / 文档化。
    """
    total_work = total_bnkv * total_gsq
    for core in range(used_core_num):
        linear_end = total_work * (core + 1) // used_core_num
        b_nkv_end[core] = linear_end // total_gsq
        g_sq_end[core] = linear_end % total_gsq


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6, 7], pass_configs=pass_configs)
def flash_attention_gqa_fwd(
    batch,
    q_seq_len,
    kv_seq_len,
    heads,      # nq = query 头数
    kv_heads,   # nkv = key/value 头数
    dim,
    dtype="float16",
):
    block_M, block_N = 64, 64
    cube_K = 64
    prelaunch = 2
    ring_slots = prelaunch + 1

    accum_dtype = "float"

    group_num = heads // kv_heads
    sm_scale = (1.0 / dim) ** 0.5

    # ---- 布局（原生 GQA；kv_seq_len 与 q_seq_len 按块 pad）----
    # 照 AscendC tiling.cpp: kvSeqLenAlign = AlignUp(kvSeqLen, BLOCK_N)
    kv_seq_len_padded = ((kv_seq_len + block_N - 1) // block_N) * block_N
    # qSeqLenAlign = AlignUp(qSeqLen, BLOCK_M)
    q_seq_len_padded = ((q_seq_len + block_M - 1) // block_M) * block_M

    q_shape = [batch, heads, q_seq_len, dim]
    kv_shape = [batch, kv_heads, kv_seq_len_padded, dim]
    out_shape = [batch, heads, q_seq_len, dim]

    # ---- tiling 参数（对照 AscendC tiling.cpp；值供 tilelang 编译，偏离处以注释标注）----
    # AscendC: s2BaseSize = SelectS2BaseSize(qSeqLen, groupNum) = 512（qSeqLen>16）
    # AscendC: mBaseSize = SelectMBaseSize(qSeqLen, s2BaseSize) = 512
    # tilelang 偏离：AscendC 默认 512 会让 Q L1（mBaseSize×dim）超 L1 预算且块过大，
    # 这里降为 block_M/block_N（64），落入 tiling.cpp 的 `mBaseSize < BLOCK_M ? BLOCK_M` 下限。
    m_base_size = 64    # tilelang 偏离 AscendC 默认 512
    s2_base_size = 64   # tilelang 偏离 AscendC 默认 512/1024；= block_N，一个 s2 外块=一个 64 块
    q_outer_blocks = math.ceil(q_seq_len_padded / m_base_size)    # CeilDiv(qSeqLenAlign, mBaseSize)
    s2_outer_blocks = math.ceil(kv_seq_len_padded / s2_base_size)  # CeilDiv(kvSeqLen, s2BaseSize)
    total_bnkv = batch * kv_heads
    total_gsq = group_num * q_outer_blocks
    total_work = total_bnkv * total_gsq
    used_core_num = min(MAX_CORES, total_work)

    # bNkvEnd/gSqEnd 数组（AscendC FillCoreRangeEnds 逻辑，仅供对照）
    b_nkv_end = [0] * MAX_CORES
    g_sq_end = [0] * MAX_CORES
    _fill_core_range_ends(total_bnkv, total_gsq, used_core_num, b_nkv_end, g_sq_end)

    tail_valid = kv_seq_len % block_N      # 尾块有效列数（K/V 尾部块的无效列 mask）
    dim_align = ((dim + 15) // 16) * 16     # AlignUp(dim, 16)（AscendC dimAlign）

    # 核内最大 (bNkv,gSq) pair 数上界：任何核的 [startLinear,endLinear) <= ceil(totalWork/usedCore)
    q_block_count_ub = math.ceil(total_work / used_core_num)

    # 线性区间反解（AscendC kernel.h：startBNkv=gSq 由 FillCoreRangeEnds 端点推）
    # tilelang kernel 里用 cid 数学等价推导（与 FillCoreRangeEnds 同款：linearEnd=totalWork*(cid+1)//usedCore）
    def _kernel_start_end():
        # startLinear = totalWork*cid//usedCore，endLinear = totalWork*(cid+1)//usedCore
        return None  # 仅供注释说明；实际在 prim_func 内用 T 表达式推导

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
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            # C0 预载的 Q 块常驻 L1，贯穿本 qOuter 的所有 s2 块（AscendC qBufL1_）

            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)

            # 线性区间端点（AscendC FillCoreRangeEnds / kernel.h，用 cid 数学等价推导）
            start_linear = (total_work * cid) // used_core_num
            end_linear = (total_work * (cid + 1)) // used_core_num

            for task_local in T.serial(q_block_count_ub):
                # ---- 2D 线性化任务反解（对照 kernel.h，逐字一致）----
                cur_linear = start_linear + task_local
                if cur_linear < end_linear:
                    cur_bnkv = cur_linear // total_gsq
                    cur_gsq = cur_linear % total_gsq
                    bz = cur_bnkv // kv_heads
                    kv_head = cur_bnkv % kv_heads
                    group_id = cur_gsq // q_outer_blocks
                    q_outer_idx = cur_gsq % q_outer_blocks
                    q_head = kv_head * group_num + group_id
                    q_row_start = q_outer_idx * m_base_size

                    # ---- C0：预载 Q 块到 L1（AscendC PrepareQ），贯穿本 qOuter 全部 s2 块 ----
                    with T.Scope("C"):
                        # TODO(tile-level, C0): T.copy(Q[bz, q_head, q_row_start:q_row_start+block_M, :], q_l1)
                        #   annotate_layout(make_zn_layout)；跨 s2 块复用；isLastS2 尾块只取 q_seq_len 内有效行
                        T.copy(Q[bz, q_head, q_row_start:q_row_start + block_M, :], q_l1)
                        T.set_cross_flag("FIX", SIG_S_READY)   # 占位：C 侧 Q 就绪

                    # ---- V0：初始化本 qOuter 的运行累加器（AscendC vecKernel.maxCache/sumCache 默认态）----
                    with T.Scope("V"):
                        # TODO(tile-level, V0): acc_o=0, sumexp=0, m_i=-inf（T.tile.fill）
                        pass

                    # ---- ring 流水主循环（对照 kernel.h ring 分支：内层 s2 外块遍历 + prelaunch）----
                    # kernel.h 把 qBlockCount 与 s2OuterBlocks 拍平成单 cycle 序列
                    # （totalTasks = qBlockCount * s2OuterBlocks），ring slot = seq % RING_SLOTS
                    # 跨 (bNkv,gSq) 连续推进；本骨架用外层 pair + 内层 s2 块两层表达，slot 每对重置。
                    for t in T.serial(s2_outer_blocks + prelaunch):
                        # ---------- 生产者：C1(t) / V1(t) ----------
                        if t < s2_outer_blocks:
                            slot_prod = t % ring_slots
                            with T.Scope("C"):
                                # TODO(tile-level, C1): S = Q_l1 @ K_t^T -> workspace_s[cid, slot_prod]
                                #   - T.copy(K[bz, kv_head, t*block_N:(t+1)*block_N, :], k_l1)
                                #   - for kk in qk_k_loops:
                                #       T.copy(q_l1[0, kk*cube_K], lhs_l0)
                                #       T.copy(k_l1[0, kk*cube_K], rhs_l0, transpose=True)
                                #       T.mma(lhs_l0, rhs_l0, acc_s_l0c, init=(kk==0))
                                #   - T.copy(acc_s_l0c, workspace_s[cid, slot_prod, :, :])
                                T.set_cross_flag("FIX", SIG_S_READY)   # C1(t) 完成 -> V1(t)

                            with T.Scope("V"):
                                # TODO(tile-level, V1): online softmax -> workspace_p[cid,slot] + meta
                                #   - T.wait_cross_flag(SIG_S_READY)
                                #   - 取 workspace_s[cid, slot_prod] 的 vid 半 tile；乘 sm_scale=1/sqrt(D)
                                #   - 尾块(t==s2_outer_blocks-1)用 mask_col 把 [tail_valid, block_N) 置 -inf
                                #   - T.reduce_max -> m_i；T.tile.max(m_i, m_i, m_i_prev)
                                #   - exp(S - m_i)；T.reduce_sum -> sumexp_i
                                #   - 写 P -> workspace_p[cid, slot_prod]；写 (m_i_prev, sumexp_i) ->
                                #     workspace_meta[cid, slot_prod, vid 行, 0/1]
                                T.wait_cross_flag(SIG_S_READY)
                                T.set_cross_flag("MTE3", SIG_P_READY)  # V1(t) 完成 -> C2(t)

                        # ---------- 消费者：C2(now_k) / V2(now_k) ----------
                        if t >= prelaunch:
                            now_k = t - prelaunch
                            slot_cons = now_k % ring_slots
                            with T.Scope("C"):
                                # TODO(tile-level, C2): O_tmp = P @ V_t -> workspace_o[cid, slot_cons]
                                #   - T.wait_cross_flag(SIG_P_READY)
                                #   - T.copy(V[bz, kv_head, now_k*block_N:(now_k+1)*block_N, :], v_l1)
                                #   - for dd in qk_k_loops:
                                #       T.copy(workspace_p[cid, slot_cons], acc_s_l1)
                                #       T.copy(v_l1[0, dd*cube_K], rhs_l0)
                                #       T.mma(lhs_l0, rhs_l0, acc_o_l0c, init=True)
                                #   - T.copy(acc_o_l0c, workspace_o[cid, slot_cons, :, :])
                                T.wait_cross_flag(SIG_P_READY)
                                T.set_cross_flag("FIX", SIG_O_READY)   # C2(now_k) 完成 -> V2(now_k)

                            with T.Scope("V"):
                                # TODO(tile-level, V2): rescale + merge O_tmp，末块归一化写 Output
                                #   - T.wait_cross_flag(SIG_O_READY)
                                #   - 从 workspace_meta[cid, slot_cons, vid 行, 0/1] 取 alpha/meta_sumexp
                                #   - acc_o *= alpha；acc_o += workspace_o 的 vid 半 tile
                                #   - sumexp = sumexp*alpha + meta_sumexp
                                #   - isLastS2: acc_o /= sumexp；写 Output[bz, q_head,
                                #     q_row_start+startRow:... , :]（带 q_seq_len 守卫）
                                T.wait_cross_flag(SIG_O_READY)

    return main
