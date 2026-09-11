#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may obtain a copy of this software and associated documentation files (the "License"), under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------

# Block-level TileLang design for GQA（cann-bench level4/gqa，BSND 布局）。
#
# 本文件是 design/tile_level/gqa.py 的结构骨架：任务划分、阶段边界、ring buffer
# 归属、跨 scope 同步全部固定；细粒度数学以 TODO(tile-level) 标注，由 tile-level
# 填充。两文件除 TODO 段外一致。
#
# ── 核心结构：G 折叠进 M（catlass FAI 对齐，investigation_catlass_expert.md §一）──
#   task = (bz, nkv, s_blk, g_blk)，一个 task 绑定一个 kv head；
#   M tile 行 = bs(S 块) × bg(组内 head 块)，head-major 行排布
#   （行 r = g_local * bs + s_local，与 golden reshape/permute 的 G*S 折叠序一致）。
#   K/V tile 一次 load 被 bs*bg 行整体复用，K/V 不按 q head 物化。
#
# ── 布局（BSND，[B, S, N, D]）─────────────────────────────────────────────
#   Q[b, s, nkv*G+g, :]；K/V[b, j, nkv, :]。
#   C0 Q 装载：per-g 循环 T.copy(q[bz, s0:s0+bs, hq0+g, :], q_l1[g*bs:(g+1)*bs, :])
#     —— 源 extents [1, bs, 1, D] 恰 2 活跃维（严禁 ≥3 活跃维 region→2D buffer，
#     见 investigation_compiler_copy.md §2），L1 行切片 splice 有测试背书。
#   epilogue 写出：per-g T.copy(acc_o_half[...], output[bz, s0+.., hq, :])，
#     dst extents [1, bs, 1, D] 同构。
#
# ── 关键尺寸 ────────────────────────────────────────────────────────────────
#   block_n=128；cube_k = min(dim_align, 128)，d_chunks = ceildiv(dim_align, 128)
#   （D=256 → 2 chunk：QK 两路独立 mma(init=True) 各落 ws_s 的 chunk 槽，
#    V1 相加得 S；PV 按 N 维 128 分 chunk 各落 ws_o 列切片。
#    规避 tilelang pto 后端 kv>64&dim>64 多 chunk mma 累加的已知崩溃 M2）。
#   m_budget = 128（dim_align≤128）/ 64（dim_align=256，UB 容量约束）；
#   bs = min(m_budget, S)，bg = min(G, m_budget//bs)，block_m = ceil16(bs*bg)
#   （M 凑 16 对齐 cube 分形；无效行 Q 零填充、epilogue 跳过写）。
#
# ── causal（右下对齐，j ≤ i + (Skv−S)）─────────────────────────────────────
#   diffS = Skv - S；kv_end = min(kv_loops, (s0+bs-1+diffS)//block_n + 1)
#   —— diffS 循环裁剪（FAI 手法）：t ≥ kv_end 的 kv 块生产/消费双端跳过；
#   对角块（(t+1)*block_n-1 > s0+diffS）在 V1 内 UB 现算 mask
#   （createvecindex 列索引 + 逐行标量 bound compare(LE) + select 填 -2^30）。
#   kv 尾块（Skv % block_n != 0）列 mask 沿用 FA 的 mask_col 加 -inf 手法。
#
# ── 任务内 KV ring 流水（FA archive 母本）─────────────────────────────────
#   for t in kv_loops + prelaunch:
#     if t < kv_end:                    C1(t)/V1(t)   # S=Q@K^T→ws_s；softmax→ws_p
#     if t>=prelaunch and t-2 < kv_end: C2(t-2)/V2(t-2) # O=P@V→ws_o；rescale merge
#   prelaunch=2 / ring_slots=3，slot = t % ring_slots；running 状态 UB ring 驻留。
#
# Tail：GM 越界读零填充 / 越界写跳过由编译器处理；g 尾块（G % bg != 0）与
#   s 尾块（S % bs != 0）由 task 内守卫保证。

from dataclasses import dataclass

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# AIC<->AIV workspace 交接信号
SIG_S_READY = 0   # C1 写完 workspace_s -> V1 可读
SIG_P_READY = 1   # V1 写完 workspace_p -> C2 可读
SIG_O_READY = 2   # C2 写完 workspace_o -> V2 可读

MAX_CORES = 20   # grid 固定物理核数；task_idx < block_num 守卫

MASK_NEG = -2 ** 30   # softmax 前 mask 填充值（替代 -inf，避免 NaN 传播路径差异）


@dataclass(frozen=True)   # frozen: tilelang.jit 需要对 cfg 取 hash 做编译缓存键
class GQAConfig:
    batch: int
    heads_q: int
    heads_kv: int
    q_seq_len: int
    kv_seq_len: int
    dim: int
    causal: bool = False
    dtype: str = "float16"


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs, target='ascendc')
def gqa_fwd(cfg: GQAConfig):
    batch = cfg.batch
    heads_q = cfg.heads_q
    heads_kv = cfg.heads_kv
    q_seq_len = cfg.q_seq_len
    kv_seq_len = cfg.kv_seq_len
    dim = cfg.dim
    causal = cfg.causal
    dtype = cfg.dtype

    assert heads_q % heads_kv == 0, "N_q % N_kv != 0"
    group = heads_q // heads_kv
    assert not (causal and q_seq_len > kv_seq_len), "causal requires S <= S_kv"

    # C0 对齐（32B/elem：fp16/bf16→16）
    c0 = 16
    dim_align = ((dim + c0 - 1) // c0) * c0

    block_n = 128
    cube_k = min(dim_align, 128)
    d_chunks = (dim_align + 127) // 128
    prelaunch = 2
    ring_slots = prelaunch + 1
    accum_dtype = "float"

    # G 折叠进 M 的行预算：dim_align>128 时 UB 驻留状态随 dim 翻倍，行预算减半
    m_budget = 128 if dim_align <= 128 else 64
    bs = min(m_budget, q_seq_len)
    bg = min(group, max(1, m_budget // bs))
    block_m = ((bs * bg + 15) // 16) * 16   # M 凑 16 对齐 cube 分形

    s_blks = (q_seq_len + bs - 1) // bs
    g_blks = (group + bg - 1) // bg
    block_num = batch * heads_kv * s_blks * g_blks
    used_core_num = MAX_CORES
    tasks_per_core = (block_num + used_core_num - 1) // used_core_num
    kv_loops = (kv_seq_len + block_n - 1) // block_n

    diff_s = kv_seq_len - q_seq_len   # causal 右下对齐偏移（非 causal 不使用）

    # BSND 布局 shape
    q_shape = [batch, q_seq_len, heads_q, dim]
    kv_shape = [batch, kv_seq_len, heads_kv, dim]
    out_shape = [batch, q_seq_len, heads_q, dim]

    tail_valid = kv_seq_len % block_n

    @T.prim_func
    def main(
        q: T.Tensor(q_shape, dtype),
        k: T.Tensor(kv_shape, dtype),
        v: T.Tensor(kv_shape, dtype),
        output: T.Tensor(out_shape, dtype),
        workspace_s: T.Tensor([block_num, ring_slots, d_chunks, block_m, block_n], accum_dtype),
        workspace_p: T.Tensor([block_num, ring_slots, block_m, block_n], dtype),
        workspace_o: T.Tensor([block_num, ring_slots, block_m, dim_align], accum_dtype),
        sm_scale: T.float32,
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # ---- L1 缓冲（Q 常驻 task 级；K/V 每 kv 块缓存；dim 维按 dim_align）----
            q_l1 = T.alloc_L1([block_m, dim_align], dtype)
            k_l1 = T.alloc_L1([block_n, dim_align], dtype)
            v_l1 = T.alloc_L1([block_n, dim_align], dtype)
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
            # TODO(tile-level, V-buffers)：running O / sumexp / m_i 驻留，
            #   alpha/sumexp_i UB ring，acc_s_ub / acc_s_half / broadcast 目标 /
            #   mask_col / causal 用的 col_idx + uint8 mask，均按 block_m//2 ×
            #   (block_n | dim_align) 申请（沿用 FA 清单 + causal 两项）。

            for local_idx in T.serial(tasks_per_core):
                task_idx = cid * tasks_per_core + local_idx
                # TIR 无 continue：用 if 守卫整个 task body
                if task_idx < block_num:
                    g_blk = task_idx % g_blks
                    s_blk = task_idx // g_blks % s_blks
                    nkv = task_idx // (g_blks * s_blks) % heads_kv
                    bz = task_idx // (g_blks * s_blks * heads_kv) % batch

                    s0 = s_blk * bs
                    hq0 = nkv * group + g_blk * bg
                    # causal 循环裁剪：本 task 可见的 kv 块数（静态循环 + 动态守卫）
                    if causal:
                        kv_end = T.min(
                            kv_loops, (s0 + bs - 1 + diff_s) // block_n + 1)
                    else:
                        kv_end = kv_loops

                    # ---- C0：Q 常驻 L1（per-g splice；越界读编译器零填充）----
                    with T.Scope("C"):
                        for gg in T.serial(bg):
                            if g_blk * bg + gg < group:
                                T.copy(q[bz, s0: s0 + bs, hq0 + gg, 0:dim],
                                       q_l1[gg * bs: (gg + 1) * bs, :])

                    # ---- V0：running 状态初始化 ----
                    with T.Scope("V"):
                        # TODO(tile-level, V0)：T.tile.fill(acc_o, 0) /
                        #   fill(sumexp, 0) / fill(m_i, MASK_NEG)
                        pass

                    # ---- 任务内 KV ring 流水 ----
                    for t in T.serial(kv_loops + prelaunch):
                        # ---------- 生产者：C1(t) / V1(t) ----------
                        if t < kv_end:
                            slot_prod = t % ring_slots

                            with T.Scope("C"):
                                # C1: S = Q @ K^T -> workspace_s[task_idx, slot_prod]
                                T.copy(k[bz, t * block_n: (t + 1) * block_n, nkv, 0:dim],
                                       k_l1)
                                # TODO(tile-level, C1)：
                                # - d_chunks 循环：q_l1/k_l1 的 cube_k 列切片进
                                #   lhs_l0/rhs_l0（K 走 transpose copy），
                                #   T.mma(..., init=True) 独立累加，
                                #   acc_s_l0c 落 ws_s[task_idx, slot_prod, chunk]
                                T.set_cross_flag("FIX", SIG_S_READY)   # C1(t) 完成 -> V1(t)

                            with T.Scope("V"):
                                # V1: scale/mask/online softmax -> workspace_p + meta
                                T.wait_cross_flag(SIG_S_READY)
                                # TODO(tile-level, V1)：
                                # - 读 ws_s 半 tile 各 d_chunk 切片，相加（chunks>1 时）
                                #   后乘 sm_scale
                                # - kv 尾块列 mask（mask_col 填 MASK_NEG，前 tail_valid
                                #   列填 0，逐行加）——tail_valid != 0 且 t == kv_end-1
                                #   注意：causal 时 kv_end 可能 < kv_loops，尾块条件按
                                #   t == kv_loops-1 判定（kv_end==kv_loops 时才重合）
                                # - causal 对角块 mask：(t+1)*block_n-1 > s0+diff_s 时，
                                #   createvecindex 列索引 + 逐行 compare(LE, bound_r) +
                                #   select 填 MASK_NEG；bound_r = s0 + (r%bs) + diff_s
                                # - online softmax：new_max/max/alpha/exp/reduce_sum，
                                #   P cast 写 ws_p；alpha/sumexp_i 落 UB ring
                                T.set_cross_flag("MTE3", SIG_P_READY)  # V1(t) 完成 -> C2(t)

                        # ---------- 消费者：C2(t-2) / V2(t-2) ----------
                        if t >= prelaunch:
                            now_k = t - prelaunch
                            if now_k < kv_end:
                                slot_cons = now_k % ring_slots

                                with T.Scope("C"):
                                    # C2: O = P @ V -> workspace_o[task_idx, slot_cons]
                                    T.wait_cross_flag(SIG_P_READY)
                                    T.copy(workspace_p[task_idx, slot_cons, :, :],
                                           acc_s_l1)
                                    T.copy(v[bz, now_k * block_n: (now_k + 1) * block_n,
                                             nkv, 0:dim], v_l1)
                                    # TODO(tile-level, C2)：
                                    # - d_chunks 循环：acc_s_l1→p_l0、v_l1 列切片→v_l0，
                                    #   T.mma(p_l0, v_l0, acc_o_l0c, init=True)，
                                    #   acc_o_l0c 落 ws_o[..., chunk*cube_k:...]
                                    T.set_cross_flag("FIX", SIG_O_READY)  # C2 -> V2

                                with T.Scope("V"):
                                    # V2: 读 meta（alpha/sumexp_i），merge O 贡献
                                    T.wait_cross_flag(SIG_O_READY)
                                    # TODO(tile-level, V2)：
                                    # - 读 ws_o 半 tile 进 acc_o_ub；
                                    #   acc_o = acc_o*alpha + acc_o_ub（broadcast mul+add）
                                    # - sumexp = sumexp*alpha + sumexp_i
                                    pass

                    # ---- 收尾：归一化并 per-g 写出（越界/g 尾块跳过）----
                    with T.Scope("V"):
                        # TODO(tile-level, V2-epilogue)：
                        # - acc_o /= sumexp（broadcast div），cast 进 acc_o_half
                        # - per-g：行区间 [g*bs, g*bs+bs) ∩ [vid*bm2, vid*bm2+bm2)
                        #   非空则 T.copy(acc_o_half 行切片[:, 0:dim],
                        #   output[bz, s0+s_lo:..., hq0+g, 0:dim])
                        pass

    return main
