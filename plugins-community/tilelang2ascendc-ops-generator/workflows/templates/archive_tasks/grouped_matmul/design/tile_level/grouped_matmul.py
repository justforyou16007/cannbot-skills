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

# Tile-level TileLang design for GroupedMatmul（slice-M），per-shape 版。
#
# deliverer 路线 Phase 3 产物。骨架对齐 design 蓝本
# archive_tasks/matmul/design/tile_level/matmul.py 的数据流（L1/L0 双缓冲 +
# T.mma 直调），但同步改为 AUTO_SYNC（见 ascendc/GENERALIZE_DEBUG_LOG.md ISSUE-1：
# 手写 flag 模式下 BT bias-init mma 的 MTE1→M 依赖手工覆盖不全，bias 失效；
# AUTO_SYNC 由编译器自动插入核内 pipe 同步，bias 正确）。
#
# 在 matmul 蓝本之上补的 grouped 能力：
#   - block_metadata 查表（grouped_gemm 模式）：每个 block 查得 [g, m_start, valid_m]
#   - 空组：host 展开 metadata 时 m_g=0 不产 block，天然跳过
#   - bias：BT bias-init mma（T.mma(..., bias=bias_bt)），bias 全程 fp32，
#     GM → L1(linear RowMajor) → BT → 首个 mma 预置 L0C（官方 conv bias 路径）
#   - transpose_weight：weight [E,N,K] 时 B_L1 按 [block_N, K_L1] 读，
#     L1→L0B 用 T.copy(..., transpose=True) 转置装载
#   - 尾块：K 尾零填充（pad_value=0，对累加无贡献）+ M/N 尾 fixpipe 钳位
#     （matmul 蓝本已实证 idiom；pad 清零防 0×inf=NaN 污染）
#
# ── metadata 布局 ─────────────────────────────────────────────────────────
#   block_metadata[total_m_blocks, 3] int32 = [g, m_start_global, valid_m]
#   total_m_blocks = Σ_g ceildiv(m_g, block_M)（空组 m_g=0 → 0 块）
#   kernel 内 bx = task // n_num 索引 metadata，by = task % n_num 决定 n_start。
#
# ── 关键尺寸（同 matmul 蓝本）──────────────────────────────────────────────
#   fp16/bf16: block_M=128, block_N=256, block_K=64, K_L1=256, S1=2, S2=2
#   fp32:      block_M=128, block_N=128, block_K=64, K_L1=128, S1=2, S2=2

from dataclasses import dataclass

import tilelang
from tilelang import language as T
from tilelang.intrinsics.ascend_layout import AscendLayout
from tilelang.utils.target import get_core_num

# 手写预取流水（P1 性能迭代，ISSUE-9）：K 循环用蓝本 matmul 的软件预取
# （`if k<loop_k-1: 预取 A/B(k+1)` 到 A_L1[(k+1)%S1]，MTE2(k+1) 与 kk 循环
# mma(k) overlap），替代 AUTO_SYNC 的顺序流水。实测（msprof）：case9 174→117us
# （-33%，追平 aclnn 122us，MTE2 0.50→0.935 带宽瓶颈）、case12 193→70us
# （-64%，MMA 0.35→0.863 compute bound，超 aclnn 73us）。AUTO_SYNC 顺序流水
# MTE2/MMA 串行（cube_util 97.5% 但 MMA 仅 35%）是大 shape 2.5x 差距的根因。
# MEMORY_PLANNING：片上 buffer 静态规划。不开 AUTO_SYNC（手写 set/wait flag）。
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CORE_NUM = get_core_num("cube")  # 编译期读目标芯片


@dataclass(frozen=True)  # frozen: tilelang.jit 需要对 cfg 取 hash 做编译缓存键
class GroupedMatmulConfig:
    m: int                 # 总 M（= group_list[-1]）
    n: int                 # N
    k: int                 # K
    e: int                 # expert 数
    dtype: str = "float16"
    has_bias: bool = False
    transpose_weight: bool = False
    total_m_blocks: int = 0  # host 预计算 Σ_g ceildiv(m_g, block_M)


def _tile_cfg(dtype):
    """per-dtype tile 配置（含义见 PERF_DESIGN.md §5）。"""
    if dtype == "float32":
        return 128, 128, 64, 128   # block_M, block_N, block_K, K_L1
    return 128, 256, 64, 256


@tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS, target='ascendc')
def grouped_matmul_fwd(cfg: GroupedMatmulConfig):
    M, N, K, E = cfg.m, cfg.n, cfg.k, cfg.e
    dtype = cfg.dtype
    accum_dtype = "float"
    has_bias = cfg.has_bias
    tw = cfg.transpose_weight

    block_M, block_N, block_K, K_L1 = _tile_cfg(dtype)
    S1, S2 = 2, 2

    n_num = (N + block_N - 1) // block_N
    total_m_blocks = cfg.total_m_blocks
    total_blocks = total_m_blocks * n_num
    loop_k = (K + K_L1 - 1) // K_L1
    loop_kk = K_L1 // block_K

    # weight 的 GM 形状：tw=false → [E, K, N]；tw=true → [E, N, K]
    Wd0, Wd1 = (N, K) if tw else (K, N)

    @T.prim_func
    def main(
        X: T.Tensor([M, K], dtype),
        W: T.Tensor([E, Wd0, Wd1], dtype),
        Bias: T.Tensor([E, N], "float"),
        Y: T.Tensor([M, N], dtype),
        Meta: T.Tensor([total_m_blocks, 3], "int32"),
    ):
        # grid = min(total_blocks, CORE_NUM)：block 数不足物理核时按实际 block 数
        # launch，避免空转核（ascendc-tiling-design 反模式：block_num < 物理核仍
        # 按物理核 launch）。
        grid = min(total_blocks, CORE_NUM)
        with T.Kernel(grid, is_npu=True) as (cid, vid):
            A_L1 = T.alloc_L1([S1, block_M, K_L1], dtype)
            # B_L1 shape 依赖 tw（编译期 python bool），单次无条件分配：
            # tw=false → [S1, K_L1, block_N]；tw=true → [S1, block_N, K_L1]
            B_L1 = T.alloc_L1(
                [S1, block_N, K_L1] if tw else [S1, K_L1, block_N], dtype)

            A_L0 = T.alloc_L0A([S2, block_M, block_K], dtype)
            B_L0 = T.alloc_L0B([S2, block_K, block_N], dtype)
            C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)

            # bias buffer 无条件分配（保持 buffer 在函数作用域内可解析；
            # 无 bias 时仅占用 1KB BT + block_N*4B L1，不使用）。bias 全程 fp32。
            bias_L1 = T.alloc_L1([1, block_N], "float")
            bias_bt = T.alloc_BT([1, block_N], "float")
            # bias_L1 要线性布局: 默认 zN 分形会把 [1,N] 按 16 列块打散,
            # copy_l1_to_bt 线性读会拿到错位的 bias
            T.annotate_layout({
                bias_L1: T.Layout([1, block_N], lambda *args: args,
                                  layout_tag=AscendLayout.kRowMajor.value)
            })

            with T.Scope("C"):
                # init_flag（蓝本 matmul 模式）：L1 双缓冲 mte1↔mte2（编号 0/1），
                # L0 双缓冲 m↔mte1（编号 0/1），fix m↔fix（编号 0）。
                T.set_flag("mte1", "mte2", 0)
                T.set_flag("mte1", "mte2", 1)
                T.set_flag("m", "mte1", 0)
                T.set_flag("m", "mte1", 1)
                T.set_flag("fix", "m", 0)
                for i in T.serial((total_blocks + CORE_NUM - 1) // CORE_NUM):
                    task = i * CORE_NUM + cid
                    if task < total_blocks:
                        bx = task // n_num
                        by = task % n_num
                        # block_metadata 查表：g=组号，m_start=组内全局行起点，
                        # valid_m=该 m 砖的组内有效行数（尾块 < block_M）。
                        g = Meta[bx, 0]
                        m_start = Meta[bx, 1]
                        valid_m = Meta[bx, 2]

                        # bias 装载（K 循环前一次）：bias[g, by*block_N:...]
                        # GM→L1(MTE2)→BT(MTE1)→首个 mma(M)。独立 flag 编号 2
                        # （mte2→mte1 / mte1→m），与 L0/L1 双缓冲的 0/1 不冲突——
                        # 解决 ISSUE-1 手写 flag 下 bias 失效（MTE1→M 依赖显式覆盖）。
                        if has_bias:
                            T.copy(Bias[g, by * block_N:(by + 1) * block_N],
                                   bias_L1[0, :])
                            T.set_flag("mte2", "mte1", 2)
                            T.wait_flag("mte2", "mte1", 2)
                            T.copy(bias_L1, bias_bt)
                            T.set_flag("mte1", "m", 2)

                        # 首个 K chunk 装载（k=0 到 A_L1[0]/B_L1[0]）
                        T.wait_flag("mte1", "mte2", 0)
                        T.copy(X[m_start:m_start + valid_m, 0:K_L1],
                               A_L1[0, :, :])
                        if tw:
                            T.copy(W[g, by * block_N:(by + 1) * block_N, 0:K_L1],
                                   B_L1[0, :, :])
                        else:
                            T.copy(W[g, 0:K_L1, by * block_N:(by + 1) * block_N],
                                   B_L1[0, :, :])
                        T.set_flag("mte2", "mte1", 0)
                        T.wait_flag("fix", "m", 0)

                        for k in T.serial(loop_k):
                            # 软件预取：k+1 chunk 搬到 A_L1[(k+1)%S1]/B_L1[(k+1)%S1]，
                            # 与下方 kk 循环的 mma(k) overlap（MTE2 提前一拍）。
                            if k < loop_k - 1:
                                T.wait_flag("mte1", "mte2", (k + 1) % S1)
                                T.copy(X[m_start:m_start + valid_m,
                                         (k + 1) * K_L1:(k + 2) * K_L1],
                                       A_L1[(k + 1) % S1, :, :])
                                if tw:
                                    T.copy(W[g, by * block_N:(by + 1) * block_N,
                                             (k + 1) * K_L1:(k + 2) * K_L1],
                                           B_L1[(k + 1) % S1, :, :])
                                else:
                                    T.copy(W[g, (k + 1) * K_L1:(k + 2) * K_L1,
                                             by * block_N:(by + 1) * block_N],
                                           B_L1[(k + 1) % S1, :, :])
                                T.set_flag("mte2", "mte1", (k + 1) % S1)

                            for kk in T.serial(loop_kk):
                                if kk == 0:
                                    T.wait_flag("mte2", "mte1", k % S1)
                                T.wait_flag("m", "mte1", kk % S2)
                                # L1 → L0（block_K 乒乓 kk%S2）
                                T.copy(A_L1[k % S1, :, kk * block_K:(kk + 1) * block_K],
                                       A_L0[kk % S2, :, :])
                                if tw:
                                    T.copy(B_L1[k % S1, :, kk * block_K:(kk + 1) * block_K],
                                           B_L0[kk % S2, :, :], transpose=True)
                                else:
                                    T.copy(B_L1[k % S1, kk * block_K:(kk + 1) * block_K, :],
                                           B_L0[kk % S2, :, :])
                                if kk == loop_kk - 1:
                                    T.set_flag("mte1", "mte2", k % S1)
                                T.set_flag("mte1", "m", kk % S2)
                                T.wait_flag("mte1", "m", kk % S2)
                                if has_bias:
                                    # 首个 mma 带 BT bias 预置 L0C，后续累加。
                                    # 首个 mma_bias 前 wait bias 的 MTE1→M（编号 2）。
                                    if T.And(k == 0, kk == 0):
                                        T.wait_flag("mte1", "m", 2)
                                        T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :],
                                              C_L0, bias=bias_bt)
                                    else:
                                        T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :],
                                              C_L0, init=False)
                                else:
                                    T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :],
                                          C_L0, init=T.And(k == 0, kk == 0))
                                T.set_flag("m", "mte1", kk % S2)

                        # L0C → GM（fixpipe 钳位 M/N 尾块，T.copy 自动钳位）。
                        # M 维用 valid_m 钳位（ISSUE-2：防跨组写竞态）；N 维尾块
                        # 由 T.copy 自动钳位（ISSUE-8：kernel 内处理，host 不 pad）。
                        T.set_flag("m", "fix", 0)
                        T.wait_flag("m", "fix", 0)
                        T.copy(C_L0, Y[m_start:m_start + valid_m,
                                       by * block_N:(by + 1) * block_N])
                        T.set_flag("fix", "m", 0)
                # clear_flag
                T.wait_flag("mte1", "mte2", 0)
                T.wait_flag("mte1", "mte2", 1)
                T.wait_flag("m", "mte1", 0)
                T.wait_flag("m", "mte1", 1)
                T.wait_flag("fix", "m", 0)

    return main
