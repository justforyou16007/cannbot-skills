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

# Tile-level TileLang design for matmul（C[M,N] = A[M,K] @ B[K,N]），per-shape 版。
#
# deliverer 路线 Phase 3 产物。结构对齐 tilelang-ascend/examples/gemm/
# example_gemm_intrinsic.py（用户裁决：用 intrinsic 手写流水版，性能更高）：
#   - 固定 20 核 + use_swizzle 任务映射 + task_idx 守卫
#   - L1 双 buffer（S1=2 slots，K_L1 chunk 预取与 mma 重叠）
#   - L0 双 buffer（S2=2 slots，block_K 乒乓）
#   - 全手写 flag（mte1/mte2、m/mte1、fix/m），不开 AUTO_SYNC
#   - T.mma 直调（不走 gemm_v0——gemm_v0 的 kL0Size 默认 128 对 fp32 需手动
#     调小，见 probe_kl0size.py 对照实验；intrinsic 版自己控制 L0 装载粒度）
#
# 在示例之上补的泛化能力：
#   - M/N/K 任意尾块（部分拷贝由编译器钳位 + 零填充；K 尾零填充对累加无贡献；
#     C 写出的 M/N 尾由 fixpipe 钳位——已在 gemm_v0 版设计上实证同 idiom 正确）
#   - fp16/bf16/fp32：fp32 用更小的 block_N/K_L1（L0/L1 容量减半）
#
# ── 关键尺寸 ────────────────────────────────────────────────────────────────
#   fp16/bf16: block_M=128, block_N=256, block_K=64, K_L1=256, S1=2, S2=2
#     （A_L1 2×128×256×2=128KB + B_L1 2×256×256×2=256KB = 384KB ≤ 512KB；
#      A_L0 2×128×64×2=32KB，B_L0 2×64×256×2=64KB 恰好；C_L0 128×256×4=128KB 恰好）
#   fp32:      block_M=128, block_N=128, block_K=64, K_L1=128, S1=2, S2=2
#     （A_L1+B_L1 = 2×128×128×4×2=256KB；A_L0/B_L0 各 64KB 恰好；C_L0 64KB）

from dataclasses import dataclass

import tilelang
from tilelang import language as T
from tilelang.utils.target import get_core_num


# 无 pass_configs：intrinsic 版全部手写同步（示例原样）
CORE_NUM = get_core_num("cube")  # 编译期读目标芯片


@dataclass(frozen=True)   # frozen: tilelang.jit 需要对 cfg 取 hash 做编译缓存键
class MatmulConfig:
    m: int
    n: int
    k: int
    dtype: str = "float16"


@tilelang.jit(out_idx=[2], target='ascendc')
def matmul_fwd(cfg: MatmulConfig):
    M, N, K = cfg.m, cfg.n, cfg.k
    dtype = cfg.dtype
    accum_dtype = "float"

    if dtype == "float32":
        block_M, block_N, block_K, K_L1 = 128, 128, 64, 128
    else:
        block_M, block_N, block_K, K_L1 = 128, 256, 64, 256
    S1, S2 = 2, 2

    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N
    loop_k = (K + K_L1 - 1) // K_L1
    loop_kk = K_L1 // block_K

    @T.macro
    def init_flag():
        T.set_flag("mte1", "mte2", 0)
        T.set_flag("mte1", "mte2", 1)
        T.set_flag("m", "mte1", 0)
        T.set_flag("m", "mte1", 1)
        T.set_flag("fix", "m", 0)

    @T.macro
    def clear_flag():
        T.wait_flag("mte1", "mte2", 0)
        T.wait_flag("mte1", "mte2", 1)
        T.wait_flag("m", "mte1", 0)
        T.wait_flag("m", "mte1", 1)
        T.wait_flag("fix", "m", 0)

    @T.prim_func
    def main(
        A: T.Tensor([M, K], dtype),
        B: T.Tensor([K, N], dtype),
        C: T.Tensor([M, N], dtype),
    ):
        with T.Kernel(CORE_NUM, is_npu=True) as (cid, vid):
            A_L1 = T.alloc_L1([S1, block_M, K_L1], dtype)
            B_L1 = T.alloc_L1([S1, K_L1, block_N], dtype)

            A_L0 = T.alloc_L0A([S2, block_M, block_K], dtype)
            B_L0 = T.alloc_L0B([S2, block_K, block_N], dtype)
            C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)

            with T.Scope("C"):
                init_flag()

                for i in T.serial((m_num * n_num + CORE_NUM - 1) // CORE_NUM):
                    # 传 padding 后的 M/N：use_swizzle 要求整除且任务数按
                    # m//block_m*n//block_n 向下取整——pad 后恰为 m_num*n_num
                    task = T.use_swizzle(i * CORE_NUM + cid, m_num * block_M,
                                         n_num * block_N, K,
                                         block_M, block_N, off=3)
                    if task < m_num * n_num:
                        bx = task // n_num
                        by = task % n_num

                        T.wait_flag("mte1", "mte2", 0)
                        T.copy(A[bx * block_M: (bx + 1) * block_M, 0:K_L1],
                               A_L1[0, :, :])
                        T.copy(B[0:K_L1, by * block_N: (by + 1) * block_N],
                               B_L1[0, :, :])
                        T.set_flag("mte2", "mte1", 0)
                        T.wait_flag("fix", "m", 0)
                        for k in T.serial(loop_k):
                            if k < loop_k - 1:
                                T.wait_flag("mte1", "mte2", (k + 1) % S1)
                                T.copy(A[bx * block_M: (bx + 1) * block_M,
                                         (k + 1) * K_L1: (k + 2) * K_L1],
                                       A_L1[(k + 1) % S1, :, :])
                                T.copy(B[(k + 1) * K_L1: (k + 2) * K_L1,
                                         by * block_N: (by + 1) * block_N],
                                       B_L1[(k + 1) % S1, :, :])
                                T.set_flag("mte2", "mte1", (k + 1) % S1)

                            for kk in T.serial(loop_kk):
                                if kk == 0:
                                    T.wait_flag("mte2", "mte1", k % S1)
                                T.wait_flag("m", "mte1", kk % S2)
                                T.copy(A_L1[k % S1, :, kk * block_K: (kk + 1) * block_K],
                                       A_L0[kk % S2, :, :])
                                T.copy(B_L1[k % S1, kk * block_K: (kk + 1) * block_K, :],
                                       B_L0[kk % S2, :, :])
                                if kk == loop_kk - 1:
                                    T.set_flag("mte1", "mte2", k % S1)
                                T.set_flag("mte1", "m", kk % S2)
                                T.wait_flag("mte1", "m", kk % S2)

                                T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :],
                                      C_L0, init=T.And(k == 0, kk == 0))

                                T.set_flag("m", "mte1", kk % S2)

                        T.set_flag("m", "fix", 0)
                        T.wait_flag("m", "fix", 0)
                        T.copy(C_L0, C[bx * block_M: (bx + 1) * block_M,
                                       by * block_N: (by + 1) * block_N])
                        T.set_flag("fix", "m", 0)

                clear_flag()

    return main
