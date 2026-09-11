// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
//
// GroupedMatmul 泛化 kernel 设备实现（gmm_run）。
// 与 Phase 4 产物 ascendc/grouped_matmul.cpp 的 gmm_run 零逻辑差异（ABI 搬运，
// GmmArgs → gmm::GmmTiling）。同步结构、buffer 偏移、copy/mma 调用逐字节保留
// （防退化规则 1；ISSUE-1/2/6/7 的修复全部含在此结构中）。

#ifndef GMM_KERNEL_H
#define GMM_KERNEL_H

#include "gmm_common.h"

namespace gmm {

template <typename T, bool HAS_BIAS, bool TW>
__aicore__ inline void gmm_run(GM_ADDR X_handle, GM_ADDR W_handle,
                               GM_ADDR Bias_handle, GM_ADDR Y_handle,
                               GM_ADDR Meta_handle, const GmmTiling& a) {
    using Cfg = TileCfg<T>;
    constexpr uint32_t block_M = Cfg::block_M;
    constexpr uint32_t block_N = Cfg::block_N;
    constexpr uint32_t block_K = Cfg::block_K;
    constexpr uint32_t K_L1 = Cfg::K_L1;
    constexpr uint32_t loop_kk = K_L1 / block_K;  // fp16:4 fp32:2

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> X;
    X.SetGlobalBuffer((__gm__ T*)X_handle);
    AscendC::GlobalTensor<T> W;
    W.SetGlobalBuffer((__gm__ T*)W_handle);
    AscendC::GlobalTensor<float> Bias;
    Bias.SetGlobalBuffer((__gm__ float*)Bias_handle);
    AscendC::GlobalTensor<T> Y;
    Y.SetGlobalBuffer((__gm__ T*)Y_handle);
    AscendC::GlobalTensor<int> Meta;
    Meta.SetGlobalBuffer((__gm__ int*)Meta_handle);

    // buffer 分配（生成码原样尺寸）
    AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
    pipe.InitBuffer(ascend_l0a, 65536);
    AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
    pipe.InitBuffer(ascend_l0b, 65536);
    AscendC::TBuf<AscendC::TPosition::A1> ascend_l1;
    pipe.InitBuffer(ascend_l1, 524032);
    AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c;
    pipe.InitBuffer(ascend_l0c, 131072);
    AscendC::TBuf<AscendC::TPosition::C2> ascend_bt;
    pipe.InitBuffer(ascend_bt, 1024);
    pipe.Destroy();

    auto cid = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        cid = cid / 2;  // 生成码原样（AIV 分支；本 kernel AIV 空，不执行）
    }

    // buffer 视图（生成码原样偏移，按 dtype 分支）
    auto bias_L1 = ascend_l1.GetWithOffset<float>(block_N, 0);
    auto bias_bt = ascend_bt.GetWithOffset<float>(block_N, 0);
    auto C_L0 = ascend_l0c.GetWithOffset<float>(Cfg::C_L0_ELEMS, 0);
    auto A_L1 = ascend_l1.GetWithOffset<T>(Cfg::S1 * Cfg::A_L1_STRIDE,
                                           Cfg::A_L1_BYTE_OFF);
    auto B_L1 = ascend_l1.GetWithOffset<T>(Cfg::S1 * Cfg::B_L1_STRIDE,
                                           Cfg::B_L1_BYTE_OFF);
    auto A_L0 = ascend_l0a.GetWithOffset<T>(Cfg::S2 * Cfg::A_L0_STRIDE, 0);
    auto B_L0 = ascend_l0b.GetWithOffset<T>(Cfg::S2 * Cfg::B_L0_STRIDE, 0);

    auto vid = AscendC::GetSubBlockIdx();

    if ASCEND_IS_AIC {
        // init_flag（手写预取流水，蓝本 matmul 模式，ISSUE-9）：L1 双缓冲
        // mte1↔mte2（编号 0/1），L0 双缓冲 m↔mte1（编号 0/1），fix m↔fix（编号 0）。
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        // 外层任务遍历（语义形式，已对照验证等价于生成码的折叠常量）：
        //   task = i*grid + cid; bx = task/n_num; by = task%n_num
        int32_t iters = (a.total_blocks + a.grid - 1) / a.grid;
        for (int32_t i = 0; i < iters; ++i) {
            int32_t task = i * a.grid + cid;
            if (task < a.total_blocks) {
                int32_t bx = task / a.n_num;
                int32_t by = task % a.n_num;
                int32_t g = Meta.GetValue(bx * 3);
                int32_t m_start = Meta.GetValue(bx * 3 + 1);
                int32_t valid_m = Meta.GetValue(bx * 3 + 2);

                // M 尾块钳位（ISSUE-2：组内有效行 valid_m，防跨组读写竞态）
                int32_t m_clamp = ((valid_m + m_start) <= a.M)
                                      ? valid_m
                                      : ((m_start < a.M) ? (a.M - m_start) : 0);
                // N 尾块钳位（ISSUE-8：host 不再 pad N，kernel 内处理 N 尾块）。
                int32_t n_clamp = ((by * (int32_t)block_N + (int32_t)block_N) <= a.N)
                                      ? (int32_t)block_N
                                      : (a.N - by * (int32_t)block_N);

                if constexpr (HAS_BIAS) {
                    // bias 装载（K 循环前一次）：GM→L1(MTE2)→BT(MTE1)→首个 mma(M)。
                    // 独立 flag 编号 2（mte2→mte1 / mte1→m），与 L0/L1 的 0/1 不冲突
                    // （ISSUE-1：手写 flag 下 bias 的 MTE1→M 依赖显式覆盖）。
                    // N 尾块 realTailN=n_clamp（ISSUE-8，编译器修复 commit e50437fc）。
                    tl::ascend::copy_gm_to_l1_linear<float, 1, block_N>(
                        bias_L1[0], Bias[g * a.N + by * block_N], a.N,
                        ((g <= a.E - 1) ? 1 : 0), n_clamp, true);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(2);
                    tl::ascend::copy_l1_to_bt<float, float>(bias_bt[0],
                                                            bias_L1[0], block_N);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2);
                }

                // 首个 K chunk（k=0）装载到 A_L1[0]/B_L1[0]
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                {
                    // k=0 的 K 尾块钳位：loop_k==1 时 k_clamp=K，否则 K_L1
                    int32_t k0_clamp = (a.loop_k <= 1) ? (a.K) : (int32_t)K_L1;
                    tl::ascend::copy_gm_to_l1<T, block_M, K_L1>(
                        A_L1[0], X[m_start * a.K], a.K, m_clamp, k0_clamp, true);
                    if constexpr (TW) {
                        tl::ascend::copy_gm_to_l1<T, block_N, K_L1>(
                            B_L1[0], W[g * a.N * a.K + by * (block_N * a.K)],
                            a.K, n_clamp, k0_clamp, true);
                    } else {
                        tl::ascend::copy_gm_to_l1<T, K_L1, block_N>(
                            B_L1[0], W[g * a.K * a.N + by * block_N],
                            a.N, k0_clamp, n_clamp, true);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);

                for (int32_t k = 0; k < a.loop_k; ++k) {
                    // 软件预取 k+1 chunk 到 A_L1[(k+1)%S1]/B_L1[(k+1)%S1]，
                    // 与下方 kk 循环的 mma(k) overlap（MTE2 提前一拍，ISSUE-9）。
                    if (k < a.loop_k - 1) {
                        int32_t kn = k + 1;
                        // k+1 chunk 的 K 尾块钳位
                        int32_t kn_clamp =
                            (kn <= a.loop_k - 2) ? (int32_t)K_L1 : (a.K - kn * K_L1);
                        int32_t slot = kn % Cfg::S1;
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
                        tl::ascend::copy_gm_to_l1<T, block_M, K_L1>(
                            A_L1[slot * Cfg::A_L1_STRIDE],
                            X[m_start * a.K + kn * K_L1], a.K, m_clamp, kn_clamp,
                            true);
                        if constexpr (TW) {
                            tl::ascend::copy_gm_to_l1<T, block_N, K_L1>(
                                B_L1[slot * Cfg::B_L1_STRIDE],
                                W[g * a.N * a.K + by * (block_N * a.K) + kn * K_L1],
                                a.K, n_clamp, kn_clamp, true);
                        } else {
                            tl::ascend::copy_gm_to_l1<T, K_L1, block_N>(
                                B_L1[slot * Cfg::B_L1_STRIDE],
                                W[g * a.K * a.N + kn * (K_L1 * a.N) + by * block_N],
                                a.N, kn_clamp, n_clamp, true);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
                    }

                    int32_t kslot = k % Cfg::S1;
                    for (int32_t kk = 0; kk < loop_kk; ++kk) {
                        if (kk == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kslot);
                        }
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kk % Cfg::S2);
                        // L1 → L0（block_K 乒乓 kk%S2）
                        tl::ascend::copy_l1_to_l0a<T, block_M, K_L1, false>(
                            A_L0[(kk % Cfg::S2) * Cfg::A_L0_STRIDE],
                            A_L1[kslot * Cfg::A_L1_STRIDE +
                                 kk * Cfg::A_L1_KK_STEP],
                            block_M, block_K);
                        if constexpr (TW) {
                            tl::ascend::copy_l1_to_l0b<T, block_N, K_L1, true>(
                                B_L0[(kk % Cfg::S2) * Cfg::B_L0_STRIDE],
                                B_L1[kslot * Cfg::B_L1_STRIDE +
                                     kk * Cfg::B_L1_KK_STEP_TW],
                                block_K, block_N);
                        } else {
                            tl::ascend::copy_l1_to_l0b<T, K_L1, block_N, false>(
                                B_L0[(kk % Cfg::S2) * Cfg::B_L0_STRIDE],
                                B_L1[kslot * Cfg::B_L1_STRIDE +
                                     kk * Cfg::B_L1_KK_STEP_NT],
                                block_K, block_N);
                        }
                        if (kk == loop_kk - 1) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kslot);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kk % Cfg::S2);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kk % Cfg::S2);
                        if constexpr (HAS_BIAS) {
                            if ((k == 0) && (kk == 0)) {
                                // 首个 mma_bias 前 wait bias 的 MTE1→M（编号 2）
                                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(2);
                                tl::ascend::mma_bias<T, float, block_M, block_N>(
                                    A_L0[0], B_L0[0], C_L0[0], bias_bt[0],
                                    (bool)0, block_K);
                            } else {
                                tl::ascend::mma<T, float, block_M, block_N>(
                                    A_L0[(kk % Cfg::S2) * Cfg::A_L0_STRIDE],
                                    B_L0[(kk % Cfg::S2) * Cfg::B_L0_STRIDE],
                                    C_L0[0], (bool)0, block_K);
                            }
                        } else {
                            tl::ascend::mma<T, float, block_M, block_N>(
                                A_L0[(kk % Cfg::S2) * Cfg::A_L0_STRIDE],
                                B_L0[(kk % Cfg::S2) * Cfg::B_L0_STRIDE],
                                C_L0[0], ((k == 0) && (kk == 0)), block_K);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kk % Cfg::S2);
                    }
                }

                // L0C → GM（fixpipe 钳位 M 尾块 valid_m + N 尾块 n_clamp，ISSUE-8）
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
                tl::ascend::copy_l0c_to_gm<float, T, layout::RowMajor, block_M,
                                           block_N, 0>(
                    Y[m_start * a.N + by * block_N], C_L0[0], a.N, m_clamp,
                    n_clamp, 0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
            }
        }
        // clear_flag
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    }
}

}  // namespace gmm

#endif  // GMM_KERNEL_H
