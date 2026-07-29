/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV2D_BASE_TILING_H
#define CONV2D_BASE_TILING_H

#include <cstdint>

// Conv2dBase on arch20 (910B) - shared host/device tile constants.
//
// GEMM view of the convolution (after host-side layout normalization):
//   A [M, K] : im2col of x, M = Ho*Wo (per image), K = Ci*kH*kW
//   B [K, N] : weight permuted to K-major ND, N = Co
//   C [M, N] : fp32 acc, equals NHWC output rows
// Host normalizes: x -> NHWC + spatial pad + Ci pad to 16, so the kernel only
// sees a pad-free NHWC problem with Ci % 16 == 0 and Co % 16 == 0.
//
// Tiling delivery: all tiling fields are few scalars -> passed directly as
// kernel launch arguments (no GM tiling buffer), the simpler reliable option.

constexpr int32_t CONV_BASE_M = 128;   // m basic block (Ho*Wo rows per L0A tile)
constexpr int32_t CONV_BASE_N = 128;   // n basic block (Co per L0B tile)
constexpr int32_t CONV_BASE_K = 64;    // k basic block (per L0 load)
constexpr int32_t CONV_C0 = 16;        // fractal inner dim (fp16/bf16)

// L1 budget for A1 + B1 panels. 910B L1 = 512KB total. P6: A1/B1 are
// depth-2 queues for cross-block/panel prefetch, so the budget accounts
// for two slots per panel and the B panel uses b1Cols <= baseN columns.
constexpr int64_t CONV_L1_BUDGET_BYTES = 448 * 1024;

// B panel column budget: full baseN only when the nEff=baseN fallback can
// trigger (Co < 64); otherwise the NZ panel only ever holds Co columns.
inline int32_t ConvB1Cols(int32_t coPad)
{
    if (coPad < 64) {
        return CONV_BASE_N;
    }
    return coPad < CONV_BASE_N ? coPad : CONV_BASE_N;
}

// K-dim fullload decision: kAL1 = largest multiple of CONV_BASE_K such that
// 2 * (baseM + b1Cols) * kAL1 * dtypeSize <= L1 budget (depth-2 queues),
// capped to ceil(K, baseK).
inline int32_t ConvComputeKAL1(int32_t K, int32_t dtypeSize, int32_t b1Cols)
{
    int64_t perK = static_cast<int64_t>(CONV_BASE_M + b1Cols) * dtypeSize * 2;
    int64_t maxK = CONV_L1_BUDGET_BYTES / perK;
    int32_t kCap = static_cast<int32_t>(maxK / CONV_BASE_K) * CONV_BASE_K;
    if (kCap < CONV_BASE_K) {
        kCap = CONV_BASE_K;
    }
    int32_t kFull = ((K + CONV_BASE_K - 1) / CONV_BASE_K) * CONV_BASE_K;
    return kCap < kFull ? kCap : kFull;
}

inline int32_t ConvRoundUp16(int32_t v)
{
    return (v + CONV_C0 - 1) / CONV_C0 * CONV_C0;
}

#endif // CONV2D_BASE_TILING_H
