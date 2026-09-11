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
// GroupedMatmul launch 声明（host gcc 侧包含；实现随 kernel 同 TU 由 bisheng 编译）。

#ifndef GROUPED_MATMUL_LAUNCH_H
#define GROUPED_MATMUL_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void *
#endif

#define GMM_DECL_LAUNCH(SUFFIX)                                               \
    extern "C" void launch_##SUFFIX(                                          \
        uint32_t grid, void *stream, GM_ADDR X_handle, GM_ADDR W_handle,      \
        GM_ADDR Bias_handle, GM_ADDR Y_handle, GM_ADDR Meta_handle,           \
        GM_ADDR tiling);

GMM_DECL_LAUNCH(fp16_bias_nt)
GMM_DECL_LAUNCH(fp16_bias_t)
GMM_DECL_LAUNCH(fp16_nobias_nt)
GMM_DECL_LAUNCH(fp16_nobias_t)
GMM_DECL_LAUNCH(bf16_bias_nt)
GMM_DECL_LAUNCH(bf16_bias_t)
GMM_DECL_LAUNCH(bf16_nobias_nt)
GMM_DECL_LAUNCH(bf16_nobias_t)
GMM_DECL_LAUNCH(fp32_bias_nt)
GMM_DECL_LAUNCH(fp32_bias_t)
GMM_DECL_LAUNCH(fp32_nobias_nt)
GMM_DECL_LAUNCH(fp32_nobias_t)

#undef GMM_DECL_LAUNCH

// bias_cast 纯 AIV kernel（ISSUE-10 方案 1）：GM(T bias)→UB→Cast<T→float>→GM(fp32)。
// 仅 half/bf16；shape 走左值标量 total（零 H2D）。
#define GMM_DECL_BIAS_CAST(SUFFIX)                                            \
    extern "C" void launch_bias_cast_##SUFFIX(                                \
        uint32_t grid, void *stream, GM_ADDR src_handle, GM_ADDR dst_handle,  \
        int32_t total);

GMM_DECL_BIAS_CAST(fp16)
GMM_DECL_BIAS_CAST(bf16)

#undef GMM_DECL_BIAS_CAST

#endif  // GROUPED_MATMUL_LAUNCH_H
