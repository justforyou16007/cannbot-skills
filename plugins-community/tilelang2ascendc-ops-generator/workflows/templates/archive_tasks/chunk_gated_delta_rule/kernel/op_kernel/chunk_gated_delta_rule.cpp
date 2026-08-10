/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule.cpp
 * \brief
 */

#include "chunk_gated_delta_rule.h"
#include "chunk_gated_delta_rule_tiling_data.h"

using namespace AscendC;
using namespace matmul;
using namespace ChunkGatedDeltaRule;

// Direct-launch (<<<>>>) tiling copy: the host passes the full tiling blob via a
// GM tiling tensor; copy its bytes into a local (stack/locally-allocated) POD
// struct, then drive the (zero-modified) CGDR kernel with it.
__aicore__ inline void CopyTilingToLocal(ChunkGatedDeltaRuleTilingData *tiling, GM_ADDR tilingGM)
{
    auto *dst = reinterpret_cast<int64_t *>(tiling);
    auto *src = reinterpret_cast<__gm__ int64_t *>(tilingGM);
    // The tiling struct is int64-heavy; copy as raw words to keep the ABI
    // independent of host/device packed struct layout.
    for (uint64_t i = 0; i < sizeof(ChunkGatedDeltaRuleTilingData) / sizeof(int64_t); ++i) {
        dst[i] = src[i];
    }
}

extern "C" __global__ __aicore__ void chunk_gated_delta_rule(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta,
                                                             GM_ADDR initialState, GM_ADDR seqlens, GM_ADDR gOptional,
                                                             GM_ADDR out, GM_ADDR finalState, GM_ADDR workspaceGM,
                                                             GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    ChunkGatedDeltaRuleTilingData tilingData;
    CopyTilingToLocal(&tilingData, tilingGM);

    __gm__ uint8_t *user = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);

    CGDR<bfloat16_t, float> op(&pipe, &tilingData);
    CGDRInitParams initParams{query, key, value, beta, initialState, seqlens, gOptional, out, finalState};
    op.Init(initParams, user);
    op.Process();
}
