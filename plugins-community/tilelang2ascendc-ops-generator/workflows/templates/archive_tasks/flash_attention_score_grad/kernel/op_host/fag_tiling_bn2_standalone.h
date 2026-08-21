/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fag_tiling_bn2_standalone.h
 * \brief FlashAttentionScoreGrad S1s2Bn2（小 shape 路径）独立 host tiling。
 *
 * 从生产版 op_host/arch22/flash_attention_score_grad_tiling_s1s2_bn2.cpp 裁剪移植，
 * 仅保留：BNSD 布局、keep_prob=1.0（无 dropout）、无 pse、sparse_mode=0
 * （可选 bool atten_mask）、无 rope、非确定性关闭、非 L1Custom、非 NZ 输出
 * （d ∈ {72,80,88,96} 的 NZ 场景由路由层排除，走 S1s2Bn2gs1s2 泛化路径）。
 * tiling 结构体复用 kernel 侧 op_kernel/flash_attention_score_grad_tiling.h。
 */

#ifndef FAG_TILING_BN2_STANDALONE_H
#define FAG_TILING_BN2_STANDALONE_H

#include <cstdint>

#include "../op_kernel/flash_attention_score_grad_tiling.h"
#include "fag_tiling_standalone.h" // FagDtype

namespace fag_standalone {

// S1s2Bn2 模板路由判定（对齐生产 IsCapable 在归档固定配置下的化简）。
// aivNum 为平台 AIV 核数（aicoreParams_.numBlocks）。返回 true 表示可走 bn2 路径。
bool FagBn2IsCapable(int64_t b, int64_t n2, int64_t g, int64_t s1, int64_t s2, int64_t d, int64_t value_d,
                     FagDtype dtype);

// 计算 S1s2Bn2 模板路径的 tiling，填充 tilingData（调用前需已 reset）。
// attenMask* 仅在 hasAttenMask=true 时有效（4D mask 的 (dim0, dim1, s1, s2)）。
// 返回用户 workspace 字节数（不含系统预留区）；blockDimOut 输出 AIC block 数。
// 失败返回 0。
uint64_t FagBn2DoOpTiling(int64_t b, int64_t n2, int64_t g, int64_t s1, int64_t s2, int64_t d, int64_t value_d,
                          FagDtype dtype, float scaleValue, bool hasAttenMask, int64_t attenMaskDim0,
                          int64_t attenMaskDim1, int64_t attenMaskS1, int64_t attenMaskS2,
                          FlashAttentionScoreGradTilingDataS1s2Bn2 *tilingData, uint32_t *blockDimOut);

} // namespace fag_standalone

#endif // FAG_TILING_BN2_STANDALONE_H
