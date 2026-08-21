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
 * \file fag_tiling_standalone.h
 * \brief FlashAttentionScoreGrad 独立 host tiling（archive re-host）。
 *
 * 从生产版 op_host/flash_attention_score_grad_tiling_s1s2_bn2gs1s2.cpp（原 op_host/arch22/）
 * 裁剪移植，仅保留：BNSD 布局、keep_prob=1.0（无 dropout）、无 pse、
 * sparse_mode=0（可选 atten_mask，如 causal 上三角掩码）、无 rope、非确定性关闭。
 * tiling 结构体直接复用 kernel 侧 op_kernel/flash_attention_score_grad_tiling.h 的定义。
 */

#ifndef FAG_TILING_STANDALONE_H
#define FAG_TILING_STANDALONE_H

#include <cstdint>

#include "../op_kernel/flash_attention_score_grad_tiling.h"

namespace fag_standalone {

enum class FagDtype { FP16 = 0, BF16 = 1, FP32 = 2 };

// 计算 S1s2Bn2gs1s2 通用模板路径的 tiling，填充 tilingData（调用前需已 reset）。
// attenMask* 仅在 hasAttenMask=true 时有效，传入 4D mask 的 (dim0, dim1, s1, s2)。
// 返回用户 workspace 字节数（不含系统预留区）；blockDimOut 输出 AIC block 数。
// 失败返回 0。
uint64_t FagS1s2DoOpTiling(int64_t b, int64_t n2, int64_t g, int64_t s1, int64_t s2, int64_t d, int64_t value_d,
                           FagDtype dtype, float scaleValue, bool hasAttenMask, int64_t attenMaskDim0,
                           int64_t attenMaskDim1, int64_t attenMaskS1, int64_t attenMaskS2,
                           FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *tilingData, uint32_t *blockDimOut);

} // namespace fag_standalone

#endif // FAG_TILING_STANDALONE_H
