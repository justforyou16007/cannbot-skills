/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

// FlashAttentionScoreGrad（BNSD 布局，keep_prob=1.0，无 pse，sparse_mode=0）。
// 返回 (dq, dk, dv)。atten_mask 为可选 bool 张量（True 表示屏蔽），scale 为 softmax 缩放。
std::vector<at::Tensor> flash_attention_score_grad(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &dy,
    const at::Tensor &softmax_max, const at::Tensor &softmax_sum, const at::Tensor &attention_in,
    const c10::optional<at::Tensor> &atten_mask, double scale);

} // namespace ascend_kernel

#endif // OPS_H
