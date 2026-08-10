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

// Chunked Gated Delta Rule.
// query (T, Nk, Dk) bf16, key (T, Nk, Dk) bf16, value (T, Nv, Dv) bf16,
// beta (T, Nv) bf16, initial_state (B, Nv, Dv, Dk) bf16/fp32,
// actual_seq_lengths (B,) int32, g (T, Nv) fp32 (zeros when absent), scale attr.
// Returns {out (T, Nv, Dv) bf16, final_state (B, Nv, Dv, Dk)}.
std::vector<at::Tensor> chunk_gated_delta_rule(const at::Tensor &query,
                                               const at::Tensor &key,
                                               const at::Tensor &value,
                                               const at::Tensor &beta,
                                               const at::Tensor &initialState,
                                               const at::Tensor &actualSeqLengths,
                                               const at::Tensor &g,
                                               double scale);

} // namespace ascend_kernel

#endif // OPS_H
