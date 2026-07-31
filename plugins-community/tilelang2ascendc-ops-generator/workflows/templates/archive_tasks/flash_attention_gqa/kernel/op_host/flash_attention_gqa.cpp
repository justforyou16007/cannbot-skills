/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// flash_attention_gqa op_host — input validation + doOpTiling + workspace/output/tiling
// tensor allocation + EXEC_KERNEL_CMD dispatch. The kernel ABI is
// (GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling),
// launched via EXEC_KERNEL_CMD with the tiling blob passed as a GM tensor.

#include <cstdint>

#include <torch/extension.h>
#include <torch/library.h>

#include "torch_kernel_helper.h"
#include "utils/flash_attention_tiling.h"

#include "aclrtlaunch_flash_attention_gqa_bf16.h"
#include "aclrtlaunch_flash_attention_gqa_fp16.h"

namespace ascend_kernel {

at::Tensor flash_attention_gqa(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v)
{
    TORCH_CHECK(q.dim() == 4, "q must be 4D [batch, qHeads, qSeqLen, dim]");
    TORCH_CHECK(k.dim() == 4, "k must be 4D [batch, kvHeads, kvSeqLen, dim]");
    TORCH_CHECK(v.dim() == 4, "v must be 4D [batch, kvHeads, kvSeqLen, dim]");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have same dtype");
    TORCH_CHECK(q.scalar_type() == v.scalar_type(), "q and v must have same dtype");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(q.device().type() == at::kPrivateUse1, "q must be on NPU/PrivateUse1 device");
    TORCH_CHECK(q.sizes()[0] == k.sizes()[0] && q.sizes()[0] == v.sizes()[0],
                "batch size of q/k/v must match");
    TORCH_CHECK(k.sizes()[1] == v.sizes()[1], "kv heads of k and v must match");
    TORCH_CHECK(k.sizes()[2] == v.sizes()[2], "kv sequence length of k and v must match");
    TORCH_CHECK(q.sizes()[3] == k.sizes()[3] && q.sizes()[3] == v.sizes()[3],
                "last dim of q/k/v must match");

    // dtype dispatch: fp16 -> flash_attention_gqa_fp16, bf16 -> flash_attention_gqa_bf16
    const char *kernelName;
    uint32_t qElementSize = 2;  // half and bfloat16 are both 16-bit
    if (q.scalar_type() == at::kHalf) {
        kernelName = "flash_attention_gqa_fp16";
    } else if (q.scalar_type() == at::kBFloat16) {
        kernelName = "flash_attention_gqa_bf16";
    } else {
        TORCH_CHECK(false, "flash_attention_gqa unsupported dtype, expected float16 or bfloat16.");
        kernelName = nullptr;
    }

    const uint32_t batch    = q.sizes()[0];
    const uint32_t nq       = q.sizes()[1];
    const uint32_t nkv      = k.sizes()[1];
    const uint32_t qSeqLen  = q.sizes()[2];
    const uint32_t kvSeqLen = k.sizes()[2];
    const uint32_t dim      = q.sizes()[3];
    TORCH_CHECK(nkv > 0, "kv heads must be positive");
    TORCH_CHECK(nq % nkv == 0, "q heads must be a multiple of kv heads");

    // Output uses the original qSeqLen (no host-side padding).
    at::Tensor output = at::zeros({static_cast<int64_t>(batch), static_cast<int64_t>(nq),
                                   static_cast<int64_t>(qSeqLen), static_cast<int64_t>(dim)},
                                  q.options());

    // Tiling: fill FlashAttentionTiling via doOpTiling and copy to device (GM tensor).
    at::Tensor tilingCpu = at::empty({static_cast<int64_t>(sizeof(FlashAttentionTiling))},
                                     at::device(at::kCPU).dtype(at::kByte));
    auto *tilingPtr = reinterpret_cast<FlashAttentionTiling *>(tilingCpu.data_ptr());
    uint64_t totalWsBytes = doOpTiling(
        batch, nq, nkv, qSeqLen, kvSeqLen, dim, qElementSize, tilingPtr);
    TORCH_CHECK(totalWsBytes == tilingPtr->totalWsBytes,
                "tiling total workspace size mismatch");
    at::Tensor tilingNpu = tilingCpu.to(at::kPrivateUse1);

    // Workspace sized from tiling totalWsBytes.
    at::Tensor workspace = at::zeros({static_cast<int64_t>(tilingPtr->totalWsBytes)},
                                     at::device(at::kPrivateUse1).dtype(at::kByte));

    const uint32_t blockDim = tilingPtr->usedCoreNum;

    if (q.scalar_type() == at::kHalf) {
        EXEC_KERNEL_CMD(flash_attention_gqa_fp16, blockDim,
                        q, k, v, output, workspace, tilingNpu);
    } else {
        EXEC_KERNEL_CMD(flash_attention_gqa_bf16, blockDim,
                        q, k, v, output, workspace, tilingNpu);
    }

    return output;
}

} // namespace ascend_kernel
