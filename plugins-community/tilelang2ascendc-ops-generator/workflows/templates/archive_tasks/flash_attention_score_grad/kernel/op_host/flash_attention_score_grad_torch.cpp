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
 * \file flash_attention_score_grad_torch.cpp
 * \brief torch 侧包装：输入校验 -> doOpTiling -> tiling/workspace 分配 -> aclrtlaunch 启动。
 * kernel ABI: (query, key, value, dy, atten_mask, softmax_max, softmax_sum, attention_in,
 *              dq, dk, dv, workspace, tiling)，按 dtype x hasAttenMask 选择 6 个入口之一。
 */

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>

#include "torch_kernel_helper.h"
#include "fag_tiling_standalone.h"
#include "fag_tiling_bn2_standalone.h"

// 6 个入口（3 dtype x 2 mask）全部包含；二进制体积调试期的 FAG_ENTRY_COUNT 条件编译已移除
#include "aclrtlaunch_flash_attention_score_grad_fp16.h"
#include "aclrtlaunch_flash_attention_score_grad_fp16_mask.h"
#include "aclrtlaunch_flash_attention_score_grad_bf16.h"
#include "aclrtlaunch_flash_attention_score_grad_bf16_mask.h"
#include "aclrtlaunch_flash_attention_score_grad_fp32.h"
#include "aclrtlaunch_flash_attention_score_grad_fp32_mask.h"
// S1s2Bn2（小 shape 路由）入口
#include "aclrtlaunch_flash_attention_score_grad_bn2_fp16.h"
#include "aclrtlaunch_flash_attention_score_grad_bn2_fp16_mask.h"
#include "aclrtlaunch_flash_attention_score_grad_bn2_bf16.h"
#include "aclrtlaunch_flash_attention_score_grad_bn2_bf16_mask.h"
#include "aclrtlaunch_flash_attention_score_grad_bn2_fp32.h"
#include "aclrtlaunch_flash_attention_score_grad_bn2_fp32_mask.h"

namespace ascend_kernel {
namespace {

// 独立构建无 opp 系统 workspace 管理，host 需为 kernel 额外预留系统区
// （kernel 内 SetSysWorkspaceForce + GetUserWorkspace 会跳过该区域），
// dav-c220 (Ascend910B2) 的 RESERVED_WORKSPACE 为 16MB。
constexpr uint64_t SYS_WORKSPACE_RESERVED = 16ULL * 1024 * 1024;

// tiling 结果缓存：同一 (shape, dtype, mask, scale) 的重复调用直接复用，
// 避免每次调用重算 MatmulApiTiling 等 host tiling（与生产算子的 tiling 缓存行为对齐）。
struct TilingCacheEntry {
    std::vector<uint8_t> blob;
    uint32_t blockDim = 0;
    uint64_t userWsBytes = 0;
    bool useBn2 = false; // true=S1s2Bn2 小 shape 路径，false=S1s2Bn2gs1s2 泛化路径
};

struct TilingKey {
    int64_t b, nkv, g, s1, s2, d, valueD;
    int dtype, hasMask, maskDim0, maskDim1;
    uint32_t scaleBits;

    bool operator==(const TilingKey &o) const
    {
        return b == o.b && nkv == o.nkv && g == o.g && s1 == o.s1 && s2 == o.s2 && d == o.d &&
               valueD == o.valueD && dtype == o.dtype && hasMask == o.hasMask && maskDim0 == o.maskDim0 &&
               maskDim1 == o.maskDim1 && scaleBits == o.scaleBits;
    }
};

struct TilingKeyHash {
    size_t operator()(const TilingKey &k) const
    {
        uint64_t h = 1469598103934665603ULL;
        auto mix = [&h](uint64_t v) {
            h ^= v;
            h *= 1099511628211ULL;
        };
        mix(k.b); mix(k.nkv); mix(k.g); mix(k.s1); mix(k.s2); mix(k.d); mix(k.valueD);
        mix(k.dtype); mix(k.hasMask); mix(k.maskDim0); mix(k.maskDim1); mix(k.scaleBits);
        return static_cast<size_t>(h);
    }
};

std::unordered_map<TilingKey, TilingCacheEntry, TilingKeyHash> g_tilingCache;
std::mutex g_tilingCacheMutex;

} // namespace

std::vector<at::Tensor> flash_attention_score_grad(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &dy,
    const at::Tensor &softmax_max, const at::Tensor &softmax_sum, const at::Tensor &attention_in,
    const c10::optional<at::Tensor> &atten_mask, double scale)
{
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && dy.dim() == 4,
                "q/k/v/dy must be 4D [B, N, S, D] (BNSD)");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() &&
                    q.scalar_type() == dy.scalar_type(),
                "q/k/v/dy must have the same dtype");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16 ||
                    q.scalar_type() == at::kFloat,
                "unsupported dtype, expected float16/bfloat16/float32");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && dy.is_contiguous(),
                "q/k/v/dy must be contiguous");
    TORCH_CHECK(q.device().type() == at::kPrivateUse1, "q must be on NPU/PrivateUse1 device");
    TORCH_CHECK(q.sizes()[0] == k.sizes()[0] && k.sizes()[0] == v.sizes()[0],
                "batch size of q/k/v must match");
    TORCH_CHECK(k.sizes()[1] == v.sizes()[1] && k.sizes()[2] == v.sizes()[2],
                "kv heads / kv seq len of k and v must match");
    TORCH_CHECK(k.sizes()[3] == q.sizes()[3], "head dim of q and k must match");
    TORCH_CHECK(q.sizes()[1] % k.sizes()[1] == 0, "q heads must be a multiple of kv heads");
    TORCH_CHECK(dy.sizes()[3] == v.sizes()[3], "head dim of dy and v must match");

    const int64_t b = q.sizes()[0];
    const int64_t nq = q.sizes()[1];
    const int64_t nkv = k.sizes()[1];
    const int64_t s1 = q.sizes()[2];
    const int64_t s2 = k.sizes()[2];
    const int64_t d = q.sizes()[3];
    const int64_t valueD = v.sizes()[3];
    const int64_t g = nq / nkv;

    const bool hasMask = atten_mask.has_value() && atten_mask->defined() && atten_mask->numel() > 0;
    at::Tensor maskNpu;
    int64_t maskDim0 = 0, maskDim1 = 0, maskS1 = 0, maskS2 = 0;
    if (hasMask) {
        maskNpu = atten_mask->to(at::kBool).contiguous();
        TORCH_CHECK(maskNpu.dim() == 4, "atten_mask must be 4D [B/1, N1/1, S1, S2]");
        maskDim0 = maskNpu.sizes()[0];
        maskDim1 = maskNpu.sizes()[1];
        maskS1 = maskNpu.sizes()[2];
        maskS2 = maskNpu.sizes()[3];
    }

    at::Tensor softmaxMaxF = softmax_max.scalar_type() == at::kFloat ? softmax_max : softmax_max.to(at::kFloat);
    at::Tensor softmaxSumF = softmax_sum.scalar_type() == at::kFloat ? softmax_sum : softmax_sum.to(at::kFloat);
    TORCH_CHECK(softmaxMaxF.is_contiguous() && softmaxSumF.is_contiguous(),
                "softmax_max/softmax_sum must be contiguous");

    fag_standalone::FagDtype dtype;
    if (q.scalar_type() == at::kHalf) {
        dtype = fag_standalone::FagDtype::FP16;
    } else if (q.scalar_type() == at::kBFloat16) {
        dtype = fag_standalone::FagDtype::BF16;
    } else {
        dtype = fag_standalone::FagDtype::FP32;
    }

    // host 侧 tiling（带缓存：重复 shape 直接复用已计算的 tiling blob）
    float scaleF = static_cast<float>(scale);
    TilingKey key{b,      nkv,      g,        s1,        s2,         d, valueD, static_cast<int>(dtype),
                  hasMask ? 1 : 0, maskDim0, maskDim1, 0};
    uint32_t scaleU32;
    static_assert(sizeof(scaleU32) == sizeof(scaleF));
    std::memcpy(&scaleU32, &scaleF, sizeof(scaleF));
    key.scaleBits = scaleU32;

    uint32_t blockDim = 0;
    uint64_t userWsBytes = 0;
    bool useBn2 = false;
    at::Tensor tilingNpu;
    {
        std::lock_guard<std::mutex> lock(g_tilingCacheMutex);
        auto it = g_tilingCache.find(key);
        if (it == g_tilingCache.end()) {
            TilingCacheEntry entry;
            // 小 shape 模板路由（对齐生产 IsCapable 条件）：命中走 S1s2Bn2
            entry.useBn2 = fag_standalone::FagBn2IsCapable(b, nkv, g, s1, s2, d, valueD, dtype);
            if (entry.useBn2) {
                entry.blob.resize(sizeof(FlashAttentionScoreGradTilingDataS1s2Bn2));
                auto *ptr = reinterpret_cast<FlashAttentionScoreGradTilingDataS1s2Bn2 *>(entry.blob.data());
                ptr->reset();
                entry.userWsBytes = fag_standalone::FagBn2DoOpTiling(
                    b, nkv, g, s1, s2, d, valueD, dtype, scaleF, hasMask, maskDim0, maskDim1, maskS1, maskS2, ptr,
                    &entry.blockDim);
            } else {
                entry.blob.resize(sizeof(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2));
                auto *ptr = reinterpret_cast<FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *>(entry.blob.data());
                ptr->reset();
                entry.userWsBytes = fag_standalone::FagS1s2DoOpTiling(
                    b, nkv, g, s1, s2, d, valueD, dtype, scaleF, hasMask, maskDim0, maskDim1, maskS1, maskS2, ptr,
                    &entry.blockDim);
            }
            TORCH_CHECK(entry.userWsBytes > 0, "flash_attention_score_grad doOpTiling failed");
            TORCH_CHECK(entry.blockDim > 0, "flash_attention_score_grad invalid blockDim");
            it = g_tilingCache.emplace(key, std::move(entry)).first;
        }
        blockDim = it->second.blockDim;
        userWsBytes = it->second.userWsBytes;
        useBn2 = it->second.useBn2;
        at::Tensor tilingCpu = at::from_blob(
            const_cast<uint8_t *>(it->second.blob.data()),
            {static_cast<int64_t>(it->second.blob.size())}, at::device(at::kCPU).dtype(at::kByte));
        tilingNpu = TorchNpuHelper::CopyTensorHostToDevice(tilingCpu);
    }

    // workspace 内容每次启动前由 kernel 内 Pre/clearWorkspace 初始化，无需 host 清零
    at::Tensor workspace = at::empty({static_cast<int64_t>(userWsBytes + SYS_WORKSPACE_RESERVED)},
                                     at::device(at::kPrivateUse1).dtype(at::kByte));

    // dq/dk/dv 全部位置由 kernel 覆写（fp16/bf16：Pre 清零 workspace + 主算累加 + Post 按
    // qSize/kvSize/vSize 全量 cast 写回；fp32：Pre 直接清零 dvGm、主算累加、Post 写 dq/dk），
    // host 侧无需 at::zeros（zeros 会额外产生 3 个 ZerosLike kernel，case0 实测占 device 时间 11.3%）
    at::Tensor dq = at::empty(q.sizes(), q.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());

    // 无 mask 入口的 atten_mask 形参传 q 的指针占位（kernel 内不会解引用）
    const at::Tensor &maskArg = hasMask ? maskNpu : q;


#define FAG_LAUNCH(PREFIX)                                                                                 \
    do {                                                                                                   \
        if (dtype == fag_standalone::FagDtype::FP16) {                                                     \
            if (hasMask) {                                                                                 \
                EXEC_KERNEL_CMD(PREFIX##_fp16_mask, blockDim, q, k, v, dy, maskArg, softmaxMaxF,           \
                                softmaxSumF, attention_in, dq, dk, dv, workspace, tilingNpu);              \
            } else {                                                                                       \
                EXEC_KERNEL_CMD(PREFIX##_fp16, blockDim, q, k, v, dy, maskArg, softmaxMaxF, softmaxSumF,   \
                                attention_in, dq, dk, dv, workspace, tilingNpu);                           \
            }                                                                                              \
        } else if (dtype == fag_standalone::FagDtype::BF16) {                                              \
            if (hasMask) {                                                                                 \
                EXEC_KERNEL_CMD(PREFIX##_bf16_mask, blockDim, q, k, v, dy, maskArg, softmaxMaxF,           \
                                softmaxSumF, attention_in, dq, dk, dv, workspace, tilingNpu);              \
            } else {                                                                                       \
                EXEC_KERNEL_CMD(PREFIX##_bf16, blockDim, q, k, v, dy, maskArg, softmaxMaxF, softmaxSumF,   \
                                attention_in, dq, dk, dv, workspace, tilingNpu);                           \
            }                                                                                              \
        } else {                                                                                           \
            if (hasMask) {                                                                                 \
                EXEC_KERNEL_CMD(PREFIX##_fp32_mask, blockDim, q, k, v, dy, maskArg, softmaxMaxF,           \
                                softmaxSumF, attention_in, dq, dk, dv, workspace, tilingNpu);              \
            } else {                                                                                       \
                EXEC_KERNEL_CMD(PREFIX##_fp32, blockDim, q, k, v, dy, maskArg, softmaxMaxF, softmaxSumF,   \
                                attention_in, dq, dk, dv, workspace, tilingNpu);                           \
            }                                                                                              \
        }                                                                                                  \
    } while (0)

    if (useBn2) {
        FAG_LAUNCH(flash_attention_score_grad_bn2);
    } else {
        FAG_LAUNCH(flash_attention_score_grad);
    }
#undef FAG_LAUNCH

    // 调试：导出 user workspace 到文件供数值比对（FAG_BN2_DEBUG_WS=1 时）
    if (std::getenv("FAG_BN2_DEBUG_WS") != nullptr) {
        at::Tensor wsUser = workspace.narrow(0, static_cast<int64_t>(SYS_WORKSPACE_RESERVED),
                                             static_cast<int64_t>(userWsBytes));
        at::Tensor wsCpu = wsUser.cpu();  // 隐式同步
        FILE *fp = std::fopen("/tmp/fag_bn2_ws.bin", "wb");
        if (fp != nullptr) {
            std::fwrite(wsCpu.data_ptr(), 1, userWsBytes, fp);
            std::fclose(fp);
        }
        printf("[bn2 debug] workspace dumped: %llu bytes -> /tmp/fag_bn2_ws.bin\n",
               (unsigned long long)userWsBytes);
    }

    return {dq, dk, dv};
}

} // namespace ascend_kernel
