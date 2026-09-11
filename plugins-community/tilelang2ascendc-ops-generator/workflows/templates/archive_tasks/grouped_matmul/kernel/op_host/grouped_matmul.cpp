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
// GroupedMatmul op_host — tiling 计算（含 N padding + block_metadata 预计算）+
// 按 (dtype, has_bias, tw) 分发 launch。设备实现：op_kernel/gmm_kernel.h。
//
// 语义见 model.py docstring（cann-bench level3/grouped_matmul）：
//   y[rows_g] = x[rows_g] @ weight[g] (+ bias[g])，空组跳过。
// 返回 [M, N]（原始 N；kernel 内部按 padded N 计算，host 切片回）。

#include <cstdint>
#include <cstring>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tiling/platform/platform_ascendc.h"

#include "op_kernel/grouped_matmul_launch.h"

namespace ascend_kernel {

// 与 op_kernel/gmm_common.h 的 gmm::GmmTiling 逐字段一致
struct GmmTilingHost {
    int32_t M, N, K, E, n_num, loop_k, total_m_blocks, total_blocks, grid, pad;
};

// per-dtype tile 配置（与 kernel TileCfg 一致）
struct TileCfgHost {
    int32_t block_M, block_N, block_K, K_L1;
};

static TileCfgHost tile_cfg(at::ScalarType dt) {
    if (dt == at::kFloat) {
        return {128, 128, 64, 128};
    }
    return {128, 256, 64, 256};  // fp16 / bf16
}

// launch 函数指针类型
using LaunchFn = void (*)(uint32_t, void *, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                          GM_ADDR, GM_ADDR);
// bias_cast launch 函数指针类型（GM src, GM dst, 左值标量 total）
using BiasCastFn = void (*)(uint32_t, void *, GM_ADDR, GM_ADDR, int32_t);

static LaunchFn select_launch(at::ScalarType dt, bool has_bias, bool tw) {
    if (dt == at::kHalf) {
        return has_bias ? (tw ? launch_fp16_bias_t : launch_fp16_bias_nt)
                        : (tw ? launch_fp16_nobias_t : launch_fp16_nobias_nt);
    }
    if (dt == at::kBFloat16) {
        return has_bias ? (tw ? launch_bf16_bias_t : launch_bf16_bias_nt)
                        : (tw ? launch_bf16_nobias_t : launch_bf16_nobias_nt);
    }
    if (dt == at::kFloat) {
        return has_bias ? (tw ? launch_fp32_bias_t : launch_fp32_bias_nt)
                        : (tw ? launch_fp32_nobias_t : launch_fp32_nobias_nt);
    }
    TORCH_CHECK(false, "unsupported dtype for grouped_matmul: ", dt);
}

// bias_cast launch 选择（仅 half/bf16；fp32 透传不 launch）
static BiasCastFn select_bias_cast(at::ScalarType dt) {
    if (dt == at::kHalf) return launch_bias_cast_fp16;
    if (dt == at::kBFloat16) return launch_bias_cast_bf16;
    TORCH_CHECK(false, "bias_cast only for half/bf16, got: ", dt);
}

at::Tensor grouped_matmul(const at::Tensor &x, const at::Tensor &weight,
                          const c10::optional<at::Tensor> &bias_opt,
                          const at::Tensor &group_list,
                          int64_t split_item, bool transpose_weight) {
    const bool tw = transpose_weight;
    const int64_t M = x.sizes()[0], K = x.sizes()[1];
    const int64_t E = weight.sizes()[0];
    const int64_t N = tw ? weight.sizes()[1] : weight.sizes()[2];
    const auto dt = x.scalar_type();
    const bool has_bias = bias_opt.has_value() && bias_opt->numel() > 0;

    TileCfgHost tc = tile_cfg(dt);
    // N 尾块在 kernel 内处理（ISSUE-8：消除 host N-padding）。weight 按原始 N
    // 读 GM（B 分形装载 n_clamp 钳位 + 尾列零填充，fixpipe 写出 N 钳位），
    // 不再 host pad 到 block_N。n_num = ceildiv(N, block_N)。
    const int64_t n_num = (N + tc.block_N - 1) / tc.block_N;
    const int64_t loop_k = (K + tc.K_L1 - 1) / tc.K_L1;

    // weight 不 pad（原始 N），直接透传。
    const at::Tensor &w = weight;

    // bias：主 kernel 的 bias 累加走 BT（fp32）。
    //  - fp32 bias：直接透传（不 launch cast）。
    //  - half/bf16 bias：launch bias_cast 纯 AIV kernel 转 fp32 到 workspace
    //    （GM→UB→Cast<T→float>→GM；shape 走左值标量零 H2D，输出 at::empty 零 H2D，
    //    Q1 裁决）。主 kernel 零改动（仍读 fp32 bias 进 BT）。
    //  - 无 bias：传 x data 指针 dummy（HAS_BIAS=0 时 kernel 不读 Bias，方案 2，
    //    删 at::zeros 避免一次 D2H/H2D + 全零写）。
    // bias 行 stride = 原始 N（与 kernel 的 a.N 一致；weight/bias 的 N 处理同源）。
    auto f32 = x.options().dtype(at::kFloat);
    at::Tensor bias_ws;  // half/bf16 时的 fp32 workspace（保持生命周期到 launch 后）
    const void *bias_ptr = nullptr;
    if (has_bias && bias_opt->scalar_type() == at::kFloat) {
        bias_ptr = bias_opt->storage().data();
    } else if (has_bias) {
        // half/bf16 → fp32 workspace（at::empty，内容全部由 bias_cast 写覆盖）
        bias_ws = at::empty({E, N}, f32);
        bias_ptr = bias_ws.storage().data();
    } else {
        bias_ptr = x.storage().data();  // dummy（不读）
    }

    // group_list → ends（cumsum）
    at::Tensor gl = group_list.scalar_type() == at::kLong
                        ? group_list
                        : group_list.to(at::kLong);
    auto gl_cpu = gl.to(at::kCPU);
    std::vector<int64_t> ends(gl_cpu.data_ptr<int64_t>(),
                              gl_cpu.data_ptr<int64_t>() + E);

    // block_metadata 预计算（host 展开 group_list；空组 m_g=0 不产 block）
    std::vector<int32_t> meta_rows;
    {
        int64_t s = 0;
        for (int64_t g = 0; g < E; ++g) {
            int64_t e = ends[g];
            int64_t m_g = e - s;
            if (m_g > 0) {
                for (int64_t m_start = s; m_start < e;
                     m_start += tc.block_M) {
                    meta_rows.push_back((int32_t)g);
                    meta_rows.push_back((int32_t)m_start);
                    meta_rows.push_back(
                        (int32_t)std::min<int64_t>(tc.block_M, e - m_start));
                }
            }
            s = e;
        }
    }
    const int64_t total_m_blocks = (int64_t)meta_rows.size() / 3;
    const int64_t total_blocks = total_m_blocks * n_num;

    // grid = min(total_blocks, cube 核数)（运行期查，不写死）
    auto *ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(ascendcPlatform != nullptr,
                "PlatformAscendCManager::GetInstance() is null");
    const int64_t cube_core_num = ascendcPlatform->GetCoreNumAic();
    const int64_t grid = std::min<int64_t>(total_blocks, cube_core_num);

    // 输出按原始 N 分配（kernel fixpipe 钳位写 N 尾块），无切片。
    // at::empty（方案 3）：group_list 完整覆盖 [0,M)，每个非空组的每个 block
    // 都被 kernel fixpipe 完整写出（空组不占行），无未初始化读取。
    at::Tensor y = at::empty({M, N}, x.options());

    // meta + tiling 合并单 buffer 单次同步 H2D（方案 4/6，Q2 裁决）：
    //   布局 [tiling 头 (sizeof(GmmTilingHost)=40B) | pad 至 32B 对齐 | meta 体]。
    //   kernel 的 Meta_handle = base + META_OFF（kernel 零改动，仅 GM 基址偏移）；
    //   tiling = base。host buffer 用函数内 at::Tensor（RAII），
    //   .to(kPrivateUse1, non_blocking=false) 同步 H2D，删 static keepalive。
    //   目标：每 case repeat 阶段 aclrtMemcpy = 1（meta+tiling 合并这一次）。
    GmmTilingHost t{(int32_t)M,  (int32_t)N, (int32_t)K, (int32_t)E,
                    (int32_t)n_num, (int32_t)loop_k, (int32_t)total_m_blocks,
                    (int32_t)total_blocks, (int32_t)grid, 0};
    constexpr int64_t HEADER_BYTES = (int64_t)sizeof(GmmTilingHost);
    constexpr int64_t META_OFF = (HEADER_BYTES + 31) / 32 * 32;  // 40 → 64
    const int64_t meta_bytes =
        std::max<int64_t>(total_m_blocks, 1) * 3 * (int64_t)sizeof(int32_t);
    const int64_t mt_bytes = META_OFF + meta_bytes;

    at::Tensor mt_cpu = at::empty({mt_bytes}, at::device(at::kCPU).dtype(at::kByte));
    auto *mt_base = reinterpret_cast<uint8_t *>(mt_cpu.data_ptr());
    std::memcpy(mt_base, &t, HEADER_BYTES);
    if (total_m_blocks > 0) {
        std::memcpy(mt_base + META_OFF, meta_rows.data(), meta_bytes);
    }
    // 同步 H2D（non_blocking=false）：host buffer RAII，无需 keepalive
    at::Tensor mt_npu = mt_cpu.to(at::kPrivateUse1, /*non_blocking=*/false);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    // half/bf16 bias：先 launch bias_cast 转 fp32 到 workspace（与主 kernel 同 stream）
    if (has_bias && bias_opt->scalar_type() != at::kFloat) {
        BiasCastFn bias_cast = select_bias_cast(dt);
        // AIV_ONLY：grid = 2×cube 物理核数（起满全部 vector 核）
        const int64_t aiv_grid = cube_core_num * 2;
        int32_t bias_total = (int32_t)(E * N);
        void *bsrc = const_cast<void *>(bias_opt->storage().data());
        void *bdst = const_cast<void *>(bias_ws.storage().data());
        bias_cast((uint32_t)aiv_grid, aclStream, bsrc, bdst, bias_total);
    }

    LaunchFn launch = select_launch(dt, has_bias, tw);
    void *mt_npu_base = const_cast<void *>(mt_npu.storage().data());
    void *tilingPtr = mt_npu_base;                                   // tiling 头 @ base
    void *metaPtr = static_cast<uint8_t *>(mt_npu_base) + META_OFF;  // meta 体 @ base+64
    launch((uint32_t)grid, aclStream,
           const_cast<void *>(x.storage().data()),
           const_cast<void *>(w.storage().data()),
           const_cast<void *>(bias_ptr),
           const_cast<void *>(y.storage().data()),
           metaPtr,
           tilingPtr);

    return y;
}

}  // namespace ascend_kernel
