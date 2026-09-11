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
// GroupedMatmul 泛化 kernel 公共头：编译期 tile 配置 + GM tiling 结构体。
// 设备实现见 gmm_kernel.h；入口与 launch 包装见 grouped_matmul_kernels.cpp。
// 变换清单与容量核算见 ascendc/GENERALIZE_REPORT.md（与 Phase 4 产物零逻辑差异）。

#ifndef GMM_COMMON_H
#define GMM_COMMON_H

#include "tl_templates/ascend/common.h"

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif
#ifndef BACKEND_HYBM
#define BACKEND_HYBM
#endif

using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

namespace gmm {

// GM tiling 结构体（8B 对齐；host 侧 op_host/grouped_matmul.cpp 的同构
// GmmTilingHost 必须逐字段一致）。shape 维度全运行期。
struct GmmTiling {
    int32_t M;               // 总 M（= group_list[-1]）
    int32_t N;               // N（原始 N，不 pad；kernel 内 n_clamp 处理 N 尾块）
    int32_t K;               // K
    int32_t E;               // expert 数
    int32_t n_num;           // ceildiv(N, block_N)（N 尾块由 kernel 钳位）
    int32_t loop_k;          // ceildiv(K, K_L1)
    int32_t total_m_blocks;  // Σ_g ceildiv(m_g, block_M)
    int32_t total_blocks;    // total_m_blocks * n_num
    int32_t grid;            // launch 核数 = min(total_blocks, 物理核数)
    int32_t pad;             // 8B 对齐填充
};

// per-dtype tile 配置（含义见 design/PERF_DESIGN.md §5）。
// 非 fp32（fp16/bf16，C0=16）：block_N=256, K_L1=256
// fp32（C0=8）：block_N=128, K_L1=128（L0/L1 容量减半）
template <typename T>
struct TileCfg {
    static constexpr uint32_t block_M = 128;
    static constexpr uint32_t block_N = (sizeof(T) == 4) ? 128 : 256;
    static constexpr uint32_t block_K = 64;
    static constexpr uint32_t K_L1 = (sizeof(T) == 4) ? 128 : 256;
    static constexpr uint32_t S1 = 2;  // L1 双缓冲槽
    static constexpr uint32_t S2 = 2;  // L0 双缓冲槽
    // 派生槽距（元素数）
    static constexpr uint32_t A_L1_STRIDE = block_M * K_L1;
    static constexpr uint32_t B_L1_STRIDE = block_N * K_L1;
    static constexpr uint32_t A_L0_STRIDE = block_M * block_K;
    static constexpr uint32_t B_L0_STRIDE = block_K * block_N;
    static constexpr uint32_t C_L0_ELEMS = block_M * block_N;
    static constexpr uint32_t BIAS_L1_ELEMS = block_N;
    static constexpr uint32_t L1_TOTAL = 524032;
    // A_L1 字节偏移（生成码原样）：fp32=512，非 fp32=1024
    static constexpr uint32_t A_L1_BYTE_OFF = (sizeof(T) == 4) ? 512 : 1024;
    static constexpr uint32_t B_L1_BYTE_OFF =
        A_L1_BYTE_OFF + S1 * A_L1_STRIDE * sizeof(T);
    // C0（分形单元列数）：fp16/bf16=16, fp32=8
    static constexpr uint32_t C0 = (sizeof(T) == 4) ? 8 : 16;
    // B_L1 内 kk 步进（L1 分形布局，生成码原样，ISSUE-6）
    static constexpr uint32_t B_L1_KK_STEP_NT = block_K * C0;
    static constexpr uint32_t B_L1_KK_STEP_TW = block_K * block_N;
    static constexpr uint32_t A_L1_KK_STEP = block_M * block_K;
};

}  // namespace gmm

#endif  // GMM_COMMON_H
