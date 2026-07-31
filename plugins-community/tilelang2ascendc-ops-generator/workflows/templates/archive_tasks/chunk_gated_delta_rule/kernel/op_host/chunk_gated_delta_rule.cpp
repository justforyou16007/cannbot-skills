/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// chunk_gated_delta_rule op_host — tiling computation (faithful port of the CANN
// chunk_gated_delta_rule_tiling.cpp) + EXEC_KERNEL_CMD launch.
//
// This is a **migration**: the tiling math (切块公式、workspace 计算、matmul tiling)
// is carried over verbatim from the operator source. Only the registration glue
// changes: gert::TilingContext* -> PlatformAscendCManager::GetInstance(),
// gert::Shape/ge::DataType -> bare int64 tensors, OP_LOG -> throw/printf, and the
// registered-op launch -> direct <<<>>> EXEC_KERNEL_CMD launch.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <torch/extension.h>
#include <torch/library.h>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_chunk_gated_delta_rule.h"

// host-side matmul tiling library
#include "adv_api/matmul/bmm_tiling.h"

#include "op_kernel/chunk_gated_delta_rule_tiling_data.h"

namespace ascend_kernel {

namespace {

// Index / dim constants ported from the source tiling.
constexpr int64_t P_NUM = 2;
constexpr int64_t STAGE_ONE_PARA_NUM = 4;
constexpr int64_t MASK_NUM = 4;
constexpr int64_t SIZE_HIGH = 4; // sizeof(float)
constexpr int64_t SIZE_LOW = 2;  // sizeof(bfloat16_t)

// Matmul tiling base dims.
constexpr uint32_t MATMUL_BASE_M = 128;
constexpr uint32_t MATMUL_BASE_K = 128;
constexpr uint32_t MATMUL_BASE_N = 128;

void CheckInputs(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                 const at::Tensor &beta, const at::Tensor &initialState,
                 const at::Tensor &actualSeqLengths, const at::Tensor &g)
{
    TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
                "query/key/value must be 3D (T, N, D)");
    TORCH_CHECK(beta.dim() == 2, "beta must be 2D (T, Nv)");
    TORCH_CHECK(g.dim() == 2, "g must be 2D (T, Nv)");
    TORCH_CHECK(initialState.dim() == 4, "initial_state must be 4D (B, Nv, Dv, Dk)");
    TORCH_CHECK(actualSeqLengths.dim() == 1, "actual_seq_lengths must be 1D (B,)");

    TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16 &&
                    value.scalar_type() == at::kBFloat16 && beta.scalar_type() == at::kBFloat16,
                "query/key/value/beta must be bfloat16");
    TORCH_CHECK(g.scalar_type() == at::kFloat, "g must be float32");
    TORCH_CHECK(actualSeqLengths.scalar_type() == at::kInt, "actual_seq_lengths must be int32");
    TORCH_CHECK(initialState.scalar_type() == at::kBFloat16,
                "initial_state must be bfloat16 (FP32 state is unsupported on this SoC)");

    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous() &&
                    beta.is_contiguous() && g.is_contiguous() && initialState.is_contiguous() &&
                    actualSeqLengths.is_contiguous(),
                "inputs must be contiguous");

    const int64_t t = query.sizes()[0];
    const int64_t nk = query.sizes()[1];
    const int64_t dk = query.sizes()[2];
    const int64_t nv = value.sizes()[1];
    const int64_t dv = value.sizes()[2];
    const int64_t b = initialState.sizes()[0];

    TORCH_CHECK(key.sizes()[0] == t && key.sizes()[1] == nk && key.sizes()[2] == dk,
                "key shape mismatch");
    TORCH_CHECK(value.sizes()[0] == t && value.sizes()[2] == dv, "value shape mismatch");
    TORCH_CHECK(beta.sizes()[0] == t && beta.sizes()[1] == nv, "beta shape mismatch");
    TORCH_CHECK(g.sizes()[0] == t && g.sizes()[1] == nv, "g shape mismatch");
    TORCH_CHECK(actualSeqLengths.sizes()[0] == b, "actual_seq_lengths shape mismatch");
    TORCH_CHECK(initialState.sizes()[1] == nv && initialState.sizes()[2] == dv &&
                    initialState.sizes()[3] == dk,
                "initial_state shape mismatch");

    TORCH_CHECK(nk > 0 && nv > 0 && dk > 0 && dv > 0 && t > 0 && b > 0, "dims must be positive");
    TORCH_CHECK(nv % nk == 0, "nv must be an integer multiple of nk");
    TORCH_CHECK(nk <= 64 && nv <= 64, "nk/nv must be <= 64");
    TORCH_CHECK(dk <= 128 && dv <= 128, "dk/dv must be <= 128");
}

} // namespace

std::vector<at::Tensor> chunk_gated_delta_rule(const at::Tensor &query,
                                               const at::Tensor &key,
                                               const at::Tensor &value,
                                               const at::Tensor &beta,
                                               const at::Tensor &initialState,
                                               const at::Tensor &actualSeqLengths,
                                               const at::Tensor &g,
                                               double scale)
{
    CheckInputs(query, key, value, beta, initialState, actualSeqLengths, g);

    // ---------------------------------------------------------------------
    // Platform info (interface replacement: gert::TilingContext ->
    //   PlatformAscendCManager::GetInstance()).  Logic identical to
    //   ChunkGatedDeltaRuleTiling::InitCompileInfo().
    // ---------------------------------------------------------------------
    auto *ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(ascendcPlatform != nullptr, "PlatformAscendCManager::GetInstance() is null");

    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t aivNum = ascendcPlatform->GetCoreNumAiv();
    const uint32_t aicNum = ascendcPlatform->GetCoreNumAic();
    TORCH_CHECK(aivNum > 0 && aicNum > 0, "aivNum/aicNum must be > 0");
    const auto socVersion = ascendcPlatform->GetSocVersion();

    ChunkGatedDeltaRule::ChunkGatedDeltaRuleTilingData tilingData = {};

    // DoOpTiling: chunking parameters (faithful port).
    const int64_t c = 64; // chunkSize
    tilingData.chunkSize = c;
    tilingData.aiCoreNum = aicNum;
    tilingData.maxGroupLength = P_NUM * tilingData.aiCoreNum * tilingData.chunkSize;
    tilingData.stageOneParaNum = STAGE_ONE_PARA_NUM;

    // shapes
    const int64_t t = static_cast<int64_t>(query.sizes()[0]);
    const int64_t nk = static_cast<int64_t>(query.sizes()[1]);
    const int64_t dk = static_cast<int64_t>(query.sizes()[2]);
    const int64_t nv = static_cast<int64_t>(value.sizes()[1]);
    const int64_t dv = static_cast<int64_t>(value.sizes()[2]);
    const int64_t b = static_cast<int64_t>(initialState.sizes()[0]);
    tilingData.t = t;
    tilingData.nk = nk;
    tilingData.dk = dk;
    tilingData.nv = nv;
    tilingData.dv = dv;
    tilingData.b = b;

    // GetOptionalInput / GetScale (interface replacement).
    tilingData.hasGamma = 1; // g is always passed by model_new_ascendc ("zeros when absent")
    tilingData.scale = static_cast<float>(scale);

    // AnalyzeDtype: FP32 state unsupported on non-Ascend950.
    tilingData.stateIsFp32 = 0; // input enforced bf16
    if (socVersion != platform_ascendc::SocVersion::ASCEND950) {
        // (keep the guard for parity; bf16 path only)
    }

    // GetStrides (faithful port of GetStrides())
    tilingData.stateStride1 = static_cast<int64_t>(dk) * dv;
    tilingData.stateStride0 = static_cast<int64_t>(nv) * tilingData.stateStride1;

    // ---------------------------------------------------------------------
    // interWorkspaceSz / stageWorkspaceSz (faithful port of DoOpTiling()).
    // ---------------------------------------------------------------------
    const int64_t s = tilingData.maxGroupLength;
    int64_t interWorkspaceSz = 0;
    interWorkspaceSz += SIZE_HIGH * nv * s;                                 // gCumExp (FP32)
    interWorkspaceSz += SIZE_LOW * nv * s * dk;                             // kCumDecay (BF16)
    interWorkspaceSz += SIZE_LOW * nv * s * dv;                             // vInner (BF16)
    interWorkspaceSz += SIZE_LOW * nv * s * dk;                             // qPrime (BF16)
    interWorkspaceSz += SIZE_LOW * nv * s * dv;                             // attnInter (BF16)
    interWorkspaceSz += SIZE_LOW * nv * s * dk;                             // kg (BF16)
    interWorkspaceSz += SIZE_LOW * nv * s * c;                              // qkt (BF16)
    if (tilingData.stateIsFp32) {
        // stateBf16Wk (FP32 state path) — unused on 910B3
    } else if (socVersion != platform_ascendc::SocVersion::ASCEND950) {
        interWorkspaceSz +=
            SIZE_HIGH * tilingData.b * nv * dv * dk; // highState_ (arch22)
    }
    interWorkspaceSz += SIZE_HIGH * c * c * tilingData.aiCoreNum * MASK_NUM; // mask (FP32)
    tilingData.interWorkspaceSz = interWorkspaceSz;

    tilingData.stageWorkspaceSz =
        SIZE_LOW * c * (2 * c + 3 * dk + dv) * tilingData.stageOneParaNum;
    tilingData.stageWorkspaceSz *= tilingData.aiCoreNum;

    // workspaceSize (faithful port of GetWorkspaceSize()).
    const int64_t SYS_WORKSPACE_SIZE = 16777216; // 16 MB
    const int64_t workspaceSize = SYS_WORKSPACE_SIZE + interWorkspaceSz + tilingData.stageWorkspaceSz;

    // ---------------------------------------------------------------------
    // Matmul tiling (DoMatmulTiling — BF16 path, FP32 state path omitted on 910B3).
    // ---------------------------------------------------------------------
    {
        uint64_t l1Size = 0;
        uint64_t l0CSize = 0;
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);

        matmul_tiling::MultiCoreMatmulTiling mm_;
        mm_.SetBufferSpace(static_cast<int32_t>(l1Size), static_cast<int32_t>(l0CSize),
                           static_cast<int32_t>(ubSize));
        mm_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_BFLOAT16, true);
        mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_BFLOAT16, true);
        mm_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_BFLOAT16);
        mm_.SetBias(false);
        mm_.SetDim(1);
        mm_.SetShape(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);
        mm_.SetOrgShape(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);
        mm_.SetFixSplit(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);
        if (mm_.GetTiling(tilingData.matmulTilingFp32) != 0) {
            throw std::runtime_error("CGDR: Get Tiling Failed!");
        }
        tilingData.matmulTilingFp32.dbL0C = 1;
        tilingData.matmulTilingFp32.stepKa = 1;
        tilingData.matmulTilingFp32.stepKb = 1;
        tilingData.matmulTilingFp32.depthA1 = 1;
        tilingData.matmulTilingFp32.depthB1 = 1;
        tilingData.matmulTilingFp32.stepM = 1;
        tilingData.matmulTilingFp32.stepN = 1;
    }

    // ---------------------------------------------------------------------
    // Allocate outputs.
    // ---------------------------------------------------------------------
    at::Tensor out = at::empty({t, nv, dv}, query.options());
    at::Tensor finalState = at::empty_like(initialState);
    if (!out.is_contiguous() || !finalState.is_contiguous()) {
        out = out.contiguous();
        finalState = finalState.contiguous();
    }

    // Workspace tensor (bytes).
    at::Tensor workspace = at::empty({static_cast<int64_t>(workspaceSize)},
                                     at::device(at::kPrivateUse1).dtype(at::kByte));

    // Tiling tensor (bytes) -> copied to GM, kernel copies back to a local struct.
    at::Tensor tilingCpu = at::empty({static_cast<int64_t>(sizeof(ChunkGatedDeltaRule::ChunkGatedDeltaRuleTilingData))},
                                     at::device(at::kCPU).dtype(at::kByte));
    std::memcpy(tilingCpu.data_ptr(), &tilingData,
                sizeof(ChunkGatedDeltaRule::ChunkGatedDeltaRuleTilingData));
    at::Tensor tilingNpu = tilingCpu.to(at::kPrivateUse1);

    const uint32_t blockDim = tilingData.aiCoreNum;
    EXEC_KERNEL_CMD(chunk_gated_delta_rule, blockDim,
                    query, key, value, beta, initialState, actualSeqLengths, g,
                    out, finalState, workspace, tilingNpu);

    return {out, finalState};
}

} // namespace ascend_kernel
