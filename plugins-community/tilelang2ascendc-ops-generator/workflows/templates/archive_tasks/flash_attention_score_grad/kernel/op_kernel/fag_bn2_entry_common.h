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
 * \file fag_bn2_entry_common.h
 * \brief FlashAttentionScoreGrad S1s2Bn2（小 shape 路径）standalone kernel entries。
 *
 * 与 fag_s1s2_entry_common.h 同一 re-host 约定，差异：
 *   - 无 Pre 阶段（生产 NO_DROPOUT 入口体即如此：核间切 B/N2，每核独占 (b,n2)，
 *     无需多核累加，省去前置 GM 清零）；
 *   - Main 为 FlashAttentionScoreGradS1s2Bn2（mm_12/mm_dq/mm_dkv 三个 matmul 对象），
 *     Post 为 TilingDataS1s2Bn2 的主模板特化（CAST_DV=true，ND）；
 *   - 固定模板实参：BNSD(mock_layout=0)、无 pse、无 dropout、ND 输出、
 *     MM_CFG_NORMAL（生产对小 shape 用 MatmulCfg=0 分支）。
 */

#ifdef KFC_L1_RESERVER_SIZE
#undef KFC_L1_RESERVER_SIZE
#define KFC_L1_RESERVER_SIZE 0 // only support Gm in and Gm out
#else
#define KFC_L1_RESERVER_SIZE 0 // only support Gm in and Gm out
#endif

#include "kernel_operator.h"
using namespace AscendC;

// basic_modules/cube_op_bn2.h 以裸标识符引用布局常量（生产 opp TU 依赖包含顺序下的
// 宽松查找；独立入口 TU 需在包含 s1s2_bn2.h 前显式提供文件作用域定义）。
// 取值与 s1s2_bn2.h 内类常量一致：BNGSD=0 / SBNGD=1 / BSNGD=2 / TND=3。
constexpr static uint32_t BNGSD = 0;
constexpr static uint32_t SBNGD = 1;
constexpr static uint32_t BSNGD = 2;
constexpr static uint32_t TND = 3;

#include "flash_attention_score_grad_tiling.h"
#include "flash_attention_score_grad_post.h"
#include "flash_attention_score_grad_s1s2_bn2.h"

// 与生产 flash_attention_score_grad.cpp 对齐的 matmul 配置
constexpr static MatmulConfig FAG_BN2_MM_CFG = GetNormalConfig(false);
constexpr static uint32_t FAG_BN2_LAYOUT_BNSD_MOCK = 0; // BNSD 的 mock layout 编码
constexpr static uint32_t FAG_BN2_NO_PSE = 0;           // PSE_CFG: INPUT_NONE
constexpr static uint32_t FAG_BN2_NO_DROPOUT = 0;       // DROPOUT_CFG: INPUT_NONE

// 生产 GetTilingKey：非 fp32 时 mm12 输出 NZ（MM_OUT_FORMAT=NZ），fp32 为 ND；
// mm345 仅 d∈{72,80,88,96} 命中 NZ（归档 bn2 路由已排除这些 d，恒 ND）。
constexpr static CubeFormat FAG_BN2_MM12_OUT_FORMAT = CubeFormat::NZ;
constexpr static CubeFormat FAG_BN2_MM345_OUT_FORMAT = CubeFormat::ND;

// 单 dtype x 单 mask 配置的通用入口体。
// INPUT_TYPE: half / bfloat16_t / float；HAS_MASK: 是否带 atten_mask。
// MM12_OUT: 按生产规则随 dtype 取 NZ（fp16/bf16）或 ND（fp32）。
#define FAG_BN2_ENTRY_BODY(INPUT_TYPE, HAS_MASK, MM12_OUT)                                                  \
    do {                                                                                                    \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                                                  \
        AscendC::SetMaskNorm();                                                                             \
        AscendC::SetSysWorkspaceForce(workspace);                                                           \
        if ASCEND_IS_AIC {                                                                                  \
            AscendC::clearWorkspace(workspace);                                                             \
        }                                                                                                   \
        __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);                                        \
                                                                                                            \
        /* GM -> 本地 tiling 结构体逐字拷贝（替代 GET_TILING_DATA_WITH_STRUCT） */                           \
        FlashAttentionScoreGradTilingDataS1s2Bn2 tilingDataBuf;                                             \
        {                                                                                                   \
            __gm__ uint64_t *srcW = reinterpret_cast<__gm__ uint64_t *>(tiling);                            \
            uint64_t *dstW = reinterpret_cast<uint64_t *>(&tilingDataBuf);                                  \
            constexpr uint32_t kWords = sizeof(FlashAttentionScoreGradTilingDataS1s2Bn2) / 8;               \
            for (uint32_t i = 0; i < kWords; i++) {                                                         \
                dstW[i] = srcW[i];                                                                          \
            }                                                                                               \
            if constexpr (sizeof(FlashAttentionScoreGradTilingDataS1s2Bn2) % 8 != 0) {                      \
                __gm__ uint8_t *srcB = reinterpret_cast<__gm__ uint8_t *>(tiling);                          \
                uint8_t *dstB = reinterpret_cast<uint8_t *>(&tilingDataBuf);                                \
                for (uint32_t i = kWords * 8; i < sizeof(tilingDataBuf); i++) {                             \
                    dstB[i] = srcB[i];                                                                      \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
        const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tilingData = &tilingDataBuf;             \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                            \
        const TCubeTiling *__restrict bmm31tiling = &(tilingData->mm31TilingData);                          \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm4TilingData);                            \
                                                                                                            \
        /* Main（生产 NO_DROPOUT_POST2KERNEL 入口：POST=true，post 处理融合在主 kernel 内，
           无独立 Pre/Post 阶段） */                                                                \
        TPipe pipeIn;                                                                               \
        FlashAttentionScoreGradS1s2Bn2<INPUT_TYPE, float, FAG_BN2_MM_CFG, MM12_OUT, FAG_BN2_NO_PSE, \
                                       HAS_MASK, FAG_BN2_NO_DROPOUT, FAG_BN2_LAYOUT_BNSD_MOCK,      \
                                       FAG_BN2_MM345_OUT_FORMAT, true, false>                       \
            op;                                                                                     \
        REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm_12, bmm1tiling, op.mm_dkv, bmm4tiling,       \
                          op.mm_dq, bmm31tiling);                                                           \
        op.Init(query, key, value, dy, nullptr, nullptr, nullptr, atten_mask, softmax_max, softmax_sum,     \
                nullptr, nullptr, nullptr, nullptr, attention_in, dq, dk, dv, nullptr, user,                \
                tilingData, &pipeIn);                                                                       \
        op.Process();                                                                                       \
    } while (0)

// 入口参数约定与 s1s2 入口一致：workspace 倒数第 2、tiling 最后 1 个 GM 参数。
#define FAG_BN2_ENTRY_DECL                                                                                  \
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR atten_mask, GM_ADDR softmax_max,         \
        GM_ADDR softmax_sum, GM_ADDR attention_in, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,                      \
        GM_ADDR workspace, GM_ADDR tiling
