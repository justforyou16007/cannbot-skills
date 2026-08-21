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
 * \file fag_s1s2_entry_common.h
 * \brief FlashAttentionScoreGrad standalone kernel entries (archive re-host).
 *
 * 本文件将生产版 opp kernel（arch22 S1s2Bn2gs1s2 通用模板路径）以独立
 * ascendc_library 方式 re-host：
 *   - 固定模板实参：BNSD 布局(mock_layout=0)、无 pse、无 dropout(keep_prob=1)、
 *     ND 输出格式（归档用例 s2%128==0 且 d 不在 {72,80,88,96}，不会命中 NZ）、
 *     非 TND pingpong、无 rope；
 *   - 按 dtype(fp16/bf16/fp32) x atten_mask(有/无) 展开为 6 个非模板入口，
 *     由 host 侧按输入直接选择对应 aclrtlaunch 符号，绕开 opp 的
 *     REGISTER_TILING_FOR_TILINGKEY 分发体系；
 *   - tiling blob 作为最后一个 GM 参数传入，入口内拷贝到本地结构体
 *     （替代 opp 专属的 GET_TILING_DATA_WITH_STRUCT）；
 *   - 独立构建的 host stub 不做系统 workspace 偏移（HAVE_WORKSPACE 未定义），
 *     因此入口内自行 SetSysWorkspaceForce + GetUserWorkspace，
 *     与 opp 运行时语义对齐；host 侧需额外分配 RESERVED_WORKSPACE(16MB)。
 */

#ifdef KFC_L1_RESERVER_SIZE
#undef KFC_L1_RESERVER_SIZE
#define KFC_L1_RESERVER_SIZE 0 // only support Gm in and Gm out
#else
#define KFC_L1_RESERVER_SIZE 0 // only support Gm in and Gm out
#endif

#include "kernel_operator.h"
using namespace AscendC;

#include "flash_attention_score_grad_tiling.h"
#include "flash_attention_score_grad_post.h"
#include "flash_attention_score_grad_s1s2_bn2gs1s2.h"
#include "flash_attention_score_grad_pre.h"

// BNSD 布局在 kernel 内部的 mock layout 编码（与生产 mock_layout 重映射一致: BNSD(2) -> 0）
constexpr static uint32_t FAG_LAYOUT_BNSD_MOCK = 0;
constexpr static uint32_t FAG_ND = 0; // Post 的 input_format: ND

// 单 dtype x 单 mask 配置的通用入口体。
// INPUT_TYPE: half / bfloat16_t / float；HAS_MASK: 是否带 atten_mask。
#define FAG_S1S2_ENTRY_BODY(INPUT_TYPE, HAS_MASK)                                                        \
    do {                                                                                                 \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                                               \
        AscendC::SetMaskNorm();                                                                          \
        /* 独立构建无 opp 系统 workspace 管理，自行建立 sys/user workspace 划分 */                        \
        AscendC::SetSysWorkspaceForce(workspace);                                                        \
        if ASCEND_IS_AIC {                                                                               \
            AscendC::clearWorkspace(workspace);                                                        \
        }                                                                                              \
        __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);                                     \
                                                                                                         \
        /* GM -> 本地 tiling 结构体逐字拷贝（设备侧无 memcpy，替代 GET_TILING_DATA_WITH_STRUCT） */      \
        FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 tilingDataBuf;                                     \
        {                                                                                                \
            __gm__ uint64_t *srcW = reinterpret_cast<__gm__ uint64_t *>(tiling);                         \
            uint64_t *dstW = reinterpret_cast<uint64_t *>(&tilingDataBuf);                               \
            constexpr uint32_t kWords = sizeof(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2) / 8;       \
            for (uint32_t i = 0; i < kWords; i++) {                                                      \
                dstW[i] = srcW[i];                                                                       \
            }                                                                                            \
            if constexpr (sizeof(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2) % 8 != 0) {              \
                __gm__ uint8_t *srcB = reinterpret_cast<__gm__ uint8_t *>(tiling);                       \
                uint8_t *dstB = reinterpret_cast<uint8_t *>(&tilingDataBuf);                             \
                for (uint32_t i = kWords * 8; i < sizeof(tilingDataBuf); i++) {                          \
                    dstB[i] = srcB[i];                                                                   \
                }                                                                                        \
            }                                                                                            \
        }                                                                                                \
        const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *__restrict tilingData = &tilingDataBuf;     \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                         \
        const TCubeTiling *__restrict bmm2tiling = &(tilingData->mm2TilingData);                         \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm3TilingData);                         \
                                                                                                         \
        TPipe pipeIn;                                                                                    \
        FlashAttentionScoreGradPre<INPUT_TYPE, float,                                                    \
                                   FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, true>                  \
            opPre;                                                                                       \
        opPre.Init(dq, nullptr, dk, nullptr, dv, nullptr, user, tilingData, &pipeIn);                    \
        opPre.Process();                                                                                 \
        opPre.SyncALLCores();                                                                            \
        pipeIn.Destroy();                                                                                \
                                                                                                         \
        TPipe pipeBase;                                                                                  \
        FlashAttentionScoreGradS1s2Bn2gs1s2<INPUT_TYPE, float, HAS_MASK, false, false, CubeFormat::ND,   \
                                            FAG_LAYOUT_BNSD_MOCK, CubeFormat::ND, false>                 \
            op;                                                                                          \
        REGIST_MATMUL_OBJ(&pipeBase, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm3, bmm2tiling,       \
                          op.mm4, bmm3tiling);                                                           \
        op.Init(key, nullptr, value, dy, query, nullptr, nullptr, nullptr, atten_mask, attention_in,     \
                softmax_max, softmax_sum, nullptr, nullptr, nullptr, dq, nullptr, dk, nullptr, dv,       \
                nullptr, user, tilingData, &pipeBase);                                                   \
        op.Process();                                                                                    \
        op.SyncALLCores();                                                                               \
        pipeBase.Destroy();                                                                              \
                                                                                                         \
        TPipe pipePost;                                                                                  \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, true,     \
                                    FAG_LAYOUT_BNSD_MOCK, FAG_ND>                                        \
            opPost;                                                                                      \
        opPost.Init(dq, nullptr, dk, nullptr, dv, nullptr, nullptr, nullptr, user, tilingData,           \
                    &pipePost);                                                                          \
        opPost.Process();                                                                                \
    } while (0)

// 入口参数约定：workspace 为倒数第 2 个、tiling 为最后 1 个 GM 参数（host stub 约定）。
#define FAG_S1S2_ENTRY_DECL                                                                              \
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR atten_mask, GM_ADDR softmax_max,      \
        GM_ADDR softmax_sum, GM_ADDR attention_in, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,                   \
        GM_ADDR workspace, GM_ADDR tiling
