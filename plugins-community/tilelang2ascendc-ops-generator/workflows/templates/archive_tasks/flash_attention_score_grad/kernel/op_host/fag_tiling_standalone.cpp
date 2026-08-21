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
 * \file fag_tiling_standalone.cpp
 * \brief FlashAttentionScoreGrad S1s2Bn2gs1s2 通用模板路径的独立 host tiling 实现。
 *
 * 移植自生产版 op_host/flash_attention_score_grad_tiling_s1s2_bn2gs1s2.cpp（原为 op_host/arch22/ 下，已随归档扁平化），
 * 将 gert::TilingContext / fe::PlatFormInfos 框架胶水替换为函数参数 +
 * PlatformAscendCManager 独立平台查询。仅保留 BNSD / keep_prob=1.0 / 无 pse /
 * sparse_mode=0 / 无 rope 路径；TND、sparse 分块、pese、dropout、rope 分支均已删除。
 * 公式与生产代码逐行对齐（含 NZ 标志计算，归档用例不会命中但保留公式以保证一致）。
 */

#include "fag_tiling_standalone.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include <tiling/tiling_api.h>
#include <tiling/platform/platform_ascendc.h>

namespace fag_standalone {
namespace {

// ---- 常量（与生产 flash_attention_score_grad_tiling_s1s2_bn2gs1s2.cpp / _common.h 一致）----
constexpr uint32_t INITIAL_S1_SPLIT_NUM = 128;
constexpr uint32_t INITIAL_S2_SPLIT_NUM = 64;
constexpr uint32_t MUL_CORE_SYNC_BUFFER = 16 * 1024;
constexpr uint32_t TEMP_BUFFER_REMAIN_SIZE = 1024 * 2;
constexpr uint32_t MATMUL_SIZE = 8 * 1024;
constexpr uint32_t INPUT_ALIGN = 16;
constexpr uint32_t WORKSPACE_NUM_ALIGN = 256;
constexpr int64_t GM_ALIGN = 512;
constexpr uint32_t SOFTMAX_PERF = 64;
constexpr uint32_t BASIC_BLOCK_MULTIPLE = 15;
constexpr uint32_t POST_NZ_COEX_NODE = 10;
constexpr uint32_t POST_COEX_NODE = 3;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t POST_NZ_RESERVED_N = 4;
constexpr uint32_t FP16_BYTES = 2;
constexpr uint32_t FP16_BLOCK_NUMS = 16;
constexpr uint32_t FP32_BYTES = 4;
constexpr uint32_t SHAPE_INFO = 32;
constexpr uint32_t C0_SIZE = 16;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MATMAL_INPUT_NUMS = 2;
constexpr uint32_t S1CV_RATIO_DEFAULT = 1;
constexpr uint32_t S2CV_RATIO_DEFAULT = 8;
constexpr uint32_t CV_RATIO_2 = 2;
constexpr uint32_t CV_RATIO_4 = 4;
constexpr uint32_t CV_RATIO_16 = 16;
constexpr uint32_t WORKSPACE_BUFFER = 20 * 1024 * 1024;
constexpr uint32_t S2_NZ_SIZE = 128;
constexpr uint32_t MM12_ND2NZ_SIZE = 5000;
constexpr uint32_t ASCENDC_API_TEMP_BUFFER = 32 * 1024 + 1024;
constexpr uint32_t API_BOOL_ALIGN = 32;
constexpr uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;
constexpr uint32_t FIX_BASEMN_128 = 128;
constexpr uint32_t FIX_BASEMN_256 = 256;
constexpr int64_t TND_NZ_IN_MAX_D = 256;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BOOL_BLOCK_NUMS = 32;
constexpr uint32_t FRACTAL_NUM = 16;
constexpr uint32_t CORE_LIST_NUM = 50;

constexpr uint32_t EMPTY_TENSOR = 0;
constexpr uint32_t NORMAL_TENSOR = 1;
constexpr uint32_t NO_MASK = 0;
constexpr uint32_t PSE_OUTER_ADD_MUL_TYPE = 1; // 生产默认值（无 pse 时仅作占位写入）
constexpr uint32_t ATTEN_MASK_TYPE_U8_BOOL = 1;
constexpr uint32_t ATTEN_MASK_SHAPE_TYPE_SS = 0;
constexpr uint32_t ATTEN_MASK_SHAPE_TYPE_B1SS = 1;
constexpr uint32_t ATTEN_MASK_SHAPE_TYPE_BNSS = 2;

// 无 band 场景下的 token 取值（与 model_new_ascendc.py 中 pre/next_tockens=65536 对齐）
constexpr int64_t NO_BAND_TOKENS = 65536;

// 中间态（对应生产 FuzzyBaseInfoParams 的 BNSD 子集）
struct FagTilingCtx {
    // platform
    int64_t coreNum = 0; // AIV 核数
    int64_t aicNum = 0;  // AIC 核数
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;

    // shape
    int64_t b = 0, n2 = 0, g = 0, s1 = 0, s2 = 0, d = 0, value_d = 0;
    uint32_t s1Align = 0, s2Align = 0;

    // dtype
    FagDtype dtype = FagDtype::FP16;
    bool isFp32 = false;
    uint32_t dataTypeSize = 0;
    uint32_t calTypeSize = 0;
    bool mm1IsNZOut = false;
    bool mm2IsNZOut = false;

    // attr
    float scaleValue = 0;
    bool hasAttenMask = false;
    uint32_t attenMaskShapeType = 0;
    int64_t attenMaskS1Size = 0;
    int64_t attenMaskS2Size = 0;

    // split
    uint32_t s1CvRatio = 1, s2CvRatio = 8;
    int64_t s1Outer = 0;
    uint32_t s1Inner = 0, s1CvInner = 0, s1Tail = 0, s1CvTail = 0;
    int64_t s2Outer = 0;
    uint32_t s2Inner = 0, cvS2Inner = 0, s2Tail = 0, s2CvTail = 0;
    uint32_t sfmgdOuter = 0, sfmgdInner = 0, sfmgdTail = 0;
    uint32_t baseMN = 0;
    uint32_t tmpBufferSize = 0;

    // block split
    uint32_t blockOuter = 0;
    int64_t blockStarts[CORE_LIST_NUM] = {0};
    int64_t blockEnds[CORE_LIST_NUM] = {0};

    // sizes
    int64_t qSize = 0, kvSize = 0, vSize = 0, dropMaskSize = 0;
    int64_t qSizeAlign = 0, kvSizeAlign = 0, vSizeAlign = 0;
};

template <class T> inline T AlignTo_(const T n, const T alignSize)
{
    if (alignSize == 0) {
        return 0;
    }
    return (n + alignSize - 1) & (~(alignSize - 1));
}

inline int64_t CeilCommon_(int64_t num1, int64_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

bool GetPlatformInfo_(FagTilingCtx &c)
{
    const char *socEnv = std::getenv("SOC_VERSION");
    const char *soc = (socEnv != nullptr && socEnv[0] != '\0') ? socEnv : "Ascend910B2";
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    if (platform == nullptr) {
        // 未显式初始化时尝试默认实例
        platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    }
    if (platform == nullptr) {
        return false;
    }
    c.coreNum = platform->GetCoreNumAiv();
    c.aicNum = platform->GetCoreNumAic();
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, c.ubSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, c.l1Size);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, c.l0cSize);
    if (c.coreNum == 0 || c.aicNum == 0) {
        return false;
    }
    c.ubSize -= MATMUL_SIZE;
    return true;
}

// 对应生产 CheckFuzzyArgsLegal
bool CheckFuzzyArgsLegal_(const FagTilingCtx &c, uint32_t s1Inner, uint32_t s2Inner)
{
    uint32_t baseMNSize = s1Inner * s2Inner * c.calTypeSize;
    if (baseMNSize > c.ubSize) {
        return false;
    }

    // simplesoftmax and dropout
    uint32_t cvS2Inner = s2Inner * c.s2CvRatio;
    uint32_t s2VSize = cvS2Inner > 256 ? 256 : cvS2Inner;
    // ascend api attenmask and dropout last dim 32 align
    if (c.hasAttenMask) { // keepProb==1，仅 attenmask 场景
        s2VSize = (s2VSize + API_BOOL_ALIGN - 1) / API_BOOL_ALIGN * API_BOOL_ALIGN;
    }
    uint32_t s1VecSize = std::min(((INITIAL_S1_SPLIT_NUM * INITIAL_S2_SPLIT_NUM + s2VSize - 1) / s2VSize), s1Inner);

    auto softmaxShape = ge::Shape({s1VecSize, s2VSize});
    auto dropoutShape = ge::Shape({s1VecSize, s2VSize});
    auto selectWithBytesMaskShape1 = ge::Shape({s1VecSize, s2VSize});
    auto selectWithBytesMaskShape2 =
        ge::Shape({s1VecSize, (s2VSize + BOOL_BLOCK_NUMS - 1) / BOOL_BLOCK_NUMS * BOOL_BLOCK_NUMS});
    uint32_t softmaxTmpSize = AscendC::GetSoftMaxMinTmpSize(softmaxShape, c.calTypeSize, true);
    uint32_t dropoutTmpSize = AscendC::GetDropOutMinTmpSize(dropoutShape, c.calTypeSize, true);
    uint32_t minValue = 0;
    uint32_t maxValue = 0;
    AscendC::GetSelectWithBytesMaskMaxMinTmpSize(selectWithBytesMaskShape1, ge::Shape({1}), c.calTypeSize,
                                                 selectWithBytesMaskShape2, sizeof(uint8_t), true, maxValue, minValue);
    uint32_t selectWithBytesMaskTmpSize = minValue;
    uint32_t maxTmpBufferSize = std::max(softmaxTmpSize, dropoutTmpSize);
    maxTmpBufferSize = std::max(maxTmpBufferSize, selectWithBytesMaskTmpSize);
    if (ASCENDC_API_TEMP_BUFFER < maxTmpBufferSize) {
        return false;
    }

    // L0c buffer
    uint32_t bufferSizeL0c = baseMNSize;
    return bufferSizeL0c <= c.l0cSize;
}

// 对应生产 FuzzyForBestSplit
std::tuple<uint32_t, uint32_t, uint32_t> FuzzyForBestSplit_(FagTilingCtx &c)
{
    uint32_t s1Inner = std::min(INITIAL_S1_SPLIT_NUM, c.s1Align);
    uint32_t s2Inner = std::min(INITIAL_S2_SPLIT_NUM, c.s2Align);

    bool left = true;
    while (!CheckFuzzyArgsLegal_(c, s1Inner, s2Inner)) {
        if (left) {
            s1Inner = s1Inner - FRACTAL_NUM;
        } else {
            s2Inner = s2Inner - FRACTAL_NUM;
        }
        left = !left;
    }

    s2Inner = s2Inner > SOFTMAX_PERF ? s2Inner / SOFTMAX_PERF * SOFTMAX_PERF : s2Inner;
    uint32_t first = s1Inner;
    uint32_t second = s2Inner;

    uint32_t tmpBufferSize =
        (c.ubSize - first * second * BASIC_BLOCK_MULTIPLE - first * SHAPE_INFO * c.calTypeSize) / BYTE_BLOCK *
        BYTE_BLOCK;
    if (c.mm1IsNZOut) {
        tmpBufferSize = tmpBufferSize - TEMP_BUFFER_REMAIN_SIZE;
    }
    c.tmpBufferSize = tmpBufferSize;

    // softmaxfront: init d split factor use s2Inner
    uint32_t third = 0;
    uint32_t dInner = std::min(static_cast<int64_t>(s2Inner), c.value_d);
    while (dInner > 0) {
        auto softmaxgradShape = ge::Shape({s1Inner, dInner});
        uint32_t softmaxgradTmpSize = AscendC::GetSoftMaxGradMinTmpSize(softmaxgradShape, c.calTypeSize, true, false);
        if (c.tmpBufferSize < softmaxgradTmpSize) {
            dInner -= FRACTAL_NUM;
        } else {
            third = dInner;
            break;
        }
    }

    third = third > SOFTMAX_PERF ? third / SOFTMAX_PERF * SOFTMAX_PERF : third;
    return std::make_tuple(std::min(first, 128u), std::min(second, 64u), std::min(third, 64u));
}

// 对应生产 DoSplit（删除 TND CV 配比与 TndS1PingPong 分支，BNSD 不命中）
bool DoSplit_(FagTilingCtx &c)
{
    c.s1CvRatio = S1CV_RATIO_DEFAULT;
    c.s2CvRatio = S2CV_RATIO_DEFAULT;
    if (c.d == 64) {
        c.s2CvRatio = CV_RATIO_16;
        if (c.s1 >= 256) {
            if (c.s2 <= 128) {
                c.s1CvRatio = CV_RATIO_4;
                c.s2CvRatio = CV_RATIO_2;
            }
        }
    }

    auto bestSplitRes = FuzzyForBestSplit_(c);
    uint32_t s1Inner = std::get<0>(bestSplitRes);
    uint32_t s1CvInner = s1Inner * c.s1CvRatio;
    if (s1CvInner == 0) {
        return false;
    }
    int64_t s1Outer = (c.s1 + s1CvInner - 1) / s1CvInner;
    uint32_t s1TailTmp = c.s1 % s1Inner;
    uint32_t s1CvTailTmp = c.s1 % s1CvInner;
    c.s1Tail = s1TailTmp == 0 ? s1Inner : s1TailTmp;
    c.s1CvTail = s1CvTailTmp == 0 ? s1CvInner : s1CvTailTmp;
    c.s1Inner = s1Inner;
    c.s1CvInner = s1CvInner;
    c.s1Outer = s1Outer;

    uint32_t s2Inner = std::get<1>(bestSplitRes);
    uint32_t cvS2Inner = s2Inner * c.s2CvRatio;
    if (cvS2Inner == 0) {
        return false;
    }
    int64_t s2Outer = (c.s2 + cvS2Inner - 1) / cvS2Inner;
    uint32_t s2TailTmp = c.s2 % s2Inner;
    uint32_t s2CvTailTmp = c.s2 % cvS2Inner;
    c.s2Tail = s2TailTmp == 0 ? s2Inner : s2TailTmp;
    c.s2CvTail = s2CvTailTmp == 0 ? cvS2Inner : s2CvTailTmp;
    c.s2Outer = s2Outer;
    c.cvS2Inner = cvS2Inner;
    c.s2Inner = s2Inner;

    c.baseMN = s1Inner * s2Inner;
    if (c.baseMN == 0 || c.s2Outer == 0 || c.s1Outer == 0) {
        return false;
    }

    uint32_t sfmgdInner = std::get<2>(bestSplitRes);
    if (sfmgdInner == 0) {
        return false;
    }
    c.sfmgdOuter = (c.value_d + sfmgdInner - 1) / sfmgdInner;
    uint32_t sfmgdTailTmp = c.value_d % sfmgdInner;
    c.sfmgdTail = sfmgdTailTmp == 0 ? sfmgdInner : sfmgdTailTmp;
    c.sfmgdInner = sfmgdInner;
    return true;
}

// 对应生产 DoSparse 的非 sparse 分支（isSparse=false）
bool DoBlockSplit_(FagTilingCtx &c)
{
    int64_t fusedOuter = c.b * c.n2 * c.g * c.s1Outer * c.s2Outer; // 总块数
    int64_t blockFactor = (fusedOuter + c.coreNum - 1) / c.coreNum; // 单核块数
    if (blockFactor == 0) {
        return false;
    }
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor; // 实际核数
    if (blockOuter > CORE_LIST_NUM) {
        return false;
    }
    c.blockOuter = blockOuter;
    for (int64_t i = 0; i < blockOuter; i++) {
        c.blockStarts[i] = blockFactor * i;
        c.blockEnds[i] = std::min(blockFactor * (i + 1), fusedOuter);
    }
    for (int64_t i = blockOuter; i < CORE_LIST_NUM; i++) {
        c.blockStarts[i] = 0;
        c.blockEnds[i] = 0;
    }
    return true;
}

// 对应生产 DoPreTiling（rope_d=0 分支已删除）
bool DoPreTiling_(FagTilingCtx &c, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *td)
{
    uint32_t castBufferLen = 60 * 1024;
    uint32_t outputBufferLen = 30 * 1024;
    uint32_t inputBufferLen = 4 * 1024;
    int64_t singleUBProcessNum = castBufferLen / 2;

    int64_t maskSize = AlignTo_(c.dropMaskSize, static_cast<int64_t>(BOOL_BLOCK_NUMS));
    int64_t singleCoreNum = AlignTo_(CeilCommon_(maskSize, static_cast<int64_t>(c.blockOuter)),
                                     static_cast<int64_t>(BOOL_BLOCK_NUMS));
    int64_t maskUsedCoreNum = static_cast<int64_t>(CeilCommon_(maskSize, singleCoreNum));

    int64_t tailCoreNum = maskSize - (maskUsedCoreNum - 1) * singleCoreNum;
    tailCoreNum = AlignTo_(tailCoreNum, static_cast<int64_t>(BOOL_BLOCK_NUMS));

    int64_t singleCoreUBLoop = static_cast<int64_t>(CeilCommon_(singleCoreNum, singleUBProcessNum));
    int64_t tailCoreUBLoop = static_cast<int64_t>(CeilCommon_(tailCoreNum, singleUBProcessNum));

    int64_t singleCoreUBLastLoopNum = static_cast<int64_t>(singleCoreNum - (singleCoreUBLoop - 1) * singleUBProcessNum);
    int64_t tailCoreUBLastLoopNum = static_cast<int64_t>(tailCoreNum - (tailCoreUBLoop - 1) * singleUBProcessNum);

    td->preTilingData.set_maskCoreNum(maskUsedCoreNum);
    td->preTilingData.set_castBufferLen(castBufferLen);
    td->preTilingData.set_outputBufferLen(outputBufferLen);
    td->preTilingData.set_inputBufferLen(inputBufferLen);
    td->preTilingData.set_singleUBProcessNum(singleUBProcessNum);
    td->preTilingData.set_maskSingleCoreNum(singleCoreNum);
    td->preTilingData.set_maskSingleCoreLoop(singleCoreUBLoop);
    td->preTilingData.set_maskLastLoopNum(singleCoreUBLastLoopNum);
    td->preTilingData.set_maskTailCoreLoop(tailCoreUBLoop);
    td->preTilingData.set_maskTailCoreLastLoopNum(tailCoreUBLastLoopNum);

    if (maskUsedCoreNum == 0) {
        return false;
    }
    int64_t qPreBlockFactor = (c.qSizeAlign + maskUsedCoreNum - 1) / maskUsedCoreNum;
    if (qPreBlockFactor == 0) {
        return false;
    }
    int64_t qPreBlockTotal = (c.qSizeAlign + qPreBlockFactor - 1) / qPreBlockFactor;
    int64_t qPreTailNumTmp = c.qSizeAlign % qPreBlockFactor;
    int64_t qPreTailNum = qPreTailNumTmp == 0 ? qPreBlockFactor : qPreTailNumTmp;

    int64_t kvPreBlockFactor = (c.kvSizeAlign + maskUsedCoreNum - 1) / maskUsedCoreNum;
    if (kvPreBlockFactor == 0) {
        return false;
    }
    int64_t kvPreBlockTotal = (c.kvSizeAlign + kvPreBlockFactor - 1) / kvPreBlockFactor;
    int64_t kvPreTailNumTmp = c.kvSizeAlign % kvPreBlockFactor;
    int64_t kvPreTailNum = kvPreTailNumTmp == 0 ? kvPreBlockFactor : kvPreTailNumTmp;

    td->preTilingData.set_kvPreBlockFactor(kvPreBlockFactor);
    td->preTilingData.set_kvPreBlockTotal(kvPreBlockTotal);
    td->preTilingData.set_kvPreBlockTail(kvPreTailNum);

    int64_t vPreBlockFactor = (c.vSizeAlign + maskUsedCoreNum - 1) / maskUsedCoreNum;
    if (vPreBlockFactor == 0) {
        return false;
    }
    int64_t vPreBlockTotal = (c.vSizeAlign + vPreBlockFactor - 1) / vPreBlockFactor;
    int64_t vPreTailNumTmp = c.vSizeAlign % vPreBlockFactor;
    int64_t vPreTailNum = vPreTailNumTmp == 0 ? vPreBlockFactor : vPreTailNumTmp;

    td->preTilingData.set_vPreBlockFactor(vPreBlockFactor);
    td->preTilingData.set_vPreBlockTotal(vPreBlockTotal);
    td->preTilingData.set_vPreBlockTail(vPreTailNum);

    int64_t maskPreBlockTotal = c.dropMaskSize;
    td->preTilingData.set_qPreBlockFactor(qPreBlockFactor);
    td->preTilingData.set_qPreBlockTotal(qPreBlockTotal);
    td->preTilingData.set_qPreBlockTail(qPreTailNum);
    td->preTilingData.set_dropoutIsDivisibleBy8(1);
    td->preTilingData.set_maskPreBlockTotal(maskPreBlockTotal);

    int64_t dropBeginAddr = SYNC_GLOBAL_WORKSPACE_SIZE;
    dropBeginAddr = (dropBeginAddr + c.qSizeAlign * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dropBeginAddr =
        (dropBeginAddr + c.kvSizeAlign * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dropBeginAddr = (dropBeginAddr + c.vSizeAlign * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    td->preTilingData.set_dropBeginAddr(dropBeginAddr);
    return true;
}

// 对应生产 DoPostTiling（rope/TND 分支已删除；mm2IsNZOut 公式保留）
bool DoPostTiling_(FagTilingCtx &c, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *td)
{
    int64_t dAlign = (c.d + FP16_BLOCK_NUMS - 1) / FP16_BLOCK_NUMS * FP16_BLOCK_NUMS;
    int64_t curPostCoexNode = c.mm2IsNZOut ? POST_NZ_COEX_NODE : POST_COEX_NODE;
    int64_t nzReservedSize = c.mm2IsNZOut ? dAlign / C0_SIZE * BLOCK_SIZE * POST_NZ_RESERVED_N : 0;
    int64_t postUbBaseSize = (c.ubSize - 2 * nzReservedSize) / curPostCoexNode / BUFFER_NUM / WORKSPACE_NUM_ALIGN *
                             WORKSPACE_NUM_ALIGN;

    int64_t qPostBaseNum = c.mm2IsNZOut ? (postUbBaseSize / c.dataTypeSize / dAlign * c.d)
                                        : (postUbBaseSize / c.dataTypeSize);
    if (qPostBaseNum == 0 || c.blockOuter == 0) {
        return false;
    }
    int64_t qPostBlockTotal = c.qSize;
    int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    int64_t qPostBlockFactor = (qPostBlockOuterTotal + c.blockOuter - 1) / c.blockOuter;

    int64_t kvPostBaseNum = qPostBaseNum;
    int64_t kvPostBlockTotal = c.kvSize;
    int64_t kvPostTailNumTmp = kvPostBlockTotal % kvPostBaseNum;
    int64_t kvPostTailNum = kvPostTailNumTmp == 0 ? kvPostBaseNum : kvPostTailNumTmp;
    int64_t kvPostBlockOuterTotal = (kvPostBlockTotal + kvPostBaseNum - 1) / kvPostBaseNum;
    int64_t kvPostBlockFactor = (kvPostBlockOuterTotal + c.blockOuter - 1) / c.blockOuter;

    td->postTilingData.set_kvPostBlockFactor(kvPostBlockFactor);
    td->postTilingData.set_kvPostBlockTotal(kvPostBlockTotal);
    td->postTilingData.set_kvPostBaseNum(kvPostBaseNum);
    td->postTilingData.set_kvPostTailNum(kvPostTailNum);
    td->postTilingData.set_kvSizeAlign(c.kvSizeAlign);

    int64_t vPostBaseNum = qPostBaseNum;
    int64_t vPostBlockTotal = c.vSize;
    int64_t vPostTailNumTmp = vPostBlockTotal % vPostBaseNum;
    int64_t vPostTailNum = vPostTailNumTmp == 0 ? vPostBaseNum : vPostTailNumTmp;
    int64_t vPostBlockOuterTotal = (vPostBlockTotal + vPostBaseNum - 1) / vPostBaseNum;
    int64_t vPostBlockFactor = (vPostBlockOuterTotal + c.blockOuter - 1) / c.blockOuter;

    td->postTilingData.set_vPostBlockFactor(vPostBlockFactor);
    td->postTilingData.set_vPostBlockTotal(vPostBlockTotal);
    td->postTilingData.set_vPostBaseNum(vPostBaseNum);
    td->postTilingData.set_vPostTailNum(vPostTailNum);
    td->postTilingData.set_vSizeAlign(c.vSizeAlign);

    td->postTilingData.set_scaleValue(c.scaleValue);
    td->postTilingData.set_coreNum(c.coreNum);
    td->postTilingData.set_postUbBaseSize(postUbBaseSize);
    td->postTilingData.set_nzReservedSize(nzReservedSize);
    td->postTilingData.set_qPostBlockFactor(qPostBlockFactor);
    td->postTilingData.set_qPostBlockTotal(qPostBlockTotal);
    td->postTilingData.set_qPostBaseNum(qPostBaseNum);
    td->postTilingData.set_qPostTailNum(qPostTailNum);
    td->postTilingData.set_qSizeAlign(c.qSizeAlign);

    int64_t workspaceOffsets = MUL_CORE_SYNC_BUFFER;
    td->postTilingData.set_dqWorkSpaceOffset(workspaceOffsets);

    workspaceOffsets = (workspaceOffsets + c.qSizeAlign * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    td->postTilingData.set_dkWorkSpaceOffset(workspaceOffsets);

    workspaceOffsets = (workspaceOffsets + c.kvSizeAlign * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    td->postTilingData.set_dvWorkSpaceOffset(workspaceOffsets);

    td->postTilingData.set_b(c.b);
    td->postTilingData.set_n2(c.n2);
    td->postTilingData.set_g(c.g);
    td->postTilingData.set_s1(c.s1);
    td->postTilingData.set_s2(c.s2);
    td->postTilingData.set_d(c.d);
    td->postTilingData.set_value_d(c.value_d);
    td->postTilingData.set_rope_d(0);
    td->postTilingData.set_t1(0); // 生产 BNSD 路径 t1/t2 保持默认 0
    td->postTilingData.set_t2(0);
    return true;
}

// 对应生产 DoLibApiTiling（仅 BNSD 分支；mm NZ 公式保留）
bool DoLibApiTiling_(FagTilingCtx &c, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *td,
                     const platform_ascendc::PlatformAscendC &platform)
{
    matmul_tiling::MatmulApiTiling mm1(platform);
    matmul_tiling::MatmulApiTiling mm2(platform);
    matmul_tiling::MatmulApiTiling mm3(platform);
    // 生产逻辑：fp32 用 DT_FLOAT，fp16/bf16 统一用 DT_BFLOAT16 计算 mm tiling
    matmul_tiling::DataType inputAType =
        c.isFp32 ? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_BFLOAT16;

    mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, false);
    mm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, true);
    mm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    int64_t mmS1 = c.s1;
    if (c.mm1IsNZOut) {
        mmS1 = c.s1Inner * c.s1CvRatio;
    }
    // BNSD: left[B,N2,G,S1,D] right[B,N2,1,S2,D] result[B,N2,G,S1,S2]
    mm1.SetOrgShape(mmS1, c.s2, c.d);
    mm1.SetShape(c.s1Inner * c.s1CvRatio, c.s2Inner * c.s2CvRatio, c.d);
    mm1.SetBias(false);

    if (c.cvS2Inner > 128) {
        if (c.d > 64) {
            uint32_t minBaseM = std::min(c.s1CvInner, FIX_BASEMN_256);
            mm1.SetFixSplit(minBaseM, FIX_BASEMN_128, -1);
        } else {
            uint32_t minBaseM = std::min(c.s1CvInner, FIX_BASEMN_128);
            mm1.SetFixSplit(minBaseM, FIX_BASEMN_256, -1);
        }
    } else {
        mm1.SetFixSplit(-1, -1, -1);
    }

    if (mm1.GetTiling(td->mm1TilingData) != 0) {
        return false;
    }
    td->mm1TilingData.shareMode = 0;
    td->mm1TilingData.shareL1Size = c.l1Size;
    td->mm1TilingData.shareL0CSize = c.l0cSize;

    // mm2: left[B,N2,G,S1,S2] right[B,N2,G,S1,D] result[B,N2,G,S2,D]
    mm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, true);
    mm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, false);
    auto outFormat = c.mm2IsNZOut ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND;
    mm2.SetCType(matmul_tiling::TPosition::GM, outFormat, matmul_tiling::DataType::DT_FLOAT);
    mm2.SetOrgShape(c.s2, c.d, c.s1);
    mm2.SetShape(c.s2Inner * c.s2CvRatio, c.d, c.s1Inner * c.s1CvRatio);
    mm2.SetBias(false);
    mm2.SetFixSplit(-1, -1, -1); // mm2IsNZOut 的 FixSplit 仅 TND 命中，BNSD 恒为 (-1,-1,-1)

    if (mm2.GetTiling(td->mm2TilingData) != 0) {
        return false;
    }
    td->mm2TilingData.shareMode = 0;
    td->mm2TilingData.shareL1Size = c.l1Size;
    td->mm2TilingData.shareL0CSize = c.l0cSize;

    // mm3: left[B,N2,G,S1,S2] right[B,N2,1,S2,D] result[B,N2,G,S1,D]
    mm3.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, false);
    mm3.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, false);
    mm3.SetCType(matmul_tiling::TPosition::GM, outFormat, matmul_tiling::DataType::DT_FLOAT);
    mm3.SetOrgShape(c.s1, c.d, c.s2);
    mm3.SetShape(c.s1Inner * c.s1CvRatio, c.d, c.s2Inner * c.s2CvRatio);
    mm3.SetBias(false);
    mm3.SetFixSplit(-1, -1, -1); // mm1IsNZOut && mm2IsNZOut 不同时成立，(-1,-1,-1)

    if (mm3.GetTiling(td->mm3TilingData) != 0) {
        return false;
    }
    td->mm3TilingData.shareMode = 0;
    td->mm3TilingData.shareL1Size = c.l1Size;
    td->mm3TilingData.shareL0CSize = c.l0cSize;

    uint32_t cvS2Inner = c.s2Inner * c.s2CvRatio;
    uint32_t s2VSize = cvS2Inner > 256 ? 256 : cvS2Inner;
    uint32_t s1VecSize =
        std::min(((INITIAL_S1_SPLIT_NUM * INITIAL_S2_SPLIT_NUM + s2VSize - 1) / s2VSize), c.s1Inner);

    auto softmaxShape = ge::Shape({s1VecSize, s2VSize});
    AscendC::SoftMaxTilingFunc(softmaxShape, c.calTypeSize, c.tmpBufferSize, td->softmaxTilingData);
    AscendC::SoftMaxGradTilingFunc(softmaxShape, c.calTypeSize, c.tmpBufferSize, td->softmaxGradTilingData, true);
    return true;
}

// 对应生产 GetWorkspaceSize（无 pse inner、dropoutIsDivisibleBy8=1）
uint64_t GetWorkspaceSize_(const FagTilingCtx &c)
{
    uint64_t workspaceSize = MUL_CORE_SYNC_BUFFER;

    // dq fp32
    workspaceSize = (workspaceSize + static_cast<size_t>(c.qSizeAlign) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // dk fp32
    workspaceSize = (workspaceSize + static_cast<size_t>(c.kvSizeAlign) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // dv fp32
    workspaceSize = (workspaceSize + static_cast<size_t>(c.vSizeAlign) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    // matmul1/matmul2 workspace
    size_t vectorCoreNum = c.coreNum;
    workspaceSize = (workspaceSize +
                     vectorCoreNum * c.s1CvRatio * c.s2CvRatio * c.baseMN * FP32_BYTES * MATMAL_INPUT_NUMS + GM_ALIGN) /
                    GM_ALIGN * GM_ALIGN;
    // CV ratio workspace (pingpong)
    workspaceSize = (workspaceSize +
                     vectorCoreNum * c.s1CvRatio * c.s2CvRatio * c.baseMN * c.dataTypeSize * 2 + GM_ALIGN) /
                    GM_ALIGN * GM_ALIGN;
    // mul workspace
    workspaceSize = (workspaceSize +
                     vectorCoreNum * c.s1CvRatio * c.s2CvRatio * c.baseMN * c.dataTypeSize * 2 + GM_ALIGN) /
                    GM_ALIGN * GM_ALIGN;
    workspaceSize += WORKSPACE_BUFFER;
    return workspaceSize;
}

// 对应生产 SaveToTilingData
void SaveToTilingData_(FagTilingCtx &c, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *td)
{
    td->s1s2BNGS1S2BaseParams.set_coreNum(c.coreNum);

    td->s1s2BNGS1S2BaseParams.set_b(c.b);
    td->s1s2BNGS1S2BaseParams.set_n2(c.n2);
    td->s1s2BNGS1S2BaseParams.set_g(c.g);
    td->s1s2BNGS1S2BaseParams.set_s1(c.s1);
    td->s1s2BNGS1S2BaseParams.set_d(c.d);
    td->s1s2BNGS1S2BaseParams.set_rope_d(0);
    td->s1s2BNGS1S2BaseParams.set_value_d(c.value_d);
    td->s1s2BNGS1S2BaseParams.set_s2(c.s2);

    td->s1s2BNGS1S2BaseParams.set_pseOptional(EMPTY_TENSOR);
    td->s1s2BNGS1S2BaseParams.set_pseType(PSE_OUTER_ADD_MUL_TYPE);
    td->s1s2BNGS1S2BaseParams.set_pseShapeType(0);
    td->s1s2BNGS1S2BaseParams.set_pseDtype(0);
    td->s1s2BNGS1S2BaseParams.set_attenMaskOptional(c.hasAttenMask ? NORMAL_TENSOR : EMPTY_TENSOR);
    td->s1s2BNGS1S2BaseParams.set_attenMaskShapeType(c.attenMaskShapeType);
    td->s1s2BNGS1S2BaseParams.set_attenMaskDtype(c.hasAttenMask ? ATTEN_MASK_TYPE_U8_BOOL : 0);
    td->s1s2BNGS1S2BaseParams.set_scaleValue(c.scaleValue);
    td->s1s2BNGS1S2BaseParams.set_keepProb(1.0f);

    td->s1s2BNGS1S2BaseParams.set_s1Token(NO_BAND_TOKENS);
    td->s1s2BNGS1S2BaseParams.set_s2Token(NO_BAND_TOKENS);

    td->s1s2BNGS1S2BaseParams.set_sparseMode(NO_MASK);
    td->s1s2BNGS1S2BaseParams.set_isSparse(0);
    td->s1s2BNGS1S2BaseParams.set_attenMaskS2Size(c.attenMaskS2Size);
    td->s1s2BNGS1S2BaseParams.set_attenMaskCompressMode(0);
    td->s1s2BNGS1S2BaseParams.set_qStartIdx(0);
    td->s1s2BNGS1S2BaseParams.set_kvStartIdx(0);

    td->s1s2BNGS1S2SplitCoreParams.set_s1CvRatio(c.s1CvRatio);
    td->s1s2BNGS1S2SplitCoreParams.set_s1Outer(c.s1Outer);
    td->s1s2BNGS1S2SplitCoreParams.set_s1Inner(c.s1Inner);
    td->s1s2BNGS1S2SplitCoreParams.set_s1CvInner(c.s1CvInner);
    td->s1s2BNGS1S2SplitCoreParams.set_s1Tail(c.s1Tail);
    td->s1s2BNGS1S2SplitCoreParams.set_s1CvTail(c.s1CvTail);
    td->s1s2BNGS1S2SplitCoreParams.set_s2Outer(c.s2Outer);
    td->s1s2BNGS1S2SplitCoreParams.set_s2CvRatio(c.s2CvRatio);
    td->s1s2BNGS1S2SplitCoreParams.set_s2Inner(c.s2Inner);
    td->s1s2BNGS1S2SplitCoreParams.set_s2Tail(c.s2Tail);
    td->s1s2BNGS1S2SplitCoreParams.set_sfmgdOuter(c.sfmgdOuter);
    td->s1s2BNGS1S2SplitCoreParams.set_sfmgdFactor(c.sfmgdInner);
    td->s1s2BNGS1S2SplitCoreParams.set_sfmgdTail(c.sfmgdTail);

    td->s1s2BNGS1S2SplitCoreParams.set_baseMN(c.baseMN);
    td->s1s2BNGS1S2SplitCoreParams.set_bandIdx(0);
    td->s1s2BNGS1S2BlockNumList.set_blockStarts(c.blockStarts);
    td->s1s2BNGS1S2BlockNumList.set_blockEnds(c.blockEnds);
    td->s1s2BNGS1S2SplitCoreParams.set_blockOuter(c.blockOuter);

    td->s1s2BNGS1S2BaseParams.set_tndSoftmaxIn(0);
}

} // namespace

uint64_t FagS1s2DoOpTiling(int64_t b, int64_t n2, int64_t g, int64_t s1, int64_t s2, int64_t d, int64_t value_d,
                           FagDtype dtype, float scaleValue, bool hasAttenMask, int64_t attenMaskDim0,
                           int64_t attenMaskDim1, int64_t attenMaskS1, int64_t attenMaskS2,
                           FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *tilingData, uint32_t *blockDimOut)
{
    if (tilingData == nullptr || blockDimOut == nullptr || n2 == 0 || g == 0) {
        return 0;
    }
    tilingData->reset();

    FagTilingCtx c;
    if (!GetPlatformInfo_(c)) {
        return 0;
    }

    // ---- shape（生产 GetBaseShapeInfo 的 BNSD 分支 + GetShapeAttrsInfo 派生量）----
    c.b = b;
    c.n2 = n2;
    c.g = g;
    c.s1 = s1;
    c.s2 = s2;
    c.d = d;
    c.value_d = value_d;
    c.s1Align = (s1 + INPUT_ALIGN - 1) / INPUT_ALIGN * INPUT_ALIGN;
    c.s2Align = (s2 + INPUT_ALIGN - 1) / INPUT_ALIGN * INPUT_ALIGN;

    c.qSize = b * n2 * g * s1 * d;
    c.kvSize = b * n2 * 1 * s2 * d;
    c.vSize = b * n2 * 1 * s2 * value_d;
    c.dropMaskSize = b * n2 * g * s2 * s1;

    c.dtype = dtype;
    c.isFp32 = (dtype == FagDtype::FP32);
    if (c.isFp32) {
        c.dataTypeSize = FP32_BYTES;
        c.calTypeSize = FP32_BYTES;
    } else {
        c.dataTypeSize = FP16_BYTES;
        c.calTypeSize = FP32_BYTES;
    }

    c.mm1IsNZOut = (s2 % S2_NZ_SIZE != 0 && s2 < MM12_ND2NZ_SIZE && !c.isFp32 && d <= TND_NZ_IN_MAX_D);
    c.mm2IsNZOut = (!c.isFp32) && (d == 72 || d == 80 || d == 88 || d == 96);

    c.qSizeAlign = c.qSize;
    c.kvSizeAlign = c.kvSize;
    c.vSizeAlign = c.vSize;
    if (c.mm2IsNZOut) {
        c.qSizeAlign = c.qSize / d * ((d + C0_SIZE - 1) / C0_SIZE * C0_SIZE);
        c.kvSizeAlign = c.kvSize / d * ((d + C0_SIZE - 1) / C0_SIZE * C0_SIZE);
        c.vSizeAlign = c.vSize / value_d * ((value_d + C0_SIZE - 1) / C0_SIZE * C0_SIZE);
    }

    c.scaleValue = scaleValue;
    c.hasAttenMask = hasAttenMask;
    if (hasAttenMask) {
        // 生产 CheckAttenMaskShape：4D mask 按 (dim0, dim1) 分类
        if (attenMaskDim0 == b && attenMaskDim1 == n2 * g) {
            c.attenMaskShapeType = ATTEN_MASK_SHAPE_TYPE_BNSS;
        } else if (attenMaskDim0 == b && attenMaskDim1 == 1) {
            c.attenMaskShapeType = ATTEN_MASK_SHAPE_TYPE_B1SS;
        } else if (attenMaskDim0 == 1 && attenMaskDim1 == 1) {
            c.attenMaskShapeType = ATTEN_MASK_SHAPE_TYPE_SS;
        } else {
            return 0;
        }
        if (attenMaskS1 != s1 || attenMaskS2 != s2) { // compressMode=0 时必须与 s1/s2 一致
            return 0;
        }
        c.attenMaskS1Size = attenMaskS1;
        c.attenMaskS2Size = attenMaskS2;
    }

    // ---- split / block / pre / post（生产 DoOpTiling 顺序）----
    if (!DoSplit_(c) || !DoBlockSplit_(c) || !DoPreTiling_(c, tilingData) || !DoPostTiling_(c, tilingData)) {
        return 0;
    }

    // ---- matmul / softmax tiling（生产 DoLibApiTiling）----
    const char *socEnv = std::getenv("SOC_VERSION");
    const char *soc = (socEnv != nullptr && socEnv[0] != '\0') ? socEnv : "Ascend910B2";
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    if (platform == nullptr) {
        platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    }
    if (platform == nullptr || !DoLibApiTiling_(c, tilingData, *platform)) {
        return 0;
    }

    // ---- workspace / save / blockdim ----
    uint64_t workspaceSize = GetWorkspaceSize_(c);
    SaveToTilingData_(c, tilingData);

    // CalcTschBlockDim：mix 1:2 kernel 以 AIC block 数启动
    int64_t blockDim = std::min(static_cast<int64_t>(c.blockOuter), c.aicNum);
    if (blockDim == 0) {
        return 0;
    }
    *blockDimOut = static_cast<uint32_t>(blockDim);
    return workspaceSize;
}

} // namespace fag_standalone
