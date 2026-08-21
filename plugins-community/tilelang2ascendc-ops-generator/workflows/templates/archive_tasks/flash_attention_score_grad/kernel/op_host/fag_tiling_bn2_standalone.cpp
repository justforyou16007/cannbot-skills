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
 * \file fag_tiling_bn2_standalone.cpp
 * \brief S1s2Bn2 模板 host tiling 的独立实现（生产 tiling_s1s2_bn2.cpp 的 BNSD 子集移植）。
 *
 * 移植范围（固定配置）：BNSD、keep_prob=1.0、无 pse、无 dropout、sparse_mode=0、
 * 无 rope、非确定性、非 L1Custom、非 NZ 输出（d∈{72,80,88,96} 由路由排除）。
 * 公式与生产代码逐项对齐；框架胶水（TilingContext/PlatformInfo）替换为函数参数 +
 * PlatformAscendCManager 独立平台查询（与 fag_tiling_standalone.cpp 同方式）。
 */

#include "fag_tiling_bn2_standalone.h"

#include <algorithm>
#include <cstdlib>

#include <tiling/tiling_api.h>
#include <tiling/platform/platform_ascendc.h>

namespace fag_standalone {
namespace {

// ---- 常量（与生产 tiling_s1s2_bn2.cpp / tiling_common.h 一致）----
constexpr uint32_t HALF_PRECISION_SIZE = 2;
constexpr uint32_t FLOAT_PRECISION_SIZE = 4;
constexpr uint32_t BASE_LEN_32 = 32;
constexpr uint32_t BASE_LEN_64 = 64;
constexpr uint32_t BASE_LEN_128 = 128;
constexpr uint32_t BASE_LEN_256 = 256;
constexpr uint32_t BASE_LEN_512 = 512;
constexpr int64_t S_LEN_512 = 512;
constexpr int64_t S_LEN_1024 = 1024;
constexpr int64_t GM_ALIGN = 512;
constexpr uint32_t MM_RATIO_4 = 4;
constexpr uint32_t MM_RATIO_8 = 8;
constexpr uint32_t MATMUL_MINI_N = 16;
constexpr uint32_t VEC_REPEAT_LIMIT = 255;
constexpr uint32_t POST_COEX_NODE = 3;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK = 32;
constexpr uint32_t FRACTAL_NUM = 16;
constexpr uint32_t BOOL_BLOCK_NUMS = 32;
constexpr uint64_t B4_L2_CACHESIZE = 96 * 1024 * 1024;
constexpr uint32_t MAX_STRIDE_LIMIT = 65535;
constexpr uint32_t B16 = 2;
constexpr uint32_t B32 = 4;
constexpr int64_t SAMEAB_MIN_S_THRESHOLD = 768;
// atten mask 相关（与生产 tiling_common 一致）
constexpr uint32_t ATTEN_MASK_TYPE_U8_BOOL = 1;
constexpr uint32_t ATTEN_MASK_SHAPE_TYPE_SS = 0;
constexpr uint32_t ATTEN_MASK_SHAPE_TYPE_B1SS = 1;
constexpr uint32_t ATTEN_MASK_SHAPE_TYPE_BNSS = 2;
constexpr uint32_t NO_COMPRESS_MODE = 0;

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

inline uint32_t AlignData_(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b * b;
}

// 中间态（对应生产 tmpData_ + aicoreParams_ 的 BNSD 子集）
struct Bn2TilingCtx {
    // platform
    int64_t aivNum = 0;   // aicoreParams_.numBlocks
    int64_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    uint64_t l2CacheSize = 0;

    // shape
    int64_t b = 0, n2 = 0, g = 0, s1 = 0, s2 = 0, d = 0, value_d = 0;
    FagDtype dtype = FagDtype::FP16;
    float scaleValue = 0;
    bool hasAttenMask = false;
    int64_t maskDim0 = 0, maskDim1 = 0, maskS1 = 0, maskS2 = 0;

    // DecideBranch 产物（归档 calcMode=HIGH_PRECISION_0：fp16 也走高精度分支）
    uint32_t dataTypeSize = FLOAT_PRECISION_SIZE; // UB 内数据类型大小（高精度分支恒 4）

    // DecideBaseMND 产物
    uint32_t baseM = 0, baseN = 0, baseMmm = 0, baseNmm = 0;
    uint32_t tensorSize = 0;
    uint32_t mmRatio = MM_RATIO_4;
    uint32_t s1Ratio = 1, s2Ratio = MM_RATIO_4;
    uint32_t singleM = 0, singleN = 0;
    uint32_t sftSingleM = 0, sftBaseM = 0, baseD = 0;
    bool needSetFixSplit = false;

    // DoBlockTiling 产物
    int64_t usedCoreNum = 0;
    bool needAdjustBlockDim = false;
};

bool GetPlatformInfo_(Bn2TilingCtx &c)
{
    const char *socEnv = std::getenv("SOC_VERSION");
    const char *soc = (socEnv != nullptr && socEnv[0] != '\0') ? socEnv : "Ascend910B2";
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    if (platform == nullptr) {
        platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    }
    if (platform == nullptr) {
        return false;
    }
    c.aivNum = platform->GetCoreNumAiv();
    c.aicNum = platform->GetCoreNumAic();
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, c.ubSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, c.l1Size);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, c.l2CacheSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, c.l0cSize);
    return c.aivNum > 0 && c.aicNum > 0 && c.ubSize > 0;
}

// ---- 生产 VectorBaseMNSplit（dropout 分支不存在，atten 分支保留）----
void VectorBaseMNSplit_(Bn2TilingCtx &c)
{
    if (c.s2 < static_cast<int64_t>(c.baseN) && c.s1 > static_cast<int64_t>(c.baseM)) {
        c.baseN = AlignData_(static_cast<uint32_t>(c.s2), MATMUL_MINI_N);
        c.baseM = c.tensorSize / c.baseN / MATMUL_MINI_N * MATMUL_MINI_N;
        c.baseM = c.baseM > VEC_REPEAT_LIMIT ? (BASE_LEN_256 - MATMUL_MINI_N * 2) : c.baseM;
    } else if (c.s2 > static_cast<int64_t>(c.baseN) && c.s1 < static_cast<int64_t>(c.baseM)) {
        c.baseM = c.s1;
        c.baseN = c.tensorSize / c.baseM / MATMUL_MINI_N * MATMUL_MINI_N;
    }

    c.baseM = c.baseM > BASE_LEN_128 ? BASE_LEN_128 : c.baseM; // simple-softmax 4KB UB 限制
    c.baseN = c.baseN > BASE_LEN_512 ? BASE_LEN_512 : c.baseN; // nz-format 上限 512

    if (c.hasAttenMask) {
        // dropout/atten-mask 只有 8K，baseN 需要 32 对齐
        c.baseM = c.tensorSize / (CeilCommon_(c.baseN, BLOCK) * BLOCK);
        c.baseN = CeilCommon_(c.baseN, BLOCK) * BLOCK;
        c.baseM = c.baseM > BASE_LEN_128 ? BASE_LEN_128 : c.baseM;
        c.baseN = c.baseN > BASE_LEN_512 ? BASE_LEN_512 : c.baseN;
    }
}

// ---- 生产 NMDStrategy（BNSD 子集；SBH 超 stride 分支不存在）----
void NMDStrategy_(Bn2TilingCtx &c)
{
    if (c.d > BASE_LEN_64) {
        if (c.s2 < S_LEN_512) {
            // 减少 S1 方向循环次数，调大 mm 基本块
            c.baseMmm = std::min(BASE_LEN_256, AlignData_(static_cast<uint32_t>(c.s1), FRACTAL_NUM));
            c.baseNmm = std::min(BASE_LEN_128, AlignData_(static_cast<uint32_t>(c.s2), FRACTAL_NUM));
            c.needSetFixSplit = true;
        } else if (c.s2 > S_LEN_512 && c.s2 < S_LEN_1024 && c.l2CacheSize > B4_L2_CACHESIZE) {
            c.mmRatio = MM_RATIO_8; // S2 方向满配比，减少重复搬运
        }
    }

    // 优先 S2 方向配比，S2 不够时 S1 方向配比，总配比不超过 mmRatio
    c.s1Ratio = 1;
    c.s2Ratio = CeilCommon_(c.s2, c.baseNmm);
    c.s2Ratio = c.s2Ratio > c.mmRatio ? c.mmRatio : c.s2Ratio;
    c.s1Ratio = c.mmRatio / c.s2Ratio;
    c.s1Ratio = c.s1Ratio * c.baseMmm > c.s1 ? CeilCommon_(c.s1, c.baseMmm) : c.s1Ratio;
    c.s1Ratio = c.s1Ratio > 0 ? c.s1Ratio : 1;
    c.s2Ratio = c.s2Ratio > 0 ? c.s2Ratio : 1;
}

// ---- 生产 SFTBaseMDSplit ----
void SFTBaseMDSplit_(Bn2TilingCtx &c)
{
    uint32_t typeSize = c.dtype == FagDtype::FP32 ? B32 : B16;
    uint32_t blockNum = BLOCK / typeSize;
    uint32_t dimDAlign = CeilCommon_(c.d, blockNum) * blockNum;

    uint32_t sftSingleM = BASE_LEN_256;
    if (c.singleM < sftSingleM) {
        sftSingleM = c.singleM;
    }

    if (dimDAlign > c.tensorSize) {
        // D 切分模板
        sftSingleM = BASE_LEN_128;
        c.sftBaseM = BASE_LEN_128;
        c.baseD = BASE_LEN_64;
    } else {
        c.baseD = dimDAlign;
    }

    c.sftBaseM = c.tensorSize / c.baseD;
    if (c.sftBaseM > sftSingleM) {
        c.sftBaseM = sftSingleM;
    }
    c.sftSingleM = sftSingleM;
}

// ---- 生产 DecideBaseMND（非 L1Custom 路径）----
void DecideBaseMND_(Bn2TilingCtx &c)
{
    c.baseM = BASE_LEN_64;
    c.baseN = BASE_LEN_128;
    c.baseMmm = BASE_LEN_128;
    c.baseNmm = BASE_LEN_128;
    c.tensorSize = c.baseM * c.baseN;
    c.mmRatio = MM_RATIO_4;

    VectorBaseMNSplit_(c);

    // Matmul 特殊处理（cacheline 友好）
    if (c.d <= BASE_LEN_64 && c.s2 >= BASE_LEN_256) {
        c.baseMmm = BASE_LEN_128;
        c.baseNmm = BASE_LEN_256;
        c.baseM = BASE_LEN_32;
        c.baseN = BASE_LEN_256;
    } else {
        c.baseMmm = BASE_LEN_128;
        c.baseNmm = BASE_LEN_128;
    }

    c.s1Ratio = 1;
    c.s2Ratio = c.mmRatio;
    NMDStrategy_(c);

    c.singleM = c.baseMmm * c.s1Ratio;
    c.singleN = c.baseNmm * c.s2Ratio;

    SFTBaseMDSplit_(c);
}

// ---- 生产 DoBlockTiling（非 TND 路径）----
void DoBlockTiling_(Bn2TilingCtx &c, FlashAttentionScoreGradTilingDataS1s2Bn2 *td)
{
    int64_t nNums = c.b * c.n2;
    int64_t formerCoreProcessNNums = CeilCommon_(nNums, c.aivNum);
    int64_t remainCoreNum = formerCoreProcessNNums * c.aivNum - nNums;
    c.usedCoreNum = c.aivNum;
    td->opInfo.set_usedCoreNum(c.aivNum);
    td->opInfo.set_castUsedCoreNum(c.aivNum); // for pre & post process

    if (nNums < c.aivNum) {
        int64_t minS = std::min(c.s1, c.s2);
        if (nNums * minS * c.d / c.aivNum * c.dataTypeSize < 8 * 1024) {
            // 后处理 UB 利用率严重不足时，减少核启动
            td->opInfo.set_usedCoreNum(nNums);
            td->opInfo.set_castUsedCoreNum(nNums);
            c.usedCoreNum = nNums;
            c.needAdjustBlockDim = true;
        }
    }

    td->opInfo.set_formerCoreNum(c.aivNum - remainCoreNum);
    td->opInfo.set_formerCoreProcessNNum(formerCoreProcessNNums);
    td->opInfo.set_remainCoreProcessNNum(static_cast<uint32_t>(formerCoreProcessNNums - 1));
}

// ---- 生产 DoCastTiling（非 NZ 输出路径）----
bool DoCastTiling_(Bn2TilingCtx &c, FlashAttentionScoreGradTilingDataS1s2Bn2 *td)
{
    int64_t dAlign = (c.d + 15) / 16 * 16;
    int64_t allNumQuery = c.b * c.n2 * c.g * c.s1 * dAlign;
    int64_t allNumKv = c.b * c.n2 * c.s2 * dAlign;
    uint32_t typeSize = c.dtype == FagDtype::FP32 ? B32 : B16;
    uint32_t usedCoreNum = td->opInfo.get_castUsedCoreNum();

    uint32_t postUbBaseSize = c.ubSize / POST_COEX_NODE / BUFFER_NUM / BASE_LEN_256 * BASE_LEN_256;
    uint32_t qPostBaseNum = postUbBaseSize / typeSize;
    if (qPostBaseNum == 0 || usedCoreNum == 0) {
        return false;
    }

    int64_t qPostBlockTotal = allNumQuery / dAlign * c.d;
    int64_t qSizeAlign = (qPostBlockTotal + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN * typeSize;
    int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    int64_t qPostBlockFactor = (qPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t kvPostBaseNum = qPostBaseNum;
    int64_t kvPostBlockTotal = allNumKv / dAlign * c.d;
    int64_t kvSizeAlign = (kvPostBlockTotal + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN * typeSize;
    int64_t kvPostTailNumTmp = kvPostBlockTotal % kvPostBaseNum;
    int64_t kvPostTailNum = kvPostTailNumTmp == 0 ? kvPostBaseNum : kvPostTailNumTmp;
    int64_t kvPostBlockOuterTotal = (kvPostBlockTotal + kvPostBaseNum - 1) / kvPostBaseNum;
    int64_t kvPostBlockFactor = (kvPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t vecQueUbSize = 183 * 1024; // 与 kernel vecQue 共用 UB，最大 183K
    int64_t totalDtypeSize = c.dtype == FagDtype::FP32 ? (B32 + B32) : (B16 + B32);
    uint32_t postMaxDataSize = vecQueUbSize / totalDtypeSize;

    td->postTilingData.set_coreNum(usedCoreNum);
    td->postTilingData.set_scaleValue(c.scaleValue);
    td->postTilingData.set_postUbBaseSize(postUbBaseSize);
    td->postTilingData.set_postMaxDataSize(postMaxDataSize);
    td->postTilingData.set_qPostBlockFactor(qPostBlockFactor);
    td->postTilingData.set_qPostBlockTotal(qPostBlockTotal);
    td->postTilingData.set_qPostBaseNum(qPostBaseNum);
    td->postTilingData.set_qPostTailNum(qPostTailNum);
    td->postTilingData.set_qSizeAlign(qSizeAlign);
    td->postTilingData.set_kvPostBlockFactor(kvPostBlockFactor);
    td->postTilingData.set_kvPostBlockTotal(kvPostBlockTotal);
    td->postTilingData.set_kvPostBaseNum(kvPostBaseNum);
    td->postTilingData.set_kvPostTailNum(kvPostTailNum);
    td->postTilingData.set_kvSizeAlign(kvSizeAlign);
    td->postTilingData.set_nzReservedSize(0);

    td->opInfo.set_dqWorkspaceLen((allNumQuery * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);
    td->opInfo.set_dkWorkspaceLen((allNumKv * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);
    td->opInfo.set_dvWorkspaceLen((allNumKv * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);

    td->postTilingData.set_b(c.b);
    td->postTilingData.set_n2(c.n2);
    td->postTilingData.set_g(c.g);
    td->postTilingData.set_s1(c.s1);
    td->postTilingData.set_s2(c.s2);
    td->postTilingData.set_d(c.d);
    td->postTilingData.set_t1(c.b * c.s1); // BNSD: t1=b*s1, t2=b*s2（仅 TND 使用，占位一致）
    td->postTilingData.set_t2(c.b * c.s2);
    return true;
}

// ---- 生产 DoPreTiling（keep_prob=1.0：dropout workspace 恒 0；bn2 入口无 Pre 阶段，
//      preTilingData 不被 kernel 读取，此处仅为字段完整性按生产公式填充）----
void DoPreTiling_(Bn2TilingCtx &c, FlashAttentionScoreGradTilingDataS1s2Bn2 *td)
{
    uint32_t castBufferLen = 60 * 1024;
    int64_t singleUBProcessNum = castBufferLen / 2;
    int64_t dropMaskSize = c.b * c.n2 * c.g * c.s1 * c.s2;
    uint32_t usedCoreNum = td->opInfo.get_castUsedCoreNum();

    int64_t maskSize = AlignTo_(dropMaskSize, static_cast<int64_t>(BOOL_BLOCK_NUMS));
    int64_t singleCoreNum = AlignTo_(CeilCommon_(maskSize, static_cast<int64_t>(usedCoreNum)),
                                     static_cast<int64_t>(BOOL_BLOCK_NUMS));
    uint32_t maskUsedCoreNum = static_cast<uint32_t>(CeilCommon_(maskSize, singleCoreNum));
    int64_t tailCoreNum = AlignTo_(maskSize - (maskUsedCoreNum - 1) * singleCoreNum,
                                   static_cast<int64_t>(BOOL_BLOCK_NUMS));
    uint32_t singleCoreUBLoop = static_cast<uint32_t>(CeilCommon_(singleCoreNum, singleUBProcessNum));
    uint32_t tailCoreUBLoop = static_cast<uint32_t>(CeilCommon_(tailCoreNum, singleUBProcessNum));
    uint32_t singleCoreUBLastLoopNum =
        static_cast<uint32_t>(singleCoreNum - (singleCoreUBLoop - 1) * singleUBProcessNum);
    uint32_t tailCoreUBLastLoopNum =
        static_cast<uint32_t>(tailCoreNum - (tailCoreUBLoop - 1) * singleUBProcessNum);

    td->preTilingData.set_maskCoreNum(maskUsedCoreNum);
    td->preTilingData.set_castBufferLen(castBufferLen);
    td->preTilingData.set_outputBufferLen(30 * 1024);
    td->preTilingData.set_inputBufferLen(4 * 1024);
    td->preTilingData.set_singleUBProcessNum(static_cast<uint32_t>(singleUBProcessNum));
    td->preTilingData.set_maskSingleCoreNum(singleCoreNum);
    td->preTilingData.set_maskSingleCoreLoop(singleCoreUBLoop);
    td->preTilingData.set_maskLastLoopNum(singleCoreUBLastLoopNum);
    td->preTilingData.set_maskTailCoreLoop(tailCoreUBLoop);
    td->preTilingData.set_maskTailCoreLastLoopNum(tailCoreUBLastLoopNum);
    td->preTilingData.set_dropoutIsDivisibleBy8(1);
    td->preTilingData.set_dropBeginAddr(0);
    td->opInfo.set_dropoutWorkspaceLen(0);
}

// ---- 生产 DoLibApiTiling：softmax/softmaxGrad tiling + 3 个 MatmulApiTiling ----
bool DoLibApiTiling_(Bn2TilingCtx &c, FlashAttentionScoreGradTilingDataS1s2Bn2 *td)
{
    // simpleSoftMax（dstShape == srcShape）
    auto simpleSoftMaxShape = ge::Shape({c.baseM, c.baseN});
    uint32_t helpLenA = c.tensorSize * c.dataTypeSize;
    AscendC::SoftMaxTilingFunc(simpleSoftMaxShape, sizeof(float), helpLenA, td->softmaxTilingData);

    // softmaxGrad：高精度分支 [sftBaseM, BLOCK/4=8]（fp32 计算）
    auto softmaxGradShape = ge::Shape({c.sftBaseM, BLOCK / c.dataTypeSize});
    uint32_t helpLenB = 2 * c.tensorSize * c.dataTypeSize;
    AscendC::SoftMaxGradTilingFunc(softmaxGradShape, c.dataTypeSize, helpLenB, td->softmaxGradTilingData, true);

    auto inputType = matmul_tiling::DataType::DT_FLOAT16;
    auto outputType = matmul_tiling::DataType::DT_FLOAT; // 归档 calcMode=0：fp16 也走高精度 fp32 输出
    if (c.dtype == FagDtype::BF16) {
        inputType = matmul_tiling::DataType::DT_BF16;
    } else if (c.dtype == FagDtype::FP32) {
        inputType = matmul_tiling::DataType::DT_FLOAT;
    }

    // mm1（BNSD: m=S1, k=D, n=S2；A 不转置，B 转置）
    {
        matmul_tiling::MatmulApiTiling mm1;
        mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputType, false);
        mm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputType, true);
        mm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, outputType);
        mm1.SetShape(c.singleM, c.singleN, c.d);
        mm1.SetOrgShape(c.singleM, c.s2, c.d, c.d);
        mm1.SetBias(false);
        mm1.SetBufferSpace(c.l1Size, c.l0cSize);
        mm1.SetFixSplit(c.baseMmm, c.baseNmm, -1);
        if (mm1.GetTiling(td->mm1TilingData) != 0) {
            return false;
        }
        td->mm1TilingData.shareMode = 0;
        td->mm1TilingData.shareL1Size = c.l1Size;
        td->mm1TilingData.shareL0CSize = c.l0cSize;
        td->mm1TilingData.iterateOrder = 1;
    }

    // mm31（dq：BNSD m=S1, k=S2, n=D；A/B 均不转置）
    {
        matmul_tiling::MatmulApiTiling mm31;
        mm31.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputType, false);
        mm31.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputType, false);
        mm31.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, outputType);
        mm31.SetShape(c.singleM, c.d, c.singleN);
        mm31.SetOrgShape(c.s1, c.d, c.s2, c.s2);
        mm31.SetBias(false);
        mm31.SetBufferSpace(c.l1Size, c.l0cSize);
        if (mm31.GetTiling(td->mm31TilingData) != 0) {
            return false;
        }
        td->mm31TilingData.shareMode = 0;
        td->mm31TilingData.shareL1Size = c.l1Size;
        td->mm31TilingData.shareL0CSize = c.l0cSize;
        td->mm31TilingData.iterateOrder = 0;
    }

    // mm4（dk/dv：BNSD m=S2, n=D, k=S1；A 转置，B 不转置）
    {
        matmul_tiling::MatmulApiTiling mm4;
        mm4.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputType, true);
        mm4.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputType, false);
        mm4.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, outputType);
        mm4.SetShape(c.singleN, c.d, c.singleM);
        mm4.SetOrgShape(c.s2, c.d, c.s1, c.s1);
        mm4.SetBias(false);
        if (c.needSetFixSplit) {
            int64_t dAlign = AlignData_(static_cast<uint32_t>(c.d), 16);
            int64_t minBaseN = std::min(BASE_LEN_128, static_cast<uint32_t>(dAlign));
            int64_t s2Align = (c.s2 + FRACTAL_NUM - 1) / FRACTAL_NUM * FRACTAL_NUM;
            int64_t minBaseM = std::min(BASE_LEN_256, static_cast<uint32_t>(s2Align));
            mm4.SetFixSplit(minBaseM, minBaseN, -1);
        }
        mm4.SetBufferSpace(c.l1Size, c.l0cSize);
        if (mm4.GetTiling(td->mm4TilingData) != 0) {
            return false;
        }
        td->mm4TilingData.shareMode = 0;
        td->mm4TilingData.shareL1Size = c.l1Size;
        td->mm4TilingData.shareL0CSize = c.l0cSize;
        td->mm4TilingData.iterateOrder = 0;
    }
    return true;
}

} // namespace

// 路由判定：生产 IsCapable 在归档固定配置（BNSD/无 pse/无 rope/非确定性/非 TND）下的化简。
bool FagBn2IsCapable(int64_t b, int64_t n2, int64_t g, int64_t s1, int64_t s2, int64_t d, int64_t value_d,
                     FagDtype dtype)
{
    Bn2TilingCtx plat;
    if (!GetPlatformInfo_(plat)) {
        return false;
    }
    const int64_t aivNum = plat.aivNum;
    if (s1 >= 1024 || s2 >= 1024) {
        return false; // 大 S 走泛化/SAB
    }
    if (d != value_d) {
        return false; // D!=Dv 生产 bn2 不支持
    }
    if (d > 64) {
        // 实测（archive 8 用例）：D>64 时本 bn2 移植路径显著慢于泛化路径
        // （case1 D128 S512：0.184~0.382ms vs 泛化 0.10~0.12ms；case5 同理），
        // 疑似 NMDStrategy 的 needSetFixSplit 分支在该形态下效率不佳；
        // D>64 一律回退泛化路径（生产在同形态下的实际选择未证实更优，不做无谓跟随）。
        return false;
    }
    if (d == 72 || d == 80 || d == 88 || d == 96) {
        return false; // NZ 输出场景不归档 bn2（走泛化路径）
    }
    if (g > 1 && b * n2 * 2 <= aivNum) {
        return false; // GQA 小 BN2 开不满核，生产路由 SAB
    }
    if (b * n2 < aivNum && (s1 > SAMEAB_MIN_S_THRESHOLD || s2 > SAMEAB_MIN_S_THRESHOLD)) {
        return false;
    }
    // 生产特殊排除：D=128 && B∈[1,6] && G≥4 && keep_prob=1 && 无 pse（路由 SAB）
    if (d == 128 && b >= 1 && b <= 6 && g >= 4) {
        return false;
    }
    (void)dtype; // fp32 在 bn2 kernel 内有分支支持（IsSameType<T1,float>）
    return true;
}

uint64_t FagBn2DoOpTiling(int64_t b, int64_t n2, int64_t g, int64_t s1, int64_t s2, int64_t d, int64_t value_d,
                          FagDtype dtype, float scaleValue, bool hasAttenMask, int64_t attenMaskDim0,
                          int64_t attenMaskDim1, int64_t attenMaskS1, int64_t attenMaskS2,
                          FlashAttentionScoreGradTilingDataS1s2Bn2 *td, uint32_t *blockDimOut)
{
    if (td == nullptr || blockDimOut == nullptr || n2 == 0 || g == 0) {
        return 0;
    }
    Bn2TilingCtx c;
    c.b = b; c.n2 = n2; c.g = g; c.s1 = s1; c.s2 = s2; c.d = d; c.value_d = value_d;
    c.dtype = dtype; c.scaleValue = scaleValue;
    c.hasAttenMask = hasAttenMask;
    c.maskDim0 = attenMaskDim0; c.maskDim1 = attenMaskDim1;
    c.maskS1 = attenMaskS1; c.maskS2 = attenMaskS2;
    if (!GetPlatformInfo_(c)) {
        return 0;
    }

    // ---- GetShapeAttrsInfo / DecideBranch（固定配置化简）----
    td->opInfo.set_B(b);
    td->opInfo.set_N2(n2);
    td->opInfo.set_G(g);
    td->opInfo.set_S1(s1);
    td->opInfo.set_S2(s2);
    td->opInfo.set_D(d);
    td->opInfo.set_scaleValue(scaleValue);
    td->opInfo.set_keepProb(1.0f);
    td->opInfo.set_preTokens(2147483647);  // 无 band
    td->opInfo.set_nextTokens(2147483647);
    td->opInfo.set_sparseMode(0);
    td->opInfo.set_isSparse(0);
    // 关键：kernel 仅在 pseType==PSE_OUTER_ADD_MUL_TYPE(1) 时对 S 施加 scale
    // （DoMulsScale 挂载在该分支上）——无 pse 也必须置 1 占位（同 s1s2 standalone）。
    td->opInfo.set_pseType(1); // PSE_OUTER_ADD_MUL_TYPE
    td->opInfo.set_pseShapeType(0);
    td->opInfo.set_bandIdx(0);
    td->opInfo.set_unpadEmptyInput(0);
    td->opInfo.set_qStartIdx(0);
    td->opInfo.set_kvStartIdx(0);
    td->opInfo.set_pseAlibiBaseS1(0);
    td->opInfo.set_pseAlibiBaseS2(0);

    // atten mask（bool，True=屏蔽；支持 [S1,S2]/[1,1,S1,S2]/[B,1,S1,S2]/[B,N1,S1,S2]）
    td->opInfo.set_existAttenMask(hasAttenMask ? 1 : 0);
    td->opInfo.set_attenMaskCompressMode(NO_COMPRESS_MODE);
    td->opInfo.set_attenMaskS2Size(hasAttenMask ? attenMaskS2 : 0);
    if (hasAttenMask) {
        td->opInfo.set_maskDataType(ATTEN_MASK_TYPE_U8_BOOL);
        uint32_t maskShapeType;
        if (attenMaskDim0 == b && attenMaskDim1 == n2 * g) {
            maskShapeType = ATTEN_MASK_SHAPE_TYPE_BNSS;
        } else if (attenMaskDim0 == b && attenMaskDim1 == 1) {
            maskShapeType = ATTEN_MASK_SHAPE_TYPE_B1SS;
        } else {
            maskShapeType = ATTEN_MASK_SHAPE_TYPE_SS;
        }
        td->opInfo.set_maskShapeType(maskShapeType);
        // CheckAttenMaskShape（NO_COMPRESS_MODE）：mask 的 S1/S2 必须与输入一致
        if (attenMaskS1 != s1 || attenMaskS2 != s2) {
            return 0;
        }
    } else {
        td->opInfo.set_maskShapeType(ATTEN_MASK_SHAPE_TYPE_SS);
        td->opInfo.set_maskDataType(ATTEN_MASK_TYPE_U8_BOOL);
    }

    // ---- 基本块与切分 ----
    DecideBaseMND_(c);

    td->opInfo.set_inputBufferLen(c.tensorSize * B32);
    td->opInfo.set_helpBufferLen(c.tensorSize * B32);
    td->splitCoreParams.set_SFTBaseM(c.sftBaseM);
    td->splitCoreParams.set_SFTSingleM(c.sftSingleM);
    td->splitCoreParams.set_dInner(c.baseD);
    td->splitCoreParams.set_baseM(c.baseM);
    td->splitCoreParams.set_baseN(c.baseN);
    td->splitCoreParams.set_singleM(c.singleM);
    td->splitCoreParams.set_singleN(c.singleN);
    td->splitCoreParams.set_s1OuterOuter(CeilCommon_(s1, c.singleM));
    td->splitCoreParams.set_s2OuterOuter(CeilCommon_(s2, c.singleN));

    DoBlockTiling_(c, td);
    if (!DoCastTiling_(c, td)) {
        return 0;
    }
    DoPreTiling_(c, td);

    if (!DoLibApiTiling_(c, td)) {
        return 0;
    }

    // ---- GetWorkspaceSize（不含 opp 系统预留；host 侧另加 RESERVED_WORKSPACE）----
    int64_t launchBlockDims = c.needAdjustBlockDim ? c.usedCoreNum : c.aivNum;
    // CalcTschBlockDim：mix 1:2 kernel 按 AIC block 数启动
    int64_t ration = c.aicNum == 0 ? 1 : c.aivNum / c.aicNum;
    if (ration <= 0) {
        ration = 1;
    }
    int64_t blockDim = (launchBlockDims + ration - 1) / ration;
    if (blockDim == 0) {
        return 0;
    }
    *blockDimOut = static_cast<uint32_t>(blockDim);

    int64_t mm1WorkspaceLen = AlignTo_(static_cast<int64_t>(c.singleM) * c.singleN * c.dataTypeSize, GM_ALIGN);
    int64_t mm2WorkspaceLen = mm1WorkspaceLen;
    int64_t typeSize = dtype == FagDtype::FP32 ? B32 : B16;
    int64_t space = AlignTo_(static_cast<int64_t>(c.singleM) * c.singleN * typeSize, GM_ALIGN);
    int64_t mm4InputWorkspaceLen = space + space; // matmul 异步 ping-pong 防踩
    int64_t mm3InputWorkspaceLen = space + space;

    td->opInfo.set_mm1WorkspaceLen(mm1WorkspaceLen);
    td->opInfo.set_mm2WorkspaceLen(mm2WorkspaceLen);
    td->opInfo.set_mm4InputWorkspaceLen(mm4InputWorkspaceLen);
    td->opInfo.set_mm3InputWorkspaceLen(mm3InputWorkspaceLen);

    int64_t dqWorkspaceLen = td->opInfo.get_dqWorkspaceLen();
    int64_t dkWorkspaceLen = td->opInfo.get_dkWorkspaceLen();
    int64_t dvWorkspaceLen = td->opInfo.get_dvWorkspaceLen();

    int64_t userWorkspace =
        (mm1WorkspaceLen + mm2WorkspaceLen) * c.usedCoreNum +
        (mm4InputWorkspaceLen + mm3InputWorkspaceLen) * td->opInfo.get_usedCoreNum() +
        dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;

    // post 侧 workspace 偏移（dq/dk/dv 依次排布在 mm 区之后）
    int64_t workspaceOffsets =
        (mm1WorkspaceLen + mm2WorkspaceLen) * c.usedCoreNum +
        (mm4InputWorkspaceLen + mm3InputWorkspaceLen) * td->opInfo.get_usedCoreNum();
    td->postTilingData.set_dqWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets += dqWorkspaceLen;
    td->postTilingData.set_dkWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets += dkWorkspaceLen;
    td->postTilingData.set_dvWorkSpaceOffset(workspaceOffsets);

    if (std::getenv("FAG_BN2_DEBUG") != nullptr) {
        printf("[bn2 tiling] usedCore=%lld former=%lld procN=%lld/%u baseM=%u baseN=%u singleM=%u singleN=%u "
               "sftBaseM=%u sftSingleM=%u baseD=%u tensorSize=%u needFix=%d blk=%u ws=%lld\n",
               (long long)c.usedCoreNum, (long long)td->opInfo.get_formerCoreNum(),
               (long long)td->opInfo.get_formerCoreProcessNNum(), td->opInfo.get_remainCoreProcessNNum(),
               c.baseM, c.baseN, c.singleM, c.singleN, c.sftBaseM, c.sftSingleM, c.baseD, c.tensorSize,
               (int)c.needSetFixSplit, *blockDimOut, (long long)userWorkspace);
        printf("[bn2 tiling] mm1: M=%u N=%u Ka=%u Kb=%u baseM=%u baseN=%u baseK=%u\n",
               td->mm1TilingData.M, td->mm1TilingData.N, td->mm1TilingData.Ka, td->mm1TilingData.Kb,
               td->mm1TilingData.baseM, td->mm1TilingData.baseN, td->mm1TilingData.baseK);
        printf("[bn2 tiling] mm31: M=%u N=%u Ka=%u Kb=%u baseM=%u baseN=%u baseK=%u\n",
               td->mm31TilingData.M, td->mm31TilingData.N, td->mm31TilingData.Ka, td->mm31TilingData.Kb,
               td->mm31TilingData.baseM, td->mm31TilingData.baseN, td->mm31TilingData.baseK);
        printf("[bn2 tiling] mm4: M=%u N=%u Ka=%u Kb=%u baseM=%u baseN=%u baseK=%u\n",
               td->mm4TilingData.M, td->mm4TilingData.N, td->mm4TilingData.Ka, td->mm4TilingData.Kb,
               td->mm4TilingData.baseM, td->mm4TilingData.baseN, td->mm4TilingData.baseK);
    }
    return static_cast<uint64_t>(userWorkspace);
}

} // namespace fag_standalone
