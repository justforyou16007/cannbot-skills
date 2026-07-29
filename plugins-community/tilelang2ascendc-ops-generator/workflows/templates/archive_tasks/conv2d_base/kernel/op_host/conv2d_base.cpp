/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file op_host/conv2d_base.cpp
 * Host entry for conv2d_base on arch20 (910B), v2: host-zero-prep.
 *
 * The host performs NO at:: layout/pad/cast transformations; raw x / weight /
 * bias tensors go straight into the kernels. All preprocessing semantics are
 * expressed as scalar tiling parameters + device-side data movement:
 *   stage0w conv2d_base_wrepack_{fp16,bf16} (AIV):
 *     raw weight [Co, Ci, kH, kW] -> K-major ws [Kpad, CoPad]
 *     (k order = (kh,kw,ci); Ci/Co tails zero-filled via UB TransDataTo5HD)
 *   stage0x conv2d_base_im2col_{fp16,bf16} (AIV):
 *     raw x (NCHW or NHWC) + spatial pad + Ci tail -> im2col ws
 *     [totalRows, Kpad] (DataCopyPad clamp/zero + UB transpose for NCHW)
 *   stage1  conv2d_base_conv_{fp16,bf16} (AIC):  (unchanged, proven)
 *     Nd2Nz -> L1 NZ -> L0A/L0B -> Mmad fp32 -> Fixpipe -> acc ws
 *   stage2  conv2d_base_epilogue_{fp16,bf16} (AIV):
 *     y = cast_T(acc_fp32 + bias_raw) written directly in the target layout
 *     (NCHW via UB 16x16 transpose tiles + DataCopyPad)
 * Tiling fields are few scalars -> passed directly as launch arguments.
 */

#include <cstdlib>
#include <list>
#include <mutex>
#include <vector>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_conv2d_base_wrepack_fp16.h"
#include "aclrtlaunch_conv2d_base_wrepack_bf16.h"
#include "aclrtlaunch_conv2d_base_im2col_fp16.h"
#include "aclrtlaunch_conv2d_base_im2col_bf16.h"
#include "aclrtlaunch_conv2d_base_conv_fp16.h"
#include "aclrtlaunch_conv2d_base_conv_bf16.h"
#include "aclrtlaunch_conv2d_base_epilogue_fp16.h"
#include "aclrtlaunch_conv2d_base_epilogue_bf16.h"
#include "aclrtlaunch_conv2d_base_xpad_fp16.h"
#include "aclrtlaunch_conv2d_base_xpad_bf16.h"
#include "aclrtlaunch_conv2d_base_xtrans_fp16.h"
#include "aclrtlaunch_conv2d_base_xtrans_bf16.h"
#include "conv2d_base_tiling.h"
#include "ops.h"

namespace ascend_kernel {

namespace {

// P1-a: weight K-major repack cache.
// The wrepack stage deterministically rewrites the full [K, CoPad] workspace
// from (weight content, shape, dtype). When the same weight tensor is
// convolved repeatedly (the standard inference pattern), the repack result is
// identical and the wrepack kernel launch can be skipped entirely.
//   - key: weight data_ptr + sizes + dtype + version counter; an in-place
//     mutation bumps the version and forces a re-repack.
//   - a strong reference to the weight is kept inside the entry, so its
//     data_ptr cannot be recycled by the allocator while cached (no stale
//     address-reuse hits).
//   - entries are a small LRU; the cached wWs tensor is kept alive by the
//     entry and fully rewritten on every miss.
//   - assumes same-stream execution across calls (stream-ordered reuse of
//     wWs), the same assumption the per-call workspaces already rely on.
// Disable with CONV2D_WREPACK_CACHE=0.
struct WeightRepackCacheEntry {
    at::Tensor weight;
    at::Tensor wWs;
    uint32_t version;
    std::vector<int64_t> sizes;
    at::ScalarType dtype;
    int32_t fused;  // k-order layout: 0 = (kh,kw,ci), 1 = (ci1,kh,kw,ci0)
};

constexpr size_t kWRepackCacheCapacity = 8;
std::list<WeightRepackCacheEntry> g_wrepackCache;
std::mutex g_wrepackCacheMutex;

bool wrepackCacheEnabled()
{
    static const bool enabled = [] {
        const char *e = getenv("CONV2D_WREPACK_CACHE");
        return !(e != nullptr && atoi(e) == 0);
    }();
    return enabled;
}

void wrepackCacheInsert(const at::Tensor &weight, const at::Tensor &wWs,
                        uint32_t version, int32_t fused)
{
    if (!wrepackCacheEnabled()) { return; }
    std::lock_guard<std::mutex> lock(g_wrepackCacheMutex);
    g_wrepackCache.push_front(
        {weight, wWs, version, weight.sizes().vec(), weight.scalar_type(),
         fused});
    while (g_wrepackCache.size() > kWRepackCacheCapacity) {
        g_wrepackCache.pop_back();
    }
}

} // namespace

at::Tensor conv2d_base(
    const at::Tensor &x, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    std::string data_format)
{
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D [Co, Ci, kH, kW]");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x must be float16 or bfloat16");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype must match x");
    TORCH_CHECK(stride.size() == 2 && padding.size() == 2 && dilation.size() == 2,
                "stride/padding/dilation must be [h, w]");
    const bool nhwc = (data_format == "NHWC");
    TORCH_CHECK(nhwc || data_format == "NCHW", "data_format must be NCHW or NHWC");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    int64_t batch, Ci, H, W;
    if (nhwc) {
        batch = x.sizes()[0];
        H = x.sizes()[1];
        W = x.sizes()[2];
        Ci = x.sizes()[3];
    } else {
        batch = x.sizes()[0];
        Ci = x.sizes()[1];
        H = x.sizes()[2];
        W = x.sizes()[3];
    }
    const int64_t Co = weight.sizes()[0];
    const int64_t kH = weight.sizes()[2];
    const int64_t kW = weight.sizes()[3];
    TORCH_CHECK(weight.sizes()[1] == Ci, "weight Ci must match x");
    TORCH_CHECK(kH * kW <= 64, "kH*kW > 64 not supported by wrepack tile");

    const bool hasBias = bias.has_value() && bias->numel() > 0;
    TORCH_CHECK(!hasBias || bias->sizes()[0] == Co, "bias must be [Co]");
    TORCH_CHECK(!hasBias || bias->scalar_type() == x.scalar_type(),
                "bias dtype must match x");

    const int64_t sH = stride[0], sW = stride[1];
    const int64_t pH = padding[0], pW = padding[1];
    const int64_t dH = dilation[0], dW = dilation[1];

    const int64_t Ho = (H + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    const int64_t Wo = (W + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    TORCH_CHECK(Ho > 0 && Wo > 0, "output spatial size must be positive");

    // ---- tiling scalars ----
    const int32_t CiPad = ConvRoundUp16(static_cast<int32_t>(Ci));
    const int32_t CoPad = ConvRoundUp16(static_cast<int32_t>(Co));
    const int64_t K = static_cast<int64_t>(CiPad) * kH * kW;
    const int64_t mImg = Ho * Wo;
    const int64_t totalRows = batch * mImg;

    // P0-b: direct 1x1 mode (NCHW, s=1, p=0, d=1, Ci%16==0). The GEMM is
    // flipped: A = raw weight [Co, Ci] (shared, skips wrepack + its cache),
    // B = raw x NCHW [Ci, H*W] (already the [K, M] ND layout, skips the
    // im2col kernel entirely), acc = [Co, H*W] fp32 written per image, and
    // the epilogue direct mode adds per-row scalar bias + casts straight
    // into the NCHW output (no transpose). All loads stay on the proven
    // Nd2Nz path (plain 2D DataCopy GM<->L1 is unreliable on this platform).
    // Alignment gate: (H*W)%16==0 keeps B' rows 32B-aligned for the Nd2Nz
    // load. P7 tried lifting this via a UB-relay xpad repack kernel: the
    // xpad itself costs ~5.4us (392B non-32B rows hit DataCopyPad's slow
    // dummy-pad path) and the flipped GEMM on narrow padded rows is slower
    // than the normal conv (8.7 vs 4.5us) — measured NET NEGATIVE for
    // case 9 (20.7 vs 19.4us), so the gate is restored and the xpad /
    // padded-epilogue paths stay dormant (they are correct, just slower).
    const bool direct1x1 = (!nhwc) && kH == 1 && kW == 1 &&
                           sH == 1 && sW == 1 && pH == 0 && pW == 0 &&
                           dH == 1 && dW == 1 && (Ci % 16 == 0) &&
                           (mImg % 16 == 0);

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int32_t aicNum = static_cast<int32_t>(ascendcPlatform->GetCoreNumAic());
    const int32_t aivNum = static_cast<int32_t>(ascendcPlatform->GetCoreNumAiv());

    const int32_t kAL1 = ConvComputeKAL1(static_cast<int32_t>(K),
                                         static_cast<int32_t>(x.element_size()),
                                         ConvB1Cols(CoPad));
    const int32_t mBlocks = static_cast<int32_t>((mImg + CONV_BASE_M - 1) / CONV_BASE_M);
    const int32_t nBlocks = static_cast<int32_t>((CoPad + CONV_BASE_N - 1) / CONV_BASE_N);
    const int32_t totalBlocks = static_cast<int32_t>(batch * mBlocks * nBlocks);
    // core-count tuning (v10, stagger-aware): MIX kernel launch stagger is
    // ~0.2us/core on this platform, so fewer-but-balanced cores beat
    // max-cores-with-a-ragged-tail-wave. conv: waves = ceil(blocks/aic),
    // dim = ceil(blocks/waves) keeps every wave full (case 0: 25 blocks ->
    // 13 cores, 12.5 -> 10.9us measured).
    const int32_t convWaves = (totalBlocks + aicNum - 1) / aicNum;
    const uint32_t convDim = static_cast<uint32_t>(
        (totalBlocks + convWaves - 1) / convWaves);
    uint32_t convDimEnv = convDim;
    if (const char *e = getenv("CONV2D_CONV_DIM")) { convDimEnv = static_cast<uint32_t>(atoi(e)); }

    const int32_t ciBlocks = CiPad / 16;
    const int32_t coBlocks = CoPad / 16;
    const int32_t wTiles = ciBlocks * coBlocks;
    const uint32_t wDim = static_cast<uint32_t>(wTiles < aivNum ? wTiles : aivNum);

    const int32_t woBlocks = static_cast<int32_t>((Wo + 15) / 16);
    const int32_t i2cTiles = static_cast<int32_t>(batch * Ho * woBlocks);
    // stagger-aware cap: ~2 tiles/core for small grids, <= 24 cores for
    // large ones (case 0: 224 tiles -> 24 cores, 42.0 -> 40.8us; case 9:
    // 14 tiles -> 7 cores, 9.1 -> 8.3us measured)
    uint32_t i2cDim;
    if (i2cTiles <= 32) {
        i2cDim = static_cast<uint32_t>((i2cTiles + 1) / 2);
    } else {
        i2cDim = static_cast<uint32_t>(i2cTiles < 24 ? i2cTiles : 24);
    }
    if (const char *e = getenv("CONV2D_I2C_DIM")) { i2cDim = static_cast<uint32_t>(atoi(e)); }

    int32_t rowsPerTile;
    int64_t epiTiles;
    if (nhwc) {
        rowsPerTile = 8192 / CoPad;
        rowsPerTile = rowsPerTile < 16 ? 16 : rowsPerTile;
        epiTiles = (totalRows + rowsPerTile - 1) / rowsPerTile;
    } else {
        // P2: big NCHW epilogue tiles, [rowsPerTile rows][coC co] per tile,
        // ~16K fp32 acc elements per tile (64KB UB for the acc buffer)
        const int32_t coC = CoPad < 64 ? CoPad : 64;
        rowsPerTile = 16384 / coC;
        rowsPerTile = rowsPerTile > 1024 ? 1024 : rowsPerTile;
        rowsPerTile = rowsPerTile / 16 * 16;
        const int64_t coChunks = CoPad / coC;
        const int64_t mChunks = (mImg + rowsPerTile - 1) / rowsPerTile;
        epiTiles = batch * mChunks * coChunks;
    }
    // stagger-aware: ~2 tiles/core (case 0: 13 tiles -> 7 cores,
    // 6.1 -> 4.7us; case 9: 8 tiles -> 4 cores, 5.6 -> 5.1us measured)
    int64_t epiDimCap = epiTiles < 2 ? 1 : (epiTiles + 1) / 2;
    if (epiDimCap > aivNum) { epiDimCap = aivNum; }
    const uint32_t epiDim = static_cast<uint32_t>(epiDimCap);
    uint32_t epiDimEnv = epiDim;
    if (const char *e = getenv("CONV2D_EPI_DIM")) { epiDimEnv = static_cast<uint32_t>(atoi(e)); }

    // ---- outputs & workspaces (raw allocations only) ----
    at::Tensor y = nhwc ? at::empty({batch, Ho, Wo, Co}, x.options())
                        : at::empty({batch, Co, Ho, Wo}, x.options());
    const int64_t es = x.element_size();

    // P5 fused img2col: the conv kernel expands the feature map on the fly
    // via the img2col LoadData3D from an L1-resident NC1HWC0 copy (xT),
    // so the im2col GM materialization stage is skipped entirely. Gates:
    //   - Ci % 16 == 0 (img2col channelSize + xtrans fractal constraint)
    //   - NCHW needs H*W even (xtrans 32B-row constraint); NHWC needs no
    //     xtrans at all (x is already [H*W, Ci] channel-inner)
    //   - xNz (Ci*H*W) + resident K-major B panel (K*b1Cols) fit L1
    // Disable with CONV2D_FUSED=0.
    const int64_t xNzBytes = Ci * H * W * es;
    const int64_t bResBytes = K * ConvB1Cols(CoPad) * es;
    bool fusedOk = (!direct1x1) && (Ci % 16 == 0) &&
                   (nhwc || ((H * W) % 2 == 0)) &&
                   (xNzBytes + bResBytes <= 500 * 1024);
    if (const char *e = getenv("CONV2D_FUSED")) { if (atoi(e) == 0) { fusedOk = false; } }

    // P1-a: reuse the cached K-major weight workspace when the weight tensor
    // is unchanged; on a hit the wrepack kernel launch below is skipped.
    // (P0-b direct 1x1 mode uses the raw weight as GEMM A and skips this.)
    const uint32_t wVersion =
        weight.unsafeGetTensorImpl()->version_counter().current_version();
    const int32_t fusedVal = fusedOk ? 1 : 0;
    at::Tensor wWs;
    bool wCacheHit = false;
    if (!direct1x1 && wrepackCacheEnabled()) {
        std::lock_guard<std::mutex> lock(g_wrepackCacheMutex);
        for (auto it = g_wrepackCache.begin(); it != g_wrepackCache.end(); ++it) {
            if (it->weight.data_ptr() == weight.data_ptr() &&
                it->dtype == weight.scalar_type() &&
                it->version == wVersion &&
                it->sizes == weight.sizes().vec() &&
                it->fused == fusedVal) {
                g_wrepackCache.splice(g_wrepackCache.begin(), g_wrepackCache, it);
                wWs = it->wWs;
                wCacheHit = true;
                break;
            }
        }
    }
    if (!wCacheHit && !direct1x1) {
        wWs = at::empty({K * CoPad * es},
                        at::device(at::kPrivateUse1).dtype(at::kByte));
    }
    at::Tensor im2colWs;
    if (!direct1x1 && !fusedOk) {
        im2colWs = at::empty({totalRows * K * es},
                             at::device(at::kPrivateUse1).dtype(at::kByte));
    }
    at::Tensor accWs = at::empty({totalRows * CoPad * static_cast<int64_t>(sizeof(float))},
                                 at::device(at::kPrivateUse1).dtype(at::kByte));

    at::Tensor biasArg = hasBias ? *bias : weight;  // dummy ptr when no bias
    const int32_t hasBiasVal = hasBias ? 1 : 0;
    int32_t isNCHW = nhwc ? 0 : 1;
    if (const char *e = getenv("CONV2D_I2C_DBG")) { if (!nhwc) { isNCHW = 1 + atoi(e); } }

    // named scalars: EXEC_KERNEL_CMD binds args by lvalue ref, no temporaries
    int32_t batchVal = static_cast<int32_t>(batch);
    int32_t hVal = static_cast<int32_t>(H);
    int32_t wVal = static_cast<int32_t>(W);
    int32_t ciVal = static_cast<int32_t>(Ci);
    int32_t ciPadVal = CiPad;
    int32_t coVal = static_cast<int32_t>(Co);
    int32_t coPadVal = CoPad;
    int32_t khVal = static_cast<int32_t>(kH);
    int32_t kwVal = static_cast<int32_t>(kW);
    int32_t kkVal = static_cast<int32_t>(kH * kW);
    int32_t hoVal = static_cast<int32_t>(Ho);
    int32_t woVal = static_cast<int32_t>(Wo);
    int32_t shVal = static_cast<int32_t>(sH);
    int32_t swVal = static_cast<int32_t>(sW);
    int32_t dhVal = static_cast<int32_t>(dH);
    int32_t dwVal = static_cast<int32_t>(dW);
    int32_t phVal = static_cast<int32_t>(pH);
    int32_t pwVal = static_cast<int32_t>(pW);
    int32_t kVal = static_cast<int32_t>(K);
    int32_t kAL1Val = kAL1;
    int32_t mBlocksVal = mBlocks;
    int32_t nBlocksVal = nBlocks;
    int32_t totalBlocksVal = totalBlocks;
    int32_t mImgVal = static_cast<int32_t>(mImg);
    int32_t totalRowsVal = static_cast<int32_t>(totalRows);
    int32_t woBlocksVal = woBlocks;
    int32_t i2cTilesVal = i2cTiles;
    int32_t rowsPerTileVal = rowsPerTile;

    int stage = 4;
    if (const char *e = getenv("CONV2D_STAGE")) { stage = atoi(e); }

    // P0-b direct 1x1 fast path: flipped GEMM, no wrepack / im2col launches
    if (direct1x1) {
        // P7: pad H*W to a multiple of 16 when misaligned (UB-relay xpad)
        const int64_t hwPad = (mImg + 15) / 16 * 16;
        const bool needXpad = (hwPad != mImg);
        at::Tensor xPadWs;
        at::Tensor xSrc = x;
        int32_t xRowsD = static_cast<int32_t>(batch * Ci);
        int32_t hwD = static_cast<int32_t>(mImg);
        int32_t hwPadD = static_cast<int32_t>(hwPad);
        if (needXpad) {
            xPadWs = at::empty({batch * Ci * hwPad * es},
                               at::device(at::kPrivateUse1).dtype(at::kByte));
            xSrc = xPadWs;
        }
        // acc holds [batch*Co, hwPad] fp32 rows in direct mode
        const int64_t accDirectBytes = batch * Co * hwPad * 4;
        if (accDirectBytes > totalRows * CoPad * 4) {
            accWs = at::empty({accDirectBytes},
                              at::device(at::kPrivateUse1).dtype(at::kByte));
        }
        int32_t mGemmD = static_cast<int32_t>(Co);
        int32_t nGemmD = hwPadD;
        int32_t kGemmD = static_cast<int32_t>(Ci);
        int32_t kAL1D = ConvComputeKAL1(kGemmD, static_cast<int32_t>(es),
                                        ConvB1Cols(nGemmD >= 64 ? nGemmD : 16));
        int32_t mBlocksD = (mGemmD + CONV_BASE_M - 1) / CONV_BASE_M;
        int32_t nBlocksD = (nGemmD + CONV_BASE_N - 1) / CONV_BASE_N;
        int32_t totalBlocksD = static_cast<int32_t>(batch) * mBlocksD * nBlocksD;
        uint32_t convDimD = static_cast<uint32_t>(
            totalBlocksD < aicNum ? totalBlocksD : aicNum);
        uint32_t xpadDim = static_cast<uint32_t>(
            (xRowsD + 63) / 64 < aivNum ? (xRowsD + 63) / 64 : aivNum);
        int32_t totalRowsD = static_cast<int32_t>(batch * Co);
        // stagger-aware: ~50K fp32 acc elements/core (case 2: 200704 ->
        // 4 cores, 15.4 -> 3.5us measured)
        int64_t epiElemsD = static_cast<int64_t>(totalRowsD) * mImg;
        uint32_t epiDimD = static_cast<uint32_t>(
            (epiElemsD + 49999) / 50000);
        if (epiDimD > static_cast<uint32_t>(aivNum)) { epiDimD = static_cast<uint32_t>(aivNum); }
        if (const char *e = getenv("CONV2D_EPI_DIM")) { epiDimD = static_cast<uint32_t>(atoi(e)); }
        int32_t rowsPerTileD = 1;
        int32_t directOn = 1;
        if (needXpad && getenv("CONV2D_XPAD_DBG")) {
            if (x.scalar_type() == at::kHalf) {
                EXEC_KERNEL_CMD(conv2d_base_xpad_fp16, xpadDim, x, xPadWs,
                                xRowsD, hwD, hwPadD);
            } else {
                EXEC_KERNEL_CMD(conv2d_base_xpad_bf16, xpadDim, x, xPadWs,
                                xRowsD, hwD, hwPadD);
            }
            auto xpView = xPadWs.view(x.scalar_type())
                              .view({batch, Ci, hwPad});
            y.copy_(xpView.narrow(2, 0, mImg).reshape({batch, Ci, H, W}));
            return y;
        }
        if (x.scalar_type() == at::kHalf) {
            if (needXpad) {
                EXEC_KERNEL_CMD(conv2d_base_xpad_fp16, xpadDim, x, xPadWs,
                                xRowsD, hwD, hwPadD);
            }
            EXEC_KERNEL_CMD(conv2d_base_conv_fp16, convDimD, weight, xSrc, accWs,
                            batchVal, mGemmD, nGemmD, kGemmD, kAL1D,
                            mBlocksD, nBlocksD, totalBlocksD, directOn,
                            hVal, wVal, ciVal, khVal, kwVal,
                            shVal, swVal, dhVal, dwVal, phVal, pwVal);
            EXEC_KERNEL_CMD(conv2d_base_epilogue_fp16, epiDimD, accWs, biasArg, y,
                            totalRowsD, hwPadD, coVal, hwD, rowsPerTileD,
                            hasBiasVal, isNCHW, directOn);
        } else {
            if (needXpad) {
                EXEC_KERNEL_CMD(conv2d_base_xpad_bf16, xpadDim, x, xPadWs,
                                xRowsD, hwD, hwPadD);
            }
            EXEC_KERNEL_CMD(conv2d_base_conv_bf16, convDimD, weight, xSrc, accWs,
                            batchVal, mGemmD, nGemmD, kGemmD, kAL1D,
                            mBlocksD, nBlocksD, totalBlocksD, directOn,
                            hVal, wVal, ciVal, khVal, kwVal,
                            shVal, swVal, dhVal, dwVal, phVal, pwVal);
            EXEC_KERNEL_CMD(conv2d_base_epilogue_bf16, epiDimD, accWs, biasArg, y,
                            totalRowsD, hwPadD, coVal, hwD, rowsPerTileD,
                            hasBiasVal, isNCHW, directOn);
        }
        return y;
    }

    int32_t directOff = 0;

    // P5 fused img2col path: [xtrans (NCHW only)] -> conv(direct=2) -> epilogue
    if (fusedOk) {
        at::Tensor xTWs;
        at::Tensor xT = x;
        if (!nhwc) {
            xTWs = at::empty({batch * Ci * H * W * es},
                             at::device(at::kPrivateUse1).dtype(at::kByte));
            xT = xTWs;
        }
        const int32_t xtT = [&] {
            int32_t t = 128;
            while (t > 16 && Ci * t * es * 2 > 48 * 1024) { t /= 2; }
            return t;
        }();
        const int32_t xtTiles = static_cast<int32_t>(batch) *
                                static_cast<int32_t>((H * W + xtT - 1) / xtT);
        uint32_t xtDim;
        if (xtTiles <= 32) {
            xtDim = static_cast<uint32_t>((xtTiles + 1) / 2);
        } else {
            xtDim = static_cast<uint32_t>(xtTiles < 24 ? xtTiles : 24);
        }
        if (const char *e = getenv("CONV2D_XT_DIM")) { xtDim = static_cast<uint32_t>(atoi(e)); }
        int32_t ciXt = static_cast<int32_t>(Ci);
        int32_t hwXt = static_cast<int32_t>(H * W);
        int32_t fusedOn = 2;
        if (x.scalar_type() == at::kHalf) {
            if (!wCacheHit) {
                EXEC_KERNEL_CMD(conv2d_base_wrepack_fp16, wDim, weight, wWs,
                                coVal, ciVal, kkVal, ciPadVal, coPadVal, fusedVal);
                wrepackCacheInsert(weight, wWs, wVersion, fusedVal);
            }
            if (!nhwc) {
                EXEC_KERNEL_CMD(conv2d_base_xtrans_fp16, xtDim, x, xTWs,
                                batchVal, ciXt, hwXt);
            }
            EXEC_KERNEL_CMD(conv2d_base_conv_fp16, convDimEnv, xT, wWs, accWs,
                            batchVal, mImgVal, coPadVal, kVal, kAL1Val,
                            mBlocksVal, nBlocksVal, totalBlocksVal, fusedOn,
                            hVal, wVal, ciVal, khVal, kwVal,
                            shVal, swVal, dhVal, dwVal, phVal, pwVal);
            EXEC_KERNEL_CMD(conv2d_base_epilogue_fp16, epiDimEnv, accWs, biasArg, y,
                            totalRowsVal, coPadVal, coVal, mImgVal, rowsPerTileVal,
                            hasBiasVal, isNCHW, directOff);
        } else {
            if (!wCacheHit) {
                EXEC_KERNEL_CMD(conv2d_base_wrepack_bf16, wDim, weight, wWs,
                                coVal, ciVal, kkVal, ciPadVal, coPadVal, fusedVal);
                wrepackCacheInsert(weight, wWs, wVersion, fusedVal);
            }
            if (!nhwc) {
                EXEC_KERNEL_CMD(conv2d_base_xtrans_bf16, xtDim, x, xTWs,
                                batchVal, ciXt, hwXt);
            }
            EXEC_KERNEL_CMD(conv2d_base_conv_bf16, convDimEnv, xT, wWs, accWs,
                            batchVal, mImgVal, coPadVal, kVal, kAL1Val,
                            mBlocksVal, nBlocksVal, totalBlocksVal, fusedOn,
                            hVal, wVal, ciVal, khVal, kwVal,
                            shVal, swVal, dhVal, dwVal, phVal, pwVal);
            EXEC_KERNEL_CMD(conv2d_base_epilogue_bf16, epiDimEnv, accWs, biasArg, y,
                            totalRowsVal, coPadVal, coVal, mImgVal, rowsPerTileVal,
                            hasBiasVal, isNCHW, directOff);
        }
        return y;
    }
    if (x.scalar_type() == at::kHalf) {
        if (!wCacheHit) {
            EXEC_KERNEL_CMD(conv2d_base_wrepack_fp16, wDim, weight, wWs,
                            coVal, ciVal, kkVal, ciPadVal, coPadVal, fusedVal);
            wrepackCacheInsert(weight, wWs, wVersion, fusedVal);
        }
        if (stage < 2) return y;
        EXEC_KERNEL_CMD(conv2d_base_im2col_fp16, i2cDim, x, im2colWs,
                        hVal, wVal, ciVal, ciPadVal, khVal, kwVal, hoVal, woVal,
                        shVal, swVal, dhVal, dwVal, phVal, pwVal,
                        kVal, mImgVal, woBlocksVal, i2cTilesVal, isNCHW);
        if (stage < 3) return y;
        EXEC_KERNEL_CMD(conv2d_base_conv_fp16, convDimEnv, im2colWs, wWs, accWs,
                        batchVal, mImgVal, coPadVal, kVal, kAL1Val,
                        mBlocksVal, nBlocksVal, totalBlocksVal, directOff,
                        hVal, wVal, ciVal, khVal, kwVal,
                        shVal, swVal, dhVal, dwVal, phVal, pwVal);
        if (stage < 4) return y;
        EXEC_KERNEL_CMD(conv2d_base_epilogue_fp16, epiDimEnv, accWs, biasArg, y,
                        totalRowsVal, coPadVal, coVal, mImgVal, rowsPerTileVal,
                        hasBiasVal, isNCHW, directOff);
    } else {
        if (!wCacheHit) {
            EXEC_KERNEL_CMD(conv2d_base_wrepack_bf16, wDim, weight, wWs,
                            coVal, ciVal, kkVal, ciPadVal, coPadVal, fusedVal);
            wrepackCacheInsert(weight, wWs, wVersion, fusedVal);
        }
        EXEC_KERNEL_CMD(conv2d_base_im2col_bf16, i2cDim, x, im2colWs,
                        hVal, wVal, ciVal, ciPadVal, khVal, kwVal, hoVal, woVal,
                        shVal, swVal, dhVal, dwVal, phVal, pwVal,
                        kVal, mImgVal, woBlocksVal, i2cTilesVal, isNCHW);
        EXEC_KERNEL_CMD(conv2d_base_conv_bf16, convDimEnv, im2colWs, wWs, accWs,
                        batchVal, mImgVal, coPadVal, kVal, kAL1Val,
                        mBlocksVal, nBlocksVal, totalBlocksVal, directOff,
                        hVal, wVal, ciVal, khVal, kwVal,
                        shVal, swVal, dhVal, dwVal, phVal, pwVal);
        EXEC_KERNEL_CMD(conv2d_base_epilogue_bf16, epiDimEnv, accWs, biasArg, y,
                        totalRowsVal, coPadVal, coVal, mImgVal, rowsPerTileVal,
                        hasBiasVal, isNCHW, directOff);
    }
    return y;
}

} // namespace ascend_kernel
