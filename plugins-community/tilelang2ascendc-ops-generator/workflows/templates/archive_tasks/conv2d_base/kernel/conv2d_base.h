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
 * @file conv2d_base.h
 * Conv2dBase main-flow kernel for arch20 (910B), AIC side.
 *
 * Input contract (produced by host + stage-0 im2col kernel):
 *   im2col : [totalRows, K] ND fp16/bf16, K = Ci*kH*kW (k order = (kh,kw,ci))
 *   w      : [K, Co] ND K-major, Co % 16 == 0
 *   acc    : [totalRows, Co] fp32 workspace (NHWC output rows, no bias)
 *
 * Data flow: GM --Nd2Nz--> L1 --LoadData--> L0A/L0B --Mmad(fp32)--> L0C
 *            --Fixpipe--> GM   (identical to the proven 8_WQB matmul path;
 *            plain 2D DataCopy GM<->L1 proved unreliable on this platform,
 *            so the im2col matrix is materialized in GM by the stage-0 kernel)
 *
 * Multi-core decomposition: blocks = batch x mBlocks x nTiles, strided over
 * AIC cores. K fullload in L1 when it fits the budget, else split into
 * kAL1 panels with L0C accumulation across panels. m/n tails: L0 computes
 * full baseM/baseN tiles on padded/stale data, Fixpipe writes only the
 * actual m/n extent (stale rows/cols are discarded).
 *
 * P3 history (v7): (a) n-adaptive L0B load + Mmad kept — nEff = nActual
 * when nActual >= 64 (halves cube work for Co=64; perf-neutral but
 * principled). nActual < 64 keeps the proven full-baseN path (nActual=16
 * triggers an L0C r/w conflict on dav-c220, empirically). m-side Mmad
 * reduction produced garbage on tail blocks and was reverted.
 * (b) baseK/kChunk=128 chunk-pipelined K loop (depth-2 A1/B1 queues) was
 * tried and REVERTED: per-chunk Mmad (~0.3us) is far smaller than the
 * chunk load (~1us), so the pipeline degenerates to load-serial, and the
 * narrower GM rows (128B/256B vs 1152B monolithic) hurt DRAM efficiency —
 * case 0 conv regressed 13.1 -> 15.1/14.6us. The kernel is load-latency
 * + wave-quantization bound, not Mmad-bound.
 *
 * P5 (fused img2col, mode 2): the A side is expanded on the fly by the
 * img2col LoadData3D from an L1-resident NC1HWC0 feature map (xT GM
 * [H*W, Ci] produced by the xtrans AIV kernel for NCHW, or the raw x for
 * NHWC); B is the K-major weight ws (resident per nIdx). No im2col GM
 * matrix is materialized. Measured hardware k order of the img2col
 * expansion is (ci1, kh, kw, ci0) — C0-channel-block OUTERMOST — so the
 * fused weight ws is repacked in that order (wrepack fused=1); the two
 * orders coincide exactly when Ci == 16 or kH*kW == 1.
 */

#ifndef CONV2D_BASE_H
#define CONV2D_BASE_H

#include "kernel_operator.h"
#include "conv2d_base_tile.h"
#include "conv2d_base_tiling.h"

using namespace AscendC;

template <typename T>
class Conv2dBaseKernel {
    static constexpr uint32_t baseM = CONV_BASE_M;
    static constexpr uint32_t baseN = CONV_BASE_N;
    static constexpr uint32_t baseK = CONV_BASE_K;

public:
    __aicore__ inline Conv2dBaseKernel() {}

    __aicore__ inline void Init(GM_ADDR im2col, GM_ADDR w, GM_ADDR acc,
                                int32_t batch, int32_t mImg, int32_t Co,
                                int32_t K, int32_t kAL1,
                                int32_t mBlocks, int32_t nBlocks, int32_t totalBlocks,
                                int32_t direct,
                                int32_t H, int32_t W, int32_t Ci,
                                int32_t kH, int32_t kW,
                                int32_t sH, int32_t sW, int32_t dH, int32_t dW,
                                int32_t pH, int32_t pW,
                                AscendC::TPipe *pipe)
    {
        mImg_ = mImg;
        Co_ = Co;
        K_ = K;
        kAL1_ = kAL1;
        mBlocks_ = mBlocks;
        nBlocks_ = nBlocks;
        totalBlocks_ = totalBlocks;
        mode_ = direct;  // 0: normal, 1: direct 1x1, 2: P5 fused img2col
        H_ = H;
        W_ = W;
        Ci_ = Ci;
        kH_ = kH;
        kW_ = kW;
        sH_ = sH;
        sW_ = sW;
        dH_ = dH;
        dW_ = dW;
        pH_ = pH;
        pW_ = pW;
        // P0-b direct 1x1 mode: A = raw weight [Co, Ci] (shared across
        // images), B = raw x NCHW [Ci, H*W] (per image). Normal mode:
        // A = im2col [rows, K] (per image), B = K-major weight (shared).
        // P5 fused mode: A = NC1HWC0 feature map in L1 expanded on the fly
        // by the img2col LoadData3D (xT loaded per image with an explicit
        // offset), B = K-major weight (shared, resident per nIdx).
        aImgStride_ = direct ? 0 : static_cast<int64_t>(mImg_) * K_;
        bImgStride_ = (direct == 1) ? static_cast<int64_t>(K_) * Co_ : 0;

        im2colGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(im2col),
                                  static_cast<int64_t>(batch) * mImg_ * K_);
        wGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(w),
                             bImgStride_ > 0 ? bImgStride_ * batch
                                             : static_cast<int64_t>(K_) * Co_);
        accGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(acc),
                               static_cast<int64_t>(batch) * mImg_ * Co_);

        // P6: depth-2 A1/B1 queues for cross-block/panel prefetch. The B
        // slot uses b1Cols_ <= baseN columns (the NZ panel only ever holds
        // Co_ columns when Co_ >= 64), which is what lets two case-0
        // panels (A1 147KB + B1 73KB) fit L1.
        b1Cols_ = Co_ < 64 ? static_cast<int32_t>(baseN)
                           : (Co_ < static_cast<int32_t>(baseN)
                                  ? Co_ : static_cast<int32_t>(baseN));
        if (mode_ == 2) {
            // P5 fused: xNz (NC1HWC0 feature map, per image, depth-1) and a
            // resident K-major B panel (per nIdx, depth-1) share L1; the
            // prefetch A1/B1 queues are unused. A tiles are produced by the
            // img2col LoadData3D straight into L0A (no im2col GM matrix).
            pipe->InitBuffer(xNzQue_, 1,
                             static_cast<uint32_t>(Ci_) * H_ * W_ * sizeof(T));
            pipe->InitBuffer(bResQue_, 1,
                             static_cast<uint32_t>(K_) * b1Cols_ * sizeof(T));
            pipe->InitBuffer(inQueueA1_, 2, 64);
            pipe->InitBuffer(inQueueB1_, 2, 64);
        } else {
            pipe->InitBuffer(xNzQue_, 1, 64);
            pipe->InitBuffer(inQueueA1_, 2, baseM * kAL1_ * sizeof(T));
            pipe->InitBuffer(inQueueB1_, 2, static_cast<uint32_t>(kAL1_) * b1Cols_ * sizeof(T));
        }
        pipe->InitBuffer(inQueueA2_, 2, baseM * baseK * sizeof(T));
        pipe->InitBuffer(inQueueB2_, 2, baseK * baseN * sizeof(T));
        pipe->InitBuffer(outQueueCO1_, 1, baseM * baseN * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (mode_ == 2) {
            ProcessFused();
            return;
        }
        int32_t numCores = AscendC::GetBlockNum();
        int32_t coreIdx = AscendC::GetBlockIdx();
        int32_t kPanels = (K_ + kAL1_ - 1) / kAL1_;

        // P6: prime the pipeline with the first (blk, panel) pair
        if (coreIdx < totalBlocks_) {
            Prefetch(coreIdx, 0);
        }
        for (int32_t blk = coreIdx; blk < totalBlocks_; blk += numCores) {
            int32_t img = blk / (mBlocks_ * nBlocks_);
            int32_t rem = blk % (mBlocks_ * nBlocks_);
            int32_t mIdx = rem / nBlocks_;
            int32_t nIdx = rem % nBlocks_;
            ComputeBlock(img, mIdx, nIdx, blk, numCores, kPanels);
        }
    }

private:
    // issue the GM->L1 loads for one (blk, panel) pair into a free queue slot
    __aicore__ inline void Prefetch(int32_t blk, int32_t kp)
    {
        int32_t img = blk / (mBlocks_ * nBlocks_);
        int32_t rem = blk % (mBlocks_ * nBlocks_);
        int32_t mIdx = rem / nBlocks_;
        int32_t nIdx = rem % nBlocks_;
        uint32_t m0 = mIdx * baseM;
        uint32_t n0 = nIdx * baseN;
        uint32_t mActual = mImg_ - m0;
        mActual = mActual > baseM ? baseM : mActual;
        uint32_t nActual = Co_ - n0;
        nActual = nActual > baseN ? baseN : nActual;
        int32_t k0 = kp * kAL1_;
        uint32_t kLen = K_ - k0;
        kLen = kLen > (uint32_t)kAL1_ ? (uint32_t)kAL1_ : kLen;
        LoadA(img, m0, mActual, k0, kLen);
        LoadB(img, n0, k0, kLen, nActual);
    }

    __aicore__ inline void ComputeBlock(int32_t img, int32_t mIdx, int32_t nIdx,
                                        int32_t blk, int32_t numCores,
                                        int32_t kPanels)
    {
        uint32_t m0 = mIdx * baseM;
        uint32_t n0 = nIdx * baseN;
        uint32_t mActual = mImg_ - m0;
        mActual = mActual > baseM ? baseM : mActual;
        uint32_t nActual = Co_ - n0;
        nActual = nActual > baseN ? baseN : nActual;
        // P3-a: n-adaptive L0B load + Mmad (guarded; see header comment).
        // round16: P0-b direct mode's spatial n-tail can be non-16-aligned
        // (e.g. H*W=196 -> 68); a non-16 channelSize skips the partial
        // fractal block in the Zn LoadData3D and corrupts those columns.
        uint32_t nEff = (nActual + 15) / 16 * 16;
        nEff = nEff >= 64 ? nEff : baseN;

        AscendC::LocalTensor<float> c1Local = outQueueCO1_.AllocTensor<float>();
        bool cmatrixInitVal = true;

        for (int32_t kp = 0; kp < kPanels; ++kp) {
            int32_t k0 = kp * kAL1_;
            uint32_t kLen = K_ - k0;
            kLen = kLen > (uint32_t)kAL1_ ? (uint32_t)kAL1_ : kLen;

            // P6: prefetch the next (blk, panel) pair while this one computes
            if (kp + 1 < kPanels) {
                Prefetch(blk, kp + 1);
            } else if (blk + numCores < totalBlocks_) {
                Prefetch(blk + numCores, 0);
            }

            AscendC::LocalTensor<T> a1Local = inQueueA1_.DeQue<T>();
            AscendC::LocalTensor<T> b1Local = inQueueB1_.DeQue<T>();

            uint32_t kTiles = (kLen + baseK - 1) / baseK;
            for (uint32_t kt = 0; kt < kTiles; ++kt) {
                uint32_t kAct = kLen - kt * baseK;
                kAct = kAct > baseK ? baseK : kAct;
                SplitA(a1Local, kt * baseM * baseK, kAct);
                SplitB(b1Local, kt * baseK * CONV_C0, kAct, kLen, nEff);
                Compute(c1Local, kAct, nEff, cmatrixInitVal);
                cmatrixInitVal = false;
            }

            inQueueA1_.FreeTensor(a1Local);
            inQueueB1_.FreeTensor(b1Local);
        }

        outQueueCO1_.EnQue(c1Local);
        CopyOut(img, m0, mActual, n0, nActual);
    }

    // ------------------------------------------------------------------
    // P5 fused mode: SAME GEMM orientation as the normal path
    //   A = im2col expansion of x, produced on the fly by the img2col
    //       LoadData3D (pad/stride/dilation in hw) from the L1-resident
    //       NC1HWC0 feature map (no im2col GM materialization)
    //   B = K-major weight ws (identical to the normal path, resident per
    //       nIdx), C = [Ho*Wo, Co] fp32 -> the normal epilogue
    // The m/n decomposition, CopyOut and epilogue are unchanged.
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessFused()
    {
        int32_t numCores = AscendC::GetBlockNum();
        int32_t coreIdx = AscendC::GetBlockIdx();
        int32_t curImg = -1;
        int32_t curNIdx = -1;
        AscendC::LocalTensor<T> xNzLocal;
        AscendC::LocalTensor<T> bLocal;

        for (int32_t blk = coreIdx; blk < totalBlocks_; blk += numCores) {
            int32_t img = blk / (mBlocks_ * nBlocks_);
            int32_t rem = blk % (mBlocks_ * nBlocks_);
            int32_t mIdx = rem / nBlocks_;
            int32_t nIdx = rem % nBlocks_;

            if (img != curImg) {
                if (curImg >= 0) {
                    xNzQue_.FreeTensor(xNzLocal);
                }
                xNzLocal = xNzQue_.AllocTensor<T>();
                // xT GM [img, H*W, Ci] -> L1 NC1HWC0 [C1][H*W][C0]
                LoadNdGmToNzL1(xNzLocal,
                               im2colGM_[static_cast<int64_t>(img) * H_ * W_ * Ci_],
                               static_cast<uint32_t>(H_ * W_),
                               static_cast<uint32_t>(Ci_),
                               static_cast<uint32_t>(Ci_),
                               static_cast<uint32_t>(H_ * W_));
                xNzQue_.EnQue(xNzLocal);
                xNzLocal = xNzQue_.DeQue<T>();
                curImg = img;
            }
            if (nIdx != curNIdx) {
                if (curNIdx >= 0) {
                    bResQue_.FreeTensor(bLocal);
                }
                uint32_t n0 = nIdx * baseN;
                uint32_t nActual = Co_ - n0;
                nActual = nActual > baseN ? baseN : nActual;
                bLocal = bResQue_.AllocTensor<T>();
                // resident K-major B panel [K, nActual] (shared across imgs)
                LoadNdGmToNzL1(bLocal, wGM_[n0],
                               static_cast<uint32_t>(K_), nActual,
                               static_cast<uint32_t>(Co_),
                               static_cast<uint32_t>(K_));
                bResQue_.EnQue(bLocal);
                bLocal = bResQue_.DeQue<T>();
                curNIdx = nIdx;
            }
            ComputeBlockFused(img, mIdx, nIdx, xNzLocal, bLocal);
        }
        if (curImg >= 0) {
            xNzQue_.FreeTensor(xNzLocal);
        }
        if (curNIdx >= 0) {
            bResQue_.FreeTensor(bLocal);
        }
    }

    __aicore__ inline void ComputeBlockFused(
        int32_t img, int32_t mIdx, int32_t nIdx,
        const AscendC::LocalTensor<T> &xNzLocal,
        const AscendC::LocalTensor<T> &bLocal)
    {
        uint32_t m0 = mIdx * baseM;
        uint32_t n0 = nIdx * baseN;
        uint32_t mActual = mImg_ - m0;
        mActual = mActual > baseM ? baseM : mActual;
        uint32_t nActual = Co_ - n0;
        nActual = nActual > baseN ? baseN : nActual;
        uint32_t nEff = (nActual + 15) / 16 * 16;
        nEff = nEff >= 64 ? nEff : baseN;

        AscendC::LocalTensor<float> c1Local = outQueueCO1_.AllocTensor<float>();
        bool cmatrixInitVal = true;

        uint32_t kTiles = (K_ + baseK - 1) / baseK;
        for (uint32_t kt = 0; kt < kTiles; ++kt) {
            uint32_t kStart = kt * baseK;
            uint32_t kAct = K_ - kStart;
            kAct = kAct > baseK ? baseK : kAct;
            LoadA3D(xNzLocal, kStart, kAct, m0, mActual);
            SplitB(bLocal, kt * baseK * CONV_C0, kAct, K_, nEff);
            Compute(c1Local, kAct, nEff, cmatrixInitVal);
            cmatrixInitVal = false;
        }

        outQueueCO1_.EnQue(c1Local);
        CopyOut(img, m0, mActual, n0, nActual);
    }

    // img2col LoadData3D: expand the NC1HWC0 feature map on the fly and
    // pick the [mActual x kAct] block at (m0, kStart) into an L0A Zz tile
    __aicore__ inline void LoadA3D(const AscendC::LocalTensor<T> &xNzLocal,
                                   uint32_t kStart, uint32_t kAct,
                                   uint32_t m0, uint32_t mActual)
    {
        AscendC::LocalTensor<T> a2Local = inQueueA2_.AllocTensor<T>();
        AscendC::LoadData3DParamsV2<T> params;
        params.l1H = H_;
        params.l1W = W_;
        params.channelSize = Ci_;
        params.padList[0] = pW_;
        params.padList[1] = pW_;
        params.padList[2] = pH_;
        params.padList[3] = pH_;
        params.strideH = sH_;
        params.strideW = sW_;
        params.filterH = kH_;
        params.filterW = kW_;
        params.dilationFilterH = dH_;
        params.dilationFilterW = dW_;
        params.kStartPt = kStart;
        params.kExtension = kAct;
        params.mStartPt = m0;
        params.mExtension = mActual;
        params.padValue = static_cast<T>(0);
        AscendC::LoadData(a2Local, xNzLocal, params);
        inQueueA2_.EnQue(a2Local);
    }

    // A panel [mActual, kLen] of im2col: GM ND -> L1 NZ (fractal stride baseM)
    __aicore__ inline void LoadA(int32_t img, uint32_t m0, uint32_t mActual,
                                 int32_t k0, uint32_t kLen)
    {
        AscendC::LocalTensor<T> a1Local = inQueueA1_.AllocTensor<T>();
        LoadNdGmToNzL1(a1Local,
                       im2colGM_[img * aImgStride_ +
                                 static_cast<int64_t>(m0) * K_ + k0],
                       mActual, kLen, K_, baseM);
        inQueueA1_.EnQue(a1Local);
    }

    // B panel [kLen, nActual] of K-major weight: GM ND -> L1 NZ (stride kLen)
    __aicore__ inline void LoadB(int32_t img, uint32_t n0, int32_t k0,
                                 uint32_t kLen, uint32_t nActual)
    {
        AscendC::LocalTensor<T> b1Local = inQueueB1_.AllocTensor<T>();
        LoadNdGmToNzL1(b1Local,
                       wGM_[img * bImgStride_ + static_cast<int64_t>(k0) * Co_ + n0],
                       kLen, nActual, Co_, kLen);
        inQueueB1_.EnQue(b1Local);
    }

    __aicore__ inline void SplitA(const AscendC::LocalTensor<T> &a1Local,
                                  uint32_t offset, uint32_t kAct)
    {
        AscendC::LocalTensor<T> a2Local = inQueueA2_.AllocTensor<T>();
        LoadNzL1ToZzL0A(a2Local, a1Local[offset], baseM, kAct, baseM);
        inQueueA2_.EnQue(a2Local);
    }

    __aicore__ inline void SplitB(const AscendC::LocalTensor<T> &b1Local,
                                  uint32_t offset, uint32_t kAct, uint32_t kLen,
                                  uint32_t nEff)
    {
        AscendC::LocalTensor<T> b2Local = inQueueB2_.AllocTensor<T>();
        LoadNzL1ToZnL0B(b2Local, b1Local[offset], kAct, nEff, kLen);
        inQueueB2_.EnQue(b2Local);
    }

    __aicore__ inline void Compute(const AscendC::LocalTensor<float> &c1Local,
                                   uint32_t kAct, uint32_t nEff,
                                   bool cmatrixInitVal)
    {
        AscendC::LocalTensor<T> a2Local = inQueueA2_.DeQue<T>();
        AscendC::LocalTensor<T> b2Local = inQueueB2_.DeQue<T>();
        AscendC::MmadParams mmadParams;
        mmadParams.m = baseM;
        mmadParams.n = nEff;
        mmadParams.k = kAct;
        mmadParams.cmatrixInitVal = cmatrixInitVal;
        AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
        inQueueA2_.FreeTensor(a2Local);
        inQueueB2_.FreeTensor(b2Local);
    }

    // L0C(fp32, NZ) -> GM acc rows (NHWC output rows, bias added by epilogue)
    __aicore__ inline void CopyOut(int32_t img, uint32_t m0, uint32_t mActual,
                                   uint32_t n0, uint32_t nActual)
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1_.DeQue<float>();
        AscendC::FixpipeParamsV220 params;
        params.nSize = nActual;
        params.mSize = mActual;
        params.srcStride = baseM;
        params.dstStride = Co_;
        params.ndNum = 1;
        params.srcNdStride = 0;
        params.dstNdStride = 0;
        AscendC::Fixpipe(accGM_[(static_cast<int64_t>(img) * mImg_ + m0) * Co_ + n0],
                         c1Local, params);
        outQueueCO1_.FreeTensor(c1Local);
    }

    int32_t mImg_;
    int32_t Co_;
    int32_t K_;
    int32_t kAL1_;
    int32_t mBlocks_;
    int32_t nBlocks_;
    int32_t totalBlocks_;
    int64_t aImgStride_;
    int64_t bImgStride_;
    int32_t b1Cols_;
    int32_t mode_ = 0;
    int32_t H_ = 0;
    int32_t W_ = 0;
    int32_t Ci_ = 0;
    int32_t kH_ = 0;
    int32_t kW_ = 0;
    int32_t sH_ = 0;
    int32_t sW_ = 0;
    int32_t dH_ = 0;
    int32_t dW_ = 0;
    int32_t pH_ = 0;
    int32_t pW_ = 0;

    AscendC::GlobalTensor<T> im2colGM_;
    AscendC::GlobalTensor<T> wGM_;
    AscendC::GlobalTensor<float> accGM_;

    AscendC::TQue<AscendC::TPosition::A1, 2> inQueueA1_;
    AscendC::TQue<AscendC::TPosition::A2, 2> inQueueA2_;
    AscendC::TQue<AscendC::TPosition::B1, 2> inQueueB1_;
    AscendC::TQue<AscendC::TPosition::A1, 1> xNzQue_;
    AscendC::TQue<AscendC::TPosition::B1, 1> bResQue_;
    AscendC::TQue<AscendC::TPosition::B2, 2> inQueueB2_;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1_;
};

#endif // CONV2D_BASE_H
