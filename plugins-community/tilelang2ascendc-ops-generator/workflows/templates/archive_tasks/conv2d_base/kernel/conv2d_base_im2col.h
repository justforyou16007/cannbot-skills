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
 * @file conv2d_base_im2col.h
 * Stage-0x (AIV-only) fused im2col for conv2d_base (v2, host-zero-prep).
 *   im2col[row, k] = x[img, hi, wi, ci], hi = ho*sH + kh*dH - pH,
 *   wi = wo*sW + kw*dW - pW (out-of-range -> 0), k order = (kh, kw, ci),
 *   K = kH*kW*CiPad, CiPad = ceil16(Ci).
 * Consumes the RAW input layout (no host permute/pad).
 *
 * Two implementations coexist, selected in Init from shape-derived UB budget:
 *
 * FAST PATH (v5, P0): bulk data movement.
 *   - NCHW load: one 2D DataCopyPad per (kh, ci-block) covering all 16 ci
 *     rows (srcStride = H*W*es), instead of 16 single-row copies. Requires
 *     uniform 2B-realign shift across ci rows -> H*W must be even; large
 *     left/right clamps fall back to per-row loads.
 *   - transpose: per (kh, ci-block) TransDataTo5HD into a full-width
 *     transAll [kH*wWinPad][CiPad] buffer.
 *   - NHWC load: whole-row slab per kh goes straight into transAll.
 *   - store: per (kh, kw) one 2D DataCopy [mCnt][CiPad] (DIRECT mode,
 *     CiPad >= 64), or per (kh, kw) one V-pipe Copy into a rowbuf
 *     [16][K] followed by ONE contiguous 2D store (ROWBUF mode, CiPad < 64).
 *     MTE2/MTE3 call counts drop by 1-2 orders of magnitude vs the legacy
 *     per-(m, ci-block) 32B chunk stores.
 *
 * LEGACY PATH: the proven small-tile implementation (per-row DataCopyPad
 * loads, per-(m, ci-block) 32B chunk stores), kept for shapes whose
 * transAll/rowbuf buffers would exceed the UB budget or whose H*W parity
 * breaks the merged-load shift assumption.
 *
 * Grid: (img, ho, woBlock) tiles of up to 16 consecutive wo.
 * All sync uses explicit SetFlag/WaitFlag (MTE2->V->MTE3 chain).
 */

#ifndef CONV2D_BASE_IM2COL_H
#define CONV2D_BASE_IM2COL_H

#include "kernel_operator.h"
#include "conv2d_base_tiling.h"

using namespace AscendC;

template <typename T>
class Conv2dIm2colKernel {
    static constexpr uint32_t C0 = CONV_C0;
    static constexpr uint32_t MTILE = 16;
    // UB budget for the fast path (192KB total; leave headroom)
    static constexpr int64_t FAST_UB_BUDGET = 160 * 1024;

public:
    __aicore__ inline Conv2dIm2colKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR im2col,
                                int32_t H, int32_t W, int32_t Ci, int32_t CiPad,
                                int32_t kH, int32_t kW, int32_t Ho, int32_t Wo,
                                int32_t strideH, int32_t strideW,
                                int32_t dilH, int32_t dilW,
                                int32_t padH, int32_t padW,
                                int32_t K, int32_t mImg, int32_t woBlocks,
                                int32_t totalTiles, int32_t isNCHW,
                                AscendC::TPipe *pipe)
    {
        H_ = H;
        W_ = W;
        Ci_ = Ci;
        CiPad_ = CiPad;
        kH_ = kH;
        kW_ = kW;
        Ho_ = Ho;
        Wo_ = Wo;
        sH_ = strideH;
        sW_ = strideW;
        dH_ = dilH;
        dW_ = dilW;
        pH_ = padH;
        pW_ = padW;
        K_ = K;
        mImg_ = mImg;
        woBlocks_ = woBlocks;
        totalTiles_ = totalTiles;
        isNCHW_ = isNCHW;
        ciBlocks_ = CiPad_ / C0;
        wWinMax_ = (MTILE - 1) * sW_ + (kW_ - 1) * dW_ + 1;
        // +1 block headroom only when the window already fills the pitch
        wWinPad_ = (wWinMax_ + C0 - 1) / C0 * C0;
        if (wWinPad_ == wWinMax_) { wWinPad_ += C0; }

        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x),
                             static_cast<int64_t>(totalTiles_) / (Ho_ * woBlocks_) *
                             H_ * W_ * Ci_);
        im2colGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(im2col),
                                  static_cast<int64_t>(totalTiles_) / (Ho_ * woBlocks_) *
                                  mImg_ * K_);

        const int64_t es = static_cast<int64_t>(sizeof(T));
        // fast-path buffer sizing
        rowbufMode_ = (CiPad_ < 64) ? 1 : 0;  // DIRECT stores need >= 128B blocks
        int64_t slabBytes = static_cast<int64_t>(wWinPad_) * CiPad_ * es;  // legacy: one (kh,cib) slab
        int64_t transLegacyBytes = static_cast<int64_t>(wWinPad_) * C0 * es;
        int64_t transAllBytes = static_cast<int64_t>(kH_) * wWinPad_ * CiPad_ * es;
        int64_t rowbufBytes = static_cast<int64_t>(MTILE) * K_ * es;
        // P4: the fast NCHW path stages ALL (kh,cib) slabs at once so the 12
        // per-tile loads share ONE MTE2->V sync point instead of 24 flag
        // pairs (sync serialization dominated: ~10us/tile -> ~1us/tile).
        int64_t fastSlabBytes = static_cast<int64_t>(kH_) * slabBytes;
        int64_t fastTotal = fastSlabBytes + transLegacyBytes + transAllBytes +
                            (rowbufMode_ ? rowbufBytes : 0);
        useFast_ = (fastTotal <= FAST_UB_BUDGET) ? 1 : 0;
        if (isNCHW_ && ((static_cast<int64_t>(H_) * W_) & 1)) {
            // merged 16-ci-row load needs a uniform 2B-realign shift across
            // ci rows (srcOff parity = f(ci) via ci*H*W) -> require H*W even
            useFast_ = 0;
        }

        pipe->InitBuffer(slabBuf_,
                         static_cast<uint32_t>(useFast_ ? fastSlabBytes : slabBytes));
        pipe->InitBuffer(transBuf_, static_cast<uint32_t>(transLegacyBytes));
        pipe->InitBuffer(transAllBuf_,
                         static_cast<uint32_t>(useFast_ ? transAllBytes : 64));
        pipe->InitBuffer(rowbufBuf_,
                         static_cast<uint32_t>((useFast_ && rowbufMode_) ? rowbufBytes : 64));

        evMte2V_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        evVMte3_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        evMte3Mte2_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        evMte2Mte3_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        evVMte2_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        evMte3V_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        if (useFast_) {
            ProcessFast();
        } else {
            ProcessLegacy();
        }
    }

    // ------------------------------------------------------------------
    // FAST PATH (v5)
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessFast()
    {
        int32_t numCores = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        int32_t coreIdx = AscendC::GetBlockIdx();

        AscendC::LocalTensor<T> slabAll = slabBuf_.Get<T>();
        AscendC::LocalTensor<T> transAll = transAllBuf_.Get<T>();
        AscendC::LocalTensor<T> rowbuf = rowbufBuf_.Get<T>();

        for (int32_t t = coreIdx; t < totalTiles_; t += numCores) {
            int32_t wb = t % woBlocks_;
            int32_t ho = (t / woBlocks_) % Ho_;
            int32_t img = t / (woBlocks_ * Ho_);
            uint32_t mCnt = Wo_ - wb * MTILE;
            mCnt = mCnt > MTILE ? MTILE : mCnt;
            uint32_t wWin = (mCnt - 1) * sW_ + (kW_ - 1) * dW_ + 1;
            int32_t wiStart = wb * MTILE * sW_ - pW_;
            int32_t ws = wiStart < 0 ? 0 : wiStart;

            if (isNCHW_) {
                // P4: one prezero + all (kh,cib) loads back-to-back + ONE
                // MTE2->V wait + all transposes back-to-back (V pipe keeps
                // program order). Replaces 2 flag pairs per (kh,cib).
                AscendC::Duplicate(slabAll, static_cast<T>(0),
                                   static_cast<int32_t>(kH_) *
                                   static_cast<int32_t>(ciBlocks_) * 16 *
                                   static_cast<int32_t>(wWinPad_));
                SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                for (int32_t kh = 0; kh < kH_; ++kh) {
                    int32_t hi = ho * sH_ + kh * dH_ - pH_;
                    for (int32_t cib = 0; cib < (int32_t)ciBlocks_; ++cib) {
                        AscendC::LocalTensor<T> slice =
                            slabAll[(kh * ciBlocks_ + cib) * 16 * wWinPad_];
                        LoadFastNCHW(img, hi, wiStart, wWin, cib, slice);
                    }
                }
                SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
                WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
                for (int32_t kh = 0; kh < kH_; ++kh) {
                    for (int32_t cib = 0; cib < (int32_t)ciBlocks_; ++cib) {
                        AscendC::LocalTensor<T> slice =
                            slabAll[(kh * ciBlocks_ + cib) * 16 * wWinPad_];
                        TransFastNCHW(kh, cib, slice, transAll);
                    }
                }
            } else {
                for (int32_t kh = 0; kh < kH_; ++kh) {
                    int32_t hi = ho * sH_ + kh * dH_ - pH_;
                    // whole-row slab loads straight into transAll[kh] slice
                    AscendC::LocalTensor<T> dst =
                        transAll[kh * wWinPad_ * CiPad_];
                    LoadSlabNHWC(img, hi, wiStart, wWin, dst);
                }
            }
            ExpandStore(img, ho, wb, mCnt, ws, transAll, rowbuf);
        }
    }

private:
    // FAST: NCHW per (kh, cib) load into a slab slice [16 ci][wWinPad].
    // Caller pre-zeroes all slices and owns the V->MTE2 / MTE2->V sync.
    __aicore__ inline void LoadFastNCHW(int32_t img, int32_t hi, int32_t wiStart,
                                        uint32_t wWin, int32_t cib,
                                        AscendC::LocalTensor<T> &slab)
    {
        if (dbgNoStore_ == 2) { return; }
        int32_t ws = wiStart < 0 ? 0 : wiStart;
        int32_t we = wiStart + static_cast<int32_t>(wWin);
        we = we > W_ ? W_ : we;
        uint32_t validW = (we > ws && hi >= 0 && hi < H_)
                              ? static_cast<uint32_t>(we - ws) : 0;
        if (validW == 0) {
            return;  // caller pre-zeroed the slice
        }
        uint32_t leftPad = static_cast<uint32_t>(ws - wiStart);
        int64_t base = ((static_cast<int64_t>(img) * Ci_ + cib * C0) * H_ + hi) * W_ + ws;
        uint32_t shift = static_cast<uint32_t>(base & 1);
        uint32_t loadW = validW + shift;
        int32_t validCiRows = Ci_ - cib * C0;
        validCiRows = validCiRows < 16 ? validCiRows : 16;

        // merged 16-ci-row load (no pad params): only when the window is
        // unclamped (validW == wWin, leftPad == 0) so the consumed region
        // [shift, shift+wWin) is entirely valid loaded data; the load spans
        // exactly [ws-shift, ws+wWin) -> zero GM over-read. Clamped tiles
        // fall back to per-row loads (prezero + pad params supply the zeros).
        bool merged = (validW == wWin) && (leftPad == 0) && (validCiRows > 0);
        if (merged) {
            uint32_t occB = (loadW * sizeof(T) + 31) / 32;
            uint32_t pitchB = wWinPad_ * sizeof(T) / 32;
            AscendC::DataCopyExtParams cp{static_cast<uint16_t>(validCiRows),
                static_cast<uint32_t>(loadW * sizeof(T)),
                static_cast<uint32_t>((static_cast<int64_t>(H_) * W_ - loadW) * sizeof(T)),
                pitchB - occB, 0};
            AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(slab, xGM_[base - shift], cp, pp);
        } else {
            // per-row fallback (large clamp offsets)
            for (uint32_t r = 0; r < 16; ++r) {
                int32_t ci = cib * C0 + r;
                if (ci >= Ci_) {
                    continue;
                }
                int64_t srcOff = base + static_cast<int64_t>(r) * H_ * W_;
                uint32_t rshift = static_cast<uint32_t>(srcOff & 1);
                srcOff -= rshift;
                uint32_t rLoadW = validW + rshift;
                uint32_t rRightPad = wWinPad_ - leftPad - rLoadW;
                uint32_t leftAligned = leftPad & ~15u;
                uint32_t leftRem = leftPad - leftAligned;
                uint32_t rRightRem = rRightPad > 16 ? 16 : rRightPad;
                AscendC::DataCopyExtParams cp{1,
                    static_cast<uint32_t>(rLoadW * sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> pp{true,
                    static_cast<uint8_t>(leftRem), static_cast<uint8_t>(rRightRem),
                    static_cast<T>(0)};
                AscendC::DataCopyPad(slab[r * wWinPad_ + leftAligned], xGM_[srcOff], cp, pp);
            }
        }
    }

    // FAST: transpose slab [16 ci][wWinPad] into transAll rows
    // [kh*wWinPad, +wWinPad) cols [cib*16, +16)
    __aicore__ inline void TransFastNCHW(int32_t kh, int32_t cib,
                                         AscendC::LocalTensor<T> &slab,
                                         AscendC::LocalTensor<T> &transAll)
    {
        if (dbgNoStore_ == 3 || dbgNoStore_ == 7) { return; }
        using TransT = typename std::conditional<std::is_same<T, half>::value,
                                                 half, uint16_t>::type;
        AscendC::LocalTensor<TransT> srcList[16];
        AscendC::LocalTensor<TransT> dstList[16];
        uint32_t rowBase = kh * wWinPad_;
        for (uint32_t i = 0; i < 16; ++i) {
            srcList[i] = slab[i * wWinPad_].template ReinterpretCast<TransT>();
            dstList[i] = transAll[(rowBase + i) * CiPad_ + cib * C0]
                             .template ReinterpretCast<TransT>();
        }
        uint32_t repeats = wWinPad_ / C0;
        // per repeat the dst advances 16 rows x CiPad elems (32B-block units)
        uint16_t dstRS = (repeats == 1) ? 0 : static_cast<uint16_t>(CiPad_ * sizeof(T) / 2);
        uint16_t srcRS = (repeats == 1) ? 0 : 1;
        AscendC::TransDataTo5HDParams params(false, false, static_cast<uint8_t>(repeats),
                                             dstRS, srcRS);
        AscendC::TransDataTo5HD<TransT>(dstList, srcList, params);
    }

    // FAST: expansion + store for one (img, ho, wb) tile.
    // transAll layout [kH*wWinPad][CiPad]; GM rows of the tile are consecutive
    // im2col rows: row0 = img*mImg + ho*Wo + wb*MTILE.
    __aicore__ inline void ExpandStore(int32_t img, int32_t ho, int32_t wb,
                                       uint32_t mCnt, int32_t ws,
                                       AscendC::LocalTensor<T> &transAll,
                                       AscendC::LocalTensor<T> &rowbuf)
    {
        int64_t row0 = static_cast<int64_t>(img) * mImg_ + ho * Wo_ + wb * MTILE;
        const uint32_t esz = sizeof(T);

        if (rowbufMode_) {
            // V-pipe expansion into rowbuf [mCnt][K] (program-ordered after
            // the transposes; NHWC needs the trailing MTE2 loads drained)
            if (!isNCHW_) {
                SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
                WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
            }
            if (dbgNoStore_ != 1 && dbgNoStore_ != 7) {
                for (int32_t kh = 0; kh < kH_; ++kh) {
                    int32_t hi = ho * sH_ + kh * dH_ - pH_;
                    uint32_t shift = ShiftOf(img, hi, ws);
                    for (int32_t kw = 0; kw < kW_; ++kw) {
                        uint32_t srcRow = kh * wWinPad_ + kw * dW_ + shift;
                        uint32_t dstCol = (kh * kW_ + kw) * CiPad_;
                        AscendC::CopyRepeatParams rp{
                            1, 1,
                            static_cast<uint16_t>(K_ * esz / 32),
                            static_cast<uint16_t>(sW_ * CiPad_ * esz / 32)};
                        AscendC::Copy(rowbuf[dstCol], transAll[srcRow * CiPad_],
                                      static_cast<uint64_t>(CiPad_),
                                      static_cast<uint8_t>(mCnt), rp);
                    }
                }
            }
            SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);
            WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);
            if (dbgNoStore_ != 1 && dbgNoStore_ != 7) {
                // tile rows are consecutive im2col rows -> one contiguous store
                AscendC::DataCopyParams cp{static_cast<uint16_t>(mCnt),
                    static_cast<uint16_t>(K_ * esz / 32), 0, 0};
                AscendC::DataCopy(im2colGM_[row0 * K_], rowbuf, cp);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
            WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
        } else {
            // DIRECT: per (kh, kw) one 2D DataCopy [mCnt][CiPad] UB -> GM
            SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);
            WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);
            if (dbgNoStore_ != 1 && dbgNoStore_ != 7) {
                for (int32_t kh = 0; kh < kH_; ++kh) {
                    int32_t hi = ho * sH_ + kh * dH_ - pH_;
                    uint32_t shift = ShiftOf(img, hi, ws);
                    for (int32_t kw = 0; kw < kW_; ++kw) {
                        uint32_t srcRow = kh * wWinPad_ + kw * dW_ + shift;
                        int64_t dstOff = row0 * K_ +
                            static_cast<int64_t>(kh * kW_ + kw) * CiPad_;
                        AscendC::DataCopyParams cp{static_cast<uint16_t>(mCnt),
                            static_cast<uint16_t>(CiPad_ * esz / 32),
                            static_cast<uint16_t>((sW_ - 1) * CiPad_ * esz / 32),
                            static_cast<uint16_t>((K_ - CiPad_) * esz / 32)};
                        AscendC::DataCopy(im2colGM_[dstOff],
                                          transAll[srcRow * CiPad_], cp);
                    }
                }
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
            WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
        }
    }

    // 2B-realign shift of the load window for one (img, hi) row; NCHW fast
    // path guarantees H*W even so the shift is uniform across ci rows.
    __aicore__ inline uint32_t ShiftOf(int32_t img, int32_t hi, int32_t ws)
    {
        if (isNCHW_) {
            return static_cast<uint32_t>(
                ((static_cast<int64_t>(img) * Ci_ * H_ + hi) * W_ + ws) & 1);
        }
        return static_cast<uint32_t>(
            (((static_cast<int64_t>(img) * H_ + hi) * W_ + ws) * Ci_) & 1);
    }

    // ------------------------------------------------------------------
    // LEGACY PATH (proven small-tile implementation, unchanged logic)
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessLegacy()
    {
        int32_t numCores = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        int32_t coreIdx = AscendC::GetBlockIdx();

        AscendC::LocalTensor<T> slab = slabBuf_.Get<T>();
        AscendC::LocalTensor<T> trans = transBuf_.Get<T>();

        for (int32_t t = coreIdx; t < totalTiles_; t += numCores) {
            int32_t wb = t % woBlocks_;
            int32_t ho = (t / woBlocks_) % Ho_;
            int32_t img = t / (woBlocks_ * Ho_);
            uint32_t mCnt = Wo_ - wb * MTILE;
            mCnt = mCnt > MTILE ? MTILE : mCnt;
            uint32_t wWin = (mCnt - 1) * sW_ + (kW_ - 1) * dW_ + 1;
            int32_t wiStart = wb * MTILE * sW_ - pW_;

            for (int32_t kh = 0; kh < kH_; ++kh) {
                int32_t hi = ho * sH_ + kh * dH_ - pH_;
                if (isNCHW_) {
                    for (int32_t cib = 0; cib < (int32_t)ciBlocks_; ++cib) {
                        LoadTileNCHW(img, hi, wiStart, wWin, cib, slab, trans);
                        StoreChunksNCHW(img, ho, wb, mCnt, kh, cib, trans);
                    }
                } else {
                    LoadSlabNHWC(img, hi, wiStart, wWin, slab);
                    StoreChunksNHWC(img, ho, wb, mCnt, kh, slab);
                }
            }
        }
    }

    // NHWC slab [wWinPad][CiPad] for one hi row (clamped, zero padded)
    __aicore__ inline void LoadSlabNHWC(int32_t img, int32_t hi, int32_t wiStart,
                                        uint32_t wWin, AscendC::LocalTensor<T> &slab)
    {
        srcShift_ = 0;  // defined value when no valid row updates it below
        int32_t ws = wiStart < 0 ? 0 : wiStart;
        int32_t we = wiStart + static_cast<int32_t>(wWin);
        we = we > W_ ? W_ : we;
        uint32_t validW = (we > ws && hi >= 0 && hi < H_)
                              ? static_cast<uint32_t>(we - ws) : 0;
        // zero the slab (V), then copy the valid window into the middle (MTE2);
        // large clamp offsets can not use DataCopyPad's uint8 left/rightPadding
        AscendC::Duplicate(slab, static_cast<T>(0), wWinPad_ * CiPad_);
        SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
        WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
        if (validW == 0) {
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(evMte2Mte3_);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evMte2Mte3_);
            return;
        }
        uint32_t leftPad = static_cast<uint32_t>(ws - wiStart) * CiPad_;
        int64_t srcOff = ((static_cast<int64_t>(img) * H_ + hi) * W_ + ws) * Ci_;
        srcShift_ = static_cast<uint32_t>(srcOff & 1);
        srcOff -= srcShift_;
        AscendC::DataCopyExtParams cp{1,
            static_cast<uint32_t>((validW + srcShift_) * Ci_ * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(slab[leftPad + srcShift_ * CiPad_], xGM_[srcOff], cp, pp);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(evMte2Mte3_);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evMte2Mte3_);
    }

    // per (m in tile, cib): store kW chunks (16 ci each) from slab row
    __aicore__ inline void StoreChunksNHWC(int32_t img, int32_t ho, int32_t wb,
                                           uint32_t mCnt, int32_t kh,
                                           AscendC::LocalTensor<T> &slab)
    {
        for (uint32_t j = 0; j < mCnt; ++j) {
            int64_t row = static_cast<int64_t>(img) * mImg_ + ho * Wo_ + wb * MTILE + j;
            uint32_t wRel = j * sW_;
            int64_t dstBase = row * K_ + static_cast<int64_t>(kh) * kW_ * CiPad_;
            for (int32_t cib = 0; cib < (int32_t)ciBlocks_; ++cib) {
                // k order (kh,kw,ci): kw block stride is CiPad on GM
                AscendC::DataCopyParams cp{
                    static_cast<uint16_t>(kW_),
                    static_cast<uint16_t>(1),
                    static_cast<uint16_t>(dW_ * (CiPad_ / C0) - 1),
                    static_cast<uint16_t>(CiPad_ / C0 - 1)};
                AscendC::DataCopy(im2colGM_[dstBase + cib * C0],
                                  slab[((wRel + srcShift_) * CiPad_) + cib * C0], cp);
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2_);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2_);
        // store (MTE3) reads slab; next load's V (Duplicate) overwrites it
        SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
        WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
    }

    // NCHW tile [16 ci][wWinPad] -> trans [wWinPad][16] for one (kh, cib)
    // DataCopyPad rule: leftPadding/rightPadding must each be <= 32B (16
    // fp16/bf16 elems); large clamp offsets therefore can not use the pad
    // params. Instead every row is pre-zeroed (V), then the valid window is
    // copied to a 16-element-aligned dst with the remainder handled by pad
    // params (each <= 15/16 elems). Pre-zero + clamped pad keeps every byte
    // of the padding region 0 (isPad=true -> dummy fill = paddingValue = 0).
    __aicore__ inline void LoadTileNCHW(int32_t img, int32_t hi, int32_t wiStart,
                                        uint32_t wWin, int32_t cib,
                                        AscendC::LocalTensor<T> &slab,
                                        AscendC::LocalTensor<T> &trans)
    {
        if (dbgNoStore_ == 2) { return; }
        srcShift_ = 0;  // defined value when all rows are out-of-range below
        for (uint32_t r = 0; r < 16; ++r) {
            AscendC::Duplicate(slab[r * wWinPad_], static_cast<T>(0), wWinPad_);
        }
        SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
        WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
        for (uint32_t r = 0; r < 16; ++r) {
            if (dbgNoStore_ == 4) {
                continue;
            }
            int32_t ci = cib * C0 + r;
            int32_t ws = wiStart < 0 ? 0 : wiStart;
            int32_t we = wiStart + static_cast<int32_t>(wWin);
            we = we > W_ ? W_ : we;
            uint32_t validW = (we > ws && hi >= 0 && hi < H_ && ci < Ci_)
                                  ? static_cast<uint32_t>(we - ws) : 0;
            if (validW == 0) {
                continue;
            }
            uint32_t leftPad = static_cast<uint32_t>(ws - wiStart);
            int64_t srcOff = ((static_cast<int64_t>(img) * Ci_ + ci) * H_ + hi) * W_ + ws;
            uint32_t shift = static_cast<uint32_t>(srcOff & 1);
            srcShift_ = shift;
            srcOff -= shift;
            uint32_t loadW = validW + shift;
            uint32_t rightPad = wWinPad_ - leftPad - loadW;
            uint32_t leftAligned = leftPad & ~15u;  // 32B-aligned dst offset
            uint32_t leftRem = leftPad - leftAligned;       // <= 15 elems
            uint32_t rightRem = rightPad > 16 ? 16 : rightPad;  // <= 16 elems
            AscendC::DataCopyExtParams cp{1,
                static_cast<uint32_t>(loadW * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pp{true,
                static_cast<uint8_t>(leftRem), static_cast<uint8_t>(rightRem),
                static_cast<T>(0)};
            AscendC::DataCopyPad(slab[r * wWinPad_ + leftAligned], xGM_[srcOff], cp, pp);
        }
        SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
        WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);

        // bf16 unsupported by TransDataTo5HD on c220: transpose as raw u16
        using TransT = typename std::conditional<std::is_same<T, half>::value,
                                                 half, uint16_t>::type;
        AscendC::LocalTensor<TransT> srcList[16];
        AscendC::LocalTensor<TransT> dstList[16];
        for (uint32_t i = 0; i < 16; ++i) {
            srcList[i] = slab[i * wWinPad_].template ReinterpretCast<TransT>();
            dstList[i] = trans[C0 * i].template ReinterpretCast<TransT>();
        }
        uint32_t repeats = wWinPad_ / C0;
        uint16_t dstRS = (repeats == 1) ? 0 : 16;
        uint16_t srcRS = (repeats == 1) ? 0 : 1;
        if (dbgNoStore_ != 3 && dbgNoStore_ != 7) {
            AscendC::TransDataTo5HDParams params(false, false, static_cast<uint8_t>(repeats),
                                                 dstRS, srcRS);
            AscendC::TransDataTo5HD<TransT>(dstList, srcList, params);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);
        WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);
    }

    __aicore__ inline void StoreChunksNCHW(int32_t img, int32_t ho, int32_t wb,
                                           uint32_t mCnt, int32_t kh, int32_t cib,
                                           AscendC::LocalTensor<T> &trans)
    {
        if (dbgNoStore_ == 1 || dbgNoStore_ == 7) { return; }
        for (uint32_t j = 0; j < mCnt; ++j) {
            int64_t row = static_cast<int64_t>(img) * mImg_ + ho * Wo_ + wb * MTILE + j;
            uint32_t wRel = j * sW_;
            int64_t dstBase = row * K_ + static_cast<int64_t>(kh) * kW_ * CiPad_ + cib * C0;
            // k order (kh,kw,ci): kw block stride is CiPad on GM
            AscendC::DataCopyParams cp{
                static_cast<uint16_t>(kW_),
                static_cast<uint16_t>(1),
                static_cast<uint16_t>(dW_ - 1),
                static_cast<uint16_t>(CiPad_ / C0 - 1)};
            AscendC::DataCopy(im2colGM_[dstBase], trans[(wRel + srcShift_) * C0], cp);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2_);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2_);
        // store (MTE3) reads trans; next load's V (transpose) overwrites it
        SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
        WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
    }

    int32_t H_;
    int32_t W_;
    int32_t Ci_;
    int32_t CiPad_;
    int32_t kH_;
    int32_t kW_;
    int32_t Ho_;
    int32_t Wo_;
    int32_t sH_;
    int32_t sW_;
    int32_t dH_;
    int32_t dW_;
    int32_t pH_;
    int32_t pW_;
    int32_t K_;
    int32_t mImg_;
    int32_t woBlocks_;
    int32_t totalTiles_;
    int32_t isNCHW_;
    uint32_t ciBlocks_;
    uint32_t wWinMax_;
    uint32_t wWinPad_;
    uint32_t srcShift_;  // odd GM-address realign offset (0/1 elements)
    int32_t useFast_;
    int32_t rowbufMode_;
    int32_t evMte2V_;
    int32_t evVMte3_;
    int32_t evMte3Mte2_;
    int32_t evMte2Mte3_;
    int32_t evVMte2_;
    int32_t evMte3V_;
public:
    int32_t dbgNoStore_ = 0;
    int32_t disableMerged_ = 0;
private:

    AscendC::GlobalTensor<T> xGM_;
    AscendC::GlobalTensor<T> im2colGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> slabBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> transBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> transAllBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowbufBuf_;
};

#endif // CONV2D_BASE_IM2COL_H
