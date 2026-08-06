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
 * @file conv2d_base_epilogue.h
 * Stage-2 (AIV-only) epilogue for conv2d_base (v2, host-zero-prep).
 *   y = cast_to_T(acc_fp32 + bias); bias is loaded raw (fp16/bf16) and cast
 *   to fp32 in-kernel (replaces the former host-side bias .to(kFloat)).
 *   acc: [totalRows, CoPad] fp32 (totalRows = batch*Ho*Wo, CoPad = ceil16(Co)).
 *
 *   NHWC: y [totalRows, Co] written row-wise (Co-trimmed via one 2D
 *         DataCopyPad per tile; bias add is a single ARA-style Add with
 *         src1RepStride = 0).
 *   NCHW (v5, P2): big tiles of [rowsPerTile rows][coC co], coC = min(CoPad,
 *         64), tiles never cross image boundaries. Per tile: one 2D acc load,
 *         one ARA bias Add, one Cast, coC/16 TransDataTo5HD calls (repeat
 *         walks the 16-row blocks), and ONE 2D DataCopyPad store
 *         [coC][rows] per tile (rows contiguous per output channel).
 *         Replaces the former 16x16-tile scheme (per-tile 16x 32B stores +
 *         scalar SetValue/GetValue staging for odd segments).
 */

#ifndef CONV2D_BASE_EPILOGUE_H
#define CONV2D_BASE_EPILOGUE_H

#include "kernel_operator.h"
#include "conv2d_base_tiling.h"

using namespace AscendC;

template <typename T>
class Conv2dEpilogueKernel {
    static constexpr uint32_t C0 = CONV_C0;
    // P0-b direct 1x1 row tile (elements of fp32 acc per inner chunk)
    static constexpr int32_t DIRECT_TILE = 4096;

public:
    __aicore__ inline Conv2dEpilogueKernel() {}

    __aicore__ inline void Init(GM_ADDR acc, GM_ADDR bias, GM_ADDR y,
                                int32_t totalRows, int32_t coPad, int32_t co,
                                int32_t mImg, int32_t rowsPerTile, int32_t hasBias,
                                int32_t isNCHW, int32_t directMode, AscendC::TPipe *pipe)
    {
        totalRows_ = totalRows;
        coPad_ = coPad;
        co_ = co;
        mImg_ = mImg;
        rowsPerTile_ = rowsPerTile;
        hasBias_ = hasBias;
        isNCHW_ = isNCHW;
        directMode_ = directMode;

        if (directMode_) {
            // P0-b direct 1x1: acc is [totalRows=batch*Co, coPad_=H*W pad]
            // (P7: acc rows may be padded), y is [totalRows, mImg=H*W]
            accGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(acc),
                                   static_cast<int64_t>(totalRows_) * coPad_);
            yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y),
                                 static_cast<int64_t>(totalRows_) * mImg_);
        } else {
            accGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(acc),
                                   static_cast<int64_t>(totalRows_) * coPad_);
            yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y),
                                 static_cast<int64_t>(totalRows_) * co_);
        }
        biasGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias), co_);

        coC_ = coPad_ < 64 ? coPad_ : 64;

        if (directMode_) {
            // P0-b direct 1x1: acc/y are [totalRows=batch*Co, mImg=H*W];
            // small row-granularity buffers only
            pipe->InitBuffer(accBuf_, DIRECT_TILE * sizeof(float));
            pipe->InitBuffer(outBuf_, DIRECT_TILE * sizeof(T));
            pipe->InitBuffer(transBuf_, 64);
        } else if (isNCHW_) {
            pipe->InitBuffer(accBuf_, rowsPerTile_ * coC_ * sizeof(float));
            pipe->InitBuffer(outBuf_, rowsPerTile_ * coC_ * sizeof(T));
            pipe->InitBuffer(transBuf_, coC_ * rowsPerTile_ * sizeof(T));
        } else {
            pipe->InitBuffer(accBuf_, rowsPerTile_ * coPad_ * sizeof(float));
            pipe->InitBuffer(outBuf_, rowsPerTile_ * coPad_ * sizeof(T));
            pipe->InitBuffer(transBuf_, 64);
        }
        // P7: in direct mode coPad_ is the acc row stride (H*W pad), not a
        // bias-related extent — size the bias buffers by co_ instead
        int32_t biasLen = directMode_ ? co_ : coPad_;
        pipe->InitBuffer(biasBuf_, biasLen * sizeof(float));
        pipe->InitBuffer(biasRawBuf_, biasLen * sizeof(T));

        evMte2V_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        evVMte3_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        evVMte2_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        evMte3V_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        int32_t numCores = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        int32_t coreIdx = AscendC::GetBlockIdx();

        AscendC::LocalTensor<float> biasF = biasBuf_.Get<float>();
        if (hasBias_) {
            AscendC::LocalTensor<T> biasRaw = biasRawBuf_.Get<T>();
            AscendC::Duplicate(biasF, 0.0f, directMode_ ? co_ : coPad_);
            // DataCopyPad: co_ need not make a 32B-multiple transfer
            AscendC::DataCopyExtParams bcp{1,
                static_cast<uint32_t>(co_ * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> bpp{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(biasRaw, biasGM_, bcp, bpp);
            SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
            WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
            AscendC::Cast(biasF, biasRaw, AscendC::RoundMode::CAST_NONE, co_);
            PipeBarrier<PIPE_V>();
        }

        if (directMode_) {
            ProcessDirect(coreIdx, numCores, biasF);
        } else if (isNCHW_) {
            ProcessNCHW(coreIdx, numCores, biasF);
        } else {
            ProcessNHWC(coreIdx, numCores, biasF);
        }
    }

private:
    // P0-b direct 1x1 path: acc/y are [batch*Co, H*W] (channel rows,
    // spatial contiguous) — the WHOLE array is contiguous, so process big
    // cross-row chunks: one load + per-row scalar Adds segments (V pipe,
    // no sync between them) + one cast + one store per chunk. Per-core
    // contiguous spans keep the cores balanced. Replaces the row-at-a-time
    // scheme whose per-row flag pairs dominated (13-15us -> ~3-4us).
    __aicore__ inline void ProcessDirect(int32_t coreIdx, int32_t numCores,
                                         const AscendC::LocalTensor<float> &biasF)
    {
        if (coPad_ != mImg_) {
            ProcessDirectPadded(coreIdx, numCores, biasF);
            return;
        }
        AscendC::LocalTensor<float> accLocal = accBuf_.Get<float>();
        AscendC::LocalTensor<T> outLocal = outBuf_.Get<T>();
        int64_t rowLen = mImg_;
        int64_t total = static_cast<int64_t>(totalRows_) * rowLen;

        // contiguous per-core span (16-element granular: fp32 loads need
        // 8-element / 32B alignment, fp16/bf16 stores need 16), chunked by
        // DIRECT_TILE
        int64_t span = (total + numCores - 1) / numCores;
        span = (span + 15) / 16 * 16;
        int64_t e0 = coreIdx * span;
        int64_t e1 = e0 + span;
        e1 = e1 < total ? e1 : total;

        for (int64_t c0 = e0; c0 < e1; c0 += DIRECT_TILE) {
            int32_t len = static_cast<int32_t>(e1 - c0);
            len = len > DIRECT_TILE ? DIRECT_TILE : len;
            // plain DataCopy requires 32B-multiple transfers: fp32 load
            // needs len%8==0, the fp16/bf16 store needs len%16==0 (a 7376B
            // store was silently truncated to 7360B -> 8 missing halfs per
            // span tail, measured). Anything else goes through DataCopyPad.
            bool aligned = (len % 8 == 0) && ((c0 % 8) == 0);
            bool alignedOut = (len % 16 == 0) && ((c0 % 16) == 0);

            if (aligned) {
                AscendC::DataCopy(accLocal, accGM_[c0], len);
            } else {
                AscendC::DataCopyExtParams cp{1,
                    static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
                AscendC::DataCopyPad(accLocal, accGM_[c0], cp, pp);
            }
            SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
            WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);

            if (hasBias_) {
                int64_t pos = c0;
                while (pos < c0 + len) {
                    int64_t r = pos / rowLen;
                    int64_t rowEnd = (r + 1) * rowLen;
                    rowEnd = rowEnd < c0 + len ? rowEnd : c0 + len;
                    float bv = biasF.GetValue(static_cast<int32_t>(r % co_));
                    AscendC::Adds(accLocal[pos - c0], accLocal[pos - c0], bv,
                                  static_cast<int32_t>(rowEnd - pos));
                    pos = rowEnd;
                }
                PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(outLocal, accLocal, AscendC::RoundMode::CAST_RINT, len);
            SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);
            WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);

            if (alignedOut) {
                AscendC::DataCopy(yGM_[c0], outLocal, len);
            } else {
                AscendC::DataCopyExtParams ocp{1,
                    static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPad(yGM_[c0], outLocal, ocp);
            }
            // buffer reuse across chunks (P1.4: skip on the last chunk)
            if (c0 + DIRECT_TILE < e1) {
                SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
                WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
            }
        }
    }

    __aicore__ inline void ProcessNHWC(int32_t coreIdx, int32_t numCores,
                                       const AscendC::LocalTensor<float> &biasF)
    {
        int32_t tiles = (totalRows_ + rowsPerTile_ - 1) / rowsPerTile_;
        AscendC::LocalTensor<float> accLocal = accBuf_.Get<float>();
        AscendC::LocalTensor<T> outLocal = outBuf_.Get<T>();

        for (int32_t t = coreIdx; t < tiles; t += numCores) {
            int64_t row0 = static_cast<int64_t>(t) * rowsPerTile_;
            int32_t rows = totalRows_ - static_cast<int32_t>(row0);
            rows = rows < rowsPerTile_ ? rows : rowsPerTile_;

            AscendC::DataCopyParams cp{static_cast<uint16_t>(rows),
                static_cast<uint16_t>(coPad_ * sizeof(float) / 32),
                static_cast<uint16_t>(0), static_cast<uint16_t>(0)};
            AscendC::DataCopy(accLocal, accGM_[row0 * coPad_], cp);
            SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
            WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);

            if (hasBias_) {
                // ARA: per repeat one acc row (coPad cols), bias replayed
                for (int32_t r0 = 0; r0 < rows; r0 += 255) {
                    uint8_t rep = rows - r0 > 255 ? 255 : static_cast<uint8_t>(rows - r0);
                    if (coPad_ <= 64) {
                        // ARA broadcast: src1BlkStride=1 (advance across the
                        // row's blocks), src1RepStride=0 (replay across rows)
                        AscendC::BinaryRepeatParams bp{1, 1, 1,
                            static_cast<uint8_t>(coPad_ * sizeof(float) / 32),
                            static_cast<uint8_t>(coPad_ * sizeof(float) / 32), 0};
                        AscendC::Add(accLocal[r0 * coPad_], accLocal[r0 * coPad_], biasF,
                                     static_cast<uint64_t>(coPad_), rep, bp);
                    } else {
                        // 64-col groups; bias slice advances with the group
                        for (int32_t g = 0; g < coPad_; g += 64) {
                            uint64_t m = coPad_ - g < 64 ? coPad_ - g : 64;
                            AscendC::BinaryRepeatParams bp{1, 1, 1,
                                static_cast<uint8_t>(coPad_ * sizeof(float) / 32),
                                static_cast<uint8_t>(coPad_ * sizeof(float) / 32), 0};
                            AscendC::Add(accLocal[r0 * coPad_ + g], accLocal[r0 * coPad_ + g],
                                         biasF[g], m, rep, bp);
                        }
                    }
                }
                PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(outLocal, accLocal, AscendC::RoundMode::CAST_RINT,
                          rows * coPad_);
            SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);
            WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);

            // write Co-trimmed rows to y [totalRows, Co]: one 2D call
            AscendC::DataCopyExtParams ocp{static_cast<uint16_t>(rows),
                static_cast<uint32_t>(co_ * sizeof(T)),
                static_cast<uint32_t>((coPad_ - co_) * sizeof(T) / 32), 0, 0};
            AscendC::DataCopyPad(yGM_[row0 * co_], outLocal, ocp);
            // buffer reuse across tiles: next MTE2 load writes accLocal (was
            // read by V Add/Cast); next V Cast writes outLocal (was read by
            // the MTE3 store). P1.4: skipped on the last tile per core (the
            // round-trip is pure latency with nothing to overlap).
            if (t + numCores < tiles) {
                SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
                WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
            }
        }
    }

    // P7 padded direct path: acc rows are coPad_-strided (H*W rounded up
    // to 16), y rows are mImg_-strided. Row groups of R = DIRECT_TILE/hw
    // rows: one 2D DataCopyPad load (drops the pad), per-row scalar Adds,
    // one Cast, one 2D DataCopyPad store into the contiguous y.
    __aicore__ inline void ProcessDirectPadded(int32_t coreIdx, int32_t numCores,
                                               const AscendC::LocalTensor<float> &biasF)
    {
        AscendC::LocalTensor<float> accLocal = accBuf_.Get<float>();
        AscendC::LocalTensor<T> outLocal = outBuf_.Get<T>();
        int64_t rows = totalRows_;
        int32_t hw = mImg_;
        uint32_t pitchFg = (static_cast<uint32_t>(hw) * sizeof(float) + 31) / 32 * 8;
        uint32_t pitchTg = (static_cast<uint32_t>(hw) * sizeof(T) + 31) / 32 *
                           (32 / sizeof(T));
        int32_t rGroup = DIRECT_TILE / hw;
        int32_t rGroupF = DIRECT_TILE / static_cast<int32_t>(pitchFg);
        int32_t rGroupT = DIRECT_TILE / static_cast<int32_t>(pitchTg);
        rGroup = rGroup < rGroupF ? rGroup : rGroupF;
        rGroup = rGroup < rGroupT ? rGroup : rGroupT;
        rGroup = rGroup < 1 ? 1 : rGroup;
        int64_t groups = (rows + rGroup - 1) / rGroup;

        for (int64_t g = coreIdx; g < groups; g += numCores) {
            int64_t r0 = g * rGroup;
            int32_t rc = static_cast<int32_t>(rows - r0);
            rc = rc > rGroup ? rGroup : rc;

            if (hw <= DIRECT_TILE) {
                // UB row pitches: a non-32B-multiple blockLen rounds the UB
                // row pitch UP to whole 32B blocks on BOTH DataCopyPad
                // directions (measured: 784B rows land at a 200-float
                // pitch, 392B rows at a 208-half pitch). Address the tiles
                // with these pitches; per-row Adds/Cast bridge the
                // different acc/out pitches.
                uint32_t pitchF = (static_cast<uint32_t>(hw) * sizeof(float) + 31) / 32 * 8;
                uint32_t pitchT = (static_cast<uint32_t>(hw) * sizeof(T) + 31) / 32 *
                                  (32 / sizeof(T));
                // one 2D load for the whole group: [rc][hw] from
                // acc[r0*coPad_], dropping the pad columns (UB rows land at
                // the rounded pitchF automatically, dstStride=0)
                AscendC::DataCopyExtParams cp{static_cast<uint16_t>(rc),
                    static_cast<uint32_t>(hw * sizeof(float)),
                    static_cast<uint32_t>((coPad_ - hw) * sizeof(float)), 0, 0};
                AscendC::DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
                AscendC::DataCopyPad(accLocal, accGM_[r0 * coPad_], cp, pp);
                SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
                WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);

                for (int32_t rr = 0; rr < rc; ++rr) {
                    if (hasBias_) {
                        float bv = biasF.GetValue(static_cast<int32_t>(r0 + rr) % co_);
                        AscendC::Adds(accLocal[rr * pitchF], accLocal[rr * pitchF],
                                      bv, hw);
                    }
                    AscendC::Cast(outLocal[rr * pitchT], accLocal[rr * pitchF],
                                  AscendC::RoundMode::CAST_RINT, hw);
                }
                PipeBarrier<PIPE_V>();
                SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);
                WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);

                // y rows are contiguous across the whole tensor: dstStride=0
                AscendC::DataCopyExtParams ocp{static_cast<uint16_t>(rc),
                    static_cast<uint32_t>(hw * sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPad(yGM_[r0 * hw], outLocal, ocp);
                SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
                WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
            } else {
                // huge H*W: per-row chunks of DIRECT_TILE
                for (int32_t rr = 0; rr < rc; ++rr) {
                    int64_t r = r0 + rr;
                    float bv = 0.0f;
                    if (hasBias_) { bv = biasF.GetValue(static_cast<int32_t>(r) % co_); }
                    for (int32_t off = 0; off < hw; off += DIRECT_TILE) {
                        int32_t len = hw - off;
                        len = len > DIRECT_TILE ? DIRECT_TILE : len;
                        AscendC::DataCopyExtParams cp{1,
                            static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
                        AscendC::DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
                        AscendC::DataCopyPad(accLocal, accGM_[r * coPad_ + off], cp, pp);
                        SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
                        WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
                        if (hasBias_) {
                            AscendC::Adds(accLocal, accLocal, bv, len);
                            PipeBarrier<PIPE_V>();
                        }
                        AscendC::Cast(outLocal, accLocal,
                                      AscendC::RoundMode::CAST_RINT, len);
                        SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);
                        WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);
                        AscendC::DataCopyExtParams ocp{1,
                            static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
                        AscendC::DataCopyPad(yGM_[r * hw + off], outLocal, ocp);
                        SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                        WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                        SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
                        WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
                    }
                }
            }
        }
    }

    // NCHW big-tile path: tiles = (img, mChunk, coChunk), no image crossing
    __aicore__ inline void ProcessNCHW(int32_t coreIdx, int32_t numCores,
                                       const AscendC::LocalTensor<float> &biasF)
    {
        int32_t coChunks = coPad_ / coC_;
        int32_t mChunks = (mImg_ + rowsPerTile_ - 1) / rowsPerTile_;
        int32_t tiles = (totalRows_ / mImg_) * mChunks * coChunks;
        AscendC::LocalTensor<float> accLocal = accBuf_.Get<float>();
        AscendC::LocalTensor<T> outLocal = outBuf_.Get<T>();
        AscendC::LocalTensor<T> trans = transBuf_.Get<T>();

        for (int32_t t = coreIdx; t < tiles; t += numCores) {
            int32_t cc = t % coChunks;
            int32_t mc = (t / coChunks) % mChunks;
            int32_t img = t / (coChunks * mChunks);
            int32_t co0 = cc * coC_;
            int32_t m0 = mc * rowsPerTile_;
            int32_t rows = mImg_ - m0;
            rows = rows < rowsPerTile_ ? rows : rowsPerTile_;
            uint32_t rowsPad = (rows + 15) / 16 * 16;
            uint32_t nb = rowsPad / 16;

            // load acc[img, m0:+rows, co0:+coC] -> [rows][coC] packed
            AscendC::DataCopyParams cp{static_cast<uint16_t>(rows),
                static_cast<uint16_t>(coC_ * sizeof(float) / 32),
                static_cast<uint16_t>((coPad_ - coC_) * sizeof(float) / 32),
                static_cast<uint16_t>(0)};
            AscendC::DataCopy(accLocal,
                              accGM_[(static_cast<int64_t>(img) * mImg_ + m0) * coPad_ + co0],
                              cp);
            SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
            WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);

            if (hasBias_) {
                for (int32_t r0 = 0; r0 < rows; r0 += 255) {
                    uint8_t rep = rows - r0 > 255 ? 255 : static_cast<uint8_t>(rows - r0);
                    AscendC::BinaryRepeatParams bp{1, 1, 1,
                        static_cast<uint8_t>(coC_ * sizeof(float) / 32),
                        static_cast<uint8_t>(coC_ * sizeof(float) / 32), 0};
                    AscendC::Add(accLocal[r0 * coC_], accLocal[r0 * coC_], biasF[co0],
                                 static_cast<uint64_t>(coC_), rep, bp);
                }
                PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(outLocal, accLocal, AscendC::RoundMode::CAST_RINT,
                          rows * coC_);
            PipeBarrier<PIPE_V>();

            // transpose [rows][coC] -> [coC][rowsPad]; per 16-col group one
            // TransDataTo5HD whose repeats walk the 16-row blocks
            using TransT = typename std::conditional<std::is_same<T, half>::value,
                                                     half, uint16_t>::type;
            for (uint32_t g = 0; g < static_cast<uint32_t>(coC_) / C0; ++g) {
                AscendC::LocalTensor<TransT> srcList[16];
                AscendC::LocalTensor<TransT> dstList[16];
                for (uint32_t i = 0; i < 16; ++i) {
                    srcList[i] = outLocal[i * coC_ + g * C0]
                                     .template ReinterpretCast<TransT>();
                    dstList[i] = trans[(g * C0 + i) * rowsPad]
                                     .template ReinterpretCast<TransT>();
                }
                // repeat r: src += 16 rows (16*coC elems), dst += 16 elems
                uint16_t srcRS = (nb == 1) ? 0
                                           : static_cast<uint16_t>(coC_ * sizeof(T) / 2);
                uint16_t dstRS = (nb == 1) ? 0 : 1;
                AscendC::TransDataTo5HDParams params(false, false,
                                                     static_cast<uint8_t>(nb), dstRS, srcRS);
                AscendC::TransDataTo5HD<TransT>(dstList, srcList, params);
            }
            SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);
            WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);

            // store [validCo][rows] to y[img, co0:+, m0:+]. Main part (rows
            // rounded down to 16) is ONE 2D call (UB-side stride is in 32B
            // blocks, so the rowsPad gap must be 16-element granular); the
            // ragged tail (< 16 rows) is stored per co channel.
            int32_t validCo = co_ - co0;
            validCo = validCo < coC_ ? validCo : coC_;
            int32_t rowsMain = rows & ~15;
            int32_t tail = rows - rowsMain;
            int64_t dstBase = (static_cast<int64_t>(img) * co_ + co0) * mImg_ + m0;
            if (rowsMain > 0) {
                AscendC::DataCopyExtParams ocp{static_cast<uint16_t>(validCo),
                    static_cast<uint32_t>(rowsMain * sizeof(T)),
                    static_cast<uint32_t>((rowsPad - rowsMain) * sizeof(T) / 32),
                    static_cast<uint32_t>((mImg_ - rowsMain) * sizeof(T)), 0};
                AscendC::DataCopyPad(yGM_[dstBase], trans, ocp);
            }
            for (int32_t c = 0; c < validCo && tail > 0; ++c) {
                AscendC::DataCopyExtParams tcp{1,
                    static_cast<uint32_t>(tail * sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPad(yGM_[dstBase + c * mImg_ + rowsMain],
                                     trans[c * rowsPad + rowsMain], tcp);
            }
            // buffer reuse across tiles (P1.4: skip on the last tile per
            // core — pure latency with nothing to overlap)
            if (t + numCores < tiles) {
                SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
                WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
            }
        }
    }

    int32_t totalRows_;
    int32_t coPad_;
    int32_t co_;
    int32_t mImg_;
    int32_t rowsPerTile_;
    int32_t hasBias_;
    int32_t isNCHW_;
    int32_t directMode_;
    int32_t coC_;
    int32_t evMte2V_;
    int32_t evVMte3_;
    int32_t evVMte2_;
    int32_t evMte3V_;

    AscendC::GlobalTensor<float> accGM_;
    AscendC::GlobalTensor<T> biasGM_;
    AscendC::GlobalTensor<T> yGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> biasBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> biasRawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> transBuf_;
};

#endif // CONV2D_BASE_EPILOGUE_H
