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
 * @file conv2d_base_xtrans.h
 * P5 stage-B (v2): AIV kernel transposing an NCHW feature map to
 * channel-inner  x [img, Ci, H, W] -> xT [img, H*W, Ci]   (Ci % 16 == 0).
 * The conv kernel's fused mode (P5) then Nd2Nz-loads xT into L1 in the
 * NC1HWC0 layout the img2col LoadData3D requires. Traffic is ~2x the
 * feature map size (vs ~K/Ci times that for full im2col materialization).
 *
 * v2 (big hw tiles): per hw tile of T rows (T=128/64/32 by UB budget):
 *   - load [16 ci][T hw] slabs per ci block (T*es-byte rows, one merged 2D
 *     DataCopyPad per ci block),
 *   - transpose 16x16 subtiles into a [T][Ci] staging buffer,
 *   - ONE contiguous store of wCnt*Ci elements (xT rows are contiguous
 *     across hw, so the whole tile is a single linear region).
 * Replaces the v1 scheme (T=16, per-tile full-slab prezero, 32B-granular
 * strided stores) which ran at ~20-60GB/s on cases 0/9.
 */

#ifndef CONV2D_BASE_XTRANS_H
#define CONV2D_BASE_XTRANS_H

#include "kernel_operator.h"
#include "conv2d_base_tiling.h"

using namespace AscendC;

template <typename T>
class Conv2dXtransKernel {
    static constexpr uint32_t C0 = CONV_C0;

public:
    __aicore__ inline Conv2dXtransKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR xT,
                                int32_t batch, int32_t Ci, int32_t HW,
                                AscendC::TPipe *pipe)
    {
        Ci_ = Ci;
        HW_ = HW;
        batch_ = batch;
        ciBlocks_ = Ci_ / C0;
        // hw rows per tile: shared with the host via ConvXtransT (UB budget
        // + exact-divisor preference, P1.5)
        T_ = ConvXtransT(Ci_, HW_, static_cast<int32_t>(sizeof(T)));
        hwBlocks_ = (HW_ + T_ - 1) / T_;
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x),
                             static_cast<int64_t>(batch) * Ci_ * HW_);
        xtGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(xT),
                              static_cast<int64_t>(batch) * Ci_ * HW_);
        pipe->InitBuffer(slabBuf_, Ci_ * T_ * sizeof(T));
        pipe->InitBuffer(stageBuf_, Ci_ * T_ * sizeof(T));
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

        AscendC::LocalTensor<T> slab = slabBuf_.Get<T>();
        AscendC::LocalTensor<T> stage = stageBuf_.Get<T>();

        // P1.5 NOTE (measured, reverted): hoisting the TransDataTo5HD
        // pointer lists out of the tile loop (v3 R1) was NET NEGATIVE here
        // (case0 xtrans 8.75 -> 9.1us, case9 11.2 -> 12.5us): 128 hoisted
        // LocalTensor objects with a runtime cib index force stack
        // materialization, while the per-call constructions are register-
        // allocated by the compiler. The divisor-of-HW T selection was also
        // NET NEGATIVE (case0 9.7 -> 12.2us: more tiles = more per-tile
        // fixed costs). Both halves of P1.5 are reverted to the original
        // per-call construction + power-of-two T.

        int64_t tiles = static_cast<int64_t>(batch_) * hwBlocks_;
        for (int64_t t = coreIdx; t < tiles; t += numCores) {
            int32_t hb = static_cast<int32_t>(t % hwBlocks_);
            int32_t img = static_cast<int32_t>(t / hwBlocks_);
            int32_t hw0 = hb * T_;
            uint32_t wCnt = HW_ - hw0;
            wCnt = wCnt > static_cast<uint32_t>(T_) ? static_cast<uint32_t>(T_)
                                                    : wCnt;

            if (wCnt != static_cast<uint32_t>(T_)) {
                // zero the slab pad region once (tail tile only)
                AscendC::Duplicate(slab, static_cast<T>(0), Ci_ * T_);
                SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
            }
            for (uint32_t cib = 0; cib < ciBlocks_; ++cib) {
                LoadBlock(img, hw0, wCnt, cib, slab);
            }
            SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
            WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
            for (uint32_t cib = 0; cib < ciBlocks_; ++cib) {
                TransBlock(cib, slab, stage);
            }
            SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);
            WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3_);

            // one contiguous store: xT rows are contiguous across hw
            int64_t dstBase = (static_cast<int64_t>(img) * HW_ + hw0) * Ci_;
            AscendC::DataCopy(xtGM_[dstBase], stage,
                              static_cast<int32_t>(wCnt) * Ci_);
            // buffer reuse across tiles (skip on the last tile per core:
            // the flag round-trip is pure latency with nothing to overlap)
            if (t + numCores < tiles) {
                SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2_);
                SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
                WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V_);
            }
        }
    }

private:
    // [16 ci][wCnt] slab rows for one (img, hw0, cib) block
    __aicore__ inline void LoadBlock(int32_t img, int32_t hw0, uint32_t wCnt,
                                     uint32_t cib, AscendC::LocalTensor<T> &slab)
    {
        int64_t base = (static_cast<int64_t>(img) * Ci_ + cib * C0) * HW_ + hw0;
        if (wCnt == static_cast<uint32_t>(T_)) {
            // merged 16-ci-row load; dst rows are T contiguous elements
            AscendC::DataCopyExtParams cp{static_cast<uint16_t>(C0),
                static_cast<uint32_t>(T_ * sizeof(T)),
                static_cast<uint32_t>((static_cast<int64_t>(HW_) - T_) * sizeof(T)),
                0, 0};
            AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(slab[cib * C0 * T_], xGM_[base], cp, pp);
        } else {
            // partial tail: per-row loads; pad params supply the zero fill
            for (uint32_t r = 0; r < C0; ++r) {
                int64_t srcOff = base + static_cast<int64_t>(r) * HW_;
                uint32_t rightPad = T_ - wCnt;
                uint32_t rightRem = rightPad > 16 ? 16 : rightPad;
                AscendC::DataCopyExtParams cp{1,
                    static_cast<uint32_t>(wCnt * sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> pp{true, 0,
                    static_cast<uint8_t>(rightRem), static_cast<T>(0)};
                AscendC::DataCopyPad(slab[cib * C0 * T_ + r * T_],
                                     xGM_[srcOff], cp, pp);
            }
        }
    }

    // transpose all 16x16 subtiles of [16 ci][T hw] into stage [T][Ci]:
    // ONE TransDataTo5HD call per ci block whose repeats walk the hw
    // subtiles (scalar setup was the v2 bottleneck: 32 list entries per
    // 16x16 subtile); src advance per repeat = 16 elems (1x 32B block),
    // dst advance = 16 stage rows = Ci 32B blocks
    __aicore__ inline void TransBlock(uint32_t cib, AscendC::LocalTensor<T> &slab,
                                      AscendC::LocalTensor<T> &stage)
    {
        using TransT = typename std::conditional<std::is_same<T, half>::value,
                                                 half, uint16_t>::type;
        AscendC::LocalTensor<TransT> srcList[16];
        AscendC::LocalTensor<TransT> dstList[16];
        for (uint32_t i = 0; i < 16; ++i) {
            srcList[i] = slab[cib * C0 * T_ + i * T_]
                             .template ReinterpretCast<TransT>();
            dstList[i] = stage[i * Ci_ + cib * C0]
                             .template ReinterpretCast<TransT>();
        }
        AscendC::TransDataTo5HDParams params(
            false, false, static_cast<uint8_t>(T_ / C0),
            static_cast<uint16_t>(Ci_), 1);
        AscendC::TransDataTo5HD<TransT>(dstList, srcList, params);
    }

    int32_t Ci_;
    int32_t HW_;
    int32_t batch_;
    int32_t T_;
    uint32_t ciBlocks_;
    int32_t hwBlocks_;
    int32_t evMte2V_;
    int32_t evVMte3_;
    int32_t evVMte2_;
    int32_t evMte3V_;

    AscendC::GlobalTensor<T> xGM_;
    AscendC::GlobalTensor<T> xtGM_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> slabBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stageBuf_;
};

#endif // CONV2D_BASE_XTRANS_H
