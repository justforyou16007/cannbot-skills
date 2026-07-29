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
 * @file conv2d_base_xpad.h
 * P7: tiny AIV kernel padding the flattened H*W dimension of an NCHW tensor.
 *   x [rows, hw] (rows = batch*Ci, fully contiguous) -> xpad [rows, hwPad]
 * Used by the P0-b direct 1x1 mode when hw is not a multiple of 16: the
 * flipped GEMM's B' Nd2Nz load needs a 32B-aligned srcDValue (hwPad
 * elements), while x itself has hw-element (possibly misaligned) rows.
 * The pad columns are never read back (GEMM columns are independent; the
 * pad columns only reach acc pad columns which the epilogue never reads),
 * so no zero fill is needed. DataCopyPad handles the arbitrary byte
 * strides (plain DataCopy would need 32B-multiple transfers).
 */

#ifndef CONV2D_BASE_XPAD_H
#define CONV2D_BASE_XPAD_H

#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class Conv2dXpadKernel {
    static constexpr int32_t UB_BUDGET = 96 * 1024;

public:
    __aicore__ inline Conv2dXpadKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR xpad,
                                int32_t rows, int32_t hw, int32_t hwPad,
                                AscendC::TPipe *pipe)
    {
        rows_ = rows;
        hw_ = hw;
        hwPad_ = hwPad;
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x),
                             static_cast<int64_t>(rows_) * hw_);
        xpGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(xpad),
                              static_cast<int64_t>(rows_) * hwPad_);
        chunkRows_ = UB_BUDGET / (hwPad_ * static_cast<int32_t>(sizeof(T)));
        chunkRows_ = chunkRows_ > 64 ? 64 : chunkRows_;
        pipe->InitBuffer(buf_, chunkRows_ * hwPad_ * sizeof(T));
        evMte2Mte3_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        evMte3Mte2_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Process()
    {
        int32_t numCores = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        int32_t coreIdx = AscendC::GetBlockIdx();
        AscendC::LocalTensor<T> buf = buf_.Get<T>();

        for (int32_t r0 = coreIdx * chunkRows_; r0 < rows_;
             r0 += numCores * chunkRows_) {
            int32_t rows = rows_ - r0;
            rows = rows > chunkRows_ ? chunkRows_ : rows;
            // GM src rows back-to-back (srcStride=0 bytes); UB dst rows
            // hwPad elements apart (dstStride in 32B blocks)
            AscendC::DataCopyExtParams cp{static_cast<uint16_t>(rows),
                static_cast<uint32_t>(hw_ * sizeof(T)), 0,
                static_cast<uint32_t>((hwPad_ - hw_) * sizeof(T) / 32), 0};
            AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(buf, xGM_[static_cast<int64_t>(r0) * hw_], cp, pp);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(evMte2Mte3_);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evMte2Mte3_);

            // UB src rows hwPad apart (srcStride in 32B blocks); GM dst rows
            // hwPad elements apart (dstStride in bytes)
            AscendC::DataCopyExtParams ocp{static_cast<uint16_t>(rows),
                static_cast<uint32_t>(hw_ * sizeof(T)),
                static_cast<uint32_t>((hwPad_ - hw_) * sizeof(T) / 32),
                static_cast<uint32_t>((hwPad_ - hw_) * sizeof(T)), 0};
            AscendC::DataCopyPad(xpGM_[static_cast<int64_t>(r0) * hwPad_], buf, ocp);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2_);
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2_);
        }
    }

private:
    int32_t rows_;
    int32_t hw_;
    int32_t hwPad_;
    int32_t chunkRows_;
    int32_t evMte2Mte3_;
    int32_t evMte3Mte2_;

    AscendC::GlobalTensor<T> xGM_;
    AscendC::GlobalTensor<T> xpGM_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_;
};

#endif // CONV2D_BASE_XPAD_H
