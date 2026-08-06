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
 * @file conv2d_base_dw.h
 * Depthwise (groups == Ci, CiG == 1, stride 1, dilation 1) conv2d for
 * fp16/bf16, pure AIV, fp32 accumulation. Port of the 7_ConvStandard2d_v3
 * R2 fp32 kernel (conv_standard2d_depthwise.cpp) to the archive dtype set:
 *   - x is staged into a fp32 GM scratch (xp, zero halo) via DataCopyPad
 *     (2D, dstStride=0, hardware 32B-rounded UB row pitch) + per-row Cast
 *     T->fp32 (CAST_NONE), then stored UB->GM with a 2D DataCopyPad whose
 *     UB-side pitch equals roundUp32(W*4) (srcStride=0).
 *   - kW shifted row windows are reloaded from xp as fp32; DataCopyPad
 *     absorbs the element-grain misaligned GM src (v3-proven pattern,
 *     compute_conv.md #20: UB-side element shifts are infeasible).
 *   - acc[ow] = bias + sum_{kh,kw} w[kh][kw] * win[kw][ih][ow] with the
 *     fp32 Axpy vector FMA chain; out-of-range ih rows are skipped on the
 *     compute side (xp halos cover W padding), so no win pre-zeroing.
 *   - per-row Cast fp32->T with CAST_RINT (same RoundMode as the archive
 *     epilogue) + per-row DataCopyPad store (multi-row dstStride==0 form
 *     only lands the first row on this platform, v3 measured).
 * Sync: narrowest immediate flag pairs only (compute_conv.md #10: AIV has
 * no MTE2_MTE3 event, staging uses the MTE2_V + V_MTE3 chain; #12: WAR
 * pairs V_MTE2 / MTE3_V immediately around buffer reuse; #15: no
 * PIPE_ALL). Scalar UB reads (weight/bias extraction) are preceded by
 * PipeBarrier<PIPE_V> (archive epilogue pattern).
 */

#ifndef CONV2D_BASE_DW_H
#define CONV2D_BASE_DW_H

#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class Conv2dDwKernel {
    static constexpr int32_t DW_STAGE_ELEMS = 8192;  // stageT/stageF capacity
    static constexpr int32_t DW_MAX_K = 64;          // kH*kW upper bound

public:
    __aicore__ inline Conv2dDwKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR xp, GM_ADDR w, GM_ADDR bias,
                                GM_ADDR y, int32_t batch, int32_t ci,
                                int32_t co, int32_t coG, int32_t H, int32_t W,
                                int32_t wpad, int32_t kH, int32_t kW,
                                int32_t sH, int32_t pH, int32_t pW,
                                int32_t dH, int32_t dW, int32_t Ho, int32_t Wo,
                                int32_t woPad, int32_t rowsSlot,
                                int32_t bandOh, int32_t haloL,
                                int32_t hasBias, int32_t totalTasks,
                                AscendC::TPipe *pipe)
    {
        batch_ = batch;
        ci_ = ci;
        co_ = co;
        coG_ = coG;
        H_ = H;
        W_ = W;
        wpad_ = wpad;
        kH_ = kH;
        kW_ = kW;
        sH_ = sH;
        pH_ = pH;
        pW_ = pW;
        dH_ = dH;
        dW_ = dW;
        Ho_ = Ho;
        Wo_ = Wo;
        woPad_ = woPad;
        rowsSlot_ = rowsSlot;
        bandOh_ = bandOh;
        haloL_ = haloL;
        hasBias_ = hasBias;
        totalTasks_ = totalTasks;

        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x),
                             static_cast<int64_t>(batch_) * ci_ * H_ * W_);
        xpGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(xp),
                              static_cast<int64_t>(batch_) * ci_ * H_ * wpad_);
        wGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(w),
                             static_cast<int64_t>(co_) * kH_ * kW_);
        bGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias), co_);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y),
                             static_cast<int64_t>(batch_) * co_ * Ho_ * Wo_);

        // UB-side row pitch of the 2D DataCopyPad staging load is
        // roundUp32(W*es) (dstStride=0 hardware rounding); the fp32 staging
        // store expects roundUp32(W*4) (srcStride=0). Per-row Casts bridge
        // the two pitches.
        pitchT_ = ((W_ * static_cast<int32_t>(sizeof(T)) + 31) / 32 * 32) /
                  static_cast<int32_t>(sizeof(T));
        pitchF_ = (W_ + 7) / 8 * 8;
        // per-row store src must be 32B aligned: outT row pitch in T elems
        pitchO_ = (woPad_ * static_cast<int32_t>(sizeof(T)) + 31) / 32 * 32 /
                  static_cast<int32_t>(sizeof(T));

        pipe->InitBuffer(stageTBuf_, DW_STAGE_ELEMS * sizeof(T));
        pipe->InitBuffer(stageFBuf_, DW_STAGE_ELEMS * sizeof(float));
        pipe->InitBuffer(winBuf_, static_cast<int64_t>(kW_) * rowsSlot_ *
                                      woPad_ * sizeof(float));
        pipe->InitBuffer(wTBuf_, DW_MAX_K * sizeof(T));
        pipe->InitBuffer(wFBuf_, DW_MAX_K * sizeof(float));
        pipe->InitBuffer(biasTBuf_, co_ * sizeof(T));
        pipe->InitBuffer(biasFBuf_, co_ * sizeof(float));
        pipe->InitBuffer(bandOutBuf_, static_cast<int64_t>(bandOh_) * woPad_ *
                                          sizeof(float));
        pipe->InitBuffer(outTBuf_, static_cast<int64_t>(bandOh_) * pitchO_ *
                                       sizeof(T));

        evMte2V_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        evStageV_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        evMte3Mte2_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        evVMte2_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        evV2Mte3_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        evMte32V_ = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        int32_t numCores = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        int32_t coreIdx = AscendC::GetBlockIdx();

        LocalTensor<T> stageT = stageTBuf_.Get<T>();
        LocalTensor<float> stageF = stageFBuf_.Get<float>();
        LocalTensor<float> win = winBuf_.Get<float>();
        LocalTensor<T> wT = wTBuf_.Get<T>();
        LocalTensor<float> wF = wFBuf_.Get<float>();
        LocalTensor<T> biasT = biasTBuf_.Get<T>();
        LocalTensor<float> biasF = biasFBuf_.Get<float>();
        LocalTensor<float> bandOut = bandOutBuf_.Get<float>();
        LocalTensor<T> outT = outTBuf_.Get<T>();

        // whole-bias load + cast once (all cores), mirrors the archive
        // epilogue bias handling
        if (hasBias_ != 0) {
            DataCopyExtParams bcp{1, static_cast<uint32_t>(co_ * sizeof(T)),
                                  0, 0, 0};
            DataCopyPadExtParams<T> bpp{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(biasT, bGM_, bcp, bpp);
            SetFlag<HardEvent::MTE2_V>(evMte2V_);
            WaitFlag<HardEvent::MTE2_V>(evMte2V_);
            AscendC::Cast(biasF, biasT, AscendC::RoundMode::CAST_NONE, co_);
        }

        const int32_t es = static_cast<int32_t>(sizeof(T));
        int32_t rowsPerChunk = DW_STAGE_ELEMS / pitchT_;
        if (rowsPerChunk > H_) { rowsPerChunk = H_; }
        if (rowsPerChunk < 1) { rowsPerChunk = 1; }  // host gates W <= 8192

        for (int32_t task = coreIdx; task < totalTasks_; task += numCores) {
            int32_t co = task % co_;
            int32_t n = task / co_;
            int32_t g = co / coG_;
            int64_t srcPlane = (static_cast<int64_t>(n) * ci_ + g) * H_ * W_;
            int64_t dstPlane = (static_cast<int64_t>(n) * ci_ + g) * H_ * wpad_;

            // ---- stage plane into fp32 xp scratch (zero halo) ----
            for (int32_t c0 = 0; c0 < H_; c0 += rowsPerChunk) {
                int32_t rc = H_ - c0 < rowsPerChunk ? H_ - c0 : rowsPerChunk;
                DataCopyExtParams lcp{static_cast<uint16_t>(rc),
                                      static_cast<uint32_t>(W_ * es), 0, 0, 0};
                DataCopyPadExtParams<T> lpp{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(stageT, xGM_[srcPlane + c0 * W_], lcp, lpp);
                // AIV has no MTE2_MTE3 event (#10): order the UB->GM store
                // after the GM->UB load via the MTE2_V + V_MTE3 chain
                SetFlag<HardEvent::MTE2_V>(evMte2V_);
                WaitFlag<HardEvent::MTE2_V>(evMte2V_);
                if (pitchT_ == W_) {
                    // rows packed contiguously (W*es 32B-aligned): one cast
                    AscendC::Cast(stageF, stageT,
                                  AscendC::RoundMode::CAST_NONE, rc * W_);
                } else {
                    // rows sit at the 32B-rounded hardware pitch; bridge
                    // pitchT_ -> pitchF_ per row (gap contents are garbage)
                    for (int32_t r = 0; r < rc; ++r) {
                        AscendC::Cast(stageF[r * pitchF_], stageT[r * pitchT_],
                                      AscendC::RoundMode::CAST_NONE, W_);
                    }
                }
                SetFlag<HardEvent::V_MTE3>(evStageV_);
                WaitFlag<HardEvent::V_MTE3>(evStageV_);
                DataCopyExtParams scp{static_cast<uint16_t>(rc),
                                      static_cast<uint32_t>(W_ * 4), 0,
                                      static_cast<uint32_t>((wpad_ - W_) * 4),
                                      0};
                DataCopyPad(xpGM_[dstPlane + c0 * wpad_ + haloL_], stageF,
                            scp);
                SetFlag<HardEvent::MTE3_MTE2>(evMte3Mte2_);
                WaitFlag<HardEvent::MTE3_MTE2>(evMte3Mte2_);
            }

            // ---- per-task weights (kH*kW <= 64), cast to fp32 ----
            {
                DataCopyExtParams wcp{1,
                    static_cast<uint32_t>(kH_ * kW_ * es), 0, 0, 0};
                DataCopyPadExtParams<T> wpp{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(wT, wGM_[static_cast<int64_t>(co) * kH_ * kW_],
                            wcp, wpp);
            }
            SetFlag<HardEvent::MTE2_V>(evMte2V_);
            WaitFlag<HardEvent::MTE2_V>(evMte2V_);
            AscendC::Cast(wF, wT, AscendC::RoundMode::CAST_NONE, DW_MAX_K);
            // scalar reads below: WaitFlag<MTE2_V> only queues a V-pipe
            // dependency and does NOT stall the scalar issue stream;
            // PipeBarrier<PIPE_V> stalls it until the queued Cast (and hence
            // the MTE2 weight/bias loads) has completed
            PipeBarrier<PIPE_V>();
            float wv[DW_MAX_K];
            for (int32_t i = 0; i < kH_ * kW_; ++i) {
                wv[i] = wF.GetValue(i);
            }
            float biasVal = hasBias_ != 0 ? biasF.GetValue(co) : 0.0f;

            // ---- bands of output rows ----
            for (int32_t oh0 = 0; oh0 < Ho_; oh0 += bandOh_) {
                int32_t oh1 = oh0 + bandOh_ < Ho_ ? oh0 + bandOh_ : Ho_;
                int32_t rowsLo = oh0 * sH_ - pH_;
                int32_t rowsHi = (oh1 - 1) * sH_ - pH_ + (kH_ - 1) * dH_;
                int32_t effLo = rowsLo > 0 ? rowsLo : 0;
                int32_t effHi = rowsHi < H_ - 1 ? rowsHi : H_ - 1;
                int32_t nRows = effHi - effLo + 1;
                int32_t slotOff = effLo - rowsLo;

                // drain V pipe before reusing win / bandOut (immediate pair)
                SetFlag<HardEvent::V_MTE2>(evVMte2_);
                WaitFlag<HardEvent::V_MTE2>(evVMte2_);
                for (int32_t j = 0; j < kW_; ++j) {
                    int32_t srcOff = haloL_ - pW_ + j * dW_;
                    int64_t jBase = static_cast<int64_t>(j) * rowsSlot_ *
                                    woPad_;
                    for (int32_t r = 0; r < nRows; ++r) {
                        DataCopyExtParams cp{1,
                            static_cast<uint32_t>(woPad_ * sizeof(float)),
                            0, 0, 0};
                        DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
                        DataCopyPad(win[jBase + (slotOff + r) * woPad_],
                                    xpGM_[dstPlane +
                                          (effLo + r) * wpad_ + srcOff],
                                    cp, pp);
                    }
                }
                SetFlag<HardEvent::MTE2_V>(evMte2V_);
                WaitFlag<HardEvent::MTE2_V>(evMte2V_);

                // compute each output row into its own bandOut slot
                for (int32_t oh = oh0; oh < oh1; ++oh) {
                    LocalTensor<float> acc = bandOut[(oh - oh0) * woPad_];
                    Duplicate(acc, biasVal, woPad_);
                    for (int32_t kh = 0; kh < kH_; ++kh) {
                        int32_t ih = oh * sH_ - pH_ + kh * dH_;
                        if (ih < 0 || ih >= H_) {
                            continue;
                        }
                        int32_t slot = ih - rowsLo;
                        for (int32_t kw = 0; kw < kW_; ++kw) {
                            int64_t wbase =
                                static_cast<int64_t>(kw) * rowsSlot_ *
                                    woPad_ +
                                static_cast<int64_t>(slot) * woPad_;
                            Axpy<float, float>(acc, win[wbase],
                                               wv[kh * kW_ + kw], woPad_);
                        }
                    }
                }
                // cast fp32 acc rows back to T (CAST_RINT, same as epilogue)
                for (int32_t oh = oh0; oh < oh1; ++oh) {
                    AscendC::Cast(outT[(oh - oh0) * pitchO_],
                                  bandOut[(oh - oh0) * woPad_],
                                  AscendC::RoundMode::CAST_RINT, woPad_);
                }
                SetFlag<HardEvent::V_MTE3>(evV2Mte3_);
                WaitFlag<HardEvent::V_MTE3>(evV2Mte3_);
                // per-row stores: multi-row UB->GM DataCopyPad with
                // dstStride==0 only lands the first row on this platform
                // (v3 measured; dstStride>0 form not applicable to the
                // contiguous y layout)
                int64_t yBase = (static_cast<int64_t>(n) * co_ + co) * Ho_ *
                                Wo_;
                for (int32_t oh = oh0; oh < oh1; ++oh) {
                    DataCopyExtParams ocp{1,
                        static_cast<uint32_t>(Wo_ * es), 0, 0, 0};
                    DataCopyPad(yGM_[yBase + oh * Wo_],
                                outT[(oh - oh0) * pitchO_], ocp);
                }
                // drain the band store before the next band/task reuses
                // bandOut / outT (immediate pair)
                SetFlag<HardEvent::MTE3_V>(evMte32V_);
                WaitFlag<HardEvent::MTE3_V>(evMte32V_);
            }
        }
    }

private:
    int32_t batch_;
    int32_t ci_;
    int32_t co_;
    int32_t coG_;
    int32_t H_;
    int32_t W_;
    int32_t wpad_;
    int32_t kH_;
    int32_t kW_;
    int32_t sH_;
    int32_t pH_;
    int32_t pW_;
    int32_t dH_;
    int32_t dW_;
    int32_t Ho_;
    int32_t Wo_;
    int32_t woPad_;
    int32_t rowsSlot_;
    int32_t bandOh_;
    int32_t haloL_;
    int32_t hasBias_;
    int32_t totalTasks_;
    int32_t pitchT_;
    int32_t pitchF_;
    int32_t pitchO_;

    int32_t evMte2V_;
    int32_t evStageV_;
    int32_t evMte3Mte2_;
    int32_t evVMte2_;
    int32_t evV2Mte3_;
    int32_t evMte32V_;

    GlobalTensor<T> xGM_;
    GlobalTensor<float> xpGM_;
    GlobalTensor<T> wGM_;
    GlobalTensor<T> bGM_;
    GlobalTensor<T> yGM_;
    TBuf<TPosition::VECCALC> stageTBuf_;
    TBuf<TPosition::VECCALC> stageFBuf_;
    TBuf<TPosition::VECCALC> winBuf_;
    TBuf<TPosition::VECCALC> wTBuf_;
    TBuf<TPosition::VECCALC> wFBuf_;
    TBuf<TPosition::VECCALC> biasTBuf_;
    TBuf<TPosition::VECCALC> biasFBuf_;
    TBuf<TPosition::VECCALC> bandOutBuf_;
    TBuf<TPosition::VECCALC> outTBuf_;
};

#endif // CONV2D_BASE_DW_H
