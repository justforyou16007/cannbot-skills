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
 * @file conv2d_base_wrepack.h
 * Stage-0w (AIV-only) weight repack for conv2d_base.
 *   raw weight [Co, Ci, kH, kW]  ->  wOut [Kpad, CoPad], k order = (kh, kw, ci)
 *   Kpad = kH*kW*CiPad, CiPad = ceil16(Ci), CoPad = ceil16(Co); tails zero-filled.
 * Per (ci-block, co-block of 16): builds a [16 co][kHkW] UB tile via
 * DataCopyPad (arbitrary byte offsets / lengths), transposes it in UB with
 * TransDataTo5HD, then stores rows of 16 co to the K-major workspace.
 * Replaces the former host-side at::permute + constant_pad_nd.
 */

#ifndef CONV2D_BASE_WREPACK_H
#define CONV2D_BASE_WREPACK_H

#include "kernel_operator.h"
#include "conv2d_base_tiling.h"

using namespace AscendC;

template <typename T>
class Conv2dWeightRepackKernel {
    static constexpr uint32_t C0 = CONV_C0;
    static constexpr uint32_t MAX_KK = 64;  // kHkW padded to 16, supports kHkW <= 64

public:
    __aicore__ inline Conv2dWeightRepackKernel() {}

    __aicore__ inline void Init(GM_ADDR w, GM_ADDR wOut,
                                int32_t Co, int32_t Ci, int32_t kHkW,
                                int32_t CiPad, int32_t CoPad, int32_t fused,
                                AscendC::TPipe *pipe)
    {
        Co_ = Co;
        Ci_ = Ci;
        kHkW_ = kHkW;
        CiPad_ = CiPad;
        CoPad_ = CoPad;
        // P5: the img2col LoadData3D expands k as (ci1, kh, kw, ci0) — the
        // C0 channel block is OUTERMOST (measured). fused=1 emits that
        // order; fused=0 keeps the im2col-kernel order (kh, kw, ci).
        fused_ = fused;
        repeats_ = (kHkW_ + C0 - 1) / C0;
        rowElems_ = repeats_ * C0;

        wGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(w),
                             static_cast<int64_t>(Co_) * Ci_ * kHkW_);
        outGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(wOut),
                               static_cast<int64_t>(CiPad_) * kHkW_ * CoPad_);

        pipe->InitBuffer(tileBuf_, 16 * rowElems_ * sizeof(T));
        pipe->InitBuffer(transBuf_, 16 * rowElems_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        int32_t ciBlocks = CiPad_ / C0;
        int32_t coBlocks = CoPad_ / C0;
        int32_t totalTiles = ciBlocks * coBlocks;
        int32_t numCores = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        int32_t coreIdx = AscendC::GetBlockIdx();

        int32_t evMte2V = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        int32_t evVMte3 = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        int32_t evMte3Mte2 = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        int32_t evMte3V = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));

        AscendC::LocalTensor<T> tile = tileBuf_.Get<T>();
        AscendC::LocalTensor<T> trans = transBuf_.Get<T>();

        // P1.9 (v3 R3/R6 port): batch DMA calls. The per-c inner loop used
        // 16 tiny per-row loads + kHkW per-row 32B stores (~26 calls per c,
        // ~416 per tile; call-count bound at ~60-70ns/call => ~20-27us cold).
        // Now: ONE 2D DataCopyPad per c loads all 16 co rows (rows pack at
        // the 32B-rounded pitch, which equals rowElems for 2B dtypes), and
        // ONE 2D DataCopyPad per c stores all kHkW rows (srcStride=0 packed,
        // dstStride>0 — the verified multi-row form). The transpose pointer
        // lists are tile/c-independent (persistent buffers) -> hoisted.
        using TransT = typename std::conditional<std::is_same<T, half>::value,
                                                 half, uint16_t>::type;

        if (kHkW_ == 1) {
            // 1x1 path: the repack degenerates to a [Co, Ci] transpose.
            // Per (cib, cob) tile: ONE 2D DataCopyPad load, one transpose,
            // 16 pipelined row stores. NOTE: a full-Co slab variant (16
            // tasks, one contiguous store per task) was measured and
            // REVERTED — better steady (16.9 vs ~20us) but consistently
            // worse in the official cold single-call metric (case9
            // 53.3-56.5 vs 52.2us baseline across 3 runs); the cold-metric
            // floor is the first-launch inflation + TransDataTo5HD call
            // count, which the slab does not reduce.
            AscendC::LocalTensor<TransT> srcList1[16];
            AscendC::LocalTensor<TransT> dstList1[16];
            for (uint32_t i = 0; i < 16; ++i) {
                srcList1[i] = tile[i * C0].template ReinterpretCast<TransT>();
                dstList1[i] = trans[C0 * i].template ReinterpretCast<TransT>();
            }
            for (int32_t t = coreIdx; t < totalTiles; t += numCores) {
                int32_t cib = t % ciBlocks;
                int32_t cob = t / ciBlocks;
                int32_t validCo = Co_ - cob * C0;
                validCo = validCo < static_cast<int32_t>(C0) ? validCo : static_cast<int32_t>(C0);
                int32_t validCi = Ci_ - cib * C0;
                validCi = validCi < static_cast<int32_t>(C0) ? validCi : static_cast<int32_t>(C0);

                // zero the co-tail rows (pad columns of the K-major output);
                // V-pipe op, disjoint rows from the MTE2 copy below, and the
                // same-pipe transpose afterwards is program-ordered
                if (validCo < static_cast<int32_t>(C0)) {
                    AscendC::Duplicate(tile[validCo * C0], static_cast<T>(0),
                                       (C0 - validCo) * C0);
                }
                // load [validCo rows][16 ci] starting at (cob*16, cib*16);
                // ci tail zero-filled via rightPadding (each row -> 32B)
                {
                    int64_t srcOff = static_cast<int64_t>(cob) * C0 * Ci_ + cib * C0;
                    AscendC::DataCopyExtParams cp{static_cast<uint16_t>(validCo),
                        static_cast<uint32_t>(validCi * sizeof(T)),
                        static_cast<uint32_t>((Ci_ - validCi) * sizeof(T)), 0, 0};
                    AscendC::DataCopyPadExtParams<T> pp{true, 0,
                        static_cast<uint8_t>(C0 - validCi), static_cast<T>(0)};
                    AscendC::DataCopyPad(tile, wGM_[srcOff], cp, pp);
                }
                SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
                WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);

                // transpose [16 co][16 ci] -> [16 ci][16 co]
                AscendC::TransDataTo5HDParams params(false, false, 1, 0, 0);
                AscendC::TransDataTo5HD<TransT>(dstList1, srcList1, params);
                SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
                WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3);

                // store 16 rows: out[(cib*16 + ci) * CoPad + cob*16] (kk = 0)
                // NOTE: a 2D batched store (blockCount=16, dstStride=CoPad)
                // measured SLOWER here than 16 pipelined row stores
                // (case9 Ci=256/Co=512: cold 24.0 -> 27.3us, reverted)
                int64_t dstBase = static_cast<int64_t>(cib) * C0 * CoPad_ +
                                  cob * C0;
                for (uint32_t ci = 0; ci < C0; ++ci) {
                    AscendC::DataCopy(outGM_[dstBase + ci * CoPad_],
                                      trans[ci * C0], C0);
                }
                SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2);
                WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2);
                // MTE3 store reads trans; next tile's V (transpose)
                // overwrites it -> MTE3 must drain before V proceeds
                SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V);
                WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V);
            }
            return;
        }

        // general path (kHkW > 1): per (cib, cob) tile, per c: ONE 2D load +
        // transpose + ONE 2D store with per-c syncs, so the pipes of
        // consecutive c iterations overlap (load c+1 during transpose c).
        // A fully batched variant (stage all 16 c, one sync per pipe)
        // measured slightly WORSE (case0 steady 12.3 -> 13.0us): it
        // serializes the MTE2 loads against the V transposes.
        uint16_t dstRS = (repeats_ == 1) ? 0 : 16;
        uint16_t srcRS = (repeats_ == 1) ? 0 : 1;
        // per-kk dst row stride in the K-major ws (elements)
        const int64_t kkStride = fused_ ? static_cast<int64_t>(C0) * CoPad_
                                        : static_cast<int64_t>(CiPad_) * CoPad_;
        AscendC::LocalTensor<TransT> srcList[16];
        AscendC::LocalTensor<TransT> dstList[16];
        for (uint32_t i = 0; i < 16; ++i) {
            srcList[i] = tile[i * rowElems_].template ReinterpretCast<TransT>();
            dstList[i] = trans[C0 * i].template ReinterpretCast<TransT>();
        }

        for (int32_t t = coreIdx; t < totalTiles; t += numCores) {
            int32_t cib = t % ciBlocks;
            int32_t cob = t / ciBlocks;
            int32_t validCo = Co_ - cob * C0;
            validCo = validCo < static_cast<int32_t>(C0) ? validCo
                                                         : static_cast<int32_t>(C0);

            // every ci of the 16-channel block: [16 co][kk] 2D load +
            // transpose + [kk][16 co] 2D store
            for (uint32_t c = 0; c < C0; ++c) {
                int32_t ci = cib * C0 + static_cast<int32_t>(c);

                if (ci < Ci_) {
                    // zero the co-tail rows (keep pad co columns defined)
                    if (validCo < static_cast<int32_t>(C0)) {
                        for (int32_t r = validCo; r < static_cast<int32_t>(C0);
                             ++r) {
                            AscendC::Duplicate(tile[r * rowElems_],
                                               static_cast<T>(0), rowElems_);
                        }
                    }
                    // ONE 2D load: [validCo co rows][kHkW] from
                    // w[cob*16 + r][ci][0]; rows pack at the 32B-rounded
                    // pitch == rowElems (2B dtypes). In-row pad columns
                    // [kHkW, rowElems) keep stale data — they transpose into
                    // trans rows kk >= kHkW, which are never stored.
                    int64_t srcOff = (static_cast<int64_t>(cob) * C0 * Ci_ +
                                      ci) * kHkW_;
                    AscendC::DataCopyExtParams cp{static_cast<uint16_t>(validCo),
                        static_cast<uint32_t>(kHkW_ * sizeof(T)),
                        static_cast<uint32_t>((Ci_ - 1) * kHkW_ * sizeof(T)),
                        0, 0};
                    AscendC::DataCopyPadExtParams<T> pp{false, 0, 0,
                                                        static_cast<T>(0)};
                    AscendC::DataCopyPad(tile, wGM_[srcOff], cp, pp);
                } else {
                    // ci tail: the K rows for ci >= Ci MUST be zero — the
                    // im2col ws pads those columns with zeros and 0 x NaN
                    // (stale GM) would poison the GEMM accumulator
                    AscendC::Duplicate(tile, static_cast<T>(0),
                                       C0 * rowElems_);
                }
                SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
                WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);

                // transpose [16 co][kk] -> [kk][16 co]
                AscendC::TransDataTo5HDParams params(false, false,
                    static_cast<uint8_t>(repeats_), dstRS, srcRS);
                AscendC::TransDataTo5HD<TransT>(dstList, srcList, params);
                SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
                WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3);

                // ONE 2D store: rows kk = 0..kHkW-1, src packed (trans rows
                // are contiguous 32B blocks), dst row stride kkStride elems
                // (> C0 always, so dstStride > 0). kIdx is (kk*CiPad + ci)
                // in im2col order, ((cib*kHkW + kk)*C0 + c) in the P5 fused
                // img2col-LoadData3D order (ci1 outermost).
                int64_t dstBase = fused_
                    ? (static_cast<int64_t>(cib) * kHkW_ * C0 + c) * CoPad_ +
                      cob * C0
                    : static_cast<int64_t>(ci) * CoPad_ + cob * C0;
                AscendC::DataCopyExtParams sp{static_cast<uint16_t>(kHkW_),
                    static_cast<uint32_t>(C0 * sizeof(T)), 0,
                    static_cast<uint32_t>((kkStride - C0) * sizeof(T)), 0};
                AscendC::DataCopyPad(outGM_[dstBase], trans, sp);
                SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2);
                WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2);
                // MTE3 store reads trans; next sub-iteration's V (transpose)
                // overwrites it -> MTE3 must drain before V proceeds
                SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V);
                WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V);
            }
        }
    }

private:
    int32_t Co_;
    int32_t Ci_;
    int32_t kHkW_;
    int32_t CiPad_;
    int32_t CoPad_;
    int32_t fused_ = 0;
    uint32_t repeats_;
    uint32_t rowElems_;

    AscendC::GlobalTensor<T> wGM_;
    AscendC::GlobalTensor<T> outGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tileBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> transBuf_;
};

#endif // CONV2D_BASE_WREPACK_H
