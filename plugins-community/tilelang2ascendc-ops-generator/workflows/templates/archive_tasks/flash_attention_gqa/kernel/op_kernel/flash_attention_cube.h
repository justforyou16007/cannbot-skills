#ifndef FLASH_ATTENTION_CUBE_H
#define FLASH_ATTENTION_CUBE_H

#include "kernel_operator.h"
#include "matmul_tile.h"
#include "flash_attention_tiling.h"
#include "kernel_common.h"
#include "workspace_queue.h"

using namespace AscendC;

template <typename QType>
class FlashAttentionCube {
    static constexpr uint32_t C0 = 32 / sizeof(QType);  // 16
    static constexpr uint32_t Q_L1_ROWS = 256;
    static constexpr uint32_t Q_L1_COLS = 192;
    static constexpr uint32_t V_L1_ROWS = 256;
    static constexpr uint32_t KP_L1_COLS = 256;
    static constexpr uint32_t Q_L1_BUF_ELEMS = Q_L1_ROWS * Q_L1_COLS;
    static constexpr uint32_t V_L1_BUF_ELEMS = V_L1_ROWS * BASE_K;
    static constexpr uint32_t KP_L1_BUF_ELEMS = BLOCK_N * KP_L1_COLS;
    static constexpr uint32_t L0_AB_BUF_COUNT = 2;
    static constexpr uint32_t L0_C_BUF_COUNT = 2;
    static constexpr uint32_t L0A_BUF_ELEMS = BLOCK_M * BASE_K;
    static constexpr uint32_t L0B_BUF_ELEMS = BASE_K * BLOCK_N;
    static constexpr uint32_t L0C_BUF_ELEMS = BLOCK_M * BASE_K;

public:
    __aicore__ inline FlashAttentionCube() {}

    __aicore__ inline void Init(const FlashAttentionTiling &tiling,
                                GlobalTensor<QType> &qGm, GlobalTensor<QType> &kGm, GlobalTensor<QType> &vGm,
                                GlobalTensor<float> &wsSGm, GlobalTensor<QType> &wsPGm,
                                GlobalTensor<float> &wsOGm)
    {
        tiling_ = tiling;
        qGm_ = qGm;
        kGm_ = kGm;
        vGm_ = vGm;
        wsSGm_ = wsSGm;
        wsPGm_ = wsPGm;
        wsOGm_ = wsOGm;
        dimAlign_ = AlignUp(tiling_.dim, C0);
        qL1BlockM_ = dimAlign_ <= 192 ? static_cast<uint32_t>(tiling_.mBaseSize) : BLOCK_M;
        if (qL1BlockM_ < BLOCK_M) {
            qL1BlockM_ = BLOCK_M;
        }
        sQueue_.Init(wsSGm_, tiling_.mBaseSize * tiling_.s2BaseSize, SIG_S_READY);
        pQueue_.Init(wsPGm_, tiling_.mBaseSize * tiling_.s2BaseSize, SIG_P_READY);
        oQueue_.Init(wsOGm_, tiling_.mBaseSize * dimAlign_, SIG_O_READY);
    }

    __aicore__ inline void InitBuffers(TPipe &pipe)
    {
        pipe.InitBuffer(qBufL1_, 2 * Q_L1_BUF_ELEMS * sizeof(QType));
        pipe.InitBuffer(vQueL1_, 2, V_L1_BUF_ELEMS * sizeof(QType));
        pipe.InitBuffer(kpQueL1_, 3, KP_L1_BUF_ELEMS * sizeof(QType));
        pipe.InitBuffer(l0ABuf_, L0_AB_BUF_COUNT * L0A_BUF_ELEMS * sizeof(QType));
        pipe.InitBuffer(l0BBuf_, L0_AB_BUF_COUNT * L0B_BUF_ELEMS * sizeof(QType));
        pipe.InitBuffer(l0CBuf_, L0_C_BUF_COUNT * L0C_BUF_ELEMS * sizeof(float));
        LocalTensor<QType> l0ABase = l0ABuf_.Get<QType>();
        LocalTensor<QType> l0BBase = l0BBuf_.Get<QType>();
        LocalTensor<float> l0CBase = l0CBuf_.Get<float>();
        for (uint32_t i = 0; i < L0_AB_BUF_COUNT; ++i) {
            l0A_[i] = l0ABase[i * L0A_BUF_ELEMS];
            l0B_[i] = l0BBase[i * L0B_BUF_ELEMS];
            l0AFreeEvent_[i] = pipe.AllocEventID<HardEvent::M_MTE1>();
            l0BFreeEvent_[i] = pipe.AllocEventID<HardEvent::M_MTE1>();
        }
        for (uint32_t i = 0; i < L0_C_BUF_COUNT; ++i) {
            l0C_[i] = l0CBase[i * L0C_BUF_ELEMS];
            l0CFreeEvent_[i] = pipe.AllocEventID<HardEvent::FIX_M>();
        }
        // Q load relies on the same implicit ordering as the baseline LoadQ:
        // DataCopy(Q) is issued before DataCopy(K) on the MTE2 pipe, and ComputeMM1's
        // kpQueL1 DeQue(K) guarantees K is loaded before MTE1 reads Q.
    }

    __aicore__ inline void BeginL0BufferEvents()
    {
        l0ABufId_ = 0;
        l0CBufId_ = 0;
        for (uint32_t i = 0; i < L0_AB_BUF_COUNT; ++i) {
            SetFlag<HardEvent::M_MTE1>(l0AFreeEvent_[i]);
            SetFlag<HardEvent::M_MTE1>(l0BFreeEvent_[i]);
        }
        for (uint32_t i = 0; i < L0_C_BUF_COUNT; ++i) {
            SetFlag<HardEvent::FIX_M>(l0CFreeEvent_[i]);
        }
    }

    __aicore__ inline void EndL0BufferEvents()
    {
        for (uint32_t i = 0; i < L0_AB_BUF_COUNT; ++i) {
            WaitFlag<HardEvent::M_MTE1>(l0AFreeEvent_[i]);
            WaitFlag<HardEvent::M_MTE1>(l0BFreeEvent_[i]);
            GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(l0AFreeEvent_[i]);
            GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(l0BFreeEvent_[i]);
        }
        for (uint32_t i = 0; i < L0_C_BUF_COUNT; ++i) {
            WaitFlag<HardEvent::FIX_M>(l0CFreeEvent_[i]);
            GetTPipePtr()->ReleaseEventID<HardEvent::FIX_M>(l0CFreeEvent_[i]);
        }
    }

    __aicore__ inline void PrepareQ(const TaskInfo &t)
    {
        LoadQInternal(t.bz, t.qHead, t.qRowStart, t.qRows);
    }

    __aicore__ inline void ReleaseQ()
    {
    }

    __aicore__ inline void LoadQInternal(uint32_t bz, uint32_t qHead, uint32_t qRowStart, uint32_t qRows)
    {
        uint32_t dim = tiling_.dim;
        uint32_t qSeqLen = tiling_.qSeqLen;
        uint32_t qChunkRows = MinU32(qRows, qL1BlockM_);
        uint64_t qOffset = ((uint64_t)bz * tiling_.nq * qSeqLen
                          + (uint64_t)qHead * qSeqLen
                          + (uint64_t)qRowStart) * dim;
        LocalTensor<QType> qL1 = qBufL1_.Get<QType>();
        uint32_t firstRows = MinU32(qChunkRows, Q_L1_ROWS);
        LoadNdGmToNzL1(qL1, qGm_[qOffset], firstRows, dim, dim, Q_L1_ROWS);
        if (qChunkRows > Q_L1_ROWS) {
            uint32_t secondRows = qChunkRows - Q_L1_ROWS;
            LoadNdGmToNzL1(qL1[Q_L1_BUF_ELEMS], qGm_[qOffset + (uint64_t)Q_L1_ROWS * dim],
                           secondRows, dim, dim, Q_L1_ROWS);
        }
    }

    __aicore__ inline void ComputeMM1(const TaskInfo &t)
    {
        int bz = t.bz;
        int kvHead = t.kvHead;
        uint32_t s2Start = t.s2Start;
        uint32_t s2Rows = t.s2Rows;
        uint32_t qRows = t.qRows;
        // qLocalOffset stays 0 (Commit 4 is pure refactor; no q-splitting yet).
        auto sSlot = sQueue_.ProducerAcquire();
        uint32_t dim = tiling_.dim;
        uint32_t kvSeqLen = tiling_.kvSeqLen;
        uint32_t outerCols = tiling_.s2BaseSize;
        uint32_t nL1Size = (dim <= BASE_K) ? 256U : BLOCK_N;
        uint32_t kTiles = dim / BASE_K;
        LocalTensor<QType> qL1 = qBufL1_.Get<QType>();

        for (uint32_t nL1Start = 0; nL1Start < s2Rows; nL1Start += nL1Size) {
            uint32_t kvRows = MinU32(nL1Size, s2Rows - nL1Start);
            uint32_t kvRowsAlign = AlignUp(kvRows, C0);
            uint64_t kOffset = ((uint64_t)bz * tiling_.nkv * kvSeqLen
                              + (uint64_t)kvHead * kvSeqLen
                              + (uint64_t)(s2Start + nL1Start)) * dim;
            auto kvL1 = kpQueL1_.AllocTensor<QType>();
            LoadNdGmToNzL1(kvL1, kGm_[kOffset], kvRows, dim, dim, kvRowsAlign);
            kpQueL1_.EnQue(kvL1);
            kvL1 = kpQueL1_.DeQue<QType>();

            for (uint32_t mL1Start = 0; mL1Start < qRows; mL1Start += Q_L1_ROWS) {
                uint32_t mL1Rows = MinU32(Q_L1_ROWS, qRows - mL1Start);
                uint32_t qBufId = mL1Start / Q_L1_ROWS;

                for (uint32_t mL0Start = 0; mL0Start < mL1Rows; mL0Start += BLOCK_M) {
                    uint32_t mRows = MinU32(BLOCK_M, mL1Rows - mL0Start);
                    uint32_t mRowsAlign = AlignUp(mRows, C0);
                    uint32_t qBufRowOffset = mL0Start * C0;

                    for (uint32_t nL0Start = 0; nL0Start < kvRows; nL0Start += BLOCK_N) {
                        uint32_t nRows = MinU32(BLOCK_N, kvRows - nL0Start);
                        uint32_t nRowsAlign = AlignUp(nRows, C0);
                        uint32_t cBufId = l0CBufId_;
                        WaitFlag<HardEvent::FIX_M>(l0CFreeEvent_[cBufId]);
                        LocalTensor<float> cL0 = l0C_[cBufId];
                        bool isFirstK = true;

                        for (uint32_t ki = 0; ki < kTiles; ++ki) {
                            uint32_t abBufId = l0ABufId_;
                            WaitFlag<HardEvent::M_MTE1>(l0AFreeEvent_[abBufId]);
                            WaitFlag<HardEvent::M_MTE1>(l0BFreeEvent_[abBufId]);
                            LocalTensor<QType> aL0 = l0A_[abBufId];
                            uint64_t qL1Off = (uint64_t)qBufId * Q_L1_BUF_ELEMS +
                                              (uint64_t)ki * Q_L1_ROWS * BASE_K + qBufRowOffset;
                            LoadNzL1ToZzL0A(aL0, qL1[qL1Off], mRowsAlign, BASE_K, Q_L1_ROWS);

                            LocalTensor<QType> bL0 = l0B_[abBufId];
                            uint64_t kL1Off = (uint64_t)ki * kvRowsAlign * BASE_K + (uint64_t)C0 * nL0Start;
                            LoadNzL1ToZnL0BWithTranspose(bL0, kvL1[kL1Off], BASE_K, nRowsAlign, kvRowsAlign);

                            SetWaitFlag<HardEvent::MTE1_M>();

                            MmadParams mp;
                            mp.m = mRowsAlign;
                            mp.n = nRowsAlign;
                            mp.k = BASE_K;
                            mp.cmatrixInitVal = isFirstK;
                            mp.cmatrixSource = false;
                            if (!isFirstK) { PipeBarrier<PIPE_M>(); }
                            Mmad(cL0, aL0, bL0, mp);
                            SetFlag<HardEvent::M_MTE1>(l0AFreeEvent_[abBufId]);
                            SetFlag<HardEvent::M_MTE1>(l0BFreeEvent_[abBufId]);
                            l0ABufId_ = 1 - abBufId;
                            isFirstK = false;
                        }

                        uint32_t kRemain = dim % BASE_K;
                        if (kRemain > 0) {
                            uint32_t kAligned = AlignUp(kRemain, C0);

                            uint32_t abBufId = l0ABufId_;
                            WaitFlag<HardEvent::M_MTE1>(l0AFreeEvent_[abBufId]);
                            WaitFlag<HardEvent::M_MTE1>(l0BFreeEvent_[abBufId]);
                            LocalTensor<QType> aL0 = l0A_[abBufId];
                            uint64_t qL1Off = (uint64_t)qBufId * Q_L1_BUF_ELEMS +
                                              (uint64_t)kTiles * Q_L1_ROWS * BASE_K + qBufRowOffset;
                            LoadNzL1ToZzL0A(aL0, qL1[qL1Off], mRowsAlign, kAligned, Q_L1_ROWS);

                            LocalTensor<QType> bL0 = l0B_[abBufId];
                            uint64_t kL1Off = (uint64_t)kTiles * kvRowsAlign * BASE_K + (uint64_t)C0 * nL0Start;
                            LoadNzL1ToZnL0BWithTranspose(bL0, kvL1[kL1Off], kAligned, nRowsAlign, kvRowsAlign);

                            SetWaitFlag<HardEvent::MTE1_M>();

                            MmadParams mp;
                            mp.m = mRowsAlign;
                            mp.n = nRowsAlign;
                            mp.k = kAligned;
                            mp.cmatrixInitVal = isFirstK;
                            mp.cmatrixSource = false;
                            if (!isFirstK) { PipeBarrier<PIPE_M>(); }
                            Mmad(cL0, aL0, bL0, mp);
                            SetFlag<HardEvent::M_MTE1>(l0AFreeEvent_[abBufId]);
                            SetFlag<HardEvent::M_MTE1>(l0BFreeEvent_[abBufId]);
                            l0ABufId_ = 1 - abBufId;
                            isFirstK = false;
                        }

                        SetWaitFlag<HardEvent::M_FIX>();

                        FixpipeParamsV220 fixParams;
                        fixParams.mSize = mRows;
                        fixParams.nSize = nRowsAlign;
                        fixParams.srcStride = mRowsAlign;
                        fixParams.dstStride = outerCols;
                        fixParams.ndNum = 1;
                        fixParams.srcNdStride = 0;
                        fixParams.dstNdStride = 0;
                        uint64_t dstOffset = ((uint64_t)mL1Start + mL0Start) * outerCols + nL1Start + nL0Start;
                        Fixpipe(sSlot[dstOffset], cL0, fixParams);
                        SetFlag<HardEvent::FIX_M>(l0CFreeEvent_[cBufId]);
                        l0CBufId_ = 1 - cBufId;
                    }
                }
            }
            kpQueL1_.FreeTensor(kvL1);
        }
        sQueue_.ProducerReleaseFix();
    }

    __aicore__ inline void ComputeMM2(const TaskInfo &t)
    {
        int bz = t.bz;
        int kvHead = t.kvHead;
        uint32_t s2Start = t.s2Start;
        uint32_t s2Rows = t.s2Rows;
        uint32_t qRows = t.qRows;
        uint32_t dim = tiling_.dim;
        uint32_t kvSeqLen = tiling_.kvSeqLen;
        uint32_t outerCols = tiling_.s2BaseSize;

        auto pSlot = pQueue_.ConsumerAcquire();
        auto oSlot = oQueue_.ProducerAcquire();

        constexpr uint32_t MM2_K_L1_SPLIT = 256;
        constexpr uint32_t V_L1_CACHE_COUNT = 2;
        uint32_t kL1Count = CeilDiv(s2Rows, MM2_K_L1_SPLIT);

        // Fixpipe params are identical for every M sub-block; build once outside the loops.
        FixpipeParamsV220 fixParams;
        fixParams.nSize = BASE_K;
        fixParams.dstStride = dimAlign_;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;

        if (kL1Count > V_L1_CACHE_COUNT) {
            // Fallback path (s2Rows > 512): V queue (depth=2) cannot hold all slices at once,
            // so each M sub-block reloads V from GM — preserves the original per-M-block load/free logic.
            for (uint32_t mL0Start = 0; mL0Start < qRows; mL0Start += BLOCK_M) {
                uint32_t mRows = MinU32(BLOCK_M, qRows - mL0Start);
                uint32_t mRowsAlign = AlignUp(mRows, C0);
                uint32_t cBufId = l0CBufId_;
                WaitFlag<HardEvent::FIX_M>(l0CFreeEvent_[cBufId]);
                LocalTensor<float> cL0 = l0C_[cBufId];
                bool isFirstK = true;
                for (uint32_t kL1Start = 0; kL1Start < s2Rows; kL1Start += MM2_K_L1_SPLIT) {
                    uint32_t kL1Size = MinU32(MM2_K_L1_SPLIT, s2Rows - kL1Start);
                    uint32_t kL1SizeAlign = AlignUp(kL1Size, C0);
                    auto pL1 = kpQueL1_.AllocTensor<QType>();
                    LoadNdGmToNzL1(pL1, pSlot[(uint64_t)mL0Start * outerCols + kL1Start],
                                   mRows, kL1Size, outerCols, mRowsAlign);
                    kpQueL1_.EnQue(pL1);
                    pL1 = kpQueL1_.DeQue<QType>();

                    auto vL1 = vQueL1_.AllocTensor<QType>();
                    uint64_t vOffset = ((uint64_t)bz * tiling_.nkv * kvSeqLen +
                                        (uint64_t)kvHead * kvSeqLen +
                                        (uint64_t)(s2Start + kL1Start)) * dim;
                    LoadNdGmToNzL1(vL1, vGm_[vOffset], kL1Size, dim, dim, kL1SizeAlign);
                    vQueL1_.EnQue(vL1);
                    vL1 = vQueL1_.DeQue<QType>();

                    for (uint32_t kL0Start = 0; kL0Start < kL1Size; kL0Start += BLOCK_N) {
                        uint32_t kL0Size = MinU32(BLOCK_N, kL1Size - kL0Start);
                        uint32_t kL0SizeAlign = AlignUp(kL0Size, C0);

                        uint32_t abBufId = l0ABufId_;
                        WaitFlag<HardEvent::M_MTE1>(l0AFreeEvent_[abBufId]);
                        WaitFlag<HardEvent::M_MTE1>(l0BFreeEvent_[abBufId]);
                        LocalTensor<QType> aL0 = l0A_[abBufId];
                        LoadNzL1ToZzL0A(aL0, pL1[(uint64_t)kL0Start * mRowsAlign],
                                        mRowsAlign, kL0SizeAlign, mRowsAlign);

                        LocalTensor<QType> bL0 = l0B_[abBufId];
                        LoadNzL1ToZnL0B(bL0, vL1[(uint64_t)kL0Start * C0],
                                        kL0SizeAlign, BASE_K, kL1SizeAlign);

                        SetWaitFlag<HardEvent::MTE1_M>();

                        MmadParams mp;
                        mp.m = mRowsAlign;
                        mp.n = BASE_K;
                        mp.k = kL0Size;
                        mp.cmatrixInitVal = isFirstK;
                        mp.cmatrixSource = false;
                        if (!isFirstK) { PipeBarrier<PIPE_M>(); }
                        Mmad(cL0, aL0, bL0, cL0, mp);
                        SetFlag<HardEvent::M_MTE1>(l0AFreeEvent_[abBufId]);
                        SetFlag<HardEvent::M_MTE1>(l0BFreeEvent_[abBufId]);
                        l0ABufId_ = 1 - abBufId;
                        isFirstK = false;
                    }
                    kpQueL1_.FreeTensor(pL1);
                    vQueL1_.FreeTensor(vL1);
                }

                SetWaitFlag<HardEvent::M_FIX>();
                fixParams.mSize = mRows;
                fixParams.srcStride = mRowsAlign;
                Fixpipe(oSlot[(uint64_t)mL0Start * dimAlign_], cL0, fixParams);
                SetFlag<HardEvent::FIX_M>(l0CFreeEvent_[cBufId]);
                l0CBufId_ = 1 - cBufId;
            }
            oQueue_.ProducerReleaseFix();
            return;
        }

        // Optimized path (s2Rows <= 512, kL1Count <= 2): preload V slices once into the
        // depth-2 V queue and reuse them across all M sub-blocks, halving V GM->L1 traffic.

        // 1. Preload V cache (up to 2 slices, held across all M sub-blocks).
        LocalTensor<QType> vCache[V_L1_CACHE_COUNT];
        uint32_t kSizesAlign[V_L1_CACHE_COUNT];
        for (uint32_t kid = 0; kid < kL1Count; ++kid) {
            uint32_t kStart = kid * MM2_K_L1_SPLIT;
            uint32_t kSize = MinU32(MM2_K_L1_SPLIT, s2Rows - kStart);
            uint32_t kAlign = AlignUp(kSize, C0);
            kSizesAlign[kid] = kAlign;
            vCache[kid] = vQueL1_.AllocTensor<QType>();
            uint64_t vOffset = ((uint64_t)bz * tiling_.nkv * kvSeqLen +
                                (uint64_t)kvHead * kvSeqLen +
                                (uint64_t)(s2Start + kStart)) * dim;
            LoadNdGmToNzL1(vCache[kid], vGm_[vOffset], kSize, dim, dim, kAlign);
            vQueL1_.EnQue(vCache[kid]);
            vCache[kid] = vQueL1_.DeQue<QType>();
        }

        // 2. M sub-blocks share the cached V.
        for (uint32_t mL0Start = 0; mL0Start < qRows; mL0Start += BLOCK_M) {
            uint32_t mRows = MinU32(BLOCK_M, qRows - mL0Start);
            uint32_t mRowsAlign = AlignUp(mRows, C0);
            uint32_t cBufId = l0CBufId_;
            WaitFlag<HardEvent::FIX_M>(l0CFreeEvent_[cBufId]);
            LocalTensor<float> cL0 = l0C_[cBufId];
            bool isFirstK = true;
            for (uint32_t kid = 0; kid < kL1Count; ++kid) {
                uint32_t kStart = kid * MM2_K_L1_SPLIT;
                uint32_t kSize = MinU32(MM2_K_L1_SPLIT, s2Rows - kStart);
                uint32_t kL1SizeAlign = kSizesAlign[kid];
                auto pL1 = kpQueL1_.AllocTensor<QType>();
                LoadNdGmToNzL1(pL1, pSlot[(uint64_t)mL0Start * outerCols + kStart],
                               mRows, kSize, outerCols, mRowsAlign);
                kpQueL1_.EnQue(pL1);
                pL1 = kpQueL1_.DeQue<QType>();

                for (uint32_t kL0Start = 0; kL0Start < kSize; kL0Start += BLOCK_N) {
                    uint32_t kL0Size = MinU32(BLOCK_N, kSize - kL0Start);
                    uint32_t kL0SizeAlign = AlignUp(kL0Size, C0);

                    uint32_t abBufId = l0ABufId_;
                    WaitFlag<HardEvent::M_MTE1>(l0AFreeEvent_[abBufId]);
                    WaitFlag<HardEvent::M_MTE1>(l0BFreeEvent_[abBufId]);
                    LocalTensor<QType> aL0 = l0A_[abBufId];
                    LoadNzL1ToZzL0A(aL0, pL1[(uint64_t)kL0Start * mRowsAlign],
                                    mRowsAlign, kL0SizeAlign, mRowsAlign);

                    LocalTensor<QType> bL0 = l0B_[abBufId];
                    LoadNzL1ToZnL0B(bL0, vCache[kid][(uint64_t)kL0Start * C0],
                                    kL0SizeAlign, BASE_K, kL1SizeAlign);

                    SetWaitFlag<HardEvent::MTE1_M>();

                    MmadParams mp;
                    mp.m = mRowsAlign;
                    mp.n = BASE_K;
                    mp.k = kL0Size;
                    mp.cmatrixInitVal = isFirstK;
                    mp.cmatrixSource = false;
                    if (!isFirstK) { PipeBarrier<PIPE_M>(); }
                    Mmad(cL0, aL0, bL0, cL0, mp);
                    SetFlag<HardEvent::M_MTE1>(l0AFreeEvent_[abBufId]);
                    SetFlag<HardEvent::M_MTE1>(l0BFreeEvent_[abBufId]);
                    l0ABufId_ = 1 - abBufId;
                    isFirstK = false;
                }
                kpQueL1_.FreeTensor(pL1);
            }

            SetWaitFlag<HardEvent::M_FIX>();
            fixParams.mSize = mRows;
            fixParams.srcStride = mRowsAlign;
            Fixpipe(oSlot[(uint64_t)mL0Start * dimAlign_], cL0, fixParams);
            SetFlag<HardEvent::FIX_M>(l0CFreeEvent_[cBufId]);
            l0CBufId_ = 1 - cBufId;
        }

        // 3. Release cached V after all M sub-blocks are done.
        for (uint32_t kid = 0; kid < kL1Count; ++kid) {
            vQueL1_.FreeTensor(vCache[kid]);
        }
        oQueue_.ProducerReleaseFix();
    }

private:
    FlashAttentionTiling tiling_;
    uint32_t dimAlign_;
    uint32_t qL1BlockM_;

    GlobalTensor<QType> qGm_, kGm_, vGm_;
    GlobalTensor<float> wsSGm_;
    GlobalTensor<QType> wsPGm_;
    GlobalTensor<float> wsOGm_;
    WorkspaceQueue<float, RING_SLOTS> sQueue_;
    WorkspaceQueue<QType, RING_SLOTS> pQueue_;
    WorkspaceQueue<float, RING_SLOTS> oQueue_;

    TBuf<TPosition::A1> qBufL1_;
    TQue<TPosition::A1, 2> vQueL1_;
    TQue<TPosition::A1, 3> kpQueL1_;

    TBuf<TPosition::A2> l0ABuf_;
    TBuf<TPosition::B2> l0BBuf_;
    TBuf<TPosition::CO1> l0CBuf_;
    LocalTensor<QType> l0A_[L0_AB_BUF_COUNT];
    LocalTensor<QType> l0B_[L0_AB_BUF_COUNT];
    LocalTensor<float> l0C_[L0_C_BUF_COUNT];
    TEventID l0AFreeEvent_[L0_AB_BUF_COUNT];
    TEventID l0BFreeEvent_[L0_AB_BUF_COUNT];
    TEventID l0CFreeEvent_[L0_C_BUF_COUNT];
    uint32_t l0ABufId_ = 0;
    uint32_t l0CBufId_ = 0;
};

#endif // FLASH_ATTENTION_CUBE_H
