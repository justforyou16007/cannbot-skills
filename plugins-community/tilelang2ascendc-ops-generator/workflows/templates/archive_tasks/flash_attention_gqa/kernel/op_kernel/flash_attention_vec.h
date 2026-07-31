#ifndef FLASH_ATTENTION_VEC_H
#define FLASH_ATTENTION_VEC_H

#include "kernel_operator.h"
#include "flash_attention_tiling.h"
#include "kernel_common.h"
#include "workspace_queue.h"

using namespace AscendC;

constexpr SoftmaxConfig FA_SOFTMAX_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};
constexpr uint32_t SOFTMAX_WORK_BUF_SIZE = 32 * 1024;
constexpr uint32_t SOFTMAX_STATE_BUF_SIZE = 2 * 1024;
constexpr float SOFTMAX_NEG_INF = -1073741824.0f;
constexpr uint32_t VEC_QUEUE_BUF_SIZE = 32 * 1024;
constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = VEC_QUEUE_BUF_SIZE / sizeof(float);

template <typename QType>
class FlashAttentionVec {
    static constexpr uint32_t C0 = 32 / sizeof(QType);
    static constexpr uint16_t BRCB_NUM = 32 / sizeof(float);
    struct MSplitInfo {
        uint32_t nBufferStartM = 0;
        uint32_t nBufferDealM = 0;
        uint32_t vecStartM = 0;
        uint32_t vecDealM = 0;
    };

public:
    __aicore__ inline FlashAttentionVec() {}

    __aicore__ inline void Init(const FlashAttentionTiling &tiling,
                                GlobalTensor<float> &wsSGm, GlobalTensor<QType> &wsPGm,
                                GlobalTensor<float> &wsOGm, GlobalTensor<float> &wsMetaGm,
                                GlobalTensor<float> &wsAccOGm,
                                GlobalTensor<QType> &outGm)
    {
        tiling_ = tiling;
        wsSGm_ = wsSGm;
        wsPGm_ = wsPGm;
        wsOGm_ = wsOGm;
        wsMetaGm_ = wsMetaGm;
        wsAccOGm_ = wsAccOGm;
        outGm_ = outGm;
        dimAlign_ = AlignUp(tiling_.dim, C0);
        sQueue_.Init(wsSGm_, tiling_.mBaseSize * tiling_.s2BaseSize, SIG_S_READY);
        pQueue_.Init(wsPGm_, tiling_.mBaseSize * tiling_.s2BaseSize, SIG_P_READY);
        oQueue_.Init(wsOGm_, tiling_.mBaseSize * dimAlign_, SIG_O_READY);
        subBlockNum_ = GetSubBlockNum();
        subBlockIdx_ = GetSubBlockIdx();
        subBlockRows_ = BLOCK_M / subBlockNum_;
        rowStart_ = subBlockIdx_ * subBlockRows_;
        vec1ChunkRows_ = BASE_BLOCK_MAX_ELEMENT_NUM / tiling_.s2BaseSize;
        if (vec1ChunkRows_ == 0) {
            vec1ChunkRows_ = 1;
        }
        // SoftmaxFlashV2 state (max/sum/exp) 写入需要 32 字节对齐，
        // 即 baseOffset 必须是 8 的倍数，因此 vec1ChunkRows_ 也必须是 8 的倍数。
        // tiling 层保证 s2BaseSize <= 1024，使得 vec1ChunkRows_ >= 8 且 buffer 不超过 32KB。
        constexpr uint32_t SM_STATE_ALIGN = 8;
        vec1ChunkRows_ = ((vec1ChunkRows_ + SM_STATE_ALIGN - 1) / SM_STATE_ALIGN) * SM_STATE_ALIGN;
        if (vec1ChunkRows_ > subBlockRows_) {
            vec1ChunkRows_ = (subBlockRows_ / SM_STATE_ALIGN) * SM_STATE_ALIGN;
        }
        if (vec1ChunkRows_ == 0) {
            vec1ChunkRows_ = SM_STATE_ALIGN;
        }
    }

    __aicore__ inline void InitBuffers(TPipe *pipe)
    {
        pipe->InitBuffer(inputQue1_, 2, VEC_QUEUE_BUF_SIZE);
        pipe->InitBuffer(outputQue1_, 1, VEC_QUEUE_BUF_SIZE);

        pipe->InitBuffer(tmpBuf_, SOFTMAX_WORK_BUF_SIZE);
        pipe->InitBuffer(softmaxMaxDefaultBuf_, SOFTMAX_STATE_BUF_SIZE);
        pipe->InitBuffer(softmaxSumDefaultBuf_, SOFTMAX_STATE_BUF_SIZE);

        // Dedicated TBuf for prev softmax max/sum state — avoids overwriting the
        // default-value buffers (which must stay at -inf/0 for isFirst=true).
        stateStride_ = SOFTMAX_STATE_BUF_SIZE / sizeof(float);
        pipe->InitBuffer(maxCacheBuf_, RING_SLOTS * SOFTMAX_STATE_BUF_SIZE);
        pipe->InitBuffer(sumCacheBuf_, RING_SLOTS * SOFTMAX_STATE_BUF_SIZE);
        pipe->InitBuffer(expCacheBuf_, RING_SLOTS * SOFTMAX_STATE_BUF_SIZE);
        maxCacheUb_ = maxCacheBuf_.Get<float>();
        sumCacheUb_ = sumCacheBuf_.Get<float>();
        expCacheUb_ = expCacheBuf_.Get<float>();

        softmaxMaxDefaultUb_ = softmaxMaxDefaultBuf_.Get<float>();
        softmaxSumDefaultUb_ = softmaxSumDefaultBuf_.Get<float>();

        Duplicate(softmaxMaxDefaultUb_, SOFTMAX_NEG_INF, SOFTMAX_STATE_BUF_SIZE / sizeof(float));
        Duplicate(softmaxSumDefaultUb_, 0.0f, SOFTMAX_STATE_BUF_SIZE / sizeof(float));
    }

    __aicore__ inline void SetTaskContext(const TaskInfo &t)
    {
        curBz_ = t.bz;
        curQHead_ = t.qHead;
        curQRowStart_ = t.qRowStart;
        curTaskRows_ = t.qRows;
        SetMSplitInfo(t.qRows);
    }

    __aicore__ inline void ComputeVec1(const TaskInfo &t, uint32_t slot, bool isFirst)
    {
        uint32_t kvRows = t.s2Rows;

        auto sSlot = sQueue_.ConsumerAcquire();
        auto pSlot = pQueue_.ProducerAcquire();
        if (mSplitInfo_.vecDealM == 0) {
            pQueue_.ProducerReleaseMte3();
            return;
        }
        uint32_t outerCols = tiling_.s2BaseSize;
        for (uint32_t chunkRow = 0; chunkRow < mSplitInfo_.vecDealM; chunkRow += vec1ChunkRows_) {
            uint32_t dealRows = MinU32(vec1ChunkRows_, mSplitInfo_.vecDealM - chunkRow);
            uint32_t tileSize = dealRows * outerCols;

            LocalTensor<float> sUb = inputQue1_.AllocTensor<float>();
            uint64_t sOffset = (uint64_t)(mSplitInfo_.vecStartM + chunkRow) * outerCols;
            DataCopy(sUb, sSlot[sOffset], tileSize);
            inputQue1_.EnQue(sUb);
            sUb = inputQue1_.DeQue<float>();
            Muls(sUb, sUb, tiling_.smScale, tileSize);

            LocalTensor<uint8_t> softmaxTmpUb = tmpBuf_.Get<uint8_t>();
            SoftMaxShapeInfo srcShape;
            srcShape.srcM = dealRows;
            srcShape.srcK = outerCols;
            srcShape.oriSrcM = dealRows;
            srcShape.oriSrcK = kvRows;
            SoftMaxTiling smTiling = SoftMaxFlashV2TilingFunc(
                srcShape, sizeof(float), sizeof(float), softmaxTmpUb.GetSize(), true, false);

            LocalTensor<float> inMaxTensor;
            LocalTensor<float> inSumTensor;
            uint32_t baseOffset = mSplitInfo_.nBufferStartM / 2 + chunkRow;
            if (isFirst) {
                inMaxTensor = softmaxMaxDefaultUb_;
                inSumTensor = softmaxSumDefaultUb_;
            } else {
                uint32_t prevSlot = (slot + RING_SLOTS - 1) % RING_SLOTS;
                uint32_t prevStateBase = prevSlot * stateStride_ + baseOffset;
                inMaxTensor = maxCacheUb_[prevStateBase];
                inSumTensor = sumCacheUb_[prevStateBase];
            }

            uint32_t stateBase = slot * stateStride_ + baseOffset;
            SoftmaxFlashV2<float, true, true, false, false, FA_SOFTMAX_CFG_WITHOUT_BRC>(
                sUb, sumCacheUb_[stateBase], maxCacheUb_[stateBase], sUb,
                expCacheUb_[stateBase], inSumTensor, inMaxTensor, softmaxTmpUb, smTiling, srcShape);

            LocalTensor<QType> pHalf = outputQue1_.AllocTensor<QType>();
            Cast(pHalf, sUb, RoundMode::CAST_ROUND, tileSize);
            outputQue1_.EnQue(pHalf);
            pHalf = outputQue1_.DeQue<QType>();
            uint64_t pOffset = (uint64_t)(mSplitInfo_.vecStartM + chunkRow) * outerCols;
            DataCopy(pSlot[pOffset], pHalf, tileSize);
            outputQue1_.FreeTensor(pHalf);
            inputQue1_.FreeTensor(sUb);
        }

        pQueue_.ProducerReleaseMte3();
    }

    __aicore__ inline void ComputeVec2(const TaskInfo &t, uint32_t slot, bool isFirst, bool isLast)
    {
        auto oSlot = oQueue_.ConsumerAcquire();
        if (mSplitInfo_.vecDealM == 0) {
            return;
        }

        uint32_t dim = dimAlign_;
        uint32_t actualDim = tiling_.dim;
        uint32_t mChunk = BASE_BLOCK_MAX_ELEMENT_NUM / dim;
        if (mChunk > 255) {
            mChunk = 255;
        }
        mChunk = (mChunk / BRCB_NUM) * BRCB_NUM;
        if (mChunk == 0) {
            mChunk = BRCB_NUM;
        }
        if (mChunk > mSplitInfo_.vecDealM) {
            mChunk = mSplitInfo_.vecDealM;
        }
        uint32_t loopCount = (mSplitInfo_.vecDealM + mChunk - 1) / mChunk;
        uint32_t tailChunk = mSplitInfo_.vecDealM - (loopCount - 1) * mChunk;
        for (uint32_t ci = 0, dealRows = mChunk; ci < loopCount; ci++) {
            if (ci == loopCount - 1) {
                dealRows = tailChunk;
            }
            uint32_t startRow = ci * mChunk;
            uint32_t chunkSize = dealRows * dim;
            uint32_t rowOffset = mSplitInfo_.vecStartM + startRow;
            uint32_t stateRowBase = slot * stateStride_ + mSplitInfo_.nBufferStartM / 2 + startRow;
            
            LocalTensor<float> oNewUb = inputQue1_.AllocTensor<float>();
            uint64_t oOffset = (uint64_t)rowOffset * dim;
            DataCopy(oNewUb, oSlot[oOffset], chunkSize);
            inputQue1_.EnQue(oNewUb);
            oNewUb = inputQue1_.DeQue<float>();

            if (!isFirst) {
                uint32_t prevSlot = (slot + RING_SLOTS - 1) % RING_SLOTS;
                uint64_t accOffset = ((uint64_t)prevSlot * tiling_.mBaseSize + rowOffset) * dim;
                LocalTensor<float> oPrevUb = inputQue1_.AllocTensor<float>();
                DataCopy(oPrevUb, wsAccOGm_[accOffset], chunkSize);
                inputQue1_.EnQue(oPrevUb);
                oPrevUb = inputQue1_.DeQue<float>();

                LocalTensor<float> tmpExpBrcbUb = tmpBuf_.Get<float>();
                LocalTensor<float> expStateUb = expCacheUb_[stateRowBase];
                Brcb(tmpExpBrcbUb, expStateUb, (dealRows + BRCB_NUM - 1) / BRCB_NUM, {1, BRCB_NUM});
                RowMuls(oPrevUb, oPrevUb, tmpExpBrcbUb, dealRows, dim, actualDim);
                Add(oNewUb, oNewUb, oPrevUb, chunkSize);
                inputQue1_.FreeTensor(oPrevUb);
            }

            if (isLast) {
                LocalTensor<float> tmpSumBrcbUb = tmpBuf_.Get<float>();
                LocalTensor<float> sumStateUb = sumCacheUb_[stateRowBase];
                Brcb(tmpSumBrcbUb, sumStateUb, (dealRows + BRCB_NUM - 1) / BRCB_NUM, {1, BRCB_NUM});
                RowDivs(oNewUb, oNewUb, tmpSumBrcbUb, dealRows, dim, actualDim);
                FinalizeOutputChunk(oNewUb, rowOffset, dealRows);
            } else {
                uint64_t accOutOffset = ((uint64_t)slot * tiling_.mBaseSize + rowOffset) * dim;
                LocalTensor<float> tmpAccUb = outputQue1_.AllocTensor<float>();
                DataCopy(tmpAccUb, oNewUb, chunkSize);
                outputQue1_.EnQue(tmpAccUb);
                tmpAccUb = outputQue1_.DeQue<float>();
                DataCopy(wsAccOGm_[accOutOffset], tmpAccUb, chunkSize);
                outputQue1_.FreeTensor(tmpAccUb);
            }
            inputQue1_.FreeTensor(oNewUb);
        }
    }

private:
    __aicore__ inline void SetMSplitInfo(uint32_t mDealSize)
    {
        mSplitInfo_.nBufferStartM = 0;
        mSplitInfo_.nBufferDealM = mDealSize;
        if (mSplitInfo_.nBufferDealM <= 16) {
            mSplitInfo_.vecDealM = mSplitInfo_.nBufferDealM;
        } else {
            mSplitInfo_.vecDealM = ((mSplitInfo_.nBufferDealM + 15) / 16 + 1) / 2 * 16;
        }
        mSplitInfo_.vecStartM = 0;
        if ((subBlockIdx_ & 1U) == 1U) {
            mSplitInfo_.vecStartM = mSplitInfo_.vecDealM;
            mSplitInfo_.vecDealM = mSplitInfo_.nBufferDealM - mSplitInfo_.vecDealM;
        }
        if (mSplitInfo_.vecStartM >= curTaskRows_) {
            mSplitInfo_.vecDealM = 0;
            return;
        }
        if (mSplitInfo_.vecStartM + mSplitInfo_.vecDealM > curTaskRows_) {
            mSplitInfo_.vecDealM = curTaskRows_ - mSplitInfo_.vecStartM;
        }
    }

    __aicore__ inline void FinalizeOutputChunk(LocalTensor<float> &oUb, uint32_t startRow, uint32_t dealRows)
    {
        uint32_t actualDim = tiling_.dim;
        uint32_t dim = dimAlign_;
        uint32_t qSeqLen = tiling_.qSeqLen;

        uint32_t globalRowStart = curQRowStart_ + startRow;
        if (globalRowStart >= (uint32_t)qSeqLen) return;
        uint32_t maxValidRows = qSeqLen - globalRowStart;
        if (dealRows > maxValidRows) dealRows = maxValidRows;

        LocalTensor<QType> outHalf = outputQue1_.AllocTensor<QType>();

        if (dim == actualDim) {
            Cast(outHalf, oUb, RoundMode::CAST_ROUND, dealRows * dim);

            uint64_t outBase = ((uint64_t)curBz_ * tiling_.nq * qSeqLen +
                                (uint64_t)curQHead_ * qSeqLen +
                                (uint64_t)curQRowStart_ +
                                (uint64_t)startRow) * actualDim;
            outputQue1_.EnQue(outHalf);
            outHalf = outputQue1_.DeQue<QType>();
            DataCopy(outGm_[outBase], outHalf, dealRows * actualDim);
        } else {
            for (uint32_t i = 0; i < dealRows; i++) {
                Cast(outHalf[i * actualDim], oUb[i * dim], RoundMode::CAST_ROUND, actualDim);
            }

            uint64_t outBase = ((uint64_t)curBz_ * tiling_.nq * qSeqLen +
                                (uint64_t)curQHead_ * qSeqLen +
                                (uint64_t)curQRowStart_ +
                                (uint64_t)startRow) * actualDim;
            outputQue1_.EnQue(outHalf);
            outHalf = outputQue1_.DeQue<QType>();
            for (uint32_t i = 0; i < dealRows; i++) {
                DataCopy(outGm_[outBase + i * actualDim], outHalf[i * actualDim], actualDim);
            }
        }
        outputQue1_.FreeTensor(outHalf);
    }

public:
    int curBz_, curQHead_;

private:
    FlashAttentionTiling tiling_;
    uint32_t dimAlign_;
    uint32_t subBlockNum_;
    uint32_t subBlockIdx_;
    uint32_t subBlockRows_;
    uint32_t rowStart_;
    uint32_t vec1ChunkRows_;
    uint32_t stateStride_;
    uint32_t curQRowStart_;
    uint32_t curTaskRows_;
    uint32_t curSubBlockRows_;
    MSplitInfo mSplitInfo_;

    GlobalTensor<float> wsSGm_;
    GlobalTensor<QType> wsPGm_;
    GlobalTensor<float> wsOGm_;
    GlobalTensor<float> wsMetaGm_;
    GlobalTensor<float> wsAccOGm_;
    GlobalTensor<QType> outGm_;
    WorkspaceQueue<float, RING_SLOTS> sQueue_;
    WorkspaceQueue<QType, RING_SLOTS> pQueue_;
    WorkspaceQueue<float, RING_SLOTS> oQueue_;

    TQue<TPosition::VECIN, 1> inputQue1_;
    TQue<TPosition::VECOUT, 1> outputQue1_;

    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> softmaxMaxDefaultBuf_;
    TBuf<TPosition::VECCALC> softmaxSumDefaultBuf_;
    LocalTensor<float> softmaxMaxDefaultUb_;
    LocalTensor<float> softmaxSumDefaultUb_;

    TBuf<TPosition::VECCALC> maxCacheBuf_;
    TBuf<TPosition::VECCALC> sumCacheBuf_;
    TBuf<TPosition::VECCALC> expCacheBuf_;
    LocalTensor<float> maxCacheUb_;
    LocalTensor<float> sumCacheUb_;
    LocalTensor<float> expCacheUb_;
};

#endif // FLASH_ATTENTION_VEC_H
