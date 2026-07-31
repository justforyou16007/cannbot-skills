#ifndef KERNEL_COMMON_H
#define KERNEL_COMMON_H

#include <type_traits>

constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = 64;
constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = 8;
constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

// Unified task context passed through the AIC/AIV pipeline.
// Pure data carrier — packing only, no scheduling logic.
struct TaskInfo {
    bool valid = false;
    uint32_t seq = 0;
    uint32_t workspaceSlot = 0;
    uint32_t bz = 0;
    uint32_t kvHead = 0;
    uint32_t qHead = 0;
    uint32_t qRowStart = 0;
    uint32_t qRows = 0;
    uint32_t s2OuterIdx = 0;
    uint32_t s2Start = 0;
    uint32_t s2Rows = 0;
    bool isFirstS2 = false;   // s2OuterIdx == 0
    bool isLastS2 = false;    // s2OuterIdx + 1 == s2OuterBlocks
};

// Integer ceil division for tiling and block scheduling
__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

template <typename T>
__aicore__ inline uint32_t GetTaskS2Start(const T &tiling, uint32_t s2OuterIdx)
{
    return s2OuterIdx * tiling.s2BaseSize;
}

template <typename T>
__aicore__ inline uint32_t GetTaskS2Rows(const T &tiling, uint32_t s2OuterIdx)
{
    uint32_t s2Start = GetTaskS2Start(tiling, s2OuterIdx);
    if (s2Start >= tiling.kvSeqLen) {
        return 0;
    }
    return MinU32(tiling.s2BaseSize, tiling.kvSeqLen - s2Start);
}

template <typename T>
__aicore__ inline uint32_t GetTileKvRows(const T &tiling, uint32_t s2Start, uint32_t localTileIdx)
{
    uint32_t rowStart = s2Start + localTileIdx * BLOCK_N;
    if (rowStart >= tiling.kvSeqLen) {
        return 0;
    }
    return MinU32(BLOCK_N, tiling.kvSeqLen - rowStart);
}

// Issue a same-core hard-event SetFlag/WaitFlag pair
template <AscendC::HardEvent evt>
__aicore__ inline void SetWaitFlag() {
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    AscendC::SetFlag<evt>(eventId);
    AscendC::WaitFlag<evt>(eventId);
}

// Copy packed tiling data from GM into a local tiling struct
template<typename T>
__aicore__ inline void CopyTiling(T *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);
    for (size_t i = 0; i < sizeof(T) / sizeof(uint32_t); ++i, ++ptr) {
        *ptr = *(tiling32 + i);
    }
}

// Row-wise divide.
// src0/dst are laid out as [dealRowCount, columnCount], and src1 is a Brcb-expanded
// broadcast buffer laid out as [dealRowCount, FP32_BLOCK_ELEMENT_NUM].
// Each row in src0 is divided by the corresponding per-row scalar block in src1,
__aicore__ inline void RowDivs(AscendC::LocalTensor<float> dstUb,
                               AscendC::LocalTensor<float> src0Ub,
                               AscendC::LocalTensor<float> src1Ub,
                               uint32_t dealRowCount,
                               uint32_t columnCount,
                               uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;
    AscendC::BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;

    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            AscendC::Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        AscendC::BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8;
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            AscendC::Div(dstUb[offset],
                src0Ub[offset],
                src1Ub[i * FP32_BLOCK_ELEMENT_NUM],
                dtypeMask,
                columnRepeatCount,
                columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        AscendC::Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

// Row-wise multiply.
// src0/dst are laid out as [dealRowCount, columnCount], and src1 is a Brcb-expanded
// broadcast buffer laid out as [dealRowCount, blockElementNum].
// Each row in src0 is multiplied by the corresponding per-row scalar block in src1,
template <typename T>
__aicore__ inline void RowMuls(AscendC::LocalTensor<T> dstUb,
                               AscendC::LocalTensor<T> src0Ub,
                               AscendC::LocalTensor<T> src1Ub,
                               uint32_t dealRowCount,
                               uint32_t columnCount,
                               uint32_t actualColumnCount)
{
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2;
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
    }

    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        if (dLoop <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                AscendC::Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            AscendC::BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8;
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8;
            for (uint32_t i = 0; i < dealRowCount; i++) {
                AscendC::Mul(dstUb[i * columnCount],
                    src0Ub[i * columnCount],
                    src1Ub[i * blockElementNum],
                    repeatElementNum,
                    dLoop,
                    columnRepeatParams);
            }
        }

        if (dRemain > 0) {
            AscendC::Mul(dstUb[dLoop * repeatElementNum],
                src0Ub[dLoop * repeatElementNum],
                src1Ub,
                dRemain,
                dealRowCount,
                repeatParams);
        }
    } else {
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            AscendC::Mul(dstUb[i * columnCount],
                src0Ub[i * columnCount],
                src1Ub[i * blockElementNum],
                repeatElementNum,
                dLoop,
                repeatParams);
            if (dRemain > 0) {
                AscendC::Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum],
                    src1Ub[i * blockElementNum],
                    dRemain,
                    1,
                    repeatParams);
            }
        }
    }
}

#endif // KERNEL_COMMON_H
