#include <cmath>

#include "flash_attention_tiling.h"

namespace {

inline uint32_t AlignUpHost(uint32_t num, uint32_t align)
{
    return (num + align - 1) / align * align;
}

inline uint32_t CeilDivHost(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

inline uint32_t SelectS2BaseSize(uint32_t qSeqLen, uint32_t groupNum)
{
    // s2BaseSize 上限 1024：保证 vec1ChunkRows_ = VEC_QUEUE_BUF_SIZE / (s2BaseSize * 4) >= 8，
    // 从而 SoftmaxFlashV2 的 state 写入地址始终 32 字节对齐。
    // UB 192KB 下 s2BaseSize > 1024 会导致 input queue buffer 超限。
    if (qSeqLen <= 16) {
        if (groupNum <= 4) {
            return 1024;
        }
        if (groupNum <= 9) {
            return 1024;
        }
        return 1024;
    }
    return 512;
}

inline uint32_t SelectMBaseSize(uint32_t qSeqLen, uint32_t s2BaseSize)
{
    uint32_t mBaseSize = 512;
    if (qSeqLen <= 16) {
        if (s2BaseSize <= 512) {
            mBaseSize = 512;
        } else if (s2BaseSize <= 1024) {
            mBaseSize = 256;
        } else if (s2BaseSize <= 2048) {
            mBaseSize = 128;
        } else if (s2BaseSize <= 4096) {
            mBaseSize = 64;
        } else {
            mBaseSize = 32;
        }
    }
    // Current kernel keeps 128x128 as the smallest executable micro tile.
    return mBaseSize < BLOCK_M ? BLOCK_M : mBaseSize;
}

inline void FillCoreRangeEnds(uint32_t totalBNkv,
                              uint32_t totalGSq,
                              uint32_t usedCoreNum,
                              FlashAttentionTiling *tiling)
{
    const uint64_t totalWork = static_cast<uint64_t>(totalBNkv) * totalGSq;
    for (uint32_t core = 0; core < MAX_CORES; ++core) {
        tiling->bNkvEnd[core] = 0;
        tiling->gSqEnd[core] = 0;
    }

    for (uint32_t core = 0; core < usedCoreNum; ++core) {
        const uint64_t linearEnd = totalWork * (core + 1) / usedCoreNum;
        tiling->bNkvEnd[core] = static_cast<uint32_t>(linearEnd / totalGSq);
        tiling->gSqEnd[core] = static_cast<uint32_t>(linearEnd % totalGSq);
    }
}

} // namespace

uint64_t doOpTiling(uint32_t batch,
                    uint32_t nq,
                    uint32_t nkv,
                    uint32_t qSeqLen,
                    uint32_t kvSeqLen,
                    uint32_t dim,
                    uint32_t qElementSize,
                    FlashAttentionTiling *tiling)
{
    const uint32_t groupNum = nq / nkv;
    const uint32_t qSeqLenAlign = AlignUpHost(qSeqLen, BLOCK_M);
    const uint32_t kvSeqLenAlign = AlignUpHost(kvSeqLen, BLOCK_N);
    const uint32_t dimAlign = AlignUpHost(dim, 16);
    const uint32_t tailValid = kvSeqLen % BLOCK_N;
    const uint32_t seqBlocks = CeilDivHost(qSeqLenAlign, BLOCK_M);
    const uint32_t kvLoops = CeilDivHost(kvSeqLen, BLOCK_N);
    const uint32_t s2BaseSize = SelectS2BaseSize(qSeqLen, groupNum);
    const uint32_t mBaseSize = SelectMBaseSize(qSeqLen, s2BaseSize);
    const uint32_t qOuterBlocks = CeilDivHost(qSeqLenAlign, mBaseSize);
    const uint32_t s2OuterBlocks = CeilDivHost(kvSeqLen, s2BaseSize);
    const uint32_t totalBNkv = batch * nkv;
    const uint32_t totalGSq = groupNum * qOuterBlocks;
    const uint64_t totalWork = static_cast<uint64_t>(totalBNkv) * totalGSq;
    const uint32_t totalBlocks = static_cast<uint32_t>(totalWork);
    const uint32_t usedCoreNum = totalWork < MAX_CORES ? static_cast<uint32_t>(totalWork) : MAX_CORES;
    const uint64_t qTotalElements = static_cast<uint64_t>(batch) * nq * qSeqLen * dim;
    const uint64_t kvTotalElements = static_cast<uint64_t>(batch) * nkv * kvSeqLen * dim;

    tiling->batch = batch;
    tiling->heads = nq;
    tiling->nq = nq;
    tiling->nkv = nkv;
    tiling->groupNum = groupNum;
    tiling->qSeqLen = qSeqLen;
    tiling->kvSeqLen = kvSeqLen;
    tiling->dim = dim;
    tiling->blockM = BLOCK_M;
    tiling->blockN = BLOCK_N;
    tiling->smScale = 1.0f / std::sqrt(static_cast<float>(dim));
    tiling->usedCoreNum = usedCoreNum;
    tiling->tailValid = tailValid;
    tiling->qSeqLenAlign = qSeqLenAlign;
    tiling->kvSeqLenAlign = kvSeqLenAlign;
    tiling->dimAlign = dimAlign;
    tiling->seqBlocks = seqBlocks;
    tiling->kvLoops = kvLoops;
    tiling->totalBlocks = totalBlocks;
    tiling->mBaseSize = mBaseSize;
    tiling->s2BaseSize = s2BaseSize;
    tiling->qOuterBlocks = qOuterBlocks;
    tiling->s2OuterBlocks = s2OuterBlocks;
    tiling->qTotalElements = qTotalElements;
    tiling->kvTotalElements = kvTotalElements;

    const uint64_t wsSSize = static_cast<uint64_t>(RING_SLOTS) * mBaseSize * s2BaseSize * sizeof(float);
    const uint64_t wsPSize = static_cast<uint64_t>(RING_SLOTS) * mBaseSize * s2BaseSize * qElementSize;
    const uint64_t wsOSize = static_cast<uint64_t>(RING_SLOTS) * mBaseSize * dimAlign * sizeof(float);
    const uint64_t wsMetaSize = static_cast<uint64_t>(RING_SLOTS) * mBaseSize * 3 * sizeof(float);
    const uint64_t wsAccOSize = static_cast<uint64_t>(RING_SLOTS) * mBaseSize * dimAlign * sizeof(float);
    const uint64_t perCoreBytes = wsSSize + wsPSize + wsOSize + wsMetaSize + wsAccOSize;
    const uint64_t totalWsBytes = perCoreBytes * usedCoreNum;

    tiling->wsSSize = wsSSize;
    tiling->wsPSize = wsPSize;
    tiling->wsOSize = wsOSize;
    tiling->wsMetaSize = wsMetaSize;
    tiling->wsAccOSize = wsAccOSize;
    tiling->perCoreBytes = perCoreBytes;
    tiling->totalWsBytes = totalWsBytes;
    FillCoreRangeEnds(totalBNkv, totalGSq, usedCoreNum, tiling);

    return totalWsBytes;
}
