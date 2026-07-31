#ifndef FLASH_ATTENTION_TILING_H
#define FLASH_ATTENTION_TILING_H

#include <cstdint>

constexpr uint32_t BLOCK_M = 128;
constexpr uint32_t BLOCK_N = 128;
constexpr uint32_t BASE_K = 128;
constexpr uint32_t PRELAUNCH = 2;
constexpr uint32_t RING_SLOTS = PRELAUNCH + 1;  // 3
constexpr uint32_t MAX_CORES = 20;

// AIC->AIV workspace queue sync signals (no free signal needed: RING_SLOTS > PRELAUNCH).
constexpr uint32_t SIG_S_READY = 0;  // AIC MM1 done -> AIV Vec1
constexpr uint32_t SIG_P_READY = 1;  // AIV Vec1 done -> AIC MM2
constexpr uint32_t SIG_O_READY = 2;  // AIC MM2 done -> AIV Vec2

#pragma pack(push, 8)
struct FlashAttentionTiling {
    uint32_t batch;
    uint32_t heads;        // equals nq for the current MHA implementation
    uint32_t nq;
    uint32_t nkv;
    uint32_t groupNum;     // nq / nkv, 1 for the current MHA implementation
    uint32_t qSeqLen;      // original Q sequence length
    uint32_t kvSeqLen;     // original KV sequence length
    uint32_t dim;
    uint32_t blockM;       // 128
    uint32_t blockN;       // 128
    float   smScale;       // 1/sqrt(dim)
    uint32_t usedCoreNum;
    uint32_t tailValid;    // kvSeqLen % BLOCK_N, valid columns in last KV tile
    uint32_t qSeqLenAlign; // aligned Q seq len (to BLOCK_M)
    uint32_t kvSeqLenAlign;// aligned KV seq len (to BLOCK_N)
    uint32_t dimAlign;
    uint32_t seqBlocks;
    uint32_t kvLoops;
    uint32_t totalBlocks;
    uint32_t mBaseSize;
    uint32_t s2BaseSize;
    uint32_t qOuterBlocks;
    uint32_t s2OuterBlocks;
    uint64_t qTotalElements;
    uint64_t kvTotalElements;
    uint64_t wsSSize;
    uint64_t wsPSize;
    uint64_t wsOSize;
    uint64_t wsMetaSize;
    uint64_t wsAccOSize;
    uint64_t perCoreBytes;
    uint64_t totalWsBytes;
    uint32_t bNkvEnd[MAX_CORES];
    uint32_t gSqEnd[MAX_CORES];
};
#pragma pack(pop)

uint64_t doOpTiling(uint32_t batch,
                    uint32_t nq,
                    uint32_t nkv,
                    uint32_t qSeqLen,
                    uint32_t kvSeqLen,
                    uint32_t dim,
                    uint32_t qElementSize,
                    FlashAttentionTiling *tiling);

#endif // FLASH_ATTENTION_TILING_H
