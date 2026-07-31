#ifndef FLASH_ATTENTION_KERNEL_H
#define FLASH_ATTENTION_KERNEL_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "flash_attention_tiling.h"
#include "kernel_common.h"
#include "flash_attention_cube.h"
#include "flash_attention_vec.h"

using namespace AscendC;

template <typename QType>
class FlashAttentionKernel {
    static constexpr uint32_t C0 = 32 / sizeof(QType);  // 16 for both half and bfloat16_t

public:
    __aicore__ inline FlashAttentionKernel() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                GM_ADDR output, GM_ADDR workspace,
                                GM_ADDR tilingGM, TPipe *pipe)
    {
        pipe_ = pipe;
        CopyTiling(&tiling_, tilingGM);

        // Use usedCoreNum from tiling so AIC and AIV use identical scheduling.
        int coreIdx;
        if ASCEND_IS_AIC {
            coreIdx = GetBlockIdx();
        }
        if ASCEND_IS_AIV {
            coreIdx = GetBlockIdx() / GetSubBlockNum();
        }
        coreIdx_ = coreIdx;

        // Set up GM tensors using original (unpadded) seq lens
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QType *>(q), tiling_.qTotalElements);
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QType *>(k), tiling_.kvTotalElements);
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QType *>(v), tiling_.kvTotalElements);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QType *>(output), tiling_.qTotalElements);

        GM_ADDR wsBase = workspace + coreIdx * tiling_.perCoreBytes;

        GM_ADDR wsSPtr = wsBase;
        GM_ADDR wsPPtr = wsSPtr + tiling_.wsSSize;
        GM_ADDR wsOPtr = wsPPtr + tiling_.wsPSize;
        GM_ADDR wsMetaPtr = wsOPtr + tiling_.wsOSize;
        GM_ADDR wsAccOPtr = wsMetaPtr + tiling_.wsMetaSize;

        wsSGm_.SetGlobalBuffer((__gm__ float *)(wsSPtr), tiling_.wsSSize / sizeof(float));
        wsPGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QType *>(wsPPtr), tiling_.wsPSize / sizeof(QType));
        wsOGm_.SetGlobalBuffer((__gm__ float *)(wsOPtr), tiling_.wsOSize / sizeof(float));
        wsMetaGm_.SetGlobalBuffer((__gm__ float *)(wsMetaPtr), tiling_.wsMetaSize / sizeof(float));
        wsAccOGm_.SetGlobalBuffer((__gm__ float *)(wsAccOPtr), tiling_.wsAccOSize / sizeof(float));

        if ASCEND_IS_AIC {
            cubeKernel_.Init(tiling_, qGm_, kGm_, vGm_, wsSGm_, wsPGm_, wsOGm_);
            cubeKernel_.InitBuffers(*pipe_);
        }

        if ASCEND_IS_AIV {
            vecKernel_.Init(tiling_, wsSGm_, wsPGm_, wsOGm_, wsMetaGm_, wsAccOGm_, outGm_);
            vecKernel_.InitBuffers(pipe_);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t totalGSq = tiling_.groupNum * tiling_.qOuterBlocks;

        uint32_t startBNkv = 0;
        uint32_t startGSq = 0;
        if (coreIdx_ > 0) {
            startBNkv = tiling_.bNkvEnd[coreIdx_ - 1];
            startGSq = tiling_.gSqEnd[coreIdx_ - 1];
        }
        uint32_t endBNkv = tiling_.bNkvEnd[coreIdx_];
        uint32_t endGSq = tiling_.gSqEnd[coreIdx_];

        uint32_t startLinear = startBNkv * totalGSq + startGSq;
        uint32_t endLinear = endBNkv * totalGSq + endGSq;
        uint32_t qBlockCount = endLinear - startLinear;
        uint32_t totalTasks = qBlockCount * tiling_.s2OuterBlocks;
        if ASCEND_IS_AIC {
            cubeKernel_.BeginL0BufferEvents();
        }
        if (totalTasks == 0) {
            if ASCEND_IS_AIC {
                cubeKernel_.EndL0BufferEvents();
            }
            return;
        }

        if (tiling_.s2OuterBlocks == 1) {
            uint32_t legacyBNkv = startBNkv;
            uint32_t legacyGSq = startGSq;
            uint32_t legacySeq = 0;
            while (legacyBNkv < endBNkv || (legacyBNkv == endBNkv && legacyGSq < endGSq)) {
                const uint32_t bz = legacyBNkv / tiling_.nkv;
                const uint32_t kvHead = legacyBNkv - bz * tiling_.nkv;
                const uint32_t groupId = legacyGSq / tiling_.qOuterBlocks;
                const uint32_t qOuterIdx = legacyGSq - groupId * tiling_.qOuterBlocks;
                const uint32_t qHead = kvHead * tiling_.groupNum + groupId;

                TaskInfo task;
                task.valid = true;
                task.seq = legacySeq;
                task.workspaceSlot = legacySeq % RING_SLOTS;
                task.bz = bz;
                task.kvHead = kvHead;
                task.qHead = qHead;
                task.qRowStart = qOuterIdx * tiling_.mBaseSize;
                task.qRows = MinU32(tiling_.mBaseSize, tiling_.qSeqLen - task.qRowStart);
                task.s2OuterIdx = 0;
                task.s2Start = GetTaskS2Start(tiling_, 0);
                task.s2Rows = GetTaskS2Rows(tiling_, 0);
                task.isFirstS2 = true;
                task.isLastS2 = true;

                if ASCEND_IS_AIC {
                    cubeKernel_.PrepareQ(task);
                }

                for (uint32_t t = 0; t < tiling_.s2OuterBlocks + PRELAUNCH; ++t) {
                    if ASCEND_IS_AIC {
                        if (t >= PRELAUNCH) {
                            cubeKernel_.ComputeMM2(task);
                        }
                        if (t < tiling_.s2OuterBlocks) {
                            cubeKernel_.ComputeMM1(task);
                        }
                    }

                    if ASCEND_IS_AIV {
                        vecKernel_.SetTaskContext(task);
                        if (t >= PRELAUNCH) {
                            vecKernel_.ComputeVec2(task, task.workspaceSlot, true, true);
                        }
                        if (t < tiling_.s2OuterBlocks) {
                            vecKernel_.ComputeVec1(task, task.workspaceSlot, true);
                        }
                    }
                }

                if ASCEND_IS_AIC {
                    cubeKernel_.ReleaseQ();
                }

                legacySeq++;
                legacyGSq++;
                if (legacyGSq == totalGSq) {
                    legacyGSq = 0;
                    legacyBNkv++;
                }
            }
            if ASCEND_IS_AIC {
                cubeKernel_.EndL0BufferEvents();
            }
            return;
        }

        TaskInfo taskRing[RING_SLOTS];
        uint32_t curBNkv = startBNkv;
        uint32_t curGSq = startGSq;
        uint32_t curS2Outer = 0;

        for (uint32_t cycle = 0; cycle < totalTasks + PRELAUNCH; ++cycle) {
            if (cycle < totalTasks) {
                const uint32_t bz = curBNkv / tiling_.nkv;
                const uint32_t kvHead = curBNkv - bz * tiling_.nkv;
                const uint32_t groupId = curGSq / tiling_.qOuterBlocks;
                const uint32_t qOuterIdx = curGSq - groupId * tiling_.qOuterBlocks;
                const uint32_t qHead = kvHead * tiling_.groupNum + groupId;

                uint32_t qRowStart = qOuterIdx * tiling_.mBaseSize;
                uint32_t qRows = MinU32(tiling_.mBaseSize, tiling_.qSeqLen - qRowStart);

                TaskInfo curTask;
                curTask.valid = true;
                curTask.seq = cycle;
                curTask.workspaceSlot = cycle % RING_SLOTS;
                curTask.bz = bz;
                curTask.kvHead = kvHead;
                curTask.qHead = qHead;
                curTask.qRowStart = qRowStart;
                curTask.qRows = qRows;
                curTask.s2OuterIdx = curS2Outer;
                curTask.s2Start = GetTaskS2Start(tiling_, curS2Outer);
                curTask.s2Rows = GetTaskS2Rows(tiling_, curS2Outer);
                curTask.isFirstS2 = (curS2Outer == 0);
                curTask.isLastS2 = (curS2Outer + 1 == tiling_.s2OuterBlocks);
                taskRing[curTask.workspaceSlot] = curTask;

                if ASCEND_IS_AIC {
                    if (curTask.isFirstS2) {
                        cubeKernel_.PrepareQ(curTask);
                    }
                    cubeKernel_.ComputeMM1(curTask);
                    if (curTask.isLastS2) {
                        cubeKernel_.ReleaseQ();
                    }
                }

                curS2Outer++;
                if (curS2Outer == tiling_.s2OuterBlocks) {
                    curS2Outer = 0;
                    curGSq++;
                    if (curGSq == totalGSq) {
                        curGSq = 0;
                        curBNkv++;
                    }
                }
            }

            if ASCEND_IS_AIC {
                if (cycle >= 1) {
                    uint32_t mm2Seq = cycle - 1;
                    if (mm2Seq < totalTasks) {
                        TaskInfo mm2Task = taskRing[mm2Seq % RING_SLOTS];
                        cubeKernel_.ComputeMM2(mm2Task);
                    }
                }
            }

            if ASCEND_IS_AIV {
                if (cycle >= 1) {
                    uint32_t vec1Seq = cycle - 1;
                    if (vec1Seq < totalTasks) {
                        TaskInfo vec1Task = taskRing[vec1Seq % RING_SLOTS];
                        vecKernel_.SetTaskContext(vec1Task);
                        vecKernel_.ComputeVec1(vec1Task, vec1Task.workspaceSlot, vec1Task.isFirstS2);
                    }
                }
                if (cycle >= PRELAUNCH) {
                    uint32_t vec2Seq = cycle - PRELAUNCH;
                    if (vec2Seq < totalTasks) {
                        TaskInfo vec2Task = taskRing[vec2Seq % RING_SLOTS];
                        vecKernel_.SetTaskContext(vec2Task);
                        vecKernel_.ComputeVec2(vec2Task, vec2Task.workspaceSlot, vec2Task.isFirstS2, vec2Task.isLastS2);
                    }
                }
            }
        }
        if ASCEND_IS_AIC {
            cubeKernel_.EndL0BufferEvents();
        }
    }

    TPipe *pipe_;
    FlashAttentionTiling tiling_;
    int coreIdx_;

    GlobalTensor<QType> qGm_, kGm_, vGm_, outGm_;
    GlobalTensor<float> wsSGm_, wsMetaGm_, wsAccOGm_, wsOGm_;
    GlobalTensor<QType> wsPGm_;

    FlashAttentionCube<QType> cubeKernel_;
    FlashAttentionVec<QType> vecKernel_;
};

#endif // FLASH_ATTENTION_KERNEL_H
