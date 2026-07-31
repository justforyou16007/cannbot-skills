#ifndef FLASH_ATTENTION_WORKSPACE_QUEUE_H
#define FLASH_ATTENTION_WORKSPACE_QUEUE_H

#include "kernel_operator.h"

template <typename T, uint32_t DEPTH>
class WorkspaceQueue {
public:
    __aicore__ inline WorkspaceQueue() {}

    __aicore__ inline void Init(const AscendC::GlobalTensor<T> &workspace, uint32_t slotSize,
                                uint16_t notifyId)
    {
        workspace_ = workspace;
        slotSize_ = slotSize;
        head_ = 0;
        tail_ = 0;
        notifyId_ = notifyId;
    }

    // Producer: get next slot to write (never blocks — RING_SLOTS > PRELAUNCH guarantees safety)
    __aicore__ inline AscendC::GlobalTensor<T> ProducerAcquire()
    {
        return workspace_[head_ % DEPTH * slotSize_];
    }

    __aicore__ inline void ProducerReleaseFix()
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(notifyId_);
        head_++;
    }

    __aicore__ inline void ProducerReleaseMte3()
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(notifyId_);
        head_++;
    }

    __aicore__ inline AscendC::GlobalTensor<T> ConsumerAcquire()
    {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(notifyId_);
        return workspace_[tail_++ % DEPTH * slotSize_];
    }

private:
    AscendC::GlobalTensor<T> workspace_;
    uint32_t slotSize_;
    uint32_t head_;
    uint32_t tail_;
    uint16_t notifyId_;
};

#endif // FLASH_ATTENTION_WORKSPACE_QUEUE_H
