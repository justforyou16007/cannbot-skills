/* BSD 3-Clause License - Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "conv2d_base_wrepack.h"

extern "C" __global__ __aicore__ void conv2d_base_wrepack_bf16(
    GM_ADDR w, GM_ADDR wOut, int Co, int Ci, int kHkW, int CiPad, int CoPad,
    int fused)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC { return; }
    AscendC::TPipe pipe;
    Conv2dWeightRepackKernel<bfloat16_t> kernel;
    kernel.Init(w, wOut, Co, Ci, kHkW, CiPad, CoPad, fused, &pipe);
    kernel.Process();
}
