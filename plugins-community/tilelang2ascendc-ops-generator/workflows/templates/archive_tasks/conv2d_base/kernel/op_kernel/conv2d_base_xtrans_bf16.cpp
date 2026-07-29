/* BSD 3-Clause License - Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "conv2d_base_xtrans.h"

extern "C" __global__ __aicore__ void conv2d_base_xtrans_bf16(
    GM_ADDR x, GM_ADDR xT, int batch, int Ci, int HW)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC { return; }
    AscendC::TPipe pipe;
    Conv2dXtransKernel<bfloat16_t> kernel;
    kernel.Init(x, xT, batch, Ci, HW, &pipe);
    kernel.Process();
}
