/* BSD 3-Clause License - Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "conv2d_base_epilogue.h"

extern "C" __global__ __aicore__ void conv2d_base_epilogue_fp16(
    GM_ADDR acc, GM_ADDR bias, GM_ADDR y,
    int totalRows, int coPad, int co, int mImg, int rowsPerTile,
    int hasBias, int isNCHW, int directMode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC { return; }
    AscendC::TPipe pipe;
    Conv2dEpilogueKernel<half> kernel;
    kernel.Init(acc, bias, y, totalRows, coPad, co, mImg, rowsPerTile,
                hasBias, isNCHW, directMode, &pipe);
    kernel.Process();
}
