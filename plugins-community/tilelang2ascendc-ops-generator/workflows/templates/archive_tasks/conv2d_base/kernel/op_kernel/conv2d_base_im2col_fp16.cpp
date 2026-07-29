/* BSD 3-Clause License - Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "conv2d_base_im2col.h"

extern "C" __global__ __aicore__ void conv2d_base_im2col_fp16(
    GM_ADDR x, GM_ADDR im2col,
    int H, int W, int Ci, int CiPad, int kH, int kW, int Ho, int Wo,
    int strideH, int strideW, int dilH, int dilW, int padH, int padW,
    int K, int mImg, int woBlocks, int totalTiles, int isNCHW)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC { return; }
    AscendC::TPipe pipe;
    Conv2dIm2colKernel<half> kernel;
    kernel.Init(x, im2col, H, W, Ci, CiPad, kH, kW, Ho, Wo,
                strideH, strideW, dilH, dilW, padH, padW,
                K, mImg, woBlocks, totalTiles, isNCHW, &pipe);
    if (isNCHW >= 2) { kernel.dbgNoStore_ = (isNCHW - 1) & 15; kernel.disableMerged_ = (isNCHW - 1) >> 4; }
    kernel.Process();
}
