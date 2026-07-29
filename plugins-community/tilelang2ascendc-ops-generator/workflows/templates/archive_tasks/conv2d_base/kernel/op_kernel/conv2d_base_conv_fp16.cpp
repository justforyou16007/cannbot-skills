/* BSD 3-Clause License - Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "conv2d_base.h"

extern "C" __global__ __aicore__ void conv2d_base_conv_fp16(
    GM_ADDR im2col, GM_ADDR w, GM_ADDR acc,
    int batch, int mImg, int Co, int K, int kAL1,
    int mBlocks, int nBlocks, int totalBlocks, int direct,
    int H, int W, int Ci, int kH, int kW,
    int sH, int sW, int dH, int dW, int pH, int pW)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    if ASCEND_IS_AIV { return; }
    Conv2dBaseKernel<half> kernel;
    kernel.Init(im2col, w, acc, batch, mImg, Co, K, kAL1, mBlocks, nBlocks,
                totalBlocks, direct, H, W, Ci, kH, kW, sH, sW, dH, dW,
                pH, pW, &pipe);
    kernel.Process();
}
