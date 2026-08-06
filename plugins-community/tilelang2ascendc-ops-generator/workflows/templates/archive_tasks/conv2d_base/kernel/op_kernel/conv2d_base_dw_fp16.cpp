/* BSD 3-Clause License - Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "conv2d_base_dw.h"

extern "C" __global__ __aicore__ void conv2d_base_dw_fp16(
    GM_ADDR x, GM_ADDR xp, GM_ADDR w, GM_ADDR bias, GM_ADDR y,
    int batch, int ci, int co, int coG, int H, int W, int wpad,
    int kH, int kW, int sH, int pH, int pW, int dH, int dW,
    int Ho, int Wo, int woPad, int rowsSlot, int bandOh, int haloL,
    int hasBias, int totalTasks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC { return; }
    AscendC::TPipe pipe;
    Conv2dDwKernel<half> kernel;
    kernel.Init(x, xp, w, bias, y, batch, ci, co, coG, H, W, wpad,
                kH, kW, sH, pH, pW, dH, dW, Ho, Wo, woPad, rowsSlot,
                bandOh, haloL, hasBias, totalTasks, &pipe);
    kernel.Process();
}
