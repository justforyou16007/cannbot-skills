/* flash_attention_kernels — 全部 __global__ 入口 + launch_* 启动包装（单文件） */

/* ---- flash_attention_gen_bf16 ---- */
/*
 * flash_attention_gen_bf16 — 标准算子工程入口（Phase 5）
 * 设备实现见 fa_gen_kernel.h（与 ascendc/flash_attention.cpp 同源的泛化 kernel）
 */

#include "fa_gen_kernel.h"

extern "C" __global__ __aicore__ void flash_attention_gen_bf16(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output, GM_ADDR workspace_s,
    GM_ADDR workspace_p, GM_ADDR workspace_o,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    fa_fwd_run<bfloat16_t>(q, k, v, output, workspace_s, workspace_p, workspace_o,
                    tiling);
}

/* ---- flash_attention_gen_fp16 ---- */
/*
 * flash_attention_gen_fp16 — 标准算子工程入口（Phase 5）
 * 设备实现见 fa_gen_kernel.h（与 ascendc/flash_attention.cpp 同源的泛化 kernel）
 */

#include "fa_gen_kernel.h"

extern "C" __global__ __aicore__ void flash_attention_gen_fp16(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output, GM_ADDR workspace_s,
    GM_ADDR workspace_p, GM_ADDR workspace_o,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    fa_fwd_run<half>(q, k, v, output, workspace_s, workspace_p, workspace_o,
                    tiling);
}

/* ---- flash_attention_gen_fp32 ---- */
/*
 * flash_attention_gen_fp32 — 标准算子工程入口（Phase 5）
 * 设备实现见 fa_gen_kernel.h（与 ascendc/flash_attention.cpp 同源的泛化 kernel）
 */

#include "fa_gen_kernel.h"

extern "C" __global__ __aicore__ void flash_attention_gen_fp32(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output, GM_ADDR workspace_s,
    GM_ADDR workspace_p, GM_ADDR workspace_o,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    fa_fwd_run<float>(q, k, v, output, workspace_s, workspace_p, workspace_o,
                    tiling);
}

/* host 启动包装（<<<>>>，经 flash_attention_launch.h 声明供 gcc 侧调用） */
extern "C" {

void launch_flash_attention_gen_bf16(
    uint32_t numBlocks, void *stream, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output, GM_ADDR workspace_s, GM_ADDR workspace_p, GM_ADDR workspace_o, GM_ADDR tiling)
{
    flash_attention_gen_bf16<<<numBlocks, nullptr, stream>>>(q, k, v, output, workspace_s, workspace_p, workspace_o, tiling);
}

void launch_flash_attention_gen_fp16(
    uint32_t numBlocks, void *stream, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output, GM_ADDR workspace_s, GM_ADDR workspace_p, GM_ADDR workspace_o, GM_ADDR tiling)
{
    flash_attention_gen_fp16<<<numBlocks, nullptr, stream>>>(q, k, v, output, workspace_s, workspace_p, workspace_o, tiling);
}

void launch_flash_attention_gen_fp32(
    uint32_t numBlocks, void *stream, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output, GM_ADDR workspace_s, GM_ADDR workspace_p, GM_ADDR workspace_o, GM_ADDR tiling)
{
    flash_attention_gen_fp32<<<numBlocks, nullptr, stream>>>(q, k, v, output, workspace_s, workspace_p, workspace_o, tiling);
}

}
