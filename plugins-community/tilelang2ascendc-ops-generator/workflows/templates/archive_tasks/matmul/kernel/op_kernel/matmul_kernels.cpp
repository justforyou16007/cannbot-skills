/* matmul_kernels — 全部 __global__ 入口 + launch_* 启动包装（单文件） */

/* ---- matmul_gen_bf16 ---- */
/*
 * matmul_gen_bf16 — 标准算子工程入口（deliverer Phase 5）
 * 设备实现见 mm_gen_kernel.h（泛化 kernel，运行期 M/N/K）
 */

#include "mm_gen_kernel.h"

extern "C" __global__ __aicore__ void matmul_gen_bf16(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    mm_gen::mm_fwd_run<bfloat16_t>(a, b, c, tiling);
}

/* ---- matmul_gen_fp16 ---- */
/*
 * matmul_gen_fp16 — 标准算子工程入口（deliverer Phase 5）
 * 设备实现见 mm_gen_kernel.h（泛化 kernel，运行期 M/N/K）
 */

#include "mm_gen_kernel.h"

extern "C" __global__ __aicore__ void matmul_gen_fp16(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    mm_gen::mm_fwd_run<half>(a, b, c, tiling);
}

/* ---- matmul_gen_fp32 ---- */
/*
 * matmul_gen_fp32 — 标准算子工程入口（deliverer Phase 5）
 * 设备实现见 mm_gen_kernel.h（泛化 kernel，运行期 M/N/K）
 */

#include "mm_gen_kernel.h"

extern "C" __global__ __aicore__ void matmul_gen_fp32(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    mm_gen::mm_fwd_run<float>(a, b, c, tiling);
}

/* host 启动包装（<<<>>>，经 matmul_launch.h 声明供 gcc 侧调用） */
extern "C" {

void launch_matmul_gen_bf16(
    uint32_t numBlocks, void *stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
{
    matmul_gen_bf16<<<numBlocks, nullptr, stream>>>(a, b, c, tiling);
}

void launch_matmul_gen_fp16(
    uint32_t numBlocks, void *stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
{
    matmul_gen_fp16<<<numBlocks, nullptr, stream>>>(a, b, c, tiling);
}

void launch_matmul_gen_fp32(
    uint32_t numBlocks, void *stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
{
    matmul_gen_fp32<<<numBlocks, nullptr, stream>>>(a, b, c, tiling);
}

}
