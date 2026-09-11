/* gqa_kernels — 全部 __global__ 入口 + launch_* 启动包装（单文件，direct-launch 构建）
 * 设备实现见 gqa_kernel.h（与 ascendc/gqa.cpp 同源的泛化 kernel，全量 20/20 验证）。
 */

#include "gqa_kernel.h"

extern "C" __global__ __aicore__ void gqa_gen_fp16(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output, GM_ADDR workspace_s,
    GM_ADDR workspace_p, GM_ADDR workspace_o, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  gqa_fwd_run<half>(q, k, v, output, workspace_s, workspace_p, workspace_o,
                    tiling);
}

extern "C" __global__ __aicore__ void gqa_gen_bf16(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output, GM_ADDR workspace_s,
    GM_ADDR workspace_p, GM_ADDR workspace_o, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  gqa_fwd_run<bfloat16_t>(q, k, v, output, workspace_s, workspace_p,
                          workspace_o, tiling);
}

/* host 启动包装（<<<>>>，经 gqa_launch.h 声明供 gcc 侧调用） */
extern "C" {

void launch_gqa_gen_fp16(uint32_t numBlocks, void *stream, GM_ADDR q,
                         GM_ADDR k, GM_ADDR v, GM_ADDR output,
                         GM_ADDR workspace_s, GM_ADDR workspace_p,
                         GM_ADDR workspace_o, GM_ADDR tiling) {
  gqa_gen_fp16<<<numBlocks, nullptr, stream>>>(q, k, v, output, workspace_s,
                                               workspace_p, workspace_o,
                                               tiling);
}

void launch_gqa_gen_bf16(uint32_t numBlocks, void *stream, GM_ADDR q,
                         GM_ADDR k, GM_ADDR v, GM_ADDR output,
                         GM_ADDR workspace_s, GM_ADDR workspace_p,
                         GM_ADDR workspace_o, GM_ADDR tiling) {
  gqa_gen_bf16<<<numBlocks, nullptr, stream>>>(q, k, v, output, workspace_s,
                                               workspace_p, workspace_o,
                                               tiling);
}

}
