/* gqa_launch — launch_* 启动包装声明（gcc 侧包含） */
#ifndef GQA_LAUNCH_H
#define GQA_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// 兼容 utils/torch_kernel_helper.h 的 EXEC_KERNEL_CMD
#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_name) launch_##kernel_name
#endif

extern "C" {

void launch_gqa_gen_fp16(uint32_t numBlocks, void *stream, GM_ADDR q,
                         GM_ADDR k, GM_ADDR v, GM_ADDR output,
                         GM_ADDR workspace_s, GM_ADDR workspace_p,
                         GM_ADDR workspace_o, GM_ADDR tiling);

void launch_gqa_gen_bf16(uint32_t numBlocks, void *stream, GM_ADDR q,
                         GM_ADDR k, GM_ADDR v, GM_ADDR output,
                         GM_ADDR workspace_s, GM_ADDR workspace_p,
                         GM_ADDR workspace_o, GM_ADDR tiling);

}

#endif  // GQA_LAUNCH_H
