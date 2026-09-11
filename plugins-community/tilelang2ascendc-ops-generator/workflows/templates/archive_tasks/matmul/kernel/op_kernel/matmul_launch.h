/* matmul_launch — launch_* 启动包装声明（gcc 侧包含） */
#ifndef MATMUL_LAUNCH_H
#define MATMUL_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif


// 兼容 utils/torch_kernel_helper.h 的 EXEC_KERNEL_CMD
#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_name) launch_##kernel_name
#endif

extern "C" {

void launch_matmul_gen_bf16(
    uint32_t numBlocks, void *stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling);

void launch_matmul_gen_fp16(
    uint32_t numBlocks, void *stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling);

void launch_matmul_gen_fp32(
    uint32_t numBlocks, void *stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling);

}

#endif // MATMUL_LAUNCH_H
