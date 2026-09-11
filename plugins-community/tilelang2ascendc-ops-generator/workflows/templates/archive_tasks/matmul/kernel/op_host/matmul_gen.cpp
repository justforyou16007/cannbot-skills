/*
 * matmul_gen op_host — tiling 计算 + kernel 启动（deliverer Phase 5）
 * 设备实现：op_kernel/mm_gen_kernel.h（泛化 kernel，运行期 M/N/K）
 * 无 GM workspace（C_L0 -> C 直写）。
 */

#include <cstdint>

#include "op_kernel/matmul_launch.h"

#include "tiling/platform/platform_ascendc.h"
#include <torch/extension.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"


namespace ascend_kernel {

// 与 op_kernel/mm_common.h 中的 MatmulGenTiling 保持一致
struct MatmulGenTilingHost {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t pad;
};

constexpr int64_t GRID = 20;  // kernel 内 task < m_num*n_num 守卫，grid 固定

#define DEFINE_MM_OP(OP_NAME, LAUNCH_FN)                                               \
    at::Tensor OP_NAME(const at::Tensor &a, const at::Tensor &b)                       \
    {                                                                                  \
        TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a/b must be 2D");                   \
        TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a/b dtypes must match");      \
        TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),                            \
                    "a/b must be contiguous");                                         \
        const int64_t m = a.sizes()[0];                                                \
        const int64_t k = a.sizes()[1];                                                \
        const int64_t n = b.sizes()[1];                                                \
        TORCH_CHECK(b.sizes()[0] == k, "K mismatch");                                  \
                                                                                       \
        at::Tensor output = at::zeros({m, n}, a.options());                            \
                                                                                       \
        MatmulGenTilingHost t;                                                         \
        t.m = m;                                                                       \
        t.n = n;                                                                       \
        t.k = k;                                                                       \
        t.pad = 0;                                                                     \
        at::Tensor tilingCpu = at::from_blob(&t, {(int64_t)sizeof(t)},                 \
                                             at::device(at::kCPU).dtype(at::kByte))    \
                                   .clone();                                           \
        at::Tensor tilingNpu = tilingCpu.to(at::kPrivateUse1);                         \
                                                                                       \
        auto aclStream = c10_npu::getCurrentNPUStream().stream(false);                 \
        LAUNCH_FN(static_cast<uint32_t>(GRID), aclStream,                              \
                  const_cast<void *>(a.storage().data()),                              \
                  const_cast<void *>(b.storage().data()),                              \
                  const_cast<void *>(output.storage().data()),                         \
                  const_cast<void *>(tilingNpu.storage().data()));                     \
        return output;                                                                 \
    }

DEFINE_MM_OP(matmul_gen_fp16, launch_matmul_gen_fp16)
DEFINE_MM_OP(matmul_gen_bf16, launch_matmul_gen_bf16)
DEFINE_MM_OP(matmul_gen_fp32, launch_matmul_gen_fp32)

} // namespace ascend_kernel
