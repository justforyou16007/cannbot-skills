/*
 * matmul_gen 算子注册（deliverer Phase 5 工程化）
 */

#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("matmul_gen_fp16(Tensor a, Tensor b) -> Tensor");
    m.def("matmul_gen_bf16(Tensor a, Tensor b) -> Tensor");
    m.def("matmul_gen_fp32(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("matmul_gen_fp16", TORCH_FN(ascend_kernel::matmul_gen_fp16));
    m.impl("matmul_gen_bf16", TORCH_FN(ascend_kernel::matmul_gen_bf16));
    m.impl("matmul_gen_fp32", TORCH_FN(ascend_kernel::matmul_gen_fp32));
}

} // namespace
