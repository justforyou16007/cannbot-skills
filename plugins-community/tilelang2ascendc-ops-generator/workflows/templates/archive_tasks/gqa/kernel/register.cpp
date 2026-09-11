/*
 * gqa_gen 算子注册（Phase 5 工程化）
 */

#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("gqa_gen_fp16(Tensor q, Tensor k, Tensor v, float scaleValue, bool isCausal) -> Tensor");
    m.def("gqa_gen_bf16(Tensor q, Tensor k, Tensor v, float scaleValue, bool isCausal) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("gqa_gen_fp16", TORCH_FN(ascend_kernel::gqa_gen_fp16));
    m.impl("gqa_gen_bf16", TORCH_FN(ascend_kernel::gqa_gen_bf16));
}

} // namespace
