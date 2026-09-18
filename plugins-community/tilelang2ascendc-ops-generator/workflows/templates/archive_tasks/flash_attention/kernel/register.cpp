/*
 * flash_attention_gen 算子注册（Phase 5 工程化）
 */

#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("flash_attention_gen_fp16(Tensor q, Tensor k, Tensor v, bool is_causal=False) -> Tensor");
    m.def("flash_attention_gen_bf16(Tensor q, Tensor k, Tensor v, bool is_causal=False) -> Tensor");
    m.def("flash_attention_gen_fp32(Tensor q, Tensor k, Tensor v, bool is_causal=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("flash_attention_gen_fp16", TORCH_FN(ascend_kernel::flash_attention_gen_fp16));
    m.impl("flash_attention_gen_bf16", TORCH_FN(ascend_kernel::flash_attention_gen_bf16));
    m.impl("flash_attention_gen_fp32", TORCH_FN(ascend_kernel::flash_attention_gen_fp32));
}

} // namespace
