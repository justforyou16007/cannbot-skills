/*
 * flash_attention_gen 算子声明
 */
#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor flash_attention_gen_fp16(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, bool isCausal);
at::Tensor flash_attention_gen_bf16(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, bool isCausal);
at::Tensor flash_attention_gen_fp32(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, bool isCausal);

} // namespace ascend_kernel

#endif // OPS_H
