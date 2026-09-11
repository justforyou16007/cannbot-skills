/*
 * matmul_gen 算子声明（deliverer Phase 5 工程化）
 */
#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor matmul_gen_fp16(const at::Tensor &a, const at::Tensor &b);
at::Tensor matmul_gen_bf16(const at::Tensor &a, const at::Tensor &b);
at::Tensor matmul_gen_fp32(const at::Tensor &a, const at::Tensor &b);

} // namespace ascend_kernel

#endif // OPS_H
