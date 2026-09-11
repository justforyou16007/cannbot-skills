/* grouped_matmul 算子声明 */
#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor grouped_matmul(const at::Tensor &x, const at::Tensor &weight,
                          const c10::optional<at::Tensor> &bias_opt,
                          const at::Tensor &group_list,
                          int64_t split_item, bool transpose_weight);

} // namespace ascend_kernel

#endif // OPS_H
