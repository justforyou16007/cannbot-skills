/* grouped_matmul 算子注册 */

#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("grouped_matmul(Tensor x, Tensor weight, Tensor? bias, Tensor group_list, int split_item, bool transpose_weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("grouped_matmul", TORCH_FN(ascend_kernel::grouped_matmul));
}

} // namespace
