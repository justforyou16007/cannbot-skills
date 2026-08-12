#ifndef SMLA_OPS_H
#define SMLA_OPS_H
#include <torch/extension.h>
namespace op_api {
at::Tensor SparseFlashMlaMetadata(
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
    const c10::optional<at::Tensor> &cuSeqlensQ,
    const c10::optional<at::Tensor> &cuSeqlensOriKv,
    const c10::optional<at::Tensor> &cuSeqlensCmpKv,
    const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedOriKv,
    const c10::optional<at::Tensor> &sequsedCmpKv,
    const c10::optional<at::Tensor> &cmpResidualKv,
    const c10::optional<at::Tensor> &oriTopkLength,
    const c10::optional<at::Tensor> &cmpTopkLength,
    const c10::optional<int64_t> &batchSize, const c10::optional<int64_t> &maxSeqlenQ,
    const c10::optional<int64_t> &maxSeqlenOriKv, const c10::optional<int64_t> &maxSeqlenCmpKv,
    const c10::optional<int64_t> &oriTopk, const c10::optional<int64_t> &cmpTopk,
    const c10::optional<int64_t> &cmpRatio, const c10::optional<int64_t> &oriMaskMode,
    const c10::optional<int64_t> &cmpMaskMode, const c10::optional<int64_t> &oriWinLeft,
    const c10::optional<int64_t> &oriWinRight, const c10::optional<std::string> &layoutQ,
    const c10::optional<std::string> &layoutKv, const c10::optional<bool> &hasOriKv,
    const c10::optional<bool> &hasCmpKv);
std::tuple<at::Tensor, at::Tensor> SparseFlashMla(
    const at::Tensor &q, const c10::optional<at::Tensor> &oriKv, const c10::optional<at::Tensor> &cmpKv,
    const c10::optional<at::Tensor> &oriSparseIndices, const c10::optional<at::Tensor> &cmpSparseIndices,
    const c10::optional<at::Tensor> &oriBlockTable, const c10::optional<at::Tensor> &cmpBlockTable,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensOriKv,
    const c10::optional<at::Tensor> &cuSeqlensCmpKv, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedOriKv, const c10::optional<at::Tensor> &sequsedCmpKv,
    const c10::optional<at::Tensor> &cmpResidualKv, const c10::optional<at::Tensor> &oriTopkLength,
    const c10::optional<at::Tensor> &cmpTopkLength, const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata, double softmaxScale, int64_t cmpRatio, int64_t oriMaskMode,
    int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, c10::string_view layoutQ,
    c10::string_view layoutKv, int64_t topkValueMode, bool returnSoftmaxLse);
} // namespace op_api
#endif
