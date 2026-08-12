#include "acl/acl_base.h"
#include "acl/acl_base_rt.h"
#include <torch/extension.h>
#include <torch/library.h>
#include "aclnn_common.h"
#include "aclnnop/aclnn_sparse_flash_mla.h"
#include "aclnnop/aclnn_sparse_flash_mla_metadata.h"
#include "ops.h"

namespace op_api {

constexpr int64_t SMLA_METADATA_SIZE = 1024;

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
    const c10::optional<bool> &hasCmpKv)
{
    int64_t batchSizeV = batchSize.value_or(0);
    int64_t maxSeqlenQV = maxSeqlenQ.value_or(0);
    int64_t maxSeqlenOriKvV = maxSeqlenOriKv.value_or(0);
    int64_t maxSeqlenCmpKvV = maxSeqlenCmpKv.value_or(0);
    int64_t oriTopkV = oriTopk.value_or(0);
    int64_t cmpTopkV = cmpTopk.value_or(0);
    int64_t cmpRatioV = cmpRatio.value_or(1);
    int64_t oriMaskModeV = oriMaskMode.value_or(4);
    int64_t cmpMaskModeV = cmpMaskMode.value_or(3);
    int64_t oriWinLeftV = oriWinLeft.value_or(127);
    int64_t oriWinRightV = oriWinRight.value_or(0);
    std::string layoutQV = layoutQ.value_or("BSND");
    std::string layoutKvV = layoutKv.value_or("BSND");
    bool hasOriKvV = hasOriKv.value_or(true);
    bool hasCmpKvV = hasCmpKv.value_or(false);
    at::Device outputDevice = at::Device(std::string("npu"));
    at::Tensor output = torch::empty({SMLA_METADATA_SIZE}, torch::dtype(torch::kInt32).device(outputDevice));
    auto cuSeqlensQVal = get_valid_tensor(cuSeqlensQ, outputDevice);
    auto cuSeqlensOriKvVal = get_valid_tensor(cuSeqlensOriKv, outputDevice);
    auto cuSeqlensCmpKvVal = get_valid_tensor(cuSeqlensCmpKv, outputDevice);
    auto sequsedQVal = get_valid_tensor(sequsedQ, outputDevice);
    auto sequsedOriKvVal = get_valid_tensor(sequsedOriKv, outputDevice);
    auto sequsedCmpKvVal = get_valid_tensor(sequsedCmpKv, outputDevice);
    auto cmpResidualKvVal = get_valid_tensor(cmpResidualKv, outputDevice);
    auto oriTopkLengthVal = get_valid_tensor(oriTopkLength, outputDevice);
    auto cmpTopkLengthVal = get_valid_tensor(cmpTopkLength, outputDevice);
    std::string layoutQStr = std::string(layoutQV);
    std::string layoutKvStr = std::string(layoutKvV);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());
    ACLNN_CMD(aclnnSparseFlashMlaMetadata, cuSeqlensQVal, cuSeqlensOriKvVal, cuSeqlensCmpKvVal, sequsedQVal,
              sequsedOriKvVal, sequsedCmpKvVal, cmpResidualKvVal, oriTopkLengthVal, cmpTopkLengthVal, numHeadsQ,
              numHeadsKv, headDim, batchSizeV, maxSeqlenQV, maxSeqlenOriKvV, maxSeqlenCmpKvV, oriTopkV, cmpTopkV,
              cmpRatioV, oriMaskModeV, cmpMaskModeV, oriWinLeftV, oriWinRightV, layoutQPtr, layoutKvPtr,
              hasOriKvV, hasCmpKvV, output);
    return output;
}

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
    c10::string_view layoutKv, int64_t topkValueMode, bool returnSoftmaxLse)
{
    at::Tensor attenOut = at::empty_like(q);
    at::Tensor softmaxLse = at::empty({}, q.options().dtype(torch::kFloat32));
    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKvStr = std::string(layoutKv);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());
    ACLNN_CMD(aclnnSparseFlashMla, q, oriKv, cmpKv, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
              cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, sequsedOriKv, sequsedCmpKv, cmpResidualKv,
              oriTopkLength, cmpTopkLength, sinks, metadata, softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode,
              oriWinLeft, oriWinRight, layoutQPtr, layoutKvPtr, topkValueMode, returnSoftmaxLse, attenOut, softmaxLse);
    return std::tuple<at::Tensor, at::Tensor>(attenOut, softmaxLse);
}

} // namespace op_api
