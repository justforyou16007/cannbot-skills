/*
 * flash_attention_gen op_host — tiling 计算 + workspace 分配 + kernel 启动
 * 设备实现：op_kernel/fa_gen_kernel.h（泛化 kernel，运行期 shape + 运行期 buffer）
 */

#include <cstdint>

#include "op_kernel/flash_attention_launch.h"
#include <cmath>

#include "tiling/platform/platform_ascendc.h"
#include <torch/extension.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"


namespace ascend_kernel {

// 与 op_kernel/fa_gen_kernel.h 中的 FlashAttentionGenTiling 保持一致
struct FlashAttentionGenTilingHost {
    int64_t batch;
    int64_t heads;
    int64_t qSeqLen;
    int64_t kvSeqLen;
    int64_t dim;
    float smScale;
    int64_t causal;
};

constexpr int64_t BLOCK_N = 128;
constexpr int64_t RING_SLOTS = 3;
constexpr int64_t GRID = 20;  // kernel 内 task_idx < block_num 守卫，grid 固定
constexpr int64_t MAX_DIM = 512;

// block_m 按 dim_align 分档（与 kernel fa_gen_kernel.h 一致）
static int64_t DeriveBlockM(int64_t dimAlign, int64_t elemSize) {
    if (elemSize == 4) {  // fp32：block_m 减半避免 L0A 别名
        return (dimAlign <= 128) ? 64 : 32;
    }
    if (dimAlign <= 128) return 128;
    return 64;
}

static void PrepareGen(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                       bool isCausal, at::Tensor &output, at::Tensor &tilingNpu,
                       int64_t &batch, int64_t &heads, int64_t &qSeqLen,
                       int64_t &kvSeqLen, int64_t &dim)
{
    TORCH_CHECK(q.dim() == 4, "q must be 4D [batch, heads, qSeqLen, dim]");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k/v must be 4D");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "q/k/v dtypes must match");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");

    batch    = q.sizes()[0];
    heads    = q.sizes()[1];
    qSeqLen  = q.sizes()[2];
    dim      = q.sizes()[3];
    kvSeqLen = k.sizes()[2];

    TORCH_CHECK(dim <= MAX_DIM, "head dim > 512: not supported");
    TORCH_CHECK(!(isCausal && qSeqLen > kvSeqLen), "causal requires qSeqLen <= kvSeqLen");

    output = at::zeros({batch, heads, qSeqLen, dim}, q.options());

    FlashAttentionGenTilingHost t;
    t.batch = batch;
    t.heads = heads;
    t.qSeqLen = qSeqLen;
    t.kvSeqLen = kvSeqLen;
    t.dim = dim;
    t.smScale = 1.0f / sqrtf(static_cast<float>(dim));
    t.causal = isCausal ? 1 : 0;

    at::Tensor tilingCpu = at::from_blob(&t, {(int64_t)sizeof(t)},
                                         at::device(at::kCPU).dtype(at::kByte)).clone();
    tilingNpu = tilingCpu.to(at::kPrivateUse1);
}

// workspace 尺寸与 kernel 内偏移公式一致（元素 → 字节）：
//   ws_s: block_num * 3 * 128 * 128 * 4B (fp32)
//   ws_p: block_num * 3 * 128 * 128 * sizeof(T)
//   ws_o: block_num * 3 * 128 * dim_align * 4B (fp32)
static void AllocateGenWorkspaces(const at::Tensor &q, int64_t batch, int64_t heads,
                                  int64_t qSeqLen, int64_t dim,
                                  at::Tensor &wsS, at::Tensor &wsP, at::Tensor &wsO)
{
    const int64_t elemSize = q.element_size();
    const int64_t c0 = 32 / elemSize;
    int64_t dimAlign = (dim + c0 - 1) / c0 * c0;
    if (dimAlign > 128) {
        dimAlign = (dim + 127) / 128 * 128;  // 128 整 chunk 取齐
    }
    const int64_t blockM = DeriveBlockM(dimAlign, elemSize);
    const int64_t blockNum = ((qSeqLen + blockM - 1) / blockM) * heads * batch;

    auto optsF32 = at::device(at::kPrivateUse1).dtype(at::kFloat);
    auto optsT = q.options();
    wsS = at::zeros({blockNum * RING_SLOTS * blockM * BLOCK_N}, optsF32);
    wsP = at::zeros({blockNum * RING_SLOTS * blockM * BLOCK_N}, optsT);
    wsO = at::zeros({blockNum * RING_SLOTS * blockM * dimAlign}, optsF32);
}

#define DEFINE_GEN_OP(OP_NAME, LAUNCH_FN)                                              \
    at::Tensor OP_NAME(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,  \
                       bool isCausal)                                                  \
    {                                                                                  \
        at::Tensor output, tilingNpu, wsS, wsP, wsO;                           \
        int64_t batch, heads, qSeqLen, kvSeqLen, dim;                                  \
        PrepareGen(q, k, v, isCausal, output, tilingNpu, batch, heads, qSeqLen,        \
                   kvSeqLen, dim);                                                     \
        AllocateGenWorkspaces(q, batch, heads, qSeqLen, dim, wsS, wsP, wsO);   \
        auto aclStream = c10_npu::getCurrentNPUStream().stream(false);                 \
        LAUNCH_FN(static_cast<uint32_t>(GRID), aclStream,                              \
                  const_cast<void *>(q.storage().data()),                              \
                  const_cast<void *>(k.storage().data()),                              \
                  const_cast<void *>(v.storage().data()),                              \
                  const_cast<void *>(output.storage().data()),                         \
                  const_cast<void *>(wsS.storage().data()),                            \
                  const_cast<void *>(wsP.storage().data()),                            \
                  const_cast<void *>(wsO.storage().data()),                            \
                  const_cast<void *>(tilingNpu.storage().data()));                     \
        return output;                                                                 \
    }

DEFINE_GEN_OP(flash_attention_gen_fp16, launch_flash_attention_gen_fp16)
DEFINE_GEN_OP(flash_attention_gen_bf16, launch_flash_attention_gen_bf16)
DEFINE_GEN_OP(flash_attention_gen_fp32, launch_flash_attention_gen_fp32)

} // namespace ascend_kernel
