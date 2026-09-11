/*
 * mha_gen op_host — tiling 计算 + workspace 分配 + kernel 启动（Phase 5）
 * 设备实现：op_kernel/mha_kernel.h（泛化 kernel，运行期 shape + 运行期 buffer）
 */

#include <cstdint>
#include <cmath>

#include "op_kernel/mha_launch.h"

#include "tiling/platform/platform_ascendc.h"
#include <torch/extension.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"


namespace ascend_kernel {

// 与 op_kernel/kernel_common.h 中的 MhaGenTiling 保持一致
struct MhaGenTilingHost {
    int64_t batch;
    int64_t heads_q;
    int64_t heads_kv;
    int64_t q_seq_len;
    int64_t kv_seq_len;
    int64_t dim;
    int64_t causal;
    float sm_scale;
    int64_t pad;
};

constexpr int64_t RING_SLOTS = 3;

// Runtime AI core count (never hardcoded): query the current device's AI core
// number so the grid matches the chip (910B A2=20 / 910C A3=24 / 950). Cached
// after the first query. MIX_AIC_1_2: grid = cube core count. The kernel reads
// the same count via GetBlockNum(), so grid and task split never diverge.
static uint32_t GetGridSize()
{
    static uint32_t grid = 0;
    if (grid == 0) {
        auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        uint32_t aic = 20;  // fallback
        if (platform != nullptr) {
            aic = platform->GetCoreNumAic();
        }
        if (aic == 0) {
            aic = 20;
        }
        grid = aic;
    }
    return grid;
}

// 派生 tiling 参数（公式与 kernel 内逐字一致，见 design/tile_level/mha.py）
struct MhaDerived {
    int64_t group, dim_align, cube_k, d_chunks, bs, bg, block_m, block_n;
    int64_t s_blks, g_blks, block_num;
};

static MhaDerived Derive(int64_t batch, int64_t heads_q, int64_t heads_kv,
                         int64_t q_seq_len, int64_t kv_seq_len, int64_t dim,
                         int64_t elemSize)
{
    MhaDerived d;
    const int64_t c0 = 32 / elemSize;
    d.group = heads_q / heads_kv;
    d.dim_align = (dim + c0 - 1) / c0 * c0;
    d.cube_k = d.dim_align < 128 ? d.dim_align : 128;
    d.d_chunks = (d.dim_align + 127) / 128;
    // HS21: per-tier row budget (must match kernel op_kernel/mha_kernel.h).
    const int64_t m_budget =
        d.dim_align <= 128 ? 128 : (d.dim_align <= 256 ? 64 : 32);
    d.bs = m_budget < q_seq_len ? m_budget : q_seq_len;
    const int64_t bg_cap = (m_budget / d.bs) > 1 ? (m_budget / d.bs) : 1;
    d.bg = d.group < bg_cap ? d.group : bg_cap;
    d.block_m = ((d.bs * d.bg + 15) / 16) * 16;
    // nRatio S2 amplification (must match kernel op_kernel/mha_kernel.h):
    // widen the KV tile 128 -> 256 for small-block_m decode/MTP shapes.
    d.block_n = (d.dim_align <= 128 && d.block_m <= 64 && kv_seq_len >= 1024)
                    ? 256 : 128;
    d.s_blks = (q_seq_len + d.bs - 1) / d.bs;
    d.g_blks = (d.group + d.bg - 1) / d.bg;
    d.block_num = batch * heads_kv * d.s_blks * d.g_blks;
    return d;
}

static void PrepareGen(const at::Tensor &q, const at::Tensor &k,
                       const at::Tensor &v, double scaleValue, bool isCausal,
                       at::Tensor &output, at::Tensor &tilingNpu,
                       MhaDerived &d)
{
    TORCH_CHECK(q.dim() == 4, "q must be 4D [batch, qSeqLen, heads_q, dim] (BSND)");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k/v must be 4D (BSND)");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "q/k/v dtypes must match");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");

    const int64_t batch    = q.sizes()[0];
    const int64_t qSeqLen  = q.sizes()[1];
    const int64_t headsQ   = q.sizes()[2];
    const int64_t dim      = q.sizes()[3];
    const int64_t kvSeqLen = k.sizes()[1];
    const int64_t headsKv  = k.sizes()[2];

    TORCH_CHECK(headsQ % headsKv == 0, "heads_q %% heads_kv != 0");
    TORCH_CHECK(!(isCausal && qSeqLen > kvSeqLen), "is_causal requires S <= S_kv");
    // HS21: dim support extended to 512. dim_align<=512 with per-tier m_budget
    // (128/64/32) and single-slot K/V at dim_align>=512 keep UB/L0A/L1 within
    // capacity (see kernel op_kernel/mha_kernel.h and ascendc/mha.cpp).
    TORCH_CHECK(dim <= 512,
                "head dim > 512: d_chunks / L0/L1 capacity only covers dim <= 512");

    // at::empty (NOT at::zeros): the output is fully overwritten by every task's
    // epilogue, and the ring cross-core flags guarantee write-before-read on all
    // workspaces (C1 writes ws_s -> SetFlag -> V1 WaitFlag reads; C2 writes ws_o
    // likewise; ws_p written by V1 read by C2). No uninitialized-read path.
    output = at::empty({batch, qSeqLen, headsQ, dim}, q.options());

    d = Derive(batch, headsQ, headsKv, qSeqLen, kvSeqLen, dim, q.element_size());

    MhaGenTilingHost t;
    t.batch = batch;
    t.heads_q = headsQ;
    t.heads_kv = headsKv;
    t.q_seq_len = qSeqLen;
    t.kv_seq_len = kvSeqLen;
    t.dim = dim;
    t.causal = isCausal ? 1 : 0;
    t.sm_scale = scaleValue > 0 ? static_cast<float>(scaleValue)
                                : 1.0f / sqrtf(static_cast<float>(dim));
    t.pad = 0;

    at::Tensor tilingCpu = at::from_blob(&t, {(int64_t)sizeof(t)},
                                         at::device(at::kCPU).dtype(at::kByte)).clone();
    tilingNpu = tilingCpu.to(at::kPrivateUse1);
}

// workspace 尺寸与 kernel 内偏移公式一致（元素）。kernel 内 workspace 按全局
// task_idx 索引（ws_*_base = task_idx * ws_*_task），故按 block_num 分配：
//   ws_s: block_num * RING * block_m * block_n  (fp32)  -- P3: no d_chunks (L0C K-accum)
//   ws_p: block_num * RING * block_m * block_n             (T)
//   ws_o: block_num * RING * block_m * dim_align           (fp32)
static void AllocateGenWorkspaces(const at::Tensor &q, const MhaDerived &d,
                                  at::Tensor &wsS, at::Tensor &wsP,
                                  at::Tensor &wsO)
{
    auto optsF32 = at::device(at::kPrivateUse1).dtype(at::kFloat);
    auto optsT = q.options();
    // at::empty (NOT at::zeros): see PrepareGen note — cross-core flags order all
    // workspace writes before reads, so no zero-init is needed.
    // P3: ws_s holds ONE accumulated slab per ring slot (C1 accumulates d_chunks
    // in L0C), so no d_chunks factor (footprint /2). ws_o keeps dim_align (C2 N-col).
    wsS = at::empty({d.block_num * RING_SLOTS * d.block_m * d.block_n}, optsF32);
    wsP = at::empty({d.block_num * RING_SLOTS * d.block_m * d.block_n}, optsT);
    wsO = at::empty({d.block_num * RING_SLOTS * d.block_m * d.dim_align}, optsF32);
}

#define DEFINE_GEN_OP(OP_NAME, LAUNCH_FN)                                              \
    at::Tensor OP_NAME(const at::Tensor &q, const at::Tensor &k,                       \
                       const at::Tensor &v, double scaleValue, bool isCausal)          \
    {                                                                                  \
        at::Tensor output, tilingNpu, wsS, wsP, wsO;                                   \
        MhaDerived d;                                                                  \
        PrepareGen(q, k, v, scaleValue, isCausal, output, tilingNpu, d);               \
        AllocateGenWorkspaces(q, d, wsS, wsP, wsO);                                    \
        auto aclStream = c10_npu::getCurrentNPUStream().stream(false);                 \
        LAUNCH_FN(GetGridSize(), aclStream,                                              \
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

DEFINE_GEN_OP(mha_gen_fp16, launch_mha_gen_fp16)
DEFINE_GEN_OP(mha_gen_bf16, launch_mha_gen_bf16)

} // namespace ascend_kernel
