// ----------------------------------------------------------------------------------------------------------
// FlashAttention forward (generalized, HS32 手写同步) - Cube 侧：C0 / C1 / C2
// 同步结构（flag 编号 / 顺序）与 tilelang 手写同步设计（design/tile_level/
// flash_attention_handsync.py）的生成码逐字节一致，禁止改动。
// ----------------------------------------------------------------------------------------------------------
#ifndef FA_GEN_CUBE_H
#define FA_GEN_CUBE_H

#include "kernel_common.h"

namespace fa_gen {

// C0：Q 常驻 L1（尾块越界读由 DataCopyPad 零填充；只取前 dim 列）
template <typename T>
CATLASS_DEVICE void FaGenC0LoadQ(FaGenContext<T> &ctx, int64_t q_base,
                                 int64_t q_tail_m) {
  fa_dyn::copy_gm_to_l1_dyn<T>(ctx.q_l1[0], ctx.q[q_base], ctx.block_m,
                               ctx.dim_align, ctx.dim, q_tail_m, ctx.dim, true);
}

// C1(t)：S = Q @ K^T -> workspace_s[slot]（d_chunks 个 K 段累 L0C）
// HS35：k_l1 双缓冲——QK 用槽 t&1，同时 MTE2 往槽 (t+1)&1 载入下一块 K，
// 载入与 QK 重叠（preload 槽 0 在主循环前）。
template <typename T>
CATLASS_DEVICE void FaGenC1(FaGenContext<T> &ctx, int64_t kv_base,
                            int64_t ws_s_base, int64_t slot, int64_t t) {
  const int64_t cur = t & 1;
  const int64_t nxt = (t + 1) & 1;
  auto k_cur = cur ? ctx.k_l1b : ctx.k_l1;
  auto k_nxt = nxt ? ctx.k_l1b : ctx.k_l1;
  // HS35 预取 K(t+1) 到槽 nxt（QK 前，MTE2 预取与 Cube QK 重叠）
  if (t + 1 < ctx.kv_loops) {
    const int64_t k_tail_next =
        min64(BLOCK_N, ctx.kv_seq_len - (t + 1) * BLOCK_N);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(nxt);
    fa_dyn::copy_gm_to_l1_dyn<T>(
        k_nxt[0], ctx.k[kv_base + (t + 1) * BLOCK_N * ctx.dim], BLOCK_N,
        ctx.dim_align, ctx.dim, k_tail_next, ctx.dim, true);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(nxt);
  }
  // QK 用槽 cur 的 K(t)
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(cur);
  const int64_t q_step = ctx.block_m * ctx.cube_k;
  const int64_t k_step = ctx.cube_k * BLOCK_N;
  int64_t q_off = 0;
  int64_t k_off = 0;
  // 首 chunk（cc==0）：跨迭代 WAR 已由 ring 尾 drain flag 3 覆盖，无需 M_MTE1/FIX_M。
  fa_dyn::copy_l1_to_l0a_dyn<T, false>(
      ctx.lhs_l0[0], ctx.q_l1[q_off], ctx.block_m,
      ctx.dim_align, ctx.block_m, ctx.cube_k);
  fa_dyn::copy_l1_to_l0b_dyn<T, true>(
      ctx.rhs_l0[0], k_cur[k_off], BLOCK_N,
      ctx.dim_align, ctx.cube_k, BLOCK_N);
  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
  fa_dyn::mma_dyn<T>(ctx.lhs_l0[0], ctx.rhs_l0[0], ctx.acc_s_l0c[0],
                     true, ctx.block_m, BLOCK_N, ctx.cube_k);
  q_off += q_step;
  k_off += k_step;
  for (int64_t cc = 1; cc < ctx.d_chunks; ++cc) {
    // WAR：上 chunk mma 读 lhs/rhs_l0 排干 -> MTE1 覆写
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    fa_dyn::copy_l1_to_l0a_dyn<T, false>(
        ctx.lhs_l0[0], ctx.q_l1[q_off], ctx.block_m,
        ctx.dim_align, ctx.block_m, ctx.cube_k);
    fa_dyn::copy_l1_to_l0b_dyn<T, true>(
        ctx.rhs_l0[0], k_cur[k_off], BLOCK_N,
        ctx.dim_align, ctx.cube_k, BLOCK_N);
    // RAW：L0 就绪 -> mma；WAR：上 chunk FIX 读 acc_s_l0c 排干 -> 本次 mma 覆写
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    fa_dyn::mma_dyn<T>(ctx.lhs_l0[0], ctx.rhs_l0[0], ctx.acc_s_l0c[0],
                       false, ctx.block_m, BLOCK_N, ctx.cube_k);
    q_off += q_step;
    k_off += k_step;
  }
  // RAW：全 chunk 累完 -> FIX 落盘
  AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
  AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
  fa_dyn::copy_l0c_to_gm_dyn<float>(
      ctx.workspace_s[ws_s_base + slot * ctx.block_m * BLOCK_N],
      ctx.acc_s_l0c[0], ctx.block_m, BLOCK_N, BLOCK_N, ctx.block_m, BLOCK_N);
  // HS35 释放槽 cur（供 C1(t+1) 预取复用）
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(cur);
  // HS19 位置 B（C1 跨迭代消费者完成，供 ring 尾 drain）
  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
  AscendC::SetFlag<AscendC::HardEvent::FIX_M>(3);
  AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0);
}

// C2(tc)：O = P @ V -> workspace_o[slot_c]（d_chunks 个 K 段逐段落盘）
template <typename T>
CATLASS_DEVICE void FaGenC2(FaGenContext<T> &ctx, int64_t kv_base,
                            int64_t ws_p_base, int64_t ws_o_base, int64_t slot_c,
                            int64_t tc) {
  AscendC::CrossCoreWaitFlag(1);
  // WAR：上迭代 MTE1 读 acc_s_l1/v_l1 排干 -> MTE2 覆写
  // HS35：flag 让位给 K 槽信用（MTE1_MTE2 0/1、MTE2_MTE1 0/1），改用 3/2。
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(3);
  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(3);
  fa_dyn::copy_gm_to_l1_dyn<T>(
      ctx.acc_s_l1[0], ctx.workspace_p[ws_p_base + slot_c * ctx.block_m * BLOCK_N],
      ctx.block_m, BLOCK_N, BLOCK_N, ctx.block_m, BLOCK_N, true);
  const int64_t v_tail_m = min64(BLOCK_N, ctx.kv_seq_len - tc * BLOCK_N);
  fa_dyn::copy_gm_to_l1_dyn<T>(ctx.v_l1[0],
                               ctx.v[kv_base + tc * BLOCK_N * ctx.dim], BLOCK_N,
                               ctx.dim_align, ctx.dim, v_tail_m, ctx.dim, true);
  // RAW：L1 载 -> L0
  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(2);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(2);
  // WAR：上迭代 mma 读 p_l0 排干 -> MTE1 覆写（p_l0 外提，单次载）
  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
  fa_dyn::copy_l1_to_l0a_dyn<T, false>(ctx.p_l0[0], ctx.acc_s_l1[0],
                                       ctx.block_m, BLOCK_N, ctx.block_m,
                                       BLOCK_N);
  const int64_t v_step = ctx.cube_k * BLOCK_N;
  int64_t v_off = 0;
  int64_t wo_off = ws_o_base + slot_c * ctx.block_m * ctx.dim_align;
  for (int64_t cc = 0; cc < ctx.d_chunks; ++cc) {
    // WAR：上 chunk mma 读 v_l0 排干 -> MTE1 覆写
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    fa_dyn::copy_l1_to_l0b_dyn<T, false>(
        ctx.v_l0[0], ctx.v_l1[v_off], BLOCK_N,
        ctx.dim_align, BLOCK_N, ctx.cube_k);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    // WAR：上 chunk/迭代 FIX 读 acc_o_l0c 排干 -> mma 覆写
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    fa_dyn::mma_dyn<T>(ctx.p_l0[0], ctx.v_l0[0], ctx.acc_o_l0c[0], true,
                       ctx.block_m, ctx.cube_k, BLOCK_N);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
    fa_dyn::copy_l0c_to_gm_dyn<float>(
        ctx.workspace_o[wo_off],
        ctx.acc_o_l0c[0], ctx.block_m, ctx.cube_k, ctx.dim_align, ctx.block_m,
        ctx.cube_k);
    v_off += v_step;
    wo_off += ctx.cube_k;
  }
  // HS19 位置 B（C2 跨迭代消费者完成，供 ring 尾 drain）
  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(5);
  AscendC::SetFlag<AscendC::HardEvent::FIX_M>(4);
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(4);
  AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(2);
}

}  // namespace fa_gen

#endif  // FA_GEN_CUBE_H
