// ----------------------------------------------------------------------------------------------------------
// FlashAttention forward (generalized，手写同步) - Vector 侧：
// V0(running 状态初始化) / V1(scale/mask/online softmax) / V2(merge) / epilogue
// 同步结构（flag 编号 / 顺序）与 tilelang 手写同步设计生成码逐字节一致，禁止改动。
// 同管 V 链（Muls/reduce/Max/Sub/Exp/Broadcast/Div）不插 PIPE_V——in-order FIFO
// 定序（sync pass Fix 1 同规则）。
// ----------------------------------------------------------------------------------------------------------
#ifndef FA_GEN_VEC_H
#define FA_GEN_VEC_H

#include "kernel_common.h"

namespace fa_gen {

// V0：running 状态初始化（acc_o / sumexp / m_i 驻留 UB）
// 免零初始化：acc_o/sumexp 由 V2(0) 直接建立，单元素/单行填充只是
// 活性锚点（生成码逐字节：Fill(acc_o[0],0,dim_align) + Fill(sumexp[0],0,1)）
template <typename T>
CATLASS_DEVICE void FaGenV0Init(FaGenContext<T> &ctx) {
  tl::ascend::Fill<float>(ctx.acc_o[0], 0.000000e+00f, ctx.dim_align);
  tl::ascend::Fill<float>(ctx.sumexp[0], 0.000000e+00f, 1);
  tl::ascend::Fill<float>(ctx.m_i[0], -1073741824, ctx.half_m);
}

// causal 对角块 mask：块内末列超出首行可见界才需要
template <typename T>
CATLASS_DEVICE void FaGenCausalMask(FaGenContext<T> &ctx, int64_t bx, int64_t t,
                                    int64_t vid) {
  // 条件：(t+1)*block_n - 1 > bx*block_m + diff_s
  if (bx * ctx.block_m + ctx.diff_s < (t + 1) * BLOCK_N - 1) {
    const float col_start = static_cast<float>(t * BLOCK_N);
    const float bound_start = static_cast<float>(
        bx * ctx.block_m + vid * ctx.half_m + ctx.diff_s);
    AscendC::CreateVecIndex<float>(ctx.col_idx[0], col_start, BLOCK_N);
    AscendC::CreateVecIndex<float>(ctx.bound_vec[0], bound_start, ctx.half_m);
    // col_2d[r][c] = t*block_n + c（沿行 broadcast）；bound_2d[r][c] = bound_start + r（沿列）
    const uint32_t s_dst[2] = {static_cast<uint32_t>(ctx.half_m),
                               static_cast<uint32_t>(BLOCK_N)};
    const uint32_t s_col[2] = {1, static_cast<uint32_t>(BLOCK_N)};
    const uint32_t s_row[2] = {static_cast<uint32_t>(ctx.half_m), 1};
    tl::ascend::Broadcast<float, 2, 0, false>(ctx.m_i_2d[0], ctx.col_idx[0],
                                              ctx.tmp_ub[0], s_dst, s_col);
    tl::ascend::Broadcast<float, 2, 1, false>(ctx.bound_2d[0], ctx.bound_vec[0],
                                              ctx.tmp_ub[0], s_dst, s_row);
    AscendC::Compare(ctx.causal_cmp[0], ctx.m_i_2d[0], ctx.bound_2d[0],
                     AscendC::CMPMODE::LE, ctx.half_m * BLOCK_N);
    AscendC::Select<float, uint8_t>(
        ctx.acc_s_ub[0], ctx.causal_cmp[0], ctx.acc_s_ub[0],
        static_cast<float>(-1.073742e+09f),
        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, ctx.half_m * BLOCK_N);
  }
}

// V1(t)：scale/mask/online softmax -> workspace_p[slot]；alpha/sumexp_i 落 UB ring
template <typename T>
CATLASS_DEVICE void FaGenV1(FaGenContext<T> &ctx, int64_t ws_s_base,
                            int64_t ws_p_base, int64_t slot, int64_t t,
                            int64_t vid, int64_t bx) {
  AscendC::CrossCoreWaitFlag(0);
  // WAR：上迭代 V 链读 acc_s_ub 排干 -> 本次 MTE2 载
  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
  fa_dyn::copy_ub_to_ub_dyn<float, float>(ctx.alpha_ring[slot * ctx.half_m],
                                          ctx.m_i[0], ctx.half_m, 1,
                                          ctx.half_m, ctx.half_m, ctx.half_m);
  fa_dyn::copy_gm_to_ub_dyn<float>(
      ctx.acc_s_ub[0],
      ctx.workspace_s[ws_s_base + slot * ctx.block_m * BLOCK_N +
                      vid * ctx.half_m * BLOCK_N],
      BLOCK_N, BLOCK_N, ctx.half_m, BLOCK_N, 0.000000e+00f);
  // RAW：MTE2 载 -> V 计算
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
  AscendC::Muls(ctx.acc_s_ub[0], ctx.acc_s_ub[0], ctx.sm_scale,
                ctx.half_m * BLOCK_N);
  // kv 尾块列 mask
  if (0 < ctx.kv_tail) {
    if (t == ctx.kv_loops - 1) {
      tl::ascend::Fill<float>(ctx.mask_col[0], -1.073742e+09f, BLOCK_N);
      tl::ascend::Fill<float>(ctx.mask_col[0], 0.000000e+00f, ctx.kv_tail);
      for (int64_t h_i = 0; h_i < ctx.half_m; ++h_i) {
        AscendC::Add(ctx.acc_s_ub[h_i * BLOCK_N], ctx.acc_s_ub[h_i * BLOCK_N],
                     ctx.mask_col[0], BLOCK_N);
      }
    }
  }
  // causal 对角块 mask
  if (ctx.causal) {
    FaGenCausalMask(ctx, bx, t, vid);
  }
  // online softmax（同管 V 链，零屏障）
  fa_dyn::reduce_max_dyn<float>(ctx.m_i[0], ctx.acc_s_ub[0], ctx.tmp_ub[0],
                                ctx.half_m, BLOCK_N);
  AscendC::Max(ctx.m_i[0], ctx.m_i[0], ctx.alpha_ring[slot * ctx.half_m],
               ctx.half_m);
  AscendC::Sub(ctx.alpha_ring[slot * ctx.half_m], ctx.alpha_ring[slot * ctx.half_m],
               ctx.m_i[0], ctx.half_m);
  AscendC::Exp(ctx.alpha_ring[slot * ctx.half_m], ctx.alpha_ring[slot * ctx.half_m],
               ctx.half_m);
  // brcb-sub（catlass CalcExp 同构，dim_align<=128 门控）：brcb 物化
  // half_m x 8（2KB）替代整幅 2D 广播（32KB）；acc_s_ub 按 [half_m,64] 两半
  // 各一次 sub_mask（行宽恒 64 与 dim_align 无关，故 da<128 同样正确）。
  // dim>128 档 UB 无余量，走原 broadcast+sub。
  if (ctx.dim_align <= 128) {
    const uint8_t brcb_rep = static_cast<uint8_t>(ctx.half_m / 8);
    const uint8_t sub_rep = static_cast<uint8_t>(ctx.half_m);
    tl::ascend::brcb<float>(ctx.brcb_tmp[0], ctx.m_i[0], brcb_rep, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::sub_mask<float>(ctx.acc_s_ub[0], ctx.acc_s_ub[0],
                                ctx.brcb_tmp[0], 0xFFFFFFFFFFFFFFFF, 0, sub_rep,
                                1, 1, 0, 16, 16, 1);
    tl::ascend::brcb<float>(ctx.brcb_tmp[0], ctx.m_i[0], brcb_rep, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::sub_mask<float>(ctx.acc_s_ub[64], ctx.acc_s_ub[64],
                                ctx.brcb_tmp[0], 0xFFFFFFFFFFFFFFFF, 0, sub_rep,
                                1, 1, 0, 16, 16, 1);
  } else {
    const uint32_t s_dst[2] = {static_cast<uint32_t>(ctx.half_m),
                               static_cast<uint32_t>(BLOCK_N)};
    const uint32_t s_src[2] = {static_cast<uint32_t>(ctx.half_m), 1};
    tl::ascend::Broadcast<float, 2, 1, false>(ctx.m_i_2d[0], ctx.m_i[0],
                                              ctx.tmp_ub[0], s_dst, s_src);
    AscendC::Sub(ctx.acc_s_ub[0], ctx.acc_s_ub[0], ctx.m_i_2d[0],
                 ctx.half_m * BLOCK_N);
  }
  AscendC::Exp(ctx.acc_s_ub[0], ctx.acc_s_ub[0], ctx.half_m * BLOCK_N);
  fa_dyn::reduce_sum_dyn<float>(ctx.sumexp_i_ring[slot * ctx.half_m],
                                ctx.acc_s_ub[0], ctx.tmp_ub[0], ctx.half_m,
                                BLOCK_N);
  // WAR：上迭代 MTE3 读 acc_s_half 排干 -> 本次 cast 写
  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
  fa_dyn::copy_ub_to_ub_dyn<T, float>(ctx.acc_s_half[0], ctx.acc_s_ub[0],
                                      ctx.half_m * BLOCK_N, ctx.half_m, BLOCK_N,
                                      BLOCK_N, BLOCK_N);
  // RAW：cast(V) -> P 写 GM(MTE3)
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
  fa_dyn::copy_ub_to_gm_dyn<T>(
      ctx.workspace_p[ws_p_base + slot * ctx.block_m * BLOCK_N +
                      vid * ctx.half_m * BLOCK_N],
      ctx.acc_s_half[0], BLOCK_N, BLOCK_N, ctx.half_m, BLOCK_N);
  AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(1);
}

// V2(tc)：UB ring 读 alpha/sumexp_i，merge 当前 O 贡献到 running 状态
// 压缩 Vector 三分支：
//   now_k==0        —— 免零初始化直接建立（copy acc_o_ub->acc_o / sumexp_i->sumexp）
//   dim_align==128  —— 整幅 broadcast-mul（平坦 Mul 要求 pitch(acc_o)==pitch(m_i_2d)
//                      ==128；da<128 时逐元素错位，实测 dim=64/80/96/112 全错，
//                      见 ascendc/GENERALIZE_DEBUG_LOG.md ISSUE-A1）
//   其余（da!=128） —— 行循环标量 mul
template <typename T>
CATLASS_DEVICE void FaGenV2(FaGenContext<T> &ctx, int64_t ws_o_base,
                            int64_t slot_c, int64_t now_k, int64_t vid) {
  AscendC::CrossCoreWaitFlag(2);
  // WAR：上迭代 V add 读 acc_o_ub 排干 -> MTE2 覆写
  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
  fa_dyn::copy_gm_to_ub_dyn<float>(
      ctx.acc_o_ub[0],
      ctx.workspace_o[ws_o_base + slot_c * ctx.block_m * ctx.dim_align +
                      vid * ctx.half_m * ctx.dim_align],
      ctx.dim_align, ctx.dim_align, ctx.half_m, ctx.dim_align, 0.000000e+00f);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
  if (now_k == 0) {
    fa_dyn::copy_ub_to_ub_dyn<float, float>(
        ctx.acc_o[0], ctx.acc_o_ub[0], ctx.half_m * ctx.dim_align, ctx.half_m,
        ctx.dim_align, ctx.dim_align, ctx.dim_align);
    fa_dyn::copy_ub_to_ub_dyn<float, float>(
        ctx.sumexp[0], ctx.sumexp_i_ring[slot_c * ctx.half_m], ctx.half_m,
        ctx.half_m, 1, 1, 1);
  } else if (ctx.dim_align == 128) {
    const uint32_t s_dst[2] = {static_cast<uint32_t>(ctx.half_m),
                               static_cast<uint32_t>(BLOCK_N)};
    const uint32_t s_src[2] = {static_cast<uint32_t>(ctx.half_m), 1};
    tl::ascend::Broadcast<float, 2, 1, false>(
        ctx.m_i_2d[0], ctx.alpha_ring[slot_c * ctx.half_m], ctx.tmp_ub[0],
        s_dst, s_src);
    AscendC::Mul(ctx.acc_o[0], ctx.acc_o[0], ctx.m_i_2d[0],
                 ctx.half_m * BLOCK_N);
    AscendC::Add(ctx.acc_o[0], ctx.acc_o[0], ctx.acc_o_ub[0],
                 ctx.half_m * BLOCK_N);
    AscendC::Mul(ctx.sumexp[0], ctx.sumexp[0], ctx.alpha_ring[slot_c * ctx.half_m],
                 ctx.half_m);
    AscendC::Add(ctx.sumexp[0], ctx.sumexp[0],
                 ctx.sumexp_i_ring[slot_c * ctx.half_m], ctx.half_m);
  } else {
    for (int64_t h_i = 0; h_i < ctx.half_m; ++h_i) {
      const float alpha = ctx.alpha_ring.GetValue(slot_c * ctx.half_m + h_i);
      AscendC::Muls(ctx.acc_o[h_i * ctx.dim_align], ctx.acc_o[h_i * ctx.dim_align],
                    alpha, ctx.dim_align);
    }
    AscendC::Add(ctx.acc_o[0], ctx.acc_o[0], ctx.acc_o_ub[0],
                 ctx.half_m * ctx.dim_align);
    AscendC::Mul(ctx.sumexp[0], ctx.sumexp[0], ctx.alpha_ring[slot_c * ctx.half_m],
                 ctx.half_m);
    AscendC::Add(ctx.sumexp[0], ctx.sumexp[0],
                 ctx.sumexp_i_ring[slot_c * ctx.half_m], ctx.half_m);
  }
}

// epilogue：归一化（broadcast + 2D div）+ cast + 写出（只写前 dim 列）
template <typename T>
CATLASS_DEVICE void FaGenEpilogue(FaGenContext<T> &ctx, int64_t bx, int64_t vid,
                                  int64_t q_base) {
  for (int64_t h_i = 0; h_i < ctx.half_m; ++h_i) {
    const float rsum = 1.0f / ctx.sumexp.GetValue(h_i);
    AscendC::Muls(ctx.acc_o[h_i * ctx.dim_align], ctx.acc_o[h_i * ctx.dim_align],
                  rsum, ctx.dim_align);
  }
  fa_dyn::copy_ub_to_ub_dyn<T, float>(ctx.acc_o_half[0], ctx.acc_o[0],
                                      ctx.half_m * ctx.dim_align, ctx.half_m,
                                      ctx.dim_align, ctx.dim_align,
                                      ctx.dim_align);
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
  const int64_t out_raw = ctx.q_seq_len - bx * ctx.block_m - vid * ctx.half_m;
  const int64_t out_tail =
      out_raw >= ctx.half_m ? ctx.half_m : (out_raw > 0 ? out_raw : 0);
  if (out_tail > 0) {
    fa_dyn::copy_ub_to_gm_dyn<T>(
        ctx.output[q_base + vid * ctx.half_m * ctx.dim], ctx.acc_o_half[0],
        ctx.dim_align, ctx.dim, out_tail, ctx.dim);
  }
}

}  // namespace fa_gen

#endif  // FA_GEN_VEC_H
