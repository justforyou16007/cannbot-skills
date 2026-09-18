// ----------------------------------------------------------------------------------------------------------
// FlashAttention forward - generalized standalone AscendC kernel (MHA, kv_heads == heads).
//
// Origin: tilelang 手写同步设计 design/tile_level/flash_attention_handsync.py 的编译产物，
// generalized：dim 运行期 + block_m 按 dim_align 分档 + d_chunks K 累加
// + causal 右下对齐 + 全显式旗语（AUTO_SYNC 关闭）。同步结构（flag 编号 / 顺序）
// 与生成码逐字节一致，禁止改动。
// ----------------------------------------------------------------------------------------------------------
#ifndef FA_GEN_KERNEL_H
#define FA_GEN_KERNEL_H

#include "kernel_common.h"
#include "flash_attention_cube.h"
#include "flash_attention_vec.h"

template <typename T>
CATLASS_DEVICE void fa_fwd_run(
    GM_ADDR q_handle, GM_ADDR k_handle, GM_ADDR v_handle, GM_ADDR output_handle,
    GM_ADDR workspace_s_handle, GM_ADDR workspace_p_handle,
    GM_ADDR workspace_o_handle, GM_ADDR tiling_handle) {
  using namespace fa_gen;

  __gm__ FlashAttentionGenTiling* tiling_gm =
      reinterpret_cast<__gm__ FlashAttentionGenTiling*>(tiling_handle);

  FaGenContext<T> ctx;
  ctx.batch = tiling_gm->batch;
  ctx.heads = tiling_gm->heads;
  ctx.q_seq_len = tiling_gm->q_seq_len;
  ctx.kv_seq_len = tiling_gm->kv_seq_len;
  ctx.dim = tiling_gm->dim;
  ctx.sm_scale = tiling_gm->sm_scale;
  ctx.causal = tiling_gm->causal;

  constexpr uint32_t C0 = 32 / sizeof(T);   // 16: fp16/bf16, 8: fp32
  ctx.dim_align = (ctx.dim + C0 - 1) / C0 * C0;
  // dim_align > 128 时按 128 整 chunk 取齐
  if (ctx.dim_align > 128) {
    ctx.dim_align = (ctx.dim + 127) / 128 * 128;
  }
  // block_m 按 dim_align 分档（fp32 减半 + dim>128 也 64）
  if constexpr (sizeof(T) == 4) {
    ctx.block_m = (ctx.dim_align <= 128) ? 64 : 32;
  } else if (ctx.dim_align <= 128) {
    ctx.block_m = 128;
  } else {
    ctx.block_m = 64;
  }
  ctx.half_m = ctx.block_m / 2;
  ctx.cube_k = ctx.dim_align < 256 ? ctx.dim_align : 256;
  ctx.d_chunks = (ctx.dim_align + 255) / 256;
  ctx.diff_s = ctx.kv_seq_len - ctx.q_seq_len;
  ctx.q_blocks = (ctx.q_seq_len + ctx.block_m - 1) / ctx.block_m;
  ctx.block_num = ctx.q_blocks * ctx.heads * ctx.batch;
  ctx.tasks_per_core = (ctx.block_num + MAX_CORES - 1) / MAX_CORES;
  ctx.kv_loops = (ctx.kv_seq_len + BLOCK_N - 1) / BLOCK_N;
  ctx.kv_tail = ctx.kv_seq_len % BLOCK_N;

  // workspace 步长（元素，per-task）
  ctx.ws_s_task = RING * ctx.block_m * BLOCK_N;            // fp32
  ctx.ws_p_task = RING * ctx.block_m * BLOCK_N;            // T
  ctx.ws_o_task = RING * ctx.block_m * ctx.dim_align;      // fp32

  AscendC::TPipe pipe;
  ctx.q.SetGlobalBuffer((__gm__ T*)q_handle);
  ctx.k.SetGlobalBuffer((__gm__ T*)k_handle);
  ctx.v.SetGlobalBuffer((__gm__ T*)v_handle);
  ctx.output.SetGlobalBuffer((__gm__ T*)output_handle);
  ctx.workspace_s.SetGlobalBuffer((__gm__ float*)workspace_s_handle);
  ctx.workspace_p.SetGlobalBuffer((__gm__ T*)workspace_p_handle);
  ctx.workspace_o.SetGlobalBuffer((__gm__ float*)workspace_o_handle);

  // ---- 片上 buffer（运行期尺寸） ---------------------------------------------
  // L1: q @0, k @block_m*da*s, acc_s @(block_m+block_n)*da*s, v @+block_m*block_n*s
  const uint32_t q_sz = ctx.block_m * ctx.dim_align * sizeof(T);
  const uint32_t k_sz = BLOCK_N * ctx.dim_align * sizeof(T);
  const uint32_t p_sz = ctx.block_m * BLOCK_N * sizeof(T);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1;
  pipe.InitBuffer(ascend_l1, q_sz + 2 * k_sz + p_sz + k_sz);  // k_l1 双槽
  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c;
  pipe.InitBuffer(ascend_l0c, 131072);

  // UB 字节偏移（含生命周期互斥别名，见下）
  const uint32_t ub_acc_o = 0;                                       // float[half_m*da]
  const uint32_t ub_sumexp = ub_acc_o + ctx.half_m * ctx.dim_align * 4;
  const uint32_t ub_m_i = ub_sumexp + ctx.half_m * 4;
  const uint32_t ub_alpha_ring = ub_m_i + ctx.half_m * 4;
  const uint32_t ub_acc_s_ub = ub_alpha_ring + RING * ctx.half_m * 4;
  const uint32_t ub_mask_col = ub_acc_s_ub + ctx.half_m * BLOCK_N * 4;
  const uint32_t ub_col_idx = ub_mask_col + BLOCK_N * 4;
  const uint32_t ub_bound_vec = ub_col_idx + BLOCK_N * 4;
  const uint32_t ub_tmp = ub_bound_vec + ctx.half_m * 4;            // uint8[half_m*256]
  const uint32_t ub_m_i_2d = ub_tmp + ctx.half_m * 256;
  // brcb_tmp：dim_align<=128 档 half_m*8 fp32，其余档 8 元素哑元
  // （与设计 builder 常量三元同构；运行期 dim 故尺寸走运行期三元。
  //  生命周期仅限 V1 内部 brcb->sub_mask，不与任何 buffer 别名）
  const uint32_t brcb_elems = (ctx.dim_align <= 128) ? ctx.half_m * 8 : 8;
  const uint32_t ub_brcb = ub_m_i_2d + ctx.half_m * BLOCK_N * 4;
  const uint32_t ub_sumexp_i_ring = ub_brcb + brcb_elems * 4;
  const uint32_t ub_acc_s_half = ub_sumexp_i_ring + RING * ctx.half_m * 4;
  const uint32_t ub_acc_o_ub = ub_acc_s_half + ctx.half_m * BLOCK_N * sizeof(T);
  const uint32_t ub_base_total = ub_acc_o_ub + ctx.half_m * ctx.dim_align * 4;
  // 别名：
  //   alpha_2d  == m_i_2d  区（V2/epilogue vs V1 生命周期互斥）
  //   acc_o_half == acc_s_ub 区（epilogue vs V1 生命周期互斥）
  //   causal_cmp == acc_s_half 区（V1 mask vs V1 cast 时序互斥）
  //   bound_2d  == acc_o_ub 区（V1 causal vs V2 生命周期互斥；需 half_m*128 <=
  //                half_m*da 即 da>=128，否则独立放在 ub_base_total 之后）
  // 条件别名（生命周期互斥，但别名目标大小必须容得下被别名对象）：
  //   causal_cmp == acc_s_half（V1 mask vs V1 cast；half_m*128 <= half_m*128*sizeof(T) 恒成立）
  //   bound_2d  == acc_o_ub（V1 causal vs V2；需 da>=128 使 half_m*128 <= half_m*da）
  //   alpha_2d  == m_i_2d（V2/epilogue vs V1；需 da<=128 使 half_m*da <= half_m*128）
  //   acc_o_half == acc_s_ub（epilogue vs V1；需 da*sizeof(T)<=512）
  // 不满足时独立分配（容量已按最坏 da<=128 / da>128 两档核算，均 <=192KB）。
  const uint32_t ub_causal_cmp = ub_acc_s_half;
  const bool bound_alias = (ctx.dim_align >= 128);
  const uint32_t ub_bound_2d = bound_alias ? ub_acc_o_ub : ub_base_total;
  const uint32_t ub_after_bound =
      bound_alias ? ub_base_total : ub_base_total + ctx.half_m * BLOCK_N * 4;
  const uint32_t ub_acc_o_half = ub_acc_o_ub;  // epilogue vs V2 生命周期错开
  const uint32_t ub_total = ub_after_bound;
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub;
  pipe.InitBuffer(ascend_ub, ub_total);
  pipe.Destroy();

  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto vid = AscendC::GetSubBlockIdx();

  // ---- buffer 视图 ------------------------------------------------------------
  ctx.q_l1 = ascend_l1.GetWithOffset<T>(ctx.block_m * ctx.dim_align, 0);
  ctx.k_l1 = ascend_l1.GetWithOffset<T>(BLOCK_N * ctx.dim_align, q_sz);
  ctx.k_l1b = ascend_l1.GetWithOffset<T>(BLOCK_N * ctx.dim_align, q_sz + k_sz);
  ctx.acc_s_l1 = ascend_l1.GetWithOffset<T>(ctx.block_m * BLOCK_N, q_sz + 2 * k_sz);
  ctx.v_l1 = ascend_l1.GetWithOffset<T>(BLOCK_N * ctx.dim_align, q_sz + 2 * k_sz + p_sz);

  // L0A/L0B 别名复用：fp32 tile 占满 64KB，QK(lhs/rhs) 与 PV(p/v) 时分复用
  // 同一 buffer；fp16/bf16 分开两半。分档后偏移由"放得下则分开、放不下则别名"决定。
  const uint32_t lhs_sz = ctx.block_m * ctx.cube_k * sizeof(T);
  const uint32_t rhs_sz = ctx.cube_k * BLOCK_N * sizeof(T);
  const uint32_t p_l0_off = (lhs_sz + p_sz <= 65536) ? lhs_sz : 0;
  const uint32_t v_l0_off = (2 * rhs_sz <= 65536) ? rhs_sz : 0;
  ctx.lhs_l0 = ascend_l0a.GetWithOffset<T>(ctx.block_m * ctx.cube_k, 0);
  ctx.p_l0 = ascend_l0a.GetWithOffset<T>(ctx.block_m * BLOCK_N, p_l0_off);
  ctx.rhs_l0 = ascend_l0b.GetWithOffset<T>(ctx.cube_k * BLOCK_N, 0);
  ctx.v_l0 = ascend_l0b.GetWithOffset<T>(BLOCK_N * ctx.cube_k, v_l0_off);
  ctx.acc_s_l0c = ascend_l0c.GetWithOffset<float>(ctx.block_m * BLOCK_N, 0);
  ctx.acc_o_l0c = ascend_l0c.GetWithOffset<float>(
      ctx.block_m * ctx.cube_k, ctx.block_m * BLOCK_N * 4);

  ctx.acc_o = ascend_ub.GetWithOffset<float>(ctx.half_m * ctx.dim_align, ub_acc_o);
  ctx.sumexp = ascend_ub.GetWithOffset<float>(ctx.half_m, ub_sumexp);
  ctx.m_i = ascend_ub.GetWithOffset<float>(ctx.half_m, ub_m_i);
  ctx.alpha_ring = ascend_ub.GetWithOffset<float>(RING * ctx.half_m, ub_alpha_ring);
  ctx.acc_s_ub = ascend_ub.GetWithOffset<float>(ctx.half_m * BLOCK_N, ub_acc_s_ub);
  ctx.mask_col = ascend_ub.GetWithOffset<float>(BLOCK_N, ub_mask_col);
  ctx.col_idx = ascend_ub.GetWithOffset<float>(BLOCK_N, ub_col_idx);
  ctx.bound_vec = ascend_ub.GetWithOffset<float>(ctx.half_m, ub_bound_vec);
  ctx.tmp_ub = ascend_ub.GetWithOffset<uint8_t>(ctx.half_m * 256, ub_tmp);
  ctx.m_i_2d = ascend_ub.GetWithOffset<float>(ctx.half_m * BLOCK_N, ub_m_i_2d);
  ctx.brcb_tmp = ascend_ub.GetWithOffset<float>(brcb_elems, ub_brcb);
  ctx.sumexp_i_ring =
      ascend_ub.GetWithOffset<float>(RING * ctx.half_m, ub_sumexp_i_ring);
  ctx.acc_s_half = ascend_ub.GetWithOffset<T>(ctx.half_m * BLOCK_N, ub_acc_s_half);
  ctx.acc_o_ub =
      ascend_ub.GetWithOffset<float>(ctx.half_m * ctx.dim_align, ub_acc_o_ub);
  ctx.acc_o_half =
      ascend_ub.GetWithOffset<T>(ctx.half_m * ctx.dim_align, ub_acc_o_half);
  ctx.causal_cmp =
      ascend_ub.GetWithOffset<uint8_t>(ctx.half_m * BLOCK_N, ub_causal_cmp);
  ctx.bound_2d =
      ascend_ub.GetWithOffset<float>(ctx.half_m * BLOCK_N, ub_bound_2d);

  // ---- task / ring 主循环 ------------------------------------------------------
  for (int64_t local_idx = 0; local_idx < ctx.tasks_per_core; ++local_idx) {
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t task_idx = cid * ctx.tasks_per_core + local_idx;
    if (task_idx < ctx.block_num) {
      const int64_t bx = task_idx % ctx.q_blocks;
      const int64_t by = (task_idx / ctx.q_blocks) % ctx.heads;
      const int64_t bz = task_idx / ctx.q_blocks / ctx.heads;
      const int64_t q_base =
          ((bz * ctx.heads + by) * ctx.q_seq_len + bx * ctx.block_m) * ctx.dim;
      const int64_t kv_base = (bz * ctx.heads + by) * ctx.kv_seq_len * ctx.dim;
      // workspace 按 task 索引（handsyc 设计语义）
      const int64_t ws_s_base = task_idx * ctx.ws_s_task;
      const int64_t ws_p_base = task_idx * ctx.ws_p_task;
      const int64_t ws_o_base = task_idx * ctx.ws_o_task;
      const int64_t q_tail_m = min64(ctx.block_m, ctx.q_seq_len - bx * ctx.block_m);

      if ASCEND_IS_AIC {
        FaGenC0LoadQ(ctx, q_base, q_tail_m);
        // preload：K 槽信用预置 + 预加载 K(0) 到槽 0
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);   // K 槽 0 空闲
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);   // K 槽 1 空闲
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        fa_dyn::copy_gm_to_l1_dyn<T>(
            ctx.k_l1[0], ctx.k[kv_base], BLOCK_N, ctx.dim_align, ctx.dim,
            min64(BLOCK_N, ctx.kv_seq_len), ctx.dim, true);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);   // K 槽 0 就绪
      }
      if ASCEND_IS_AIV {
        FaGenV0Init(ctx);
      }

      for (int64_t t = 0; t < ctx.kv_loops + PRELAUNCH; ++t) {
        if (t < ctx.kv_loops) {
          const int64_t slot = t % RING;
          if ASCEND_IS_AIC {
            FaGenC1(ctx, kv_base, ws_s_base, slot, t);
          }
          if ASCEND_IS_AIV {
            FaGenV1(ctx, ws_s_base, ws_p_base, slot, t, vid, bx);
          }
        }
        if (PRELAUNCH <= t) {
          const int64_t tc = t - PRELAUNCH;
          const int64_t slot_c = tc % RING;
          if ASCEND_IS_AIC {
            FaGenC2(ctx, kv_base, ws_p_base, ws_o_base, slot_c, tc);
          }
          if ASCEND_IS_AIV {
            FaGenV2(ctx, ws_o_base, slot_c, tc, vid);
          }
        }
        // ring 尾定向 drain（替代哨兵 PIPE_ALL）
        if ASCEND_IS_AIC {
          if (t < ctx.kv_loops) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(3);
          }
          if (PRELAUNCH <= t) {
            const int64_t now_k = t - PRELAUNCH;
            if (now_k < ctx.kv_loops) {
              AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(5);
              AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(4);
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(4);
            }
          }
        }
      }
      // task 尾 drain：K 槽 0/1 空闲信用清零（每槽残留 1 个 set）
      if ASCEND_IS_AIC {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
      }
      if ASCEND_IS_AIV {
        FaGenEpilogue(ctx, bx, vid, q_base);
      }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }
}

#endif  // FA_GEN_KERNEL_H
