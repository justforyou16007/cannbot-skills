// ----------------------------------------------------------------------------------------------------------
// GQA forward (generalized) - kernel 编排：tiling 读取 + buffer 分配 + task/ring 主循环
//
// 与 ascendc/gqa.cpp 的 gqa_fwd_kernel 同源（设备代码零逻辑改动，仅 ABI 适配：
// shape/标量从 GM tiling 读取，删除 tilelang JIT 残留的 fftsAddr）。
// 同步结构逐字节保留，禁止"顺手优化"。
// ----------------------------------------------------------------------------------------------------------
#ifndef GQA_GEN_KERNEL_H
#define GQA_GEN_KERNEL_H

#include "kernel_common.h"

template <typename T>
CATLASS_DEVICE void gqa_fwd_run(
    GM_ADDR q_handle, GM_ADDR k_handle, GM_ADDR v_handle, GM_ADDR output_handle,
    GM_ADDR workspace_s_handle, GM_ADDR workspace_p_handle,
    GM_ADDR workspace_o_handle, GM_ADDR tiling_handle) {
  using namespace gqa_gen;

  // ---- tiling 读取（GM） ----------------------------------------------------
  __gm__ GqaGenTiling* tiling_gm =
      reinterpret_cast<__gm__ GqaGenTiling*>(tiling_handle);
  const int64_t batch = tiling_gm->batch;
  const int64_t heads_q = tiling_gm->heads_q;
  const int64_t heads_kv = tiling_gm->heads_kv;
  const int64_t q_seq_len = tiling_gm->q_seq_len;
  const int64_t kv_seq_len = tiling_gm->kv_seq_len;
  const int64_t dim = tiling_gm->dim;
  const int64_t causal = tiling_gm->causal;
  const float sm_scale = tiling_gm->sm_scale;

  constexpr uint32_t C0 = 32 / sizeof(T);   // 16: fp16/bf16

  // ---- runtime shape variables (mirror design/tile_level/gqa.py) -----------
  const int64_t group = heads_q / heads_kv;
  const int64_t dim_align = (dim + C0 - 1) / C0 * C0;   // on-chip buffer width
  const int64_t cube_k = min64(dim_align, 128);
  const int64_t d_chunks = (dim_align + 127) / 128;
  // HS21 (D extension to 512): per-tier row budget keeps UB/L0A/L1 within
  // capacity: dim_align<=128 -> 128 (dc=1); <=256 -> 64 (dc=2); <=512 -> 32.
  const int64_t m_budget =
      dim_align <= 128 ? 128 : (dim_align <= 256 ? 64 : 32);
  // HS21: K/V L1 slots. dim_align>=512 -> single slot (double-slot K/V alone
  // would fill L1); P2 prefetch disabled when kv_slots==1. Slot indices are
  // %kv_slots; flag ids unchanged (K=0/1, V=2/3, unused ids still paired).
  const int64_t kv_slots = dim_align >= 512 ? 1 : 2;
  const int64_t bs = min64(m_budget, q_seq_len);
  const int64_t bg = min64(group, (m_budget / bs) > 1 ? (m_budget / bs) : 1);
  const int64_t block_m = ((bs * bg + 15) / 16) * 16;
  const int64_t bm2 = block_m / 2;             // rows per AIV subcore
  // FAS-alignment (nRatio-style S2 amplification, production FIAS
  // S1s2Bn2gs1 CalcNRatio): widen the KV tile 128 -> 256 for small-block_m
  // shapes (decode/MTP), halving the ring iteration count. L0B (64KB) caps a
  // single mma N-chunk at 128 cols, so the widened tile is consumed as
  // n_chunks = BLOCK_N/128 sub-chunks (C1: per-chunk mma+FIX into ws_s column
  // chunks; C2: K-dim accumulate init=(nc==0), single FIX). Capacity gate:
  // block_m<=64 (L0A/UB), dim_align<=128 (L1), kv_seq_len>=1024 (at
  // kv_loops=2 the ring never reaches steady state and 256 REGRESSES ~4%,
  // case 17 A/B). Must match host Derive (op_host/gqa_gen.cpp).
  const int64_t BLOCK_N =
      (dim_align <= 128 && block_m <= 64 && kv_seq_len >= 1024) ? 256 : 128;
  const int64_t n_chunks = BLOCK_N / BLOCK_N_BASE;   // L0B col chunks/iter
  const int64_t s_blks = (q_seq_len + bs - 1) / bs;
  const int64_t g_blks = (group + bg - 1) / bg;
  const int64_t block_num = batch * heads_kv * s_blks * g_blks;
  // Runtime core count (never hardcoded): under MIX_AIC_1_2 both AIC and AIV
  // GetBlockNum() return the host numBlocks (= cube core count), so the task
  // split is consistent on both sides; the host launches with the AI core count.
  const int64_t num_cores = AscendC::GetBlockNum();
  const int64_t tasks_per_core = (block_num + num_cores - 1) / num_cores;
  const int64_t kv_loops = (kv_seq_len + BLOCK_N - 1) / BLOCK_N;
  const int64_t tail_valid = kv_seq_len % BLOCK_N;
  // L1Reuse (K/V full residency in L1): when dim_align<=128 and the FULL K/V
  // (kv_loops*BLOCK_N*dim_align elems each) + Q + acc_s fit L1, each core loads
  // the (bz,nkv) group's entire K/V ONCE at the group head into a resident L1
  // region; later same-group tasks (g_blk/s_blk fast-varying, per-core tasks
  // contiguous) reuse it and skip ALL K/V GM->L1 traffic AND slot-credit
  // handshakes. Group-head test is pure arithmetic (prev task = task_idx-1).
  // Byte-formula capacity gate (64KB margin) excludes nRatio/d>=256 shapes.
  const int64_t kv_align = kv_loops * BLOCK_N;
  const bool kv_resident =
      (dim_align <= 128) &&
      ((block_m * dim_align + 2 * kv_align * dim_align + block_m * BLOCK_N) *
           (int64_t)sizeof(T) <=
       448 * 1024) &&
      (s_blks * g_blks >= 4);
  const int64_t kv_l1_slots = kv_resident ? kv_loops : kv_slots;
  const int64_t diff_s = kv_seq_len - q_seq_len;   // causal right-align offset

  const int64_t bands_per_half = (bm2 % bs == 0) ? (bm2 / bs) : 0;
  const int64_t sub_per_band = (bs % bm2 == 0) ? (bs / bm2) : 0;

  // workspace strides (elements)
  // P3 (D256 L0C K-dim accumulate): C1's d_chunks accumulate into one acc_s_l0c
  // via mma init=(cc==0), so ws_s holds ONE slab per ring slot (d_chunks factor
  // removed -> /2). C2 UNCHANGED (its d_chunks slice output N columns, not K).
  const int64_t ws_s_task = RING * block_m * BLOCK_N;                // fp32 (P3: no d_chunks)
  const int64_t ws_p_task = RING * block_m * BLOCK_N;              // T
  const int64_t ws_o_task = RING * block_m * dim_align;            // fp32

  AscendC::TPipe pipe;

  AscendC::GlobalTensor<T> q;
  q.SetGlobalBuffer((__gm__ T*)q_handle);
  AscendC::GlobalTensor<T> k;
  k.SetGlobalBuffer((__gm__ T*)k_handle);
  AscendC::GlobalTensor<T> v;
  v.SetGlobalBuffer((__gm__ T*)v_handle);
  AscendC::GlobalTensor<T> output;
  output.SetGlobalBuffer((__gm__ T*)output_handle);
  AscendC::GlobalTensor<float> workspace_s;
  workspace_s.SetGlobalBuffer((__gm__ float*)workspace_s_handle);
  AscendC::GlobalTensor<T> workspace_p;
  workspace_p.SetGlobalBuffer((__gm__ T*)workspace_p_handle);
  AscendC::GlobalTensor<float> workspace_o;
  workspace_o.SetGlobalBuffer((__gm__ float*)workspace_o_handle);

  // ---- on-chip buffers (runtime-sized) -------------------------------------
  // L1 (handwritten-sync, K/V double-buffered; layout order q, k[2], v[2],
  // acc_s). See ascendc/gqa.cpp for the full handwritten-sync notes.
  const uint32_t L1_Q = block_m * dim_align * sizeof(T);
  // L1Reuse: kv_resident widens K/V to the FULL kv extent (kv_loops slots).
  const uint32_t L1_K = kv_l1_slots * BLOCK_N * dim_align * sizeof(T);  // HS21 slots
  const uint32_t L1_V = kv_l1_slots * BLOCK_N * dim_align * sizeof(T);
  const uint32_t L1_S = block_m * BLOCK_N * sizeof(T);
  const uint32_t l1_q_off = 0;
  const uint32_t l1_k_off = l1_q_off + L1_Q;
  const uint32_t l1_v_off = l1_k_off + L1_K;
  const uint32_t l1_s_off = l1_v_off + L1_V;
  const uint32_t l1_total = l1_s_off + L1_S;

  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1;
  pipe.InitBuffer(ascend_l1, l1_total);
  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c;
  pipe.InitBuffer(ascend_l0c, 131072);

  // UB byte-offset chain (runtime widths; fixed order). alpha_2d aliases
  // acc_s_ub (see ascendc/gqa.cpp note; liveness never overlaps on one AIV).
  const uint32_t bn = BLOCK_N;
  auto a32 = [](uint32_t x) { return (x + 31u) / 32u * 32u; };
  const uint32_t ub_acc_o = 0;                              // float[bm2*da]
  const uint32_t ub_sumexp = ub_acc_o + a32(bm2 * dim_align * 4);
  const uint32_t ub_m_i = ub_sumexp + a32(bm2 * 4);
  const uint32_t ub_alpha_ring = ub_m_i + a32(bm2 * 4);     // float[RING*bm2]
  const uint32_t ub_acc_s_ub = ub_alpha_ring + a32(RING * bm2 * 4);  // float[bm2*bn]
  uint32_t ub_acc_s_tmp = ub_acc_s_ub;                      // alias when dc==1
  uint32_t ub_after_s = ub_acc_s_ub + a32(bm2 * bn * 4);
  if (d_chunks > 1) {
    ub_acc_s_tmp = ub_after_s;                              // float[bm2*bn]
    ub_after_s = ub_acc_s_tmp + a32(bm2 * bn * 4);
  }
  const uint32_t ub_col_idx = ub_after_s;                   // float[bn]
  const uint32_t ub_mask_col = ub_col_idx + a32(bn * 4);    // float[bn]
  // P4: causal_cmp widened from uint8[bn] to uint8[bm2*bn] (whole-tile mask).
  const uint32_t ub_causal_cmp = ub_mask_col + a32(bn * 4); // uint8[bm2*bn]
  // P4: bound_vec holds the per-row visible bound for the whole-tile compare.
  const uint32_t ub_bound_vec = ub_causal_cmp + a32(bm2 * bn);  // float[bm2]
  const uint32_t ub_tmp = ub_bound_vec + a32(bm2 * 4);      // uint8[bm2*bn*2]
  const uint32_t ub_m_i_2d = ub_tmp + a32(bm2 * bn * 2);    // float[bm2*bn]
  const uint32_t ub_sumexp_i_ring = ub_m_i_2d + a32(bm2 * bn * 4);  // float[RING*bm2]
  const uint32_t ub_acc_s_half = ub_sumexp_i_ring + a32(RING * bm2 * 4);  // T[bm2*bn]
  const uint32_t ub_acc_o_ub = ub_acc_s_half + a32(bm2 * bn * sizeof(T));  // float[bm2*da]
  const uint32_t ub_acc_o_half = ub_acc_o_ub + a32(bm2 * dim_align * 4);   // T[bm2*da]
  uint32_t ub_total = ub_acc_o_half + a32(bm2 * dim_align * sizeof(T));
  // HS22 (alpha_2d overflow fix): alpha_2d (V2/epilogue broadcast target, width
  // bm2*dim_align) aliases acc_s_ub (width bm2*BLOCK_N) only safely while
  // dim_align <= 2*BLOCK_N. At dim_align=512 the bm2*512 write overruns into
  // sumexp_i_ring / acc_s_half -> inf in the epilogue. Alias for dim_align<=256
  // (byte-identical to verified layout), independent tail allocation for
  // dim_align>256 (d384/d512).
  const bool alpha_2d_indep = dim_align > 256;
  const uint32_t ub_alpha_2d =
      alpha_2d_indep ? ub_total : ub_acc_s_ub;            // float[bm2*da]
  if (alpha_2d_indep) {
    ub_total = ub_alpha_2d + a32(bm2 * dim_align * 4);
  }
  // P4 whole-tile mask broadcast targets, aliased onto buffers idle at the V1
  // mask point: col_2d reuses acc_o_ub (read only in V2), bound_2d reuses
  // m_i_2d (written only by the broadcast AFTER the mask).
  const uint32_t ub_col_2d = ub_acc_o_ub;                   // alias float[bm2*bn]
  const uint32_t ub_bound_2d = ub_m_i_2d;                   // alias float[bm2*bn]

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub;
  pipe.InitBuffer(ascend_ub, ub_total);
  pipe.Destroy();

  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto vid = AscendC::GetSubBlockIdx();

  // ---- buffer views ---------------------------------------------------------
  auto q_l1 = ascend_l1.GetWithOffset<T>(block_m * dim_align, l1_q_off);
  // K/V are 2-slot double buffers: slot s occupies [s*BLOCK_N*dim_align, ...).
  auto k_l1 = ascend_l1.GetWithOffset<T>(kv_l1_slots * BLOCK_N * dim_align, l1_k_off);
  auto v_l1 = ascend_l1.GetWithOffset<T>(kv_l1_slots * BLOCK_N * dim_align, l1_v_off);
  auto acc_s_l1 = ascend_l1.GetWithOffset<T>(block_m * BLOCK_N, l1_s_off);
  // element offset of K/V slot s within its double buffer (zN tile is contiguous).
  const uint32_t kv_slot_elems = BLOCK_N * dim_align;

  // HS17 task 1 (Q resident in L0A): lhs_l0 spans ALL d_chunks chunks
  // (d_chunks*block_m*cube_k elems @ offset 0), loaded ONCE before the ring and
  // reused across all ring iterations (Q is loop-invariant). Capacity invariant
  // (verified over all 20 cases): d_chunks*block_m*cube_k*2B <= 32768B, so the
  // resident Q never overlaps p_l0 @ 32768.
  auto lhs_l0 = ascend_l0a.GetWithOffset<T>(d_chunks * block_m * cube_k, 0);
  auto p_l0 = ascend_l0a.GetWithOffset<T>(block_m * BLOCK_N, 32768);
  auto rhs_l0 = ascend_l0b.GetWithOffset<T>(BLOCK_N_BASE * cube_k, 0);
  auto v_l0 = ascend_l0b.GetWithOffset<T>(BLOCK_N_BASE * cube_k, 32768);
  auto acc_s_l0c = ascend_l0c.GetWithOffset<float>(block_m * BLOCK_N_BASE, 0);
  auto acc_o_l0c = ascend_l0c.GetWithOffset<float>(block_m * cube_k, 65536);

  auto acc_o = ascend_ub.GetWithOffset<float>(bm2 * dim_align, ub_acc_o);
  auto sumexp = ascend_ub.GetWithOffset<float>(bm2, ub_sumexp);
  auto m_i = ascend_ub.GetWithOffset<float>(bm2, ub_m_i);
  auto alpha_ring = ascend_ub.GetWithOffset<float>(RING * bm2, ub_alpha_ring);
  auto acc_s_ub = ascend_ub.GetWithOffset<float>(bm2 * bn, ub_acc_s_ub);
  auto acc_s_tmp = ascend_ub.GetWithOffset<float>(bm2 * bn, ub_acc_s_tmp);
  auto col_idx = ascend_ub.GetWithOffset<float>(bn, ub_col_idx);
  auto mask_col = ascend_ub.GetWithOffset<float>(bn, ub_mask_col);
  // P4: whole-tile causal mask (bm2*bn) + bound vector + aliased 2D broadcasts.
  auto causal_cmp = ascend_ub.GetWithOffset<uint8_t>(bm2 * bn, ub_causal_cmp);
  auto bound_vec = ascend_ub.GetWithOffset<float>(bm2, ub_bound_vec);
  auto col_2d = ascend_ub.GetWithOffset<float>(bm2 * bn, ub_col_2d);
  auto bound_2d = ascend_ub.GetWithOffset<float>(bm2 * bn, ub_bound_2d);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(bm2 * bn * 2, ub_tmp);
  auto m_i_2d = ascend_ub.GetWithOffset<float>(bm2 * bn, ub_m_i_2d);
  auto sumexp_i_ring =
      ascend_ub.GetWithOffset<float>(RING * bm2, ub_sumexp_i_ring);
  auto acc_s_half = ascend_ub.GetWithOffset<T>(bm2 * bn, ub_acc_s_half);
  auto acc_o_ub = ascend_ub.GetWithOffset<float>(bm2 * dim_align, ub_acc_o_ub);
  auto acc_o_half = ascend_ub.GetWithOffset<T>(bm2 * dim_align, ub_acc_o_half);
  auto alpha_2d = ascend_ub.GetWithOffset<float>(bm2 * dim_align, ub_alpha_2d);

  for (int64_t local_idx = 0; local_idx < tasks_per_core; ++local_idx) {
    // HS19: task-loop-top PipeBarrier<PIPE_ALL> REMOVED (de-PIPE_ALL). Cross-task
    // protection is covered downstream: AIC drained at ring tail (position B) +
    // C0 preset slot credits; AIV acc_o_half cross-task WAR covered by task-tail
    // MTE3_V(7) (position C). V0's tile.fill writes acc_o/sumexp/m_i (distinct UB
    // offsets from acc_o_half) -> no address overlap. Pure redundant barrier.
    const int64_t task_idx = cid * tasks_per_core + local_idx;
    if (task_idx < block_num) {
      const int64_t g_blk = task_idx % g_blks;
      const int64_t s_blk = (task_idx / g_blks) % s_blks;
      const int64_t nkv = (task_idx / (g_blks * s_blks)) % heads_kv;
      const int64_t bz = task_idx / (g_blks * s_blks * heads_kv);

      const int64_t s0 = s_blk * bs;
      const int64_t hq0 = nkv * group + g_blk * bg;
      const int64_t kv_end =
          causal ? min64(kv_loops, (s0 + bs - 1 + diff_s) / BLOCK_N + 1)
                 : kv_loops;

      const int64_t q_base =
          bz * (q_seq_len * heads_q * dim) + s0 * (heads_q * dim) + hq0 * dim;
      const int64_t kv_base = bz * (kv_seq_len * heads_kv * dim) + nkv * dim;
      const int64_t ws_s_base = task_idx * ws_s_task;
      const int64_t ws_p_base = task_idx * ws_p_task;
      const int64_t ws_o_base = task_idx * ws_o_task;
      const int64_t q_rows = min64(bs, q_seq_len - s0);

      // ---- C0: preset K/V slot credits + Q resident in L1 + preload K0/V0 ---
      // Handwritten slot-credit scheme (mirrors design/tile_level/gqa.py and
      // ascendc/gqa.cpp): MTE1_MTE2 = slot free / MTE2_MTE1 = slot ready; K slots
      // ids 0/1, V slots ids 2/3, Q resident id 4, acc_s_l1 id 5. C1 L0: MTE1_M
      // id 2, M_FIX id 3, reverse M_MTE1 id 4 / FIX_M id 5. C2 L0: MTE1_M id 7,
      // M_FIX id 0, reverse M_MTE1 id 6 / FIX_M id 1.
      if ASCEND_IS_AIC {
        // L1Reuse: resident mode issues no slot credits (the ring neither
        // consumes nor drains them -- symmetric removal keeps levels balanced).
        if (!kv_resident) {
          // Preset K/V slot-free credits (closed within this task).
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);   // K slot 0 free
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);   // K slot 1 free
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);   // V slot 0 free
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(3);   // V slot 1 free
        }
        // P6 (decode Q-splice DMA merge): when bs==1 the bg group-band rows are
        // CONTIGUOUS in GM (same s0 row, heads hq0.. step by dim -> a [g_valid,
        // dim] tile with row-stride dim). Merge the bg small per-g DMAs into ONE
        // [g_valid,dim] copy. L1 row layout UNCHANGED (bs==1 -> row gg = head
        // hq0+gg), so the epilogue row mapping is untouched. bs>1 keeps per-g.
        if (bs == 1) {
          const int64_t g_valid = min64(bg, group - g_blk * bg);
          gqa_dyn::copy_gm_to_l1_dyn<T>(
              q_l1[0], q[q_base], block_m, dim_align, /*realSrcN=*/dim,
              /*tailM=*/g_valid, /*tailN=*/dim, /*need_clear=*/true);
        } else {
          for (int64_t gg = 0; gg < bg; ++gg) {
            if (g_blk * bg + gg < group) {
              gqa_dyn::copy_gm_to_l1_dyn<T>(
                  q_l1[gg * bs * C0],
                  q[q_base + gg * dim],
                  block_m, dim_align, heads_q * dim, q_rows, dim,
                  /*need_clear=*/(bg == 1));
            }
          }
        }
        // L1Reuse group-head load: on this core's first task or a (bz,nkv)
        // group switch, load the group's ENTIRE K/V block-by-block into the
        // resident L1 region (per-block copy keeps tail need_clear zero-fill).
        // WAR safety: the previous task's ring-tail M_MTE1/FIX_M drains
        // guarantee all its MTE1 reads of the resident K/V already completed.
        if (kv_resident) {
          const int64_t tasks_per_grp = g_blks * s_blks;
          const int64_t grp = task_idx / tasks_per_grp;
          if (local_idx == 0 || (task_idx - 1) / tasks_per_grp != grp) {
            for (int64_t t2 = 0; t2 < kv_loops; ++t2) {
              const int64_t rows = min64(BLOCK_N, kv_seq_len - t2 * BLOCK_N);
              gqa_dyn::copy_gm_to_l1_dyn<T>(
                  k_l1[t2 * kv_slot_elems],
                  k[kv_base + t2 * BLOCK_N * heads_kv * dim], BLOCK_N,
                  dim_align, heads_kv * dim, rows, dim, true);
            }
            for (int64_t t2 = 0; t2 < kv_loops; ++t2) {
              const int64_t rows = min64(BLOCK_N, kv_seq_len - t2 * BLOCK_N);
              gqa_dyn::copy_gm_to_l1_dyn<T>(
                  v_l1[t2 * kv_slot_elems],
                  v[kv_base + t2 * BLOCK_N * heads_kv * dim], BLOCK_N,
                  dim_align, heads_kv * dim, rows, dim, true);
            }
          }
        }
        // HS29-B2 (C0 reorder, forward-fixed): K0/V0 preload issued BETWEEN
        // set(4) and wait(4) -- the positional marker covers ONLY Q, so Q->L0A
        // no longer spans the K/V DMAs (v9's pre-set(4) placement stretched
        // Q's fence: neutral on A2, -0.26 online on A3), while K/V DMAs still
        // overlap the Q wait + L0A fill. Credit accounting unchanged (HS29
        // residue discipline: per-task preset+drain, kernel-end level 0).
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(4);
        if (!kv_resident) {
          if (kv_end > 0) {
            const int64_t k0_rows = min64(BLOCK_N, kv_seq_len - 0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);   // K slot 0 free
            gqa_dyn::copy_gm_to_l1_dyn<T>(
                k_l1[0], k[kv_base + 0], BLOCK_N, dim_align, heads_kv * dim,
                k0_rows, dim, true);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);    // K slot 0 ready
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);   // V slot 0 free
            gqa_dyn::copy_gm_to_l1_dyn<T>(
                v_l1[0], v[kv_base + 0], BLOCK_N, dim_align, heads_kv * dim,
                k0_rows, dim, true);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(2);    // V slot 0 ready
          }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(4);
        // HS17 task 1 (Q resident in L0A): load ALL d_chunks of Q from L1 into
        // the L0A resident region ONCE here (ring-out), so C1's ring loop reuses
        // it without reloading. The MTE1 write of L0A -> first ring mma read of
        // L0A handshake is covered by C1's MTE1_M(2).
        for (int64_t cc = 0; cc < d_chunks; ++cc) {
          gqa_dyn::copy_l1_to_l0a_dyn<T, false>(
              lhs_l0[cc * block_m * cube_k], q_l1[cc * cube_k * block_m],
              block_m, dim_align, block_m, cube_k);
        }
        // (HS29-B2: K0/V0 preload moved between set(4) and wait(4))
      }
      // ---- V0: running state init ------------------------------------------
      if ASCEND_IS_AIV {
        tl::ascend::Fill<float>(acc_o[0], 0.000000e+00f, bm2 * dim_align);
        tl::ascend::Fill<float>(sumexp[0], 0.000000e+00f, bm2);
        tl::ascend::Fill<float>(m_i[0], MASK_NEG, bm2);
      }

      for (int64_t t = 0; t < kv_loops + PRELAUNCH; ++t) {
        // (handwritten-sync) ring-top barrier REMOVED (it blocked the P2
        // prefetch<->mma overlap); cross-iteration K/V/Q reuse is governed by
        // the C-side slot flags, with a single ring-tail PipeBarrier<PIPE_ALL>.
        if (t < kv_end) {
          const int64_t slot_prod = t % RING;
          // ================= C1: S = Q @ K^T -> workspace_s ================
          if ASCEND_IS_AIC {
            // P2 double-slot overlap: prefetch K(t+1) into slot (t+1)%kv_slots
            // (MTE2 overlaps the MTE1/mma below), then consume K(t) in slot
            // t%kv_slots. HS21: prefetch disabled when kv_slots==1.
            // L1Reuse: resident mode has K fully resident -- no prefetch, no
            // slot credits, no readiness wait; the ring reads k_l1[t] directly.
            if (!kv_resident) {
              if (kv_slots > 1 && t + 1 < kv_end) {
                const int64_t kn_rows = min64(BLOCK_N, kv_seq_len - (t + 1) * BLOCK_N);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((t + 1) % kv_slots);  // slot free
                gqa_dyn::copy_gm_to_l1_dyn<T>(
                    k_l1[((t + 1) % kv_slots) * kv_slot_elems],
                    k[kv_base + (t + 1) * BLOCK_N * heads_kv * dim], BLOCK_N,
                    dim_align, heads_kv * dim, kn_rows, dim, true);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((t + 1) % kv_slots);  // slot ready
              }
              // HS21 single-slot (kv_slots==1): K(t) not on-chip (no prefetch);
              // load synchronously into slot 0 (C0 preload covers only t==0).
              if (kv_slots == 1 && t > 0) {
                const int64_t kt_rows = min64(BLOCK_N, kv_seq_len - t * BLOCK_N);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);   // slot 0 free
                gqa_dyn::copy_gm_to_l1_dyn<T>(
                    k_l1[0], k[kv_base + t * BLOCK_N * heads_kv * dim], BLOCK_N,
                    dim_align, heads_kv * dim, kt_rows, dim, true);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);    // slot 0 ready
              }
              // Consume K(t) in slot t%kv_slots: wait for its MTE2 readiness.
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(t % kv_slots);
            }
            // L1Reuse: resident reads block t of the resident region directly.
            const int64_t k_slot = kv_resident ? t : t % kv_slots;
            if (d_chunks == 1) {
              // nRatio S2 amplification (BLOCK_N=256): consume the widened K
              // tile as n_chunks 128-col L0B sub-chunks; each chunk is an
              // independent mma(init=true) + immediate FIX into its ws_s
              // column chunk (L0C stays block_m*128). Chunk-reuse handshakes:
              // M_MTE1(4) (mma(nc-1) read rhs_l0 -> L0B write(nc)) and
              // FIX_M(5) (FIX(nc-1) read acc_s_l0c -> mma(nc) write).
              // n_chunks==1 degenerates to the exact pre-nRatio sequence.
              for (int64_t nc = 0; nc < n_chunks; ++nc) {
                if (nc > 0) {
                  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(4);  // mma(nc-1) done reading rhs_l0
                }
                // K chunk rows [nc*128, nc*128+128) of the zN slot tile:
                // element offset nc*128*C0 (zN row pitch = C0 elems).
                gqa_dyn::copy_l1_to_l0b_dyn<T, true>(
                    rhs_l0[0],
                    k_l1[k_slot * kv_slot_elems +
                         nc * BLOCK_N_BASE * C0],
                    BLOCK_N, dim_align, cube_k, BLOCK_N_BASE);
                // L0 load (MTE1) -> mma (M): MTE1_M handshake (id 2).
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(2);
                if (nc > 0) {
                  AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(5);  // FIX(nc-1) done reading acc_s_l0c
                }
                gqa_dyn::mma_dyn<T, float>(lhs_l0[0], rhs_l0[0], acc_s_l0c[0],
                                           /*init=*/true, block_m,
                                           BLOCK_N_BASE, cube_k);
                if (nc + 1 < n_chunks) {
                  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(4);  // this chunk mma done reading L0
                }
                // mma (M) -> L0C->GM (FIX): M_FIX handshake (id 3); ws_s slab
                // row stride is BLOCK_N, column chunk offset nc*128.
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(3);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(3);
                gqa_dyn::copy_l0c_to_gm_dyn<float>(
                    workspace_s[ws_s_base + slot_prod * block_m * BLOCK_N +
                                nc * BLOCK_N_BASE],
                    acc_s_l0c[0], block_m, BLOCK_N_BASE, BLOCK_N, block_m,
                    BLOCK_N_BASE);
                if (nc + 1 < n_chunks) {
                  AscendC::SetFlag<AscendC::HardEvent::FIX_M>(5);  // this chunk FIX done reading acc_s_l0c
                }
              }
            } else {
            for (int64_t cc = 0; cc < d_chunks; ++cc) {
              // P3 (D256 L0C K-dim accumulate): d_chunks slice QK^T's K
              // (reduction) dim; accumulate into the SAME acc_s_l0c via mma
              // init=(cc==0), write ws_s ONCE after the last chunk. Handshake:
              // M_MTE1(4) kept (L0 reuse race), FIX_M(5) removed (intermediate
              // chunks no longer FIX-drain acc_s_l0c). See ascendc/gqa.cpp.
              // HS17 task 1: lhs_l0 (Q) is now resident (loaded once before the
              // ring), so M_MTE1(4) now protects ONLY rhs_l0 (K) reuse.
              if (cc > 0) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(4);
              }
              // HS17 task 1 (Q resident): Q is NOT reloaded here; the mma reads
              // the resident chunk lhs_l0[cc*block_m*cube_k] directly.
              gqa_dyn::copy_l1_to_l0b_dyn<T, true>(
                  rhs_l0[0], k_l1[k_slot * kv_slot_elems + cc * cube_k * BLOCK_N],
                  BLOCK_N, dim_align, cube_k, BLOCK_N);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2);
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(2);
              gqa_dyn::mma_dyn<T, float>(lhs_l0[cc * block_m * cube_k], rhs_l0[0],
                                         acc_s_l0c[0],
                                         /*init=*/(cc == 0), block_m, BLOCK_N,
                                         cube_k);
              if (cc + 1 < d_chunks) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(4);
              }
            }
            // Single write-out of the accumulated acc_s_l0c after the last chunk.
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(3);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(3);
            gqa_dyn::copy_l0c_to_gm_dyn<float>(
                workspace_s[ws_s_base + slot_prod * block_m * BLOCK_N],
                acc_s_l0c[0], block_m, BLOCK_N, BLOCK_N, block_m, BLOCK_N);
            }  // end d_chunks>1 accumulate path (d_chunks==1 uses the nc loop)
            // HS19 (position B, C1 cross-iteration, unconditional so d_chunks==1
            // is covered): this iteration's mma finished READING rhs_l0
            // (M_MTE1(3)) and FIX finished READING acc_s_l0c (FIX_M(3)); the ring
            // tail waits on both -> replaces the AIC ring-tail PipeBarrier<PIPE_ALL>.
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(3);
            // Release K slot t%kv_slots for a later prefetch.
            if (!kv_resident) {
              // Release K slot t%kv_slots for a later prefetch (resident: none).
              AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(t % kv_slots);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(slot_prod);   // SIG_S_READY: C1(t) -> V1(t)
          }
          // ============ V1: scale/mask/online softmax -> ws_p + meta =======
          if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag(slot_prod);   // SIG_S_READY(slot_prod)
            const int64_t h_start = vid * bm2;
            gqa_dyn::copy_ub_to_ub_dyn<float, float>(
                alpha_ring[slot_prod * bm2], m_i[0], bm2, 1, bm2, bm2, bm2);
            // HS19 (position A, V1 head): drain the V pipe before this MTE2 write
            // of acc_s_ub, so all PRIOR V-pipe touches of the acc_s_ub-aliased
            // region complete first (V1(t-1) reads + V2(t-3) alpha_2d Broadcast
            // write, which aliases acc_s_ub). Same-iteration V->MTE2 self-pair,
            // safe id 1 (id 0 is the copy_gm_to_ub helper's Duplicate pair).
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            // P3: ws_s holds a SINGLE accumulated slab per ring slot (C1
            // accumulated d_chunks in L0C); read chunk 0 only, the old
            // d_chunks>1 Add-sum loop is removed.
            gqa_dyn::copy_gm_to_ub_dyn<float>(
                acc_s_ub[0],
                workspace_s[ws_s_base + slot_prod * block_m * BLOCK_N +
                            h_start * BLOCK_N],
                BLOCK_N, BLOCK_N, bm2, BLOCK_N, 0.000000e+00f);
            // ws_s GM->UB (MTE2) completes -> later V compute may read acc_s_ub.
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
            AscendC::Muls(acc_s_ub[0], acc_s_ub[0], sm_scale, bm2 * bn);
            if (tail_valid != 0) {
              if (t == kv_loops - 1) {
                tl::ascend::Fill<float>(mask_col[0], MASK_NEG, BLOCK_N);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::Fill<float>(mask_col[0], 0.000000e+00f, tail_valid);
                for (int64_t h_i = 0; h_i < bm2; ++h_i) {
                  AscendC::PipeBarrier<PIPE_V>();
                  AscendC::Add(acc_s_ub[h_i * BLOCK_N], acc_s_ub[h_i * BLOCK_N],
                               mask_col[0], BLOCK_N);
                }
              }
            }
            if (causal) {
              if ((t + 1) * BLOCK_N - 1 > s0 + diff_s) {
                if (sub_per_band > 0) {
                  // P4 whole-tile vectorized mask (sub_per_band: bs>=bm2, this
                  // half's bm2 rows lie within a single band). HS24: the per-row
                  // bound uses the BAND-LOCAL row index s_local=(h_start+h_i)%bs;
                  // the old s_local=h_start+h_i (no wrap) under-masked band>=1
                  // when bg>=2 (e.g. S=48,bg=2,bs=bm2). Single-band halves give
                  // (h_start+h_i)%bs==(h_start%bs)+h_i, so start at h_start%bs.
                  AscendC::CreateVecIndex<float>(col_idx[0], (float)(t * BLOCK_N),
                                                 BLOCK_N);
                  AscendC::CreateVecIndex<float>(
                      bound_vec[0], (float)(s0 + (h_start % bs) + diff_s), bm2);
                  AscendC::PipeBarrier<PIPE_V>();
                  const uint32_t col_dst[2] = {(uint32_t)bm2, (uint32_t)bn};
                  const uint32_t col_src[2] = {1, (uint32_t)bn};
                  tl::ascend::Broadcast<float, 2, 0, false>(
                      col_2d[0], col_idx[0], tmp_ub[0], col_dst, col_src);
                  AscendC::PipeBarrier<PIPE_V>();
                  const uint32_t bnd_dst[2] = {(uint32_t)bm2, (uint32_t)bn};
                  const uint32_t bnd_src[2] = {(uint32_t)bm2, 1};
                  tl::ascend::Broadcast<float, 2, 1, false>(
                      bound_2d[0], bound_vec[0], tmp_ub[0], bnd_dst, bnd_src);
                  AscendC::PipeBarrier<PIPE_V>();
                  AscendC::Compare(causal_cmp[0], col_2d[0], bound_2d[0],
                                   AscendC::CMPMODE::LE, bm2 * bn);
                  AscendC::PipeBarrier<PIPE_V>();
                  AscendC::Select<float, uint8_t>(
                      acc_s_ub[0], causal_cmp[0], acc_s_ub[0], (float)MASK_NEG,
                      AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, bm2 * bn);
                } else {
                  // bands_per_half (bs<bm2, decode/MTP): s_local = r%bs wraps,
                  // bound non-linear; keep per-row compare+select. selMask
                  // reuses the first bn bytes of causal_cmp as a single-row mask.
                  AscendC::CreateVecIndex<float>(col_idx[0], (float)(t * BLOCK_N),
                                                 BLOCK_N);
                  for (int64_t h_i = 0; h_i < bm2; ++h_i) {
                    const int64_t s_local = (h_start + h_i) % bs;
                    const float bound = (float)(s0 + s_local + diff_s);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::CompareScalar(causal_cmp[0], col_idx[0], bound,
                                           AscendC::CMPMODE::LE, BLOCK_N);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Select<float, uint8_t>(
                        acc_s_ub[h_i * BLOCK_N], causal_cmp[0],
                        acc_s_ub[h_i * BLOCK_N], (float)MASK_NEG,
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, BLOCK_N);
                  }
                }
              }
            }
            AscendC::PipeBarrier<PIPE_V>();
            gqa_dyn::reduce_max_dyn<float>(m_i[0], acc_s_ub[0], tmp_ub[0], bm2,
                                           bn);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(m_i[0], m_i[0], alpha_ring[slot_prod * bm2], bm2);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(alpha_ring[slot_prod * bm2],
                         alpha_ring[slot_prod * bm2], m_i[0], bm2);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(alpha_ring[slot_prod * bm2], alpha_ring[slot_prod * bm2],
                         bm2);
            const uint32_t s_dst[2] = {(uint32_t)bm2, (uint32_t)bn};
            const uint32_t s_src[2] = {(uint32_t)bm2, 1};
            tl::ascend::Broadcast<float, 2, 1, false>(m_i_2d[0], m_i[0],
                                                      tmp_ub[0], s_dst, s_src);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(acc_s_ub[0], acc_s_ub[0], m_i_2d[0], bm2 * bn);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(acc_s_ub[0], acc_s_ub[0], bm2 * bn);
            AscendC::PipeBarrier<PIPE_V>();
            gqa_dyn::reduce_sum_dyn<float>(sumexp_i_ring[slot_prod * bm2],
                                           acc_s_ub[0], tmp_ub[0], bm2, bn);
            // HS19 (position A, V1 cast): drain the MTE3 pipe before this V-write
            // of acc_s_half, so the PREVIOUS iteration's ws_p copy_ub_to_gm MTE3
            // read of acc_s_half completes first. Same-iteration MTE3->V
            // self-pair, safe id 2 (id 0 helper, id 7 task-tail C).
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(2);
            gqa_dyn::copy_ub_to_ub_dyn<T, float>(acc_s_half[0], acc_s_ub[0],
                                                 bm2 * bn, bm2, bn, bn, bn);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
            gqa_dyn::copy_ub_to_gm_dyn<T>(
                workspace_p[ws_p_base + slot_prod * block_m * BLOCK_N +
                            h_start * BLOCK_N],
                acc_s_half[0], BLOCK_N, BLOCK_N, bm2, BLOCK_N);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(3 + slot_prod);  // SIG_P_READY: V1(t) -> C2(t)
          }
        }
        // HS19 directional-sync (position A, V1/V2 boundary): the AIV-only
        // PipeBarrier<PIPE_ALL> (ISSUE-HS4) is REMOVED here, replaced by the TWO
        // directed single-pipe-pair drains at the CONSUMER side inside V1 (the
        // V1-head V_MTE2(1) and V1-cast MTE3_V(2) self-pairs above).
        //   hazard 1: acc_s_half MTE3-read(t) -> V-write(t+1): MTE3_V(2).
        //   hazard 2: acc_s_ub V-read(t) -> MTE2-write(t+1): V_MTE2(1).
        //   hazard 3: alpha_2d(=acc_s_ub alias) V-write(t-2) -> MTE2-write(t+1):
        //             also V_MTE2(1) (the V-pipe drain at V1's head flushes ALL
        //             prior V-pipe touches of the aliased region, incl. V2's).
        // The AIC is never drained here (P2 prefetch<->mma overlap preserved).
        if (PRELAUNCH <= t) {
          const int64_t now_k = t - PRELAUNCH;
          if (now_k < kv_end) {
            const int64_t slot_cons = now_k % RING;
            // ================= C2: O = P @ V -> workspace_o ================
            if ASCEND_IS_AIC {
              AscendC::CrossCoreWaitFlag(3 + slot_cons);   // SIG_P_READY(slot_cons)
              // ws_p -> acc_s_l1 (GM->L1, MTE2); mark completion (id 5) for the
              // L1->L0A (MTE1) below.
              gqa_dyn::copy_gm_to_l1_dyn<T>(
                  acc_s_l1[0],
                  workspace_p[ws_p_base + slot_cons * block_m * BLOCK_N],
                  block_m, BLOCK_N, BLOCK_N, block_m, BLOCK_N, true);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(5);
              // P2 double-slot overlap: prefetch V(now_k+1) into slot
              // (now_k+1)%kv_slots (MTE2 overlaps the MTE1/mma below), then
              // consume slot now_k%kv_slots. HS21: prefetch disabled when
              // kv_slots==1.
              // L1Reuse: resident mode has V fully resident -- prefetch,
              // single-slot sync load and the readiness wait are all removed.
              if (!kv_resident) {
                if (kv_slots > 1 && now_k + 1 < kv_end) {
                  const int64_t vn_rows = min64(BLOCK_N, kv_seq_len - (now_k + 1) * BLOCK_N);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2 + (now_k + 1) % kv_slots);  // V slot free
                  gqa_dyn::copy_gm_to_l1_dyn<T>(
                      v_l1[((now_k + 1) % kv_slots) * kv_slot_elems],
                      v[kv_base + (now_k + 1) * BLOCK_N * heads_kv * dim], BLOCK_N,
                      dim_align, heads_kv * dim, vn_rows, dim, true);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(2 + (now_k + 1) % kv_slots);  // V slot ready
                }
                // HS21 single-slot (kv_slots==1): V(now_k) not on-chip; load
                // synchronously into V slot 0 (id 2). C0 preload covers now_k==0.
                if (kv_slots == 1 && now_k > 0) {
                  const int64_t vv_rows = min64(BLOCK_N, kv_seq_len - now_k * BLOCK_N);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);   // V slot 0 free
                  gqa_dyn::copy_gm_to_l1_dyn<T>(
                      v_l1[0], v[kv_base + now_k * BLOCK_N * heads_kv * dim], BLOCK_N,
                      dim_align, heads_kv * dim, vv_rows, dim, true);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(2);    // V slot 0 ready
                }
              }
              // L1Reuse: resident reads block now_k of the resident region.
              const int64_t v_slot = kv_resident ? now_k : now_k % kv_slots;
              // Wait for acc_s_l1 (id 5) MTE2 readiness (V slot wait when streaming).
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(5);
              if (!kv_resident) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(2 + now_k % kv_slots);  // V slot ready
              }
              if (d_chunks == 1) {
                // nRatio S2 amplification (BLOCK_N=256): the mma K-dim (=kv
                // axis) is consumed as n_chunks 128-wide sub-chunks,
                // ACCUMULATED into acc_o_l0c via init=(nc==0), single FIX
                // after the last chunk. M_MTE1(6) guards v_l0 chunk reuse.
                // p_l0 is loaded as n_chunks 128-wide column chunks (NOT one
                // [block_m, 256] copy): the 256-wide L1->L0A copy scrambles
                // the SECOND 16-row fractal block (HS28, identity-V probe).
                // n_chunks==1 degenerates to the exact pre-nRatio sequence.
                for (int64_t nc = 0; nc < n_chunks; ++nc) {
                  gqa_dyn::copy_l1_to_l0a_dyn<T, false>(
                      p_l0[nc * BLOCK_N_BASE * block_m],
                      acc_s_l1[nc * BLOCK_N_BASE * block_m],
                      block_m, BLOCK_N, block_m, BLOCK_N_BASE);
                }
                for (int64_t nc = 0; nc < n_chunks; ++nc) {
                  if (nc > 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(6);  // mma(nc-1) done reading v_l0
                  }
                  // V chunk rows [nc*128, nc*128+128) of the zN slot tile.
                  gqa_dyn::copy_l1_to_l0b_dyn<T, false>(
                      v_l0[0],
                      v_l1[v_slot * kv_slot_elems +
                           nc * BLOCK_N_BASE * C0],
                      BLOCK_N, dim_align, BLOCK_N_BASE, cube_k);
                  // L0 load (MTE1) -> mma (M): MTE1_M handshake (id 7).
                  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(7);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(7);
                  // A-operand P column chunk: zZ column offset nc*128 ->
                  // nc*128*block_m elements.
                  gqa_dyn::mma_dyn<T, float>(p_l0[nc * BLOCK_N_BASE * block_m],
                                             v_l0[0], acc_o_l0c[0],
                                             /*init=*/(nc == 0), block_m,
                                             cube_k, BLOCK_N_BASE);
                  if (nc + 1 < n_chunks) {
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(6);  // this chunk mma done reading L0
                  }
                }
                // mma (M) -> L0C->GM (FIX): M_FIX handshake (id 0).
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
                gqa_dyn::copy_l0c_to_gm_dyn<float>(
                    workspace_o[ws_o_base + slot_cons * block_m * dim_align],
                    acc_o_l0c[0], block_m, cube_k, dim_align, block_m, cube_k);
              } else {
              for (int64_t cc = 0; cc < d_chunks; ++cc) {
                // d_chunks>1 cross-chunk reverse handshake (C2: M_MTE1(6)/FIX_M(1));
                // only cc>0 waits / cc+1<d_chunks sets.
                if (cc > 0) {
                  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(6);
                }
                gqa_dyn::copy_l1_to_l0a_dyn<T, false>(p_l0[0], acc_s_l1[0],
                                                      block_m, BLOCK_N, block_m,
                                                      BLOCK_N);
                gqa_dyn::copy_l1_to_l0b_dyn<T, false>(
                    v_l0[0], v_l1[v_slot * kv_slot_elems + cc * cube_k * BLOCK_N],
                    BLOCK_N, dim_align, BLOCK_N, cube_k);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(7);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(7);
                if (cc > 0) {
                  AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
                }
                gqa_dyn::mma_dyn<T, float>(p_l0[0], v_l0[0], acc_o_l0c[0], true,
                                           block_m, cube_k, BLOCK_N);
                if (cc + 1 < d_chunks) {
                  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(6);
                }
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
                gqa_dyn::copy_l0c_to_gm_dyn<float>(
                    workspace_o[ws_o_base + slot_cons * block_m * dim_align +
                                cc * cube_k],
                    acc_o_l0c[0], block_m, cube_k, dim_align, block_m, cube_k);
                if (cc + 1 < d_chunks) {
                  AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
                }
              }
              }  // end d_chunks>1 C2 path (d_chunks==1 uses the nc loop)
              // HS19 (position B, C2 cross-iteration, unconditional): this
              // iteration's mma finished READING v_l0 (M_MTE1(5)), FIX finished
              // READING acc_o_l0c (FIX_M(4)), and the L1->L0A MTE1 reads of the
              // single-slot acc_s_l1 staging buffer are done (MTE1_MTE2(4)). The
              // ring tail waits on all three -> replaces the AIC ring-tail
              // PipeBarrier<PIPE_ALL>. (acc_s_l1's MTE2_MTE1(5) only covers the
              // intra-iteration MTE2->MTE1 direction, not the cross-iteration
              // MTE1-read(t)->MTE2-write(t+1) WAR; d256/d_chunks==2 hits it hardest.)
              AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(5);
              AscendC::SetFlag<AscendC::HardEvent::FIX_M>(4);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(4);
              if (!kv_resident) {
                // Release V slot now_k%kv_slots for a later prefetch.
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2 + now_k % kv_slots);
              }
              AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(6 + slot_cons);  // SIG_O_READY: C2 -> V2
            }
            // ============ V2: alpha-rescale merge into running state =======
            if ASCEND_IS_AIV {
              AscendC::CrossCoreWaitFlag(6 + slot_cons);   // SIG_O_READY(slot_cons)
              const int64_t h_start = vid * bm2;
              // HS19 (position A, acc_o_ub): drain the V pipe (V -> MTE2) before
              // this MTE2 write of acc_o_ub, so the prior iteration's V2 Add
              // V-read of acc_o_ub completes first (the simulator-flagged
              // add/v -> copy_gm_to_ub/mte2 hazard on acc_o_ub). Same-iteration
              // set->wait self-pair, safe id 2.
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
              gqa_dyn::copy_gm_to_ub_dyn<float>(
                  acc_o_ub[0],
                  workspace_o[ws_o_base + slot_cons * block_m * dim_align +
                              h_start * dim_align],
                  dim_align, dim_align, bm2, dim_align, 0.000000e+00f);
              const uint32_t o_dst[2] = {(uint32_t)bm2, (uint32_t)dim_align};
              const uint32_t o_src[2] = {(uint32_t)bm2, 1};
              tl::ascend::Broadcast<float, 2, 1, false>(
                  alpha_2d[0], alpha_ring[slot_cons * bm2], tmp_ub[0], o_dst,
                  o_src);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Mul(acc_o[0], acc_o[0], alpha_2d[0], bm2 * dim_align);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Add(acc_o[0], acc_o[0], acc_o_ub[0], bm2 * dim_align);
              AscendC::Mul(sumexp[0], sumexp[0], alpha_ring[slot_cons * bm2],
                           bm2);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Add(sumexp[0], sumexp[0], sumexp_i_ring[slot_cons * bm2],
                           bm2);
            }
          }
        }
        // HS19 directional (ring tail): the AIC-side PipeBarrier<PIPE_ALL> is
        // decomposed into FIVE directed cross-iteration consumer-drain waits (the
        // load-bearing hazards, HS19-1): cross-iteration L0B/L0C/acc_s_l1 reuse.
        // Each wait pairs a set issued THIS iteration after the corresponding
        // consumer (C1 mma-read rhs_l0 -> M_MTE1(3); C1 FIX-read acc_s_l0c ->
        // FIX_M(3); C2 mma-read v_l0 -> M_MTE1(5); C2 FIX-read acc_o_l0c ->
        // FIX_M(4); C2 L1->L0A MTE1-read acc_s_l1 -> MTE1_MTE2(4)). Draining ONLY
        // these consumers replaces the full-pipe drain: the MTE2 K(t+1)/V(t+1)
        // prefetch and the MTE1 L1->L0A path are NOT stalled -> P2 overlap kept.
        // Waits are guarded by the SAME condition as their sets (level semantics:
        // a wait on a never-set flag would consume a stale level). The AIV needs
        // NO ring-tail wait (V-pipe program order + per-stage MTE2_V + position-A
        // handshakes cover its cross-iteration WAR).
        if ASCEND_IS_AIC {
          if (t < kv_end) {  // C1(t) ran this iteration -> drain its consumers
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);  // mma read rhs_l0 done
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(3);   // FIX read acc_s_l0c done
          }
          if (PRELAUNCH <= t && (t - PRELAUNCH) < kv_end) {  // C2(t-2) ran
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(5);  // mma read v_l0 done
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(4);   // FIX read acc_o_l0c done
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(4);  // MTE1 read acc_s_l1 done
          }
        }
      }
      // ---- epilogue: normalize + cast + per-g write out --------------------
      if ASCEND_IS_AIV {
        // HS23 (F217 all-masked-row protection): golden zeroes rows whose softmax
        // denominator comes from an all-masked row. Our kernel masks with the
        // FINITE MASK_NEG; an all-masked row would otherwise yield mean(V) != 0.
        // A row is all-masked iff m_i was never updated off the MASK_NEG init.
        // Zero those rows before the Div. (Only reachable for S>S_kv causal
        // violations; a no-op for valid inputs where m_i != MASK_NEG.)
        if (causal) {
          for (int64_t row = 0; row < bm2; ++row) {
            if (m_i.GetValue(row) == MASK_NEG) {   // all-masked row
              AscendC::Duplicate<float>(acc_o[row * dim_align], 0.0f, dim_align);
            }
          }
          AscendC::PipeBarrier<PIPE_V>();
        }
        const uint32_t o_dst[2] = {(uint32_t)bm2, (uint32_t)dim_align};
        const uint32_t o_src[2] = {(uint32_t)bm2, 1};
        tl::ascend::Broadcast<float, 2, 1, false>(alpha_2d[0], sumexp[0],
                                                  tmp_ub[0], o_dst, o_src);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(acc_o[0], acc_o[0], alpha_2d[0], bm2 * dim_align);
        AscendC::PipeBarrier<PIPE_V>();
        gqa_dyn::copy_ub_to_ub_dyn<T, float>(acc_o_half[0], acc_o[0],
                                             bm2 * dim_align, bm2, dim_align,
                                             dim_align, dim_align);
        if (sub_per_band > 0) {
          const int64_t g_local = (vid * bm2) / bs;
          const int64_t s_local0 = (vid * bm2) % bs;
          if (g_blk * bg + g_local < group) {
            const int64_t out_raw = q_seq_len - s0 - s_local0;
            const int64_t out_rows =
                out_raw >= bm2 ? bm2 : (out_raw > 0 ? out_raw : 0);
            if (out_rows > 0) {
              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
              gqa_dyn::copy_ub_to_gm_dyn<T>(
                  output[q_base + g_local * dim + s_local0 * heads_q * dim],
                  acc_o_half[0], dim_align, heads_q * dim, out_rows, dim);
            }
          }
        } else if (bands_per_half > 0) {
          // bm2 % bs == 0 (bm2 >= bs): subcore vid holds bands_per_half whole
          // bands, one copy each (fixed s0 lower bound, no vid-divide guard).
          // (P8 decode per-q-head segment merge was MEASURED and ROLLED BACK:
          // it slowed the decode cases ~1-2%; the epilogue write is a tiny
          // fraction of decode kernel time. See ascendc/gqa.cpp ISSUE-HS8.)
          const int64_t out_raw = q_seq_len - s0;
          const int64_t out_rows =
              out_raw >= bs ? bs : (out_raw > 0 ? out_raw : 0);
          for (int64_t i_b = 0; i_b < bands_per_half; ++i_b) {
            const int64_t band = vid * bands_per_half + i_b;
            if (band < bg) {
              if (g_blk * bg + band < group) {
                if (out_rows > 0) {
                  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
                  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
                  gqa_dyn::copy_ub_to_gm_dyn<T>(
                      output[q_base + band * dim],
                      acc_o_half[i_b * bs * dim_align], dim_align,
                      heads_q * dim, out_rows, dim);
                }
              }
            }
          }
        } else {
          // HS20 generic fallback: bs and bm2 are NOT integer multiples of each
          // other (bs in (64,128) non-16-aligned, bg==1 -> block_m=AlignUp(bs,16),
          // bm2=block_m/2, so bm2<bs and bs%bm2!=0). Neither static per-band form
          // applies; without this branch acc_o_half is never written -> output
          // rows left as garbage (gqa_81 S=127 all-wrong signature). Write
          // row-by-row: row r in [vid*bm2, vid*bm2+bm2) maps to band=r/bs,
          // s_local=r%bs (head-major, same as the Q L1 layout). One single-row
          // copy per valid row.
          for (int64_t r = 0; r < bm2; ++r) {
            const int64_t row = vid * bm2 + r;   // global row within block_m
            const int64_t band = row / bs;
            const int64_t s_local = row % bs;
            if (band < bg && (g_blk * bg + band) < group &&
                (s0 + s_local) < q_seq_len) {
              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
              gqa_dyn::copy_ub_to_gm_dyn<T>(
                  output[q_base + band * dim + s_local * heads_q * dim],
                  acc_o_half[r * dim_align], dim_align, heads_q * dim, 1, dim);
            }
          }
        }
      }
      // ---- task tail: cross-task backstop + C-side slot-credit drain --------
      // HS19 directional (task tail, de-PIPE_ALL; mirrors ascendc/gqa.cpp's
      // verified form): the load-bearing hazard is AIV-side -- acc_o_half's
      // cross-task WAR (this task's epilogue copy_ub_to_gm MTE3-read of
      // acc_o_half vs the NEXT task's epilogue cast V-write, ISSUE-5). A single
      // MTE3_V(7) self-pair on the AIV drains this task's output MTE3 read before
      // the next task's cast V-write. The AIC has no acc_o_half hazard and
      // already drained at the ring tail, so it skips. Then drain the C-side K/V
      // slot-free credits (MTE1_MTE2 0/1/2/3) so no level residue leaks across
      // tasks.
      if ASCEND_IS_AIV {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
      }
      if ASCEND_IS_AIC {
        // L1Reuse: resident mode presets/consumes no credits -- skip the drain.
        if (!kv_resident) {
          AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(3);
        }
      }
    }
  }
}

#endif  // GQA_GEN_KERNEL_H
