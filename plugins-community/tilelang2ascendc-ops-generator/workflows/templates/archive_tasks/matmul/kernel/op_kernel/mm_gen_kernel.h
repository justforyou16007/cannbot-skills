// ----------------------------------------------------------------------------------------------------------
// Matmul (generalized) - 编排：tiling 读取、buffer 分配、task/swizzle 主循环、
// K 维双 buffer 流水（L1 S1=2 预取 + L0 S2=2 乒乓）。
//
// 与 design/tile_level/matmul.py 的 tilelang 编译产物同源（fp16/fp32 两份 dump
// 逐行对照；同步结构/flag 编号/槽位偏移逐字节保留）。泛化：M/N/K 运行期
// （tiling 传入），block 尺寸 per-dtype 编译期常量（mm_common.h MmCfg）。
//
// 偏移公式（元素，两个 dtype 的生成码均已验证）：
//   A_L1 槽距 = BLOCK_M*K_L1；B_L1 槽距 = K_L1*BLOCK_N
//   A_L0 槽距 = BLOCK_M*BLOCK_K；B_L0 槽距 = BLOCK_K*BLOCK_N
//   A_L1 内 kk 列片偏移 = kk*BLOCK_K*BLOCK_M（zN 列步长 = roundup16(BLOCK_M)）
//   B_L1 内 kk 行片偏移 = kk*BLOCK_K*C0（zN K 行步长 = C0=32B/sizeof(T)，实测值）
// ----------------------------------------------------------------------------------------------------------
#ifndef MM_GEN_KERNEL_H
#define MM_GEN_KERNEL_H

#include "mm_common.h"

namespace mm_gen {

template <typename T>
CATLASS_DEVICE void mm_fwd_run(GM_ADDR a_handle, GM_ADDR b_handle,
                               GM_ADDR c_handle, GM_ADDR tiling_handle) {
  using Cfg = MmCfg<T>;
  constexpr int64_t BLOCK_N = Cfg::BLOCK_N;
  constexpr int64_t BLOCK_K = Cfg::BLOCK_K;
  constexpr int64_t K_L1 = Cfg::K_L1;
  constexpr int64_t LOOP_KK = K_L1 / BLOCK_K;
  constexpr uint32_t C0 = 32 / sizeof(T);

  constexpr uint32_t A_L1_SLOT = BLOCK_M * K_L1;        // 元素
  constexpr uint32_t B_L1_SLOT = K_L1 * BLOCK_N;
  constexpr uint32_t A_L0_SLOT = BLOCK_M * BLOCK_K;
  constexpr uint32_t B_L0_SLOT = BLOCK_K * BLOCK_N;
  constexpr uint32_t A_KK_OFF = BLOCK_K * BLOCK_M;
  constexpr uint32_t B_KK_OFF = BLOCK_K * C0;

  __gm__ MatmulGenTiling* tiling_gm =
      reinterpret_cast<__gm__ MatmulGenTiling*>(tiling_handle);
  const int64_t M = tiling_gm->m;
  const int64_t N = tiling_gm->n;
  const int64_t K = tiling_gm->k;

  const int64_t m_num = (M + BLOCK_M - 1) / BLOCK_M;
  const int64_t n_num = (N + BLOCK_N - 1) / BLOCK_N;
  const int64_t block_num = m_num * n_num;
  const int64_t tasks_per_core = (block_num + CORE_NUM - 1) / CORE_NUM;
  const int64_t loop_k = (K + K_L1 - 1) / K_L1;

  AscendC::TPipe pipe;

  AscendC::GlobalTensor<T> A;
  A.SetGlobalBuffer((__gm__ T*)a_handle);
  AscendC::GlobalTensor<T> B;
  B.SetGlobalBuffer((__gm__ T*)b_handle);
  AscendC::GlobalTensor<T> C;
  C.SetGlobalBuffer((__gm__ T*)c_handle);

  // 片上 buffer：尺寸全部是编译期常量，与生成码一致
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1;
  pipe.InitBuffer(ascend_l1, (A_L1_SLOT + B_L1_SLOT) * S1 * sizeof(T));
  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c;
  pipe.InitBuffer(ascend_l0c, 131072);
  pipe.Destroy();

  auto A_L1 = ascend_l1.GetWithOffset<T>(S1 * A_L1_SLOT, 0);
  auto B_L1 = ascend_l1.GetWithOffset<T>(S1 * B_L1_SLOT,
                                         S1 * A_L1_SLOT * sizeof(T));
  auto A_L0 = ascend_l0a.GetWithOffset<T>(S2 * A_L0_SLOT, 0);
  auto B_L0 = ascend_l0b.GetWithOffset<T>(S2 * B_L0_SLOT, 0);
  auto C_L0 = ascend_l0c.GetWithOffset<float>(BLOCK_M * BLOCK_N, 0);

  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }

  if ASCEND_IS_AIC {
    // init_flag
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);

    for (int64_t i = 0; i < tasks_per_core; ++i) {
      const int64_t task = swizzle_zn3(i * CORE_NUM + cid, m_num, n_num);
      if (task < block_num) {
        const int64_t bx = task / n_num;
        const int64_t by = task % n_num;
        const int64_t tail_m = min64(BLOCK_M, M - bx * BLOCK_M);
        const int64_t tail_n = min64(BLOCK_N, N - by * BLOCK_N);
        const int64_t a_base = bx * BLOCK_M * K;
        const int64_t b_base = by * BLOCK_N;
        const int64_t c_base = bx * BLOCK_M * N + by * BLOCK_N;

        // 首个 K chunk
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        mm_dyn::copy_gm_to_l1_dyn<T>(A_L1[0], A[a_base], BLOCK_M, K_L1, K,
                                     tail_m, min64(K_L1, K), true);
        mm_dyn::copy_gm_to_l1_dyn<T>(B_L1[0], B[b_base], K_L1, BLOCK_N, N,
                                     min64(K_L1, K), tail_n, true);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);

        for (int64_t k = 0; k < loop_k; ++k) {
          // 预取下一个 K chunk 到空闲 L1 槽
          if (k < loop_k - 1) {
            const int64_t slot_next = (k + 1) % S1;
            const int64_t k_next = min64(K_L1, K - (k + 1) * K_L1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot_next);
            mm_dyn::copy_gm_to_l1_dyn<T>(A_L1[slot_next * A_L1_SLOT],
                                         A[a_base + (k + 1) * K_L1], BLOCK_M,
                                         K_L1, K, tail_m, k_next, true);
            mm_dyn::copy_gm_to_l1_dyn<T>(B_L1[slot_next * B_L1_SLOT],
                                         B[(k + 1) * K_L1 * N + b_base], K_L1,
                                         BLOCK_N, N, k_next, tail_n, true);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(slot_next);
          }
          // L0 乒乓 + mma
          const int64_t slot_cur = k % S1;
          for (int64_t kk = 0; kk < LOOP_KK; ++kk) {
            const int64_t slot_l0 = kk % S2;
            if (kk == 0) {
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(slot_cur);
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(slot_l0);
            mm_dyn::copy_l1_to_l0a_dyn<T, false>(
                A_L0[slot_l0 * A_L0_SLOT],
                A_L1[slot_cur * A_L1_SLOT + kk * A_KK_OFF], BLOCK_M, K_L1,
                BLOCK_M, BLOCK_K);
            mm_dyn::copy_l1_to_l0b_dyn<T, false>(
                B_L0[slot_l0 * B_L0_SLOT],
                B_L1[slot_cur * B_L1_SLOT + kk * B_KK_OFF], K_L1, BLOCK_N,
                BLOCK_K, BLOCK_N);
            if (kk == LOOP_KK - 1) {
              AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot_cur);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(slot_l0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(slot_l0);
            tl::ascend::mma<T, float, BLOCK_M, BLOCK_N>(
                A_L0[slot_l0 * A_L0_SLOT], B_L0[slot_l0 * B_L0_SLOT], C_L0[0],
                (k == 0) && (kk == 0), BLOCK_K);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(slot_l0);
          }
        }

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        mm_dyn::copy_l0c_to_gm_dyn<T>(C[c_base], C_L0[0], BLOCK_M, BLOCK_N, N,
                                      tail_m, tail_n);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
      }
    }

    // clear_flag
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
  }
}

}  // namespace mm_gen

#endif  // MM_GEN_KERNEL_H
