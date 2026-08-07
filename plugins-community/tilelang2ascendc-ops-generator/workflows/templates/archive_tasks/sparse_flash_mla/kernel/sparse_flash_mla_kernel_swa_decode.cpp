#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR q_handle,  GM_ADDR kv_all_handle,  GM_ADDR sinks_handle,  GM_ADDR idx_handle,  GM_ADDR output_handle,  GM_ADDR workspace_1_handle,  GM_ADDR workspace_2_handle,  GM_ADDR workspace_3_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<bfloat16_t> q;
  q.SetGlobalBuffer((__gm__ bfloat16_t*)q_handle);
  AscendC::GlobalTensor<bfloat16_t> kv_all;
  kv_all.SetGlobalBuffer((__gm__ bfloat16_t*)kv_all_handle);
  AscendC::GlobalTensor<float> sinks;
  sinks.SetGlobalBuffer((__gm__ float*)sinks_handle);
  AscendC::GlobalTensor<float> idx;
  idx.SetGlobalBuffer((__gm__ float*)idx_handle);
  AscendC::GlobalTensor<bfloat16_t> output;
  output.SetGlobalBuffer((__gm__ bfloat16_t*)output_handle);
  AscendC::GlobalTensor<float> workspace_1;
  workspace_1.SetGlobalBuffer((__gm__ float*)workspace_1_handle);
  AscendC::GlobalTensor<bfloat16_t> workspace_2;
  workspace_2.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_2_handle);
  AscendC::GlobalTensor<float> workspace_3;
  workspace_3.SetGlobalBuffer((__gm__ float*)workspace_3_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto q_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 0);
  auto kv_l1 = ascend_l1.GetWithOffset<bfloat16_t>(65536, 32768);
  auto scores_l0c = ascend_l0c.GetWithOffset<float>(4096, 0);
  auto p_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 163840);
  auto acc_o_l0c = ascend_l0c.GetWithOffset<float>(16384, 16384);
  auto idx_ub = ascend_ub.GetWithOffset<float>(128, 0);
  auto m_i = ascend_ub.GetWithOffset<float>(16, 512);
  auto sumexp = ascend_ub.GetWithOffset<float>(16, 576);
  auto acc_o_ub = ascend_ub.GetWithOffset<float>(8192, 640);
  auto scores_ub = ascend_ub.GetWithOffset<float>(2048, 33408);
  auto mask_lo = ascend_ub.GetWithOffset<float>(128, 41600);
  auto mask_hi = ascend_ub.GetWithOffset<float>(128, 42112);
  auto mask_col = ascend_ub.GetWithOffset<float>(128, 42624);
  auto m_old = ascend_ub.GetWithOffset<float>(16, 43136);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(8192, 43200);
  auto rowsum = ascend_ub.GetWithOffset<float>(16, 51392);
  auto p_ub = ascend_ub.GetWithOffset<bfloat16_t>(2048, 51456);
  auto acc_o_new = ascend_ub.GetWithOffset<float>(8192, 55552);
  auto acc_o_half = ascend_ub.GetWithOffset<bfloat16_t>(8192, 88320);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t h = 0; h < 4; ++h) {
      AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(3);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(3);
      tl::ascend::copy_gm_to_l1<bfloat16_t, 32, 512>(q_l1[0], q[(h * 16384)], 512, 32, 512, true);
      for (int32_t t = 0; t < 16; ++t) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(5);
        tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 512>(kv_l1[0], kv_all[(t * 65536)], 512, 128, 512, true);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(6);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(6);
        tl::ascend::gemm_v0<bfloat16_t, float, 32, 128, 512, false, true, 128>(q_l1[0], kv_l1[0], scores_l0c[0], ascend_l0a, ascend_l0b, (bool)1, 128);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(2);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(2);
        AscendC::PipeBarrier<PIPE_FIX>();
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 32, 128, 0>(workspace_1[0], scores_l0c[0], 128, 0, 0);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(0);
        AscendC::CrossCoreWaitFlag(1);
        tl::ascend::copy_gm_to_l1<bfloat16_t, 32, 128>(p_l1[0], workspace_2[0], 128, 0, 0, true);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(3);
        tl::ascend::gemm_v0<bfloat16_t, float, 32, 512, 128, false, false, 128>(p_l1[0], kv_l1[0], acc_o_l0c[0], ascend_l0a, ascend_l0b, (bool)1, 512);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(4);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(4);
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 32, 512, 0>(workspace_3[0], acc_o_l0c[0], 512, 0, 0);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(2);
      }
    }
  }
  if ASCEND_IS_AIV {
    tl::ascend::copy_gm_to_ub<float, 128>(idx_ub[0], idx[0], 128, 1, 128, 0.000000e+00f);
    for (int32_t h_1 = 0; h_1 < 4; ++h_1) {
      tl::ascend::copy_gm_to_ub<float, 16>(m_i[0], sinks[((h_1 * 32) + (vid * 16))], 128, 1, 16, 0.000000e+00f);
      tl::ascend::Fill<float>(sumexp[0], 1.000000e+00f, 16);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::Fill<float>(acc_o_ub[0], 0.000000e+00f, 8192);
      for (int32_t t_1 = 0; t_1 < 16; ++t_1) {
        AscendC::CrossCoreWaitFlag(0);
        tl::ascend::copy_gm_to_ub<float, 128, 16>(scores_ub[0], workspace_1[(vid * 2048)], 128, 16, 128, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);
        {
        AscendC::Muls(scores_ub[0], scores_ub[0], 4.419417e-02f, 2048);
        }
        float lo2f = ((float)(1920 - ((t_1 > 14 ? t_1 : 14) * 128)));
        {
        AscendC::Adds(mask_lo[0], idx_ub[0], (1.000000e+00f - lo2f), 128);
        }
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Maxs(mask_lo[0], mask_lo[0], 0.000000e+00f, 128);
        }
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Mins(mask_lo[0], mask_lo[0], 1.000000e+00f, 128);
        }
        {
        AscendC::Muls(mask_hi[0], idx_ub[0], -1.000000e+00f, 128);
        }
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Adds(mask_hi[0], mask_hi[0], 1.280000e+02f, 128);
        }
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Maxs(mask_hi[0], mask_hi[0], 0.000000e+00f, 128);
        }
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Mins(mask_hi[0], mask_hi[0], 1.000000e+00f, 128);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(mask_lo[0], mask_lo[0], mask_hi[0], 128);
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Muls(mask_hi[0], mask_lo[0], -1.000000e+00f, 128);
        }
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Adds(mask_hi[0], mask_hi[0], 1.000000e+00f, 128);
        }
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Muls(mask_col[0], mask_hi[0], -1.073742e+09f, 128);
        }
        for (int32_t hi = 0; hi < 16; ++hi) {
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Mul(scores_ub[(hi * 128)], scores_ub[(hi * 128)], mask_lo[0], 128);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(scores_ub[(hi * 128)], scores_ub[(hi * 128)], mask_col[0], 128);
        }
        tl::ascend::copy_ub_to_ub<float, float, 16>(m_old[0], m_i[0], 1, 16, 16, 1, 16, 16);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::reduce_max<float, 16, 128, -1>(m_i[0], scores_ub[0], tmp_ub[0], true);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(m_i[0], m_i[0], m_old[0], 16);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(m_old[0], m_old[0], m_i[0], 16);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(m_old[0], m_old[0], 16);
        for (int32_t hi_1 = 0; hi_1 < 16; ++hi_1) {
          AscendC::PipeBarrier<PIPE_V>();
          {
          AscendC::PipeBarrier<PIPE_ALL>();
          auto m_i_scalar = -(float)m_i.GetValue(hi_1);
          AscendC::Adds(scores_ub[(hi_1 * 128)], scores_ub[(hi_1 * 128)], m_i_scalar, 128);
          }
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(scores_ub[0], scores_ub[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::reduce_sum<float,  16,  128,  -1>(rowsum[0], scores_ub[0], tmp_ub[0], true);
        AscendC::Mul(sumexp[0], sumexp[0], m_old[0], 16);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(sumexp[0], sumexp[0], rowsum[0], 16);
        tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(p_ub[0], scores_ub[0], 16, 128, 128, 16, 128, 128);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 16>(workspace_2[(vid * 2048)], p_ub[0], 128, 16, 128);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(1);
        AscendC::CrossCoreWaitFlag(2);
        for (int32_t hi_2 = 0; hi_2 < 16; ++hi_2) {
          AscendC::PipeBarrier<PIPE_V>();
          {
          AscendC::PipeBarrier<PIPE_ALL>();
          auto m_old_scalar = m_old.GetValue(hi_2);
          AscendC::Muls(acc_o_ub[(hi_2 * 512)], acc_o_ub[(hi_2 * 512)], m_old_scalar, 512);
          }
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
          tl::ascend::copy_gm_to_ub<float, 512, 16>(acc_o_new[0], workspace_3[(vid * 8192)], 512, 16, 512, 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(acc_o_ub[(hi_2 * 512)], acc_o_ub[(hi_2 * 512)], acc_o_new[(hi_2 * 512)], 512);
        }
      }
      for (int32_t hi_3 = 0; hi_3 < 16; ++hi_3) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::PipeBarrier<PIPE_ALL>();
        auto sumexp_scalar = 1.0f / (float)sumexp.GetValue(hi_3);
        AscendC::Muls(acc_o_ub[(hi_3 * 512)], acc_o_ub[(hi_3 * 512)], sumexp_scalar, 512);
      }
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<bfloat16_t, float, 8192>(acc_o_half[0], acc_o_ub[0], 16, 512, 512, 16, 512, 512);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 512, 16>(output[((h_1 * 16384) + (vid * 8192))], acc_o_half[0], 512, 16, 512);
    }
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* q_handle, uint8_t* kv_all_handle, uint8_t* sinks_handle, uint8_t* idx_handle, uint8_t* output_handle, uint8_t* workspace_1_handle, uint8_t* workspace_2_handle, uint8_t* workspace_3_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<1, nullptr, stream>>>(q_handle, kv_all_handle, sinks_handle, idx_handle, output_handle, workspace_1_handle, workspace_2_handle, workspace_3_handle, fftsAddr);
}
