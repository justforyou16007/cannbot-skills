// ----------------------------------------------------------------------------------------------------------
// FlashAttention forward (generalized) - 公共定义：常量 / tiling / 上下文 / dyn copy 包装
//
// 本文件是 kernel 多文件拆分的公共底座（对齐 kernel_reference 的 kernel_common.h 角色）：
//   kernel_common.h        —— 本文件：常量、tiling、FaGenContext、fa_dyn 运行期宽度 copy 包装
//   flash_attention_cube.h —— Cube 侧：C0(Q load) / C1(S=Q@K^T) / C2(O=P@V)
//   flash_attention_vec.h  —— Vector 侧：V0(init) / V1(online softmax) / V2(merge) / epilogue
//   fa_gen_kernel.h        —— fa_fwd_run 编排（buffer 分配 + task/ring 主循环）
//
// 同步结构（SetFlag/WaitFlag/PipeBarrier/CrossCoreFlag）与已验证的单文件版逐字节一致，
// 禁止"顺手优化"。任何修改必须官方脚本全量回归（50 case）。
// ----------------------------------------------------------------------------------------------------------
#ifndef FA_GEN_COMMON_H
#define FA_GEN_COMMON_H

// catlass v2.0.0 的 arch 特化头由 CATLASS_ARCH 门控；ascendc cmake 的 host_bisheng
// 子构建不继承 ascendc_compile_definitions，因此在头文件内兜底定义（A2=2201）。
#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif
#ifndef BACKEND_HYBM
#define BACKEND_HYBM
#endif
#include "tl_templates/ascend/common.h"
#include "acl/acl.h"

using namespace Catlass;
using uint = unsigned int;

__aicore__ __forceinline__ int64_t min64(int64_t a, int64_t b) {
  return a < b ? a : b;
}

// ---- 编译期常量 -------------------------------------------------------------
namespace fa_gen {
constexpr int64_t BLOCK_N = 128;          // KV 块宽（恒 128）
constexpr int64_t RING = 3;               // ring_slots = prelaunch + 1
constexpr int64_t PRELAUNCH = 2;
constexpr int64_t MAX_CORES = 20;         // grid 固定；task_idx < block_num 守卫
}  // namespace fa_gen

// Host 侧填入的 tiling 参数（GM 传入，kernel 启动时读取）
struct FlashAttentionGenTiling {
  int64_t batch;
  int64_t heads;
  int64_t q_seq_len;
  int64_t kv_seq_len;
  int64_t dim;
  float sm_scale;
  int64_t causal;  // 0 / 1
};

// 运行期上下文：shape 派生量 + GM 张量 + 片上 buffer 视图
template <typename T>
struct FaGenContext {
  // runtime dims（Init 时从 tiling 计算）
  int64_t batch, heads, q_seq_len, kv_seq_len, dim;
  int64_t dim_align;        // AlignUp(dim, C0)，片上 buffer 宽度
  // HS32：block_m 按 dim_align 分档（GQA 同档），cube_k 单 chunk=128，
  // d_chunks 为 K 维 chunk 数
  int64_t block_m, half_m, cube_k, d_chunks;
  int64_t q_blocks, block_num, tasks_per_core, kv_loops, kv_tail;
  int64_t ws_s_task, ws_p_task, ws_o_task;  // per-core workspace 步长（元素）
  int64_t diff_s;           // kv_seq_len - q_seq_len（causal 右下对齐）
  int64_t causal;
  float sm_scale;

  // GM 张量
  AscendC::GlobalTensor<T> q, k, v, output, workspace_p;
  AscendC::GlobalTensor<float> workspace_s, workspace_o;

  // L1 视图（q/k/v 宽 dim_align；acc_s_l1 为 BLOCK_N 定宽 P 方块）
  AscendC::LocalTensor<T> q_l1, k_l1, k_l1b, acc_s_l1, v_l1;
  // L0 视图（fp32 时 lhs/p、rhs/v 别名复用，偏移由 constexpr 分支决定——已验证布局）
  AscendC::LocalTensor<T> lhs_l0, p_l0, rhs_l0, v_l0;
  AscendC::LocalTensor<float> acc_s_l0c, acc_o_l0c;
  // UB 视图（alpha/sumexp_i 按 ring slot UB 驻留；m_i_2d/alpha_2d 为 broadcast 目标）
  AscendC::LocalTensor<float> acc_o, sumexp, m_i, acc_s_ub, mask_col;
  AscendC::LocalTensor<float> alpha_ring, sumexp_i_ring, m_i_2d, acc_o_ub;
  // HS36 brcb-sub：m_i 的 8 路广播物化（dim_align<=128 档；其余档 8 元素哑元）
  AscendC::LocalTensor<float> brcb_tmp;
  AscendC::LocalTensor<uint8_t> tmp_ub;
  AscendC::LocalTensor<T> acc_s_half, acc_o_half;
  // causal mask 专用 UB（HS32 新增；仅 causal=True 且块内末列超界时使用）
  AscendC::LocalTensor<float> col_idx, bound_vec, bound_2d;
  AscendC::LocalTensor<uint8_t> causal_cmp;
};

// ---- 运行期宽度 copy 包装（fa_dyn） ------------------------------------------
// 与 tl::ascend::copy_* 逻辑相同，但 tile 宽度是运行期值（生成码里它们是模板常量）。
// catlass tla 布局的动态维是 uint32_t 类型成员，布局"类型"与数值无关，
// TileCopyTla 的编译期分派不受影响，只是数值从编译期挪到运行期。
namespace fa_dyn {

using namespace AscendC;
using namespace tla;
using namespace Catlass::Gemm::Tile;
using tl::ascend::ArchTag;
using tl::ascend::LayoutGM;   // layout::RowMajor
using tl::ascend::LayoutL0A;  // layout::zZ
using tl::ascend::LayoutL0B;  // layout::nZ
using tl::ascend::LayoutL1;   // layout::zN
using tl::ascend::LayoutL1T;  // layout::nZ

template <typename T>
CATLASS_DEVICE void copy_gm_to_l1_dyn(LocalTensor<T> dstTensor,
                                      GlobalTensor<T> srcTensor, uint32_t dstM,
                                      uint32_t dstN, uint32_t realSrcN,
                                      uint32_t tailM, uint32_t tailN,
                                      bool need_clear) {
  if (need_clear && (tailM != dstM || tailN != dstN)) {
    AscendC::InitConstValue(
        dstTensor,
        {1, static_cast<uint16_t>(dstM * dstN * sizeof(T) / 32), 0, 0});
    AscendC::PipeBarrier<PIPE_MTE2>();
  }
  auto layout = MakeLayoutFromTag(LayoutGM{tailM, realSrcN});
  auto src_LAYOUT = GetTileLayout(layout, tla::MakeShape(tailM, tailN),
                                  tla::MakeShape(uint32_t(0), uint32_t(0)));
  auto src = tla::MakeTensor(srcTensor, src_LAYOUT, Arch::PositionGM{});
  auto layoutInL1 = tla::MakeLayout<T, LayoutL1>(dstM, dstN);
  auto dst = tla::MakeTensor(dstTensor, layoutInL1, Arch::PositionL1{});
  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

template <typename T, bool transpose>
CATLASS_DEVICE void copy_l1_to_l0a_dyn(LocalTensor<T> dstTensor,
                                       LocalTensor<T> srcTensor, uint32_t srcM,
                                       uint32_t srcN, uint32_t dstM,
                                       uint32_t dstN) {
  using LayoutL1Tag_ = std::conditional_t<transpose, LayoutL1T, LayoutL1>;
  auto layout = transpose ? tla::MakeLayout<T, LayoutL1Tag_>(srcN, srcM)
                          : tla::MakeLayout<T, LayoutL1Tag_>(srcM, srcN);
  auto src_LAYOUT = GetTileLayout(layout, tla::MakeShape(dstM, dstN),
                                  tla::MakeShape(uint32_t(0), uint32_t(0)));
  auto src = tla::MakeTensor(srcTensor, src_LAYOUT, Arch::PositionL1{});
  auto layoutAInL0 = tla::MakeLayout<T, LayoutL0A>(dstM, dstN);
  auto dst = tla::MakeTensor(dstTensor, layoutAInL0, Arch::PositionL0A{});
  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

template <typename T, bool transpose>
CATLASS_DEVICE void copy_l1_to_l0b_dyn(LocalTensor<T> dstTensor,
                                       LocalTensor<T> srcTensor, uint32_t srcM,
                                       uint32_t srcN, uint32_t dstM,
                                       uint32_t dstN) {
  using LayoutL1Tag_ = std::conditional_t<transpose, LayoutL1T, LayoutL1>;
  auto layout = transpose ? tla::MakeLayout<T, LayoutL1Tag_>(srcN, srcM)
                          : tla::MakeLayout<T, LayoutL1Tag_>(srcM, srcN);
  auto src_LAYOUT = GetTileLayout(layout, tla::MakeShape(dstM, dstN),
                                  tla::MakeShape(uint32_t(0), uint32_t(0)));
  auto src = tla::MakeTensor(srcTensor, src_LAYOUT, Arch::PositionL1{});
  auto layoutBInL0 = tla::MakeLayout<T, LayoutL0B>(dstM, dstN);
  auto dst = tla::MakeTensor(dstTensor, layoutBInL0, Arch::PositionL0B{});
  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

template <typename T2>
CATLASS_DEVICE void copy_l0c_to_gm_dyn(GlobalTensor<T2> dstTensor,
                                       LocalTensor<float> srcTensor,
                                       uint32_t srcM, uint32_t srcN,
                                       uint32_t realDstN, uint32_t tailM,
                                       uint32_t tailN) {
  auto layoutInL0C = tla::MakeLayoutL0C(srcM, srcN);
  auto src = tla::MakeTensor(srcTensor, layoutInL0C, Arch::PositionL0C{});
  auto layout = MakeLayoutFromTag(LayoutGM{tailM, realDstN});
  auto dst_LAYOUT = GetTileLayout(layout, tla::MakeShape(tailM, tailN),
                                  tla::MakeShape(uint32_t(0), uint32_t(0)));
  auto dst = tla::MakeTensor(dstTensor, dst_LAYOUT, Arch::PositionGM{});
  CopyL0CToGmTla<ArchTag, decltype(src), decltype(dst),
                 ScaleGranularity::NO_QUANT, false>
      tileCopier;
  tileCopier(dst, src, 0);
}

template <typename T>
CATLASS_DEVICE void copy_gm_to_ub_dyn(LocalTensor<T> dstTensor,
                                      GlobalTensor<T> srcTensor,
                                      uint32_t dstPitch, uint32_t realSrcN,
                                      uint32_t maskShapeM, uint32_t maskShapeN,
                                      T padValue) {
  // 与 tl::ascend::copy_gm_to_ub 相同的混合 padding 方案，UB 行距为运行期值。
  bool isPad = true;
  uint32_t rightPadding = 1;
  if (maskShapeN == dstPitch || (maskShapeN * sizeof(T)) % 32 == 0) {
    isPad = false;
    rightPadding = 0;
  }
  if (maskShapeM * dstPitch != maskShapeM * maskShapeN) {
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::Duplicate<T>(dstTensor, padValue, maskShapeM * dstPitch);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
  }
  AscendC::DataCopyExtParams dataCopyParams(
      maskShapeM, maskShapeN * sizeof(T), (realSrcN - maskShapeN) * sizeof(T),
      (dstPitch - maskShapeN) * sizeof(T) / 32, 0);
  AscendC::DataCopyPadExtParams<T> padParams(isPad, 0, rightPadding, padValue);
  AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
}

template <typename T>
CATLASS_DEVICE void copy_ub_to_gm_dyn(GlobalTensor<T> dstTensor,
                                      LocalTensor<T> srcTensor,
                                      uint32_t srcPitch, uint32_t realDstN,
                                      uint32_t maskShapeM,
                                      uint32_t maskShapeN) {
  AscendC::DataCopyExtParams dataCopyParams(
      maskShapeM, maskShapeN * sizeof(T), (srcPitch - maskShapeN) * sizeof(T) / 32,
      (realDstN - maskShapeN) * sizeof(T), 0);
  AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
}

template <typename T1, typename T2>
CATLASS_DEVICE void copy_ub_to_ub_dyn(LocalTensor<T1> dstTensor,
                                      LocalTensor<T2> srcTensor, uint32_t len,
                                      uint32_t rows, uint32_t cols,
                                      uint32_t src_stride, uint32_t dst_stride) {
  // 带运行期行距的 Cast/DataCopy（tl::ascend::copy_ub_to_ub 的长度是模板常量）。
  // Cast RoundMode 按 tl 源码分 dtype：升宽用 CAST_NONE，其余 CAST_RINT（68 实战
  // ISSUE-G2：升宽一律 RINT 会把 |x|<0.5 小数全舍成 0）。
  constexpr auto cast_mode = []() {
    if constexpr (std::is_same_v<T1, float> &&
                  (std::is_same_v<T2, half> || std::is_same_v<T2, bfloat16_t> ||
                   std::is_same_v<T2, int16_t>)) {
      return AscendC::RoundMode::CAST_NONE;
    } else if constexpr (std::is_same_v<T1, half> &&
                         std::is_same_v<T2, int8_t>) {
      return AscendC::RoundMode::CAST_NONE;
    } else if constexpr (std::is_same_v<T1, int16_t> &&
                         std::is_same_v<T2, int32_t>) {
      return AscendC::RoundMode::CAST_NONE;
    } else {
      return AscendC::RoundMode::CAST_RINT;
    }
  }();
  if (cols == src_stride && cols == dst_stride) {
    if constexpr (std::is_same_v<T1, T2>) {
      AscendC::DataCopy(dstTensor, srcTensor, len);
    } else {
      AscendC::Cast(dstTensor, srcTensor, cast_mode, len);
    }
  } else {
    for (uint32_t i = 0; i < rows; i++) {
      if constexpr (std::is_same_v<T1, T2>) {
        AscendC::DataCopy(dstTensor[i * dst_stride], srcTensor[i * src_stride],
                          cols);
      } else {
        AscendC::Cast(dstTensor[i * dst_stride], srcTensor[i * src_stride],
                      cast_mode, cols);
      }
    }
  }
}

// ---- 运行期 M/N/K 的 mma / reduce 包装（HS32 新增，block_m 分档后 M 运行期） ----

// 直接调 Mmad：MmadParams.m/n/k 本来就是运行期字段，M/N 模板常量只用来填它。
template <typename T>
CATLASS_DEVICE void mma_dyn(LocalTensor<T> A, LocalTensor<T> B,
                            LocalTensor<float> C, bool init, uint32_t M,
                            uint32_t N, uint32_t K) {
  MmadParams mmadParams;
  mmadParams.m = M;
  mmadParams.n = N;
  mmadParams.k = K;
  mmadParams.cmatrixInitVal = init;
  mmadParams.cmatrixSource = false;
  mmadParams.unitFlag = 0;
  Mmad(C, A, B, mmadParams);
}

// reduce dim=-1（行维）：M 运行期（=half_m）。
template <typename T>
CATLASS_DEVICE void reduce_max_dyn(LocalTensor<T> dst, LocalTensor<T> src,
                                   LocalTensor<uint8_t> tmp, uint32_t M,
                                   uint32_t N) {
  uint32_t shape[] = {M, N};
  AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape,
                                                       true);
}

template <typename T>
CATLASS_DEVICE void reduce_sum_dyn(LocalTensor<T> dst, LocalTensor<T> src,
                                   LocalTensor<uint8_t> tmp, uint32_t M,
                                   uint32_t N) {
  uint32_t shape[] = {M, N};
  AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR>(dst, src, tmp, shape,
                                                       true);
}

}  // namespace fa_dyn

#endif  // FA_GEN_COMMON_H
