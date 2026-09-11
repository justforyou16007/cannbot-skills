// ----------------------------------------------------------------------------------------------------------
// GQA forward (generalized) - 公共定义：常量 / tiling / dyn copy 包装
//
// 设备实现与 ascendc/gqa.cpp 同源（tilelang 编译产物的手工泛化，全量 20/20 验证）。
// 同步结构（SetFlag/WaitFlag/PipeBarrier/CrossCoreFlag）与已验证的单文件版逐字节一致，
// 禁止"顺手优化"。
//
// catlass v2.0.0 的 arch 特化头由 CATLASS_ARCH 门控；ascendc cmake 的 host_bisheng
// 子构建不继承 ascendc_compile_definitions，因此在头文件内兜底定义（A2=2201）。
// ----------------------------------------------------------------------------------------------------------
#ifndef GQA_GEN_COMMON_H
#define GQA_GEN_COMMON_H

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
namespace gqa_gen {
constexpr int64_t BLOCK_N_BASE = 128;  // L0B chunk width (rhs+v fill 64KB L0B)
constexpr int64_t RING = 3;          // ring_slots = prelaunch + 1
constexpr int64_t PRELAUNCH = 2;
// Core count is queried at runtime (kernel: GetBlockNum(); host: GetCoreNumAic)
// with a task_idx < block_num guard -- never hardcoded.
constexpr float MASK_NEG = -1073741824;  // -2**30, softmax mask fill
}  // namespace gqa_gen

// Host 侧填入的 tiling 参数（GM 传入，kernel 启动时读取）。8B 对齐。
struct GqaGenTiling {
  int64_t batch;
  int64_t heads_q;
  int64_t heads_kv;
  int64_t q_seq_len;
  int64_t kv_seq_len;
  int64_t dim;
  int64_t causal;
  float sm_scale;
  int64_t pad;  // 8B 对齐占位
};

namespace gqa_dyn {

using namespace AscendC;
using namespace tla;
using Catlass::Gemm::Tile::CopyL0CToGmTla;
using Catlass::Gemm::Tile::TileCopyTla;
using tl::ascend::ArchTag;
using tl::ascend::LayoutGM;   // layout::RowMajor
using tl::ascend::LayoutL0A;  // layout::zZ
using tl::ascend::LayoutL0B;  // layout::nZ
using tl::ascend::LayoutL1;   // layout::zN
using tl::ascend::LayoutL1T;  // layout::nZ

// ---- runtime-width variants of the tl::ascend wrappers ---------------------
// Identical logic to tl::ascend::copy_* / mma / reduce_* but the tile extents
// (which the generated code passes as template constants) are runtime values.

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
                 tl::ascend::ScaleGranularity::NO_QUANT, false>
      tileCopier;
  tileCopier(dst, src, 0);
}

template <typename T>
CATLASS_DEVICE void copy_gm_to_ub_dyn(LocalTensor<T> dstTensor,
                                      GlobalTensor<T> srcTensor,
                                      uint32_t dstPitch, uint32_t realSrcN,
                                      uint32_t maskShapeM, uint32_t maskShapeN,
                                      T padValue) {
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

// UB->UB cast/copy with runtime row pitch. RoundMode follows the tl::ascend
// source per-dtype: widening casts use CAST_NONE, everything else CAST_RINT.
// GQA only ever uses the narrowing float->half/bf16 path (CAST_RINT) and the
// float->float flat copy, but the per-dtype dispatch keeps the helper correct.
template <typename T1, typename T2>
CATLASS_DEVICE void copy_ub_to_ub_dyn(LocalTensor<T1> dstTensor,
                                      LocalTensor<T2> srcTensor, uint32_t len,
                                      uint32_t rows, uint32_t cols,
                                      uint32_t src_stride, uint32_t dst_stride) {
  constexpr bool kWiden =
      (std::is_same_v<T1, float> &&
       (std::is_same_v<T2, half> || std::is_same_v<T2, bfloat16_t> ||
        std::is_same_v<T2, int16_t>)) ||
      (std::is_same_v<T1, half> && std::is_same_v<T2, int8_t>) ||
      (std::is_same_v<T1, int16_t> && std::is_same_v<T2, int32_t>);
  constexpr AscendC::RoundMode kMode =
      kWiden ? AscendC::RoundMode::CAST_NONE : AscendC::RoundMode::CAST_RINT;
  if (cols == src_stride && cols == dst_stride) {
    if constexpr (std::is_same_v<T1, T2>) {
      AscendC::DataCopy(dstTensor, srcTensor, len);
    } else {
      AscendC::Cast(dstTensor, srcTensor, kMode, len);
    }
  } else {
    for (uint32_t i = 0; i < rows; i++) {
      if constexpr (std::is_same_v<T1, T2>) {
        AscendC::DataCopy(dstTensor[i * dst_stride], srcTensor[i * src_stride],
                          cols);
      } else {
        AscendC::Cast(dstTensor[i * dst_stride], srcTensor[i * src_stride],
                      kMode, cols);
      }
    }
  }
}

// Runtime-M/N/K mma: tl::ascend::mma takes M/N as template constants;
// MmadParams .m/.n/.k are runtime fields, so call Mmad directly. GQA always
// uses init=True (each d_chunks chunk accumulates independently).
template <typename T1, typename T2>
CATLASS_DEVICE void mma_dyn(LocalTensor<T1> const A, LocalTensor<T1> const B,
                            LocalTensor<T2> const C, bool init, uint32_t M,
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

// Runtime-shape reduce (dim=-1, AR pattern, clear=true).
template <typename T>
CATLASS_DEVICE void reduce_max_dyn(LocalTensor<T> const &dstTensor,
                                   LocalTensor<T> const &srcTensor,
                                   LocalTensor<uint8_t> const &sharedTmpBuffer,
                                   uint32_t M, uint32_t N) {
  uint32_t shape[] = {M, N};
  AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR>(dstTensor, srcTensor,
                                                      sharedTmpBuffer, shape,
                                                      true);
}

template <typename T>
CATLASS_DEVICE void reduce_sum_dyn(LocalTensor<T> const &dstTensor,
                                   LocalTensor<T> const &srcTensor,
                                   LocalTensor<uint8_t> const &sharedTmpBuffer,
                                   uint32_t M, uint32_t N) {
  uint32_t shape[] = {M, N};
  AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR>(dstTensor, srcTensor,
                                                      sharedTmpBuffer, shape,
                                                      true);
}

}  // namespace gqa_dyn

#endif  // GQA_GEN_COMMON_H
