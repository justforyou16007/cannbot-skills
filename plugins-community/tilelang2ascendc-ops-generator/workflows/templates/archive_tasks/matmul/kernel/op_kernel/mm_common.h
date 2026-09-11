// ----------------------------------------------------------------------------------------------------------
// Matmul (generalized) - 公共定义：tiling / mm_dyn 运行期宽度 copy 包装
//
// deliverer Phase 4/5 产物。设备语义来源：design/tile_level/matmul.py 的
// tilelang 编译产物（intrinsic 手写流水版，精简集 9/9 验证），泛化规则：
//   - M/N/K -> 运行期 int64（tiling 传入）；block 尺寸全部保持编译期常量
//     （per-dtype：fp32 用 128x128xK_L1=128，fp16/bf16 用 128x256xK_L1=256）
//   - L1/L0 buffer 尺寸与偏移全部是编译期常量（与生成码逐字节一致）
//   - flag / 同步结构与已验证生成码一致，禁止"顺手优化"
//   - thread_block_swizzle 的模板形状 -> 运行期整数实现 swizzle_zn3
//     （算法与 catlass GemmIdentityBlockSwizzle<3, Zn> 逐步对应）
// ----------------------------------------------------------------------------------------------------------
#ifndef MM_GEN_COMMON_H
#define MM_GEN_COMMON_H

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

namespace mm_gen {

constexpr int64_t BLOCK_M = 128;
constexpr int64_t S1 = 2;   // L1 双 buffer 槽数
constexpr int64_t S2 = 2;   // L0 双 buffer 槽数
constexpr int64_t CORE_NUM = 20;

// per-dtype block 配置（容量推导见 design/tile_level/matmul.py 头注释）
template <typename T>
struct MmCfg;
template <>
struct MmCfg<half> {
  static constexpr int64_t BLOCK_N = 256;
  static constexpr int64_t BLOCK_K = 64;
  static constexpr int64_t K_L1 = 256;
};
template <>
struct MmCfg<bfloat16_t> {
  static constexpr int64_t BLOCK_N = 256;
  static constexpr int64_t BLOCK_K = 64;
  static constexpr int64_t K_L1 = 256;
};
template <>
struct MmCfg<float> {
  static constexpr int64_t BLOCK_N = 128;
  static constexpr int64_t BLOCK_K = 64;
  static constexpr int64_t K_L1 = 128;
};

// Host 侧填入的 tiling 参数（GM 传入，kernel 启动时读取）
struct MatmulGenTiling {
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t pad;  // 8B 对齐占位
};

// thread_block_swizzle<M_pad, N_pad, K, bM, bN, off=3, Zn> 的运行期等价实现
// （catlass GemmIdentityBlockSwizzle::GetBlockCoord，loopsMN=(m_num, n_num) 由
// 调用者按 pad 后尺寸算好传入；task -> m_idx * n_num + n_idx）
__aicore__ __forceinline__ int64_t swizzle_zn3(int64_t pid, int64_t m_num,
                                               int64_t n_num) {
  constexpr int64_t OFF = 3;
  const int64_t inner = pid % (m_num * n_num);
  const int64_t tile_block_loop = (m_num + OFF - 1) / OFF;
  const int64_t tile_block_idx = inner / (OFF * n_num);
  const int64_t in_tile_block_idx = inner % (OFF * n_num);
  int64_t n_row = OFF;
  if (tile_block_idx == tile_block_loop - 1) {
    n_row = m_num - OFF * tile_block_idx;
  }
  const int64_t m_idx = tile_block_idx * OFF + in_tile_block_idx % n_row;
  int64_t n_idx = in_tile_block_idx / n_row;
  if (tile_block_idx % 2 == 1) {
    n_idx = n_num - n_idx - 1;
  }
  return m_idx * n_num + n_idx;
}

}  // namespace mm_gen

// ---- 运行期宽度 copy 包装（mm_dyn） ------------------------------------------
// 与 tl::ascend::copy_* 逻辑相同，但 tile 宽度/GM 行距是运行期值（生成码里它们
// 是模板/字面量常量）。catlass tla 布局的动态维是 uint32_t 类型成员，布局"类型"
// 与数值无关，TileCopyTla 的编译期分派不受影响。
namespace mm_dyn {

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

}  // namespace mm_dyn

#endif  // MM_GEN_COMMON_H
