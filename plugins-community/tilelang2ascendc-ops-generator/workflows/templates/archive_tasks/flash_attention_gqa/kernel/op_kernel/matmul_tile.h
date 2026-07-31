#ifndef MATMUL_TILE_H
#define MATMUL_TILE_H

#include "kernel_operator.h"

// GM(ND) -> L1(Nz)
// m % 16 == 0 && n % c0 == 0
template<typename T>
__aicore__ inline void LoadNdGmToNzL1(const AscendC::LocalTensor<T> &dst,
                                      const AscendC::GlobalTensor<T> &src,
                                      uint32_t m, uint32_t n, uint32_t ld)
{
    AscendC::Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = m;
    params.dValue = n;
    params.srcNdMatrixStride = 0;
    params.srcDValue = ld;
    params.dstNzC0Stride = m;
    params.dstNzNStride = 1;
    params.dstNzMatrixStride = 0;
    AscendC::DataCopy(dst, src, params);
}

// GM(ND) -> L1(Nz) with explicit dstNzC0Stride
// When loading a tail tile where m (actual rows) != mAligned (aligned rows),
// we need dstNzC0Stride = mAligned so that L1 layout matches L0B expectations.
template<typename T>
__aicore__ inline void LoadNdGmToNzL1(const AscendC::LocalTensor<T> &dst,
                                      const AscendC::GlobalTensor<T> &src,
                                      uint32_t m, uint32_t n, uint32_t ld,
                                      uint32_t dstNzC0Stride)
{
    AscendC::Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = m;
    params.dValue = n;
    params.srcNdMatrixStride = 0;
    params.srcDValue = ld;
    params.dstNzC0Stride = dstNzC0Stride;
    params.dstNzNStride = 1;
    params.dstNzMatrixStride = 0;
    AscendC::DataCopy(dst, src, params);
}

// L1(Nz) -> L0A(Zz)
// m % 16 == 0 && k % c0 == 0
template<typename T>
__aicore__ inline void LoadNzL1ToZzL0A(const AscendC::LocalTensor<T> &dst,
                                       const AscendC::LocalTensor<T> &src,
                                       uint32_t m, uint32_t k, uint32_t colC0Stride)
{
    AscendC::LoadData3DParamsV2<T> params;
    params.l1H = 1;
    params.l1W = colC0Stride;
    params.channelSize = k;
    params.kExtension = k;
    params.mExtension = m;
    params.strideH = 1;
    params.strideW = 1;
    params.filterH = 1;
    params.filterW = 1;
    params.dilationFilterH = 1;
    params.dilationFilterW = 1;
    AscendC::LoadData(dst, src, params);
}

// L1(Nz) -> L0B(Zn)
// Used for B matrix (or K^T in flash attention)
template<typename T>
__aicore__ inline void LoadNzL1ToZnL0B(const AscendC::LocalTensor<T> &dst,
                                       const AscendC::LocalTensor<T> &src,
                                       uint32_t k, uint32_t n, uint32_t colC0Stride)
{
    AscendC::LoadData3DParamsV2<T> params;
    params.l1H = 1;
    params.l1W = colC0Stride;
    params.channelSize = n;
    params.kExtension = n;
    params.mExtension = k;
    params.strideH = 1;
    params.strideW = 1;
    params.filterH = 1;
    params.filterW = 1;
    params.dilationFilterH = 1;
    params.dilationFilterW = 1;
    AscendC::LoadData(dst, src, params);
}

// L1(Nz) -> L0B(Zn) with transpose semantics
template<typename T>
__aicore__ inline void LoadNzL1ToZnL0BWithTranspose(const AscendC::LocalTensor<T> &dst,
                                                    const AscendC::LocalTensor<T> &src,
                                                    uint32_t k, uint32_t n, uint32_t colC0Stride)
{
    constexpr uint32_t c0 = 32 / sizeof(T);
    AscendC::LoadData2DParams p;
    p.startIndex = 0;
    p.srcStride = 1;
    p.dstGap = 0;
    p.ifTranspose = false;
    if (colC0Stride == n) {
        p.repeatTimes = (k / c0) * (n / c0);
        AscendC::LoadData(dst, src, p);
    } else {
        p.repeatTimes = n / c0;
        for (uint32_t i = 0; i < (k / c0); ++i) {
            AscendC::LoadData(dst[(uint64_t)n * i * c0], src[(uint64_t)colC0Stride * i * c0], p);
        }
    }
}

// L0C(Nz) -> GM(ND), with custom dstStride for writing sub-tiles
template<typename T>
__aicore__ inline void FixpipeNzL0cToNdGm(const AscendC::GlobalTensor<T> &dst,
                                          const AscendC::LocalTensor<T> &src,
                                          uint32_t m, uint32_t n,
                                          uint32_t dstStride = 0)
{
    AscendC::FixpipeParamsV220 params;
    params.nSize = n;
    params.mSize = m;
    params.srcStride = m;
    params.dstStride = (dstStride == 0) ? n : dstStride;
    params.ndNum = 1;
    params.srcNdStride = 0;
    params.dstNdStride = 0;
    AscendC::Fixpipe(dst, src, params);
}

#endif // MATMUL_TILE_H
