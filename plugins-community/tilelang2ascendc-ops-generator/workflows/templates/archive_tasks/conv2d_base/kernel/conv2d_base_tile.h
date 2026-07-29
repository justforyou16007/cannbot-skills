/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file conv2d_base_tile.h
 * GM<->L1<->L0<->GM transfer wrappers for conv2d_base (arch20).
 * Mirrors the proven wrappers in 8_WeightQuantBatchmatmul/matmul_tile.h.
 */

#ifndef CONV2D_BASE_TILE_H
#define CONV2D_BASE_TILE_H

#include "kernel_operator.h"

// GM(ND) -> L1(Nz), rows m, cols n, leading dim ld, explicit dstNzC0Stride
// (used for the K-major weight panel B [kLen, nActual] with ld = Co)
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

#endif // CONV2D_BASE_TILE_H
