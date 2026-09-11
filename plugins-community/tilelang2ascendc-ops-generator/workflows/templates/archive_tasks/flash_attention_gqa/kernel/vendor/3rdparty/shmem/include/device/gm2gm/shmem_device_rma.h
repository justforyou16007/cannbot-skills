/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_DEVICE_GM2GM_RMA_H
#define SHMEM_DEVICE_GM2GM_RMA_H

#include "kernel_operator.h"
#include "device/gm2gm/engine/shmem_device_mte.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "device/gm2gm/shmem_device_mo.h"
#include "host/shmem_host_def.h"
#include "device/shmem_def.h"

/**
 * @brief Standard RMA Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |half       | half      |
 * |float      | float     |
 * |double     | double    |
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 * |int64      | int64     |
 * |uint8      | uint8     |
 * |uint16     | uint16    |
 * |uint32     | uint32    |
 * |uint64     | uint64    |
 * |char       | char      |
 * |bfloat16   | bfloat16  |
 */
#define ACLSHMEM_TYPE_FUNC(FUNC) \
    FUNC(half, half);            \
    FUNC(float, float);          \
    FUNC(double, double);        \
    FUNC(int8, int8_t);          \
    FUNC(int16, int16_t);        \
    FUNC(int32, int32_t);        \
    FUNC(int64, int64_t);        \
    FUNC(uint8, uint8_t);        \
    FUNC(uint16, uint16_t);      \
    FUNC(uint32, uint32_t);      \
    FUNC(uint64, uint64_t);      \
    FUNC(char, char);            \
    FUNC(bfloat16, bfloat16_t)

/**
 * @brief Standard test Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |float      | float     |
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 * |int64      | int64     |
 * |uint8      | uint8     |
 * |uint16     | uint16    |
 * |uint32     | uint32    |
 * |uint64     | uint64    |
 * |char       | char      |
 */
#define ACLSHMEM_TEST_TYPE_FUNC(FUNC) \
    FUNC(float, float);               \
    FUNC(int8, int8_t);               \
    FUNC(int16, int16_t);             \
    FUNC(int32, int32_t);             \
    FUNC(int64, int64_t);             \
    FUNC(uint8, uint8_t);             \
    FUNC(uint16, uint16_t);           \
    FUNC(uint32, uint32_t);           \
    FUNC(uint64, uint64_t);           \
    FUNC(char, char)

#define ACLSHMEM_TYPENAME_P_AICORE(NAME, TYPE)                                              \
    /**                                                                                     \
     * @brief Provide a low latency put capability for single element of most basic types.  \
     *                                                                                      \
     * @param dst               [in] Symmetric address of the destination data on local PE. \
     * @param value             [in] The element to be put.                                 \
     * @param pe                [in] The number of the remote PE.                           \
     */                                                                                     \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_p(__gm__ TYPE *dst, const TYPE value, int pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_P_AICORE);

#define ACLSHMEM_TYPENAME_G_AICORE(NAME, TYPE)                                              \
    /**                                                                                     \
     * @brief Provide a low latency get capability for single element of most basic types.  \
     *                                                                                      \
     * @param src               [in] Symmetric address of the destination data on local PE. \
     * @param pe                [in] The number of the remote PE.                           \
     * @return A single element of type specified in the input pointer.                     \
     */                                                                                     \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_g(__gm__ TYPE *src, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_G_AICORE);

/**
 * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *                       address on the local PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmem_getmem(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe);

#define ACLSHMEM_GET_TYPENAME_MEM(NAME, TYPE)                                                                    \
    /**                                                                                                          \
     * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to           \
     *                               address on the local PE.                                                    \
     *                                                                                                           \
     * @param dst               [in] Pointer on local device of the destination data.                            \
     * @param src               [in] Pointer on Symmetric memory of the source data.                             \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                           \
     * @param pe                [in] PE number of the remote PE.                                                 \
     */                                                                                                          \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM);

/**
 * @brief Synchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmem_putmem(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe);

#define ACLSHMEM_PUT_TYPENAME_MEM(NAME, TYPE)                                                                     \
    /**                                                                                                           \
     * @brief Synchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE. \
     *                                                                                                            \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                         \
     * @param src               [in] Pointer on local device of the source data.                                  \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                     \
     * @param pe                [in] PE number of the remote PE.                                                  \
     */                                                                                                           \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM);

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *                                              address on the local PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmem_getmem_nbi(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe);

#define ACLSHMEM_GET_TYPENAME_MEM_NBI(NAME, TYPE)                                                                    \
    /**                                                                                                              \
     * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to              \
     *                                                   address on the local PE.                                    \
     *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the             \
     *                 same PE are not supported.                                                                    \
     *                                                                                                               \
     * @param dst               [in] Pointer on local device of the destination data.                                \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                 \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                               \
     * @param pe                [in] PE number of the remote PE.                                                     \
     */                                                                                                              \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_NBI);

#define ACLSHMEM_GET_TYPENAME_MEM_DETAILED_NBI(NAME, TYPE)                                                         \
    /**                                                                                                            \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                   \
     *        on symmetric memory from the specified PE to address on the local device.                            \
     *                                                                                                             \
     * @param dst               [in] Pointer on local device of the destination data.                              \
     * @param src               [in] Pointer on Symmetric memory of the source data.                               \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.         \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(__gm__ TYPE *dst, __gm__ TYPE *src,                             \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_DETAILED_NBI);

#define ACLSHMEM_GET_TYPENAME_MEM_TENSOR_NBI(NAME, TYPE)                                                                 \
    /**                                                                                                                  \
     * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to                  \
     *                           address on the local PE.                                                                \
     *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the                 \
     *                 same PE are not supported.                                                                        \
     *                                                                                                                   \
     * @param dst               [in] GlobalTensor on local device of the destination data.                               \
     * @param src               [in] GlobalTensor on Symmetric memory of the source data.                                \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                                   \
     * @param pe                [in] PE number of the remote PE.                                                         \
     */                                                                                                                  \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                 uint32_t elem_size, int pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_TENSOR_NBI);

#define ACLSHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED_NBI(NAME, TYPE)                                                        \
    /**                                                                                                                  \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                         \
     *        on symmetric memory from the specified PE to address on the local device.                                  \
     *                                                                                                                   \
     * @param dst               [in] GlobalTensor on local device of the destination data.                               \
     * @param src               [in] GlobalTensor on Symmetric memory of the source data.                                \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.               \
     * @param pe                [in] PE number of the remote PE.                                                         \
     */                                                                                                                  \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                 const non_contiguous_copy_param &copy_params, int pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_NBI(NAME, TYPE)                                                                    \
    /**                                                                                                              \
     * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.     \
     *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the             \
     *                 same PE are not supported.                                                                    \
     *                                                                                                               \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                            \
     * @param src               [in] Pointer on local device of the source data.                                     \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                        \
     * @param pe                [in] PE number of the remote PE.                                                     \
     */                                                                                                              \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_DETAILED_NBI(NAME, TYPE)                                                         \
    /**                                                                                                            \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                   \
     *        on local PE to symmetric address on the specified PE.                                                \
     *                                                                                                             \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                          \
     * @param src               [in] Pointer on local device of the source data.                                   \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.         \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(__gm__ TYPE *dst, __gm__ TYPE *src,                             \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_DETAILED_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_NBI(NAME, TYPE)                                                                 \
    /**                                                                                                                  \
     * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.         \
     *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the                 \
     *                 same PE are not supported.                                                                        \
     *                                                                                                                   \
     * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                           \
     * @param src               [in] GlobalTensor on local device of the source data.                                    \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                            \
     * @param pe                [in] PE number of the remote PE.                                                         \
     */                                                                                                                  \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                 uint32_t elem_size, int pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED_NBI(NAME, TYPE)                                                        \
    /**                                                                                                                  \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                         \
     *        on local PE to symmetric address on the specified PE.                                                      \
     *                                                                                                                   \
     * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                           \
     * @param src               [in] GlobalTensor on local device of the source data.                                    \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.               \
     * @param pe                [in] PE number of the remote PE.                                                         \
     */                                                                                                                  \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                 const non_contiguous_copy_param &copy_params, int pe)

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED_NBI);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmem_putmem_nbi(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe);

#define ACLSHMEM_TEST(NAME, TYPE)                                                                             \
    /**                                                                                                       \
     * @brief Synchronous interface. Provide a high-performance way to compare data                           \
     *        on local UB to symmetric address on the specified PE.                                           \
     *                                                                                                        \
     * @param ivar               [in] Pointer on local device of the destination data.                        \
     * @param cmp                [in] Compare type i.e. SHMEM_CMP_EQ, SHMEM_CMP_GT, etc.                      \
     * @param cmp_value          [in] Expected data for comparation                                           \
     */                                                                                                       \
    ACLSHMEM_DEVICE int aclshmem_##NAME##_test(__gm__ TYPE *ivar, int cmp, TYPE cmp_value)

ACLSHMEM_TEST_TYPE_FUNC(ACLSHMEM_TEST);

/**
 * @brief Set necessary parameters for put or get.
 *
 * @param offset                [in] The start address on UB.
 * @param ub_size               [in] The Size of Temp UB Buffer.
 * @param event_id              [in] Sync ID for put or get.
 */
ACLSHMEM_DEVICE void aclshmemx_set_mte_config(uint64_t offset, uint32_t ub_size, uint32_t event_id);

#include "gm2gm/shmem_device_rma.hpp"
#endif