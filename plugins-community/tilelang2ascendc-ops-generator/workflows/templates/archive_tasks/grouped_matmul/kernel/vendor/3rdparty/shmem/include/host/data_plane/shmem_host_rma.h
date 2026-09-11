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
#ifndef SHMEM_HOST_RMA_H
#define SHMEM_HOST_RMA_H

#include "acl/acl.h"
#include "host/shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
* @brief Standard RMA Types and Names valid on Host
*
* |NAME       | TYPE      |
* |-----------|-----------|
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
*/
#define ACLSHMEM_TYPE_FUNC(FUNC) \
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
    FUNC(char, char)

/**
 * @addtogroup group_enums
 * @{
*/
/// \brief Enumeration indicating whether the method is blocking or non-blocking
enum aclshmem_block_mode_t{
    NO_NBI = 0,  ///< Non-blocking mode disabled (method is blocking)
    NBI          ///< Non-blocking mode enabled (method is non-blocking, NBI = Non-Blocking Interface)
};
/**@} */ // end of group_enums

/**
 * @brief Translate an local symmetric address to remote symmetric address on the specified PE.
 *        Firstly, check whether the input address is legal on local PE. Then translate it into remote address
 *        on specified PE. Otherwise, returns a null pointer.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] The number of the remote PE.
 * @return If the input address is legal, returns a remote symmetric address on the specified PE that can be
 *         accessed using memory loads and stores. Otherwise, a null pointer is returned.
 */
ACLSHMEM_HOST_API void* aclshmem_ptr(void *ptr, int pe);
#define shmem_ptr aclshmem_ptr

#define ACLSHMEM_TYPE_PUT(NAME, TYPE)                                                                                 \
    /**                                                                                                               \
    * @brief Synchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.      \
    *                                                                                                                 \
    * @param dest               [in] Pointer on Symmetric memory of the destination data.                             \
    * @param source             [in] Pointer on local device of the source data.                                      \
    * @param nelems             [in] Number of elements in the destination and source arrays.                         \
    * @param pe                 [in] PE number of the remote PE.                                                      \
    */                                                                                                                \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_put(TYPE *dest, TYPE *source, size_t nelems, int pe)
    

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_PUT);
#define shmem_put_float_mem aclshmem_float_put
#define shmem_put_double_mem aclshmem_double_put
#define shmem_put_int8_mem aclshmem_int8_put
#define shmem_put_int16_mem aclshmem_int16_put
#define shmem_put_int32_mem aclshmem_int32_put
#define shmem_put_int64_mem aclshmem_int64_put
#define shmem_put_uint8_mem aclshmem_uint8_put
#define shmem_put_uint16_mem aclshmem_uint16_put
#define shmem_put_uint32_mem aclshmem_uint32_put
#define shmem_put_uint64_mem aclshmem_uint64_put
#define shmem_put_char_mem aclshmem_char_put
#undef ACLSHMEM_TYPE_PUT

#define ACLSHMEM_TYPE_PUT_NBI(NAME, TYPE)                                                                             \
    /**                                                                                                               \
    * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.     \
    *                                                                                                                 \
    * @param dest               [in] Pointer on Symmetric memory of the destination data.                             \
    * @param source             [in] Pointer on local device of the source data.                                      \
    * @param nelems             [in] Number of elements in the destination and source arrays.                         \
    * @param pe                 [in] PE number of the remote PE.                                                      \
    */                                                                                                                \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_put_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe)
    

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_PUT_NBI);

#define shmem_put_float_mem_nbi aclshmem_float_put_nbi
#define shmem_put_double_mem_nbi aclshmem_double_put_nbi
#define shmem_put_int8_mem_nbi aclshmem_int8_put_nbi
#define shmem_put_int16_mem_nbi aclshmem_int16_put_nbi
#define shmem_put_int32_mem_nbi aclshmem_int32_put_nbi
#define shmem_put_int64_mem_nbi aclshmem_int64_put_nbi
#define shmem_put_uint8_mem_nbi aclshmem_uint8_put_nbi
#define shmem_put_uint16_mem_nbi aclshmem_uint16_put_nbi
#define shmem_put_uint32_mem_nbi aclshmem_uint32_put_nbi
#define shmem_put_uint64_mem_nbi aclshmem_uint64_put_nbi
#define shmem_put_char_mem_nbi aclshmem_char_put_nbi


#undef ACLSHMEM_TYPE_PUT_NBI

#define ACLSHMEM_TYPE_GET(NAME, TYPE)                                                                                 \
    /**                                                                                                               \
    * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the  \
    *   local PE.                                                                                                     \
    *                                                                                                                 \
    * @param dest               [in] Pointer on local device of the destination data.                                 \
    * @param source             [in] Pointer on Symmetric memory of the source data.                                  \
    * @param nelems             [in] Number of elements in the destination and source arrays.                         \
    * @param pe                 [in] PE number of the remote PE.                                                      \
    */                                                                                                                \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_get(TYPE *dest, TYPE *source, size_t nelems, int pe)
    

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_GET);

#define shmem_get_float_mem aclshmem_float_get
#define shmem_get_double_mem aclshmem_double_get
#define shmem_get_int8_mem aclshmem_int8_get
#define shmem_get_int16_mem aclshmem_int16_get
#define shmem_get_int32_mem aclshmem_int32_get
#define shmem_get_int64_mem aclshmem_int64_get
#define shmem_get_uint8_mem aclshmem_uint8_get
#define shmem_get_uint16_mem aclshmem_uint16_get
#define shmem_get_uint32_mem aclshmem_uint32_get
#define shmem_get_uint64_mem aclshmem_uint64_get
#define shmem_get_char_mem aclshmem_char_get

#undef ACLSHMEM_TYPE_GET

#define ACLSHMEM_TYPE_GET_NBI(NAME, TYPE)                                                                             \
    /**                                                                                                               \
    * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the \
    * local PE.                                                                                                       \
    *                                                                                                                 \
    * @param dest               [in] Pointer on local device of the destination data.                                 \
    * @param source             [in] Pointer on Symmetric memory of the source data.                                  \
    * @param nelems             [in] Number of elements in the destination and source arrays.                         \
    * @param pe                 [in] PE number of the remote PE.                                                      \
    */                                                                                                                \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_get_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe)
    

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_GET_NBI);

#define shmem_get_float_mem_nbi aclshmem_float_get_nbi
#define shmem_get_double_mem_nbi aclshmem_double_get_nbi
#define shmem_get_int8_mem_nbi aclshmem_int8_get_nbi
#define shmem_get_int16_mem_nbi aclshmem_int16_get_nbi
#define shmem_get_int32_mem_nbi aclshmem_int32_get_nbi
#define shmem_get_int64_mem_nbi aclshmem_int64_get_nbi
#define shmem_get_uint8_mem_nbi aclshmem_uint8_get_nbi
#define shmem_get_uint16_mem_nbi aclshmem_uint16_get_nbi
#define shmem_get_uint32_mem_nbi aclshmem_uint32_get_nbi
#define shmem_get_uint64_mem_nbi aclshmem_uint64_get_nbi
#define shmem_get_char_mem_nbi aclshmem_char_get_nbi

#undef ACLSHMEM_TYPE_GET_NBI

#define ACLSHMEM_TYPENAME_P(NAME, TYPE)                                                     \
    /**                                                                                     \
    * @brief Provide a low latency put capability for single element of most basic types.   \
    *                                                                                       \
    * @param dst               [in] Symmetric address of the destination data on local PE.  \
    * @param value             [in] The element to be put.                                  \
    * @param pe                [in] The number of the remote PE.                            \
    */                                                                                      \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_p(TYPE* dst, const TYPE value, int pe)
    

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_P);
#define shmem_float_p aclshmem_float_p
#define shmem_double_p aclshmem_double_p
#define shmem_int8_p aclshmem_int8_p
#define shmem_int16_p aclshmem_int16_p
#define shmem_int32_p aclshmem_int32_p
#define shmem_int64_p aclshmem_int64_p
#define shmem_uint8_p aclshmem_uint8_p
#define shmem_uint16_p aclshmem_uint16_p
#define shmem_uint32_p aclshmem_uint32_p
#define shmem_uint64_p aclshmem_uint64_p
#define shmem_char_p aclshmem_char_p
#undef ACLSHMEM_TYPENAME_P

#define ACLSHMEM_TYPENAME_G(NAME, TYPE)                                                     \
    /**                                                                                     \
    * @brief Provide a low latency get single element of most basic types.                  \
    *                                                                                       \
    * @param src               [in] Symmetric address of the destination data on local PE.  \
    * @param pe                [in] The number of the remote PE.                            \
    * @return A single element of type specified in the input pointer.                      \
    */                                                                                      \
    ACLSHMEM_HOST_API TYPE aclshmem_##NAME##_g(TYPE* src, int32_t pe)
    

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_G);
#define shmem_float_g aclshmem_float_g
#define shmem_double_g aclshmem_double_g
#define shmem_int8_g aclshmem_int8_g
#define shmem_int16_g aclshmem_int16_g
#define shmem_int32_g aclshmem_int32_g
#define shmem_int64_g aclshmem_int64_g
#define shmem_uint8_g aclshmem_uint8_g
#define shmem_uint16_g aclshmem_uint16_g
#define shmem_uint32_g aclshmem_uint32_g
#define shmem_uint64_g aclshmem_uint64_g
#define shmem_char_g aclshmem_char_g
#undef ACLSHMEM_TYPENAME_G

/**
* @brief Synchronous interface. Copy contiguous data on symmetric memory from local PE to address on the specified PE.
*
* @param dst                [in] Pointer on Symmetric addr of local PE.
* @param src                [in] Pointer on local memory of the source data.
* @param elem_size          [in] size of elements in the destination and source addr.
* @param pe                 [in] PE number of the remote PE.
*/
ACLSHMEM_HOST_API void aclshmem_putmem(void* dst, void* src, size_t elem_size, int32_t pe);
#define shmem_putmem aclshmem_putmem


/**
* @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to
*        address on the local PE.
*
* @param dst                [in] Pointer on local device of the destination data.
* @param src                [in] Pointer on Symmetric memory of the source data.
* @param elem_size          [in] size of elements in the destination and source addr.
* @param pe                 [in] PE number of the remote PE.
*/
ACLSHMEM_HOST_API void aclshmem_getmem(void* dst, void* src, size_t elem_size, int32_t pe);
#define shmem_getmem aclshmem_getmem



/**
* @brief Asynchronous interface. Copy contiguous data on symmetric memory from local PE to address on the specified PE.
*
* @param dst                [in] Pointer on Symmetric addr of local PE.
* @param src                [in] Pointer on local memory of the source data.
* @param elem_size          [in] size of elements in the destination and source addr.
* @param pe                 [in] PE number of the remote PE.
*/
ACLSHMEM_HOST_API void aclshmem_putmem_nbi(void* dst, void* src, size_t elem_size, int32_t pe);
#define shmem_putmem_nbi aclshmem_putmem_nbi

/**
* @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to
*        address on the local PE.
*
* @param dst                [in] Pointer on local device of the destination data.
* @param src                [in] Pointer on Symmetric memory of the source data.
* @param elem_size          [in] size of elements in the destination and source addr.
* @param pe                 [in] PE number of the remote PE.
*/
ACLSHMEM_HOST_API void aclshmem_getmem_nbi(void* dst, void* src, size_t elem_size, int32_t pe);
#define shmem_getmem_nbi aclshmem_getmem_nbi

/**
    * @brief Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
    *
    * @param dst               [in] Pointer on local device of the destination data.
    * @param src               [in] Pointer on Symmetric memory of the source data.
    * @param elem_size         [in] Number of elements in the dest and source arrays.
    * @param pe                [in] PE number of the remote PE.
    * @param stream            [in] copy used stream(use default stream if stream == NULL).
 */
ACLSHMEM_HOST_API void aclshmemx_getmem_on_stream(void* dst, void* src, size_t elem_size,
                                            int32_t pe, aclrtStream stream);
#define shmemx_getmem_on_stream aclshmemx_getmem_on_stream

/**
    * @brief Copy contiguous data on local PE to symmetric address on the specified PE.
    *
    * @param dst               [in] Pointer on local device of the destination data.
    * @param src               [in] Pointer on Symmetric memory of the source data.
    * @param elem_size         [in] Number of elements in the dest and source arrays.
    * @param pe                [in] PE number of the remote PE.
    * @param stream            [in] copy used stream(use default stream if stream == NULL).
 */
ACLSHMEM_HOST_API void aclshmemx_putmem_on_stream(void* dst, void* src, size_t elem_size,
                                            int32_t pe, aclrtStream stream);
#define shmemx_putmem_on_stream aclshmemx_putmem_on_stream
/**
 * @brief Set necessary parameters for put or get.
 *
 * @param offset                [in] The start address on UB.
 * @param ub_size               [in] The Size of Temp UB Buffer.
 * @param event_id              [in] Sync ID for put or get.
 * @return Returns 0 on success or an error code on failure.
 */
ACLSHMEM_HOST_API int aclshmemx_set_mte_config(uint64_t offset, uint32_t ub_size, uint32_t event_id);
#define shmem_set_mte_config aclshmemx_set_mte_config

#ifdef __cplusplus
}
#endif

#endif