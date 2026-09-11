/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_GM2GM_RMA_HPP
#define SHMEM_DEVICE_GM2GM_RMA_HPP

#include "kernel_operator.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "device/shmem_def.h"
#include "gm2gm/engine/shmemi_device_rdma.h"
#include "host/shmem_host_def.h"
#include "shmemi_device_rma.h"

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

#define ACLSHMEM_TYPENAME_P_AICORE(NAME, TYPE)                                              \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_p(__gm__ TYPE *dst, const TYPE value, int pe)    \
    {                                                                                       \
        auto ptr = aclshmem_ptr(dst, pe);                                                   \
        __gm__ TYPE *addr_gm = reinterpret_cast<__gm__ TYPE *>(ptr);                        \
                                                                                            \
        *addr_gm = value;                                                                   \
        dcci_cacheline((__gm__ uint8_t *)addr_gm);                                          \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_P_AICORE);

#define ACLSHMEM_TYPENAME_G_AICORE(NAME, TYPE)                                              \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_g(__gm__ TYPE *src, int32_t pe)                  \
    {                                                                                       \
        auto ptr = aclshmem_ptr(src, pe);                                                   \
        __gm__ TYPE *addr_gm = reinterpret_cast<__gm__ TYPE *>(ptr);                        \
                                                                                            \
        dcci_cacheline((__gm__ uint8_t *)addr_gm);                                          \
        return *addr_gm;                                                                    \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_G_AICORE);

ACLSHMEM_DEVICE void aclshmem_getmem(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)
{
    /* ROCE */
    /* RDMA */
    /* MTE  */
    /* Global State Get */
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
    aclshmemx_mte_get_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
    aclshmem_quiet();
}

#define ACLSHMEM_GET_TYPENAME_MEM(NAME, TYPE)                                                                           \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe)      \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                      \
        /* CopyUB Config Set */                                                                                         \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                        \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                       \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                         \
        aclshmemx_mte_get_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,        \
                              copy_event_id);                                                                           \
        aclshmem_quiet();                                                                                               \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM);

ACLSHMEM_DEVICE void aclshmem_putmem(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)
{
    /* ROCE */
    /* RDMA */
    /* MTE  */
    /* Global State Get */
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
    aclshmemx_mte_put_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
    aclshmem_quiet();
}

#define ACLSHMEM_PUT_TYPENAME_MEM(NAME, TYPE)                                                                           \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe)      \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                      \
        /* CopyUB Config Set */                                                                                         \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                        \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                       \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                         \
        aclshmemx_mte_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,        \
                              copy_event_id);                                                                           \
        aclshmem_quiet();                                                                                               \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM);

ACLSHMEM_DEVICE void aclshmem_getmem_nbi(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)
{
    /* ROCE */
    /* RDMA */
    /* MTE  */
    /* Global State Get */
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
    aclshmemx_mte_get_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
}

#define ACLSHMEM_GET_TYPENAME_MEM_NBI(NAME, TYPE)                                                                           \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe)      \
    {                                                                                                                       \
        /* Global State Get */                                                                                              \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                          \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                                         \
            /* MTE  */                                                                                                      \
            /* CopyUB Config Set */                                                                                         \
            uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                        \
            uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                       \
            AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                         \
            aclshmemx_mte_get_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,        \
                                    copy_event_id);                                                                         \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                                 \
            /* RoCE */                                                                                                      \
            auto ptr = aclshmem_roce_ptr(src, pe);                                                                          \
            if (ptr == nullptr) return;                                                                                     \
            /* Create LocalTensor */                                                                                        \
            AscendC::LocalTensor<uint32_t> ub_tensor_32;                                                                    \
            ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                              \
            ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);             \
            ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;                                                                  \
            AscendC::LocalTensor<uint64_t> ub_tensor_64;                                                                    \
            ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                              \
            ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR               \
                                                                             + UB_ALIGN_SIZE);                              \
            ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;                                                                  \
            aclshmemi_roce_read((__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(TYPE),                \
                                ub_tensor_64, ub_tensor_32);                                                                \
        }                                                                                                                   \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_NBI);

#define ACLSHMEM_GET_TYPENAME_MEM_DETAILED_NBI(NAME, TYPE)                                                              \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(__gm__ TYPE *dst, __gm__ TYPE *src,                                  \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe)              \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                      \
        /* CopyUB Config Set */                                                                                         \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                        \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                       \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                         \
        aclshmemx_mte_get_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, copy_params, pe,      \
                              copy_event_id);                                                                           \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_DETAILED_NBI);

#define ACLSHMEM_GET_TYPENAME_MEM_TENSOR_NBI(NAME, TYPE)                                                                 \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                 uint32_t elem_size, int pe)                                             \
    {                                                                                                                    \
        /* Global State Get */                                                                                           \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                       \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                                      \
            /* MTE  */                                                                                                   \
            /* CopyUB Config Set */                                                                                      \
            uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                     \
            /* Create LocalTensor */                                                                                     \
            AscendC::LocalTensor<TYPE> ub_tensor;                                                                        \
            ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                               \
            ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                         \
            ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                               \
            AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                      \
            aclshmemx_mte_get_nbi(dst, src, ub_tensor, elem_size, pe, copy_event_id);                                    \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                              \
            /* RoCE */                                                                                                   \
            auto ptr = aclshmem_roce_ptr((__gm__ void *)src.GetPhyAddr(), pe);                                           \
            if (ptr == nullptr) return;                                                                                  \
            /* Create LocalTensor */                                                                                     \
            AscendC::LocalTensor<uint32_t> ub_tensor_32;                                                                 \
            ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                           \
            ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);          \
            ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;                                                               \
            AscendC::LocalTensor<uint64_t> ub_tensor_64;                                                                 \
            ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                           \
            ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR            \
                                                                            + UB_ALIGN_SIZE);                            \
            ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;                                                               \
            aclshmemi_roce_read((__gm__ uint8_t*)(dst.GetPhyAddr()), (__gm__ uint8_t*)ptr, pe, 0,                        \
                                elem_size * sizeof(TYPE), ub_tensor_64, ub_tensor_32);                                   \
        }                                                                                                                \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_TENSOR_NBI);

#define ACLSHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED_NBI(NAME, TYPE)                                                        \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_get_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                 const non_contiguous_copy_param &copy_params, int pe)                   \
    {                                                                                                                    \
        /* ROCE */                                                                                                       \
        /* RDMA */                                                                                                       \
        /* MTE  */                                                                                                       \
        /* Global State Get */                                                                                           \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                       \
        /* CopyUB Config Set */                                                                                          \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                         \
        /* Create LocalTensor */                                                                                         \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                            \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                   \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                             \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                                   \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                          \
        aclshmemx_mte_get_nbi(dst, src, ub_tensor, copy_params, pe, copy_event_id);                                      \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_NBI(NAME, TYPE)                                                                           \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe)      \
    {                                                                                                                       \
        /* Global State Get */                                                                                              \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                          \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                                         \
            /* MTE  */                                                                                                      \
            /* CopyUB Config Set */                                                                                         \
            uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                        \
            uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                       \
            AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                         \
            aclshmemx_mte_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, elem_size, pe,        \
                                    copy_event_id);                                                                         \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                                 \
            /* RoCE */                                                                                                      \
            auto ptr = aclshmem_roce_ptr(dst, pe);                                                                          \
            if (ptr == nullptr) return;                                                                                     \
            /* Create LocalTensor */                                                                                        \
            AscendC::LocalTensor<uint32_t> ub_tensor_32;                                                                    \
            ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                              \
            ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);             \
            ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;                                                                  \
            AscendC::LocalTensor<uint64_t> ub_tensor_64;                                                                    \
            ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                              \
            ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR               \
                                                                            + UB_ALIGN_SIZE);                               \
            ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;                                                                  \
            aclshmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, 0, elem_size * sizeof(TYPE),               \
                                    ub_tensor_64, ub_tensor_32);                                                            \
        }                                                                                                                   \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_DETAILED_NBI(NAME, TYPE)                                                            \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(__gm__ TYPE *dst, __gm__ TYPE *src,                                \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe)            \
    {                                                                                                                 \
        /* ROCE */                                                                                                    \
        /* RDMA */                                                                                                    \
        /* MTE  */                                                                                                    \
        /* Global State Get */                                                                                        \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                    \
        /* CopyUB Config Set */                                                                                       \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                      \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                     \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                       \
        aclshmemx_mte_put_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE *>(copy_ub), copy_ub_size, copy_params, pe,    \
                              copy_event_id);                                                                         \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_DETAILED_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_NBI(NAME, TYPE)                                                                 \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                 uint32_t elem_size, int pe)                                             \
    {                                                                                                                    \
        /* Global State Get */                                                                                           \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                       \
        if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE) {                                                      \
            /* MTE  */                                                                                                   \
            /* CopyUB Config Set */                                                                                      \
            uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                     \
            /* Create LocalTensor */                                                                                     \
            AscendC::LocalTensor<TYPE> ub_tensor;                                                                        \
            ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                               \
            ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                         \
            ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                               \
            AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                      \
            aclshmemx_mte_put_nbi(dst, src, ub_tensor, elem_size, pe, copy_event_id);                                    \
        } else if (device_state->topo_list[pe] & ACLSHMEM_TRANSPORT_ROCE) {                                              \
            /* RoCE */                                                                                                   \
            auto ptr = aclshmem_roce_ptr((__gm__ void *)dst.GetPhyAddr(), pe);                                           \
            if (ptr == nullptr) return;                                                                                  \
            /* Create LocalTensor */                                                                                     \
            AscendC::LocalTensor<uint32_t> ub_tensor_32;                                                                 \
            ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                           \
            ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);          \
            ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;                                                               \
            AscendC::LocalTensor<uint64_t> ub_tensor_64;                                                                 \
            ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);                           \
            ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR            \
                                                                            + UB_ALIGN_SIZE);                            \
            ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;                                                               \
            aclshmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)(src.GetPhyAddr()), pe, 0,                       \
                                                    elem_size * sizeof(TYPE), ub_tensor_64, ub_tensor_32);               \
        }                                                                                                                \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED_NBI(NAME, TYPE)                                                        \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_put_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,     \
                                                 const non_contiguous_copy_param &copy_params, int pe)                   \
    {                                                                                                                    \
        /* ROCE */                                                                                                       \
        /* RDMA */                                                                                                       \
        /* MTE  */                                                                                                       \
        /* Global State Get */                                                                                           \
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();                                       \
        /* CopyUB Config Set */                                                                                          \
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;                                                         \
        /* Create LocalTensor */                                                                                         \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                            \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                   \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                             \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                                   \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                          \
        aclshmemx_mte_put_nbi(dst, src, ub_tensor, copy_params, pe, copy_event_id);                                      \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED_NBI);

ACLSHMEM_DEVICE void aclshmem_putmem_nbi(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)
{
    /* MTE  */
    /* Global State Get */
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
    aclshmemx_mte_put_nbi(reinterpret_cast<__gm__ char *>(dst), reinterpret_cast<__gm__ char *>(src),
                          reinterpret_cast<__ubuf__ char *>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);
}

#define ACLSHMEM_TEST(NAME, TYPE)                                                                                \
    ACLSHMEM_DEVICE int aclshmem_##NAME##_test(__gm__ TYPE *ivar, int cmp, TYPE cmp_value)                       \
    {                                                                                                            \
        aclshmem_fence();                                                                                        \
        return aclshmemi_test(ivar, cmp, cmp_value);                                                             \
    }

ACLSHMEM_TEST_TYPE_FUNC(ACLSHMEM_TEST);

// Set Memcpy Interfaces necessary UB Buffer.
ACLSHMEM_DEVICE void aclshmemx_set_mte_config(uint64_t offset, uint32_t ub_size, uint32_t event_id)
{
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    
    device_state->mte_config.aclshmem_ub = offset;
    device_state->mte_config.ub_size = ub_size;
    device_state->mte_config.event_id = event_id;
}

#endif