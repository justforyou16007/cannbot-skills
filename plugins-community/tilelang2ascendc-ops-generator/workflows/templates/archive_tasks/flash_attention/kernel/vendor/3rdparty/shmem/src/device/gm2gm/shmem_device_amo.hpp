/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_AMO_HPP
#define SHMEM_DEVICE_AMO_HPP

#include "kernel_operator.h"
#include "shmemi_device_cc.h"

#define ACLSHMEM_ATOMIC_ADD_TYPENAME(NAME, TYPE, ATOMIC_TYPE)                                                      \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE *dst, TYPE value, int32_t pe)                    \
    {                                                                                                              \
        /* ROCE */                                                                                                 \
        /* RDMA */                                                                                                 \
        /* MTE  */                                                                                                 \
        dcci_atomic();                                                                                             \
        dsb_all();                                                                                                 \
        set_st_atomic_cfg(ATOMIC_TYPE, ATOMIC_SUM);                                                                \
        st_atomic<TYPE>(value, (__gm__ TYPE *)aclshmem_ptr(dst, pe));                                              \
        dcci_atomic();                                                                                             \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_INT(ACLSHMEM_ATOMIC_ADD_TYPENAME);

#define ACLSHMEM_ATOMIC_ADD_TYPENAME_FLOAT(NAME, TYPE, ATOMIC_TYPE)                                                \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE *dst, TYPE value, int32_t pe)                    \
    {                                                                                                              \
        /* ROCE */                                                                                                 \
        /* RDMA */                                                                                                 \
        /* MTE  */                                                                                                 \
        dcci_atomic();                                                                                             \
        dsb_all();                                                                                                 \
        set_st_atomic_cfg(ATOMIC_TYPE, ATOMIC_SUM);                                                                \
        st_atomic(value, (__gm__ TYPE *)aclshmem_ptr(dst, pe));                                                    \
        dcci_atomic();                                                                                             \
    }

ACLSHMEM_TYPE_FUNC_ATOMIC_FLOAT(ACLSHMEM_ATOMIC_ADD_TYPENAME_FLOAT);

#endif