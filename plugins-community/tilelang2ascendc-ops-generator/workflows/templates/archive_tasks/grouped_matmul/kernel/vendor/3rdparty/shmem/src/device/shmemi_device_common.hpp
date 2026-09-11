/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_COMMON_HPP
#define SHMEM_DEVICE_COMMON_HPP

#include "shmemi_def.h"

#ifdef BACKEND_HYBM
#include "shmemi_device_meta.h"
ACLSHMEM_DEVICE __gm__ aclshmem_device_host_state_t *aclshmemi_get_state() {
    return reinterpret_cast<__gm__ aclshmem_device_host_state_t *>(aclshmemi_get_extra_context_addr(0));
}
#else

// rdma
constexpr uint64_t SMEM_SHM_DEVICE_PRE_META_SIZE = 128UL; // 128B
constexpr uint64_t SMEM_SHM_DEVICE_GLOBAL_META_SIZE = SMEM_SHM_DEVICE_PRE_META_SIZE; // 128B
constexpr uint64_t SMEM_OBJECT_NUM_MAX = 511UL; // entity最大数量
constexpr uint64_t SMEM_SHM_DEVICE_META_SIZE = SMEM_SHM_DEVICE_PRE_META_SIZE * SMEM_OBJECT_NUM_MAX
                                        + SMEM_SHM_DEVICE_GLOBAL_META_SIZE; // 64K

constexpr uint64_t SMEM_SHM_DEVICE_USER_CONTEXT_PRE_SIZE = 64UL * 1024UL; // 64K
constexpr uint64_t SMEM_SHM_DEVICE_INFO_SIZE = SMEM_SHM_DEVICE_USER_CONTEXT_PRE_SIZE * SMEM_OBJECT_NUM_MAX
                                        + SMEM_SHM_DEVICE_META_SIZE; // 元数据+用户context,总大小32M, 对齐2M
constexpr uint64_t SMEM_SHM_DEVICE_META_ADDR = SVM_END_ADDR - SMEM_SHM_DEVICE_INFO_SIZE;

ACLSHMEM_DEVICE __gm__ aclshmem_device_host_state_t *aclshmemi_get_state() {
    return reinterpret_cast<__gm__ aclshmem_device_host_state_t *>((__gm__ void*)(SVM_END_ADDR - GLOBAL_STATE_SIZE));
}
#endif

ACLSHMEM_DEVICE int aclshmemi_get_my_pe()
{
    return aclshmemi_get_state()->mype;
}

ACLSHMEM_DEVICE int aclshmemi_get_total_pe()
{
    return aclshmemi_get_state()->npes;
}

ACLSHMEM_DEVICE uint64_t aclshmemi_get_heap_size()
{
    return aclshmemi_get_state()->heap_size;
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_store(__gm__ T *addr, T val)
{
    *((__gm__ T *)addr) = val;
}

template<typename T>
ACLSHMEM_DEVICE T aclshmemi_load(__gm__ T *cache)
{
    return *((__gm__ T *)cache);
}

#endif