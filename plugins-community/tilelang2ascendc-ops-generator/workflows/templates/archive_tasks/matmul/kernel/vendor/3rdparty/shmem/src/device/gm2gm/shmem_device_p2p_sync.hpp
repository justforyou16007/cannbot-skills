/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_P2P_SYNC_HPP
#define SHMEM_DEVICE_P2P_SYNC_HPP

#include "host_device/shmem_common_types.h"
#include "shmemi_device_p2p_sync.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"

ACLSHMEM_DEVICE int32_t aclshmemi_signal_wait_until_eq(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) != cmp_val);

    return ret;
}

ACLSHMEM_DEVICE int32_t aclshmemi_signal_wait_until_ne(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) == cmp_val);

    return ret;
}

ACLSHMEM_DEVICE int32_t aclshmemi_signal_wait_until_gt(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) <= cmp_val);

    return ret;
}

ACLSHMEM_DEVICE int32_t aclshmemi_signal_wait_until_ge(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) < cmp_val);

    return ret;
}

ACLSHMEM_DEVICE int32_t aclshmemi_signal_wait_until_lt(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) >= cmp_val);

    return ret;
}

ACLSHMEM_DEVICE int32_t aclshmemi_signal_wait_until_le(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) > cmp_val);

    return ret;
}

ACLSHMEM_DEVICE int32_t aclshmemi_signal_wait_until(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val)
{
    switch (cmp) {
        case ACLSHMEM_CMP_EQ:
            return aclshmemi_signal_wait_until_eq(sig_addr, cmp_val);
        case ACLSHMEM_CMP_NE:
            return aclshmemi_signal_wait_until_ne(sig_addr, cmp_val);
        case ACLSHMEM_CMP_GT:
            return aclshmemi_signal_wait_until_gt(sig_addr, cmp_val);
        case ACLSHMEM_CMP_GE:
            return aclshmemi_signal_wait_until_ge(sig_addr, cmp_val);
        case ACLSHMEM_CMP_LT:
            return aclshmemi_signal_wait_until_lt(sig_addr, cmp_val);
        case ACLSHMEM_CMP_LE:
            return aclshmemi_signal_wait_until_le(sig_addr, cmp_val);
    }
    return -1;
}

ACLSHMEM_DEVICE int32_t aclshmemi_signal_fetch(__gm__ int32_t *sig_addr)
{
    dcci_cacheline((__gm__ uint8_t *)sig_addr);
    return *sig_addr;
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_wait_until(__gm__ T *sig_addr, int cmp, T cmp_val)
{
    switch (cmp) {
        case ACLSHMEM_CMP_EQ:
            do {
                dcci_cacheline((__gm__ uint8_t *)sig_addr);
            } while (!(*sig_addr == cmp_val));
            break;
        case ACLSHMEM_CMP_NE:
            do {
                dcci_cacheline((__gm__ uint8_t *)sig_addr);
            } while (!(*sig_addr != cmp_val));
            break;
        case ACLSHMEM_CMP_GT:
            do {
                dcci_cacheline((__gm__ uint8_t *)sig_addr);
            } while (!(*sig_addr > cmp_val));
            break;
        case ACLSHMEM_CMP_GE:
            do {
                dcci_cacheline((__gm__ uint8_t *)sig_addr);
            } while (!(*sig_addr >= cmp_val));
            break;
        case ACLSHMEM_CMP_LT:
            do {
                dcci_cacheline((__gm__ uint8_t *)sig_addr);
            } while (!(*sig_addr < cmp_val));
            break;
        case ACLSHMEM_CMP_LE:
            do {
                dcci_cacheline((__gm__ uint8_t *)sig_addr);
            } while (!(*sig_addr <= cmp_val));
            break;
    }
}

template <typename T>
ACLSHMEM_DEVICE int aclshmemi_test(__gm__ T *sig_addr, int cmp, T cmp_val)
{
    dcci_cacheline((__gm__ uint8_t *)sig_addr);
    switch (cmp) {
        case ACLSHMEM_CMP_EQ:
            return *sig_addr == cmp_val;
        case ACLSHMEM_CMP_NE:
            return *sig_addr != cmp_val;
        case ACLSHMEM_CMP_GT:
            return *sig_addr > cmp_val;
        case ACLSHMEM_CMP_GE:
            return *sig_addr >= cmp_val;
        case ACLSHMEM_CMP_LT:
            return *sig_addr < cmp_val;
        case ACLSHMEM_CMP_LE:
            return *sig_addr <= cmp_val;
    }
    return 0;
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_wait_until_all(__gm__ T *sig_addr, size_t nelems, const int *status, int cmp, T cmp_val,
                                        int stride)
{
    for (int i = 0; i < nelems; i++) {
        if (status && status[i] != 0) {
            continue;
        }

        aclshmemi_wait_until(sig_addr + i * stride, cmp, cmp_val);
    }
}

template <typename T>
ACLSHMEM_DEVICE int aclshmemi_test_all(__gm__ T *sig_addr, size_t nelems, const int *status, int cmp, T cmp_val,
                                 int stride)
{
    int ret = 1;

    for (int i = 0; i < nelems; i++) {
        if (status && status[i] != 0) {
            continue;
        }

        dcci_cacheline((__gm__ uint8_t *)(sig_addr + i * stride));
        switch (cmp) {
            case ACLSHMEM_CMP_EQ:
                ret &= (*(sig_addr + i * stride) == cmp_val);
                break;
            case ACLSHMEM_CMP_NE:
                ret &= (*(sig_addr + i * stride) != cmp_val);
                break;
            case ACLSHMEM_CMP_GT:
                ret &= (*(sig_addr + i * stride) > cmp_val);
                break;
            case ACLSHMEM_CMP_GE:
                ret &= (*(sig_addr + i * stride) >= cmp_val);
                break;
            case ACLSHMEM_CMP_LT:
                ret &= (*(sig_addr + i * stride) < cmp_val);
                break;
            case ACLSHMEM_CMP_LE:
                ret &= (*(sig_addr + i * stride) <= cmp_val);
                break;
        }
    }

    return ret;
}

#ifdef __cplusplus
extern "C" {
#endif

ACLSHMEM_DEVICE int32_t aclshmem_signal_wait_until(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val)
{
    return aclshmemi_signal_wait_until(sig_addr, cmp, cmp_val);
}

#ifdef __cplusplus
}
#endif

#endif