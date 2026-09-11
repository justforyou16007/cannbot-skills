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

/*
    WARNING：

    Our barrier implementation ensures that:
        On systems with only HCCS: All operations of all pes of a team ON EXECUTING/INTERNAL STREAMs
        before the barrier are visiable to all pes of the team after the barrier.

    Refer to shmem_device_sync.h for using restrictions.
*/

#ifndef SHMEM_HOST_SYNC_H
#define SHMEM_HOST_SYNC_H

#include "acl/acl.h"
#include "host/shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn ACLSHMEM_HOST_API uint64_t util_get_ffts_config(void)
 * @brief Get runtime ffts config. This config should be passed to MIX Kernel and set by MIX Kernel
 * using aclshmemx_set_ffts. Refer to aclshmemx_set_ffts for more details.
 *
 * @return ffts config
 */
ACLSHMEM_HOST_API uint64_t util_get_ffts_config(void);
#define shmemx_get_ffts_config util_get_ffts_config

/**
 * @fn ACLSHMEM_HOST_API void aclshmemx_handle_wait(aclshmem_handle_t handle, aclrtStream stream)
 * @brief Wait asynchronous RMA operations to finish.
 *
 * @param handle              [in] handle use to wait asynchronous RMA operations to finish
 * @param stream              [in] specifed stream to do wait
 */
ACLSHMEM_HOST_API void aclshmemx_handle_wait(aclshmem_handle_t handle, aclrtStream stream);
#define shmem_handle_wait aclshmemx_handle_wait

/**
 * @brief This routine can be used to implement point-to-point synchronization between PEs or between threads within
 *        the same PE. A call to this routine blocks until the value of sig_addr at the calling PE satisfies the wait
 *        condition specified by the comparison operator, cmp, and comparison value, cmp_val.
 *
 * @param sig_addr              [in] Local address of the source signal variable.
 * @param cmp                   [in] The comparison operator that compares sig_addr with cmp_val. Supported operators:
 *                                   ACLSHMEM_CMP_EQ/ACLSHMEM_CMP_NE/ACLSHMEM_CMP_GT/ACLSHMEM_CMP_GE/ACLSHMEM_CMP_LT/ACLSHMEM_CMP_LE.
 * @param cmp_val               [in] The value against which the object pointed to by sig_addr will be compared.
 * @param stream                [in] Used stream(use default stream if stream == NULL).
 */
ACLSHMEM_HOST_API void aclshmemx_signal_wait_until_on_stream(int32_t *sig_addr, int cmp, int32_t cmp_val, aclrtStream stream);
#define shmem_signal_wait_until_on_stream aclshmemx_signal_wait_until_on_stream

#ifdef __cplusplus
}
#endif

#endif