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
#ifndef SHMEM_HOST_LOG_H
#define SHMEM_HOST_LOG_H

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Set the log print function for the SHMEM library.
 *
 * @param func the logging function, takes level and msg as parameter
 * @return Returns 0 on success or an error code on failure
 */
ACLSHMEM_HOST_API int32_t aclshmemx_set_extern_logger(void (*func)(int level, const char *msg));
#define shmem_set_extern_logger aclshmemx_set_extern_logger


/**
 * @brief Set the logging level.
 *
 * @param level the logging level. 0-debug, 1-info, 2-warn, 3-error
 * @return Returns 0 on success or an error code on failure
 */
ACLSHMEM_HOST_API int32_t aclshmemx_set_log_level(int level);
#define shmem_set_log_level aclshmemx_set_log_level

#ifdef __cplusplus
}
#endif

#endif