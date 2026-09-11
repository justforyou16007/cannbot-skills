/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_GM2GM_MTE_HPP
#define SHMEM_DEVICE_GM2GM_MTE_HPP

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "ub2gm/mte/shmemi_device_mte.h"
#include "shmemi_device_common.h"

ACLSHMEM_DEVICE bool is_host_mem_heap(__gm__ void *ptr)
{
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    if (device_state->host_heap_base == nullptr) {
        return false;
    }
    uint64_t ptr_uint = reinterpret_cast<uint64_t>(ptr);
    uint64_t base_uint = reinterpret_cast<uint64_t>(device_state->host_heap_base);
    uint64_t end_uint = base_uint + device_state->heap_size;
    return (ptr_uint >= base_uint && ptr_uint <= end_uint);
}

ACLSHMEM_DEVICE __gm__ void *aclshmem_ptr(__gm__ void *ptr, int pe)
{
    // Get Global State
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    if (is_host_mem_heap(ptr)) {
        uint64_t host_heap_offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(device_state->host_heap_base);
        uint64_t remote_host_ptr = reinterpret_cast<uint64_t>(device_state->p2p_host_heap_base[pe]) + host_heap_offset;
        return reinterpret_cast<__gm__ void *>(remote_host_ptr);
    }
    // Back to root address
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(device_state->heap_base);
    // Address translate
    uint64_t remote_ptr = reinterpret_cast<uint64_t>(device_state->p2p_device_heap_base[pe]) + offset;
    return reinterpret_cast<__gm__ void *>(remote_ptr);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                        uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = aclshmem_ptr(src, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    // block_size: dataMove Unit
    uint64_t block_size = ub_size / sizeof(T) * sizeof(T);
    uint64_t remain = (elem_size * sizeof(T)) % block_size;

    uint64_t repeat_times = (elem_size * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        aclshmemi_copy_gm2ub(buf, remote_ptr + i * repeat_elem, block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        aclshmemi_copy_ub2gm(dst + i * repeat_elem, buf, block_size);
        if (i != loop_times - 1) {  // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        aclshmemi_copy_gm2ub(buf, remote_ptr + repeat_times * repeat_elem, remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        aclshmemi_copy_ub2gm(dst + repeat_times * repeat_elem, buf, remain);
    }
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                        const non_contiguous_copy_param &copy_params, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = aclshmem_ptr(src, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    AscendC::GlobalTensor<T> src_tensor;
    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> dst_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(remote_ptr));
    dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst));

    uint64_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint64_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    aclshmemi_copy_gm2ub(ub_tensor, src_tensor, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    aclshmemi_copy_ub2gm(dst_tensor, ub_tensor, data_copy_params_ub2gm);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                        AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = aclshmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    // block_size: dataMove Unit
    uint64_t block_size = buf.GetSize() * sizeof(T);
    uint64_t remain = (elem_size * sizeof(T)) % block_size;

    uint64_t repeat_times = (elem_size * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        aclshmemi_copy_gm2ub(buf, remote_buff[i * repeat_elem], block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        aclshmemi_copy_ub2gm(dst[i * repeat_elem], buf, block_size);
        if (i != loop_times - 1) {  // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        aclshmemi_copy_gm2ub(buf, remote_buff[repeat_times * repeat_elem], remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        aclshmemi_copy_ub2gm(dst[repeat_times * repeat_elem], buf, remain);
    }
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                        AscendC::LocalTensor<T> buf, const non_contiguous_copy_param &copy_params,
                                        int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = aclshmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    uint64_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint64_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    aclshmemi_copy_gm2ub(buf, remote_buff, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    aclshmemi_copy_ub2gm(dst, buf, data_copy_params_ub2gm);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                        uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = aclshmem_ptr(dst, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    // block_size: dataMove Unit
    uint64_t block_size = ub_size / sizeof(T) * sizeof(T);
    uint64_t remain = (elem_size * sizeof(T)) % block_size;

    uint64_t repeat_times = (elem_size * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        aclshmemi_copy_gm2ub(buf, src + i * repeat_elem, block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        aclshmemi_copy_ub2gm(remote_ptr + i * repeat_elem, buf, block_size);
        if (i != loop_times - 1) {  // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        aclshmemi_copy_gm2ub(buf, src + repeat_times * repeat_elem, remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        aclshmemi_copy_ub2gm(remote_ptr + repeat_times * repeat_elem, buf, remain);
    }
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                        const non_contiguous_copy_param &copy_params, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = aclshmem_ptr(dst, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    AscendC::GlobalTensor<T> src_tensor;
    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> dst_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src));
    dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(remote_ptr));

    uint64_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint64_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    aclshmemi_copy_gm2ub(ub_tensor, src_tensor, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    aclshmemi_copy_ub2gm(dst_tensor, ub_tensor, data_copy_params_ub2gm);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                        AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = aclshmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    // block_size: dataMove Unit
    uint64_t block_size = buf.GetSize() * sizeof(T);
    uint64_t remain = (elem_size * sizeof(T)) % block_size;

    uint64_t repeat_times = (elem_size * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        aclshmemi_copy_gm2ub(buf, src[i * repeat_elem], block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        aclshmemi_copy_ub2gm(remote_buff[i * repeat_elem], buf, block_size);
        if (i != loop_times - 1) {  // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        aclshmemi_copy_gm2ub(buf, src[repeat_times * repeat_elem], remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        aclshmemi_copy_ub2gm(remote_buff[repeat_times * repeat_elem], buf, remain);
    }
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                        AscendC::LocalTensor<T> buf, const non_contiguous_copy_param &copy_params,
                                        int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = aclshmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    uint64_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint64_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    aclshmemi_copy_gm2ub(buf, src, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    aclshmemi_copy_ub2gm(remote_buff, buf, data_copy_params_ub2gm);
}

ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(__gm__ int8_t* dst, __gm__ int8_t* src, __ubuf__ int8_t* buf,
    uint32_t ub_size, uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID, bool enable_L2)
{
    auto ptr = aclshmem_ptr(src, pe);
    __gm__ int8_t* remote_ptr = reinterpret_cast<__gm__ int8_t*>(ptr);

    // block_size: dataMove Unit
    uint64_t block_size = ub_size;
    uint64_t remain = (elem_size) % block_size;

    uint64_t repeat_times = (elem_size) / block_size;
    uint64_t repeat_elem = block_size;
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        aclshmemi_copy_gm2ub(buf, remote_ptr + i * repeat_elem, block_size, enable_L2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        aclshmemi_copy_ub2gm(dst + i * repeat_elem, buf, block_size, enable_L2);
        if (i != loop_times - 1) {      // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        aclshmemi_copy_gm2ub(buf, remote_ptr + repeat_times * repeat_elem, remain, enable_L2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        aclshmemi_copy_ub2gm(dst + repeat_times * repeat_elem, buf, remain, enable_L2);
    }
}

ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(__gm__ int8_t* dst, __gm__ int8_t* src, __ubuf__ int8_t* buf,
    uint32_t ub_size, uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID, bool enable_L2)
{
    auto ptr = aclshmem_ptr(dst, pe);
    __gm__ int8_t* remote_ptr = reinterpret_cast<__gm__ int8_t*>(ptr);

    // block_size: dataMove Unit
    uint64_t block_size = ub_size;
    uint64_t remain = (elem_size) % block_size;

    uint64_t repeat_times = (elem_size) / block_size;
    uint64_t repeat_elem = block_size;
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        aclshmemi_copy_gm2ub(buf, src + i * repeat_elem, block_size, enable_L2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        aclshmemi_copy_ub2gm(remote_ptr + i * repeat_elem, buf, block_size, enable_L2);
        if (i != loop_times - 1) {      // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        aclshmemi_copy_gm2ub(buf, src + repeat_times * repeat_elem, remain, enable_L2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        aclshmemi_copy_ub2gm(remote_ptr + repeat_times * repeat_elem, buf, remain, enable_L2);
    }
}

ACLSHMEM_DEVICE void aclshmemx_mte_get_nbi(__gm__ int8_t* dst, __gm__ int8_t* src, uint32_t elem_size,
    int32_t pe, bool enable_L2)
{
    /* Global State Get */
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
    aclshmemx_mte_get_nbi(dst, src, reinterpret_cast<__ubuf__ int8_t*>(copy_ub), copy_ub_size,
        elem_size, pe, copy_event_id, enable_L2);
}

ACLSHMEM_DEVICE void aclshmemx_mte_put_nbi(__gm__ int8_t* dst, __gm__ int8_t* src, uint32_t elem_size, int32_t pe,
    bool enable_L2)
{
        /* Global State Get */
        __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
        /* CopyUB Config Set */
        uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
        uint32_t copy_ub_size = device_state->mte_config.ub_size;
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
        aclshmemx_mte_put_nbi(dst, src, reinterpret_cast<__ubuf__ int8_t*>(copy_ub), copy_ub_size,
            elem_size, pe, copy_event_id, enable_L2);
}

#endif