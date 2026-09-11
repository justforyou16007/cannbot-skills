/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_RDMA_H
#define SHMEMI_DEVICE_RDMA_H

#include "kernel_operator.h"
#include "device/shmem_def.h"

enum class ACLSHMEMAIVOPCODE : uint32_t {
    OP_SEND = 0,
    OP_SEND_WITH_INV,
    OP_SEND_WITH_IMM,
    OP_RDMA_WRITE,
    OP_RDMA_WRITE_WITH_IMM,
    OP_RDMA_READ
};

struct ACLSHMEMAIVRDMAInfo {
    uint32_t qpNum; // number of QP per connection
    uint64_t sqPtr; // pointer to send queue address array of size [PE_NUM][qpNum]
    uint64_t rqPtr; // pointer to receive queue address array of size [PE_NUM][qpNum]
    uint64_t scqPtr; // pointer to send completion queue address array of size [PE_NUM][qpNum]
    uint64_t rcqPtr; // pointer to receive completion queue address array of size [PE_NUM][qpNum]
    uint64_t memPtr; // pointer to memory region array of size [MAX_PE_NUM]
};

struct ACLSHMEMmemInfo {
    uint64_t size; // size of the memory region
    uint64_t addr; // start address of the memory region
    uint32_t lkey; // local key of the memory region
    uint32_t rkey; // remote key of the memory region
};

enum class ACLSHMEMDBMode : int32_t { INVALID_DB = -1, HW_DB = 0, SW_DB };

struct ACLSHMEMWQCtx {
    uint32_t wqn; // work queue number
    uint64_t bufAddr; // start address of ring buffer
    uint32_t wqeSize; // size of each WQE
    uint32_t depth; // depth of ring buffer
    uint64_t headAddr; // work queue head (Producer Index) address
    uint64_t tailAddr; // work queue tail (Consumer Index) address
    ACLSHMEMDBMode dbMode;
    uint64_t dbAddr; // doorbell address
    uint32_t sl; // service level
};

struct ACLSHMEMCQCtx {
    uint32_t cqn; // completion queue number
    uint64_t bufAddr; // start address of ring buffer
    uint32_t cqeSize; // size of each CQE
    uint32_t depth; // depth of ring buffer
    uint64_t headAddr; // work queue head (Producer Index) address
    uint64_t tailAddr; // work queue tail (Consumer Index) address
    ACLSHMEMDBMode dbMode;
    uint64_t dbAddr; // doorbell address
};

struct ACLSHMEMwqeCtx {
    uint32_t byte4;
    uint32_t msgLen;
    uint32_t immtdata;
    uint32_t byte16;
    uint32_t byte20;
    uint32_t rkey;
    uint64_t va;
};

struct ACLSHMEMsegCtx {
    uint32_t len;
    uint32_t lkey;
    uint64_t addr;
};

struct ACLSHMEMcqeCtx {
    uint32_t byte4;
    uint32_t immtdata;
    uint32_t byte12;
    uint32_t byte16;
    uint32_t byteCnt;
    uint32_t smac;
    uint32_t byte28;
    uint32_t byte32;
};

struct ACLSHMEMHybmDeviceMeta {
    uint32_t entityId;
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t extraContextSize;
    uint64_t symmetricSize;
    uint64_t qpInfoAddress;
    uint64_t reserved[12];  // total 128B, equal HYBM_DEVICE_PRE_META_SIZE
};

ACLSHMEM_DEVICE __gm__ ACLSHMEMAIVRDMAInfo* aclshmemi_qp_info_fetch();

ACLSHMEM_DEVICE void aclshmemi_roce_poll_cq_update_info(AscendC::LocalTensor<uint64_t> &ubLocal64,
    AscendC::LocalTensor<uint32_t> &ubLocal32, uint32_t &curTail, uint32_t &rRankId, uint32_t &qpIdx);

ACLSHMEM_DEVICE void aclshmemi_rdma_post_send_update_info(AscendC::LocalTensor<uint64_t> &ubLocal64,
    AscendC::LocalTensor<uint32_t> &ubLocal32, uint32_t &curHead, __gm__ ACLSHMEMWQCtx *&qpCtxEntry);

/**
 * @brief RDMA Poll Completion Queue (CQ) function. Return status: 0 means success, non-zero means error.
 *
 * @param remoteRankId           [in] destination rank ID
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param idx                    [in] expect completion queue consumer index after polling
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 */
ACLSHMEM_DEVICE uint32_t aclshmemi_roce_poll_cq(uint32_t remoteRankId, uint32_t qpIdx, uint32_t idx,
                                          AscendC::LocalTensor<uint64_t> ubLocal64,
                                          AscendC::LocalTensor<uint32_t> ubLocal32);

ACLSHMEM_DEVICE void aclshmemi_roce_poll_cq_update_info(AscendC::LocalTensor<uint64_t> &ubLocal64,
    AscendC::LocalTensor<uint32_t> &ubLocal32, uint32_t &curTail, uint32_t &remoteRankId, uint32_t &qpIdx);

/**
 * @brief AIV direct RDMA helper function for post send, prepare WQE and ring doorbell.
 *
 * @param remoteAddr             [in] address in remote HBM
 * @param localAddr              [in] address in lcoal HBM
 * @param destRankId             [in] destination rank ID
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param opcode                 [in] rdma opcode in ACLSHMEMAIVOPCODE enum class
 * @param messageLen             [in] message length in Bytes
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 */
ACLSHMEM_DEVICE void aclshmemi_rdma_post_send(__gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
                                        uint32_t destRankId, uint32_t qpIdx,
                                        ACLSHMEMAIVOPCODE opcode, uint64_t messageLen,
                                        AscendC::LocalTensor<uint64_t> ubLocal64,
                                        AscendC::LocalTensor<uint32_t> ubLocal32);

ACLSHMEM_DEVICE void aclshmemi_rdma_post_send_update_info(AscendC::LocalTensor<uint64_t> &ubLocal64,
    AscendC::LocalTensor<uint32_t> &ubLocal32, uint32_t &curHead, __gm__ ACLSHMEMWQCtx *&qpCtxEntry);

/**
 * @brief Asynchronous RDMA Write function.
 *
 * @param destDmaAddr            [in] destination address in remote HBM
 * @param srcDmaAddr             [in] source address in local HBM
 * @param destRankId             [in] destination rank ID
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param messageLen             [in] message length in Bytes
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 */
template<typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_write(__gm__ T* destDmaAddr, __gm__ T* srcDmaAddr, uint32_t destRankId,
                                    uint32_t qpIdx, uint64_t messageLen,
                                    AscendC::LocalTensor<uint64_t> ubLocal64,
                                    AscendC::LocalTensor<uint32_t> ubLocal32);

/**
 * @brief Asynchronous RDMA READ function.
 *
 * @param destDmaAddr            [in] destination address in local HBM
 * @param srcDmaAddr             [in] source address in remote HBM
 * @param srcRankId              [in] destination rank ID
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param messageLen             [in] message length in Bytes
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 */
template<typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_read(__gm__ T* destDmaAddr, __gm__ T* srcDmaAddr, uint32_t srcRankId,
                                   uint32_t qpIdx, uint64_t messageLen,
                                   AscendC::LocalTensor<uint64_t> ubLocal64,
                                   AscendC::LocalTensor<uint32_t> ubLocal32);

/**
 * @brief RDMA Quiet function. This synchronous function ensures all previous RDMA WQEs are completed
 * (data has arrived at the destination NIC).
 *
 * @param remoteRankId           [in] destination rank ID
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param ubLocal64              [in] temporary UB local tensor of uint64_t used as workspace
 * @param ubLocal32              [in] temporary UB local tensor of uint32_t used as workspace
 */
ACLSHMEM_DEVICE void aclshmemi_roce_quiet(uint32_t remoteRankId, uint32_t qpIdx,
                                    AscendC::LocalTensor<uint64_t> ubLocal64,
                                    AscendC::LocalTensor<uint32_t> ubLocal32);

ACLSHMEM_DEVICE void aclshmemi_roce_qpinfo_test(__gm__ uint8_t* gva, uint32_t destRankId, uint32_t qpIdx);

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_pollcq_test(__gm__ T* srcDmaAddr, __gm__ T* destDmaAddr, uint32_t destRankId,
                                                    uint32_t qpIdx, uint64_t messageLen,
                                                    AscendC::LocalTensor<uint64_t> ubLocal64,
                                                    AscendC::LocalTensor<uint32_t> ubLocal32, __gm__ uint8_t* gva);

#endif