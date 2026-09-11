/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_RDMA_HPP
#define SHMEM_DEVICE_RDMA_HPP

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "shmemi_device_rdma.h"

ACLSHMEM_DEVICE __gm__ ACLSHMEMAIVRDMAInfo* aclshmemi_qp_info_fetch()
{
#ifdef BACKEND_HYBM
    __gm__ ACLSHMEMAIVRDMAInfo* RDMAInfo = (__gm__ ACLSHMEMAIVRDMAInfo*)(aclshmemi_get_qp_info_address(0));
    return RDMAInfo;
#else
    __gm__ aclshmem_device_host_state_t  *device_state = aclshmemi_get_state();
    __gm__ ACLSHMEMAIVRDMAInfo* RDMAInfo = (__gm__ ACLSHMEMAIVRDMAInfo*)(device_state->qp_info);
    return RDMAInfo;
#endif
}

ACLSHMEM_DEVICE uint32_t aclshmemi_roce_poll_cq(uint32_t remoteRankId, uint32_t qpIdx, uint32_t idx,
                                          AscendC::LocalTensor<uint64_t> ubLocal64,
                                          AscendC::LocalTensor<uint32_t> ubLocal32)
{
    if (idx == 0) {
        return 0;
    }
    __gm__ ACLSHMEMAIVRDMAInfo* RDMAInfo = aclshmemi_qp_info_fetch();

    uint32_t qpNum = RDMAInfo->qpNum;
    __gm__ ACLSHMEMCQCtx* cqCtxEntry = (__gm__ ACLSHMEMCQCtx*)(RDMAInfo->scqPtr
        + (remoteRankId * qpNum + qpIdx) * sizeof(ACLSHMEMCQCtx));
    auto cqBaseAddr = cqCtxEntry->bufAddr;
    auto cqeSize = cqCtxEntry->cqeSize;
    auto depth = cqCtxEntry->depth;
    auto curHardwareTailAddr = cqCtxEntry->tailAddr;
    dcci_cachelines((__gm__ uint8_t*)curHardwareTailAddr, 8);
    uint32_t curTail = *(__gm__ uint32_t*)(curHardwareTailAddr);

    const uint32_t shiftWidth = 7;
    AscendC::DataCopyExtParams copyParamsTail{1, 1 * sizeof(uint32_t), 0, 0, 0};
    while (curTail != idx) {
        __gm__ ACLSHMEMcqeCtx* cqeAddr = (__gm__ ACLSHMEMcqeCtx*)(cqBaseAddr + cqeSize * (curTail & (depth - 1)));
        uint32_t cqeByte4 = *(__gm__ uint32_t*)cqeAddr;
        while (((cqeByte4 & (1 << shiftWidth)) != 0) == ((curTail & depth) != 0)) {
            int64_t tmp = AscendC::GetSystemCycle(); // reserved for timeout check
            dcci_cachelines((__gm__ uint8_t*)cqeAddr, 32);
            cqeByte4 = *(__gm__ uint32_t*)cqeAddr;
        }
        curTail++;
        uint32_t wqn = cqeAddr->byte16 & 0xFFFFFF; // reserved for multi WQ share the same CQ

        // Check CQE status
        uint32_t status = (cqeAddr->byte4 >> 8) & 0xFF;
        if (status) {
            return status;
        }
    }

    // Update CQ tail
    ubLocal32.SetValue(0, (uint32_t)curTail);
    AscendC::GlobalTensor<uint32_t> TailGlobalTensor;
    TailGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)curHardwareTailAddr);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(TailGlobalTensor, ubLocal32, copyParamsTail);
    AscendC::PipeBarrier<PIPE_ALL>();

    aclshmemi_roce_poll_cq_update_info(ubLocal64, ubLocal32, curTail, remoteRankId, qpIdx);
    return 0;
}

ACLSHMEM_DEVICE void aclshmemi_roce_poll_cq_update_info(AscendC::LocalTensor<uint64_t> &ubLocal64,
    AscendC::LocalTensor<uint32_t> &ubLocal32, uint32_t &curTail, uint32_t &remoteRankId, uint32_t &qpIdx)
{
    __gm__ ACLSHMEMAIVRDMAInfo* RDMAInfo = aclshmemi_qp_info_fetch();

    uint32_t qpNum = RDMAInfo->qpNum;
    __gm__ ACLSHMEMCQCtx* cqCtxEntry = (__gm__ ACLSHMEMCQCtx*)(RDMAInfo->scqPtr
        + (remoteRankId * qpNum + qpIdx) * sizeof(ACLSHMEMCQCtx));
    AscendC::DataCopyExtParams copyParamsTail{1, 1 * sizeof(uint32_t), 0, 0, 0};

    // Ring CQ Doorbell
    auto cqDBAddr = cqCtxEntry->dbAddr;
    if (cqCtxEntry->dbMode == ACLSHMEMDBMode::SW_DB) {
        ubLocal32.SetValue(0, (uint32_t)(curTail & 0xFFFFFF));
        AscendC::GlobalTensor<uint32_t> CQDBGlobalTensor;
        CQDBGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)cqDBAddr);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(CQDBGlobalTensor, ubLocal32, copyParamsTail);
        AscendC::PipeBarrier<PIPE_ALL>();
    } else if (cqCtxEntry->dbMode == ACLSHMEMDBMode::HW_DB) {
        uint64_t doorBellInfo = 0;
        doorBellInfo |= cqCtxEntry->cqn; // [0:23] DB_TAG = qp_num
        doorBellInfo |= 3 << 24; // [24:27] DB_CMD = HNS_ROCE_V2_CQ_DB_PTR(3)
        doorBellInfo |= (uint64_t)(curTail & 0xFFFFFF) << 32; // [32:55] DB_CQ_CI = cq.tail
        doorBellInfo |= (uint64_t)1 << 56; // [56:56] DB_CQ_CMD_SN = 1
        ubLocal64.SetValue(0, doorBellInfo);
        AscendC::GlobalTensor<uint64_t> DBGlobalTensor;
        DBGlobalTensor.SetGlobalBuffer((__gm__ uint64_t*)cqDBAddr);
        AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(DBGlobalTensor, ubLocal64, copyParams);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Update WQ tail
    __gm__ ACLSHMEMWQCtx* wqCtxEntry = (__gm__ ACLSHMEMWQCtx*)(RDMAInfo->sqPtr
        + (remoteRankId * qpNum + qpIdx) * sizeof(ACLSHMEMWQCtx));
    auto curWQTailAddr = wqCtxEntry->tailAddr;
    dcci_cachelines((__gm__ uint8_t*)curWQTailAddr, 8);
    uint32_t curWQTail = *(__gm__ uint32_t*)(curWQTailAddr);
    ubLocal32.SetValue(0, curTail);
    AscendC::GlobalTensor<uint32_t> WQTailGlobalTensor;
    WQTailGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)curWQTailAddr);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(WQTailGlobalTensor, ubLocal32, copyParamsTail);
    AscendC::PipeBarrier<PIPE_ALL>();
}

ACLSHMEM_DEVICE void aclshmemi_rdma_post_send(__gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
                                        uint32_t destRankId, uint32_t qpIdx,
                                        ACLSHMEMAIVOPCODE opcode, uint64_t messageLen,
                                        AscendC::LocalTensor<uint64_t> ubLocal64,
                                        AscendC::LocalTensor<uint32_t> ubLocal32)
{
    __gm__ ACLSHMEMAIVRDMAInfo* RDMAInfo = aclshmemi_qp_info_fetch();

    uint32_t qpNum = RDMAInfo->qpNum;
    __gm__ ACLSHMEMWQCtx* qpCtxEntry = (__gm__ ACLSHMEMWQCtx*)(RDMAInfo->sqPtr
        + (destRankId * qpNum + qpIdx) * sizeof(ACLSHMEMWQCtx));
    auto ACLSHMEMmemInfoTable = RDMAInfo->memPtr;
    auto sqBaseAddr = qpCtxEntry->bufAddr;
    auto wqeSize = qpCtxEntry->wqeSize;
    auto curHardwareHeadAddr = qpCtxEntry->headAddr;
    dcci_cachelines((__gm__ uint8_t*)curHardwareHeadAddr, 8);
    uint32_t curHead = *(__gm__ uint32_t*)(curHardwareHeadAddr);
    auto curHardwareTailAddr = qpCtxEntry->tailAddr;
    auto depth = qpCtxEntry->depth;
    auto shift = 13;
    AscendC::PipeBarrier<PIPE_ALL>();

    // Poll CQ if send queue is full
    dcci_cachelines((__gm__ uint8_t*)curHardwareTailAddr, 8);
    if ((curHead + 10) % depth == (*(__gm__ uint32_t*)(curHardwareTailAddr)) % depth) {
        aclshmemi_roce_poll_cq(destRankId, qpIdx, *(__gm__ uint32_t*)(curHardwareTailAddr) +
            ACLSHMEM_NUM_CQE_PER_POLL_CQ, ubLocal64, ubLocal32);
    }

    // Write WQE to HBM
    __gm__ uint8_t* wqeAddr = (__gm__ uint8_t*)(sqBaseAddr + wqeSize * (curHead % depth));
    uint64_t ownBit = (curHead >> shift) & 0x1;
    uint32_t byte4 = (uint32_t)opcode & 0x1F;       // [0:4] opcode
    byte4 |= ((~ownBit) << 7) & (1 << 7); // [7] owner_bit
    byte4 |= 1 << 8;                      // [8] IBV_SEND_SINGNALED
    *(__gm__ uint32_t*)(wqeAddr) = byte4; // control set by local parameter, see above lines
    *(__gm__ uint32_t*)(wqeAddr + 4) = messageLen; // message size in bytes
    *(__gm__ uint32_t*)(wqeAddr + 8) = 0; // immtdata is always 0 till we provide poll CQ flow in AIV
    *(__gm__ uint32_t*)(wqeAddr + 12) = 1 << 24; // [120:127] num_sge = 1
    *(__gm__ uint32_t*)(wqeAddr + 16) = 0; // [128:151] start_sge_index = 0
    __gm__ ACLSHMEMmemInfo* remoteMemInfo = (__gm__ ACLSHMEMmemInfo*)(ACLSHMEMmemInfoTable + sizeof(ACLSHMEMmemInfo) * destRankId);
    *(__gm__ uint32_t*)(wqeAddr + 20) = remoteMemInfo->rkey; // rkey
    *(__gm__ uint64_t*)(wqeAddr + 24) = (uint64_t)remoteAddr; // remote VA

    // Write SGE to HBM
    __gm__ uint8_t* sgeAddr = wqeAddr + sizeof(ACLSHMEMwqeCtx);
    *(__gm__ uint32_t*)(sgeAddr) = messageLen; // message size in bytes
    __gm__ ACLSHMEMmemInfo* localMemInfo = (__gm__ ACLSHMEMmemInfo*)(ACLSHMEMmemInfoTable
        + sizeof(ACLSHMEMmemInfo) * aclshmemi_get_my_pe());
    *(__gm__ uint32_t*)(sgeAddr + 4) = localMemInfo->lkey; // lkey
    *(__gm__ uint64_t*)(sgeAddr + 8) = (uint64_t)localAddr; // local VA

    // WQE & SGE cache flush
    dcci_cachelines(wqeAddr, sizeof(ACLSHMEMwqeCtx) + sizeof(ACLSHMEMsegCtx));
    AscendC::PipeBarrier<PIPE_ALL>();
    curHead++;

    aclshmemi_rdma_post_send_update_info(ubLocal64, ubLocal32, curHead, qpCtxEntry);
}

ACLSHMEM_DEVICE void aclshmemi_rdma_post_send_update_info(AscendC::LocalTensor<uint64_t> &ubLocal64,
    AscendC::LocalTensor<uint32_t> &ubLocal32, uint32_t &curHead, __gm__ ACLSHMEMWQCtx *&qpCtxEntry)
{
    uint64_t doorBellInfo = 0;
    doorBellInfo |= qpCtxEntry->wqn; // [0:23] DB_TAG = qp_num
    doorBellInfo |= 0 << 24; // [24:27] DB_CMD = HNS_ROCE_V2_SQ_DB(0)
    doorBellInfo |= ((uint64_t)curHead % 65536) << 32; // [32:47] DB_PI = sq.head
    doorBellInfo |= (uint64_t)(qpCtxEntry->sl) << 48; // [48:50] DB_SL = qp.sl

    auto curHardwareHeadAddr = qpCtxEntry->headAddr;

    __gm__ uint64_t* doorBellAddr = (__gm__ uint64_t*)(qpCtxEntry->dbAddr);
    AscendC::PipeBarrier<PIPE_ALL>();

    ubLocal64.SetValue(0, doorBellInfo);
    AscendC::GlobalTensor<uint64_t> DBGlobalTensor;
    DBGlobalTensor.SetGlobalBuffer(doorBellAddr);
    AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(DBGlobalTensor, ubLocal64, copyParams);
    AscendC::PipeBarrier<PIPE_ALL>();

    ubLocal32.SetValue(0, (uint32_t)curHead);
    AscendC::GlobalTensor<uint32_t> HeadGlobalTensor;
    HeadGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)curHardwareHeadAddr);
    AscendC::DataCopyExtParams copyParamsHead{1, 1 * sizeof(uint32_t), 0, 0, 0};
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(HeadGlobalTensor, ubLocal32, copyParamsHead);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_write(__gm__ T* destDmaAddr, __gm__ T* srcDmaAddr, uint32_t destRankId,
                                    uint32_t qpIdx, uint64_t messageLen,
                                    AscendC::LocalTensor<uint64_t> ubLocal64,
                                    AscendC::LocalTensor<uint32_t> ubLocal32)
{
    aclshmemi_rdma_post_send(destDmaAddr, srcDmaAddr, destRankId, qpIdx, ACLSHMEMAIVOPCODE::OP_RDMA_WRITE,
                            messageLen, ubLocal64, ubLocal32);
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_read(__gm__ T* destDmaAddr, __gm__ T* srcDmaAddr, uint32_t srcRankId,
                                   uint32_t qpIdx, uint64_t messageLen,
                                   AscendC::LocalTensor<uint64_t> ubLocal64,
                                   AscendC::LocalTensor<uint32_t> ubLocal32)
{
    aclshmemi_rdma_post_send(srcDmaAddr, destDmaAddr, srcRankId, qpIdx, ACLSHMEMAIVOPCODE::OP_RDMA_READ,
                            messageLen, ubLocal64, ubLocal32);
}

ACLSHMEM_DEVICE void aclshmemi_roce_quiet(uint32_t remoteRankId, uint32_t qpIdx,
                                    AscendC::LocalTensor<uint64_t> ubLocal64,
                                    AscendC::LocalTensor<uint32_t> ubLocal32)
{
    __gm__ ACLSHMEMAIVRDMAInfo* RDMAInfo = aclshmemi_qp_info_fetch();

    uint32_t qpNum = RDMAInfo->qpNum;
    __gm__ ACLSHMEMWQCtx* qpCtxEntry = (__gm__ ACLSHMEMWQCtx*)(RDMAInfo->sqPtr
        + (remoteRankId * qpNum + qpIdx) * sizeof(ACLSHMEMWQCtx));
    auto curHardwareHeadAddr = qpCtxEntry->headAddr;
    dcci_cachelines((__gm__ uint8_t*)curHardwareHeadAddr, 8);
    uint32_t curHead = *(__gm__ uint32_t*)(curHardwareHeadAddr);
    aclshmemi_roce_poll_cq(remoteRankId, qpIdx, curHead, ubLocal64, ubLocal32);
}

ACLSHMEM_DEVICE void aclshmemi_roce_qpinfo_test(__gm__ uint8_t* gva, uint32_t destRankId, uint32_t qpIdx)
{
    __gm__ ACLSHMEMAIVRDMAInfo* RDMAInfo = aclshmemi_qp_info_fetch();

    *(__gm__ uint64_t*)(gva) = (uint64_t)RDMAInfo;
    uint32_t qpNum = RDMAInfo->qpNum;
    *(__gm__ uint64_t*)(gva + 8) = (uint64_t)qpNum;
    __gm__ ACLSHMEMWQCtx* qpCtxEntry = (__gm__ ACLSHMEMWQCtx*)(RDMAInfo->sqPtr +
        (destRankId * qpNum + qpIdx) * sizeof(ACLSHMEMWQCtx));
    *(__gm__ uint64_t*)(gva + 16) = (uint64_t)qpCtxEntry;
    auto ACLSHMEMmemInfoTable = RDMAInfo->memPtr;
    *(__gm__ uint64_t*)(gva + 24) = (uint64_t)ACLSHMEMmemInfoTable;
    auto sqBaseAddr = qpCtxEntry->bufAddr;
    *(__gm__ uint64_t*)(gva + 32) = (uint64_t)sqBaseAddr;
    auto wqeSize = qpCtxEntry->wqeSize;
    *(__gm__ uint64_t*)(gva + 40) = (uint64_t)wqeSize;
    auto curHardwareHeadAddr = qpCtxEntry->headAddr;
    *(__gm__ uint64_t*)(gva + 48) = (uint64_t)curHardwareHeadAddr;
    dcci_cachelines((__gm__ uint8_t*)curHardwareHeadAddr, 8);
    uint32_t curHead = *(__gm__ uint32_t*)(curHardwareHeadAddr);
    *(__gm__ uint64_t*)(gva + 56) = (uint64_t)curHead;
    auto curHardwareTailAddr = qpCtxEntry->tailAddr;
    *(__gm__ uint64_t*)(gva + 64) = (uint64_t)curHardwareTailAddr;
    auto depth = qpCtxEntry->depth;
    *(__gm__ uint64_t*)(gva + 72) = (uint64_t)depth;
    *(__gm__ uint64_t*)(gva + 80) = (uint64_t)(qpCtxEntry->sl);
    auto shift = 15;
    AscendC::PipeBarrier<PIPE_ALL>();

    // Write WQE to HBM
    __gm__ uint8_t* wqeAddr = (__gm__ uint8_t*)(sqBaseAddr + wqeSize * (curHead % depth));
    __gm__ ACLSHMEMmemInfo* remoteMemInfo = (__gm__ ACLSHMEMmemInfo*)(ACLSHMEMmemInfoTable + sizeof(ACLSHMEMmemInfo) * destRankId);
    *(__gm__ uint64_t*)(gva + 88) = (uint64_t)(remoteMemInfo->rkey);

    // Write SGE to HBM
    __gm__ ACLSHMEMmemInfo* localMemInfo = (__gm__ ACLSHMEMmemInfo*)(ACLSHMEMmemInfoTable
        + sizeof(ACLSHMEMmemInfo) * aclshmemi_get_my_pe());
    *(__gm__ uint64_t*)(gva + 96) = (uint64_t)(localMemInfo->lkey);; // lkey

    __gm__ uint64_t* doorBellAddr = (__gm__ uint64_t*)(qpCtxEntry->dbAddr);
    *(__gm__ uint64_t*)(gva + 104) = (uint64_t)doorBellAddr;
    *(__gm__ uint64_t*)(gva + 112) = (uint64_t)gva;
    AscendC::PipeBarrier<PIPE_ALL>();
}

template<typename T>
ACLSHMEM_DEVICE void aclshmemi_roce_pollcq_test(__gm__ T* srcDmaAddr, __gm__ T* destDmaAddr, uint32_t destRankId,
                                                    uint32_t qpIdx, uint64_t messageLen,
                                                    AscendC::LocalTensor<uint64_t> ubLocal64,
                                                    AscendC::LocalTensor<uint32_t> ubLocal32, __gm__ uint8_t* gva)
{
    aclshmemi_rdma_post_send(destDmaAddr, srcDmaAddr, destRankId, qpIdx, ACLSHMEMAIVOPCODE::OP_RDMA_WRITE,
                            messageLen, ubLocal64, ubLocal32);
    uint32_t idx = 1;
    __gm__ ACLSHMEMAIVRDMAInfo* RDMAInfo = aclshmemi_qp_info_fetch();

    uint32_t qpNum = RDMAInfo->qpNum;
    __gm__ ACLSHMEMCQCtx* cqCtxEntry = (__gm__ ACLSHMEMCQCtx*)(RDMAInfo->scqPtr + (destRankId * qpNum + qpIdx) * sizeof(ACLSHMEMCQCtx));
    *(__gm__ uint64_t*)(gva) = (uint64_t)cqCtxEntry;
    auto cqBaseAddr = cqCtxEntry->bufAddr;
    auto cqeSize = cqCtxEntry->cqeSize;
    auto depth = cqCtxEntry->depth;
    *(__gm__ uint64_t*)(gva + 8) = (uint64_t)cqBaseAddr;
    *(__gm__ uint64_t*)(gva + 16) = (uint64_t)cqeSize;
    *(__gm__ uint64_t*)(gva + 24) = (uint64_t)depth;
    auto curHardwareTailAddr = cqCtxEntry->tailAddr;
    *(__gm__ uint64_t*)(gva + 32) = (uint64_t)curHardwareTailAddr;
    dcci_cachelines((__gm__ uint8_t*)curHardwareTailAddr, 8);
    uint32_t curTail = *(__gm__ uint32_t*)(curHardwareTailAddr);
    *(__gm__ uint64_t*)(gva + 40) = (uint64_t)curTail;

    AscendC::DataCopyExtParams copyParamsTail{1, 1 * sizeof(uint32_t), 0, 0, 0};
    
    __gm__ ACLSHMEMcqeCtx* cqeAddr = (__gm__ ACLSHMEMcqeCtx*)(cqBaseAddr + cqeSize * (curTail & (depth - 1)));
    uint32_t cqeByte4 = *(__gm__ uint32_t*)cqeAddr;
    while (!(cqeByte4 & (1 << 7))) {
        int64_t tmp = AscendC::GetSystemCycle();
        dcci_cachelines((__gm__ uint8_t*)cqeAddr, 32);
        cqeByte4 = *(__gm__ uint32_t*)cqeAddr;
    }
    *(__gm__ uint64_t*)(gva + 56) = (uint64_t)(cqeAddr->byte4);
    *(__gm__ uint64_t*)(gva + 64) = (uint64_t)(cqeAddr->immtdata);
    *(__gm__ uint64_t*)(gva + 72) = (uint64_t)(cqeAddr->byte12);
    *(__gm__ uint64_t*)(gva + 80) = (uint64_t)(cqeAddr->byte16);
    *(__gm__ uint64_t*)(gva + 88) = (uint64_t)(cqeAddr->byteCnt);
    *(__gm__ uint64_t*)(gva + 96) = (uint64_t)(cqeAddr->smac);
    curTail++;
    // Process each CQE, and update WQ tail
    uint32_t wqn = cqeAddr->byte16 & 0xFFFFFF;
    __gm__ ACLSHMEMWQCtx* wqCtxEntry = (__gm__ ACLSHMEMWQCtx*)(RDMAInfo->sqPtr + (destRankId * qpNum + qpIdx) * sizeof(ACLSHMEMWQCtx));
    *(__gm__ uint64_t*)(gva + 104) = (uint64_t)(wqCtxEntry->wqn == wqn);
    auto curWQTailAddr = wqCtxEntry->tailAddr;
    dcci_cachelines((__gm__ uint8_t*)curWQTailAddr, 8);
    uint32_t curWQTail = *(__gm__ uint32_t*)(curWQTailAddr);
    ubLocal32.SetValue(0, curWQTail + 1);
    AscendC::GlobalTensor<uint32_t> WQTailGlobalTensor;
    WQTailGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)curWQTailAddr);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(WQTailGlobalTensor, ubLocal32, copyParamsTail);
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cachelines((__gm__ uint8_t*)curWQTailAddr, 8);
    
    // Check CQE status
    uint32_t status = (cqeAddr->byte4 >> 8) & 0xFF;
    *(__gm__ uint64_t*)(gva + 112) = status;
    if (status) {
        return;
    }

    // Update tail
    ubLocal32.SetValue(0, (uint32_t)curTail);
    AscendC::GlobalTensor<uint32_t> TailGlobalTensor;
    TailGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)curHardwareTailAddr);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(TailGlobalTensor, ubLocal32, copyParamsTail);
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cachelines((__gm__ uint8_t*)curHardwareTailAddr, 8);

    // Ring CQ Doorbell
    auto cqDBAddr = cqCtxEntry->dbAddr;
    ubLocal32.SetValue(0, (uint32_t)(curTail & 0xFFFFFF));
    AscendC::GlobalTensor<uint32_t> CQDBGlobalTensor;
    CQDBGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)cqDBAddr);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(CQDBGlobalTensor, ubLocal32, copyParamsTail);
    AscendC::PipeBarrier<PIPE_ALL>();
    dcci_cachelines((__gm__ uint8_t*)cqDBAddr, 8);
}

ACLSHMEM_DEVICE __gm__ void *aclshmem_roce_ptr(__gm__ void *ptr, int pe)
{
    // Get Global State
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();

    // Back to root address
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(device_state->heap_base);
#ifdef BACKEND_HYBM
    uint64_t remote_ptr = reinterpret_cast<uint64_t>(device_state->p2p_device_heap_base[pe]) + offset;
#else
    uint64_t remote_ptr = reinterpret_cast<uint64_t>(device_state->rdma_device_heap_base[pe]) + offset;
#endif

    return reinterpret_cast<__gm__ void *>(remote_ptr);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe)
{
    auto ptr = aclshmem_roce_ptr(src, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_read((__gm__ uint8_t*)dst, (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T),
        ub_tensor_64, ub_tensor_32);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_get_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
    AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe)
{
    auto ptr = aclshmem_roce_ptr((__gm__ void *)src.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_read((__gm__ uint8_t*)dst.GetPhyAddr(), (__gm__ uint8_t*)ptr, pe, 0, elem_size * sizeof(T),
        ub_tensor_64, ub_tensor_32);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe)
{
    auto ptr = aclshmem_roce_ptr(dst, pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)src, pe, 0, elem_size * sizeof(T),
        ub_tensor_64, ub_tensor_32);
}

template <typename T>
ACLSHMEM_DEVICE void aclshmemx_roce_put_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
    AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = aclshmem_roce_ptr((__gm__ void *)dst.GetPhyAddr(), pe);
    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr());
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(buf.GetPhyAddr()) + UB_ALIGN_SIZE;
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;
    aclshmemi_roce_write((__gm__ uint8_t*)ptr, (__gm__ uint8_t*)(src.GetPhyAddr()), pe, 0,
        elem_size * sizeof(T), ub_tensor_64, ub_tensor_32);
}

#endif