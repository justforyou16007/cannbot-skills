// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
//
// GroupedMatmul 全部 dtype/bias/tw 入口（单文件，direct-launch 构建）+ launch 包装。
// 每 (dtype, has_bias, tw) 一个 __global__ 入口（结构轴，12 实例）；shape 维度
// 全运行期（GM tiling 读）。KERNEL_TASK_TYPE_DEFAULT 放入口内（MIX_AIC_1_2：
// 纯 cube 体，AIC 工作 / AIV 空转返回）。

#include "gmm_kernel.h"

// ---------------------------------------------------------------------------
// bias_cast 纯 AIV kernel（ISSUE-10 提交合规修复，方案 1/Q1）。
// 主 grouped_matmul kernel 的 bias 累加走 BT（fp32），host 侧若 bias 为
// half/bf16 需先转 fp32。原实现用 host `bias_opt->to(at::kFloat)`（一次额外
// D2H/H2D + 中间张量），合规预算超标。改为独立的 GM→UB→Cast<T→float>→GM
// 纯 AIV kernel：shape 走左值标量参数（零 H2D），输出 workspace 由 host
// at::empty 分配（零 H2D）。主 kernel 零改动（仍读 fp32 bias 进 BT）。
// 非 fp32 bias 才 launch；fp32 bias 直接透传（不 launch）。
// ---------------------------------------------------------------------------
namespace gmm {

template <typename T>
__aicore__ inline void bias_cast_run(GM_ADDR src_handle, GM_ADDR dst_handle,
                                     int32_t total) {
    // 纯 AIV 体：cid = GetBlockIdx()（不除 2），核数运行期 GetBlockNum()
    // （调试纪律 §10：AIV_ONLY 无孪生，照抄 cid/2 会半覆盖）。
    AscendC::TPipe pipe;
    // UB 双缓冲槽：每槽 CHUNK 个 T 输入 + CHUNK 个 float 输出。
    constexpr uint32_t CHUNK = 8192;  // 8K 元素/槽，fp32 输出 32KB，输入 ≤16KB
    AscendC::TBuf<AscendC::TPosition::VECIN> inBuf;
    AscendC::TBuf<AscendC::TPosition::VECOUT> outBuf;
    pipe.InitBuffer(inBuf, CHUNK * sizeof(T));
    pipe.InitBuffer(outBuf, CHUNK * sizeof(float));

    AscendC::GlobalTensor<T> Src;
    Src.SetGlobalBuffer((__gm__ T*)src_handle);
    AscendC::GlobalTensor<float> Dst;
    Dst.SetGlobalBuffer((__gm__ float*)dst_handle);

    auto src_ub = inBuf.Get<T>();
    auto dst_ub = outBuf.Get<float>();

    const int32_t cid = AscendC::GetBlockIdx();
    const int32_t n_cores = AscendC::GetBlockNum();
    // 核间均分（按 CHUNK 块数 ceil 分，末核吃尾块）
    const int32_t n_chunks = (total + (int32_t)CHUNK - 1) / (int32_t)CHUNK;
    for (int32_t c = cid; c < n_chunks; c += n_cores) {
        const int32_t off = c * (int32_t)CHUNK;
        const int32_t len = ((off + (int32_t)CHUNK) <= total)
                                ? (int32_t)CHUNK
                                : (total - off);
        // GM→UB（DataCopyPad 处理非对齐尾块）
        AscendC::DataCopyExtParams cpIn{1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padIn{false, 0, 0, 0};
        AscendC::DataCopyPad(src_ub, Src[off], cpIn, padIn);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        // Cast T→float（升宽，CAST_NONE；对齐到 32B 的整 CHUNK 用向量化 Cast）
        AscendC::Cast(dst_ub, src_ub, AscendC::RoundMode::CAST_NONE, len);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        // UB→GM（fp32，DataCopyPad 处理非对齐尾块）
        AscendC::DataCopyExtParams cpOut{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(Dst[off], dst_ub, cpOut);
        // 下一迭代复用 buffer 前等待 MTE3 完成（跨迭代 WAR）
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }
    pipe.Destroy();
}

}  // namespace gmm

#define GMM_DEFINE_KERNEL(SUFFIX, T, HAS_BIAS, TW)                            \
    extern "C" __global__ __aicore__ void gmm_kernel_##SUFFIX(                \
        GM_ADDR X_handle, GM_ADDR W_handle, GM_ADDR Bias_handle,              \
        GM_ADDR Y_handle, GM_ADDR Meta_handle, GM_ADDR tiling) {              \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                    \
        AscendC::GlobalTensor<int> ArgsGM;                                    \
        ArgsGM.SetGlobalBuffer((__gm__ int*)tiling);                          \
        gmm::GmmTiling a;                                                     \
        a.M = ArgsGM.GetValue(0);                                             \
        a.N = ArgsGM.GetValue(1);                                             \
        a.K = ArgsGM.GetValue(2);                                             \
        a.E = ArgsGM.GetValue(3);                                             \
        a.n_num = ArgsGM.GetValue(4);                                         \
        a.loop_k = ArgsGM.GetValue(5);                                        \
        a.total_m_blocks = ArgsGM.GetValue(6);                                \
        a.total_blocks = ArgsGM.GetValue(7);                                  \
        a.grid = ArgsGM.GetValue(8);                                          \
        gmm::gmm_run<T, HAS_BIAS, TW>(X_handle, W_handle, Bias_handle,        \
                                      Y_handle, Meta_handle, a);              \
    }

GMM_DEFINE_KERNEL(fp16_bias_nt, half, true, false)
GMM_DEFINE_KERNEL(fp16_bias_t, half, true, true)
GMM_DEFINE_KERNEL(fp16_nobias_nt, half, false, false)
GMM_DEFINE_KERNEL(fp16_nobias_t, half, false, true)
GMM_DEFINE_KERNEL(bf16_bias_nt, bfloat16_t, true, false)
GMM_DEFINE_KERNEL(bf16_bias_t, bfloat16_t, true, true)
GMM_DEFINE_KERNEL(bf16_nobias_nt, bfloat16_t, false, false)
GMM_DEFINE_KERNEL(bf16_nobias_t, bfloat16_t, false, true)
GMM_DEFINE_KERNEL(fp32_bias_nt, float, true, false)
GMM_DEFINE_KERNEL(fp32_bias_t, float, true, true)
GMM_DEFINE_KERNEL(fp32_nobias_nt, float, false, false)
GMM_DEFINE_KERNEL(fp32_nobias_t, float, false, true)

// host 可调用的 launch 包装（direct-launch：本文件经 bisheng -xasc 编译，
// <<<>>> 可用）。grid 由 host 按 min(total_blocks, 物理核数) 传入。
#define GMM_DEFINE_LAUNCH(SUFFIX)                                             \
    extern "C" void launch_##SUFFIX(                                          \
        uint32_t grid, void *stream, GM_ADDR X_handle, GM_ADDR W_handle,      \
        GM_ADDR Bias_handle, GM_ADDR Y_handle, GM_ADDR Meta_handle,           \
        GM_ADDR tiling) {                                                     \
        gmm_kernel_##SUFFIX<<<grid, nullptr, (aclrtStream)stream>>>(          \
            X_handle, W_handle, Bias_handle, Y_handle, Meta_handle, tiling);  \
    }

GMM_DEFINE_LAUNCH(fp16_bias_nt)
GMM_DEFINE_LAUNCH(fp16_bias_t)
GMM_DEFINE_LAUNCH(fp16_nobias_nt)
GMM_DEFINE_LAUNCH(fp16_nobias_t)
GMM_DEFINE_LAUNCH(bf16_bias_nt)
GMM_DEFINE_LAUNCH(bf16_bias_t)
GMM_DEFINE_LAUNCH(bf16_nobias_nt)
GMM_DEFINE_LAUNCH(bf16_nobias_t)
GMM_DEFINE_LAUNCH(fp32_bias_nt)
GMM_DEFINE_LAUNCH(fp32_bias_t)
GMM_DEFINE_LAUNCH(fp32_nobias_nt)
GMM_DEFINE_LAUNCH(fp32_nobias_t)

// ---------------------------------------------------------------------------
// bias_cast 入口（纯 AIV，AIV_ONLY）+ launch 包装。仅 half/bf16 两个 dtype
// （fp32 bias 直接透传不 launch）。grid 由 host 按 2×物理核数 传入（起满
// 全部 vector 核，调试纪律 §10）。shape 走左值标量 total（零 H2D）。
// ---------------------------------------------------------------------------
#define GMM_DEFINE_BIAS_CAST(SUFFIX, T)                                       \
    extern "C" __global__ __aicore__ void gmm_bias_cast_##SUFFIX(             \
        GM_ADDR src_handle, GM_ADDR dst_handle, int32_t total) {              \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                       \
        gmm::bias_cast_run<T>(src_handle, dst_handle, total);                 \
    }                                                                         \
    extern "C" void launch_bias_cast_##SUFFIX(                                \
        uint32_t grid, void *stream, GM_ADDR src_handle, GM_ADDR dst_handle,  \
        int32_t total) {                                                      \
        gmm_bias_cast_##SUFFIX<<<grid, nullptr, (aclrtStream)stream>>>(       \
            src_handle, dst_handle, total);                                   \
    }

GMM_DEFINE_BIAS_CAST(fp16, half)
GMM_DEFINE_BIAS_CAST(bf16, bfloat16_t)
