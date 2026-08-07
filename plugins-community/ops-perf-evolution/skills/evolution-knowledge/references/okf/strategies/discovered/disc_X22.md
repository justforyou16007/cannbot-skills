---
schema_version: okf.v1
kind: discovered_strategy
id: X22
origin: discovered
title: Depthwise 直接卷积（绕过 im2col+GEMM）
description: depthwise(groups==Cin, cig==1) 卷积用单 AIV fp32 向量卷积内核直接计算逐通道 kh*kw dot product，替代 im2col+GEMM 4 阶段流水线——无 im2col、无 GEMM、无 fp16 分裂、无 GM workspace、无 Cube。Tile=(image,co-block8,oh-block) 保证 32B 对齐存储；对齐 interior 向量化 + 标量 head/tail 独立 corr 缓冲消除 S/V hazard。
confidence: single_op
status: active
op_families: ['convolution']
bottlenecks: ['scalar_compute', 'mte3_stall', 'compute_bound']
tags: ['depthwise', 'conv2d', 'vector-conv', 'im2col-bypass']
discovered_round: 1
discovered_from: round_1/parallel_2
base_speedup: 0.1104x
---

# Strategy X22: Depthwise 直接卷积（绕过 im2col+GEMM）

## 核心思路
depthwise（groups==Cin, cig==1）卷积在 im2col+GEMM 流水线中极低效：GEMM 退化为 M×Kpad @ Kpad×1 逐通道点积，fp16 hi/lo 分裂使 Cube 工作量 3x，且 4 阶段流水线物化 4 个 GM workspace。X22 用**单 AIV fp32 向量卷积内核**直接计算逐通道 kh*kw dot product：无 im2col、无 GEMM、无 fp16 分裂、无 GM workspace、无 Cube。

## 适用场景
- **算子类型**: depthwise 卷积（groups==Cin, cig==1），且 sW==1、Wo>=8、W>=8、Co%8==0、Co>=8 时收益最大
- **瓶颈类型**: im2col 标量 gather、GM workspace 物化、Cube 利用率极低（GEMM M×1）
- **预期收益**: depthwise gen-time 提升 18-84x（实测 case7 2694→32us, case6 1500→49us, case45 305→17us）

## 实现要点
1. **Tile = (image, co-block=8, oh-block)**：co-block 8 使 NMC [B,M,Co] 输出 32B 对齐（blockLen=8*4B）；acc 按 co-major [8][plane] 计算，UB 内 transpose 到 [m][8]，一次 2D DataCopyPad 存储（dstStride=(Co-8)*4）
2. **对齐 interior + 标量 head/tail 分离**：32B 对齐 interior 用向量 Muls/Add（per-tap 对齐 tapBuf 从 GM DataCopyPad 加载）；<=7 elems 的 head/tail 写入独立 corr 缓冲（scalar S-pipe），oh-block 末尾一次向量 Add 并入——消除 acc 的 S/V hazard，热循环只需 MTE2_S/MTE2_V/V_MTE2 事件等待
3. **fp32 精确累加**（rel_p99 ~3e-6）
4. **平台坑**: `PipeBarrier<PIPE_S>` 在 910B2C 崩溃 → 用 `PipeBarrier<PIPE_ALL>`；DataCopyPad UB 目标必须 32B 对齐（weight 缓冲按 RUp8(kH*kW) 步长）；corr 清零需覆盖完整 plane
5. **Host 门控**: cig==1 && sW==1 && Wo>=8 && W>=8 && Co%8==0 && Co>=8 走 fused 内核，其余回退通用流水线

## 来源
自动发现于第 1 轮进化，算子 7_ConvStandard2d，speedup 0.099→0.1104x（depthwise case 提升 18-84x）

## Grouped 推广（round 2 更新）
把 depthwise 直接卷积推广到 general grouped conv（groups>1, cig∈[4,8]）：8 个输出通道块（同一 group）共享 CiG 输入行，
**CiG 通道维 2D DataCopyPad burst**（srcStride=H*W-W, 一条 MTE2 事务取全部 CiG 行）消 undersize transfer；
kW tap burst 合并到一条 MTE2_V 等待、FMA 合并到一条 V_MTE2 等待。实测 7_ConvStandard2d geomean 0.1104→0.1578（+45%）。
关键门控：W%8==0（sub-32B DataCopyPad 在 910B2C 崩溃）、cog%8==0、Co%8==0、Wo%8==0、cig*kk<=128。
