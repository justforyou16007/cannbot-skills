---
schema_version: okf.v1
kind: discovered_strategy
id: X14
origin: discovered
title: Mix-Kernel 跨核生产者/消费者融合 (Pairwise AIV-Producer/AIC-Consumer Fusion)
description: 当流水线中存在独立的前置 VEC pass（如 dequant/cast），且硬件路径（int8 直接进 cube）被证伪时，将该 VEC pass 与 cube kernel 融合进**单个 mix kernel**：AIC ci 与其配对 AIV（2ci, 2ci+1）组成生产者/消费者对，AIV 按 AIC 的消费顺序（如 (nIdx,kIdx) strip）生产中间数据到 write-on
confidence: single_op
status: active
op_families: ['matmul']
bottlenecks: ['compute_bound', 'l2_cache_thrash', 'mte2_stall', 'mte3_stall', 'no_overlap', 'scalar_compute']
tags: ['compute_bound', 'l2_cache_thrash', 'matmul', 'mte2_stall', 'mte3_stall', 'no_overlap', 'quantization', 'scalar_compute']
discovered_round: 3
discovered_from: round_3/parallel_1
base_speedup: 2.44x
---
# Strategy X14: Mix-Kernel 跨核生产者/消费者融合 (Pairwise AIV-Producer/AIC-Consumer Fusion)

## 核心思路
当流水线中存在独立的前置 VEC pass（如 dequant/cast），且硬件路径（int8 直接进 cube）被证伪时，将该 VEC pass 与 cube kernel 融合进**单个 mix kernel**：AIC ci 与其配对 AIV（2ci, 2ci+1）组成生产者/消费者对，AIV 按 AIC 的消费顺序（如 (nIdx,kIdx) strip）生产中间数据到 write-once GM workspace，用 **GM flag 协议**同步（128B 步长 flag、MTE3 DataCopyPad 发布、AIC 侧 `__gm__ volatile` 有界轮询 + DCCI CACHELINE_ALL），实现跨核软件流水，launch 数减 1 且 cast 与 cube 计算重叠。

## 适用场景
- 三 kernel（或多 kernel）串行流水线，前置 VEC pass 占 kernel 时间 >30%（如量化 matmul 的独立 dequant pass）
- 硬件融合路径不可用（芯片特性表证伪，如 DAV_2201 ifMmadInstrSupportAntiQuant=false）
- c220 等平台上 CrossCoreSetFlag/WaitFlag（PIPE_S）非法、用户态 ffts 握手不成立时，GM-flag 是可靠替代
- 瓶颈类型：no_overlap / mte2_stall，且 profiling 显示独立 pass 与主计算串行

## 实现要点
- 配对关系先上板探针验证（910B2C：AIC ids [0,24)、AIV ids [0,48)、GetBlockNum()=24）
- 生产顺序必须与消费顺序严格一致（strip 序）；write-once workspace 避免读写竞争
- 三大坑（实测）：(1) AIC 侧标量 GM 轮询读会被编译器 CSE 提升，必须 `__gm__ volatile` 指针读，否则空转；(2) AIV 标量 GM 写会丢失，必须用 MTE3 DataCopyPad 发布 flag；(3) 跨核 wait 是聚合阈值语义，SyncAll 为正确性必需（裸 set/wait 可致精度全挂）
- 生产者侧用 TQue 深度 2 流水化进一步隐藏 cast 延迟
- 保留 legacy 多 kernel 路径作环境变量回退（如 WQBM_LEGACY_CUBE=1），平台路由兜底

## 来源
自动发现于第 3 轮进化，算子 level3_weight_quant_batch_matmul，speedup 2.44x（父 2.03x，+20.2%；msprof 实测 wqbm_dequant_kernel launch=0，原占 38.5%）。同轮 n1_s1_cont 变体以 stage-flag 变体（2/8 阶段 cast + CrossCoreSetFlag<0x2, PIPE_MTE3> + SyncAll）在 cube-only 路径独立验证同思路（1.51x→1.92x），两种同步原语均可行。与 [[disc_X13]]（双引擎分流）正交可叠加。
