---
schema_version: okf.v1
kind: discovered_strategy
id: X10
origin: discovered
title: 设备侧分组调度 + NZ plane 权重缓存（Grouped Matmul Cube 化）
description: Grouped matmul（group_list 为 device 端 cumsum）的 Cube 化不要在 host 侧展开分组计划：每个 AIC 核在设备侧自行从 group_list 构建本核的 (group, M-block) 调度（empty group 自然跳过，无 host 同步）；每个专家的 weight 以 NZ plane 形式作为静态缓存放入 GM workspace，ti
confidence: single_op
status: active
op_families: ['matmul', 'moe']
bottlenecks: ['compute_bound', 'mte3_stall', 'no_overlap', 'scalar_compute', 'tiling_imbalance']
tags: ['compute_bound', 'matmul', 'moe', 'mte3_stall', 'no_overlap', 'quantization', 'scalar_compute', 'tiling_imbalance']
discovered_round: 1
discovered_from: round_1/parallel_1
base_speedup: 4.8213x (vs PyTorch golden; baseline kernel 0.0026x)
---
# Strategy X10: 设备侧分组调度 + NZ plane 权重缓存（Grouped Matmul Cube 化）

## 核心思路
Grouped matmul（group_list 为 device 端 cumsum）的 Cube 化不要在 host 侧展开分组计划：每个 AIC 核在设备侧自行从 group_list 构建本核的 (group, M-block) 调度（empty group 自然跳过，无 host 同步）；每个专家的 weight 以 NZ plane 形式作为静态缓存放入 GM workspace，tile 排序采用 m-fastest，使多核共享同一 weight strip，weight 流量从朴素 M×K×N 降到约 ceil(m_g/baseM)×K×N（约 2 个数量级）。epilogue（dequant/激活/再量化）放独立 AIV kernel 全向量流水完成。

## 适用场景
- GroupedMatmul / MoE 类算子（group_list cumsum 语义、group 大小不均、存在 empty group）
- 基线为 AIV 向量模拟 matmul、权重零复用的场景（收益为数量级）
- 需要保持任意 group 分布正确性且不能引入 host-device 同步的场景

## 实现要点
- AIC kernel：MatmulImpl int8×int8→int32；设备侧读 group_list 建调度；64×256 tile（L0C 双缓冲上限；baseM=128 在 dbL0C=2 溢出、dbL0C=1 结果错，不可用）
- **fixpipe NZ 写回 n1 步长 = orgM×16 原始值（不对齐 16）**——按 align16 读取会在 orgM≥16 非对齐时每行 col≥16 全错
- AIV epilogue kernel：TQue 双缓冲流水，NZ 行重组 → fp32 dequant（weight_scale UB 缓存）→ 高层 Silu API（注意非原地安全，内部 Div(dst,src,dst)）→ Max 折叠 absmax → Div 量化 → 队列化 DataCopyPad 输出
- Host：MatmulApiTiling 建 tiling，GM blob 多槽 ring 传入；cube 不可用（E>64/N>10240 等）时回退 VEC kernel

## 来源
自动发现于第 1 轮进化，算子 level4_grouped_matmul_swiglu_quant（cann-bench submit 基线），speedup 4.8213x vs PyTorch（基线 0.0026x），精度 20/20
