---
schema_version: okf.v1
kind: discovered_strategy
id: X43
origin: discovered
title: Cube 输出 FixPipe 融合 Epilogue（C_TYPE=输出 dtype + SetBias，acc 不落 GM）
description: Matmul 流水最后一段 L0C→FixPipe→GM 输出通路。C_TYPE 直接设为输出 dtype（fp16/bf16）+ EnableBias/SetBias，在 FixPipe 内做 fp32 acc + bias + F322F16/F322BF16 写回，消除独立 epilogue kernel + acc fp32 [M,N] 的 GM 往返。与 Split-K 互斥（低精度 C 上 enAtomic 损坏）；SetBias 需 per-N-tile 全局列偏移（SetBias(biasGlobal[nIdx*singleN])）；一个 ascendc_library 只放一个 MIX cube kernel。可与 X14 dequant 融合叠加成单 MIX kernel。
confidence: two_independent
status: active
op_families: ['matmul', 'cv_fusion', 'quantization']
bottlenecks: ['mte3_stall', 'no_overlap', 'compute_bound']
tags: ['cv-fusion', 'epilogue-fusion', 'fixpipe', 'cube-output', 'launch-reduction', 'matmul', 'SetBias', 'acc-no-gm-roundtrip']
strategy_ids: [X36]
discovered_round: 1
discovered_from: round_1/parallel_0 (wf1) + round_1/parallel_2 (x0)
base_speedup: 0.6331x vs PyTorch（基线 geomean 0.451x，+40%；两路独立发现 0.620x/0.633x）
---

# Strategy X36: Cube 输出 FixPipe 融合 Epilogue（C_TYPE=输出 dtype + SetBias，acc 不落 GM）

## 核心思路
Matmul 流水最后一段是 L0C → FixPipe → GM 的输出通路。把独立 epilogue kernel（bias-add +
cast + requant）下沉进 cube 的 FixPipe 阶段：**C_TYPE 直接设为输出 dtype（fp16/bf16）**，
`EnableBias` + `SetBias` 在 FixPipe 内做 fp32 acc + bias 后再 F322F16/F322BF16 写回 GM。
效果：**acc fp32 [M,N] 完全不落 GM**，消除 epilogue kernel launch（launch 数 -1）+ acc 的
GM 写/读往返。对 3-kernel 串行流水（dequant→matmul→epilogue）是第一个结构性消除点。

## 适用场景
- matmul + 向量后处理（bias/cast/轻量 elementwise）的 CV 融合算子，后处理不含跨行归约
- 独立 epilogue kernel 占 kernel 时间显著（小/中 shape 尤其明显，launch + GM 往返开销大）
- 输出 dtype 与 L0C acc dtype 不同的算子（fp32 acc → fp16/bf16 输出，最典型）

## 实现要点
1. **C_TYPE = 输出 dtype + EnableBias(hasBias)**：quantMode==0（无 requant）路径。C_TYPE
   从 fp32 改为输出 dtype，FixPipe 在 L0C fp32 上自动加 bias 并 cast 后写 GM。
2. **SetBias 必须 per-N-tile 全局列偏移**：`mm.SetBias(biasGlobal[nIdx*singleN])` —— Matmul
   内置 bias 的基址偏移只按核内 fragment 索引计算，grid-stride 切分下不自动按 nIdx 列块
   偏移，否则多 N-tile 时 bias 加错列（实测 max_abs=12-16）。
3. **一个 ascendc_library 只能放一个 MIX cube kernel**：融合入口需放独立 ascendc_library，
   否则同库多 MIX 入口导致 kernel 注册被破坏（实测输出全零）。
4. **保留 quantMode>=1 fallback**：requant 场景仍走 fp32 acc + 独立 epilogue。
5. **互斥检查**：与 Split-K（P72）互斥——fp16/bf16 输出 C 上 enAtomic GM 原子累加会损坏结果。
6. **与 dequant 融合叠加**：可与 X14（AIV producer/AIC consumer）或 SyncAll 单 MIX kernel
   叠加，把 3-kernel 压到 1-kernel。

## 来源
自动发现于第 1 轮进化（WeightQuantBatchmatmul），wf1（workflow_guided, M7/P17 路径）与
x0（open_exploration）两路独立实现同手法：3-kernel→2-kernel（wf1）或 3-kernel→1-kernel
（x0，叠加 dequant 融合），geomean 0.451x→0.62x/0.633x（+37-40%），10/10 bit-exact。
