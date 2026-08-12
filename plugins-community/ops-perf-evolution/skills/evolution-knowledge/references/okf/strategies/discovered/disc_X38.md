---
schema_version: okf.v1
kind: discovered_strategy
id: X38
origin: discovered
title: GM-Workspace-Ring KV-tile 宽度缩放 + Host 批量权重打包
description: "针对采用 GM workspace ring 队列（AIC 写 S/P/O 经 GM 中转给 AIV）的 FA 算子，每 KV tile 固定握手开销主导（AIC scalar_ratio~0.8, AIV~0.7, cube MAC 仅 4-6%）时：1) kvN 自适应（seq<=64 用 64 避免尾块浪费，seq>64 用 128）直接减半 GM ring 往返次数 + 翻倍 MM1/MM2 cube GEMM 尺寸，workspace/buffer 按 128 max 分配仅 slot stride/MM n-k/fixpipe/softmax srcK/mask 变运行时 kvN，sPad align_up(seq,kvN) 修越界；2) host 侧融合投影的 per-head 权重打包 copy-storm（48+ 次 narrow().copy_()，fp32 case 占墙钟~40%）批量化为 cat+pad+reshape+单次 strided copy（heads 多时），输出改 at::empty。FlashAttentionV2 mean 0.579→0.856（geomean 0.530→0.794）。"
confidence: single_op
status: active
op_families: ['flash_attention', 'attention', 'matmul']
bottlenecks: ['scalar_loading', 'scalar_compute', 'mte2_stall', 'undersize_transfer', 'compute_bound']
tags: ['flash_attention', 'workspace-ring', 'kv-tile', 'host-packing', 'scalar_loading', 'cube-efficiency', 'causal', 'gqa']
discovered_round: 1
discovered_from: round_1/parallel_2
base_speedup: 0.856x
---

# Strategy X38: GM-Workspace-Ring KV-tile 宽度缩放 + Host 批量权重打包

## 核心思路
FA 算子若采用 GM workspace ring 队列（AIC 写 S/P/O → GM → AIV 读）而非 L1 常驻，则每 KV tile 产生固定次数的 GM 往返 + 跨核握手 + 标量开销。此时放大 KV tile 宽度（kvN）直接减半 GM ring 往返次数、翻倍 MM1/MM2 的 cube GEMM 尺寸。同时，融合 QKV/out 投影算子的 host 侧 per-head 权重打包构成独立 copy-storm，可批量化为 cat + pad + reshape + 单次 strided copy。

## 适用场景
- FA 算子使用 GM workspace ring 握手（AIC↔AIV 经 GM 中转 S/P/O），每 KV tile 固定握手开销主导
- 融合投影 GEMM 且 heads 数多（per-head copy 次数 > 12），host 侧权重打包成为墙钟拖累
- 硬件：Ascend 910B 系列 MIX AIC/AIV

## 实现要点
- kvN 自适应：seq<=64 用 64；seq>64 用 128。workspace/buffer 按 128 max 分配，slot stride/MM n-k/fixpipe width/softmax srcK/mask/causal 对角索引改运行时 kvN。
- sPad align_up(seq, kvN) 修末 tile 越界（NaN 根因）。
- host 批量打包：heads 多时 cat + constant_pad_nd + reshape + 单次 strided copy；heads 少时保留逐 head copy（pad 固定开销 30-50us 拖累小 shape）。输出改 at::empty。

## 来源
自动发现于第 1 轮进化，算子 FlashAttentionV2，round_1/parallel_2（open_exploration）：mean 0.579→0.856（geomean 0.530→0.794）。代表 case：case 35 bf16 [1,463,1280] 0.423→0.767x，case 20 causal 0.674→1.026x，case 34 fp16 0.860→1.499x。
