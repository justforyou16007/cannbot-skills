---
schema_version: okf.v1
kind: discovered_strategy
id: X39
origin: discovered
title: host 侧全写 workspace 免零初始化（at::zeros → at::empty）
description: 算子 host 每次 forward 分配大块 workspace/输出 buffer 且 kernel 写满全部区域时，用 at::empty 替代 at::zeros 消除每调用 GB 级 memset；配合动态 blockDim=min(24,totalTasks)。大 case memory-bound 地板降低，实测大 case 1.35-2.3x。
confidence: single_op
status: active
op_families: [attention, matmul, datapath, host-overhead]
bottlenecks: [memory_bound, launch_bound, no_overlap]
tags: [host-side, workspace, memset-elimination, at-empty, dynamic-core, gqa, prefill]
discovered_round: 1
discovered_from: GqaSparseFwd evolution round_1/parallel_2 (x0 open_exploration)
base_speedup: 大 case case39 2.297x / case40 1.592x / case41 1.353x（vs PyTorch）；geomean 0.405->0.447x
---

# X39: host 侧全写 workspace 免零初始化（at::zeros → at::empty）

## 核心思路
host 侧大块 workspace/输出 buffer 若被 kernel **写满全部区域**，用 `at::empty` 替代 `at::zeros` 消除每次调用的 GB 级 memset。

## 适用场景
- 大 workspace 全写算子（如 flash-attention 类 workspace S/P/O 中转）
- 大 shape case 的 memory-bound 地板；小 case 配合动态核数

## 实现要点
- 逐区域审计"无读前未写"后再改 empty
- 输出 buffer 若 model_new 只读实际写行则安全
- blockDim = min(coreNum, totalTasks)

## 来源
GqaSparseFwd 进化 round_1/parallel_2（x0 open_exploration），50/50 精度过，大 case 1.35-2.3x。
