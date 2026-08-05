---
schema_version: okf.v1
kind: discovered_strategy
id: X20
origin: discovered
title: Host 侧权重预打包结果 LRU 缓存（Weight Repack Cache）
description: 权重布局变换（K-major repack、fractal 重排、量化打包等）的输出是 （weight 内容, shape, dtype, 目标布局）的**确定函数**。推理场景同一权重张量 被反复卷积/矩阵乘，repack 结果逐 bit 相同——host 侧做 LRU 缓存，命中时 **跳过整个 repack kernel 发射**，直接复用缓存的 workspace。
confidence: single_op
status: active
op_families: ['cv_fusion', 'elementwise', 'matmul']
bottlenecks: ['l2_cache_thrash', 'no_overlap']
tags: ['conv2d', 'cv_fusion', 'elementwise', 'l2_cache_thrash', 'matmul', 'no_overlap', 'quantization']
discovered_round: 2
discovered_from: conv2d_base (tilelang2ascendc-ops-generator) + level3_conv_2d-cannbench-web-submit
base_speedup: 省去整个 wrepack kernel 发射（每次调用 ~5-15us 级，推理式重复调用全命中）
---
> **已晋升为 P131**（card + precondition + playbook 三件套见 `cards/`、`preconditions/`、`playbooks/`）。本文件保留为实战详细记录（含实测数据与否定项依据），卡片内容以 P131 为准。


# Strategy X20: Host 侧权重预打包结果 LRU 缓存（Weight Repack Cache）

## 核心思路
权重布局变换（K-major repack、fractal 重排、量化打包等）的输出是
（weight 内容, shape, dtype, 目标布局）的**确定函数**。推理场景同一权重张量
被反复卷积/矩阵乘，repack 结果逐 bit 相同——host 侧做 LRU 缓存，命中时
**跳过整个 repack kernel 发射**，直接复用缓存的 workspace。

## 适用场景
- **算子类型**: 任何带权重预打包步骤的算子（conv、matmul、quant matmul），
  推理式重复调用模式
- **瓶颈类型**: launch-bound；或 repack kernel 本身耗时占比可测
- **前提条件**: repack 无随机性、无输入数据依赖；same-stream 执行
  （workspace 的流序复用假设，与 per-call workspace 相同）
- **预期收益**: 每次调用省 1 次 kernel 发射 + repack 计算（~5-15us 级）；
  评测/推理循环中命中率 ~100%

## 实现要点
1. 缓存键：`data_ptr + sizes + dtype + version counter`——**version 必需**：
   in-place 修改（`copy_`/`fill_` 等）会 bump version 强制重打包，否则脏数据
2. **条目内持有 weight 强引用**：防止 allocator 回收 data_ptr 后被新张量
   复用地址造成 stale 命中
3. 缓存的 workspace 张量由条目持活，miss 时整体重写；LRU 容量小（个位数条目）
4. 多目标布局（如 normal k 序 vs fused k 序，见 X12）时**布局标志进缓存键**
5. 留环境变量开关（如 `*_WREPACK_CACHE=0`）便于 A/B 与排障
6. 纯 host C++/ATen 逻辑，无 device 侧改动，全平台可移植

## 来源
conv2d_base P1-a 首验（推理权重不变场景净赚）；cann-bench conv_2d 评测
（同一 case 反复调用）全命中，四级流水下 wrepack 从每调用一次变为仅首调。
