---
schema_version: okf.v1
kind: discovered_strategy
id: X13
origin: discovered
title: Shape 自适应双引擎分流 (M-Threshold Hybrid Routing)
description: "当单一执行引擎（cube / vector）无法覆盖全部输入 shape 时，在 host 侧（op plugin）按 shape 特征（如 M 维度）路由到两条完整的独立 kernel 路径：小 M（GEMV 类）走纯 VEC streaming 内核吃带宽极限，中大 M 走 cube 内核吃 MAC 吞吐。分发阈值不靠猜，用实测证据校准：中间区间（如 M∈[3,4]）按 K*N 流量是否足以摊"
confidence: single_op
status: active
op_families: ['matmul']
bottlenecks: ['compute_bound', 'scalar_compute', 'tiling_imbalance']
tags: ['compute_bound', 'matmul', 'quantization', 'scalar_compute', 'tiling_imbalance']
discovered_round: 2
discovered_from: round_2/parallel_1
base_speedup: 2.03x
---
# Strategy X13: Shape 自适应双引擎分流 (M-Threshold Hybrid Routing)

## 核心思路
当单一执行引擎（cube / vector）无法覆盖全部输入 shape 时，在 host 侧（op plugin）按 shape 特征（如 M 维度）路由到两条完整的独立 kernel 路径：小 M（GEMV 类）走纯 VEC streaming 内核吃带宽极限，中大 M 走 cube 内核吃 MAC 吞吐。分发阈值不靠猜，用实测证据校准：中间区间（如 M∈[3,4]）按 K*N 流量是否足以摊销 cube 路径的固定 launch+workspace 开销做二级判定。

## 适用场景
- 瘦矩阵乘 / weight-quant matmul / GEMV-like 算子，输入 shape 跨 GEMV（M=1-8）与 GEMM（M>=32）两个域
- 任一单引擎路径在部分 shape 区间存在结构性短板（cube 小 M 空转 mac_ratio<0.2；VEC 大 M 受指令发射延迟硬上限）
- 瓶颈类型：tiling_imbalance（cube 低利用）或 compute_bound（VEC 发射上限），且 case 集 shape 跨度大

## 实现要点
- 两条路径必须是各自完整正确的实现，合并进同一 kernel 源文件，host tiling 各自独立；路由只换执行引擎不改数值（共享同一精确代数分解，如量化算子的 y = s*(x@w) + (off*s)*rowsum(x) + bias）
- 路由器放 host plugin（按 M 读 tensor shape 选择 tiling 分支），零 device 侧分发开销
- 阈值校准方法：对边界 shape 各跑两路径实测设备时间，3 次复现确认后再固化阈值；固定开销主导的小 shape 让 VEC 路径上浮承接
- workspace 按路径条件分配（VEC 路径可免去 cube 路径的大块 dequant workspace）
- 芯片相关常量一律运行时查询（PlatformAscendC），核数查询结果可 static 缓存避免小 kernel 上反复 host 查询引入可测延迟
- 尾块守卫、DataCopyPad 对齐兜底从父路径继承，不随路由弱化

## 来源
自动发现于第 2 轮进化，算子 level3_weight_quant_batch_matmul，speedup 2.03x（VEC-only 父 0.98x / cube-only 父 1.51x → 混合 2.03x，geomean vs PyTorch）。基于 [[disc_X12]] 的纯 VEC 路径与 n1 系 cube 路径（K1 dequant + K2 MatmulImpl + K3 epilogue）组合分发。
