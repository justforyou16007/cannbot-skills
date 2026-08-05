---
schema_version: okf.v1
kind: discovered_strategy
id: X15
origin: discovered
title: "递归拆分-并行投影外提 (Recurrent Split: Hoist Parallel Projection)"
description: "对 RNN/LSTM/GRU/scan 类含串行循环的算子，将计算拆为\"非递归部分\"与\"严格串行部分\"： 非递归投影（如全部时间步的 x@W_ih）与循环无关，可一次性合并为单个大 GEMM，外提到 host 侧 调用框架/库的高性能 GEMM（自动走 Cube 单元）；自定义内核只保留严格串行的递归残差 （如每步 h@W_hh + 门激活 + 状态更新），并将递归权重（W_hh）在 UB 常驻、"
confidence: single_op
status: active
op_families: ['matmul']
bottlenecks: ['compute_bound', 'mte3_stall', 'scalar_compute']
tags: ['compute_bound', 'matmul', 'mte3_stall', 'scalar_compute']
discovered_round: 1
discovered_from: round_1/parallel_1
base_speedup: 0.852x
---
# Strategy X15: 递归拆分-并行投影外提 (Recurrent Split: Hoist Parallel Projection)

## 核心思路
对 RNN/LSTM/GRU/scan 类含串行循环的算子，将计算拆为"非递归部分"与"严格串行部分"：
非递归投影（如全部时间步的 x@W_ih）与循环无关，可一次性合并为单个大 GEMM，外提到 host 侧
调用框架/库的高性能 GEMM（自动走 Cube 单元）；自定义内核只保留严格串行的递归残差
（如每步 h@W_hh + 门激活 + 状态更新），并将递归权重（W_hh）在 UB 常驻、超规格时回退分块。
串行链上的计算量从 O(seq×(I+H)×4H) 降到 O(seq×H×4H)，且大头走了 Cube。

## 适用场景
- 含时间步/序列串行依赖的算子（LSTM/GRU/RNN/线性 scan/指数衰减累加）
- 每步同时含"只依赖当前输入的投影"与"依赖上步状态的递归投影"
- 小 batch 场景自定义 Vector AXPY 模拟 GEMV 远低于框架 GEMM 效率时收益最大

## 实现要点
- 外提部分：将 x 重排为 (seq×batch, I) 单次 matmul，输出 permute 成 gate-major [seq,4,batch,H]
  便于内核按门连续访问；fp16 输入建议 fp32 累保精度
- 内核部分：递归权重 W_hh (H×4H) 按 UB 余量切 chunk 常驻（H≤512 fp16 可全驻留 2MB 内）；
  每步只做 gates += h @ W_hh^T（向量 AXPY 或小块 matmul）+ sigmoid/tanh + c/h 更新
- 回退路径：W_hh 超 UB 容量时按 chunk 双缓冲轮转重载，或回退 host 分步 GEMM
- 注意评测口径：外提 GEMM 在 ModelNew.forward 内、计时区间内，不算作弊；但数学必须完整

## 来源
自动发现于第 1 轮进化，算子 LSTM（910B2C），mean speedup 0.202x→0.852x（geomean 0.772x），
小 batch 用例提升最显著（[1,32,768] 0.33x→1.31x）
