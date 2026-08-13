---
schema_version: okf.v1
kind: discovered_strategy
id: X44
origin: discovered
title: fp32 Cube mmad 全量重写替代向量模拟 matmul（SetHF32(false) 全精度）
description: 基线内核用 AIV 向量标量模拟 matmul（16×16 tile 逐点 GetValue 累加）完全绕过 Cube 矩阵单元，是数量级性能杀手；改为 Matmul 高阶 API（ASCENDC_CUBE_ONLY + REGIST_MATMUL_OBJ + IterateAll）让 fp32 输入直接走 Cube，SetHF32(false) 获得全精度（MERE ~1e-6）。attention 的 QK^T/PV/W_o 三个 matmul 全部 Cube 重写后 geomean 0.0013x→4.149x vs PyTorch，50/50 PASS。
confidence: single_op
status: active
op_families: ['attention', 'matmul']
bottlenecks: ['scalar_compute', 'launch_overhead', 'compute_bound']
tags: ['cube-mmad', 'vector-simulated-matmul', 'set-hf32-false', 'fp32-cube', 'full-precision', 'attention', 'multi-matmul', 'matmul-highlevel-api']
strategy_ids: [X37]
discovered_round: 1
discovered_from: round_1/parallel_2 (x0, open_exploration)
base_speedup: 0.0013x → 4.149x geomean vs PyTorch（基线 geomean 81μs → 336μs ref；小 shape 2-3x、大 shape 7-9x）
---

# Strategy X37: fp32 Cube mmad 全量重写替代向量模拟 matmul（SetHF32(false) 全精度）

## 核心思路
生成类基线内核常见病态模式：matmul 不用 Cube 矩阵单元，而是用 AIV 向量标量模拟——
每个 16×16 输出 tile 做 K×256 次向量 Mul + 标量 GetValue 累加，完全绕过 Cube，是**数量级
性能杀手**。把算子内全部 matmul（attention 的 QK^T / PV / W_o）改用 Matmul 高阶 API
（`ASCENDC_CUBE_ONLY` + `REGIST_MATMUL_OBJ` + `IterateAll`），fp32 输入直接走 Cube；
910B2C 上 `SetHF32(false)` 关闭 HF32 精度截断获得全精度 fp32（实测 MERE ~1e-6）。

## 适用场景
- 任何 matmul 被实现为向量标量模拟（AIV GetValue 逐点累加）的算子——这是 Tiling 层
  /生成器偷懒绕开 Cube 的典型写法，改 Cube 是跨越式收益
- attention 族多 matmul 流水（QK^T→softmax→PV→W_o），且 fp32 精度路径不再需要
  fp16 hi/lo 拆分（HF32 拆分 3 次 gemm 模拟一次 fp32 gemm 的做法被 SetHF32(false)
  全精度单次 gemm 取代）
- 小/中 shape 为主、gemm 非瓶颈上限的算子：全精度 fp32 Cube 比 HF32 慢，但 shape 小
  时性能损失可接受

## 实现要点
1. **SetHF32(false) 全精度**：910B2C 无原生 fp32 Cube，默认 HF32(true) 只有 10-bit
   尾数，不够 fp32 rtol=1.2e-4；显式 SetHF32(false) 后 MERE ~1e-6，50/50 PASS。
2. **Matmul 高阶 API 走 Cube**：`ASCENDC_CUBE_ONLY` + `REGIST_MATMUL_OBJ` +
   `IterateAll`，AIC 执行；配合单次发射（P54/X24 同类思想）消除 host 侧嵌套 launch 循环。
3. **fp32 向量 op count 必须 ≥8 且为 8 的倍数**：手写 halving 归约在 nPad 非 2 幂时
   越界，改用 WholeReduceMax/Sum（chunked + ORDER_ONLY_VALUE + 树合并）。
4. **uint8 位图 mask 向量操作必须 256-bit block 对齐**：count 须是 256 的倍数，否则
   nPad<256 时 mask 写入不完整 → softmax 输出均匀 1/32（全 -1e9）。
5. **小 seqLen 统一 sR16 ≥ 64**：sR16=16 会触发上述 mask 对齐问题；sR16=max(64,
   ceil(S/16)*16) 后稳定。代价是小 seqLen 的 gemm padding，但 Cube 足够快可承受。
6. **exp 必须保留行 max 减法**：输入符号相反时 logit 可达 -141，无 max 减法则 exp
   下溢为 0 → 行和 0 → NaN。

## 来源
自动发现于第 1 轮进化（LongformerAttention），round_1/parallel_2（x0，open_exploration
不读策略库从第一原理推理）。7 次发射流水 lf_prep(AIV)→lf_gemm_nt_fp32(AIC QK^T)→
lf_softmax(AIV)→lf_gemm_nn_fp32(AIC PV)→lf_merge(AIV)→lf_gemm_nt_fp32(AIC W_o)→
lf_bias(AIV)，三个 matmul 全部 Cube 重写。geomean 0.0013x→4.149x（基线 368ms 标量模拟
+ ~1000 次串行 host launch → 最优 81μs），50/50 PASS，MERE ~1e-6。
