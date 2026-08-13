---
schema_version: okf.v1
kind: proven_solution
id: x38
title: "fp32-on-Cube：host 预拆分 fp16 hi/lo 3-GEMM 绕跨核 workspace 可见性缺陷"
source: "14_WindowAttention evolution, round 7, parallel 1 (r7a)"
speedup: "fp32 geomean 0.272x → 0.866x（+3.2x）；整体 geomean 0.49x → 1.135x vs PyTorch；50/50 精度 PASS（fp32 MERE 8e-6~6e-5）"
operator_type: "attention（QK^T → softmax → PV 多 matmul 流水 + QKV/输出投影）"
hardware: "Ascend910B2C (DAV_2201)"
confidence: single_op
status: active
op_families: [attention, flash_attention, matmul]
strategy_ids: [P128, P95]
bottlenecks: [compute_bound, scalar_compute]
tags: [fp32-on-cube, hilo, host-presplit, fp16, 3-gemm, mmad, barrier, crosscore, cube, attention]
---

# X38: fp32-on-Cube —— host 预拆分 fp16 hi/lo 3-GEMM 绕跨核 workspace 可见性缺陷

## Source
14_WindowAttention evolution, round 7, parallel 1 (r7a)

## Speedup
fp32 geomean **0.272x → 0.866x**（+3.2x，基线纯 AIV 标量 FMA）；整体 geomean
**0.49x → 1.135x** vs PyTorch；50/50 精度 PASS（fp32 MERE 8e-6~6e-5，阈值 1.22e-4）。

## Technique description
FP32 路径（K1 QKV 投影 / K2 QK^T+softmax+PV / K3 输出投影）从纯 AIV 标量 GetValue
FMA 反模式迁移到 Cube 3-GEMM。核心创新是**绕开 P128 的跨核 workspace 可见性缺陷**：

1. **host 预拆分**：fp32→hi/lo 拆分成在 host（torch cast），kernel launch 边界全同步，
   AIC 直接读预拆分 fp16 面板，与读原生数据等价。彻底规避 r6a 确认的「AIV MTE3 写 →
   AIC MTE2 读大 workspace L2 可见性非确定」缺陷。
2. **fp16 而非 bf16**：bf16 2-way 拆分精度 ~2^-17 不够（MERE~2e-4 > 1.22e-4 阈值，
   6/17 失败）；fp16 有 10-bit 尾数，2-way 给 ~2^-20（MERE~6e-6）。fp16 指数范围
   ±65504 足够（Q/K/V ~±17，P∈[0,1]），乘积在 L0C 以 fp32 累加无溢出。
3. **3 次 Mmad 间必须加 `PipeBarrier<PIPE_ALL>`**：无 barrier 时 hdp=16 case 的 GEMM
   输出变垃圾（Cube L0C 累加 WAR 依赖），是 r4a 之前 1e36 爆炸的同源根因。

## Key code change
```cpp
// Before: FP32 纯 AIV 标量 FMA（GetValue 逐元素 + Muls/Add 向量链），绕过 Cube
// After:  host 预拆分 fp16 hi/lo 面板，AIC 3-GEMM（hi·hi + hi·lo + lo·hi）
for (kt tile) {
    LoadNdGmToNzL1(qhi, qlo, khi, klo, ...);   // 预拆分 fp16 面板，无 AIV→AIC workspace
    Mmad(sL0, a2_hi, b2_hi, mp);               // hi·hi
    PipeBarrier<PIPE_ALL>();                   // ★ 关键：L0C WAR 依赖必须 barrier
    Mmad(sL0, a2_hi, b2_lo, mp_noinit);        // hi·lo
    PipeBarrier<PIPE_ALL>();
    Mmad(sL0, a2_lo, b2_hi, mp_noinit);        // lo·hi
    PipeBarrier<PIPE_ALL>();
}
```

## Applicability conditions
- fp32 输入、Cube 仅吃 fp16/bf16 的 GEMM 类算子，且需多核 MIX_AIC_1_2 流水
- 原实现为 AIV 标量 FMA 反模式（scalar_compute 瓶颈）
- host 侧可做 fp32→fp16 hi/lo 预拆分（额外 GM 流量换取 Cube 吞吐）
- 容差 fp32 rtol≈1e-4：必须用 fp16 hi/lo（bf16 hi/lo 精度不够，见 f4）

## 后续方向（更高优先）
- **X44 的 SetHF32(false) 更优**：LongformerAttention 已证 910B2C 用 Matmul 高阶 API +
  `SetHF32(false)` 可直接全精度 fp32 Cube（MERE~1e-6），**无需 hi/lo 拆分**（省 3x 工作量）。
  若 WindowAttention 的 fp32 matmul 也走 Matmul 高阶 API + SetHF32(false)，应比本方案的
  fp16 hi/lo 3-GEMM 更快。本方案仅在手写 Mmad 报 fp32 NaN、且未试 Matmul 高阶 API 的
  情况下作为可用的中间态。

## Related
- f6（跨核 workspace 可见性缺陷）
- f4（bf16 hi/lo 精度不足）
- X17 / P128（hi/lo 3-GEMM 算法本体）
- X44（SetHF32(false) 全精度 fp32 Cube，更优）
- x32（GroupedQueryAttention 投影 GEMM 的 fp16 hi/lo 落地）
