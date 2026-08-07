---
schema_version: okf.v1
kind: discovered_strategy
id: X24
title: "X24: AIC_ONLY 任务类型压设备 launch 地板（固定开销主导算子）"
origin: discovered
discovered_round: 2
discovered_from: "5_MatmulTransB evolution (manual mmad 攻坚轮), round 2, parallel 2 (x0_x1)"
base_speedup: 1.757x vs 基线 / 0.616x vs PyTorch
confidence: single_op
status: active
op_families: [matmul, elementwise, reduction]
bottlenecks: [scalar_compute, no_overlap, compute_bound]
tags: [aic-only, device-launch-floor, fixed-overhead-dominated, tiny-shape, KERNEL_TYPE]
strategy_ids: [X24]
---

### X24: AIC_ONLY 任务类型压设备 launch 地板（固定开销主导算子）

**Source**: 5_MatmulTransB evolution（手动 mmad 攻坚轮），round 2, parallel 2（x0_x1, open_exploration）
**Speedup**: 设备侧 12.09μs → 6.88μs（1.757x vs 基线 / 0.616x vs PyTorch，50/50 精度 PASS）；单 case 最高 1.31x（[25,50] bf16）
**Operator type**: 微小 shape matmul（M∈[1,512], N∈[1,1024], K∈[1,1024]），固定开销主导
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
小 shape、固定开销主导的算子（kernel 时间与 shape 几乎无关，cube_util 仅 3.95%、aic_scalar_ratio 0.661），其设备时间大部分是「每 AIC 调度启动」的固定开销而非计算。X24 通过把任务类型从默认的 `KERNEL_TYPE_MIX_AIC_1_2`（1 AIC + 2 AIV 向量核协同）改为 **`KERNEL_TYPE_AIC_ONLY`**（纯 cube 核，不启动 AIV 向量核），去掉每个 AIC 调度的 2 个空转 AIV 任务启动，**把设备固定地板从 ~7-8μs 压到 ~4.3-4.5μs**（实测，最优变体核心来源）。配合：紧凑 tiling 加载（GM 标量读取从 ~65 int32 削减到 4 int32）、单 tile 整块装入 L0 手动 mmad，共同逼近 910B 的 kernel launch 物理地板。

**Key code change**:
```cpp
// 任务类型声明（*_fp16.cpp / *_bf16.cpp / *_fp32.cpp 的 kernel 注册处）：
// Before: 默认 MIX（1 AIC + 2 AIV），AIV 核空转等待
// After: 纯 cube，去掉 AIV 任务启动
KERNEL_TYPE_AIC_ONLY  // 见 k3_platform 宏 / SetTaskType
// 实测效果：每 launch 省 ~1.5-2us；叠加 compact tiling 头（前 4-12 int32）
// 后，fp32[1,1] 从 MatmulImpl ~13us 降至手动 ~6.1us
```

**Applicability conditions**:
- **固定开销主导**的算子（判断依据：不同 shape 的 kernel 时间几乎相同、cube_util 远低于 scalar 占比）
- 小 shape、单 tile 可装入 L0（blockM×blockN、blockM×blockK、blockN×blockK ≤ L0 容量）
- 无 AIV 侧必做工作（纯 cube 计算，无需向量/标量协同 epilogue）时收益最大
- 与手动 mmad / 低开销 tiling 加载叠加效果最佳；对计算密集大 shape 无显著收益

**Strategy IDs**: X24
