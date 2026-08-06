---
schema_version: okf.v1
kind: proven_solution
title: "X23: Batched Spatial 全宽向量化 + 编译器驱动标量预取 (ConvStandard3d)"
source: "8_ConvStandard3d evolution, round 1, parallel 2 (x0, open_exploration)"
speedup: "geomean 0.36x → 3.51x（60/60 精度 PASS）；case46 3599us→5.1us"
operator_type: "conv (kw==1, W 维逐点 FMA), fp32 pure-Vector"
hardware: "Ascend910B2C (DAV_2201)"
confidence: single_op
status: active
op_families: [conv, cv_fusion, elementwise]
strategy_ids: [X23]
bottlenecks: [scalar_loading, scalar_compute]
tags: [batched-spatial, s-pipe-latency-hiding, pragma-unroll]
---

### X23: Batched Spatial 全宽向量化 + 编译器驱动标量预取

**Source**: 8_ConvStandard3d evolution, round 1, parallel 2 (x0, open_exploration)
**Speedup**: geomean 0.36x → 3.51x（60/60 精度 PASS）；常量标量诊断实验 case46 3599us→21us（~170x）
**Operator type**: conv (kw==1, W 维逐点 FMA), fp32 pure-Vector
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
滑窗卷积按「逐输出位置」切任务时 Axpy 只利用 1/4~1/8 向量 lane（W∈{8..32}, PAR=64），且每个 (co,tap) 一次权重 GetValue S-pipe 标量读（~43cy/op）完全串行化。X23 把多个连续空间位置打包进全宽向量：task=(n,cg,od)，acc[G,Hout*W] 覆盖整条 od 行，P=floor(64/W) 个连续 oh 打包进 64-lane fp32 向量，每 (co,tap) 1 次权重标量读驱动 P*W 宽 Axpy。Boundary 用零填充 + strided DataCopyPad 加载合法子区间，compute 统一全宽（零有效性判断）。Store 用 3 次 strided DataCopyPad 避免 (W,G)->(G,W) 标量转置。**核心使能**：内层 co-loop `#pragma unroll 4` 驱动编译器对 GetValue(Axpy) 链软件流水，隐藏 ~43cy S-pipe 延迟。⚠️ 手工多标量预读局部变量会触发编译器误编译（结果错误、速度虚高），必须保持 GetValue 紧跟其 Axpy 只用 pragma 驱动。

**Key code change**:
```cpp
// Before: task=(n,cg,od,oh), acc(G,W), 每 (co,tap) GetValue + W 宽 Axpy
for (g...) { AscendC::Axpy(acc[g*WP_], xLocal_[tt*WP_], wLocal_.GetValue(g*TcP_+tt), W_); }

// After: task=(n,cg,od), acc[G, Hout*W], P 个 oh 打包进 64-lane 全宽向量
#pragma unroll 4   // ← 关键：编译器软件流水隐藏 S-pipe 标量读延迟
for (g...) { AscendC::Axpy(acc[g*plane_], xBatch_[0], wLocal_.GetValue(g*K_+tap), P*W_); }
// xBatch_ Duplicate 清零 + strided DataCopyPad 加载合法 oh 子区间；compute 统一全宽
```

**Applicability conditions**:
- 滑窗卷积 kw==1 / W 维逐点 FMA，W∈[8,32] 且 W%8==0（32B 对齐），G 路 Axpy 形态
- 任意「小空间维 + 标量权重驱动向量 FMA」的算子，当 GetValue S-pipe 串行化是主瓶颈
- 仅可用 `#pragma unroll` 驱动流水；手工多标量预取禁用（编译器 bug）

**Strategy IDs**: X23
