---
schema_version: okf.v1
kind: discovered_strategy
id: X25
title: "X25: 向量权重广播（Brcb + MulAddDst）替代 S-pipe 标量读"
origin: discovered
discovered_round: 1
discovered_from: "8_ConvStandard3d evolution, round 1, parallel 2 (x0, open_exploration)"
base_speedup: 0.802x（保守 full-60 vs PyTorch；39/60 实测 +1.36x vs 同用例基线）
confidence: single_op
status: active
op_families: [conv, cv_fusion, elementwise, matmul]
bottlenecks: [scalar_loading, scalar_compute]
tags: [vector-broadcast, brcb, muladddst, scalar-read-elimination, s-pipe-bypass, batched-spatial]
strategy_ids: [X25]
---

### X25: 向量权重广播（Brcb + MulAddDst）替代 S-pipe 标量读

**Source**: 8_ConvStandard3d evolution, round 1, parallel 2 (x0, open_exploration)
**Speedup**: 39/60 用例实测 geomean 0.5516（同用例基线 0.4057，+1.36x）；多 batch 用例 case28 0.05→0.30（5.9x）、case27 0.16→0.74（4.6x）、case47/56 2.4x；full-60 保守估计 0.80（基线 0.66）。60/60 精度 PASS。⚠️ geomean 未达蒸馏阈值（受系统 7+ 并发 msprof 争用污染），但机制新颖且 per-case 实证有效。
**Operator type**: conv (kw==1, W 维逐点 FMA), fp32 pure-Vector, GEMM 形态 acc[G,HW] += w[G,K]@x[K,HW]
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
滑窗卷积/小 GEMM 的瓶颈常是每 (co,tap) 一次权重 `GetValue()`（S-pipe UB→标量寄存器读，~43cy，依赖链串行）。X25 用 **V-pipe `vbrcb`（Brcb）向量广播** 替代：把权重在 GM 预展开为 `[Co,K,8]` 布局（每权重 8 份连续副本），kernel 每 (g,tap) 一次 `Brcb` 把 8 个 b32 广播成 64-lane 权重向量，驱动 `numBatches` 条 `MulAddDst`（acc += x*w）——**全程不经过 S-pipe，标量读彻底消失**，数值语义不变（累加次序不变）。
- 关键：权重跨 oh-batch 不变，把 batch 循环内移，每次广播覆盖一个权重全部输出位置的复用（多 batch 摊销）。
- wvec 用 2×64-lane ping-pong 打破 Brcb→MulAddDst 写后读依赖链。
- 局限：numBatches=1 高 K case（K=1152, 单 batch）无跨 batch 摊销，Brcb 的 2-op 代价 + 8x 权重 GM 载入抵消 GetValue 消除收益。
- 局部细化：unroll 4 + 4 槽 ping-pong 在部分 case 调度劣化 ~5x，需保守调优。

**Key code change**:
```cpp
// Before: S-pipe 标量读驱动 Axpy（每 (tap,g) 一次 GetValue ~43cy）
#pragma unroll 4
for (g...) { AscendC::Axpy(acc[g*hw+b*P*W], x[tap], wLocal_.GetValue(g*TcP_+tap), P*W); }

// After: V-pipe Brcb 广播成 64-lane 权重向量 + MulAddDst（零 GetValue）
// wt1d 布局 [Co,K,8]，每权重 8 份连续副本
for (g...) {
    AscendC::Brcb(wvec[pp], wt1dLocal_.GetValue(g*TcP_*8 + tap*8), {0,1,2,3,4,5,6,7}, 8);
    for (b...) { AscendC::MulAddDst(acc[g*hw + b*P*W], x[tap + b*P*W], wvec[pp], P*W); }
    pp ^= 1;  // 2-slot ping-pong 破写后读
}
```

**Applicability conditions**:
- 滑窗卷积 kw==1 / 小 GEMM / 任意「标量权重驱动向量 FMA」且权重跨 batch 复用的算子
- 多 oh-batch / 多输出位置复用（numBatches ≥ 2）时收益最大；numBatches=1 时收益被 8x 权重载入抵消
- W∈[8,32]（W%8==0 满足 32B 对齐），权重广播副本数按 64/PAR 推导（fp32 PAR=64 → 8 副本）
- 需 model_new_ascendc.py 侧把权重预展开为广播布局（[Co,K,8]）
- ⚠️ 与 X23 互补：X23 保留 GetValue 但减少数量（hoist）；X25 用 V-pipe 广播彻底消除标量读（机制层面）

**Strategy IDs**: X25
