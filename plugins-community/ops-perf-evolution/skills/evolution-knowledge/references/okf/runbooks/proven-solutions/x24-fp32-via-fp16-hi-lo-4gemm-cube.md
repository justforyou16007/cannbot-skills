---
schema_version: okf.v1
kind: proven_solution
title: "FP32-via-FP16 Hi/Lo 4-GEMM Cube Decomposition"
source: "WindowAttention evolution, round 1, parallel 2"
speedup: "0.47x -> 0.98x geomean (fp32 geomean 0.1x -> 0.956x; case48 20ms -> 0.59ms)"
operator_type: "flash_attention/matmul/conv"
hardware: "Ascend 910B2C (DAV_2201)"
confidence: single_op
status: active
op_families: ['flash_attention', 'matmul', 'conv']
bottlenecks: ['scalar_compute', 'compute_bound']
strategy_ids: ['X24', 'P128']
tags: ['fp32', 'fp16', 'cube', 'hi-lo', 'decomposition', 'precision']
---

### FP32-via-FP16 Hi/Lo 4-GEMM Cube (X24)

**Source**: WindowAttention evolution, round 1, parallel 2
**Speedup**: 0.47x → 0.98x geomean (fp32 geomean 0.1x → 0.956x; case48 20ms → 0.59ms)
**Operator type**: flash_attention / matmul / conv（fp32 GEMM 精度路径）
**Hardware**: Ascend 910B2C (DAV_2201)

**Technique description**:
Cube 不吃 fp32（910B2C fp32 cube MMA NaN）且精度要求 fp32 allclose（atol≈2e-5，bf16 3-GEMM 的 ~1e-4 不够）时，把每个 fp32 操作数拆为 fp16 hi/lo 双分量（fp16 11 尾数位，hi+lo 捕获 ~23bit），用 4 次 fp16 Mmad（Ah@Bh + Ah@Bl + Al@Bh + Al@Bl，含 lo·lo）在 fp32 L0C 累加，误差 ~9e-6。独立纯 AIV split 预 pass 产出 hi/lo（8-deep ring UB 规避 MTE3→V WAR），K1/K3 与 K2 注意力 S/O GEMM 均可套用；每 k-chunk `cmatrixInitVal=true`（AIV 分别累加 O）。小 shape 建议 M*C 阈值路由回向量 FMA（split 预 pass 固定开销主导）。

**Key code change**:
```cpp
// Before (fp32 向量 FMA，标量 GetValue 内层循环):
for (r) for (k) { float av = aC.GetValue(r*BK+k);
                  Muls(tmp, bC[k*BN], av, BN); Add(acc, acc, tmp, BN); }

// After (fp16 hi/lo 4-Mmad 上 Cube):
// AIV split: hi = Cast_fp16(x, CAST_RINT); lo = Cast_fp16(x - fp32(hi), CAST_RINT);
// AIC: Mmad(s, Ah, Bh, mp); Mmad(s, Ah, Bl, mp); Mmad(s, Al, Bh, mp); Mmad(s, Al, Bl, mp); // fp32 L0C
```

**Applicability conditions**: fp32 输入 GEMM，Cube 仅 fp16/bf16，容差 ≤ ~1e-5（fp32 allclose atol 2e-5）；fp16 动态范围覆盖值域；小 shape 回退向量 FMA。

**Strategy IDs**: X24（与 X17/P128 bf16 3-GEMM 互补：X17 容差 ~1e-4，X24 容差 ~9e-6 过 fp32 allclose）
