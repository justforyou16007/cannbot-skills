---
schema_version: okf.v1
kind: proven_solution
title: "X44: fp32 Cube mmad 全量重写替代向量模拟 matmul（SetHF32(false) 全精度）"
source: "LongformerAttention evolution, round 1, parallel 2 (x0, open_exploration)"
speedup: "geomean 0.0013x → 4.149x vs PyTorch（基线 368ms / ~1000 次串行 host launch + 向量标量模拟 matmul → 最优 81μs）；小 shape 2-3x、大 shape 7-9x；50/50 精度 PASS（MERE ~1e-6）"
operator_type: "attention（多 matmul 流水）"
hardware: "Ascend910B2C (DAV_2201)"
confidence: single_op
status: active
op_families: [attention, matmul]
strategy_ids: [X44]
bottlenecks: [scalar_compute, launch_overhead, compute_bound]
tags: [cube-mmad, vector-simulated-matmul, set-hf32-false, fp32-cube, full-precision, attention, multi-matmul, matmul-highlevel-api]
---

### X44: fp32 Cube mmad 全量重写替代向量模拟 matmul（SetHF32(false) 全精度）

**Source**: LongformerAttention evolution, round 1, parallel 2 (x0, open_exploration)

**Speedup**: geomean 0.0013x → **4.149x** vs PyTorch（基线 368ms → 最优 81μs），小
shape 2-3x、大 shape 7-9x；50/50 精度 PASS（MERE ~1e-6）。本轮其余变体对比：
w1（workflow 策略路径）0.843x、n2（P54+P1 仅发射消除）0.050x —— Cube 重写是跨越关键。

**Operator type**: attention（QK^T → softmax → PV → W_o 多 matmul 流水）

**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
基线内核把全部 matmul 用 AIV 向量标量模拟（16×16 tile 逐点 GetValue 累加）完全绕过
Cube，且 host 侧 ~1000+ 次串行 launch（每次 synchronize）。重写为 7 次单次发射 kernel：
lf_prep(AIV) → lf_gemm_nt_fp32(AIC QK^T) → lf_softmax(AIV) → lf_gemm_nn_fp32(AIC PV) →
lf_merge(AIV) → lf_gemm_nt_fp32(AIC W_o) → lf_bias(AIV)。三个 matmul 全部走 Matmul 高阶
API（`ASCENDC_CUBE_ONLY` + `REGIST_MATMUL_OBJ` + `IterateAll`）fp32 Cube；`SetHF32(false)`
关闭 HF32 10-bit 尾数截断获得全精度 fp32（MERE ~1e-6，满足 fp32 rtol=1.2e-4）。

**Key code change**:
```cpp
// Before: AIV 向量标量模拟 matmul —— 16×16 tile 逐点 Mul + GetValue 累加，绕过 Cube
// After:  Matmul 高阶 API + SetHF32(false) 全精度 fp32 Cube
Mmz pmat;
pmat.SetHF32(false);                          // 全精度 fp32（默认 HF32 仅 10-bit 尾数）
REGIST_MATMUL_OBJ(&pmat, 1, Tiling, CUBE_ONLY);
pmat.SetTensorA(A, aL0, N1, M0, ...);         // Q/K 或 P/V，fp32 直接走 Cube
pmat.SetTensorB(B, bL0, N1, M0, ...);
pmat.IterateAll();                            // AIC 执行，coreIdx 步进覆盖全部任务
```

**Applicability conditions**:
- 基线 matmul 为向量标量模拟（AIV GetValue 逐点累加）的算子 —— 改 Cube 是跨越式收益
- attention 族多 matmul 流水，fp32 精度路径可放弃 fp16 hi/lo 拆分（HF32 拆分 3 次 gemm
  被全精度单次 gemm 取代）
- 小/中 shape 为主、gemm 非瓶颈上限的算子（全精度 fp32 比 HF32 慢但 shape 小可接受）
- 配套工程约束：uint8 位图 mask 向量操作须 256-bit block 对齐（count 是 256 的倍数）；
  小 seqLen 统一 sR16≥64；fp32 向量 op count ≥8 且为 8 的倍数；exp 保留行 max 减法

**Strategy IDs**: X44（OKF 卡 disc_X44；与 P54 发射消除、X24 手动 mmad 思想同族但覆盖
「向量模拟→Cube 全量重写」这一数量级跨越场景）
