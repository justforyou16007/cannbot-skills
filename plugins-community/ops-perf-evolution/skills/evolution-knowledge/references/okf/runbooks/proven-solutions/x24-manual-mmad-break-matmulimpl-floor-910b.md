---
schema_version: okf.v1
kind: proven_solution
title: "X24: 手动 mmad 打破 MatmulImpl 固定开销地板（910B 微小 matmul）"
source: "5_MatmulTransB evolution（手动 mmad 攻坚轮）, round 2, parallel 1 (n1) + parallel 2 (x0_x1)"
speedup: "设备侧 12.09μs → 6.88μs（1.757x vs 基线 / 0.616x vs PyTorch）；小 shape fp16[64,64]=3.38μs、fp32[1,1]~6.1μs；50/50 精度 PASS"
operator_type: "matmul（微小 shape，固定开销主导）"
hardware: "Ascend910B2C (DAV_2201)"
confidence: single_op
status: active
op_families: [matmul]
strategy_ids: [P87, X24]
bottlenecks: [scalar_compute, no_overlap, compute_bound]
tags: [manual-mmad, transpose-workaround, aic-only, fixpipe-noquant, fractal-loading]
device_floor_us: "4.3-5.6"
---

### X24: 手动 mmad 打破 MatmulImpl 固定开销地板（910B 微小 matmul）

**Source**: 5_MatmulTransB evolution（手动 mmad 攻坚轮），round 2, parallel 1 (n1, 完整手动) + parallel 2 (x0_x1, 最优)
**Speedup**: 设备侧 12.09μs → 6.88μs（1.757x vs 基线 / 0.616x vs PyTorch，50/50 精度 PASS）；小 shape fp16[64,64]=**3.38μs**、fp32[1,1]~6.1μs（对比 MatmulImpl 的 ~12-13μs）；14/50 case 快于 PyTorch（最高 1.31x）
**Operator type**: matmul，C[M,N]=A[M,K]@B[N,K]^T（B 转置存储），M∈[1,512]、N∈[1,1024]、K∈[1,1024]，全部微小/小 shape
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
微小 shape matmul 的瓶颈是 **MatmulImpl 高阶 API 的库固定开销**（Init/IterateAll/End 标量地板 ~6-7μs，cube_util 仅 3.95%、aic_scalar_ratio 0.661，96% 算力空转）。本方案用手动 mmad kernel 完整绕过 MatmulImpl，四个关键突破：

1. **host B 预转置绕过 910B 转置限制**：B 按 [N,K] 存储（transB），设备转置原语（LoadData3D enTranspose / LoadData2D ifTranspose / AIV vtranspose>16x16）在 910B 上坏/挂起。host 侧先把 B 转成 Bt[K,N] 写入 workspace，kernel 读 Bt 走**无转置**的 L0B 加载，完全绕开硬件转置。
2. **TQue<TPosition::A1/B1> + CopyIn 正确分形加载**（P87 两次失败的死因修复）：不能用 LoadData3D（只对 16x16x16 identity 正确）；必须用 `TQue<A1/B1>` + `CopyIn` 做 GM→L1 ND→Nz（L0A）/ ND→Zn（L0B）分形转换，对任意 M/N/K（含非 16 对齐、尾块）正确。
3. **KERNEL_TYPE_AIC_ONLY 压设备地板**（X24）：去掉每 AIC 调度的 2 个空转 AIV，设备固定地板从 ~7-8μs → ~4.3-4.5μs。
4. **紧凑 tiling 加载 + fp32 Fixpipe NoQuant**：manual 路径 GM 标量读取从 ~65 int32 削减到 4-12 int32；并验证 `FixpipeParamsV220 + QuantMode_t::NoQuant` 可在 910B 输出 fp32（**推翻「Fixpipe 不支持 float→float」的旧结论**），fp32 小 shape 从 MatmulImpl ~13μs 降到手动 ~6μs。

单 tile 整块装入 L0（L0A/L0B/L0C 各 64-128KB），一条 Mmad + Fixpipe；fp16/bf16 用 TBuf<c0=16> 轻量 pipe（无 Alloc/Free），fp32 用 TQue<c0=8> + NoQuant 输出。

**Key code change**:
```cpp
// host 侧（matmul_trans_b.cpp）：B 预转置 + 紧凑 tiling
// B[N,K] → Bt[K,N] 写入 workspace（每 shape 一次，绕开设备转置）
transpose(B_workspace, bGm, N, K);   // 伪代码：host 或首调用一次

// device 侧（matmul_trans_b_manual.h）：手动 mmad，无 MatmulImpl
// 1) 紧凑头：manual 路径只读前 4-12 个 int32（替代全量 ~65 个逐标量加载）
// 2) L0A/L0B 分形加载：TQue<A1/B1> + CopyIn（GM ND → L1 Nz/Zn → L0A/L0B）
// 3) 单 tile 整块：L0A[blockM,blockK] × L0B[blockK,blockN] → L0C，一条 Mmad
// 4) fp32：FixpipeParamsV220 + QuantMode_t::NoQuant 输出 fp32
// 5) 同步：set_flag/wait_flag 轻量 HardEvent 链（MTE2→MTE1, MTE1→M, M→FIX）
// 6) 任务类型：KERNEL_TYPE_AIC_ONLY（无 AIV）
```

**Applicability conditions**:
- 微小 shape matmul（单 tile 可装入 L0：blockM×blockK≤128、blockN×blockK≤128、blockM×blockN≤128、L1≤384KB），**固定开销主导**
- transB 类算子（B 转置存储）：必须配 host B 预转置（910B 设备转置不可用）
- 精度要求严格时：大 shape（超 L0 容量）保留 MatmulImpl 兜底；多核 K-slice 手动在无 double-buffer 时**反而不如 MatmulImpl**，勿强上
- 上限认知：910B 每 kernel launch 设备地板 ~4.3-5.6μs，PyTorch 参考 geomean ~4.2μs——目标加速比低于硬件地板（如 2.0x 需 ≤2.1μs）时物理不可达，应先确认目标可行性

**Strategy IDs**: P87, X24
