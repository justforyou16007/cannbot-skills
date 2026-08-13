---
schema_version: okf.v1
kind: proven_solution
title: "Zero-Host-Padding AIV Small-Matmul"
source: "BatchMatmul evolution, round 1, parallel 3"
speedup: "0.375x -> 0.61x geomean (small-shape 0.15x -> 0.98x; [1,2,4] up to 1.56x)"
operator_type: "matmul"
hardware: "Ascend 910B2C (npu13)"
confidence: single_op
status: active
op_families: ['matmul']
bottlenecks: ['undersize_transfer', 'fake_mte2_bound', 'scalar_loading']
strategy_ids: ['X23']
tags: ['aiv', 'vector', 'host-padding', 'small-shape', 'zero-pad']
---

### Zero-Host-Padding AIV Small-Matmul (X23)

**Source**: BatchMatmul evolution, round 1, parallel 3
**Speedup**: 0.375x → 0.61x geomean (small-shape 0.15x → 0.98x; [1,2,4] up to 1.56x)
**Operator type**: matmul (bmm / small non-aligned shapes)
**Hardware**: Ascend 910B2C (npu13)

**Technique description**:
Host 不再把 M/N/K 零填充到 Cube 对齐粒度（constant_pad_nd 的 pad kernel 占小 shape 官方时长的 83%），而是直接下发原始 shape；AIV-only kernel 核内精确计算 C[m][n]=Σ_k A[m][k]·B[k][n]，fp32 累加 + 32B 对齐行展开（DataCopyPad）+ M 行多核切分。准入 M*K≤256，超阈值回退 Cube MatmulImpl。

**Key code change**:
```cpp
// Before (host): 16 对齐零填充 + Cube MatmulImpl
aUse = at::constant_pad_nd(aUse, {0, kp - k, 0, mp - m});  // pad kernel 17-30us
tilingApi.SetShape(mp, np, kp);

// After (host): 不 pad 直接下发，按 shape 分派
if (m * k <= 256) {  // AIV 精确路径
    EXEC_KERNEL_CMD(batch_matmul_vector_fp32, batch * mChunks, a, b, c, tilingNpu);
} else {             // Cube MatmulImpl 路径
    tilingApi.SetShape(mp, np, kp);
    EXEC_KERNEL_CMD(batch_matmul_fp32, blockDim, aUse, bUse, c, w, tilingNpu);
}
```

**Applicability conditions**:
- matmul 系算子非 16 对齐小 shape（M*K≤256）
- 官方评估将 host pad kernel 计入总时长
- fp16/bf16/fp32（fp32 累加，CAST_ROUND/CAST_RINT 输出）

**Strategy IDs**: X23 (与 X13 双引擎分流、n1 SetFixSplit 多核 Cube 正交可组合)
