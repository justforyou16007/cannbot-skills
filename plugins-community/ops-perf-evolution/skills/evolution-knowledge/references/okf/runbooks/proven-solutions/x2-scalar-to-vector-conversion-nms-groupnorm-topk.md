---
schema_version: okf.v1
kind: proven_solution
title: "X2: Scalar-to-Vector Conversion (NMS + GroupNorm + TopK)"
source: "NMS (30_NMS) evolution, GroupNorm (11_GroupNorm_evo) P0, TopK (TopK_evo_0629) Round 1-3"
speedup: "11.25x (NMS), 1.84x geomean (GroupNorm fp32: 0.50x→, fp16: 1.93x, bf16: 4.31x), 4.08x (TopK, from 0.66x)"
operator_type: "elementwise, reduction, normalization, sort, topk, nms"
hardware: "Ascend 910B2C"
confidence: cross_op_verified
status: active
op_families: ['elementwise', 'normalization', 'reduction']
strategy_ids: X2 (cross-operator validation: works on NMS, GroupNorm, TopK; pattern abstracted to P85-P89 strategy cards)
---

### X2: Scalar-to-Vector Conversion (NMS + GroupNorm + TopK)

**Source**: NMS (30_NMS) evolution, GroupNorm (11_GroupNorm_evo) P0, TopK (TopK_evo_0629) Round 1-3
**Speedup**: 11.25x (NMS), 1.84x geomean (GroupNorm fp32: 0.50x→, fp16: 1.93x, bf16: 4.31x), 4.08x (TopK, from 0.66x)
**Operator type**: elementwise, reduction, normalization, sort, topk, nms
**Hardware**: Ascend 910B2C

**Technique description**:
TileLang-generated AscendC code frequently contains scalar for-loops that bypass the NPU Vector unit entirely. The systematic replacement of these patterns with AscendC Vector API calls — ReduceSum for accumulation loops, Adds/Muls chains for element-wise math, Sort32/MrgSort for sorting/selection, and zero-copy sub-tensor slicing for batch operations — is the single highest-impact optimization for generated kernels. A hybrid dispatch (vector path for large workloads, scalar path for small ones) prevents regression on edge cases.

**Key code change**:
```cpp
// Pattern A: Scalar reduction → ReduceSum (+2-5x)
// Before: for(i) sum += x[i]; sum_sq += x[i]*x[i];
// After:
AscendC::ReduceSum<float, true>(scalarOut, xTensor, reduceTmp, count);

// Pattern B: Scalar element-wise → Adds/Muls chain (+3-10x)
// Before: for(i) y[i] = (x[i] - mean) * inv_std * w + b;
// After:
AscendC::Adds<float>(y, x, -mean, n);
AscendC::Muls<float>(y, y, inv_std, n);
AscendC::Muls<float>(y, y, w_val, n);
AscendC::Adds<float>(y, y, b_val, n);

// Pattern C: Scalar sort → Sort32/MrgSort (+5-100x for large n)
// Before: O(n²) bubble sort or O(nk) selection sort
// After:
AscendC::Sort<float, true>(vals, vals, indices, tmpBuf, n / 64);
```

**Applicability conditions**:
- Kernel contains `for` loops with scalar arithmetic (detected by grep for `for.*++.*sum\s*+=` patterns)
- Reduction dimension ≥ 64 elements (smaller: scalar overhead negligible)
- Element-wise operations ≥ 3 in chain (fewer: vector launch overhead may dominate)
- Sort/selection n ≥ 32 (smaller: scalar selection sort is acceptable)

**Strategy IDs**: X2 (cross-operator validation: works on NMS, GroupNorm, TopK; pattern abstracted to P85-P89 strategy cards)
