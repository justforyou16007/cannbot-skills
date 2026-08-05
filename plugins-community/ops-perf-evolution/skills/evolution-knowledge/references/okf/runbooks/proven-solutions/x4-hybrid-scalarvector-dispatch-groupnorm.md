---
schema_version: okf.v1
kind: proven_solution
title: "X4: Hybrid Scalar/Vector Dispatch (GroupNorm)"
source: "GroupNorm (11_GroupNorm_evo) P0 optimization"
speedup: "eliminated regression cases; fp32 geomean improved from baseline 0.22x to 0.50x"
operator_type: "reduction, normalization (any operator with variable workload size)"
hardware: "Ascend 910B2C"
confidence: cross_op_verified
status: active
op_families: ['normalization', 'reduction']
strategy_ids: X4 (captured in P89 strategy card)
---

### X4: Hybrid Scalar/Vector Dispatch (GroupNorm)

**Source**: GroupNorm (11_GroupNorm_evo) P0 optimization
**Speedup**: eliminated regression cases; fp32 geomean improved from baseline 0.22x to 0.50x
**Operator type**: reduction, normalization (any operator with variable workload size)
**Hardware**: Ascend 910B2C

**Technique description**:
Vector API overhead (EnQue/DeQue pipeline synchronization) can dominate on small workloads. A hybrid architecture with an automatic dispatch threshold (e.g., spatial_size >= 128 → vector path, else → scalar path) ensures that small workloads don't suffer from vectorization overhead while large workloads get the full vector benefit. The scalar path uses AllocTensor → DataCopyPad → raw pointer → FreeTensor (no EnQue/DeQue), while the vector path uses DeQue → ReduceSum/Adds/Muls → EnQue.

**Key code change**:
```cpp
constexpr int32_t VECTOR_MIN_SPATIAL = 128;
if (count_spatial >= VECTOR_MIN_SPATIAL) {
    ProcessBatchedVectorized(...);  // EnQue/DeQue + Vector API
} else {
    ProcessBatchedScalar(...);      // AllocTensor + pointer + FreeTensor
}
```

**Applicability conditions**:
- Operator has variable per-tile workload size (e.g., different spatial dimensions)
- Vector path uses EnQue/DeQue (synchronization overhead ~tens of cycles)
- Requires profiling to tune the threshold for specific hardware

**Strategy IDs**: X4 (captured in P89 strategy card)

---
