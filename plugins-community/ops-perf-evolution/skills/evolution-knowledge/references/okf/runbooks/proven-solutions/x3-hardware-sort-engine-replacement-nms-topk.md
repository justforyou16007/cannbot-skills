---
schema_version: okf.v1
kind: proven_solution
title: "X3: Hardware Sort Engine Replacement (NMS + TopK)"
source: "NMS (30_NMS) sort optimization, TopK (TopK_evo_0629) Round 1"
speedup: "NMS case3: 2181x, TopK: 0.66x → 4.08x (hardware sort is key contributor)"
operator_type: "sort, topk, nms (any operator with sorting/selection)"
hardware: "Ascend 910B2C"
confidence: cross_op_verified
status: active
op_families: [sort, topk]
strategy_ids: X3 (captured in P87 strategy card)
---

### X3: Hardware Sort Engine Replacement (NMS + TopK)

**Source**: NMS (30_NMS) sort optimization, TopK (TopK_evo_0629) Round 1
**Speedup**: NMS case3: 2181x, TopK: 0.66x → 4.08x (hardware sort is key contributor)
**Operator type**: sort, topk, nms (any operator with sorting/selection)
**Hardware**: Ascend 910B2C

**Technique description**:
AscendC provides hardware-accelerated Sort/Sort32/MrgSort APIs executed on the Vector unit. These replace any scalar sorting or top-k selection algorithm (bubble sort, selection sort, heap-based top-k) with O(n log² n) bitonic sort hardware execution. Key patterns: (1) Negate → Sort ascending → Negate for descending sort; (2) Sort32(repeat=BATCH_ROWS) for batch-parallel small-row sorting; (3) MrgSort for tiled large-row scenarios; (4) Sort + Extract for top-k from sorted results.

**Key code change**:
```cpp
// NMS: Negate → Sort → Negate for descending
for (int32_t k = 0; k < M; k++) svScore(k) = -svScore(k);
AscendC::Sort<float, true>(svScore, svScore, orderU32, tmpF, SORT_SZ / 64);
for (int32_t k = 0; k < M; k++) svScore(k) = -svScore(k);

// TopK: Sort32(repeat=B) batch parallel for small rows
AscendC::Sort<float, true>(packedBuf, packedBuf, orderBuf, tmpBuf, 
                           curBatch * rowLen / 64);
```

**Applicability conditions**:
- Any kernel with sorting, top-k selection, or NMS-style threshold selection
- Element count ≥ 32 (below: scalar methods may be faster)
- Requires 2× data buffer + 1× index buffer in UB for Sort32

**Strategy IDs**: X3 (captured in P87 strategy card)
