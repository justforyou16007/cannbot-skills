---
schema_version: okf.v1
kind: proven_solution
title: "X1: Causal S2 Block Skipping (FlashAttentionSimple)"
source: "FlashAttentionSimple evolution, arch_round8"
speedup: "1.003x → 1.83x (45.5% latency reduction at S=32K)"
operator_type: "attention"
hardware: "Ascend 910B3"
confidence: single_op
status: active
op_families: ['attention', 'flash_attention']
strategy_ids: X-series (novel, discovered during evolution)
---

### X1: Causal S2 Block Skipping (FlashAttentionSimple)

**Source**: FlashAttentionSimple evolution, arch_round8
**Speedup**: 1.003x → 1.83x (45.5% latency reduction at S=32K)
**Operator type**: attention
**Hardware**: Ascend 910B3

**Technique description**:
In causal attention, entire S2 blocks where all entries are masked (below the diagonal)
are skipped entirely — no cube matmul, no vector post-processing, no DMA.
The valid S2 range is computed per S1 block on the host side using the causal constraint:
`valid_s2_end = s1_end` (the last row of the current Q block determines the furthest valid K column).

**Key code change**:
```cpp
// Before: iterate over all S2 blocks
for (int s2 = 0; s2 < totalS2Blocks; s2++) { ProcessBlock(s1, s2); }

// After: skip invalid blocks
int validS2End = (s1BlockEnd) / blockSize + 1;
for (int s2 = 0; s2 < min(validS2End, totalS2Blocks); s2++) { ProcessBlock(s1, s2); }
```

**Applicability conditions**:
- Attention operator with causal (lower-triangular) mask
- Block-tiled computation (not element-wise attention)
- Sequence length > 512 (smaller sequences have few blocks to skip)

**Strategy IDs**: X-series (novel, discovered during evolution)

---

---
