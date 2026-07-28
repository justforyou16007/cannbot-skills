---
source_key: evolution-strategies#card/P106_matmul_batch_merge_to_m
id: P106
title: Matmul Batch Merge to M
doc_path: cards/P106_matmul_batch_merge_to_m.md
bottlenecks: [parallelism_underutil, tiling_imbalance]
op_families: [matmul]
complexity: L1
triggers:
  - BatchMatMul with batchB == 1 and !transA
  - batchA > 1 and M is small leaving AIC cores idle
  - IterateBatch scheduling overhead dominates over per-batch compute
conflicts_with: []
synergizes_with: [P104, P16]
requires: []
has_preconditions: true
has_playbook: true
references:
  - ops/ascendc-tiling-design/references/matmul/ascendc-api-matmul-tiling.md
---

# P106: Matmul Batch 维度合并到 M（Batch Merge to M）

## 核心思想

在 BatchMatMul 中，当 B 矩阵无 batch（`batchB == 1`）且 A 未转置（`!transA`）时，可以把 A 的 batch 维度折叠进 M 维度：`M' = batchA * M`、`batchA' = 1`。这样原先把每个 batch 当作独立调度单元的问题，转化为更大的 M 维度问题，更容易触发多核切分扩展（P104），减少 `IterateBatch` 的跨 batch 调度开销。

## 代码骨架

```cpp
// === 改造前：batchA 与 M 分开处理 ===
uint32_t batchA = aShape.size() > 2 ? aShape[aShape.size() - 3] : 1;
uint32_t batchB = bShape.size() > 2 ? bShape[bShape.size() - 3] : 1;
uint32_t M      = aShape[aShape.size() - 2];

// === 改造后：在合法条件下把 batchA 合并进 M ===
if (batchB == 1 && !transA) {
    M = batchA * M;
    batchA = 1;
    // 后续 tiling 以新的 M 为基准
}
```

## 关键修改点

1. **前提严格判断**：仅当 `batchB == 1 && !transA` 时合并；`transA` 为 true 时 A 的 batch 与 M 在物理上不连续，不能折叠。
2. **Broadcast 表更新**：合并后 C 的 batch 由 `batchC` 重新按广播规则计算，避免维度错位。
3. **Kernel offset 不变**：因为 batch 已合并进 M，Kernel 端仍按合并后的 `(M', N, K)` 计算 `blockIdx → (mIdx, nIdx)`，无需额外 batch 索引。
4. **Bias 处理**：若带 bias 且 bias 原本按 batch 排布，需确认合并后 bias 广播语义仍成立。

## 常见陷阱

⚠️ `transA == true` 时仍合并，导致 A 的 stride 错误，结果错位。
⚠️ 合并后未更新 `batchC`，Host/KERNEL 参数不一致。
⚠️ 合并前 `batchA * M` 溢出 `uint32_t`，应使用 `int64_t` 中间计算。
⚠️ Bias 或 Scale 带有 batch 维度时，合并后广播失败。

## 代码搜索关键词

```bash
grep -nE "batchB|batchA|transA|MultiBatchInfo|IterateBatch|batchC" \
    op_host/*_tiling.cpp op_kernel/*.cpp
```

## 来源

- `ops/ascendc-tiling-design/references/matmul/ascendc-api-matmul-tiling.md` §Batch 合并到 M
