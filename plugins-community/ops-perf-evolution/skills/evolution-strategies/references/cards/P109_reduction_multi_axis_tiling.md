---
source_key: evolution-strategies#card/P109_reduction_multi_axis_tiling
id: P109
title: Reduction 多轴 Tiling 系统化设计
doc_path: cards/P109_reduction_multi_axis_tiling.md
complexity: L1
op_families: [reduction, normalization, softmax]
triggers: [multi_axis_reduction, shape_collapse, ar_ara_routing]
references:
  - ops/ascendc-tiling-design/references/reduction/patterns.md
  - ops/ascendc-tiling-design/references/reduction/algorithms.md
  - ops/ascendc-tiling-design/references/reduction/multi-axis-transform.md
  - ops/ascendc-tiling-design/references/reduction/tiling-fields.md
bottlenecks: [tiling_mismatch, scalar_compute]
conflicts_with: []
synergizes_with: [P110, P68, A2]
has_preconditions: true
has_playbook: true
---

# P109: Reduction 多轴 Tiling 系统化设计

## 核心思想

将任意 N 维 shape + axes 先合轴为 A/R 交替序列，再按单轴归约（AR/ARA）逐维处理。避免在多维场景下直接手写嵌套归约循环，而是把问题分解为：

1. **合轴**：消除 size=1 维度，合并相邻同类型轴。
2. **模式选择**：A0=1 走 AR（尾轴连续），A0>1 走 ARA（非尾轴块）。
3. **多轴展开**：ARARA 等场景使用 Shape 三步变换，把每个 R 维单独归约。

## 代码骨架

```cpp
// === ARA 模式：非尾轴归约 ===
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);
uint32_t srcShape[] = {R, alignedCols};
ReduceMax<float, Pattern::Reduce::RA>(resultLocal, xLocal, tmpLocal, srcShape, true);

// === AR 模式：尾轴归约 ===
ReduceSum<float, true>(scalarLocal, xLocal, reduceTmp, rLength);
```

## 关键修改点

1. **合轴函数**：host/tiling 侧先把 `shape` + `axes` 转成 A/R 序列，再决定 kernel 循环结构。
2. **A0 判断**：若合轴后首维为 1，说明 R 是尾轴，用 Level 2 Reduce API；否则用 `Pattern::Reduce::RA`。
3. **多轴归约**：每处理完一个 R 维，输出 shape 降维，继续处理下一个 R。
4. **`rLengthAlign` 用于 buffer 分配和 offset，`rLength` 用于 Reduce API count 和 DataCopyPad blockLen**。

## 常见陷阱

- ❌ 不归约的维度被误合并到 R 中 → 导致结果 shape 错误。
- ❌ AR/ARA 选择只看 `axis` 是否尾轴，未考虑合轴后的 A0。
- ❌ `alignedCols` 未按 32B 对齐 → `Pattern::Reduce::RA` 行为异常。
- ❌ 多轴场景直接一次性归约所有轴 → UB 放不下，应逐级处理。

## 代码搜索关键词

```bash
grep -nE "ReduceMax|ReduceSum|Pattern::Reduce::RA|alignedCols|rLength|rLengthAlign" op_kernel/*.cpp op_host/*_tiling.cpp
```
