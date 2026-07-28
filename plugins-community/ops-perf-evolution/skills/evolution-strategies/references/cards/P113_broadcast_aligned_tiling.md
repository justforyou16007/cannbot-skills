---
source_key: evolution-strategies#card/P113_broadcast_aligned_tiling
id: P113
title: Broadcast 对齐 Tiling 设计
doc_path: cards/P113_broadcast_aligned_tiling.md
complexity: L1
op_families: [broadcast, elementwise]
triggers: [broadcast, nddma, ub_brc, dimension_collapse]
references:
  - ops/ascendc-tiling-design/references/broadcast/patterns.md
  - ops/ascendc-tiling-design/references/broadcast/nddma-broadcast.md
  - ops/ascendc-tiling-design/references/broadcast/ub-broadcast.md
  - ops/ascendc-tiling-design/references/broadcast/dynamic-ub-broadcast.md
  - ops/ascendc-tiling-design/references/broadcast/onedim.md
bottlenecks: [mte2_stall, ub_pressure, tail_block]
conflicts_with: []
synergizes_with: [P112, P12, P7, P1]
has_preconditions: true
has_playbook: true
---

# P113: Broadcast 对齐 Tiling 设计

## 核心思想

Broadcast 算子先通过 DimensionCollapse 把多轴问题降维，再判断是 OneDim（合轴后单维）还是多维广播。多维场景根据硬件版本与数据特征选择 UB Broadcast（BRC）或 NDDMA 硬件广播，并按 256B 对齐切分 UB，确保多核负载均衡。

## 代码骨架

```cpp
// DimensionCollapse + stride 计算
for (int i = shapeLen - 1; i >= 0; --i) {
    flag[i] = computeBroadcastFlag(dims, i);
    if (i < shapeLen - 1 && flag[i] == flag[i+1]) {
        collapsedDims[i] *= collapsedDims[i+1];
        // 合并轴
    }
    strides[i] = (dims[j][i] == 1 && outDim > 1) ? 0 : runningStride;
    runningStride *= collapsedDims[i];
}

// UB 切分：找到 ubSplitAxis
for (int i = shapeLen - 1; i >= 0; --i) {
    curProduct *= dims[i];
    if (curProduct > maxElemNum) { ubSplitAxis = i; break; }
}
```

## 关键修改点

1. **补维 + 合轴**：输入维度不足时左侧补 1；相邻同 flag 轴合并。
2. **广播轴 stride 置 0**，非广播轴按连续 stride 计算。
3. **广播方式选择**：
   - NLast + 尾轴大 → UB BRC
   - INT8/FP16/BF16 + 尾轴 32B 对齐 → UB BRC
   - 其他 → NDDMA
4. **NDDMA >5 轴**：外层用 Kernel for-loop，最内 5 轴交给 `DataCopy<T,5,config>`。
5. **对齐**：OneDim 128B，多维 256B。

## 常见陷阱

- ❌ 未补维导致 flag 位图错位。
- ❌ 广播轴 stride 未置 0 → DMA 读到错误数据。
- ❌ NLast 场景误选 NDDMA → dcache 被反复冲刷。
- ❌ NDDMA 维度超过 5 未拆分 → 编译失败。

## 代码搜索关键词

```bash
grep -nE "DimensionCollapse|broadcast|flag|stride|ubSplitAxis|NDDMA|BRC|DataCopy<.*5" op_kernel/*.cpp op_host/*_tiling.cpp
```
