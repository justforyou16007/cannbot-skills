---
source_key: evolution-strategies#card/P110_reduction_ub_accumulator_reuse
id: P110
title: Reduction UB 累加器复用
doc_path: cards/P110_reduction_ub_accumulator_reuse.md
complexity: L1
op_families: [reduction, normalization, softmax]
triggers: [reduction_fullload, accumulator_buffer, tmp_buf, ub_reuse]
references:
  - ops/ascendc-tiling-design/references/reduction/algorithms.md
  - ops/ascendc-tiling-design/references/reduction/patterns.md
  - ops/ascendc-tiling-design/references/reduction/tiling-fields.md
bottlenecks: [ub_pressure, scalar_compute]
conflicts_with: []
synergizes_with: [P109, P68, A2]
has_preconditions: true
has_playbook: true
---

# P110: Reduction UB 累加器复用

## 核心思想

在 AR/ARA FullLoad 模式下，整行/整块数据一次驻留 UB，归约后的中间结果直接复用 UB buffer 作为累加器，避免重复搬入；在 Split 模式下，分 chunk 归约后通过 Max/Add 跨 chunk 合并，同样需要合理分配 `tmpBuf` 与 accumulator。

## 代码骨架

```cpp
// === FullLoad：中间结果直接复用 UB ===
DataCopyPad(xLocal, gmInput, blockLen);          // 一次搬入整行
ReduceMax<float, true>(outLocal, xLocal, tmpLocal, rLength);
DataCopyPad(gmOutput, outLocal, ...);

// === Split：分 chunk 合并 ===
for (uint32_t chunk = 0; chunk < numChunks; ++chunk) {
    DataCopyPad(partLocal, gmInput + chunk * Ar, Ar);
    ReduceMax<float, true>(partLocal, partLocal, tmpLocal, Ar);
    // 跨 chunk 合并
    if (chunk == 0) { Copy(accLocal, partLocal); }
    else { Max(accLocal, accLocal, partLocal, ...); }
}
```

## 关键修改点

1. **Buffer 预算**：`tmpBufSize` 必须按 `rLengthAlign` 分配，Reduce API workspace 单独计算。
2. **FullLoad 条件**：UB 至少能放 1 整行 R（AR）或 `R × tileA0Len`（ARA）。
3. **Split 合并**：跨 chunk 使用与算子语义对应的二元 API（Sum 用 Add，Max 用 Max）。
4. **精度敏感**：大 R sum 可叠加 Welford Online 或二分累加算法。

## 常见陷阱

- ❌ FullLoad 误判为 Split → 多轮 DataCopy 浪费带宽。
- ❌ `tmpLocal` 与 `outLocal` 复用同一块 UB → Reduce API 覆盖输出。
- ❌ Split 场景 chunk 边界未按 32B 对齐 → DataCopyPad 参数错误。
- ❌ 未区分 `rLength` 与 `rLengthAlign` → buffer 越界或结果不对。

## 代码搜索关键词

```bash
grep -nE "tmpBuf|tmpLocal|accumulator|accLocal|rLengthAlign|FullLoad|ColSplit|RowSplit" op_kernel/*.cpp op_host/*_tiling.cpp
```
