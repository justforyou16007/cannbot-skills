---
id: P96
bottlenecks: [scalar_compute, compute_bound]
op_families: [special, index_scatter]
complexity: L2
conflicts_with: []
synergizes_with: [P1, P7, P67]
requires: []
has_preconditions: true
has_playbook: true
---

# P96: 硬件排序引擎替代标量排序/选择 (Hardware Sort Engine Replacement)

## 核心思想
检测 kernel 中的标量排序（bubble sort、selection sort）或 top-k 选择算法，用 AscendC 硬件 Sort32/MrgSort/Extract API 替代。910B Vector 单元提供硬件加速的 bitonic sort，算法复杂度从 O(M²) 或 O(n×k) 降至 O(n log² n)。降序场景（NMS）用 Negate → Sort ascending → Negate 模式；小行场景（TopK 3D/4D，rowLength ≤ 32）多行打包为一次 Sort32(repeat=B) 共享一次 barrier，加速 2-6x；大行场景用 Tiled Sort + MrgSort 跨 tile 合并。

## 代码骨架

```cpp
// === 改造前（基线）：标量排序/选择 ===
// 反模式 A：冒泡排序 O(M²)
for (int32_t i = 0; i < M; ++i)
    for (int32_t j = 0; j < M - 1; ++j)
        if (scores[j] < scores[j+1]) { swap(scores[j], scores[j+1]); }

// 反模式 B：Selection-sort topK (O(n×k))
for (int32_t i = 0; i < K; ++i) {
    float maxVal = -INFINITY; int maxIdx = -1;
    for (int32_t j = 0; j < n; ++j)
        if (vals[j] > maxVal) { maxVal = vals[j]; maxIdx = j; }
    result[i] = maxVal; vals[maxIdx] = -INFINITY;
}

// 反模式 C：逐元素标量比较交换 (O(M²))
for (int32_t i = 0; i < M; ++i)
    for (int32_t j = i + 1; j < M; ++j)
        if (compare(scores[i], scores[j])) swap_all_fields(i, j);

// === 改造后：硬件 Sort32/MrgSort ===
// 模式 A：降序排序（Negate → Sort → Negate，用于 NMS）
for (int32_t k = 0; k < M; k++) svScore(k) = -svScore(k);    // Negate
for (int32_t k = M; k < SORT_SZ; k++) svScore(k) = 1e38f;     // Pad 到对齐大小
AscendC::Sort<float, true>(svScore, svScore, orderU32, tmpF, SORT_SZ / 64);
for (int32_t k = 0; k < M; k++) svScore(k) = -svScore(k);    // Restore

// 模式 B：批量并行小行排序（Sort32(repeat=B)，用于 TopK 3D/4D 小行）
constexpr int32_t BATCH_ROWS = 32;
for (int32_t r = 0; r < totalRows; r += BATCH_ROWS) {
    int32_t curBatch = min(BATCH_ROWS, totalRows - r);
    // 将 curBatch 行数据打包到连续 UB 区域
    AscendC::Sort<T, true>(packedBuf, packedBuf, orderBuf, tmpBuf,
                           curBatch * rowLen / 64);
}

// 模式 C：大行 Tiled Sort + MrgSort（用于 TopK 长行）
// 对每个 tile 内部 Sort → Extract topK → 跨 tile MrgSort 合并
AscendC::Sort<T, true>(tileVals, tileVals, tileIdx, tmpF, tileCap / 64);
// Extract topK from sorted result
// MrgSort 合并跨 tile 的 topK 结果
```

## 关键修改点

1. **Sort32 参数**：`Sort<T, true>(dst, src, index, tmp, repeat)` 中 `repeat = totalElements / 64`（64 是硬件数据块大小，**非行数**）；`true` = 同步输出排序后索引
2. **Buffer 需求**：2× 数据 buffer（源 + 目标） + 1× 索引 buffer（uint32_t，同元素数） + 1× 临时 buffer（float，同元素数）
3. **Negate 降序技巧**：`AscendC::Sort` 仅支持升序；降序 = Negate → Sort → Negate，padding 元素用 +∞ (1e38f) 填充防止干扰
4. **Sort-index hoist**：`CreateVecIndex` 每 kernel launch 只调一次，消除 tile × row 次冗余 Sort dispatch
5. **预期收益**：n ≥ 4096 时 100-2000x（O(n²) vs O(n log² n)）；n = 32-256 时 2-10x（硬件加速 + 批量并行）

## 常见陷阱

⚠️ Sort32 要求 `repeatTimes = totalElements / 64` 整除；不整除需 pad 到 64 的倍数
⚠️ Padding 元素用 +∞（升序沉底）/ -∞ 填充，防止干扰有效元素的排序位置
⚠️ Sort-index 必须用 `uint32_t`；排序后索引指向排序前位置（非排序后位置）
⚠️ MrgSort 只合并两个已排序序列；合并多于 2 个序列需多次 MrgSort
⚠️ Sort32(repeat=B) 的 repeat 参数含义是**数据块数**（每块 64 元素），非行数

## 代码搜索关键词

```bash
grep -nE "Sort|Sort32|MrgSort|Extract|CreateVecIndex|orderU32|bitonic" op_kernel/*.cpp
```

## 来源

- NMS 进化 (30_NMS): Sort32 替代 O(M²) bubble sort，关键贡献因子，geomean 11.25x
- TopK 进化 (TopK_evo_0629): Sort32 + MrgSort 替代 O(n×k) selection sort，geomean 4.08x (from 0.66x)
