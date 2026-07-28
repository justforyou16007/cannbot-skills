---
source_key: evolution-strategies#card/P111_sort_two_level_merge_tiling
id: P111
title: Sort 两级归并 Tiling 系统化设计
doc_path: cards/P111_sort_two_level_merge_tiling.md
complexity: L1
op_families: [sort, topk]
triggers: [large_sort, two_level_merge, topk, mrgsort]
references:
  - ops/ascendc-tiling-design/references/sort/patterns.md
  - ops/ascendc-tiling-design/references/sort/alg-two-level-mrgsort.md
bottlenecks: [ub_overflow, sync_overhead, scalar_compute]
conflicts_with: []
synergizes_with: [P96, P1, P7]
has_preconditions: true
has_playbook: true
---

# P111: Sort 两级归并 Tiling 系统化设计

## 核心思想

当 `N > tileSize × coreNum` 时，Sort / TopK 必须走 Pattern C：先在各核内部把多个有序 tile 归并为 1 路，再跨核归并。通过 workspace 双缓存、每轮归并后的 `SyncAll`、以及 Top-K 截断，将跨核同步次数降到 `ceil(log_M(coreNum)) + 1`。

## 代码骨架

```cpp
__aicore__ inline void Process()
{
    // Phase 1: 各核并行排序 tile
    for (uint32_t t = 0; t < myTiles; ++t) {
        SortInSingleCore(curTileSize, globalOffset);
    }
    // Phase 2/3
    pipe_.Reset();
    InitMergeBuffers();
    Phase2IntraCoreMerge();
    SyncAll();
    Phase3InterCoreMerge();
    // Phase 4
    pipe_.Reset();
    InitOutputBuffers();
    Phase4FinalMergeAndOutput();
}
```

## 关键修改点

1. **tileSize 取 4096**：对齐到 32，且通常实测最优。
2. **四阶段划分**：Phase 1 排序；Phase 2 核内归并；Phase 3 跨核归并；Phase 4 Core 0 输出。
3. **workspace 双缓存**：`workSpaceFlag` 每轮翻转，输入输出分离。
4. **截断逻辑**：当 `currentElements × M ≥ K` 时，后续只读取前 K 个元素。
5. **SyncAll 时机**：Phase 2 结束一次；Phase 3 每轮一次；Phase 4 结束一次。

## 常见陷阱

- ❌ `truncationFlag` 在归并前设置 → 首次触发时数据不完整。
- ❌ Phase 3 每 group 后 SyncAll → 过度同步；应在每轮结束后同步。
- ❌ `UpdateSortInfo()` 未调用 → 死循环。
- ❌ Workspace 按总元素数而非核数分配 → 尾核越界。

## 代码搜索关键词

```bash
grep -nE "Sort<|MrgSort|SyncAll|workSpaceFlag|truncationFlag|onceMaxElements|tileSize" op_kernel/*.cpp op_host/*_tiling.cpp
```
