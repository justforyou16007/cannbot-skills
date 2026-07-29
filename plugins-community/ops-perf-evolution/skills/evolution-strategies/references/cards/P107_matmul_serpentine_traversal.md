---
source_key: evolution-strategies#card/P107_matmul_serpentine_traversal
id: P107
title: Matmul Serpentine Traversal
doc_path: cards/P107_matmul_serpentine_traversal.md
bottlenecks: [memory_bound, mte2_bound]
op_families: [matmul]
complexity: L1
triggers:
  - L2 tiling enabled with nTileCntL2 > 1
  - B matrix rows can be reused between adjacent L2 tiles in the N direction
  - M/N large enough for multi-tile L2 traversal
conflicts_with: []
synergizes_with: [P105, P52, P104]
requires: [P105]
has_preconditions: true
has_playbook: true
references:
  - ops/ascendc-tiling-design/references/matmul/ascendc-api-matmul-tiling.md
---

# P107: Matmul Serpentine 反向遍历（Serpentine Traversal）

## 核心思想

在 L2 级 Tiling 中，若按行优先正向遍历 N 方向 tile，每切换一行 M 就要重新从 L2 加载远端 B 行。P107 采用 **serpentine 遍历**：偶数行 M 正向走 N，奇数行 M 反向走 N，使相邻两行在 L2 tile 边界处共享同一组 B 行，提升 L2 命中率并减少 MTE2 读 B。

## 代码骨架

```cpp
// === 改造前：正向双层循环 ===
for (uint64_t mTile = 0; mTile < mTileCntL2; ++mTile) {
    for (uint64_t nTile = 0; nTile < nTileCntL2; ++nTile) {
        ProcessL2Tile(mTile, nTile);
    }
}

// === 改造后：Serpentine 反向遍历 ===
bool reverse = true;
for (uint64_t mTile = 0; mTile < mTileCntL2; ++mTile) {
    reverse = !reverse;
    for (uint64_t t = 0; t < nTileCntL2; ++t) {
        uint64_t nTile = reverse ? (nTileCntL2 - t - 1) : t;
        ProcessL2Tile(mTile, nTile);
    }
}
```

## 关键修改点

1. **作用域**：仅当 `nTileCntL2 > 1` 时生效；只有 1 列 tile 时无需反向。
2. **与 calOrder 配合**：`calOrder` 决定 L2 tile 内 base 块遍历方向；serpentine 决定 L2 tile 之间的遍历方向，二者正交。
3. **同步语义不变**：每个 L2 tile 内部仍按原有 Iterate/SetSingleShape 流程执行，只是下发顺序改变。
4. **负载均衡**：serpentine 不改变 `totalTileCnt`，也不改变 Block 级核间分配。

## 常见陷阱

⚠️ 反向遍历导致 kernel 端依赖 tile 下发顺序（如 Atomic Add 顺序敏感）时结果非确定性。
⚠️ `nTileCntL2 == 1` 时仍加入 serpentine，引入无意义分支。
⚠️ 与 `COL_FIRST` 组合时，优先复用 B 列而非 B 行，需重新评估收益。
⚠️ 若 `ProcessL2Tile` 内部有 workspace 状态跨 tile 依赖，serpentine 会打破顺序。

## 代码搜索关键词

```bash
grep -nE "serpentine|reverse\s*=|mTileCntL2|nTileCntL2|ProcessL2Tile|calOrder" \
    op_host/*_tiling.cpp op_kernel/*.cpp
```

## 来源

- `ops/ascendc-tiling-design/references/matmul/ascendc-api-matmul-tiling.md` §Serpentine 反向遍历
