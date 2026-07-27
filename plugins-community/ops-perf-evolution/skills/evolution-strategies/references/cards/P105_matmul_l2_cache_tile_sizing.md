---
source_key: evolution-strategies#card/P105_matmul_l2_cache_tile_sizing
id: P105
title: Matmul L2 Cache Tile Sizing
doc_path: cards/P105_matmul_l2_cache_tile_sizing.md
bottlenecks: [memory_bound, mte2_bound]
op_families: [matmul]
complexity: L1
triggers:
  - M * N * dtypeSize > L2 / 2 (large enough to benefit from L2 tiling)
  - B matrix rows/columns reused across multiple base blocks
  - Observed L2 cache thrashing or MTE2-bound with high busy but low bandwidth utilization
conflicts_with: []
synergizes_with: [P52, P107, P104]
requires: []
has_preconditions: true
has_playbook: true
references:
  - ops/ascendc-tiling-design/references/matmul/ascendc-api-matmul-tiling.md
---

# P105: Matmul L2 Cache Tile 尺寸规划（L2 Cache Tile Sizing）

## 核心思想

大矩阵乘中，若一次性把所有 base 块顺序下发，A/B 子块在 L2 中会被后续 tile 反复挤出，造成 MTE2 重读。P105 在 Host 侧增加 **L2 级 Tiling**：把 M/N 方向切分为若干 L2 tile，每个 L2 tile 内的 base 块共享 L2 中的 A 行 / B 列，并保证 `mTileBlock * nTileBlock * dtypeSize` 不超过 L2 容量的一半，从而提升数据局部性、降低 MTE2 压力。

## 代码骨架

```cpp
// === 改造前：只有 Block/BaseBlock 两级切分 ===
L2cacheTilePara l2Para{}; // 未使用或 mTileCntL2 = nTileCntL2 = 1

// === 改造后：引入 L2 级切分 ===
constexpr uint32_t L2_SIZE = 2 * 1024 * 1024; // 以实际 platform 为准
L2cacheTilePara l2Para{};

// 小算子退化
if (M * N * dtypeSize > L2_SIZE / 2) {
    l2Para.mTileCntL2 = ceil(mTotalCnt, l2Para.mTileBlock);
    l2Para.nTileCntL2 = ceil(nTotalCnt, l2Para.nTileBlock);
    // 让单个 L2 tile 的 base 块 footprint ≈ L2/2
    // 例如：l2Para.mTileBlock = ceil(mTotalCnt, l2Para.mTileCntL2);
    //       l2Para.nTileBlock = ceil(nTotalCnt, l2Para.nTileCntL2);
} else {
    l2Para.mTileCntL2 = 1;
    l2Para.nTileCntL2 = 1;
}

l2Para.calOrder = (M >= N) ? ROW_FIRST : COL_FIRST;
```

## 关键修改点

1. **容量 guard**：单个 L2 tile 内 A/B 的工作集应 `≤ L2 / 2`，否则退化为 1。
2. **calOrder 选择**：`M ≫ N` 选 `ROW_FIRST` 复用 A 行；`N ≫ M` 选 `COL_FIRST` 复用 B 列。
3. **派生计数**：`mCnt = ceil(mTotalCnt, mTileCntL2)`，`nCnt = ceil(nTotalCnt, nTileCntL2)`，`totalTileCnt = mCnt * nCnt`。
4. **与 Block 级配合**：Block 级仍按 `totalTileCnt` 分配核，L2 级只改变遍历顺序。

## 常见陷阱

⚠️ 小算子仍启用 L2 切分，引入不必要的调度开销。
⚠️ `mTileBlock / nTileBlock` 未对齐到 base 块整数倍，导致 L2 tile 边界尾块处理错误。
⚠️ `calOrder` 与 serpentine 遍历方向冲突：ROW_FIRST 时仍可用 serpentine，只是优先维度不同。
⚠️ L2 tile footprint 计算未包含双缓冲 / L1 占用，实际可用容量比 L2_SIZE / 2 更小。

## 代码搜索关键词

```bash
grep -nE "L2cacheTilePara|mTileCntL2|nTileCntL2|mTileBlock|nTileBlock|calOrder|ROW_FIRST|COL_FIRST" \
    op_host/*_tiling.cpp op_kernel/*.cpp
```

## 来源

- `ops/ascendc-tiling-design/references/matmul/ascendc-api-matmul-tiling.md` §L2 级 Tiling
