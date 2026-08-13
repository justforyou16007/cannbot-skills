---
id: P139
bottlenecks: [near_optimal]
op_families: [flash_attention, attention, elementwise]
complexity: L1
conflicts_with: []
synergizes_with: [P14, P40, P137]
requires: []
has_preconditions: true
has_playbook: true
quantified_gain:
  - shape: "SparseFlashAttention 20-case 910c（同 kernel 连续复测）"
    baseline_us: 0
    optimized_us: 0
    speedup: 1.0
    source: "cannbench-evo round_6 verify_1/2/3：偶发 NaN/崩溃 → 连续 3 轮 20/20 稳定"
---

# P139: 多槽流水 workspace 全槽零初始化（防 padded 行 NaN → 偶发崩溃）

## 核心思想

P14 式 CV 预发射用 `kPreloadNum=2` 双槽流水时，一次性零初始化（`ZeroWorkspaceRows`）**只填 slot0 的 K/V**，slot1（或后续槽）的 padded 行 `[validCount, tileDim)` 从未初始化——gather 只写紧凑的 `[0, validCount)`，而 AIC 的 CubeQK 读整个 tile 宽度。残留内存若含 NaN/inf 位模式 → `Q·K^T` 溢出 → score/输出 NaN → **偶发崩溃或精度错**（内容随机，非 shape 决定，本地/服务端表现不同）。修复：零初始化**遍历所有槽**，且对 Cube 直接读取的 score/prob 等 padded 区域一并清零。

## 代码骨架

```cpp
// 只填 slot0 = BUG（slot1 的 padded 行未初始化 → NaN）
// 修复：遍历全部 kPreloadNum 槽
for (uint32_t slot = 0; slot < kPreloadNum; ++slot) {
    ZeroRows(kvLocal[slot], laneBegin, laneEnd);   // 每槽 K/V 行段
}
// 对 AIC 直接读的 prob/score padded 行（[activeG, mPad)）同样一次性清零
```

## 关键修改点

1. **零初始化范围 = 所有被读取的 padded 区域**：gather 只写 `[0, validCount)`，但消费方（AIC GEMM / softmax）读整个 tile 宽——凡"紧凑写 + 整块读"的缓冲都必须覆盖。
2. **失败定位**：偶发崩溃 + 本地不复现（`torch::empty` 残留偶发有限）→ 服务端暴露。先查"写入范围 < 读取范围"的 workspace 槽，而非只看数据越界。
3. **与 P40 分工**：P40 是 workspace 双缓冲的**结构**策略；P139 是双（多）槽**初始化完整性**的稳定性策略，叠加使用。

## 实测验证

- SparseFlashAttention 在 910c 上随机崩溃（case 9 与 case 13 同 shape 一个崩一个不崩）→ K/V 全槽 + prob 零填充后连续 3 轮 20/20 稳定、无 NaN 抖动，性能损失 <1%（零填充开销可忽略）。
