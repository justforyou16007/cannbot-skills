---
id: P138
bottlenecks: [near_optimal, db_not_effective]
op_families: [attention, flash_attention, matmul]
complexity: L1
conflicts_with: []
synergizes_with: [P135]
requires: []
has_preconditions: true
has_playbook: true
quantified_gain:
  - shape: "SparseFlashAttention 20-case 910c（BenchSite 同 kernel 复测）"
    baseline_us: 0
    optimized_us: 0
    speedup: 1.0
    source: "cannbench-evo round_11→12，job_e23199ece961：随机 SIGSEGV(10/20 无性能数据) → 修复后 20/20 全测，score 59.24"
---

# P138: host tiling 查询 L0A/L0C 容量并按需自适应降维（跨硬件稳定性）

## 核心思想

Cube 侧 host tiling 常用**当前 SoC 实测值写死**（如 `groupTile` / `baseM` 按 910B3 的 128KB L0C 定值）。但 L0A/L0C 容量随芯片变化（910B2C vs 910c），写死的 tile 在更小 L0 的芯片上 `InitBuffer` 溢出 → **随机 SIGSEGV / 部分 case 无性能数据**（时序相关，难复现，且与 shape 无确定对应）。修复：host tiling 用 `platform->GetCoreMemSize(L0_A / L0_C)` **运行时查询** L0 容量，`groupTile`（或 baseM）不满足时**自适应减半**，直到 L0 与 UB 约束同时满足。

## 代码骨架

```cpp
// host tiling（防 InitBuffer 溢出，跨硬件可移植）
uint64_t l0aSize = 0, l0cSize = 0;
platform->GetCoreMemSize(CoreMemType::L0_A, l0aSize);
platform->GetCoreMemSize(CoreMemType::L0_C, l0cSize);
// device 端 l0C_ = groupTile × 256 × 4B；若占满查询到的 L0C 则减半
while (l0cSize > 0 && groupTile * 256 * 4 > l0cSize) { groupTile /= 2; }
if (groupTile <= 0) groupTile = 1;
```

## 关键修改点

1. **L0 容量是运行时资源，不是编译期常量**：`GetCoreMemSize(L0_A/L0_C)` 在 host tiling 里查询，device `InitBuffer` 的 l0A_/l0C_ 布局必须与查询到的容量一致。
2. **与 P135 的分工**：P135 是"L0C 占满时减半 baseN 开双缓冲"的**性能**策略；P138 是"容量不满足时降维防溢出"的**稳定性/可移植性**策略，可叠加。
3. **症状识别**：随机 SIGSEGV、同 kernel 不同卡/芯片复测崩溃的 case 集不同、且非 shape 决定——优先怀疑写死的 L0 尺寸，而非数据越界。

## 实测验证

- SparseFlashAttention 在 910c（BenchSite）上随机崩溃（两次复测崩互补的一半 case）→ 加 L0 查询 + groupTile 自适应减半后 **20/20 全部有真实性能数据**，score 59.24，后续所有提交稳定。
