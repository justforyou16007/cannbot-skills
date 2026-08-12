---
id: P137
bottlenecks: [scalar_loading, cube_bound]
op_families: [attention, flash_attention]
complexity: L2
conflicts_with: []
synergizes_with: [P110, P14, P129]
requires: []
has_preconditions: true
has_playbook: true
quantified_gain:
  - shape: "SparseFlashAttention GQA topK512 Dk128 kvSeq4096（case1, 910B2C→910c）"
    baseline_us: 27251.88
    optimized_us: 1578.0
    speedup: 17.28
    source: "cannbench-evo round_18（紧凑化→Cube 全程，cube_utilization≈98%）"
---

# P137: Token 级稀疏 Attention 的 workspace 紧凑化走 Cube（top-k 不再回退 Vector）

## 核心思想

Token 级稀疏（top-k）FA 被误认为"不规则索引无法用 Cube"。实测否定该假设：**每个 (head, qSeq) 的 topK 索引先经 AIV 标量 gather 搬成 workspace 里行紧凑的 Q/K/V tile（有效 token 前置、尾部零填充到 Cube tile 对齐），再喂 AIC mmad**。QK^T / PV 的 GEMM 全部落在 Cube（cube_utilization≈98%），AIV 只做不规则 gather 与 online-softmax，与 AIC GEMM 经 CrossCore flag 流水重叠。稀疏不改变"用 Cube"的结论，只是多一层 gather→紧凑化的搬运契约。

## 代码骨架

```cpp
// workspace 布局（每 core 一份）：compact Q[M_tile×Dk] + K/V[preload×N_tile×Dk] + score/prob
// AIV 侧：逐 token 按 topK 索引 gather K/V 到紧凑行（DataCopyPad 批量搬，标量寻址）
// AIC 侧：CubeQK（M_tile×Dk 乘 Dk×N_tile）+ CubePV，padded 行恒 0（见 P139 零填充）
// 同步：gather 完成 Set signal → AIC 读该槽 GEMM；AIC 算完 Set → AIV 复用槽
```

## 关键修改点

1. **紧凑化契约**：K/V 按 [validCount, tileDim) 前置有效、尾部 `memset 0`，使 AIC 读整个 256 行不读未初始化内存（配合 P139 全槽零填充）。
2. **M-tile 由共享 K/V 的查询行数决定**：GQA 只有 groupSize 行（小），MLA 可到 qHeads（见 P140）——紧凑化把"M 天生小"从"Vector 全标量"变成"小 M Cube GEMM + 大 gather 摊薄"。
3. **先消 gather 标量瓶颈再谈 M 维融合**（已验证失败路径）：M-merge 增大的 Cube/score 流量超过 task 摊薄收益，gather 未向量化时反而慢 11-15%。顺序必须是 gather 向量化 → 再融合。

## 实测验证

- 全程 27251.88μs → 1578μs（17.28x）；`aic_mac` 从近乎 0 上升，`cube_utilization≈98.6%`。
- 服务端（910c）score 从 53.53（Vector 路径）→ 64.20（紧凑化→Cube 路径），case 7 达 1.96x vs golden。
- 反例留存：`sparse_design.md` 早期占位断言"Token-level top-k → Vector"已由本卡实测推翻。
