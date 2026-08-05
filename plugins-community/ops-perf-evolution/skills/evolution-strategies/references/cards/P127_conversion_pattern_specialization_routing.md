---
id: P127
bottlenecks: [mte2_stall, scalar_compute, compute_bound]
op_families: [special, elementwise]
complexity: L1
conflicts_with: []
synergizes_with: [P26, P114, P7, P1]
requires: []
has_preconditions: true
has_playbook: true
---

# P127: Conversion 模式特化路由（Transpose/Permute 禁止通用 gather 主路径）

## 核心思想

Layout-transform 类算子（permute / transpose / reshape 触发拷贝 / split / concat）的全部代价在地址重排。一个标量 gather kernel 处理所有 permutation 的实现，性能比模式特化低 1-2 个数量级。正确做法是 **R1→R5 五段路由**：

1. **R1 View 短路**：identity permutation 或仅移动 size-1 维度 → 直接返回 view，不分配输出、不启动 kernel。
2. **R2 连续维合并降 rank**：置换后仍连续的维度先 reshape 合并，降低有效 rank 再分发。例：`[A,B,M,N]→[A,B,N,M]` 合并为 `(A*B,M,N)` 后走 batch transpose。
3. **R3 退化 shape 路由**：batch=1 时 batch transpose 降级到 2D transpose kernel。
4. **R4 模式特化分发**：2D transpose / batch transpose / swap-adjacent / reverse dims 等高频模式各有专用 kernel；small-channel（C≤16）走 P114（TransDataTo5HD 路径）；单维转置走 P26（stride 零拷贝搬运）。
5. **R5 通用 fallback**：仅罕见 permutation 保留 1D gather/scatter fallback，正确性优先于性能。

## 两条禁令

- ❌ **禁止单一 generic gather 冒充特化**：所有输入走同一个标量 gather/scatter kernel 视为未特化，必须重写；fallback 分支不得覆盖 2D/batch transpose 等高频模式。
- ❌ **专用 kernel 内禁止逐元素索引**：不允许逐元素 div/mod 或 GetValue/SetValue 做地址计算；专用 kernel 内部必须是 tile-based 规则连续搬运（DataCopyPad/DataCopy 的 stride/blockCount/blockLen 参数化）。

## 代码骨架

```cpp
// host 侧模式分发（伪码，变量名/tile 需按实际 shape 设计）
if (IsIdentityPerm(dims) || OnlyMovesSizeOneDims(shape, dims))
    return View(out_shape);                       // R1
auto [mergedShape, mergedDims] = MergeContiguousDims(shape, dims);  // R2
if (mergedDims == {1, 0})
    return LaunchTranspose2D(mergedShape);        // R4: P26 stride 零拷贝
if (IsBatchTranspose(mergedDims)) {
    if (batch == 1) return LaunchTranspose2D(...); // R3 退化
    return LaunchBatchTranspose(mergedShape);      // R4 专用 kernel
}
if (IsSwapAdjacent(mergedDims)) return LaunchSwapAdjacent(...);
if (SmallChannel(mergedShape))  return LaunchP114Path(...);  // C≤16 → P114
return LaunchGenericPermute(mergedShape);         // R5 fallback，正确性优先
```

## 关键修改点

1. 先查 R1/R2：大量"permute"调用实际可以零拷贝解决，kernel 都不用写
2. 每个专用 kernel 的 grid 按物理核数限制，kernel 内循环处理多 block，避免 grid 远大于核数的调度开销
3. 专用 kernel 的 tile 参数（BLOCK_M/BLOCK_N）列入调优空间，按 M/N 选优

## 常见陷阱

- ⚠️ fallback 分支覆盖了高频模式 → 性能悬崖且不易察觉（功能正确）
- ⚠️ 专用 kernel 里用逐元素 div/mod 算偏移 → 等价于标量 gather，白特化
- ⚠️ 跳过 R2 直接分发 → 高维 kernel 的 div/mod 寻址开销吃掉 tile 搬运收益

## 代码搜索关键词

```bash
grep -nE "permute|transpose|Permute|Transpose|GetValue|SetValue" op_kernel/*.cpp op_host/*.cpp
```

## 来源

用户提供的《Triton Ascend Layout-transform / Permutation 代码生成要点》（2026-08），翻译为 AscendC 语境并与 P26/P114 分工对齐。
