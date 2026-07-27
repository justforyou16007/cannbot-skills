---
source_key: evolution-strategies#card/P112_elementwise_vectorized_tile_sizing
id: P112
title: Elementwise 向量化 Tile 尺寸设计
doc_path: cards/P112_elementwise_vectorized_tile_sizing.md
complexity: L1
op_families: [elementwise]
triggers: [elementwise, vectorized, tile_sizing, ub_split]
references:
  - ops/ascendc-tiling-design/references/elewise/patterns.md
  - ops/ascendc-tiling-design/references/elewise/tiling.md
bottlenecks: [vec_underutilization, mte2_stall, tail_block]
conflicts_with: []
synergizes_with: [P1, P5, P10, P69]
has_preconditions: true
has_playbook: true
---

# P112: Elementwise 向量化 Tile 尺寸设计

## 核心思想

Elementwise 算子无跨元素依赖，应优先展平为 1D，并按 Vector 单元宽度与 UB 容量切分 tile。通过 `maxElemNum` 公式把 `bufferNum`、`maxDtypeBits`、`minDtypeBits` 纳入预算，同时保证多核对齐（512）与 UB 对齐（256B），实现 Vector 单元满载。

## 代码骨架

```cpp
// 1D 处理骨架
for (uint32_t blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
    uint32_t offset = blockIdx * blockFormer * sizeof(T);
    uint32_t curLen = (blockIdx == blockNum - 1) ? blockTail : blockFormer;
    CopyIn(inputUb, gmInput + offset, curLen);
    Compute(outputUb, inputUb, curLen);
    CopyOut(gmOutput + offset, outputUb, curLen);
}
```

## 关键修改点

1. **Shape 判定**：输入输出 shape 必须完全相同；否则走 Broadcast 路径。
2. **dtype 分支**：FP16/BF16 加减链路在 spec 未声明同量级时，应 Cast→FP32→Cast。
3. **maxElemNum 计算**：
   ```
   maxElemNum = (ubSize - extraSize) * 8 / (bufferNum * maxDtypeBits)
   maxElemNum = floor_align(maxElemNum, 256 * 8 / minDtypeBits)
   ```
4. **多核切分**：把 `ubSplitAxis` 及外层轴展平为 `fusedProduct`，均分给 `coreNum`。
5. **核利用率不足**：循环缩小 `maxElemNum`（每次减 `CACHE_LINE`）重算，直到喂满更多核。

## 常见陷阱

- ❌ 把 broadcast 场景误当 elementwise 处理 → 结果 shape 错误。
- ❌ FP16 加减未升精度 → 大数吃小数导致精度不达标。
- ❌ `maxElemNum` 未按 dtype 位宽折算 → UB 溢出。
- ❌ 尾 block 未单独处理 → 越界或向量 mask 错误。

## 代码搜索关键词

```bash
grep -nE "maxElemNum|ubFormer|blockFormer|blockTail|ELEM_ALIGN_FACTOR|CACHE_LINE" op_host/*_tiling.cpp op_kernel/*.cpp
```
