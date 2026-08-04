---
id: P131
bottlenecks: [compute_bound]
op_families: [matmul, special, quantization]
complexity: L1
conflicts_with: []
synergizes_with: [P129, P130]
requires: []
has_preconditions: true
has_playbook: true
---

# P131: Host 侧权重预打包结果 LRU 缓存（Weight Repack Cache）

> 晋升自 X20（cann-bench conv_2d 评测全命中，四级流水下 wrepack 从每调用一次变为仅首调）。详细实战记录见 `discovered/disc_X20.md`。

## 核心思想

权重布局变换（K-major repack、fractal 重排、量化打包等）的输出是（weight 内容, shape, dtype, 目标布局）的**确定函数**。推理场景同一权重张量被反复卷积/矩阵乘，repack 结果逐 bit 相同——host 侧做 LRU 缓存，命中时**跳过整个 repack kernel 发射**，直接复用缓存的 workspace（每次调用省 ~5-15us 级，评测/推理循环命中率 ~100%）。

## 适用场景

- 任何带权重预打包步骤的算子（conv、matmul、quant matmul），推理式重复调用模式
- 瓶颈：launch-bound；或 repack kernel 本身耗时占比可测
- 前提：repack 无随机性、无输入数据依赖；same-stream 执行

## 关键修改点

1. 缓存键：`data_ptr + sizes + dtype + version counter`——**version 必需**：in-place 修改（`copy_`/`fill_` 等）会 bump version 强制重打包，否则脏数据
2. **条目内持有 weight 强引用**：防止 allocator 回收 data_ptr 后被新张量复用地址造成 stale 命中
3. 缓存的 workspace 张量由条目持活，miss 时整体重写；LRU 容量小（个位数条目）
4. 多目标布局（如 normal k 序 vs fused k 序）时**布局标志进缓存键**
5. 留环境变量开关（如 `*_WREPACK_CACHE=0`）便于 A/B 与排障
6. 纯 host C++/ATen 逻辑，无 device 侧改动，全平台可移植

## 代码搜索关键词

```bash
grep -nE "repack|wrepack|Repack|weight.*pack" op_host/*.cpp op_kernel/*.cpp
```
