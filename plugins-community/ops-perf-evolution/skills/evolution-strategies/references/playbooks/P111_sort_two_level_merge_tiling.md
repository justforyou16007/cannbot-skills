# P111 Playbook: Sort 两级归并 Tiling

> **强制流程**。本 Playbook 指导子 agent 对 Sort / TopK 大数据量场景实现 Pattern C 两级归并。必须按 Step 1-5 逐步执行。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p111_locations.txt`：

```bash
# 1.1 Sort / MrgSort API
grep -nE "Sort<|MrgSort|GetConcatTmpSize|GetSortTmpSize" \
    shared/original/op_kernel/*.{cpp,h} > /tmp/p111_locations.txt
# 1.2 tileSize / coreNum / N / K
grep -nE "tileSize|coreNum|totalElements|topK|K_" \
    shared/original/op_host/*_tiling.{cpp,h} >> /tmp/p111_locations.txt
# 1.3 SyncAll / workspace / workSpaceFlag
grep -nE "SyncAll|workspace|workSpaceFlag|GetCoreWsOffset" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p111_locations.txt
# 1.4 truncation / UpdateSortInfo
grep -nE "truncationFlag|UpdateSortInfo|stopThreshold" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p111_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，列出现有 Phase 划分与 API。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| Pattern | ? | A/B/C | op_host/*_tiling.cpp:L? |
| tileSize | ? | 4096（对齐32） | op_host/*_tiling.cpp:L? |
| Phase 1 Sort | ? | 各核并行 tile 排序 | op_kernel/*.cpp:L? |
| Phase 2/3 归并 | ? | 核内 + 跨核 M-way 归并 | op_kernel/*.cpp:L? |
| Phase 4 输出 | ? | Core 0 Extract + Cast | op_kernel/*.cpp:L? |
| workspace 双缓存 | 无 | workSpaceFlag 翻转 | op_kernel/*.cpp:L? |
| 截断逻辑 | 无 | truncationFlag | op_kernel/*.cpp:L? |

**排除规则**：
- `N ≤ tileSize` → 走 Pattern A，本 Playbook 不适用。
- 算子不含排序语义 → 不适用。

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — Pattern B**：`tileSize < N ≤ tileSize × coreNum`，只有 Phase 1 + 跨核归并。
- **形态 β — Pattern C**：`N > tileSize × coreNum`，需要完整四阶段 + 截断。

声明 `form: alpha | beta`。

### 3B. Canonical Template（形态 β）

```cpp
__aicore__ inline void Process()
{
    // Phase 1: 各核并行排序 tile
    uint32_t myTiles = (blockIdx_ < usedCore - 1) ? frontCoreTiles_ : lastCoreTiles_;
    for (uint32_t t = 0; t < myTiles; ++t) {
        uint32_t globalOffset = blockIdx_ * frontCoreTiles_ * tileSize_ + t * tileSize_;
        uint32_t curTileSize = (blockIdx_ == usedCore - 1 && t == myTiles - 1) ? lastTileSize_ : tileSize_;
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

### 3C. Variant Notes

- **形态 α（Pattern B）**：跳过 Phase 2，各核排序后直接跨核归并。
- **Top-K 截断**：当 `currentElements * M >= K` 时，每轮归并后设置 `truncationFlag=true`，下一轮只读前 K。
- **非 Top-K 全排序**：始终 `truncationFlag=false`，处理全量数据。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P111 约束核查：
  1. N = ___, tileSize = ___, coreNum = ___, Pattern = A/B/C
  2. tileSize 对齐 32？ yes/no
  3. UB 中 Concat/Sort tmp 通过 GetConcatTmpSize / GetSortTmpSize 获取？ yes/no
  4. workspace 大小 = ___ bytes，按核数分配？ yes/no
  5. onceMaxElementsMerge = ___, onceMaxElementsOutput = ___
  6. 总 SyncAll 次数 = ceil(log_M(coreNum)) + 1 = ___
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: Sort / MrgSort API 存在
(grep -cE "Sort\s*<|MrgSort\s*<" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 2: tileSize 取 4096 并对齐 32
(grep -cE "tileSize\s*=\s*4096|tileSize_\s*=\s*4096" modified_files/op_host/*_tiling.{cpp,h} modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 3: SyncAll 出现在 Phase 2 后及 Phase 3 循环内
(grep -cE "SyncAll\s*\(\s*\)" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 2（Pattern C）或 >= 1（Pattern B，需说明）

# 检查 4: workSpaceFlag 翻转
(grep -cE "workSpaceFlag\s*=\s*\(\s*workSpaceFlag\s*\+\s*1\s*\)\s*%\s*2" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 5: UpdateSortInfo 调用
(grep -cE "UpdateSortInfo\s*\(" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P111 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: form={alpha|beta}, 4-phase template applied
Step 4: constraints checked (Pattern=___, tileSize=___, SyncAll=___)
Step 5: all 5 grep checks passed
```

## 常见陷阱

- `truncationFlag` 设置时机错误 → 首次截断数据缺失。
- Phase 3 每 group 后 SyncAll → 过度同步。
- `UpdateSortInfo` 遗漏 → 死循环。
- workspace 按总元素数分配 → 尾核越界。
- `GetSortTmpSize` 硬编码 → 不同芯片 UB 占用不同。
