# P113 Playbook: Broadcast 对齐 Tiling 设计

> **强制流程**。本 Playbook 指导子 agent 对 Broadcast 算子执行 DimensionCollapse、广播方式选择与 UB 对齐切分。必须按 Step 1-5 逐步执行。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p113_locations.txt`：

```bash
# 1.1 shape / dims / strides / broadcast 标记
grep -nE "shape|dims|strides|broadcast|flag" \
    shared/original/op_host/*_tiling.{cpp,h} > /tmp/p113_locations.txt
# 1.2 DataCopy / NDDMA / BRC / Broadcast API
grep -nE "DataCopy|NDDMA|Broadcast|BRC|SetValue|GetValue" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p113_locations.txt
# 1.3 ubSplitAxis / maxElemNum
grep -nE "ubSplitAxis|maxElemNum|blockFormer" \
    shared/original/op_host/*_tiling.{cpp,h} >> /tmp/p113_locations.txt
# 1.4 OneDim / 多维分支
grep -nE "OneDim|shapeLen|collapsed" \
    shared/original/op_host/*_tiling.{cpp,h} >> /tmp/p113_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，列出广播相关代码位置。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 补维逻辑 | ? | 左侧补 1 | op_host/*_tiling.cpp:L? |
| 合轴函数 | ? | 合并相邻同 flag 轴 | op_host/*_tiling.cpp:L? |
| 广播轴 stride | ? | dim=1 且输出>1 时置 0 | op_host/*_tiling.cpp:L? |
| 广播方式 | ? | UB BRC / NDDMA | op_kernel/*.cpp:L? |
| ubSplitAxis | ? | 最内轴向外第一个放不下 | op_host/*_tiling.cpp:L? |
| NDDMA >5 轴处理 | ? | WITH_LOOP | op_kernel/*.cpp:L? |

**排除规则**：输入输出 shape 完全相同 → 走 P112 Elementwise。

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — OneDim**：合轴后只剩 1 维，可能含标量输入。
- **形态 β — 多维 + UB BRC**：NLast 或 dtype 对齐场景，调用 UB Broadcast API。
- **形态 γ — 多维 + NDDMA**：一般场景，GM→UB 阶段硬件广播。
- **形态 δ — NDDMA >5 轴**：最内 5 轴 NDDMA，外层 for-loop。

声明 `form: alpha | beta | gamma | delta`。

### 3B. Canonical Template（形态 γ）

```cpp
// Host tiling: 补维 + 合轴 + stride 计算
for (int j = 0; j < inputNum; ++j) {
    PadShapeToMaxRank(shapes[j], maxRank);
}
for (int i = maxRank - 1; i >= 0; --i) {
    flag[i] = 0;
    for (int j = 0; j < inputNum; ++j) {
        if (shapes[j][i] == 1 && outShape[i] > 1) flag[i] |= (1 << j);
    }
    if (i < maxRank - 1 && flag[i] == flag[i+1]) {
        collapsedDims[i] *= collapsedDims[i+1]; // 合并
    }
    strides[j][i] = (shapes[j][i] == 1 && outShape[i] > 1) ? 0 : runningStride;
    runningStride *= collapsedDims[i];
}

// UB 切分
uint32_t curProduct = 1;
int ubSplitAxis = 0;
for (int i = shapeLen - 1; i >= 0; --i) {
    curProduct *= dims[i];
    if (curProduct > maxElemNum) { ubSplitAxis = i; break; }
}
```

### 3C. Variant Notes

- **形态 α（OneDim）**：走 [onedim.md]，128B 对齐，可能直接调用 `Adds`/`Muls` 的 scalar broadcast 版本。
- **形态 β（UB BRC）**：NLast 且尾轴大，或 dtype 为 INT8/FP16/BF16 且尾轴 32B 对齐时，使用 `Broadcast` API。
- **形态 δ（>5 轴）**：
  ```cpp
  if (axesAfterSplit <= 5) {
      DataCopy<T, 5, config>(...);
  } else {
      for (uint32_t o = 0; o < outerLoops; ++o) {
          DataCopy<T, 5, config>(...); // 偏移随 o 变化
      }
  }
  ```

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P113 约束核查：
  1. 是否完成补维 + 合轴？ yes/no
  2. 广播轴 stride = 0？ yes/no
  3. 选择的广播方式 = UB BRC / NDDMA，依据？ ___
  4. ubSplitAxis = ___，maxElemNum = ___
  5. NDDMA 维度 = ___，是否 >5？ yes/no，是否拆 WITH_LOOP？ yes/no
  6. 对齐：OneDim 128B / 多维 256B 满足？ yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 合轴 / flag 计算存在
(grep -cE "flag\[|Collapse|collapsed" modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 1

# 检查 2: 广播轴 stride 置 0
(grep -cE "stride\s*=\s*0" modified_files/op_host/*_tiling.{cpp,h} modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 3: Broadcast / NDDMA / DataCopy API 存在
(grep -cE "Broadcast|NDDMA|DataCopy\s*<" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 4: ubSplitAxis / maxElemNum 存在
(grep -cE "ubSplitAxis|maxElemNum" modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 2

# 检查 5: >5 轴场景 WITH_LOOP 处理
(grep -cE "axesAfterSplit|WITH_LOOP|DataCopy\s*<\s*T\s*,\s*5" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1（形态 δ）或 0（其他形态，需说明）
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P113 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: form={alpha|beta|gamma|delta}, broadcast tiling applied
Step 4: constraints checked (broadcast=___, ubSplitAxis=___)
Step 5: all 5 grep checks passed
```

## 常见陷阱

- 未补维导致 flag 位图错位。
- 广播轴 stride 未置 0 → 读到错误位置。
- NLast 场景误用 NDDMA → dcache 被反复冲刷。
- NDDMA 维度超过 5 未拆分 → 编译失败。
- UB BRC 对齐条件不满足 → 性能下降或结果异常。
