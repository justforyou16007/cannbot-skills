# P109 Playbook: Reduction 多轴 Tiling 系统化设计

> **强制流程**。本 Playbook 指导子 agent 对 reduction 算子按 Ascend C Tiling 方法论进行多轴合轴与 AR/ARA 模式重构。必须按 Step 1-5 逐步执行。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p109_locations.txt`：

```bash
# 1.1 shape / axes / reduce 相关 tiling 参数
grep -nE "shape|axes|reduce|axis|rLength|rLengthAlign|tileA0Len" \
    shared/original/op_host/*_tiling.{cpp,h} > /tmp/p109_locations.txt
# 1.2 Reduce API 使用现状
grep -nE "ReduceMax|ReduceSum|ReduceMin|Pattern::Reduce::RA|Level2" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p109_locations.txt
# 1.3 多维循环 / 嵌套归约
grep -nE "for\s*\(.*reduce|while\s*\(.*reduce|nested" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p109_locations.txt
```

**交付物**（写入 `implementation_note.txt` 的 "Playbook Step 1"）：
- shape/axes 定义位置（file:line）。
- 当前使用的 Reduce API 及其位置。
- 是否存在显式多维归约循环。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 合轴函数 | ? | 输出 A/R 序列 + collapsedShape | op_host/*_tiling.cpp:L? |
| A0 值 | ? | 1（AR）或 >1（ARA） | op_host/*_tiling.cpp:L? |
| Reduce API | ? | Level2 / Pattern::Reduce::RA | op_kernel/*.cpp:L? |
| alignedCols | ? | 32B 对齐值 | op_host/*_tiling.cpp:L? |
| 多轴处理 | 单循环 | 逐级处理每个 R | op_kernel/*.cpp:L? |

**排除规则**：
- 算子不是归约语义 → 本策略不适用。
- 已有完整合轴 + AR/ARA 路由 → 跳过。

## Step 3: 代码改造

### 3A. 形态识别

判断当前算子属于：

- **形态 α — 单轴 AR**：合轴后为 `(A, R)`，A0=1，R 为尾轴。
- **形态 β — 单轴 ARA**：合轴后为 `(A, R, A)`，A0>1，R 非尾轴。
- **形态 γ — 多轴 ARAR...**：需要 Shape 三步变换后逐级归约。

在 `implementation_note.txt` 中声明：`form: alpha | beta | gamma`。

### 3B. Canonical Template（形态 α / β）

```cpp
// === Host tiling：合轴 + A/R 标记 ===
auto collapsed = CollapseShape(shape, axes);   // 返回 {collapsedShape, arFlags}
uint32_t A0 = collapsed.dim[0];
uint32_t R  = collapsed.dim[1];
bool isARA = (collapsed.flags.size() == 3);    // A,R,A

if (!isARA) {
    // AR: 使用 Level2 Reduce
    ReduceSum<T, true>(outLocal, inLocal, tmpLocal, R);
} else {
    // ARA: 使用 Pattern::Reduce::RA
    uint32_t tileA0Len = min(A0, perCoreA0);
    uint32_t alignedCols = ((tileA0Len * sizeof(T) + 31) / 32) * 32 / sizeof(T);
    uint32_t srcShape[] = {R, alignedCols};
    ReduceMax<T, Pattern::Reduce::RA>(outLocal, inLocal, tmpLocal, srcShape, true);
}
```

### 3C. Variant Notes

- **形态 γ（多轴）**：先按 [multi-axis-transform.md] 把 ARARA 压缩，按最外层 R 切分 tile，每轮输出 shape 降维后继续处理下一个 R。
- **AR 但 UB 放不下整行 R**：走 AR-ColSplit，分 chunk 后用 Add/Max 合并。
- **ARA 但 UB 放不下 R×tileA0Len**：走 ARA-RowSplit，每次搬入 r < R 行。
- **需要 with-index（argmax）**：AR-FullLoad 用 `ReduceMax(calIndex=true)`；ARA 用 Compare+Select 逐行迭代。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P109 约束核查：
  1. 合轴后是否为 A/R 交替序列？ yes/no
  2. A0 = ___，模式 = AR/ARA
  3. alignedCols = ___，是否 32B 对齐？ yes/no
  4. Reduce API count 使用 rLength=___（有效数据） yes/no
  5. Buffer 分配使用 rLengthAlign=___（对齐后） yes/no
  6. 多轴场景是否逐级处理？ yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 合轴 / A/R 标记代码存在
(grep -cE "CollapseShape|arFlags|A0|isARA" modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 1

# 检查 2: ARA 模式下出现 Pattern::Reduce::RA
(grep -cE "Pattern::Reduce::RA" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1（ARA 形态）或 0（纯 AR 形态，需在 note 说明）

# 检查 3: alignedCols 按 32B 对齐计算
(grep -cE "alignedCols\s*=.*\+\s*31.*\/\s*32" modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 1

# 检查 4: 未使用原始多维标量归约循环
BEFORE=$(grep -cE "for\s*\([^)]*\)\s*\{[^}]*\+=[^}]*\[" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
AFTER=$(grep -cE "for\s*\([^)]*\)\s*\{[^}]*\+=[^}]*\[" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "multi-dim scalar loops: before=$BEFORE after=$AFTER"
# 期望: AFTER <= BEFORE

# 检查 5: Reduce API 使用增加
RB=$(grep -cE "\bReduceSum|ReduceMax|ReduceMin\s*\(" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
RA=$(grep -cE "\bReduceSum|ReduceMax|ReduceMin\s*\(" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "Reduce APIs: before=$RB after=$RA"
# 期望: RA >= RB
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P109 Playbook Completion]
Step 1: done, locations written to /tmp/p109_locations.txt
Step 2: plan table filled
Step 3: form={alpha|beta|gamma}, template applied
Step 4: constraints checked (alignedCols=___, mode=AR/ARA)
Step 5: all 5 grep checks passed
```

## 常见陷阱

- 合轴时把应保持的 A 维误合并进 R → shape 错误。
- `alignedCols` 计算未用 `sizeof(T)` 折算 → 32B 对齐失败。
- 多轴场景一次性调用 Reduce 所有轴 → UB 溢出。
