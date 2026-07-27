# P110 Playbook: Reduction UB 累加器复用

> **强制流程**。本 Playbook 指导子 agent 在 reduction 算子中利用 UB 作为累加器/合并 buffer，避免重复 GM 搬入。必须按 Step 1-5 逐步执行。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p110_locations.txt`：

```bash
# 1.1 tmpBuf / accumulator buffer 声明与分配
grep -nE "tmpBuf|tmpLocal|accumulator|accLocal|sumBuf|VECCALC" \
    shared/original/op_kernel/*.{cpp,h} > /tmp/p110_locations.txt
# 1.2 Reduce API 调用点
grep -nE "ReduceSum|ReduceMax|ReduceMin|Pattern::Reduce::RA" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p110_locations.txt
# 1.3 rLength / rLengthAlign 使用
grep -nE "rLength|rLengthAlign" \
    shared/original/op_host/*_tiling.{cpp,h} >> /tmp/p110_locations.txt
# 1.4 Split / chunk 循环
grep -nE "chunk|ColSplit|RowSplit|split" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p110_locations.txt
```

**交付物**：file:line 列表写入 `implementation_note.txt` "Playbook Step 1"。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 模式 | ? | FullLoad / Split | op_host/*_tiling.cpp:L? |
| tmpBuf 大小 | ? | 按 rLengthAlign | op_host/*_tiling.cpp:L? |
| Reduce workspace | ? | 单独 TBuf >= 8KB | op_kernel/*.cpp:L? |
| 合并方式 | 无 | Max/Add per chunk | op_kernel/*.cpp:L? |
| accumulator buffer | 无 | accLocal / outLocal 复用 | op_kernel/*.cpp:L? |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — FullLoad**：UB 能一次放下整行/整块，Reduce 后结果直接复用 UB 输出。
- **形态 β — ColSplit（AR）**：R 方向分 chunk，每个 chunk Reduce 后跨 chunk Add/Max 合并。
- **形态 γ — RowSplit（ARA）**：R 行方向分 chunk，每批 r 行 Reduce 后跨 batch 合并。
- **形态 δ — 精度敏感大 R**：在 Split 基础上叠加 Welford Online 或二分累加。

声明 `form: alpha | beta | gamma | delta`。

### 3B. Canonical Template（形态 α）

```cpp
// === FullLoad：一次搬入 + Reduce + 复用 UB 输出 ===
DataCopyPad(inLocal, gmInput + offset, rLengthAlign);
ReduceSum<T, true>(outLocal, inLocal, tmpLocal, rLength);
DataCopyPad(gmOutput + outOffset, outLocal, tileA0Len);
```

### 3C. Variant Notes

- **形态 β（ColSplit）**：
  ```cpp
  for (uint32_t c = 0; c < numChunks; ++c) {
      DataCopyPad(partLocal, gmInput + c * Ar, ArAlign);
      ReduceMax<T, true>(partLocal, partLocal, tmpLocal, Ar);
      if (c == 0) Copy(accLocal, partLocal);
      else Max(accLocal, accLocal, partLocal, alignedA0);
  }
  ```
- **形态 γ（RowSplit）**：每次搬入 r 行，Reduce 后结果与之前 chunk 按列方向合并。
- **形态 δ（精度敏感）**：Sum 大 R 时启用二分累加；var/std 启用 Welford Online。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P110 约束核查：
  1. 模式 = FullLoad / ColSplit / RowSplit
  2. tmpLocal / workspace 大小 >= ___ bytes
  3. rLengthAlign = ___，用于 DataCopyPad blockLen / buffer 分配
  4. rLength = ___，用于 Reduce API count
  5. Split 场景合并 API = Add / Max / 其他
  6. UB 新增占用 = ___ <= tile 预算剩余 ___
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: tmpBuf / accumulator buffer 声明存在
(grep -cE "tmpBuf|tmpLocal|accLocal|sumBuf|TBuf\s*<\s*VECCALC\s*>" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 2: rLengthAlign 出现在 buffer 分配或 offset 计算
(grep -cE "rLengthAlign" modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 1

# 检查 3: rLength 出现在 Reduce API count
(grep -cE "ReduceSum|ReduceMax|ReduceMin\s*<[^>]+>\s*\([^,]+,[^,]+,[^,]+,\s*rLength" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 4: Split 场景存在合并操作
(grep -cE "Add\s*\(\s*accLocal|Max\s*\(\s*accLocal" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1（Split 形态）或 0（FullLoad 形态，需说明）

# 检查 5: DataCopyPad 使用 rLengthAlign
(grep -cE "DataCopyPad.*rLengthAlign" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P110 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: form={alpha|beta|gamma|delta}, template applied
Step 4: constraints checked (mode=___, rLength=___, rLengthAlign=___)
Step 5: all 5 grep checks passed
```

## 常见陷阱

- `tmpLocal` 与输出 buffer 复用 → Reduce 覆盖结果。
- Split 合并 API 与算子语义不匹配（Sum 用 Max）。
- `rLengthAlign` 用于 API count → 归约范围错误。
- workspace 未 8B 对齐 → `GetValue` 返回异常。
