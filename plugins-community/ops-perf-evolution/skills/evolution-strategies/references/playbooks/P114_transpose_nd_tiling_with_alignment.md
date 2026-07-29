# P114 Playbook: Transpose 多维对齐 Tiling 设计

> **强制流程**。本 Playbook 指导子 agent 对 small-channel transpose 算子按 `[C, N] → [N, C]` 模型进行 Tiling 与对齐设计。必须按 Step 1-5 逐步执行。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p114_locations.txt`：

```bash
# 1.1 transpose / permute / shape 相关
grep -nE "transpose|permute|perm|NCHW|NHWC" \
    shared/original/op_host/*_tiling.{cpp,h} > /tmp/p114_locations.txt
# 1.2 TransDataTo5HD / vnchwconv / Gather
grep -nE "TransDataTo5HD|vnchwconv|Gather|Cast" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p114_locations.txt
# 1.3 tileN / tileNA / repeats / C
grep -nE "tileN|tileNA|repeats|smallChannel|channel" \
    shared/original/op_host/*_tiling.{cpp,h} >> /tmp/p114_locations.txt
# 1.4 offset table
grep -nE "offsetBuff|SetValue|offset.*table" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p114_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，列出 transpose 相关位置。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 通道维 C | ? | ≤16 | op_host/*_tiling.cpp:L? |
| 长轴 N | ? | 其余轴乘积 | op_host/*_tiling.cpp:L? |
| tileN | ? | 32 对齐 | op_host/*_tiling.cpp:L? |
| tileNA | ? | AlignUp(tileN,32) | op_host/*_tiling.cpp:L? |
| repeats | ? | tileNA/16 ≤255 | op_host/*_tiling.cpp:L? |
| UB 预算 | ? | tileNA*(16*C+32) | op_host/*_tiling.cpp:L? |
| offset table | 无 | 按 tileNA 构建 | op_kernel/*.cpp:L? |

**排除规则**：通道维过大或重排后轴不再相邻 → 走通用 transpose API 或其他策略。

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 纯 small-channel transpose**：`[C, N] → [N, C]`，无后处理。
- **形态 β — transpose + elementwise 后处理**：如转置后再 Cast/Relu。
- **形态 γ — 多小通道轴合并**：多条相邻小轴合成一个 C。

声明 `form: alpha | beta | gamma`。

### 3B. Canonical Template（形态 α）

```cpp
// Host tiling
uint32_t C = channelDim;
uint32_t N = totalElements / C;
uint32_t ubBudget = ubSize - reservedBytes;
uint32_t perElemBytes = 16 * C + 32;
uint32_t tileNMax = AlignDown(ubBudget / perElemBytes, 32);
tileNMax = Min(tileNMax, 255 * 16);

uint32_t tileN = AlignUp(CeilDiv(N, blockDim), 32);
if (tileN > tileNMax) {
    uint32_t minTiles = CeilDiv(N, tileNMax);
    uint32_t alignedTiles = CeilDiv(minTiles, blockDim) * blockDim;
    tileN = AlignUp(CeilDiv(N, alignedTiles), 32);
}
uint32_t tileNA = AlignUp(tileN, 32);
uint32_t repeats = tileNA / 16;

// Kernel
for (uint32_t t = startTile; t < endTile; ++t) {
    CopyIn(t);
    Compute(t);   // elementwise -> half -> vnchwconv -> gather
    CopyOut(t);
}
```

### 3C. Variant Notes

- **形态 β**：把 elementwise 后处理融入 Compute，与 transpose 共用 UB pipeline，不要回 GM。
- **形态 γ**：确保被合并的小轴在目标 layout 中仍然相邻，否则不能强行合并。
- **尾块**：`curN` 可能小于 `tileNA`，但 `tileNA` 保持对齐用于 UB 组织，CopyOut 按 `curN * C` 字节写回。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P114 约束核查：
  1. C = ___，是否 ≤16？ yes/no
  2. N = ___，totalTiles = ___，blockDim = ___
  3. tileN = ___，tileNA = ___，repeats = ___，是否 ≤255？ yes/no
  4. UB 预算 = tileNA*(16*C+32) = ___，是否 ≤ ubSize？ yes/no
  5. offset table 是否按 tileNA 构建？ yes/no
  6. 尾块是否用 curN 区分有效长度？ yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 统一建模 [C,N] 存在
(grep -cE "C\s*=|channelDim|N\s*=.*totalElements" modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 1

# 检查 2: tileNA / repeats 计算
(grep -cE "tileNA|repeats" modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 2

# 检查 3: TransDataTo5HD / vnchwconv / Gather 出现
(grep -cE "TransDataTo5HD|vnchwconv|Gather" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 2

# 检查 4: offset table 构建
(grep -cE "offsetBuff|SetValue.*p \* C \+ c" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 5: tileN 对齐 32
(grep -cE "AlignUp\(.*tileN.*32\)|tileN.*AlignUp.*32" modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 1
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P114 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: form={alpha|beta|gamma}, transpose tiling applied
Step 4: constraints checked (C=___, tileNA=___, repeats=___)
Step 5: all 5 grep checks passed
```

## 常见陷阱

- C 轴与 N 轴展平顺序错误 → layout 错误。
- `repeats` 超过 255 → 编译失败。
- offset table 按 tileN 而非 tileNA 构建 → Gather 错位。
- 尾块未区分 curN 与 tileNA → 越界。
- 未把后处理融合进同一 pipeline → 额外 GM 往返。
