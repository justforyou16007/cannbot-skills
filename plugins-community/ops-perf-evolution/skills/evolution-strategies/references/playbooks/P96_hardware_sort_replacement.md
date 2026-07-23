# P96 Playbook: 硬件排序引擎替代标量排序/选择

> **强制流程**。P96 把 kernel 中的标量排序（bubble sort、selection sort）或 top-k 选择算法替换为 AscendC 硬件 `Sort32`/`MrgSort`/`Extract` API。算法复杂度从 O(M²) 或 O(n×k) 降至 O(n log² n)。**必须**按本 Playbook 逐步执行。

## Step 1: 定位标量排序/选择反模式

执行下面的 grep，把结果写入 `/tmp/p96_locations.txt`：

```bash
# 1.1 嵌套循环比较交换（冒泡/选择排序特征）
grep -nE "for.*for.*(swap|SWAP|compare|Compare)" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null > /tmp/p96_locations.txt

# 1.2 selection-sort topK
grep -nE "for.*for.*max_val|for.*for.*maxVal|selection.*sort" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p96_locations.txt

# 1.3 逐元素 SetValue/GetValue 选择（TopK 分值/索引更新）
grep -nE "for\s*\([^)]*\+\+[^)]*\).*SetValue.*GetValue" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p96_locations.txt

# 1.4 已有 Sort API 使用（避免重复应用）
grep -nE "AscendC::Sort|Sort32|MrgSort|CreateVecIndex|Extract" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p96_locations.txt

# 1.5 排序维度 / topK 参数
grep -nE "rowLength|topK|k_value|sort.*size|M" \
    shared/original/op_host/*_tiling.cpp shared/original/op_kernel/*.cpp 2>/dev/null >> /tmp/p96_locations.txt
```

**交付物**（`implementation_note.txt` "Playbook Step 1"）：
- **排序反模式位置**：嵌套循环 swap / selection topK 的文件 + 行号
- **SetValue/GetValue 选择位置**：逐元素更新 topK 的位置
- **已有 Sort API**：是否已使用 Sort32/MrgSort
- **排序维度**：rowLength / k 的范围

## Step 2: 排序改造计划表（强制）

**不填完此表不得进入 Step 3**：

| 调用点 (file:line) | 反模式 | 数据规模 | 目标方案 | buffer 需求 | 预计收益 |
|---|---|---|---|---|---|
| kernel.cpp:145 | bubble sort | M≤32 | `Sort32(repeat=1)` | 2×data + idx + tmp | 5-20x |
| kernel.cpp:234 | selection topK | n≤4096, k≤64 | `Sort + Extract` | 同上 | 50-200x |
| ... | ... | ... | ... | ... | ... |

**排除规则**（下列情况**不改**）：
- **排序维度固定 ≤ 8 且次数极少**：标量开销占比小，改造收益低
- **已有 Sort32/MrgSort**：已经是硬件排序路径
- **需要稳定排序且 k 极小**：若仅需 top-1，可考虑 `ReduceMax` + 索引扫描（P55/P94）
- **数据类型不支持 Sort32**：确认 AscendC Sort API 支持的 dtype

## Step 3: 改造实施

### 3A. 形态识别

- **形态 α — 小行批量排序（rowLength ≤ 32/64）**：多行打包为 `Sort32(repeat=B)`
- **形态 β — 大行 Tiled Sort + MrgSort**：先 tile 内 Sort，再跨 tile MrgSort
- **形态 γ — 降序排序**：Negate → Sort → Negate（如 NMS）
- **形态 δ — topK 选择**：Sort 后 `Extract` 前 K 个

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma | delta`

### 3B. Canonical Template（形态 α — 批量小行排序）

```cpp
// === 改造前：标量 bubble sort ===
for (int32_t i = 0; i < M; ++i)
    for (int32_t j = 0; j < M - 1; ++j)
        if (scores[j] < scores[j+1]) { swap(scores[j], scores[j+1]); }

// === 改造后：Sort32(repeat=B) 批量并行 ===
constexpr int32_t BATCH_ROWS = 32;
for (int32_t r = 0; r < totalRows; r += BATCH_ROWS) {
    int32_t curBatch = min(BATCH_ROWS, totalRows - r);
    // packedBuf 已按 [batch, rowLen] 排布
    AscendC::Sort<T, true>(packedBuf, packedBuf, orderBuf, tmpBuf,
                           curBatch * rowLen / 64);
}
```

**关键替换要点**：
1. `repeat = totalElements / 64`（64 是 Sort32 硬件数据块大小）
2. `true` = 同步输出排序后索引到 `orderBuf`
3. Padding 元素用 +∞（升序沉底）或 -∞ 填充，防止干扰有效元素
4. `orderBuf` 类型为 `uint32_t`，元素数与数据同等

### 3C. Variant Notes

**形态 γ — 降序排序**：
```cpp
for (int32_t k = 0; k < M; k++) svScore(k) = -svScore(k);   // Negate
for (int32_t k = M; k < SORT_SZ; k++) svScore(k) = 1e38f;    // Pad
AscendC::Sort<float, true>(svScore, svScore, orderU32, tmpF, SORT_SZ / 64);
for (int32_t k = 0; k < M; k++) svScore(k) = -svScore(k);   // Restore
```

**形态 β — 大行 Tiled Sort + MrgSort**：
```cpp
// 每个 tile 内部 Sort
AscendC::Sort<T, true>(tileVals, tileVals, tileIdx, tmpF, tileCap / 64);
// Extract topK from sorted result
// MrgSort 合并跨 tile 的 topK 结果
```

**与其他策略协同**：
- + **P1 双缓冲**：Sort 前后的大块搬运可重叠
- + **P7 数据对齐**：确保 Sort buffer 256B/512B 对齐
- + **P67 Counter 模式**：Sort 后的向量处理可用 Counter 模式

## Step 4: Buffer + 对齐约束复核

```
P96 改造约束核查：
  1. Sort32 repeat 整除: totalElements % 64 == 0 (pad 后) yes/no
  2. Padding 值: +∞ (升序) / -∞ (降序 after negate) yes/no
  3. orderBuf 类型: uint32_t yes/no
  4. tmpBuf 大小: >= totalElements * sizeof(T) yes/no
  5. 数据 buffer 双份: src + dst yes/no
  6. UB 总增量: ___ bytes <= 预算 yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 嵌套排序循环减少
BEFORE=$(grep -cE "for\s*\([^)]*\+\+[^)]*\)\s*for" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
AFTER=$(grep -cE "for\s*\([^)]*\+\+[^)]*\)\s*for" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "nested loops: before=$BEFORE after=$AFTER"
# 期望：AFTER < BEFORE

# 检查 2: Sort/MrgSort/CreateVecIndex 使用增加
SORT_B=$(grep -cE "\b(AscendC::Sort|Sort32|MrgSort|CreateVecIndex)\b" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
SORT_A=$(grep -cE "\b(AscendC::Sort|Sort32|MrgSort|CreateVecIndex)\b" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "Sort APIs: before=$SORT_B after=$SORT_A"
# 期望：SORT_A > SORT_B

# 检查 3: selection topK 反模式减少
SEL_B=$(grep -cE "for.*maxVal|for.*max_val|selection.*sort" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
SEL_A=$(grep -cE "for.*maxVal|for.*max_val|selection.*sort" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "selection topK: before=$SEL_B after=$SEL_A"
# 期望：SEL_A < SEL_B

# 检查 4: orderBuf uint32_t 声明
ORDER=$(grep -cE "LocalTensor\s*<\s*uint32_t\s*>.*order|orderBuf.*uint32_t" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "orderBuf uint32_t: $ORDER"
# 期望：>= 1

# 检查 5: Pad 值正确（升序 +∞）
grep -nE "1e38f|1e30f|INFINITY|1e37f" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | head -5
# agent 人工确认 padding 值不会把有效元素挤出 top 位置
```

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| Sort32 输出顺序错误 | `repeatTimes = totalElements / 64` 计算错；或 padding 值把有效元素挤走 |
| 编译失败：Sort 模板参数 | `Sort<T, true>` 的 T 必须是支持的 dtype；`orderBuf` 必须是 uint32_t |
| 降序排序结果反了 | 必须用 Negate → Sort → Negate；不能直接用 `>` 比较器 |
| topK 索引指向错误位置 | `CreateVecIndex` 只应创建一次，不要在每 tile 重复创建 |
| 性能不如预期 | 检查是否满足批量条件（rowLength ≤ 32/64）；单行 Sort32 repeat=1 指令开销大 |

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P96 Playbook Completion]
Step 1: done, N sort/topK anti-patterns identified
Step 2: refactor plan filled (M sites, form={alpha|beta|gamma|delta})
Step 3: Sort32/MrgSort applied, repeat=totalElements/64, pad=+inf
Step 4: alignment=___, orderBuf=uint32_t, UB delta=___ ✓
Step 5: all 5 grep checks passed (nested loops: X→Y, Sort APIs: X→Y)
Step 6: no pitfalls / {列出触发的}
```
