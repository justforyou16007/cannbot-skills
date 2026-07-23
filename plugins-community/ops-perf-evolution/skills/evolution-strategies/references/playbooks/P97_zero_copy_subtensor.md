# P97 Playbook: 零拷贝子张量切片 + 批量向量运算

> **强制流程**。P97 用 `LocalTensor<T> view = tensor[offset]` 零拷贝切片替代逐元素标量 gather 或 gather+pad 到连续 buffer 的做法，然后对视图执行批量 Vector 运算。零拷贝只创建视图元数据（偏移量 + 长度），不触发任何数据搬运。**必须**按本 Playbook 逐步执行。

## Step 1: 定位可切片替代的反模式

执行下面的 grep，把结果写入 `/tmp/p97_locations.txt`：

```bash
# 1.1 从大 tensor 逐元素取值
grep -nE "for\s*\([^)]*j[^)]*\+\+[^)]*\).*\[.*\].*=" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null > /tmp/p97_locations.txt

# 1.2 gather/copy 到连续 buffer 的模式
grep -nE "for.*gather|for.*collect|for.*copy.*\[.*\+" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p97_locations.txt

# 1.3 已有 SubTensor / 切片视图使用
grep -nE "SubTensor|LocalTensor.*=.*\[.*\]" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p97_locations.txt

# 1.4 批量向量运算机会（Max/Min/SubRelu/Mul/Div 链）
grep -nE "\b(Max|Min|SubRelu|Mul|Div|Duplicate)\s*\(" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p97_locations.txt
```

**交付物**（`implementation_note.txt` "Playbook Step 1"）：
- **逐元素 gather 位置**：`for + svX[j]` 类代码的文件 + 行号
- **连续 buffer gather 位置**：需要拷贝到临时 buffer 再做向量运算的位置
- **已有切片视图**：是否已使用 `SubTensor` 或 `tensor[offset]`
- **批量向量运算机会**：Max/Min/SubRelu/Mul/Div 链的位置

## Step 2: 零拷贝改造计划表（强制）

| 调用点 (file:line) | 当前反模式 | 源 tensor | 偏移/范围 | 批量运算 | batch 大小 | 预计收益 |
|---|---|---|---|---|---|---|
| kernel.cpp:145 | for svX1[j] | svX1 | [bStart, bStart+bCnt) | IoU vector chain | 32 | 3-5x |
| ... | ... | ... | ... | ... | ... | ... |

**排除规则**（下列情况**不改**）：
- **源 tensor 不是 UB LocalTensor**：零拷贝切片只适用于 UB 内 tensor
- **切片后数据不连续或 stride 不规则**：零拷贝视图要求底层数据连续
- **batch 大小 < 4**：批量向量运算收益不明显
- **Compare API 可直接使用且对齐满足**：若 `count*sizeof(T) % 256B == 0`，可用 Compare 替代标量比较

## Step 3: 改造实施

### 3A. 形态识别

- **形态 α — 子张量 + 向量算术链**：如 NMS IoU 计算
- **形态 β — 子张量 + 归约**：如按行 ReduceSum
- **形态 γ — 多子张量拼接**：如 GatherMask 前准备数据

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 α — NMS IoU 式向量链）

```cpp
// === 改造前：逐元素标量 gather ===
for (int32_t j = 0; j < numCandidates; ++j) {
    float cx1 = svX1[j];
    float cy1 = svY1[j];
    float cx2 = svX2[j];
    float cy2 = svY2[j];
    // ... 标量 IoU ...
}

// === 改造后：零拷贝切片 + 批量向量运算 ===
LocalTensor<float> candX1 = svX1[bStart];
LocalTensor<float> candY1 = svY1[bStart];
LocalTensor<float> candX2 = svX2[bStart];
LocalTensor<float> candY2 = svY2[bStart];

AscendC::Duplicate<float>(w0, curX1, bCnt);
AscendC::Max<float>(w2, w0, candX1, bCnt);
AscendC::Min<float>(w0, curX2, candX2, bCnt);
AscendC::SubRelu<float>(w0, w0, w2, bCnt);
// ... 后续向量链 ...
```

**关键替换要点**：
1. `LocalTensor<T> view = tensor[offset]` 创建视图，`offset` 为元素偏移（非字节偏移）
2. 视图长度 = `原长度 - offset`；如需限制长度用 `tensor[offset].template SubTensor<0, len>()`
3. 批量大小 `bCnt = min(IOU_BATCH, remaining)`，受 UB 容量限制
4. Compare API 不满足对齐时回退标量 `GetValue(k)` 比较（小批量开销可忽略）

### 3C. Variant Notes

**形态 β — 子张量归约**：
```cpp
LocalTensor<float> rowView = xIn[rowOffset];
AscendC::ReduceSum<float, true>(scalarLocal, rowView, reduceTmp, rowLen);
```

**形态 γ — 多视图拼接后 GatherMask**：零拷贝视图（`evenSrc = src[0]` / `oddSrc = src[1]`）作为 GatherMask 的输入做固定间隔解交织。**GatherMask 参数与完整改造模板见 P100 Playbook Step 3B**，此处不重复。

**与其他策略协同**：
- + **P95 向量链**：切片后的批量运算用 Adds/Muls/Max/Min 等向量指令链
- + **P100 GatherMask**：按固定间隔取元素时，GatherMask 比零拷贝切片更高效
- + **P98 混合调度**：小 batch 走标量，大 batch 走向量

## Step 4: 约束复核

```
P97 改造约束核查：
  1. 源 tensor 是 UB LocalTensor: yes/no
  2. 切片底层数据连续: yes/no
  3. offset 在有效范围内: yes/no
  4. bCnt * sizeof(T) * 中间 tensor数 <= UB 剩余: yes/no
  5. 视图生命周期 <= 原 tensor FreeTensor 前: yes/no
  6. 多个视图不重叠写入: yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 逐元素 gather 循环减少
BEFORE=$(grep -cE "for\s*\([^)]*j[^)]*\+\+[^)]*\).*sv.*\[.*\]" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
AFTER=$(grep -cE "for\s*\([^)]*j[^)]*\+\+[^)]*\).*sv.*\[.*\]" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "scalar gather loops: before=$BEFORE after=$AFTER"
# 期望：AFTER < BEFORE

# 检查 2: 切片视图使用增加
VIEW_B=$(grep -cE "LocalTensor\s*<.*>\s*\w+\s*=\s*\w+\s*\[" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
VIEW_A=$(grep -cE "LocalTensor\s*<.*>\s*\w+\s*=\s*\w+\s*\[" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "tensor[offset] views: before=$VIEW_B after=$VIEW_A"
# 期望：VIEW_A > VIEW_B

# 检查 3: 临时 gather buffer 拷贝循环减少
GATHER_B=$(grep -cE "for.*gather|for.*copy.*buf|collect.*for" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
GATHER_A=$(grep -cE "for.*gather|for.*copy.*buf|collect.*for" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "gather copy loops: before=$GATHER_B after=$GATHER_A"
# 期望：GATHER_A < GATHER_B

# 检查 4: 批量 Vector 运算增加
VEC_B=$(grep -cE "\b(Max|Min|SubRelu|Mul|Div|Duplicate)\s*\(" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
VEC_A=$(grep -cE "\b(Max|Min|SubRelu|Mul|Div|Duplicate)\s*\(" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "vector ops: before=$VEC_B after=$VEC_A"
# 期望：VEC_A >= VEC_B（至少不减少）

# 检查 5: 无越界视图（人工确认）
# 在 implementation_note.txt 列出每个 tensor[offset] 的 offset 来源和范围检查
```

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 视图数据错乱 | offset 计算错（应是元素偏移，不是字节偏移） |
| 运行时崩溃 | 原 tensor FreeTensor 后仍使用视图；确保视图生命周期在原 tensor 内 |
| 视图写入冲突 | 多个视图重叠写入同一 UB 区域；调整 batch 大小或 buffer 分配 |
| batch 过大 UB 溢出 | 减小 IOU_BATCH 或分批处理 |
| 小 batch 性能不如标量 | 走 P98 混合调度 |

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P97 Playbook Completion]
Step 1: done, N scalar gather / copy loops identified
Step 2: refactor plan filled (M sites, form={alpha|beta|gamma})
Step 3: zero-copy views applied, batch=bCnt, vector chain = ...
Step 4: source=UB_LocalTensor, continuity=___, offset valid=___, UB delta=___ ✓
Step 5: all 5 grep checks passed (scalar gather: X→Y, views: X→Y, gather copy: X→Y)
Step 6: no pitfalls / {列出触发的}
```
