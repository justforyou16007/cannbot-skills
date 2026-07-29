# P106 Playbook: Matmul Batch 合并到 M

> **强制流程**。采纳 P106 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
> 核心目标：在 `batchB == 1 && !transA` 的 BatchMatMul 中，把 A 的 batch 维度折叠进 M，提升单核任务粒度并简化多核扩展。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p106_locations.txt`：

```bash
# 1.1 Batch 维度读取
find shared/original/op_host -type f \( -name "*.cpp" -o -name "*.h" \) | xargs \
    grep -nE "batchA|batchB|batchC|BatchMatMul|MultiBatchInfo" > /tmp/p106_locations.txt

# 1.2 transA / transB 标志
find shared/original/op_host -type f \( -name "*.cpp" -o -name "*.h" \) | xargs \
    grep -nE "transA|transB|isTransA|isTransB" >> /tmp/p106_locations.txt

# 1.3 M 维度提取
find shared/original/op_host -type f \( -name "*.cpp" -o -name "*.h" \) | xargs \
    grep -nE "\bM\b.*=.*shape|M\s*=\s*aShape|GetShapeSize" >> /tmp/p106_locations.txt

# 1.4 Kernel 端 batch 遍历
find shared/original/op_kernel -type f \( -name "*.cpp" -o -name "*.h" \) | xargs \
    grep -nE "IterateBatch|batchUsedCoreNum|iterBatch|batchTileBlock" >> /tmp/p106_locations.txt
```

**交付物**（记录到 `implementation_note.txt` "Playbook Step 1"）：
- `batchA / batchB / batchC` 读取位置
- `transA / transB` 取值位置
- `M` 维度计算位置
- `IterateBatch` 或 `MultiBatchInfo` 使用位置

## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| batchA | `≥ 1` | 合并后 `1`（条件满足时） | `op_host/*_tiling.cpp:L?` |
| M | `M` | `batchA * M` | `op_host/*_tiling.cpp:L?` |
| batchC | 按广播 | 重新校验 | `op_host/*_tiling.cpp:L?` |
| MultiBatchInfo | 使用 | 可能简化 | `op_host/*_tiling.cpp:L?` |

**排除规则**（下列情况不改）：
- `batchB != 1`
- `transA == true`
- `batchA * M` 超过 `int32_t` 表达范围
- 算子依赖 per-batch 的 bias/scale 且无法广播

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 简单 BatchMatMul**：仅 A 有 batch，B 无 batch，直接合并。
- **形态 β — 带 Batch 的 Bias/Scale**：合并后需确认 bias/scale 的 batch 维度可广播。
- **形态 γ — 多维 Broadcast**：A batch = N，B batch = M，不能直接合并到 M。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template

```cpp
// 读取原始 batch 与 M
int64_t batchA = aShape.size() > 2 ? aShape[aShape.size() - 3] : 1;
int64_t batchB = bShape.size() > 2 ? bShape[bShape.size() - 3] : 1;
int64_t M      = aShape[aShape.size() - 2];
bool transA    = /* 从 op attr 读取 */;

// Batch 合并到 M
if (batchB == 1 && !transA) {
    M = batchA * M;
    batchA = 1;
}

// 重新计算 batchC（按广播规则）
int64_t batchC = /* broadcast(batchA, batchB) */;
```

### 3C. Variant Notes

- 合并后 `TilingHeader` 中 `cubeTiling.M` 应写合并后的 `M`。
- 若 Kernel 端原本使用 `IterateBatch`，合并后 batchC 可能为 1，可删除外层 batch 循环。
- 带 bias 时：bias shape 通常为 `[1, N]` 或 `[batchC, N]`；合并后仍可按 broadcast 处理，但需显式校验。

## Step 4: 约束复核

```
P106 改造约束核查：
  1. 仅在 batchB == 1 && !transA 时合并: yes/no
  2. 合并后的 M 用 int64_t 计算，未溢出: yes/no
  3. batchC 已按新 batchA 重新计算: yes/no
  4. cubeTiling.M 写入的是合并后的 M: yes/no
  5. Kernel 端无遗留 IterateBatch 期望原 batchA: yes/no
  6. Bias/Scale 的 batch 广播语义仍成立: yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 合并条件存在
MG=$(grep -cE "batchB\s*==\s*1\s*&&\s*!\s*transA|batchB\s*==\s*1\s*&&\s*transA\s*==\s*false" \
    modified_files/op_host/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "Batch merge guard: $MG"
# 期望：>= 1

# 检查 2: M = batchA * M 出现
MM=$(grep -cE "M\s*=\s*batchA\s*\*\s*M|M\s*=\s*M\s*\*\s*batchA" \
    modified_files/op_host/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "M = batchA * M assignment: $MM"
# 期望：>= 1

# 检查 3: batchA 重置为 1
BA=$(grep -cE "batchA\s*=\s*1\s*;" \
    modified_files/op_host/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "batchA reset to 1: $BA"
# 期望：>= 1

# 检查 4: 合并后的 M 写入 cubeTiling
CM=$(grep -cE "cubeTiling\.M\s*=\s*M" \
    modified_files/op_host/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "cubeTiling.M assignment: $CM"
# 期望：>= 1

# 检查 5: batchC 重新计算
BC=$(grep -cE "batchC\s*=|broadcast.*batch" \
    modified_files/op_host/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "batchC (re)compute: $BC"
# 期望：>= 1
```

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P106 Playbook Completion]
Step 1: done, /tmp/p106_locations.txt written
Step 2: plan table filled (batchA: __ → 1, M: __ → __)
Step 3: form = alpha/beta/gamma, batch merge applied
Step 4: all 6 constraints passed (yes/no list above)
Step 5: all 5 grep checks passed (MG=__, MM=__, BA=__, CM=__, BC=__)
```
