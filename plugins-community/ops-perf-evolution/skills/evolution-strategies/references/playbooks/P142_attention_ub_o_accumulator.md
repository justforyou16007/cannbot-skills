# P142 Playbook: Attention UB 常驻 O 累加器

> 本 Playbook 为**强制流程**。采纳 P142 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P142 的核心是**在 Flash Attention 中把 O_tile 常驻 UB，跨 KV tile 流式更新，消除 per-tile GM 往返**。

## Step 1: 定位关键结构

```bash
grep -nE "O_.*GM|O_Gm|WriteO|UpdateO|RescaleO|DataCopy.*O" \
    shared/original/op_kernel/*.cpp > /tmp/p142_locations.txt
grep -nE "for.*kv|kvTile|kvN|nloops|seq" \
    shared/original/op_kernel/*.cpp >> /tmp/p142_locations.txt
grep -nE "AllocTensor\|FreeTensor\|LocalTensor.*O\|O_tile" \
    shared/original/op_kernel/*.cpp >> /tmp/p142_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- O 更新位置：文件 + 行号
- KV tile 循环结构：文件 + 行号
- 当前 O 搬运模式：每次 tile 读写 GM / 已有 workspace ring / UB 常驻
- UB buffer 分配位置：文件 + 行号

## Step 2: 改造计划表（强制填写）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| O 位置 | `?` (GM / workspace / UB) | UB 常驻 | `op_kernel/*.cpp:L?` |
| M_tile × d | `?` bytes | ≤ UB 预算 20% | `op_kernel/*.cpp:L?` |
| 更新次数 | `?` (kvTiles) | 不变 | `op_kernel/*.cpp:L?` |
| 写回 GM | `?` (每 tile / 循环后) | 循环后一次 | `op_kernel/*.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造

### 3A. 形态识别

判断当前代码属于哪种形态：
- **形态 α — 已有 GM workspace ring（AIC↔AIV 经 GM）**：需先评估是否改为 UB 常驻，或保留 ring 但用 X38 放大 kvN。
- **形态 β — O 每 tile 写回 GM 再由 AIV/自己读回**：直接改为 UB 常驻。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 β — 直接 GM 往返改 UB 常驻）

```cpp
// === 改造前 ===
for (int kv = 0; kv < kvTiles; ++kv) {
    ComputePTile(...);
    LocalTensor<float> O_old = LoadFromGM(O_Gm);   // MTE2/MTE3
    LocalTensor<float> O_new = AllocTmp(...);
    O_new = Rescale(O_old) + P_tile @ V_tile;
    WriteToGM(O_Gm, O_new);
    FreeTmp(O_new);
}

// === 改造后 ===
LocalTensor<float> O_ub = ubQueue.template AllocTensor<float>(M_tile * d);
SetZero(O_ub);
for (int kv = 0; kv < kvTiles; ++kv) {
    ComputePTile(...);
    // L0C 算 PV，结果 rescale 后加到 UB O
    OUpdateInUB(O_ub, P_tile, V_tile, m_old, m_new, l_old, l_new);
}
DataCopy(O_Gm, O_ub, M_tile * d);
ubQueue.FreeTensor(O_ub);
```

### 3C. Variant Notes

- **与 X38 的边界**：若当前已用 X38（GM workspace ring + kvN 放大），UB 常驻是替代路径，不要叠加；选择依据是 O_tile 是否能完整放进 UB。
- **容量公式**：`M_tile * d * sizeof(fp32) <= UB_budget * 0.2`，否则拆小 M_tile 或 fallback。
- **精度**：rescaling 在 UB 中保持 fp32；最终写回前 cast 到输出 dtype。

## Step 4: 约束复核

```
约束 1: M_tile * d * 4B ≤ UB 容量的 20%
约束 2: O_ub 在 KV 循环外 Alloc、循环后 Free
约束 3: KV 循环内对 O_ub 的 rescale+add 必须同步完成再进入下一轮
约束 4: 循环结束后必须有一次 DataCopy 将 O_ub 写回 GM
约束 5: 不能同时保留 per-tile GM 写回路径（二选一）
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `M_tile = ?, d = ?, O_tile_size = ? bytes`
- `UB 容量 = ? bytes`, `占用比例 = ?%`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 循环外 AllocTensor O_ub
grep -cE "AllocTensor.*O|LocalTensor.*O_ub" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 循环后 FreeTensor O_ub
grep -cE "FreeTensor.*O" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: 循环内有 O 更新（无 GM 搬运）
grep -cE "for.*UpdateO\|for.*RescaleO" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 循环内不写 O_Gm
grep -cE "for.*DataCopy.*O_Gm\|for.*WriteToGM.*O" modified_files/op_kernel/*.cpp
# 期望: == 0

# 检查 5: 循环结束后写回 O_Gm
grep -cE "DataCopy.*O_Gm" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| UB 溢出 | 减小 M_tile 或 fallback 到 GM ring |
| 精度下降 | rescale 保持 fp32，最终写回前再 cast |
| 与 X38 叠加导致冲突 | 二选一，不要同时保留两条路径 |
| O_ub 在循环内被覆盖 | 确保 rescale+add 完成后再进下一轮 |
| 循环结束后未写回 GM | 必须有一次 DataCopy O_ub → O_Gm |

---

**完成清单**：
```
[P142 Playbook Completion]
Step 1: done (/tmp/p142_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: M_tile=? d=? O_tile_size=? UB=? ratio=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
