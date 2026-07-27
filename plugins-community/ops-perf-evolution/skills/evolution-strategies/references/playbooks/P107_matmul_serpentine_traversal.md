# P107 Playbook: Matmul Serpentine 遍历

> **强制流程**。采纳 P107 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
> 核心目标：在 L2 级 Tiling 的 N 方向 tile 遍历中，相邻 M 行交替反向，复用上一行末端缓存的 B 行，降低 MTE2 读 B 次数。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p107_locations.txt`：

```bash
# 1.1 L2 tile 遍历循环
find shared/original/op_kernel -type f \( -name "*.cpp" -o -name "*.h" \) | xargs \
    grep -nE "mTileCntL2|nTileCntL2|for\s*\(\s*uint64_t\s+mTile|for\s*\(\s*uint64_t\s+nTile" > /tmp/p107_locations.txt

# 1.2 ProcessL2Tile / L2 tile 处理函数
find shared/original/op_kernel -type f \( -name "*.cpp" -o -name "*.h" \) | xargs \
    grep -nE "ProcessL2Tile|ProcessTile|L2Tile|mTile\s*,\s*nTile" >> /tmp/p107_locations.txt

# 1.3 已有的 reverse / serpentine 模式
find shared/original/op_kernel -type f \( -name "*.cpp" -o -name "*.h" \) | xargs \
    grep -nE "reverse|serpentine" >> /tmp/p107_locations.txt

# 1.4 Host 侧 L2 参数
find shared/original/op_host -type f \( -name "*.cpp" -o -name "*.h" \) | xargs \
    grep -nE "mTileCntL2|nTileCntL2|calOrder" >> /tmp/p107_locations.txt
```

**交付物**（记录到 `implementation_note.txt` "Playbook Step 1"）：
- L2 tile 双层循环位置（文件 + 行号）
- `ProcessL2Tile(mTile, nTile)` 调用位置
- 当前是否已有 reverse/serpentine
- Host 侧 `nTileCntL2` 计算位置

## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| N 方向遍历 | 正向 0..nTileCntL2-1 | 偶数行正向、奇数行反向 | `op_kernel/*.cpp:L?` |
| reverse 标志 | 无 | 每切换一行 M 取反 | `op_kernel/*.cpp:L?` |
| nTile 计算 | `nTile = t` | `nTile = reverse ? (nTileCntL2 - t - 1) : t` | `op_kernel/*.cpp:L?` |

**排除规则**（下列情况不改）：
- `nTileCntL2 == 1`（只有 1 列 tile，反向无意义）
- `ProcessL2Tile` 内部有跨 tile 顺序依赖（如非可交换归约）
- 已存在 serpentine 遍历

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — Kernel 端新增 serpentine**：Host 已传 `mTileCntL2 / nTileCntL2`，Kernel 端改写遍历。
- **形态 β — Host 端遍历下发**：若 Host 负责 tile 下发，则 serpentine 放在 Host 循环中。
- **形态 γ — 与 P105 同时引入**：先完成 P105，再在相同位置叠加 serpentine。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（Kernel 端）

```cpp
// === 改造前：正向双层循环 ===
for (uint64_t mTile = 0; mTile < mTileCntL2; ++mTile) {
    for (uint64_t nTile = 0; nTile < nTileCntL2; ++nTile) {
        ProcessL2Tile(mTile, nTile);
    }
}

// === 改造后：Serpentine 反向遍历 ===
bool reverse = true;
for (uint64_t mTile = 0; mTile < mTileCntL2; ++mTile) {
    reverse = !reverse;
    for (uint64_t t = 0; t < nTileCntL2; ++t) {
        uint64_t nTile = reverse ? (nTileCntL2 - t - 1) : t;
        ProcessL2Tile(mTile, nTile);
    }
}
```

### 3C. Variant Notes

- 若 `nTileCntL2` 在编译期可能为 1，保留退化分支：`if (nTileCntL2 > 1) { serpentine } else { forward }`。
- `reverse` 初始值可任意取 `true` 或 `false`，只需保证相邻行方向相反。
- 与 Atomic Add 顺序敏感场景冲突时，放弃 P107。

## Step 4: 约束复核

```
P107 改造约束核查：
  1. nTileCntL2 > 1 时才启用 serpentine: yes/no
  2. 相邻 M 行方向严格相反: yes/no
  3. ProcessL2Tile 内部无跨 tile 顺序依赖: yes/no
  4. calOrder 与 serpentine 方向不冲突: yes/no
  5. 当 nTileCntL2 == 1 时退化为正向循环: yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: reverse 标志声明
RV=$(grep -cE "bool\s+reverse\s*=" \
    modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "reverse flag declaration: $RV"
# 期望：>= 1

# 检查 2: reverse 在每行 M 切换时取反
TF=$(grep -cE "reverse\s*=\s*!\s*reverse|reverse\s*=\s*not\s+reverse" \
    modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "reverse toggle: $TF"
# 期望：>= 1

# 检查 3: nTile 条件反向赋值
NT=$(grep -cE "nTile\s*=\s*reverse\s*\?\s*\(\s*nTileCntL2\s*-\s*t\s*-\s*1\s*\)\s*:\s*t" \
    modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "Conditional nTile reverse: $NT"
# 期望：>= 1

# 检查 4: ProcessL2Tile 仍被调用
PT=$(grep -cE "ProcessL2Tile\s*\(\s*mTile\s*,\s*nTile\s*\)" \
    modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "ProcessL2Tile call preserved: $PT"
# 期望：>= 1

# 检查 5: nTileCntL2 > 1 守卫
NG=$(grep -cE "nTileCntL2\s*>\s*1" \
    modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "nTileCntL2 > 1 guard: $NG"
# 期望：>= 1
```

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P107 Playbook Completion]
Step 1: done, /tmp/p107_locations.txt written
Step 2: plan table filled (nTileCntL2: __, reverse initial: __)
Step 3: form = alpha/beta/gamma, serpentine applied
Step 4: all 5 constraints passed (yes/no list above)
Step 5: all 5 grep checks passed (RV=__, TF=__, NT=__, PT=__, NG=__)
```
