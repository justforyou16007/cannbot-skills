# P105 Playbook: Matmul L2 Cache Tile Sizing

> **强制流程**。采纳 P105 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
> 核心目标：为大矩阵乘增加 L2 级 Tiling，使每个 L2 tile 的 base 块工作集不超过 L2 容量的一半，并通过 `calOrder` 选择 A 行 / B 列优先复用方向。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p105_locations.txt`：

```bash
# 1.1 TilingHeader / L2cacheTilePara 定义
find shared/original/op_host -type f \( -name "*.cpp" -o -name "*.h" \) | xargs \
    grep -nE "L2cacheTilePara|TilingHeader|TCubeTiling|mTileBlock|nTileBlock|mTileCntL2|nTileCntL2" > /tmp/p105_locations.txt

# 1.2 Host 侧 Tiling 计算主体
find shared/original/op_host -type f \( -name "*.cpp" -o -name "*.h" \) | xargs \
    grep -nE "GetTiling|mTotalCnt|nTotalCnt|totalBlock|calOrder" >> /tmp/p105_locations.txt

# 1.3 Kernel 端 L2 tile 遍历或 ProcessL2Tile
find shared/original/op_kernel -type f \( -name "*.cpp" -o -name "*.h" \) | xargs \
    grep -nE "mTileCntL2|nTileCntL2|ProcessL2Tile|IterateAll|IterateBatch" >> /tmp/p105_locations.txt
```

**交付物**（记录到 `implementation_note.txt` "Playbook Step 1"）：
- `TilingHeader / L2cacheTilePara` 定义位置
- `GetTiling` 后派生计数 `mTotalCnt / nTotalCnt` 位置
- 当前是否有 L2 tile 遍历结构（无则 Step 3 新增）

## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| L2 tile 结构 | 无 / 未启用 | 新增/填充 `L2cacheTilePara` | `op_host/*_tiling.cpp:L?` |
| mTileCntL2 / nTileCntL2 | 1 | 由 footprint 决定 | `op_host/*_tiling.cpp:L?` |
| mTileBlock / nTileBlock | 1 | ceil(mTotalCnt, mTileCntL2) | `op_host/*_tiling.cpp:L?` |
| calOrder | 默认 | ROW_FIRST if M≥N else COL_FIRST | `op_host/*_tiling.cpp:L?` |
| 小算子退化 | 无 | `M*N*dtypeSize ≤ L2/2` 时跳过 | `op_host/*_tiling.cpp:L?` |

**排除规则**（下列情况不改）：
- `M * N * dtypeSize ≤ L2 / 2`（小算子，L2 切分无收益）
- 算子无 L2cacheTilePara 字段且不能修改 TilingHeader ABI
- 已存在手动 L2 tiling 且经评估效果良好

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — Host 新增完整 L2 计算**：从无到有创建 `L2cacheTilePara` 并填充。
- **形态 β — 已有 L2 字段但退化或未优化**：调整 `mTileBlock / nTileBlock` 与 `calOrder`。
- **形态 γ — Kernel 端遍历顺序优化**：Host 已计算 L2 tile，Kernel 端改为 serpentine（见 P107）。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template

```cpp
// 获取平台 L2 大小（示例值，实际应查 platform 或 CANN 宏）
constexpr uint32_t L2_SIZE = 2 * 1024 * 1024;
L2cacheTilePara l2Para{};

// 小算子退化
bool needL2Tiling = (M * N * dtypeSize > L2_SIZE / 2);
if (needL2Tiling) {
    // 方式 1：直接按目标 tile 数拆分（需保证 footprint ≤ L2/2）
    uint32_t targetTileCnt = std::max(1U, static_cast<uint32_t>(
        (M * N * dtypeSize) / (L2_SIZE / 2)));
    l2Para.mTileCntL2 = static_cast<uint32_t>(std::ceil(std::sqrt(targetTileCnt)));
    l2Para.nTileCntL2 = static_cast<uint32_t>(
        std::ceil(static_cast<double>(targetTileCnt) / l2Para.mTileCntL2));

    l2Para.mTileBlock = ceil(mTotalCnt, l2Para.mTileCntL2);
    l2Para.nTileBlock = ceil(nTotalCnt, l2Para.nTileCntL2);
    l2Para.calOrder   = (M >= N) ? ROW_FIRST : COL_FIRST;
} else {
    l2Para.mTileCntL2 = 1;
    l2Para.nTileCntL2 = 1;
    l2Para.mTileBlock = mTotalCnt;
    l2Para.nTileBlock = nTotalCnt;
    l2Para.calOrder   = ROW_FIRST;
}
```

### 3C. Variant Notes

-  footprint 计算可简化为 `l2Para.mTileBlock * singleCoreM * l2Para.nTileBlock * singleCoreN * dtypeSize * 2`（A+B 两张），应 `≤ L2 / 2`。
- `calOrder` 与 P107 serpentine 正交；ROW_FIRST 下 serpentine 复用 B 行收益最大。
- 若 P104 多核扩展也在使用，`totalTileCnt` 应使用 L2 tile 内的 base 块数 `mCnt * nCnt`。

## Step 4: 约束复核

```
P105 改造约束核查：
  1. 小算子已退化（M*N*dtypeSize ≤ L2/2 时 mTileCntL2=nTileCntL2=1）: yes/no
  2. L2 tile footprint ≤ L2/2: yes/no
  3. mTileBlock / nTileBlock 是 base 块整数倍: yes/no
  4. calOrder 与 M/N 比例一致: yes/no
  5. L2 参数已序列化到 device tensor: yes/no
  6. Kernel 端读取了 l2Para 并按 mTileCntL2/nTileCntL2 遍历: yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: L2cacheTilePara 已使用
L2=$(grep -cE "L2cacheTilePara\s+\w+|L2cacheTilePara\s*{" \
    modified_files/op_host/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "L2cacheTilePara usage: $L2"
# 期望：>= 1

# 检查 2: mTileCntL2 / nTileCntL2 被赋值
TC=$(grep -cE "mTileCntL2\s*=|nTileCntL2\s*=" \
    modified_files/op_host/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "L2 tile count assignments: $TC"
# 期望：>= 2

# 检查 3: mTileBlock / nTileBlock 被赋值
TB=$(grep -cE "mTileBlock\s*=|nTileBlock\s*=" \
    modified_files/op_host/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "L2 tile block assignments: $TB"
# 期望：>= 2

# 检查 4: calOrder 被设置
CO=$(grep -cE "calOrder\s*=\s*(ROW_FIRST|COL_FIRST)" \
    modified_files/op_host/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "calOrder assignment: $CO"
# 期望：>= 1

# 检查 5: 小算子退化分支存在
DG=$(grep -cE "L2_SIZE\s*/\s*2|M\s*\*\s*N\s*\*\s*dtypeSize\s*<\s*=" \
    modified_files/op_host/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "Small-op degenerate guard: $DG"
# 期望：>= 1
```

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P105 Playbook Completion]
Step 1: done, /tmp/p105_locations.txt written
Step 2: plan table filled (mTileCntL2: __, nTileCntL2: __, calOrder: __)
Step 3: form = alpha/beta/gamma, L2 tiling applied
Step 4: all 6 constraints passed (yes/no list above)
Step 5: all 5 grep checks passed (L2=__, TC=__, TB=__, CO=__, DG=__)
```
