# P104 Playbook: Matmul 多核切分扩展

> **强制流程**。采纳 P104 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
> 核心目标：当 `MatmulApiTiling::GetTiling` 返回 `totalBlock < aicCoreNum` 时，手动缩小 `singleCoreM / singleCoreN`，使所有 AIC 核都有任务。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p104_locations.txt`：

```bash
# 1.1 Host 侧 MatmulApiTiling 调用
find shared/original/op_host -name "*_tiling.cpp" -o -name "*_tiling.h" | xargs \
    grep -nE "MatmulApiTiling|GetTiling|TCubeTiling|TilingHeader" > /tmp/p104_locations.txt

# 1.2 singleCore / baseCore / usedCoreNum 字段
find shared/original/op_host -name "*_tiling.cpp" -o -name "*_tiling.h" | xargs \
    grep -nE "singleCoreM|singleCoreN|baseM|baseN|usedCoreNum|GetCoreNumAic" >> /tmp/p104_locations.txt

# 1.3 派生计数与尾块
find shared/original/op_host -name "*_tiling.cpp" -o -name "*_tiling.h" | xargs \
    grep -nE "mTotalCnt|nTotalCnt|totalBlock|mBaseTail|nBaseTail|CeilDiv" >> /tmp/p104_locations.txt

# 1.4 Kernel 端 blockIdx 与 SetSingleShape
find shared/original/op_kernel -type f \( -name "*.cpp" -o -name "*.h" \) | xargs \
    grep -nE "GetBlockIdx|SetSingleShape|totalBlock|singleCoreM|singleCoreN" >> /tmp/p104_locations.txt
```

**交付物**（记录到 `implementation_note.txt` "Playbook Step 1"）：
- `GetTiling` 调用位置（文件 + 行号）
- 当前 `singleCoreM / singleCoreN / baseM / baseN` 赋值位置
- `usedCoreNum / blockDim` 计算位置
- Kernel 端 `blockIdx → (mIdx, nIdx)` 与 `SetSingleShape` 位置

## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| singleCoreM | `?` | `≥ baseM`，16 对齐，`totalBlock ≥ aicCoreNum` | `op_host/*_tiling.cpp:L?` |
| singleCoreN | `?` | `≥ baseN`，16 对齐 | `op_host/*_tiling.cpp:L?` |
| usedCoreNum | `?` | `min(totalBlock, aicCoreNum)` | `op_host/*_tiling.cpp:L?` |
| mBaseTail / nBaseTail | 无 | 新增计算 | `op_host/*_tiling.cpp:L?` |
| Kernel 尾块尺寸 | 可能固定 | 按 `mIdx/nIdx` 动态 | `op_kernel/*.cpp:L?` |

**排除规则**（下列情况不改）：
- `totalBlock` 已经 `≥ aicCoreNum × 2`
- `M < baseM` 或 `N < baseN`，无法再分
- 算子为分组/稀疏 matmul，分核语义特殊

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — M/N 双向扩展**：M、N 均足够大，交替对半缩小 M/N。
- **形态 β — 仅 M 向扩展**：N 较小或 `singleCoreN == baseN`，只能缩小 M。
- **形态 γ — 已有多核扩展但 usedCoreNum 未更新**：只需修正 `usedCoreNum` 和尾块。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template

```cpp
constexpr uint32_t ALIGNED_H = 16;
auto singleCoreM = cubeTiling.singleCoreM;
auto singleCoreN = cubeTiling.singleCoreN;
auto baseM = cubeTiling.baseM;
auto baseN = cubeTiling.baseN;
auto aicCoreNum = platform.GetCoreNumAic();

int64_t mTotalCnt = CeilDiv(M, static_cast<int64_t>(singleCoreM));
int64_t nTotalCnt = CeilDiv(N, static_cast<int64_t>(singleCoreN));
int64_t totalBlock = mTotalCnt * nTotalCnt;

if (totalBlock < aicCoreNum) {
    while (totalBlock < aicCoreNum && singleCoreM > baseM) {
        singleCoreM = std::max(baseM, (singleCoreM / 2) / ALIGNED_H * ALIGNED_H);
        mTotalCnt = CeilDiv(M, static_cast<int64_t>(singleCoreM));
        totalBlock = mTotalCnt * nTotalCnt;
    }
    while (totalBlock < aicCoreNum && singleCoreN > baseN) {
        singleCoreN = std::max(baseN, (singleCoreN / 2) / ALIGNED_H * ALIGNED_H);
        nTotalCnt = CeilDiv(N, static_cast<int64_t>(singleCoreN));
        totalBlock = mTotalCnt * nTotalCnt;
    }
}

cubeTiling.singleCoreM = singleCoreM;
cubeTiling.singleCoreN = singleCoreN;
cubeTiling.usedCoreNum = static_cast<uint32_t>(std::min(totalBlock, aicCoreNum));

int32_t mBaseTail = M - (mTotalCnt - 1) * singleCoreM;
int32_t nBaseTail = N - (nTotalCnt - 1) * singleCoreN;
// 将 mBaseTail / nBaseTail 一并放入 TilingHeader 传给 device
```

### 3C. Variant Notes

- 若 `singleCoreM == baseM` 仍不足，记录为“已达下限”，不再继续拆分，避免 cube 发射异常。
- 可配合 P4 多核负载均衡，把尾块均匀分散到所有核。
- 若同时启用 P105 L2 tiling，`totalBlock` 应使用 L2 tile 内的 base 块数参与核分配，而不是全局 base 块数。

## Step 4: 约束复核

```
P104 改造约束核查：
  1. singleCoreM / singleCoreN 仍是 ALIGNED_H=16 的整数倍: yes/no
  2. singleCoreM ≥ baseM 且 singleCoreN ≥ baseN: yes/no
  3. usedCoreNum == min(totalBlock, aicCoreNum): yes/no
  4. Host 计算了 mBaseTail / nBaseTail 并传给 Kernel: yes/no
  5. Kernel 入口有 if (GetBlockIdx() >= totalBlock) return;: yes/no
  6. Kernel 端 SetSingleShape 使用 curSingleM/curSingleN 尾块: yes/no
  7. blockDim = min(usedCoreNum, totalBlock): yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 多核扩展代码存在
EX=$(grep -cE "singleCoreM\s*=\s*std::max\(baseM|singleCoreN\s*=\s*std::max\(baseN" \
    modified_files/op_host/*_tiling.cpp 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "Expansion loops: $EX"
# 期望：>= 1

# 检查 2: usedCoreNum 被更新
UC=$(grep -cE "usedCoreNum\s*=\s*std::min\(totalBlock|usedCoreNum\s*=\s*std::min.*aicCoreNum" \
    modified_files/op_host/*_tiling.cpp 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "usedCoreNum update: $UC"
# 期望：>= 1

# 检查 3: Kernel 端 totalBlock 守卫
GD=$(grep -cE "GetBlockIdx\(\)\s*>=\s*totalBlock\s*\)\s*return;" \
    modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "Kernel totalBlock guard: $GD"
# 期望：>= 1

# 检查 4: Kernel 端 SetSingleShape 使用动态尾块
SS=$(grep -cE "SetSingleShape.*curSingleM|SetSingleShape.*curSingleN|curSingleM\s*=|curSingleN\s*=" \
    modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "SetSingleShape with tail: $SS"
# 期望：>= 1

# 检查 5: blockDim 取 min
BD=$(grep -cE "blockDim\s*=\s*std::min\(usedCoreNum|blockDim\s*=\s*std::min.*totalBlock" \
    modified_files/op_host/*_tiling.cpp 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "blockDim min guard: $BD"
# 期望：>= 1
```

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P104 Playbook Completion]
Step 1: done, /tmp/p104_locations.txt written
Step 2: plan table filled (singleCoreM: __ → __, singleCoreN: __ → __)
Step 3: form = alpha/beta/gamma, expansion applied
Step 4: all 7 constraints passed (yes/no list above)
Step 5: all 5 grep checks passed (EX=__, UC=__, GD=__, SS=__, BD=__)
```
