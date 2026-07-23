# P99 Playbook: BroadCast + 批量向量运算替代逐行标量循环

> **强制流程**。P99 把小向量（如 `rstd[gRows]`）广播到大矩阵（`[gRows, D]`）后，用 1 条 `Mul`/`Add` 指令覆盖全量数据，替代逐行 `GetValue + Muls/Adds`。从 gRows 条 Vec 指令减少到 2 条（BroadCast + 批量运算），指令 dispatch 开销下降约 gRows/2 倍。**必须**按本 Playbook 逐步执行。

## Step 1: 定位逐行标量缩放反模式

执行下面的 grep，把结果写入 `/tmp/p99_locations.txt`：

```bash
# 1.1 逐行 GetValue + Muls/Adds
grep -nE "for\s*\([^)]*\+\+[^)]*\).*GetValue.*(Muls|Adds|Mul|Add)" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null > /tmp/p99_locations.txt

# 1.2 小向量变量名线索（rstd/inv_std/scale/shift）
grep -nE "rstd|inv_std|scale.*Local|shift.*Local|bias.*Local" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p99_locations.txt

# 1.3 已有 BroadCast 使用
grep -nE "BroadCast|bcast_tmp" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p99_locations.txt

# 1.4 行偏移计算（逐行访问大矩阵）
grep -nE "for.*gRows|for.*group.*UD|off\s*=.*\*\s*UD" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p99_locations.txt

# 1.5 扩展：标量驱动的 Matvec（qRot / Pi × q 场景）
grep -nE "GetValue\(.*\).*Muls.*\[.*\*\s*D|for.*\{.*GetValue.*Muls.*Add\(" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p99_locations.txt
```

**交付物**（`implementation_note.txt` "Playbook Step 1"）：
- **逐行缩放位置**：`for + GetValue + Muls/Adds` 的文件 + 行号
- **小向量来源**：rstd/inv_std/scale/shift 的 LocalTensor 定义位置
- **标量驱动 Matvec 位置**：`GetValue` 后按行 `Muls` + `Add` 累加的位置（qRot 类场景）
- **大矩阵 shape**：gRows / UD 等维度定义
- **已有 BroadCast**：是否已使用

## Step 2: BroadCast 改造计划表（强制）

| 调用点 (file:line) | 小向量 | 大矩阵 shape | 广播 dim | 当前指令数 | 目标指令数 | UB 增量 | 预计收益 |
|---|---|---|---|---|---|---|---|
| kernel.cpp:145 | rstdLocal [gRows,1] | xF [gRows,UD] | dim=1 | gRows Muls | 1 BroadCast + 1 Mul | gRows*UD*sizeof(T) | gRows/2x |
| ... | ... | ... | ... | ... | ... | ... | ... |

**排除规则**（下列情况**不改**）：
- **gRows < 4**：BroadCast + PipeBarrier 开销可能超过逐行 Muls
- **大矩阵非连续存储**：BroadCast 要求 dst 连续
- **小向量不是 [gRows,1] 形态**：需先 reshape 或切片
- **已有 BroadCast 且工作正常**：不重复改

## Step 3: 改造实施

### 3A. 形态识别

- **形态 α — 一维小向量广播到二维矩阵**：最常见，如 rstd [gRows,1] → [gRows,UD]
- **形态 β — 一维标量广播到一维 tensor**：用 `Duplicate` 即可（P12），不属于 P99 核心场景
- **形态 γ — 多维广播**：NDIM > 2，需仔细设计 shape 数组
- **形态 δ — 标量驱动的 matvec**：`for i: qi=qF.GetValue(i); Muls(prod, pit[i*D], qi, D); Add(out, out, prod, D)`（qRot 类场景，D 次 V→S 往返是主耗时）

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma | delta`

### 3B. Canonical Template（形态 α）

```cpp
// === 改造前（基线）：逐行标量-向量 Muls ===
for (int32_t i = 0; i < gRows; i++) {
    int32_t off = (gStart + i) * UD;
    float rstd = rstdLocal.GetValue(i);
    Muls(xF[off], xF[off], rstd, UD);
}

// === 改造后：BroadCast 展开 + 1 条 Mul 覆盖全量 ===
{
    const uint32_t src_shape[2] = {static_cast<uint32_t>(gRows), 1};
    const uint32_t dst_shape[2] = {static_cast<uint32_t>(gRows), UD};
    BroadCast<float, 2, 1>(xSq, rstdLocal, dst_shape, src_shape, bcast_tmp);
    PipeBarrier<PIPE_V>();
    Mul(xF[gStart * UD], xF[gStart * UD], xSq, gRows * UD);
}
```

**关键替换要点**：
1. `shape` 数组必须是 `const uint32_t[N]` 格式
2. `NDIM` 为维度数，`BROADCAST_DIM` 为被广播的维度（0-based）
3. `src_shape` 中 BROADCAST_DIM 维必须为 1
4. BroadCast 写 `xSq` 后、Mul 读 `xSq` 前加 `PipeBarrier<PIPE_V>()`
5. `bcast_tmp` 大小 ≥ `gRows * UD * sizeof(T)` 字节

### 3C. Variant Notes

**形态 β — 一维标量广播**：
```cpp
// 用 Duplicate 而非 BroadCast
Duplicate<float>(scaleLocal, scaleScalar, tileSize);
Mul(yLocal, xIn, scaleLocal, tileSize);
```
（此形态更偏向 P12，但可与 P99 组合使用）

**形态 γ — 多维广播**：
```cpp
const uint32_t src_shape[3] = {B, 1, D};
const uint32_t dst_shape[3] = {B, C, D};
BroadCast<float, 3, 1>(xSq, scaleLocal, dst_shape, src_shape, bcast_tmp);
```

**形态 δ — 标量驱动 matvec → BroadCast + Mul + ReduceSum**：
```cpp
// qF [D, 1] → qFBcast [D, D]
const uint32_t q_shape[2] = {(uint32_t)D_, 1};
const uint32_t pi_shape[2] = {(uint32_t)D_, (uint32_t)D_};
BroadCast<float, 2, 1>(qFBcast, qF, pi_shape, q_shape, bcast_tmp);
PipeBarrier<PIPE_V>();
Mul(qPi, qFBcast, pitUb, D_ * D_);
ReduceSum<float, true>(qRot, qPi, reduceTmp, D_ * D_);
```
⚠️ UB 预算：`D*D` fp32 ≈ 64KB（D=128），不足时改用 Host 预计算（X5）。
⚠️ 替代方案：**P94 形态 ε（复制-Mul-批量逐行 ReduceSum）**，UB 占用相同、归约结果直接落 UB 零 GetValue，两者任选其一，不要混用。

**与其他策略协同**：
- + **P95 向量链**：BroadCast 后的批量运算可继续接 Adds/Muls 链
- + **P69 UB 融合链**：BroadCast + Mul + 后续向量链全部保留在 UB
- + **P97 零拷贝切片**：需要从 UB tensor 取子集时配合使用

## Step 4: 约束复核

```
P99 改造约束核查：
  1. src_shape[BROADCAST_DIM] == 1: yes/no
  2. dst 连续且 32B 对齐: yes/no
  3. bcast_tmp >= gRows * UD * sizeof(T): yes/no
  4. xSq != xF（不同 buffer）或已加正确 barrier: yes/no
  5. gRows >= 4: yes/no
  6. NDIM/BROADCAST_DIM 与 shape 数组严格匹配: yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 逐行 GetValue + Muls/Adds 减少
BEFORE=$(grep -cE "for\s*\([^)]*\+\+[^)]*\).*GetValue.*(Muls|Adds)" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
AFTER=$(grep -cE "for\s*\([^)]*\+\+[^)]*\).*GetValue.*(Muls|Adds)" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "row-wise GetValue+Muls/Adds: before=$BEFORE after=$AFTER"
# 期望：AFTER < BEFORE

# 检查 2: BroadCast 使用增加
BCAST_B=$(grep -cE "BroadCast\s*<" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
BCAST_A=$(grep -cE "BroadCast\s*<" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "BroadCast calls: before=$BCAST_B after=$BCAST_A"
# 期望：BCAST_A > BCAST_B

# 检查 3: 大矩阵批量 Mul/Add 出现
BULK_B=$(grep -cE "Mul\s*\(.*gRows\s*\*\s*UD|Add\s*\(.*gRows\s*\*\s*UD" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
BULK_A=$(grep -cE "Mul\s*\(.*gRows\s*\*\s*UD|Add\s*\(.*gRows\s*\*\s*UD" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "bulk Mul/Add: before=$BULK_B after=$BULK_A"
# 期望：BULK_A > BULK_B

# 检查 4: PipeBarrier<PIPE_V> 在 BroadCast 与 Mul 之间
PB=$(grep -nE "BroadCast.*\n.*PipeBarrier|PipeBarrier.*\n.*Mul" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | wc -l)
# 因跨行 grep 复杂，agent 人工检查并在 implementation_note.txt 列出 barrier 位置

# 检查 5: bcast_tmp buffer 已声明
BTMP=$(grep -cE "bcast_tmp|BroadCast.*tmp" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "bcast_tmp markers: $BTMP"
# 期望：>= 1
```

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：BroadCast 模板参数 | NDIM/BROADCAST_DIM 与 shape 数组维度不匹配；src_shape 广播维必须为 1 |
| 运行结果错乱 | dst 与 src 指向同一 buffer 且无 barrier；改为不同 buffer 或加 PipeBarrier<PIPE_V> |
| 性能反而变差 | gRows < 4，BroadCast 开销超过逐行 Muls；保留原实现 |
| UB 溢出 | bcast_tmp 过大；减小 tile 或改用 P97 切片 |
| 精度下降 | BroadCast 后的 Mul 中间结果类型不一致；保持 FP32 累加 |

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P99 Playbook Completion]
Step 1: done, N row-wise GetValue+Muls/Adds identified
Step 2: refactor plan filled (small vec shape=___, big matrix shape=___, broadcast dim=___, form={alpha|beta|gamma|delta})
Step 3: BroadCast + bulk Mul/Add applied
Step 4: src_shape[dim]=1, alignment=___, bcast_tmp=___ bytes, gRows>=4 ✓
Step 5: row-wise loops: X→Y, BroadCast: X→Y, bulk Mul/Add: X→Y
Step 6: no pitfalls / {列出触发的}
```
