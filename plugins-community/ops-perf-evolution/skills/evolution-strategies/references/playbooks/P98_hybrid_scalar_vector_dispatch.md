# P98 Playbook: 混合标量/向量路径自适应调度

> **强制流程**。P98 根据 per-tile workload 大小自动选择标量路径或向量路径，消除向量化在极小 workload 上的退化。大 workload 走 EnQue/DeQue + Vector API；小 workload 走 AllocTensor → DataCopyPad → raw pointer → FreeTensor（无队列同步开销）。**必须**按本 Playbook 逐步执行。

## Step 1: 定位单一路径与 workload 信息

执行下面的 grep，把结果写入 `/tmp/p98_locations.txt`：

```bash
# 1.1 单一 Compute/Process 函数（无 dispatch 分支）
grep -nE "void\s+Process|void\s+Compute|void\s+Calc" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null > /tmp/p98_locations.txt

# 1.2 现有条件分支（已有 dispatch 的线索）
grep -nE "if\s*\(.*spatial|if\s*\(.*rowLength|if\s*\(.*tile.*size|if\s*\(.*count" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p98_locations.txt

# 1.3 workload 维度定义
grep -nE "spatial|rowLength|tile_size|count_spatial|numCol|coreNum" \
    shared/original/op_host/*_tiling.cpp shared/original/op_host/*_tiling.h 2>/dev/null >> /tmp/p98_locations.txt

# 1.4 已使用的 Vector API 和 EnQue/DeQue
grep -nE "EnQue|DeQue|ReduceSum|Adds|Muls|SetMaskCount" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p98_locations.txt
```

**交付物**（`implementation_note.txt` "Playbook Step 1"）：
- **单一路径函数**：无 dispatch 的 Process/Compute 函数位置
- **workload 维度**：spatial / rowLength / tile_size 等变量和取值范围
- **已有 Vector API**：是否使用 ReduceSum/Adds/Muls/EnQue/DeQue

## Step 2: Dispatch 改造计划表（强制）

| 调用点 (file:line) | workload 变量 | 阈值（元素数） | 标量路径条件 | 向量路径条件 | 备注 |
|---|---|---|---|---|---|
| kernel.cpp:145 | count_spatial | 128 | <128 | >=128 | 初始阈值 |
| kernel.cpp:234 | rowLength | 32 | <=32 | >32 | Sort32 batch |
| ... | ... | ... | ... | ... | ... |

**排除规则**（下列情况**不改**）：
- **所有 case 的 workload 都集中在阈值一侧**：无需 dispatch
- **算子已经是纯标量且无 Vector API**：无向量路径可切
- **算子已经是纯向量且无标量路径**：已经是全向量化
- **dispatch 分支复杂度会爆炸**：>3 级分支且收益不明确时保持现状

## Step 3: 改造实施

### 3A. 形态识别

- **形态 α — 单阈值双路径**：一个 workload 变量，一个阈值
- **形态 β — 多级 dispatch**：如 TopK 按 rowLength 分 3-5 级
- **形态 γ — dtype/对齐混合 dispatch**：根据 `count*sizeof(T)` 和对齐性综合决策

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 α — 单阈值双路径）

```cpp
constexpr int32_t VECTOR_MIN_SPATIAL = 128;  // 初始阈值，可基于 profiling 调优

void ProcessBatched(...) {
    if (count_spatial >= VECTOR_MIN_SPATIAL) {
        ProcessBatchedVectorized(...);
    } else {
        ProcessBatchedScalar(...);
    }
}

// 标量路径：无 EnQue/DeQue
__aicore__ void ProcessBatchedScalar(...) {
    LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    DataCopyPad(xLocal, xGm + offset, count);
    float* xPtr = reinterpret_cast<float*>(xLocal.GetPhyAddr());
    float sum = 0.0f;
    for (int32_t i = 0; i < count; ++i) {
        sum += xPtr[i];
    }
    // ... 标量处理 ...
    inQueueX.FreeTensor(xLocal);
}

// 向量路径：EnQue/DeQue + ReduceSum
__aicore__ void ProcessBatchedVectorized(...) {
    LocalTensor<float> xIn = inQueueX.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    AscendC::ReduceSum<float, true>(scalarLocal, xIn, reduceTmp, count);
    // ...
    outQueueY.EnQue<float>(yLocal);
    inQueueX.FreeTensor(xIn);
}
```

**关键替换要点**：
1. 标量路径不使用 EnQue/DeQue，直接 `AllocTensor → DataCopyPad → raw pointer → FreeTensor`
2. 阈值初始值 128 元素（910B），后续基于 profiling 反馈调整（±16 步长）
3. 阈值应基于 `per_tile_elements * sizeof(T)` 而非仅元素数
4. 不同硬件（910B/950）阈值可能不同，需 per-arch 配置

### 3C. Variant Notes

**形态 β — 多级 dispatch**：
```cpp
if (rowLength <= 32) {
    ComputeBatchSmall(...);    // Sort32(repeat=B)
} else if (rowLength <= 128 && is32BAligned(rowLength)) {
    ComputeBatchAligned(...);  // 单 MTE2/MTE3 burst
} else if (rowLength <= 512) {
    ComputeBatchStaged(...);   // slot staging
} else {
    ComputeRow(...);           // 单行 tiled
}
```

**与其他策略协同**：
- + **P94/P95**：大 workload 走向量归约/向量链；小 workload 走标量
- + **P96 Sort32**：小行批量走 Sort32(repeat=B)；大行走 Tiled Sort + MrgSort
- + **P67 Counter 模式**：向量路径内可用 Counter 减少 repeat 计算

## Step 4: 约束复核

```
P98 改造约束核查：
  1. 阈值基于 per_tile_elements * sizeof(T): yes/no
  2. 标量路径无 EnQue/DeQue: yes/no
  3. 向量路径保留 EnQue/DeQue + Vector API: yes/no
  4. 两路径结果一致性验证通过: yes/no
  5. 阈值两侧都有实际 case: yes/no
  6. dispatch 不跨函数 / 不污染外部状态: yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 存在 dispatch 分支
DISPATCH=$(grep -cE "VECTOR_MIN_SPATIAL|if\s*\(.*count.*>=|if\s*\(.*rowLength.*<=|Process.*Scalar|Process.*Vectorized" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "dispatch markers: $DISPATCH"
# 期望：>= 1

# 检查 2: 标量路径无 EnQue/DeQue（在 ProcessBatchedScalar 等函数内）
# agent 人工检查：标量路径函数体内无 EnQue/DeQue

# 检查 3: 向量路径保留 Vector API
VEC=$(grep -cE "ReduceSum|Adds|Muls|SetMaskCount" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "Vector APIs: $VEC"
# 期望：>= 1

# 检查 4: 阈值常量已定义
THRESH=$(grep -cE "VECTOR_MIN_SPATIAL|rowLength.*<=|count.*>=" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "threshold markers: $THRESH"
# 期望：>= 1

# 检查 5: 无单函数内混合路径（即一个函数里又标量又向量交错）
# agent 人工确认：标量路径与向量路径为独立函数或清晰分支
```

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 阈值过高，大部分走标量 | 下调阈值，让更多 case 走向量路径 |
| 阈值过低，小 case 退化 | 上调阈值，小 workload 保留标量 |
| 两路径精度不一致 | 检查标量路径的 rounding、数据类型转换是否与向量路径一致 |
| dispatch 增加代码体积导致 icache miss | 合并公共代码，减少分支内重复 |
| 阈值对某类 case 始终不适用 | 引入第二维度阈值（如 dtype、对齐性） |

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P98 Playbook Completion]
Step 1: done, N single-path functions identified
Step 2: dispatch plan filled (threshold=___, form={alpha|beta|gamma})
Step 3: scalar path (no EnQue/DeQue) + vector path (Vector API) implemented
Step 4: threshold basis=___, consistency check=___ ✓
Step 5: dispatch markers=___, Vector APIs=___, thresholds=___
Step 6: no pitfalls / {列出触发的}
```
