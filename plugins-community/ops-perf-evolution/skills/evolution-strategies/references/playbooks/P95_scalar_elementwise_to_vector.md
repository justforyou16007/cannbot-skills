# P95 Playbook: 标量逐元素运算向量化 (Scalar Element-wise → Adds/Muls Vector Chain)

> **强制流程**。P95 把 kernel 中逐元素标量算术循环（`for(i) y[i] = f(x[i])`）替换为 AscendC 标量-向量指令（`Adds`/`Muls`/`Subs`）和向量-向量指令（`Add`/`Mul`/`Div`/`Sub`）链。`Adds`/`Muls` 直接接收标量参数，无需 `Duplicate` 广播 buffer（Vector 单元 8×FP32/16×FP16 并行，理论加速 8-16x）。**必须**按本 Playbook 逐步执行。

## Step 1: 定位标量逐元素运算循环

执行下面的 grep，把结果写入 `/tmp/p95_locations.txt`：

```bash
# 1.1 逐元素标量算术循环
grep -nE "for\s*\([^)]*\+\+[^)]*\).*y\[.*\].*=.*x\[" \
    shared/original/op_kernel/*.{cpp,h} 2>/dev/null > /tmp/p95_locations.txt
# 1.2 raw pointer 标量计算
grep -nE "reinterpret_cast\s*<\s*(float|half)\s*\*\s*>|\.GetPhyAddr\(\)" \
    shared/original/op_kernel/*.{cpp,h} 2>/dev/null >> /tmp/p95_locations.txt
# 1.3 已有的 Adds/Muls/Duplicate 使用（参考基线 / 避免重复）
grep -nE "\b(Adds|Muls|Subs|Duplicate|SubRelu)\s*\(" \
    shared/original/op_kernel/*.{cpp,h} 2>/dev/null >> /tmp/p95_locations.txt
# 1.4 标量取值后参与向量运算（典型反模式）
grep -nE "GetValue\(.*\).*Muls|GetValue\(.*\).*Adds|GetValue\(.*\).*Mul|GetValue\(.*\).*Axpy" \
    shared/original/op_kernel/*.{cpp,h} 2>/dev/null >> /tmp/p95_locations.txt
# 1.5 标量循环包裹向量 BLAS-like 调用（形态 ε 候选）
grep -nE "for\s*\([^)]*\+\+[^)]*\).*\{.*\b(Axpy|Muls|Adds|Add|Mul)\s*\(" \
    shared/original/op_kernel/*.{cpp,h} 2>/dev/null >> /tmp/p95_locations.txt
```

**交付物**（`implementation_note.txt` "Playbook Step 1"）：
- **标量逐元素循环位置**：每个 `for + y[i] = f(x[i])` 的文件 + 行号
- **raw pointer 位置**：`GetPhyAddr` / `reinterpret_cast<float*>` 使用点
- **GetValue 后接向量运算**：小向量逐行缩放的反模式位置
- **标量循环包裹向量 BLAS-like 调用**：`for` 内 `Axpy`/`Muls`/`Adds` 的位置

## Step 2: 向量链改造计划表（强制）

**不填完此表不得进入 Step 3**：

| 调用点 (file:line) | 当前标量表达式 | 目标向量链 | 中间 tensor 数 | UB 增量 | 预计收益 |
|---|---|---|---|---|---|
| kernel.cpp:145 | `y[i]=(x[i]-mean)*inv_std*w+b` | `Adds→Muls→Muls→Adds` | 1 | count×sizeof(T) | 4-8x |
| kernel.cpp:234 | `y[i]=x[i]*scale+shift` | `Muls→Adds` | 1 | count×sizeof(T) | 3-6x |

**排除规则**（下列情况**不改**）：
- **count < 128**：向量化 EnQue/DeQue 开销可能超过收益，考虑 P98 混合调度
- **标量表达式含复杂控制流 / 分支**：如 `if (x[i] > 0) y[i] = ... else ...`，需先转 `Select`/`Compare`（P12/P10）
- **中间结果需要写回 GM 被其他 tile 消费**：向量链应保持中间结果仅在 UB
- **dtype 不在 Vector 指令支持列表**：需先 Cast，权衡 Cast 开销

## Step 3: 改造实施

### 3A. 形态识别

判断你的算子属于以下哪种主导形态：

- **形态 α — 单目/双目逐元素**：如 `y[i] = x[i] * scale + shift`
- **形态 β — 归一化链**：如 `y[i] = (x[i] - mean) * inv_std * w + b`
- **形态 γ — 多输入融合链**：如 `y[i] = (a[i] - b[i]) * c[i] + d[i]`
- **形态 δ — 小向量逐行缩放**：如 `for i in gRows: Muls(x[off], x[off], rstd.GetValue(i), UD)`（应升级 P99 BroadCast）
- **形态 ε — 标量循环包裹向量 BLAS-like 调用**：外层 `for` 内调用 `Axpy`/`Muls`，scale 来自 `GetValue` 标量

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma | delta | epsilon`

### 3B. Canonical Template（形态 β — 归一化链，最常见）

```cpp
// === 改造前（基线）：标量逐元素归一化 ===
float* xPtr = reinterpret_cast<float*>(xLocal.GetPhyAddr());
float* yPtr = reinterpret_cast<float*>(yLocal.GetPhyAddr());
for (int32_t i = 0; i < count_spatial; ++i) {
    float val = xPtr[i];
    float normalized = (val - mean) * inv_std;
    yPtr[i] = normalized * w_val + b_val;
}

// === 改造后：Adds/Muls 标量-向量指令链（4 条指令） ===
AscendC::Adds<float>(yLocal, xIn, -mean, count_spatial);       // y = x - mean
AscendC::Muls<float>(yLocal, yLocal, inv_std, count_spatial);  // y = y * inv_std
AscendC::Muls<float>(yLocal, yLocal, w_val, count_spatial);    // y = y * weight
AscendC::Adds<float>(yLocal, yLocal, b_val, count_spatial);    // y = y + bias
```

**关键替换要点**：
1. 优先标量-向量指令（`Adds`/`Muls`/`Subs`），避免 `Duplicate` 广播→`Add`/`Mul` 两步操作
2. 连续 Vector 写同一 buffer 后如需 MTE3 搬运，必须加 `PipeBarrier<PIPE_V>()`
3. 向量链中间结果不写回 GM，全程保留在 UB
4. 标量参数类型必须与 tensor 元素类型一致（FP32 tensor + float scalar）

### 3C. Variant Notes

**形态 α — 单目缩放**：
```cpp
AscendC::Muls<float>(yLocal, xIn, scale, count_spatial);
AscendC::Adds<float>(yLocal, yLocal, shift, count_spatial);
```

**形态 γ — 多输入融合链**：
```cpp
AscendC::Sub<float>(t1, aIn, bIn, count_spatial);
AscendC::Mul<float>(t1, t1, cIn, count_spatial);
AscendC::Add<float>(yLocal, t1, dIn, count_spatial);
```

**形态 δ — 小向量逐行缩放 → 升级 P99**：`BroadCast<float, 2, 1>(xSq, rstdLocal, dst_shape, src_shape, bcast_tmp)` + `PipeBarrier<PIPE_V>()` + 1 条 `Mul(xF, xF, xSq, gRows * UD)` 覆盖全量（完整模板见 P99 Playbook Step 3B）。

**形态 ε — 标量循环包裹向量 BLAS-like 调用**：
看到 `Axpy` 不代表已向量化——标量 scale 的 V→S 往返仍是瓶颈：
```cpp
// === 反模式 ===
for (int32_t t = 0; t < nTok; t++) {
    float pt = small.GetValue(144 + t);           // 标量取值
    Axpy(acc, vBlkF[t * D_], pt, (int32_t)D_);    // 向量 op，但 scale 标量
}
```
优化方向（按 nTok 规模选择）：
1. **nTok 小**：`Muls(tmp, vBlkF[t*D_], pt, D)` + `Add(acc, acc, tmp, D)` 替代 `Axpy`，标量 scale 显式化为 `Muls` 参数，便于 Adds/Muls 链融合
2. **nTok 大**：`BroadCast` 把 `small[144..144+nTok]` 展开到 `[nTok, D]`，一次 `Mul` + `ReduceSum` 消除外层循环（升级 P99/P94 形态 ε）。⚠️ scalar-issue-bound 场景下 broadcast 新增的 vector pipe 工作量可能抵消标量节省（实测曾净中性），仅在 vector pipe 有空余时采用

**与其他策略协同**：
- + **P67 Counter 模式**：向量链用 Counter 模式统一处理主+尾块
- + **P69 UB 融合链**：多条向量指令中间结果保留 UB 直通消费，减少 GM 往返
- + **P97 零拷贝切片**：需从 UB 大 tensor 取子集时用 `tensor[offset]` 视图替代 gather

## Step 4: UB + 对齐约束复核

**必须**在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P95 改造约束核查：
  1. 向量链长度: N 条指令；中间 tensor 总数: M 个
  2. 中间 tensor 总大小: M * count * sizeof(T) = ___ bytes <= UB 剩余 ___ bytes ✓/✗
  3. count 对齐: count * sizeof(T) % 32B == 0 ? yes/no
  4. 标量参数类型与 T 一致: yes/no
  5. 链尾到 MTE3 搬运之间有 PipeBarrier<PIPE_V>: yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。禁止把失败 grep 标记为"不适用"。

```bash
# 检查 1: 标量逐元素循环减少（期望 AFTER < BEFORE）
BEFORE=$(grep -cE "for\s*\([^)]*\+\+[^)]*\).*y\[.*\].*=.*x\[" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
AFTER=$(grep -cE "for\s*\([^)]*\+\+[^)]*\).*y\[.*\].*=.*x\[" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "scalar elementwise loops: before=$BEFORE after=$AFTER"

# 检查 2: Adds/Muls/Subs 使用增加（期望 VEC_A > VEC_B）
VEC_B=$(grep -cE "\b(Adds|Muls|Subs)\s*\(" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
VEC_A=$(grep -cE "\b(Adds|Muls|Subs)\s*\(" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "Adds/Muls/Subs: before=$VEC_B after=$VEC_A"

# 检查 3: GetValue 后接 Muls/Adds 反模式减少（期望 GETVEC_A < GETVEC_B，或转为 BroadCast 形式）
GETVEC_B=$(grep -cE "GetValue\(.*\).*\b(Muls|Adds|Mul|Add)\b" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
GETVEC_A=$(grep -cE "GetValue\(.*\).*\b(Muls|Adds|Mul|Add)\b" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "GetValue+Vec: before=$GETVEC_B after=$GETVEC_A"

# 检查 4: PipeBarrier<PIPE_V> 存在（期望 >= 1）
grep -cE "PipeBarrier\s*<\s*PIPE_V\s*>" modified_files/op_kernel/*.{cpp,h}

# 检查 5: 无新增 raw pointer 逐元素循环（期望 == 0）
grep -cE "for\s*\([^)]*\+\+[^)]*\).*reinterpret_cast.*GetPhyAddr" modified_files/op_kernel/*.{cpp,h}
```

**在 implementation_note.txt "Playbook Step 5" 列出每条实际输出**。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：Adds/Muls 模板推导失败 | scalar 参数类型与 tensor 元素类型不一致；用 `static_cast<float>(scalar)` |
| 运行精度不对 | 漏了 `PipeBarrier<PIPE_V>()`；连续 Vector 写后未同步就 MTE3 搬运 |
| 小 count 性能退化 | 走 P98 混合调度，count<128 保留标量路径 |
| UB 溢出 | 向量链中间 tensor 过多；减少链长或分批处理 |
| 形态 δ 收益不明显 | 小 gRows（<4）时 BroadCast 开销可能超过逐行 Muls，保留原实现 |
| 复杂分支表达式无法直接用 Adds/Muls | 先用 Compare+Select（P12）或 GatherMask（P100）处理分支 |

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P95 Playbook Completion]
Step 1: done, N scalar elementwise loops identified
Step 2: refactor plan filled (M sites, form={alpha|beta|gamma|delta|epsilon})
Step 3: vector chain applied, key APIs = Adds/Muls/Subs/Add/Mul
Step 4: UB delta=___ bytes, alignment=___, PipeBarrier=___ ✓
Step 5: all 5 grep checks passed (scalar loops: X→Y, Adds/Muls: X→Y, GetValue+Vec: X→Y)
Step 6: no pitfalls / {列出触发的}
```
