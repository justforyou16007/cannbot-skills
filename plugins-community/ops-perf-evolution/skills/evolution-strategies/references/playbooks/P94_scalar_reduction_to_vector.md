# P94 Playbook: 标量归约向量化 (Scalar Reduction → Vector Reduce API)

> **强制流程**。P94 把 kernel 中逐元素的标量归约循环（`for(i) sum += x[i]` / `for(i) if (x[i] > max) max = x[i]`）替换为 AscendC `ReduceSum`/`ReduceMax`/`WholeReduceSum` Vector API（Vector 单元 8×FP32/16×FP16 并行 + 硬件树归约 O(log n) 延迟）。**必须**按本 Playbook 逐步执行。

## Step 1: 定位标量归约循环

执行下面的 grep，把结果写入 `/tmp/p94_locations.txt`：

```bash
# 1.1 标量归约循环（累加 / 求最大 / 求最小）
grep -nE "for\s*\([^)]*\+\+[^)]*\).*\{\s*(sum|max_val|min_val|result)\s*[\+\-]?=" \
    shared/original/op_kernel/*.{cpp,h} 2>/dev/null > /tmp/p94_locations.txt
# 1.2 raw pointer 使用（标量计算前兆）
grep -nE "reinterpret_cast.*GetPhyAddr|\.GetPhyAddr\(\)" \
    shared/original/op_kernel/*.{cpp,h} 2>/dev/null >> /tmp/p94_locations.txt
# 1.3 已有的 Reduce API（参考基线 / 避免重复）
grep -nE "ReduceSum|WholeReduceSum|ReduceMax|ReduceMin|scalarBuf|VECCALC" \
    shared/original/op_kernel/*.{cpp,h} 2>/dev/null >> /tmp/p94_locations.txt
# 1.4 扩展：Vector Reduce 后的二次标量聚合（形态 ζ 候选）
grep -nE "for\s*\([^)]*\+\+[^)]*\).*\{.*GetValue.*\+=" \
    shared/original/op_kernel/*.{cpp,h} 2>/dev/null >> /tmp/p94_locations.txt
# 1.5 扩展：GetValue 标量操作数驱动的 matvec（形态 ε 候选）
grep -nE "for\s*\([^)]*\+\+[^)]*\).*\{[^}]*GetValue[^}]*Muls" \
    shared/original/op_kernel/*.{cpp,h} 2>/dev/null >> /tmp/p94_locations.txt
# 1.6 归约维度相关 tiling 参数
grep -nE "count_spatial|spatial_size|rowLength|numCol|reduce_dim|axis" \
    shared/original/op_host/*_tiling.{cpp,h} 2>/dev/null >> /tmp/p94_locations.txt
```

**交付物**（`implementation_note.txt` "Playbook Step 1"）：
- **标量归约位置**：每个 `for + sum/max/min` 的文件 + 行号（含 `GetPhyAddr` raw pointer 使用点）
- **已有 Reduce API**：是否已使用 ReduceSum/WholeReduceSum
- **归约维度**：`count_spatial` / `rowLength` 等关键变量值范围

## Step 2: 归约改造计划表（强制）

**不填完此表不得进入 Step 3**：

| 调用点 (file:line) | 归约类型 | 当前标量循环变量 | 元素数范围 | 目标 API | 是否需要 sum_sq | 预计收益 |
|---|---|---|---|---|---|---|
| kernel.cpp:145 | sum | `sum += xPtr[i]` | 1024-4096 | `ReduceSum<float, true>` | 是 | 3-5x |
| kernel.cpp:234 | max | `if (x[i] > max)` | 256-1024 | `ReduceMax<float, true>` | 否 | 2-4x |

**排除规则**（下列情况**不改**）：
- **归约维度 < 64**：向量化开销可能超过收益，考虑 P98 混合调度
- **已有 ReduceSum/WholeReduceSum**：已经是向量化路径，不重复改
- **归约涉及跨 tile 原子累加**：需先确认 AtomicAdd 是否能被 Vector 归约替代
- **归约后仍需逐元素使用原始值**：`Mul(xIn, xIn, xIn)` in-place 会破坏原始值，需先拷贝

## Step 3: 改造实施

### 3A. 形态识别

判断你的算子属于以下哪种主导形态：

- **形态 α — 单次 sum 归约**：如 GroupNorm 的 `sum += x[i]`
- **形态 β — 同时需要 sum 与 sum_sq**：如 LayerNorm/GroupNorm 需要 mean + variance
- **形态 γ — max/min 归约**：如 Softmax/TopK 需要 `ReduceMax`
- **形态 δ — 大行分块归约**：count > 65536，先 `Add` 折半再 `WholeReduceSum`
- **形态 ε — GetValue 驱动的 matvec**：`for i: qi=qF.GetValue(i); Muls(prod, pit[i*D], qi, D); Add(out, out, prod, D)`，D 次 V→S 往返是主耗时（实测 qrot kernel scalar 占比 45%）
- **形态 ζ — Vector Reduce 后的二次标量聚合**：归约结果被 `GetValue` 收集到标量数组，再在外层标量循环二次累加

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma | delta | epsilon | zeta`

### 3B. Canonical Template（形态 α — 最常见）

```cpp
// === 改造前（标量归约循环）===
float* xPtr = reinterpret_cast<float*>(xLocal.GetPhyAddr());
float sum = 0.0f;
for (int32_t i = 0; i < count_spatial; ++i) { sum += xPtr[i]; }

// === 改造后：ReduceSum Level 2 API ===
LocalTensor<float> scalarLocal = scalarBuf.Get<float>();
LocalTensor<float> reduceTmp = sumBuf.Get<float>();
AscendC::ReduceSum<float, true>(scalarLocal, xIn, reduceTmp, count_spatial);
float sum = scalarLocal.GetValue(0);
```

**关键替换要点**：
1. `scalarBuf`（`TBuf<VECCALC>`，≥ 64 × sizeof(float)，8B 对齐）存放标量输出；`sumBuf`（`TBuf<VECCALC>`，≥ 8KB）作为 Level 2 workspace
2. `isTail=true`：数据可能不足一个 repeat 时设 `true`；精确整除时设 `false` 可获更好性能
3. 先 `GetValue(0)` 取值再调度下一次 ReduceSum，否则结果被覆盖

### 3C. Variant Notes

**形态 β — sum + sum_sq**：
```cpp
AscendC::ReduceSum<float, true>(scalarLocal, xIn, reduceTmp, count_spatial);
float sum = scalarLocal.GetValue(0);
AscendC::Mul<float>(xIn, xIn, xIn, count_spatial);   // in-place x²，原值不再可用
AscendC::ReduceSum<float, true>(scalarLocal, xIn, reduceTmp, count_spatial);
float sum_sq = scalarLocal.GetValue(0);
```

**形态 γ — ReduceMax**：`AscendC::ReduceMax<float>(scalarLocal, xIn, reduceTmp, count_spatial);`

**形态 δ — 大行分块 + WholeReduceSum**（count > 65536）：先按 256B 分块局部 `Add` 折叠缩小到 256B 以内，再用 `WholeReduceSum` 一次完成（完整模板见 P68）。

**形态 ε — matvec 去 GetValue（复制-Mul-批量归约）**：
```cpp
// 1. qF [D] 复制为 D 行 [D, D]（向量 Copy，无标量）
for (int64_t j = 0; j < D; j++) { Copy(repl[j * D], qF, D); }
PipeBarrier<PIPE_V>();
// 2. 与 Pi 逐元素 Mul（一次 D*D 向量指令）
Mul(repl, repl, pitUb, D * D);
// 3. D 条背靠背 ReduceSum，结果直接落 UB（qRot[j]），
//    每行独立 work slice（redWork[j*16]）消除 buffer 共享串行化
for (int64_t j = 0; j < D; j++) {
    ReduceSum<float>(qRot[j], repl[j * D], redWork[j * 16], D);
}
```
⚠️ UB 预算：D*D fp32 在 D=128 时单份 64KB，repl + pit 双份 ~128KB，必须先过 Step 4 核算；不足时退化为 Host 预计算（X5）或分块复制。
⚠️ 归约轴对应：改造后为 `out[j] = sum_i q[i]*Pi[j][i]`（沿行归约），与原写法 `out += q[i]*Pi[i][:]`（按列累加）数学等价，但要求 Pi 布局与归约轴一致，必要时配合转置（P26/P59）。归约结果直接落 UB，全程零 GetValue，不要画蛇添足再加标量读回。
⚠️ 替代方案：P99 形态 δ（BroadCast + Mul + 整段 ReduceSum），UB 占用相同但指令数更少。

**形态 ζ — 二次标量聚合 → 二次 ReduceSum**：
```cpp
// === 改造前：Reduce 结果收集到标量数组再循环累加 ===
for (int32_t t = 0; t < nTok; t++) { pArr[t] = small.GetValue(144 + t); lAdd += pArr[t]; }
// === 改造后：直接对连续区间做第二次 ReduceSum ===
LocalTensor<float> lAddScalar = scalarBuf.Get<float>();
ReduceSum<float, true>(lAddScalar, small[144], reduceTmp, nTok);
PipeBarrier<PIPE_V>();
float lAdd = lAddScalar.GetValue(0);
```

**与其他策略协同**：
- + **P67 Counter 模式**：归约后的向量链用 Counter 模式减少 repeat/tail 计算
- + **P68 低延迟归约**：大行场景 `BlockReduceSum + WholeReduceSum` 组合降低延迟
- + **P95 Adds/Muls 链**：归约后的 mean/variance 缩放用 `Adds`/`Muls` 一次完成

## Step 4: Buffer + 对齐约束复核

**必须**在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P94 改造约束核查：
  1. scalarBuf: TBuf<VECCALC> ✓/✗, >= 64*sizeof(T) ✓/✗, 8B 对齐 ✓/✗
  2. reduceTmp (sumBuf): TBuf<VECCALC> ✓/✗, >= 8KB ✓/✗
  3. isTail: count_spatial 整除 repeat 长度 yes/no → isTail=__
  4. UB: 新增 buffer 总大小 = ___ bytes <= tile 预算剩余 ___ bytes ✓/✗
  5. (仅形态 ε) repl + pit 双份 <= UB 剩余 ✓/✗；Pi 布局与归约轴一致 ✓/✗
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。

```bash
# 检查 1: 标量归约循环减少（期望 AFTER < BEFORE）
BEFORE=$(grep -cE "for\s*\([^)]*\+\+[^)]*\).*sum\s*\+=\s*xPtr\[" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
AFTER=$(grep -cE "for\s*\([^)]*\+\+[^)]*\).*sum\s*\+=\s*xPtr\[" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "scalar reduction loops: before=$BEFORE after=$AFTER"

# 检查 2: ReduceSum / ReduceMax 使用增加（期望 REDUCE_A > REDUCE_B）
REDUCE_B=$(grep -cE "\b(ReduceSum|ReduceMax|WholeReduceSum)\s*\(" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
REDUCE_A=$(grep -cE "\b(ReduceSum|ReduceMax|WholeReduceSum)\s*\(" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "Reduce APIs: before=$REDUCE_B after=$REDUCE_A"

# 检查 3: raw pointer GetPhyAddr 使用减少（期望 PHY_A <= PHY_B）
PHY_B=$(grep -cE "\.GetPhyAddr\(\)" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
PHY_A=$(grep -cE "\.GetPhyAddr\(\)" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "GetPhyAddr: before=$PHY_B after=$PHY_A"

# 检查 4: scalarBuf / VECCALC 已声明（期望 >= 1）
grep -cE "scalarBuf|TBuf\s*<\s*VECCALC\s*>" modified_files/op_kernel/*.{cpp,h}

# 检查 5: GetValue(0) 取值在下次 ReduceSum 之前
# agent 人工检查：每个 ReduceSum 后先 `float xxx = scalarLocal.GetValue(0);` 再进入下一次归约
# 在 implementation_note.txt Step 5 列出每个 ReduceSum 的取值证据（file:line）
```

**在 implementation_note.txt "Playbook Step 5" 列出每条实际输出**。任一失败 → 回 Step 3 重做。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 运行结果全 0 或数值异常 | `scalarBuf` 未用 `TBuf<VECCALC>` 或未 8B 对齐 |
| 精度不过（max_abs_diff 大）| `Mul(xIn, xIn, xIn)` in-place 破坏了后续仍需的原始值；先拷贝再平方 |
| 小 count（<64）性能反而变差 | 走 P98 混合调度，<128 元素保留标量路径 |
| ReduceSum 后取值被覆盖 | 先 `GetValue(0)` 再调用下一次 ReduceSum |
| 大行（>65536）编译失败/性能差 | 改用 P68 BlockReduceSum + WholeReduceSum 组合 |
| 归约类型不匹配 | `sum` 用 ReduceSum，`max/min` 用 ReduceMax/ReduceMin，不能混用 |
| 形态 ε 数值与基线不符或结果互相覆盖 | Pi 布局与归约轴不对应（行/列搞反）；或每行未用独立 work slice（redWork[j*16]）；先过 Step 4 预算 |

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P94 Playbook Completion]
Step 1: done, N scalar reduction loops identified
Step 2: refactor plan filled (M sites, form={alpha|beta|gamma|delta|epsilon|zeta})
Step 3: template applied, key replacement = ReduceSum/ReduceMax/WholeReduceSum
Step 4: scalarBuf=VECCALC, reduceTmp>=8KB, isTail=__, UB delta=___ ✓
Step 5: all 5 grep checks passed (scalar loops: X→Y, Reduce APIs: X→Y, GetPhyAddr: X→Y)
Step 6: no pitfalls / {列出触发的}
```
