# P119 Playbook: Two-Pass Softmax → Online Softmax 单遍改写

> **强制流程**。本 Playbook 指导子 agent 把两遍（或逐 token）softmax attention 改写为 online softmax 单遍 KV 流式结构。这是 L2 架构级改写，必须按 Step 1-5 逐步执行，严禁跳步。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p119_locations.txt`：

```bash
# 1.1 两遍结构证据（pass 注释 / 重复 KV 循环）
grep -nE "pass|PASS|for.*S_kv|for.*skv" \
    shared/original/op_kernel/*.{cpp,h} > /tmp/p119_locations.txt
# 1.2 逐 token 点积 / 标量累加证据
grep -nE "ReduceSum|WholeReduceSum|GetValue|SetValue|Kahan|kahan" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p119_locations.txt
# 1.3 K/V 加载位置（统计 GM 读取次数）
grep -nE "DataCopy|DataCopyPad" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p119_locations.txt
# 1.4 状态 buffer（max/sum/acc 当前形态）
grep -nE "max|sum|acc|Exp" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p119_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，明确：K 被读几次、V 被读几次、max/sum/acc 当前如何更新、是否含 Kahan。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| K 的 GM 读取次数 | 2（pass1+pass2） | 1 | op_kernel/*.cpp:L? |
| 行 max | 预扫描 | running max（-INF 初始化） | op_kernel/*.cpp:L? |
| 分母 sum | 第二遍累加 | `l = l*alpha + rowsum(p)` | op_kernel/*.cpp:L? |
| 分子 acc | 逐 token Kahan | `acc = acc*alpha + p@V`（fp32） | op_kernel/*.cpp:L? |
| Kahan 补偿 | 有 | 删除 | op_kernel/*.cpp:L? |
| 状态 buffer | 临时分配 | UB 常驻（配 P38） | op_kernel/*.cpp:L? |
| 全 mask 行 | 未处理 | l==0 → out=0 | op_kernel/*.cpp:L? |

## Step 3: 代码改造

### 3A. 状态初始化

```cpp
Duplicate(rowMax, -1e30f /* -INF */, s1Rows);   // 必须 -INF，exp(m_old - m_new)=0
Duplicate(rowSum, 0.0f, s1Rows);
Duplicate(acc,    0.0f, s1Rows * dAlign);       // fp32
```

### 3B. S2 单遍循环（7 步顺序不可乱）

```cpp
for (int64_t s2 = 0; s2 < skv; s2 += s2Base) {
    // 1) sTile = QK^T * scale（cube 或 vector，按当前实现能力）
    // 2) tileMax = RowMax(sTile); newMax = Max(rowMax, tileMax)
    // 3) p = Exp(sTile - broadcast(newMax))
    // 4) alpha = Exp(rowMax - newMax)
    // 5) rowSum = rowSum * alpha + RowSum(p)
    // 6) acc = acc * bcast(alpha) + Cast(p) @ V_tile
    // 7) rowMax = newMax
}
```

**顺序铁律**：先 rescale（5/6 的左项）再累加当前 tile（5/6 的右项）；7 必须最后。

### 3C. 删除 pass1 与 Kahan

整段删除 pass1 max 预扫描循环、逐 token Kahan 补偿（y/t/c 临时 buffer 一并回收）。

### 3D. 归一化兜底

```cpp
for (row : s1Rows) {
    float l = rowSum.GetValue(row);
    if (l > 0.0f) Muls(acc[row], acc[row], 1.0f / l, dAlign);
    else          Duplicate(acc[row], 0.0f, dAlign);   // 全 mask 行
}
```

### 3E. 验证顺序

先小 shape 非 causal → 小 shape causal（重点查前几个近退化行）→ 大 shape。每步与 golden 对比后再推进。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P119 约束核查：
  1. K 的 GM 读取次数 2→1？ yes/no，证据 = ___
  2. rowMax 初值 = -INF？ yes/no
  3. rescale 先于累加（acc*alpha + p@V）？ yes/no
  4. p 已减全局 running max（非 tile max）？ yes/no
  5. Kahan 已删除？ yes/no
  6. 全 mask 行 l==0 兜底？ yes/no
  7. causal 小 shape 前几行精度已验证？ yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: online 状态三件套
(grep -cE "rowMax|rowSum|acc" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 3

# 检查 2: rescale 因子
(grep -cE "alpha|Exp\\(.*rowMax.*-" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 3: pass1 / Kahan 已删除
(grep -cE "PASS 1|pass1|Kahan|kahan" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: == 0

# 检查 4: -INF 初始化
(grep -cE "NEG_INF|-1e30|-inf|FLT_MIN" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 5: 除零兜底
(grep -cE ">\\s*0\\.0f|!=\\s*0|l\\s*>\\s*0" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P119 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: online softmax applied (kv_reads=1, kahan_removed=yes)
Step 4: constraints checked (rescale_order=___, zero_guard=___)
Step 5: all 5 grep checks passed
```

## 常见陷阱

- rescale/累加顺序颠倒 → 结果整体错误。
- rowMax 用 0 初始化 → 第一块 acc 缩放错误。
- 保留逐 token Kahan → 流水被打碎，改写收益被吃掉。
- p 只减 tile max → 大 score 溢出 inf。
- 全 mask 行不兜底 → NaN 输出。
- 只改 softmax 不改 K/V 加载 → K 仍读两遍，账面收益拿不到。
