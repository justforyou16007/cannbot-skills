# P116 Playbook: GQA Grouped-M 头合并

> **强制流程**。本 Playbook 指导子 agent 把 GQA decode 场景（Sq≤4）的逐头小 matmul 改写为 G→M 维合并的单 task matmul。必须按 Step 1-5 逐步执行。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p116_locations.txt`：

```bash
# 1.1 GQA 维度参数：nq / nkv / G / sq
grep -nE "nq|nkv|numHeads|kvHeadNum|gSize|sq|seqLen" \
    shared/original/op_host/*.{cpp,h} > /tmp/p116_locations.txt
# 1.2 逐头循环 / 小 M matmul 证据
grep -nE "headsPerBlock|h_q|hidx|SetShape|SetOrgShape|IterateAll|IterateBatch" \
    shared/original/op_kernel/*.{cpp,h} shared/original/op_host/*.cpp >> /tmp/p116_locations.txt
# 1.3 layout 与行 stride
grep -nE "LAYOUT_|layout|rowStride|queryRowStride|stride" \
    shared/original/op_host/*.{cpp,h} shared/original/op_kernel/*.{cpp,h} >> /tmp/p116_locations.txt
# 1.4 mask 行数 / causal
grep -nE "causal|attenMask|maskRows|maskS2" \
    shared/original/op_host/*.{cpp,h} >> /tmp/p116_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，列出 G、Sq、当前 matmul 的 M/N/K 与行 stride。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| G = nq/nkv | ? | 确认 > 1 | op_host/*_tiling.cpp:L? |
| Sq | ? | 确认 ≤ 4（否则放弃 P116） | op_host/*_tiling.cpp:L? |
| 合并开关 | 无 | `groupedQueryM = sq<=4 && G>1` | op_host/*_tiling.cpp:L? |
| logicalB / logicalSq / logicalG | B / Sq / G | B*Sq / G / 1 | op_host/*_tiling.cpp:L? |
| queryRowStride | nq*d | d（合并时） | op_host/*_tiling.cpp:L? |
| layoutType | BSH | BNSD（合并时） | op_host/*_tiling.cpp:L? |
| mask 行数 | sq | sq * G（合并时） | op_host/*_tiling.cpp:L? |

## Step 3: 代码改造

### 3A. Host tiling 合并判定 + 维度重映射

```cpp
const int64_t originalG = nq / nkv;
const bool groupedQueryM = sq <= 4 && originalG > 1;
const int64_t logicalB  = groupedQueryM ? b * sq : b;
const int64_t logicalSq = groupedQueryM ? originalG : sq;
const int64_t logicalG  = groupedQueryM ? 1 : originalG;
const int64_t queryRowStride = groupedQueryM ? d : nq * d;
// layoutType / attenMaskS2Size / maskRows / task 总数同步切换
const int64_t totalTasks = logicalB * nkv * logicalG * s1Outer;
```

### 3B. Matmul OrgShape 用重映射后 stride

```cpp
bmm1.SetShape(MinHost(s1Base, logicalSq), bmm1N, d);
bmm1.SetOrgShape(logicalSq, s2Base * nRatio, queryRowStride, kvRowStride);
// bmm2 同理：A=P（行 stride 不变），B=V 用 kvRowStride
```

### 3C. mask 行扩展（配合 P118）

```cpp
const int64_t maskGroupSize = groupedQueryM ? originalG : 1;
const int64_t maskRows = isCausal ? sq * maskGroupSize : 0;
// mask kernel 内：queryRow = row / groupSize —— 同一 s 的 G 行内容相同
```

### 3D. 输出写回确认

确认输出 GM 偏移按原 BSH 语义计算（view 是纯重解释，输出无需 transpose）；若 kernel 内按 view 写出，核对 `b*sq + s` → 原 `b, s` 的映射。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P116 约束核查：
  1. G = ___, Sq = ___，满足 sq<=4 && G>1？ yes/no（no 则回退）
  2. G 个 query head 在内存中相邻（stride == d）？ yes/no，依据的布局 = ___
  3. logicalB/logicalSq/logicalG/queryRowStride 四处是否同步切换？ yes/no
  4. layoutType 是否切换为 BNSD？ yes/no
  5. mask 行数是否 ×G（sq*G）？ yes/no / 非 causal 说明
  6. 输出写回映射是否逐 case 核对？ yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 合并判定存在
(grep -cE "groupedQueryM|sq\s*<=\s*4" modified_files/op_host/*.cpp; true)
# 期望: >= 1

# 检查 2: 维度重映射存在
(grep -cE "logicalB|logicalSq|logicalG" modified_files/op_host/*.cpp; true)
# 期望: >= 3

# 检查 3: queryRowStride 切换
(grep -cE "queryRowStride\s*=\s*groupedQueryM\s*\?\s*d" modified_files/op_host/*.cpp; true)
# 期望: >= 1

# 检查 4: mask 行扩展
(grep -cE "maskGroupSize|maskRows.*\*\s*.*[gG]" modified_files/op_host/*.cpp; true)
# 期望: >= 1（非 causal 算子可豁免，需在 note 说明）

# 检查 5: OrgShape 使用新 stride
(grep -cE "SetOrgShape\(.*queryRowStride" modified_files/op_host/*.cpp; true)
# 期望: >= 1
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P116 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: grouped-M reshape applied (G=___, Sq=___)
Step 4: constraints checked (contiguity=___, mask_rows=___)
Step 5: all 5 grep checks passed
```

## 常见陷阱

- 非相邻布局（TND/head-major）强合并 → 读错数据，先验证 stride==d。
- Sq>4 仍合并 → task 数减少、核闲置，收益反转。
- mask 行数未 ×G → 越界或脏 mask，causal 精度全挂。
- 只改 layout 不改 SetOrgShape 行 stride → 整体错位。
- 与 P57 同时拆 G → 一个合一个拆，语义冲突；先 P116，核吃不饱再评估 P57。
