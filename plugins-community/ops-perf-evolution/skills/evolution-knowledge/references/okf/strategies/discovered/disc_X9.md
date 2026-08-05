---
schema_version: okf.v1
kind: discovered_strategy
id: X9
origin: discovered
title: Deferred Scaling — 反量化仿射结构的代数重排（延迟缩放）
description: 反量化 decode 类 kernel 的典型结构是： ``` 对 nTok*D 个元素逐一施加 per-token 缩放/偏移（D-wide Muls/Adds × nTok） → 然后马上被归约（ReduceSum / 加权累加）消费成低维量 ``` P94/P95 能把包裹这些向量指令的**标量外壳**向量化，但向量指令本身的数量 （2~3 × nTok 条 D-wide op）并未减少。本
confidence: single_op
status: active
op_families: [quantization, reduction]
bottlenecks: [bus_contention, l2_cache_thrash, mte2_stall, scalar_compute]
tags: [affine, bus_contention, dequant, l2_cache_thrash, mte2_stall, quantization, reduction, scalar_compute]
discovered_from: best_decode evolution round_1/parallel_3 (open_exploration x0, 910B, msprof)
base_speedup: 单核 aiv_time -33% (15.19→10.12us); task_duration 18.23→12.91us; whole-kernel asc_mean -5.7%
---
# Strategy X9: Deferred Scaling — 反量化仿射结构的代数重排（延迟缩放）

## 核心思路

反量化 decode 类 kernel 的典型结构是：

```
对 nTok*D 个元素逐一施加 per-token 缩放/偏移（D-wide Muls/Adds × nTok）
→ 然后马上被归约（ReduceSum / 加权累加）消费成低维量
```

P94/P95 能把包裹这些向量指令的**标量外壳**向量化，但向量指令本身的数量
（2~3 × nTok 条 D-wide op）并未减少。本策略更进一步：**利用反量化仿射/
乘性结构的分配律，把 per-element 缩放推迟到归约之后的低维量上做**，
在 fp32 等价语义下直接消掉指令：

- **K-fold（乘性）**：`score_t = s_t * sum(q · c_t)` —— K 块全程保持在
  未缩放单位（centroid/idx），per-token 因子 `s_t` 乘到 ReduceSum 之后的
  分数上（1 条 tb-lane Mul），而不是乘到全部 nTok*D 个 K 元素上
  （nTok 条 D-wide Muls）。
- **V-fold（仿射）**：`sum_t p_t*(idx_t*vs_t + vz_t) = sum_t (p_t*vs_t)*idx_t + (sum_t p_t*vz_t)` ——
  V 块保持原始 index 单位，2*nTok 条 per-token D-wide Muls/Adds 折叠为
  复合权重 `w_t = p_t*vs_t` 的 Axpy + 1 条 D-wide Adds（`sumPvz` 修正项）。
- **归约后低维量的派生计算也留在向量域**：如 softmax 权重 `w = p*vs`、
  `lAdd = ReduceSum(p)`、`sumPvz = ReduceSum(p*vz)` 全部走 vector pipe；
  Newton rsqrt 精化在 tb-lane 向量上执行（乘 2^-1 与 fp 舍入可交换，
  bit-exact），消除标量 GetValue/FLOP 链。

## 适用场景

- **算子类型**：量化 KV cache decode attention、dequant+reduce 融合、
  带 per-row scale/zero 的批量反量化。
- **触发条件**（需同时满足）：
  1. profiling 为 scalar-issue-bound / 发射受限（如 `aiv_scalar_ratio` > 30%，
     `aiv_mte2_ratio` < 15%），瓶颈是指令数而非带宽；
  2. 存在 per-token/per-row 的 `Muls(x[t*D], s_t)` / `Adds(x[t*D], z_t)` 循环，
     且其结果**只被归约或加权累加消费**（不被逐元素写出）；
  3. 缩放量与数据量满足仿射/乘性关系（`y = idx*s + z`、`y = s*c`）。

## 改造范式

### 范式 A：乘性缩放延迟（K-fold）

```cpp
// 基线：先缩放全部元素，再归约
for (t) { Muls(kBlk[t*D], kBlk[t*D], sArr[t], D); }   // nTok 条 D-wide
for (t) { Mul(prod[t*D], q, kBlk[t*D], D); ReduceSum(score[t], ...); }

// 改造：先归约（未缩放单位），再缩放低维结果
for (t) { Mul(prod[t*D], q, kBlk[t*D], D); ReduceSum(scoreRaw[t], ...); }
Mul(score, scoreRaw, sVec, tb);                        // 1 条 tb-lane
```

### 范式 B：仿射分解（V-fold）

```cpp
// 基线：逐 token 反量化 + 逐 token 加权
for (t) { Muls(vBlk[t*D], vBlk[t*D], vsArr[t], D);
          Adds(vBlk[t*D], vBlk[t*D], vzArr[t], D); }    // 2*nTok 条 D-wide
for (t) { Axpy(acc, vBlk[t*D], pArr[t], D); }

// 改造：sum_t p_t*(idx*vs+vz) = sum_t (p_t*vs)*idx + sum_t p_t*vz
Mul(wVec, pVec, vsVec, tb);                            // 复合权重（tb-lane）
for (t) { Axpy(acc, vIdxBlk[t*D], w_t, D); }           // V 保持 idx 单位
Adds(acc, acc, sumPvz, D);                             // 1 条修正项
```

### 范式 C：派生量全向量域（消除 V→S 往返）

`FetchSlotScalars` 类函数把 per-token fp16 标量 Gather+Cast 到 UB 后，
**不要** GetValue 回标量数组再喂给向量指令——让数据留在 UB 直接供
vector pipe 消费（配合块级 broadcast 展开）。热路径每 chunk
可消除 3*tb 次 V→S round trip；仅尾 chunk（stale lanes）按需现取。

## 关键前提与陷阱

⚠️ **fp32 等价性必须逐条论证**：分配律在 fp32 下对"缩放-归约"次序交换
通常是安全的（缩放在归约外只改变一次舍入位置），但需实测 rtol/atol +
全 case 验证；Newton rsqrt 类精化需确认每步与舍入可交换（如 ×2^-1）。

⚠️ **stale lanes 不能参与归约**：尾 chunk（nTok < tb）的 stale lanes 经
Exp 后是非零有限值，范式 B/C 的向量化归约只能在**满 chunk** 使用，
尾 chunk 保留标量路径（按 `nTok == tb_` 分派）。

⚠️ **负证据：勿用 Gather/LUT 路线替代本方案**。910B 的 UB Gather 是
per-lane 慢操作，用 gather 表查找替换廉价块算术实测回退 20-45%/case
（vector pipe 时间膨胀超过省下的发射槽）。本策略保持块算术、只做
代数重排，不引入 gather。（与 P100 的适用边界互补：P100 针对
SetValue/GetValue 解交织，不针对廉价块算术。）

⚠️ **不要动跨 chunk 的 online-softmax 标量状态**（`mPrev`/`lPrev`），
只向量化 chunk 内的派生量。

⚠️ **收益强依赖 splitKv 结构，须逐 case 验证回退条件**（2026-07-20
深度对拍实证）：case9（S=1）partial -47%，case1（S=3）中性，
case2（S=8 + normCorr）**回退 +40%**——chunks/unit 多时 Newton rsqrt /
w / pvz 的机械成本 > GetValue 消除收益。机械应用本策略前按 chunks/unit
评估派生量计算的实际消除率；后续"normCorr 阈值条件化"修复方向本身
也已实测封印（勿再尝试）。

## 实现步骤

1. 标记所有 per-token/per-row 缩放点（`Muls/Adds(x[t*D], s_t)`）；
2. 沿数据流确认其唯一消费者是归约/加权累加；
3. 按乘性（A）或仿射（B）写出代数等价式，重排缩放位置；
4. 归约后的低维派生计算（softmax 权重、rsqrt 精化）改 tb-lane 向量 op；
5. 满/尾 chunk 分派：满 chunk 走向量化路径，尾 chunk 保留标量路径；
6. 全 case 精度验证（建议 ≥3 轮，关注归约次序改变带来的尾差）。

## 预期收益

- 直接消除 2~3×nTok 条 D-wide 向量指令 + 配套标量操作数准备；
- 实证（best_decode，910B，TB=8，D=128，MSE 反量化 decode）：
  单核 aiv_time **-33%**，task_duration -29%，12/12 精度 matched
  （MERE ≤ 1.2e-6，阈值 9.8e-4）。

## 与现有策略的关系

- **P94/P95**：标量归约/逐元素 → 向量 API（指令数不变）；本策略在此基础上
  **减少向量指令本身**。二者串联使用：先本策略代数重排消指令，残余标量
  外壳再按 P94/P95 处理。本策略处理的是"已经是向量 API
  但数量可代数消减"的场景。
- **P100**：Gather 替换逐元素读写；本策略的负证据限定了 P100 类路线在
  "廉价块算术 + scalar-issue-bound"场景下会回退。
- **X4**：Host 预计算常量表；与本策略正交，可叠加。

## 组合实测（best_decode 完整优化链）

本策略的成果（round_1/parallel_3，149.58us）作为后续进化的基线，与其余
validated X 策略逐层正交叠加，构成完整优化链：

```
X5 延迟缩放基线 149.58us
 → +X4（host Pi 转置缓存，R1 主线）        125.64us (-16.0%)
 → +tail 向量化/WholeReduce 归约重构        115.93us (-22.5%)
 → +X3（TB=16 UB 预算化，x1）              114.71us (-23.3%)
 → +X2（reduce 融合）= merge_final         99.53us  (-33.5%, geomean 43.34x)
```

四张 validated X 卡（X2/X3/X4/X5）收益均经同窗口 asc 口径实测且互相正交，
可叠加；叠加时注意 X2 新增 buffer 须计入 X3 的 PickTb UB 预算公式
（超预算回退 TB=8，merge_final 冲突裁决记录）。

## 代码搜索关键词

```bash
# per-token 缩放后接归约（范式 A/B 候选）
grep -nE "for.*\{[^}]*Muls\s*\([^,]*\[t\s*\*" op_kernel/*.cpp

# 反量化 scale+zero 循环（范式 B 候选）
grep -nE "Muls.*vs|Adds.*vz|scale.*zero" op_kernel/*.cpp

# Gather+Cast 后 GetValue 回读标量数组（范式 C 候选）
grep -nE "GetValue\(t\)|GetValue\(16|GetValue\(32" op_kernel/*.cpp
```

## 来源

发现于 best_decode 进化 round_1/parallel_3（open_exploration，2026-07-17）：
- K-fold：MSE centroid 反量化的 `s_t` 延迟到 score 之后；
- V-fold：`idx*vs+vz` 仿射分解 + 复合 Axpy 权重；
- softmax 权重向量化 + FetchSlotScalars GetValue 死代码消除；
- 负证据：LUT/gather 解表路线实测回退 20-45%/case 后放弃。
