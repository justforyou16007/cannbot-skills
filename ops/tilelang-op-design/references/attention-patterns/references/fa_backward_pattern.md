# Attention FA Backward（FAG）Pattern

一句话：FA 反向模式处理的是"以前向统计量 + dy 重建 P 并反传 dq/dk/dv"的算子形态；它的独特性在于 **P 直接重建（非在线 softmax）、dS 双消费（dS 与 dS^T）、三段式 Pre/Main/Post 的 fp32 workspace 累加结构**，而不是前向的 online-softmax 流水。

## 先读这个

命中信号：输入含 `dy`（输出梯度）、`softmax_max`/`softmax_sum`（前向统计量）、`attention_in`（前向输出），输出为 `dq/dk/dv` 三个梯度张量。

本模式只负责：FA 反向的计算图组织与 CV 融合结构。布局（TND/Paged）、mask/causal、GQA 等仍由各专项模式文档决定——反向只是给它们套上了"梯度计算"的外壳。

参照实现：`output/flash_attention_score_grad_archive/`（生产 arch22 S1s2Bn2gs1s2 泛化路径 re-host + design/ 设计复刻，8/8 精度 PASS，实测性能画像齐备）。

## 1. 识别条件

- 输入签名含 `dy, softmax_max, softmax_sum, attention_in`（或 `softmax_lse` 变体），输出 `(dq, dk, dv)`。
- `softmax_max/sum` 形状 `(B, Nq, Sq, 8)` fp32，只用 `[..., :1]`——存在即反向的强信号。
- 与前向的关键区别：**反向不做在线 softmax**，直接用前向统计量重建 P，无 running max/sumexp 状态。

## 2. 核心规则

### 计算流程（6 步，5 个 matmul）

```text
S  = Q @ K^T * scale           (Cube)
P  = exp(S - softmax_max) / softmax_sum    (Vector，mask 位置置零)
dP = dy @ V^T                  (Cube)
sfmg = rowsum(dy ∘ attention_in)           (Vector，FlashSoftmaxGrad)
dS = P ∘ (dP - sfmg) * scale   (Vector)
dq += dS @ K ; dk += dS^T @ Q ; dv += P^T @ dy   (Cube，fp32 累加)
```

### 三段式结构（Pre / Main / Post）

```text
Pre : 清零 fp32 workspace（dq/dk/dv 累积区；fp32 dtype 时直接清 dvGm）
Main: 上述 6 步的 CV 融合主计算（核间 BN2GS1 合并轴，核内 S1/S2 切分，
      CV 双子图 ping-pong + Cube 双发 preload）
Post: fp32 workspace → Muls(scale) + Cast → 按 numel 全量写回 dq/dk/dv
```

关键规则：

- **转置全部由 mma transpose 标志消化**（K^T、V^T、dS^T、P^T），禁止显式物化转置张量。
- **dS 双消费**：dq 用 dS、dk 用 dS^T——同一份 dS workspace 被两个 mma 以不同转置位读取。
- **fp32 workspace 累加**：Cube 输入 fp16/bf16、L0C/GM workspace fp32；Post 才 cast 回输出 dtype。这是精度契约（fp16 容差 2e-2 实测余量两个数量级）。
- **输出全量覆写 → host 可 `at::empty`**：Post 的写回范围恰好铺满 numel（qPostBlockTotal == qSize == b·n1·s1·d），host 侧 `at::zeros` 是纯冗余（实测 3 个 ZerosLike kernel 占 device 时间 11.3%）。fp32 特例：dv 由 Pre 清 dvGm + Main 直写，Post 跳过（CAST_DV 条件）。
- **GQA**：dk/dv 按 kv head（N2）输出，G 组内归约经 workspace 多核/多块累加；golden 侧以 `repeat_interleave` + `view-sum` 对齐语义。
- **mask 语义**：`atten_mask` bool，**True 表示屏蔽**；causal 用上三角 mask 表达（对应生产 sparse_mode=0 + 显式 mask，而非 next_tockens 稀疏路径）。

### CV 配比与流水（A2 训练系列）

- AIC:AIV = 1:2；CV 数据量配比 1:16（Cube 基本块 128×1024 fp16 vs Vector 8×1024 fp32 预算；S1 nRatio=1、S2 ratio=8 组成）。
- Cube 双发（CV preload）：C1 同窗口连发 mm(S) 与 mm(dP)，覆盖 V1 的 Vector 计算时长；SubGrapA/B 双通道 ping-pong 隐藏 CV 交互。
- sfmg 的 D 方向切分（sfmgd outer/inner/tail）：大 value_d 时多轮累加。

## 3. 模板路由（生产多模板体系，设计侧复现的价值点）

泛化模板（S1s2Bn2gs1s2）支持所有 shape 但小 shape 结构成本高
（实测 aic scalar 75~93%、mac 仅 2~12%、核间不均最高 31.8%）：

| 条件 | 路由 | 收益来源 |
| --- | --- | --- |
| S1<1024 && S2<1024 && (B·N2·2>CoreNum \|\| G==1) | S1s2Bn2（核间 B/N2） | 小 S 减少 CV 交互 |
| G·S1·S2≤32KB && G==1 && S2<1536 | Ngs1s2Bn | 核内切 N2.i 加大 CV 基本块 |
| N2·G·S1·S2≤64·128 | Bngs1s2B | 极小内层，B.i 入 CV 块 |
| S1≥1024 or S2≥1024 | SameAB（Cube 主核） | V0/V1 复用左矩阵 + L0C 累加 |
| 其他 | S1s2Bn2gs1s2（泛化兜底） | 全 shape 可用 |

实测对照（Ascend910B2，8 用例归档）：泛化路径 vs 生产算子（按上表路由）
纯 kernel geomean 0.91x；vs torch 组合实现 2.20x；非常规 head dim
（D=177、D≠Dv=192/128）反超 1.2~2.0x。

## 4. 边界

- 本模式只覆盖 BNSD / keep_prob=1.0 / 无 pse / sparse_mode=0 / 无 rope 的
  最简反向形态；TND（varlen）、sink、rope、dScale 量化属 V4 接口的扩展参数域，
  分别按 tnd_pattern / sink_pattern 等专项文档组合。
- dropout（keep_prob<1）会在 dP/dV 路径插入 dropmask 生成与施加，不在本模式内。
- NZ 输出（mm12IsNZOut/mm345IsNZOut）仅在特定 s2/d 组合命中，泛化设计保持 ND 即可。
