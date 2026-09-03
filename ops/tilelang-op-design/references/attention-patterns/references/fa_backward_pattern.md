# Attention FA Backward（FAG）Pattern

一句话：FA 反向模式处理的是"以前向统计量 + dy 重建 P 并反传 dq/dk/dv"的算子形态；它的独特性在于 **P 直接重建（非在线 softmax）、dS 双消费（dS 与 dS^T）、三段式 Pre/Main/Post 的 fp32 workspace 累加结构**，而不是前向的 online-softmax 流水。

## 先读这个

命中信号：输入含 `dy`（输出梯度）、`softmax_max`/`softmax_sum`（前向统计量）、`attention_in`（前向输出），输出为 `dq/dk/dv` 三个梯度张量。

本模式只负责：FA 反向的计算图组织与 CV 融合结构。布局（TND/Paged）、mask/causal、GQA 等仍由各专项模式文档决定——反向只是给它们套上了"梯度计算"的外壳。

参照实现：生产版 FAG 算子（arch22 / Atlas A2\A3 训练系列）S1s2Bn2gs1s2 泛化路径。

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

实测对照（Ascend910B2，8 用例）：泛化路径 vs 生产算子（按上表路由）
纯 kernel geomean 0.91x；vs torch 组合实现 2.20x；非常规 head dim
（D=177、D≠Dv=192/128）反超 1.2~2.0x。

## 4. 边界

- 本模式只覆盖 BNSD / keep_prob=1.0 / 无 pse / sparse_mode=0 / 无 rope 的
  最简反向形态；TND（varlen）、sink、rope、dScale 量化属 V4 接口的扩展参数域，
  分别按 tnd_pattern / sink_pattern 等专项文档组合。
- dropout（keep_prob<1）会在 dP/dV 路径插入 dropmask 生成与施加，不在本模式内。
- NZ 输出（mm12IsNZOut/mm345IsNZOut）仅在特定 s2/d 组合命中，泛化设计保持 ND 即可。

## 5. 扩展语义：sliding window 与 softcap（50 case 实证）

生产算子泛化路径不支持的两种常见变体，DSL 设计可原生包含（50 case 全覆盖验证）：

- **sliding window（window_left/right）**：mask 条件 `rel < -window_left` 或
  `rel > window_right`（`rel = col - (row + Skv - Sq)`），与 causal 可叠加；
  window_left/right = -1 表示该侧不限。
- **softcap**：`scores = softcap * tanh(scores_raw / softcap)`（softcap>0 时）；
  反传 dS 需乘 `(1 - tanh²)` 链式项。tanh 用 Vector 侧 Tanh API + 临时 buffer
  （host 侧无 GetTanhMaxMinTmpSize 声明时按 [2048]fp32 给足）。
- **设计规则：causal/window 的全 mask tile 跳过应进设计**（收益实测 c1 767→185us、
  v1 593→215us）。按 hi/lo 边界求 tile 与有效带区的相交，空相交则整个 tile 的
  C+V 计算全跳。注意生产算子 sparse_mode=0（全 tile 计算）路径无此技术——它来自
  生产 sparse 分支，是超越泛化路径的合法设计点。

## 6. 结构设计决策：阶段式拆分 vs CV 融合单入口（50 case 量化证据）

| 结构 | 特征 | 实测水位（稳态 event） |
| --- | --- | --- |
| 阶段式拆分（C1→V1→C2→post 多 launch） | launch 边界即全设备同步，Cube/Vector 严格互闲；同步简单、正确性优先 | 大 case vs torch 稳态 0.53x |
| CV 融合单入口（Pre→Main→Post） | mm1/mm(dP) 双发 + SubGraph ping-pong + V 侧 double buffer | vs torch 稳态 3.85x（50/50） |

决策指导：**首版用阶段式拆分换正确性是可接受的**（TileLang 设计表达期记录为
待验证项），但性能迭代必须以 CV 融合为目标结构（大 shape 收益 ~1.9x）。
配套设计项：wsS/wsDP 以 fp16 存储（fixpipe CAST_RINT 随路，流量减半且与 golden
舍入一致）；核间用连续条带切分（替代 task+=cores 轮询，改善 L2 局部性）。

## 7. 设计期精度契约（设计文档必须显式声明，实测取证）

1. **标量乘口径 per-dtype**：golden 中 `fp16 tensor × python float` 标量先舍入 fp16；
   `bf16 × float` 标量保持 fp32 opmath 后再舍入；fp32 全精度。host 侧 scale/softcap
   的舍入规则必须 per-dtype 声明（用错引入 ~2^-9(bf16)/2^-12(fp16) 系统偏差）。
2. **GQA 组内归约的舍入镜像**：fp16/bf16 golden 对 matmul 输出**逐头 RINT 实体化后
   再 group-sum**——设计 dk/dv 累加时，组内 L0C 直接累加（fp32 精度路径）会破坏镜像；
   要么 g-slab 按头实体化 + post 归约（fp16/bf16），要么仅 fp32 走 CAST_DV 直写。
3. **pad 区 NaN 免疫**：任何按字节取整装载的索引/数据，pad 尾部必须覆盖装载或显式
   清零——`at::empty` 垃圾（Inf/NaN 位型）经 `NaN*0=NaN` 会污染 pad 列参与的全宽
   GEMM；此类 bug 仅在连跑内存复现时暴露，设计期应消除而非依赖验证期发现。
4. **尾 k 块分形宽度**：GEMM 两侧分形条带宽度与 Mmad k 必须一致（如 A 侧固定 64，
   B 侧也必须固定 64），pad 来自 ZeroL1 清零区保证数学等价。

## 8. 基准对比的有效性边界（实测警示）

- **参照基准在其已验证 case 包络外的 shape 上可能静默出错**（实测：g=4 GQA +
  非 64 对齐 seq 尾块组合下基准实质性错误而 DSL 实现逐位一致）——以任何外部
  实现为性能基准前必须先做 golden 抽查，对比结论仅限基准已验证包络内。
- **quick 批量测量含输入构造**（~17ms 地板压缩比值至 ~1.0x），per-case 绝对值
  不可横向分析；性能判定以稳态 event（输入外置、warmup、多次平均、同次背靠背）
  为准，DVFS run 间波动 ±25%。
