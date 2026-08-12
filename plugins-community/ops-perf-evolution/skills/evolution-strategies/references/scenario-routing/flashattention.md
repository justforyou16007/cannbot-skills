# FlashAttention 优化工作流（MHA / GQA / MLA 分支 + 正交 trait）

> 适用：**QK^T → softmax → PV** 的 softmax 注意力及其 head 结构变体（MHA / GQA / MQA / MLA），以及正交 trait（causal / sparse / paged / decode / 量化 / phase）。
> ⚠️ 迁移注记：旧 S0 形态判定已并入 Step 2 分支路由表；历史观测文档若引用 S0-S6 阶段名，以本文 Step 2-4 为准。
> FA 族策略收益与依赖均有序：**结构改写 > CV 流水 > 内存搬运 > mask/稀疏特化 > 调度均衡**。从未完成的最早阶段选策略，每轮落地一个阶段。

## Step 0 — 族判定确认

命中 [README.md](README.md) 判定表第 1 行（QK^T 指纹 / online softmax 状态机 / head 数关系）即归本族。
注意：**通道注意力**（SE/SK/Triplet：global pool→FC→scale，无 QK^T）**不归本族**（落 reduction/elementwise）；**无 QK^T 的递归/scan**（GDN-KDA/GLA）归 [ssm-scan.md](ssm-scan.md)。

## Step 1 — 通用前置

打开 [_common-prerequisites.md](_common-prerequisites.md) 过 C1-C5。FA 族常见命中：**C1**（老版纯向量点积实现 → 直接转 S1 的 P120，它就是 P102 在 FA 的落地配方）、**C5**（softmax 必配 A5）。

## Step 2 — 分支路由（先判 head 结构，按序路由）

| 分支 | 判定条件 | 专用路径 | 跳过条件 |
|---|---|---|---|
| standard | `num_q_heads == num_kv_heads`（MHA/self/cross） | 基础链 S1（P119/P120）→ S2 → S3 → S4/S5 | 已是 online softmax + cube 路径则跳过 S1 |
| gqa | `num_q_heads == G×num_kv_heads (G>1)`；`num_kv_heads==1` → MQA（G=num_q 特例） | P115（GQA tiling 路由）+ 基础链；S5 用 **P116**（G→M 合并）+ **P126**（变长调度）；decode 侧重 **X5/X4** | 非 grouped head |
| mla | KV 经 latent c 压缩（`c→[nope\|rope]→K/V` 升维 matmul），decode 特征 | S1 **P124**（nope/rope 拼 K 维）+ P120 + **P140**（kvHeads=1 M-tile 放大到 qHeads）；S5 P126；decode 侧重 P57/P58 | 非 latent 结构 |

> ⚠️ 小 S launch-bound 优先 host 侧（P54/X28），勿机械套 tiling/流水（P-GQA-7 教训）。

## Step 3 — 基础链（S1→S6 按序推进）

| 阶段 | 目标 | 策略（按优先级） | 跳过条件 |
|---|---|---|---|
| S1 结构改写 | 消除算法级浪费（收益最大） | P119（two-pass→online softmax）、P120（向量点积→cube mmad）、P124（仅 mla 分支） | 已是 online softmax + cube 路径则整阶段跳过 |
| S2 CV 流水 | AIC/AIV 生产者-消费者重叠 | P121（SplitFuse 流水，依赖 P120）、P14（CV 预发射）、P125（跨任务预取，依赖 P121）、P29 | 仍在纯 Vector 路径时先回 S1 完成 P120 |
| S3 内存与搬运 | 摊薄握手/启动开销，L1/L2 常驻 | P122（KV 大块化 + L1 预算）、P18/P53（L1 常驻）、P38（softmax 状态常驻）、P60（decode KV NZ）、P61（L2 cache） | mte2_stall 非主瓶颈且 L1/UB 预算已达标 |
| S4 mask/稀疏特化 | 消除无效计算与搬运 | P123（causal 跳过 + 边界 mask）、P118（mask 预生成独立 AIV）、P76/P77（sparse KV 预合并）、P80（Sink task） | 非 causal/sparse/sink 形态 |
| S5 调度与负载均衡 | 多核利用率与变长调度 | P116（GQA G→M 合并）、P117（host MatmulApiTiling 配方）、P126（GM 任务表变长调度）、P57/P58（FlashDecode 分区） | tiling_imbalance 非瓶颈且核已打满 |
| S6 通用收尾 | 基础数据通路与数值稳定 | D1、P1、P5、P7、A4、A5、P90；Bwd 另加 P91 | — |

## Step 4 — 正交 trait 追加表（基础链之上叠加；phase 切换阶段组，其余不重排链序）

| trait | 判定条件 | 追加卡 | 跳过条件 |
|---|---|---|---|
| phase=fwd（默认） | 正向传播 | 走正向基础链 S1-S6 | — |
| phase=bwd | 反向传播（FlashAttentionBwd / SDPA-Bwd / BlockSparseAttnBwd） | **切换阶段组**为反向链：重算 softmax P → dS=dO@V^T、dV=P^T@dO → dP=softmax 反向 → dQ=dS@K、dK=dS^T@Q → P91；mask/稀疏 trait 照常叠加 | 非反向算子 |
| causal | 下三角 mask | S4 P123 + P118 | 全 attention 无 mask |
| sparse:block_table | block_table/block_size 分块稀疏加载（SFA/FlashSparse 形态，7_SparseFlashAttention） | **P137**（紧凑化走 Cube：top-k 经 AIV gather 搬成紧凑 tile 再喂 AIC mmad，不再回退 Vector）+ P76/P77 + 稳定性 **P138**（L0 容量守卫）/ **P139**（多槽零初始化） | 稠密 attention |
| sparse:causal-window | band/窗口/stride mask skip | S4 P123 + P118 | 稠密 attention |
| paged/decode | Q len=1 + paged KV cache | S3 P60 + P61；S5 P126 + P57/P58 | prefill 全量 |
| 量化 | dtype∈{int8/fp8}、有 scale | 转 [quantization.md](quantization.md) 支路（P48/P49/P70），量化是 trait 不换族 | 无量化 |

## 规则

1. **分支 + trait 顺序**：先 Step 2 定分支（head 结构），再 Step 3 走基础链，最后 Step 4 按命中 trait 追加策略；`workflow_ref.stage_order` = 基础链 + branch 专属 + traits 追加。
2. **顺序不可颠倒**：cube 路径（S1）→ CV 流水（S2）→ 大块化（S3）是硬依赖链，跳阶段会产生编译失败或负收益变体。
3. **同轮组合必须过 [../strategy-compatibility.md](../strategy-compatibility.md) §1/§2** 的依赖/互斥检查（如 P121 依赖 P120；P123 与逐元素 mask 兜底互斥须删除旧路径）。
4. **workflow 分支首步读 baseline profiling 判定瓶颈域**：launch-bound → host 侧（P54/X28）；compute-bound 长 S → S2/S5 流水深化（P-GQA-7/MQA 观测教训）。
5. **Refine 阶段若 profiling 证据明确指向某瓶颈**，以 strategy-index.md "按瓶颈类型查表"优先，本工作流顺序让位于证据。
