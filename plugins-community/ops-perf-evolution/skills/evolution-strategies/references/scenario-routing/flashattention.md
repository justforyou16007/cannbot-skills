# FlashAttention 优化工作流

> 适用：Flash Attention 及全部变体（MHA / GQA / MQA / MLA / Sparse / Paged / Sliding-Window / Fwd / Bwd 等）。
> FA 族策略收益与依赖均有序：**结构改写 > CV 流水 > 内存搬运 > mask/稀疏特化 > 调度均衡**。从未完成的最早阶段选策略，每轮落地一个阶段，不要跨阶段平铺堆叠。

## Step 0 — 族判定确认

命中 [README.md](README.md) 判定表第 1 行（QK^T+softmax+PV 主干 / online softmax 状态机 / head 数关系）即归本族。形态细分（GQA/MHA/MLA + 量化/稀疏/feature flags）见 [../families/flashattention_guide.md](../families/flashattention_guide.md)。

## Step 1 — 通用前置

打开 [_common-prerequisites.md](_common-prerequisites.md) 过 C1-C5。FA 族常见命中：**C1**（老版纯向量点积实现 → 直接转 S1 的 P120，它就是 P102 在 FA 的落地配方）、**C5**（softmax 必配 A5）。

## Step 2 — 族内阶段（S0→S6 按序推进）

| 阶段 | 目标 | 策略（按优先级） | 跳过条件 |
|---|---|---|---|
| S0 形态判定 | 确定基础形态（MHA/GQA/MLA）与正交 trait（稀疏/量化/decode/paged） | P115（Tiling 路由） | 每次必做，决定后续各阶段适用卡 |
| S1 结构改写 | 消除算法级浪费（收益最大） | P119（two-pass→online softmax）、P120（向量点积→cube mmad）、P124（MLA nope/rope 拼 K 维） | 已是 online softmax + cube 路径（如官方 FIA 移植实现）则整阶段跳过 |
| S2 CV 流水 | AIC/AIV 生产者-消费者重叠 | P121（SplitFuse 流水，依赖 P120）、P14（CV 预发射）、P125（跨任务预取，依赖 P121）、P29（跨核同步） | 仍在纯 Vector 路径时先回 S1 完成 P120 |
| S3 内存与搬运 | 摊薄握手/启动开销，L1/L2 常驻 | P122（KV 大块化 + L1 预算公式）、P18/P53（L1 常驻）、P38（softmax 状态常驻）、P60（decode KV NZ）、P61（L2 cache） | mte2_stall 非主瓶颈且 L1/UB 预算已达标 |
| S4 mask/稀疏特化 | 消除无效计算与搬运 | P123（causal 跳过 + 边界 mask 特化）、P118（mask 预生成独立 AIV）、P76/P77（sparse KV 预合并/聚合）、P80（Sink task 注入） | 非 causal/sparse/sink 形态 |
| S5 调度与负载均衡 | 多核利用率与变长调度 | P116（GQA G→M 合并）、P117（host MatmulApiTiling 配方）、P126（GM 任务表变长调度）、P57/P58（FlashDecode 分区） | tiling_imbalance 非瓶颈且核已打满 |
| S6 通用收尾 | 基础数据通路与数值稳定 | D1、P1、P5、P7、A4、A5、P90；Bwd 另加 P91 | — |

## 规则

1. 顺序不可颠倒：cube 路径（S1）→ CV 流水（S2）→ 大块化（S3）是硬依赖链，跳阶段会产生编译失败或负收益变体，浪费进化轮次。
2. 同轮组合必须过 [../strategy-compatibility.md](../strategy-compatibility.md) §1/§2 的依赖/互斥检查（如 P121 依赖 P120；P123 与逐元素 mask 兜底互斥须删除旧路径）。
3. Refine 阶段若 profiling 证据明确指向某瓶颈，以 strategy-index.md "按瓶颈类型查表"优先，本工作流顺序让位于证据。
