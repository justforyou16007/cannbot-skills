---
name: tilelang2ascend-tiling-sink
description: |
  动态 seq-len serving tiling 下沉（策略 X69）部署形态参考。提供成本门判据、
  producer AIV/AICPU 选型、aclgraph 图重放三原则，以及 TileLang 各阶段
  （设计/转译/调优）能做与不能做清单。
  Use when（仅显式激活，默认流程不自动触发）:
  - 用户显式要求使用 tiling 下沉 / metadata 算子下沉 / device-side tiling
  - 用户要求动态 seq-len（varlen/TND/actual_seq_lens）算子做零 D2H / 图重放兼容的部署形态优化
---

# 动态 seq-len tiling 下沉（X69）

本 skill 是**显式按需**的部署形态参考，不参与 tilelang2ascendc 默认设计/转译/调优流程：
designer、translator、tilelang-perf-optimization 的路由与强制门禁都不指向这里，
仅当用户明确提出 tiling 下沉类需求时才使用。

## 使用方式

1. 读 [references/dynamic_seqlen_tiling_sink.md](references/dynamic_seqlen_tiling_sink.md)——
   先核对"适用特征"，再过**成本门**（AICPU producer 固定开销 130-317µs + 双 aclnn 串行
   150-190µs，主 kernel ≳500µs 的 kernel-bound 负载才净赚；launch-bound 净亏），
   再按阶段查"各阶段动作"表。
2. 成本门不达标 → 如实告知不适用，不套用此形态。
3. producer 算子（AICPU/AIV metadata 前置算子）与双 aclnn 两段式属**集成层**
   （vllm-ascend / 算子仓侧），不在 TileLang / AscendC 生成物内。

## 边界

- 本 skill 自包含完整判据（成本门/形态选择/各阶段动作），证据出处见卡内
  "证据与延伸"。
- 不修改默认流程：不要在 designer/translator/perf 的产物中主动引用本 skill，
  除非用户显式要求。
