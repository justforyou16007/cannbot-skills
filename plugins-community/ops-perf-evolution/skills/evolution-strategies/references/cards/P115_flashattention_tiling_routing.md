---
source_key: evolution-strategies#card/P115_flashattention_tiling_routing
id: P115
title: FlashAttention Tiling 路由
doc_path: cards/P115_flashattention_tiling_routing.md
complexity: L1
op_families: [attention, flash_attention]
triggers: [flash_attention, tiling_routing, gqa, mha, mla]
references:
  - ops/ascendc-tiling-design/references/flashattention/patterns.md
  - ops/ascendc-tiling-design/references/flashattention/foundation/fundamentals.md
  - ops/ascendc-tiling-design/references/flashattention/foundation/roofline.md
  - ops/ascendc-tiling-design/references/flashattention/design/_governance.md
  - ops/ascendc-tiling-design/references/flashattention/subfamilies/base_design.md
bottlenecks: [tiling_mismatch, cube_underutilization, sync_overhead]
conflicts_with: []
synergizes_with: [P14, P18, P38, P55]
has_preconditions: true
has_playbook: true
---

# P115: FlashAttention Tiling 路由

## 核心思想

FlashAttention 类算子采用四层文档架构（L1 路由 → L2 地基 → L3 设计流 → L4 变体）。设计时先选基础形态（GQA / MHA / MLA），再叠加正交 trait（量化 / 稀疏 / feature flags），然后按节点 G→N12 推进，确保 Roofline、多核切分、内存划分、流水编排一致。

## 代码骨架

```cpp
// 示意：设计节点对应的关键决策映射
N1 宏观形态:   kernel type / UB mode / Cube block / pipe depth
N2 并行策略:   task-level vs split-KV reduce
N3 Roofline:   mEff / AI 可行域
N4 Softmax:    V1(max-trick) vs V2(online)
N6 多核切分:   task 维度 + 负载均衡
N7 内存划分:   UB / L1 / L0 / GM
N8 编译特化:   constexpr / macro / runtime-if 三分
N10 流水编排:  slot / SyncAll / HardEvent
```

## 关键修改点

1. **基础形态三选一**：默认 GQA；`G=1` 退化到 MHA；含 latent absorption 走 MLA。
2. **正交 trait 叠加**：量化（MXFP8/INT8/AntiQuant）、稀疏、PSE/RoPE/Sink 等 feature flag 按需加载。
3. **按节点推进**：必须 G→N12 顺序执行，不可跳过。
4. **组合可行性校核**：未验证组合（如量化 × 稀疏）需先在 `composition.md` 确认。
5. **常量语义分离**：跨核握手槽、L1 rotation 槽、V2 自读自写槽用不同常量。

## 常见陷阱

- ❌ 把 GQA / MHA / MLA 当作互斥子族 → 它们只是基础形态变化。
- ❌ 跳过 N3 Roofline 直接定 tile → 性能达不到理论极限。
- ❌ 未校核 trait 组合 → 量化 × 稀疏等组合可能未验证。
- ❌ 同一算子混用多种基础形态 → 违反架构约束。

## 代码搜索关键词

```bash
grep -nE "GQA|MHA|MLA|kvHeadNum|gMaxPerTask|splitKV|FlashAttention|online_softmax|SetFlag|WaitFlag" op_kernel/*.cpp op_host/*_tiling.cpp
```
