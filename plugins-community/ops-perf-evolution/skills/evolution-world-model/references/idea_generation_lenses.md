# 多镜头 Idea 生成指南

本文档定义了演化系统中 6+1 种 idea 生成镜头（lens），供 Agent 在 Init、Refine 和 Supervisor 介入阶段使用。

## 总览

| Lens | 驱动方式 | 使用时机 | 典型 idea 数量 |
|------|---------|---------|--------------|
| `profiling_driven` | 确定性 + LLM | 有 profiling evidence 时 | 2-3 |
| `method_transfer` | 确定性 + LLM | 算子特征已知时 | 3-5 |
| `gap_analysis` | gap_map 驱动 | 每轮 | 2-5 |
| `continuation` | 实验结果驱动 | Refine 阶段（passed exp） | 2-3 |
| `open_exploration` | 纯 LLM | 填充剩余槽位 / 停滞破局 | 1-2 |
| `workflow_guided` | workflow 文件 | 归族命中时 | 1 |
| `supervisor_sourced` | Supervisor 外部复核 | Supervisor 介入后 | ≤3 |

## 执行优先级

1. **Init 阶段**：profiling_driven → method_transfer → gap_analysis → workflow_guided → open_exploration
2. **Refine 阶段**：continuation → profiling_driven → gap_analysis → open_exploration
3. **停滞破局**：提高 open_exploration 占比到 ceil(n/2)
4. **Supervisor 介入**：将 Supervisor 返回的 MCTS 形状 `new_nodes[]` 映射为此 lens 的 KG ideas

---

## Lens 1: profiling_driven — Profiling 驱动

**输入**：
- `baseline_evidence` 或 experiment 的 `profiling_evidence`
- `BOTTLENECK_STRATEGY_MAP`（profiling_evidence.py）

**确定性路径**：
```bash
python3 idea_generation.py generate <evo_dir> \
  --baseline-evidence <evidence.json> --parallel-num 3
```

**LLM 补充**：
Agent 读取 profiling 数据后，从以下角度细化：
- 空泡模式（periodic / sporadic / tail_dominant）是否暗示特定策略？
- DMA 效率数据是否暗示搬运粒度调整？
- 是否有多个瓶颈同时存在（需要组合策略）？

**输出 idea 示例**：
```yaml
title: "针对 mte2_stall 瓶颈的主要优化: P1, P10"
strategy_tags: [P1, P10]
mode: strategy_guided
generation_lens: profiling_driven
thesis: "基线 profiling 显示 MTE2 D类空泡占比 62%，双缓冲 + 向量化搬运可隐藏延迟"
```

---

## Lens 2: method_transfer — 方法迁移

**输入**：
- 算子特征（memory_bound, compute_bound, reduction, broadcast, elementwise, has_tail, multi_core）
- strategy-index.md 中策略的 "When to Apply" 列

**确定性路径**：
trait → strategy 映射表（内置于 idea_generation.py）

**LLM 补充**：
Agent 读取 strategy-index.md，判断：
- 该算子的计算模式是否匹配某策略的适用条件？
- 有哪些跨领域策略可以迁移（如 conv 领域的策略用于 attention）？
- 是否有策略的前置条件已被满足但尚未尝试？

**输出 idea 示例**：
```yaml
title: "方法迁移 (memory_bound): P25, P66"
strategy_tags: [P25, P66]
mode: strategy_guided
generation_lens: method_transfer
thesis: "算子为内存密集型，DataCopyPad 对齐 + 512B 对齐可减少带宽折损"
```

---

## Lens 3: gap_analysis — Gap 分析驱动

**输入**：
- `gap_map.md`（open gaps 按 priority 排序）

**确定性路径**：
```bash
python3 gap_analysis.py analyze <evo_dir>
```
从 gap 描述中提取策略 ID → 生成 idea

**LLM 补充**：
Agent 读取 gap_map.md，对每个 open gap：
- 这个 gap 暗示什么优化方向？
- 是否需要 open_exploration（gap 无明确策略时）？
- gap 的 evidence 是否指向特定的策略组合？

**输出 idea 示例**：
```yaml
title: "Gap G3: P2, P4 优化"
strategy_tags: [P2, P4]
mode: strategy_guided
generation_lens: gap_analysis
target_gaps: [G3]
thesis: "tiling_imbalance 瓶颈的主要策略 P2+P4 尚未组合测试"
```

---

## Lens 4: continuation — 成功实验深化

**输入**：
- passed experiment 的 profiling_evidence
- 父 idea 的 strategy_tags
- BOTTLENECK_STRATEGY_MAP

**确定性路径**：
```bash
python3 idea_generation.py from_experiment <evo_dir> \
  --experiment-slug <slug> --parallel-num 3
```

**LLM 补充**：
Agent 分析实验结果后：
- 残余瓶颈是什么？需要叠加什么策略？
- 哪些正交方向（双缓冲、分块、精度）可以叠加？
- 是否达到了某个瓶颈的天花板，需要转换方向？

**输出 idea 示例**：
```yaml
title: "深化 r1-p0: 叠加 P10, P25 解决残余 mte2_stall"
strategy_tags: [P1, P7, P10, P25]
mode: strategy_guided
generation_lens: continuation
parent_experiment: exp:r1-p0
thesis: "在 2.3x 基础上解决残余 mte2_stall (35%) 瓶颈"
```

---

## Lens 5: open_exploration — 开放探索

**输入**：
- 内核源代码
- profiling 数据
- query_pack.md（包含失败 ideas 禁忌表和已验证 claims）

**确定性路径**：
生成占位 idea（strategy_tags=[], mode=open_exploration），由 LLM 在生成阶段填充

**LLM 推理要求**：
- **禁止读取策略库**（strategy-index.md）
- 从最优内核代码 + profiling 数据出发
- 第一原理推理：阅读代码、分析数据流、发现优化机会
- 参考 query_pack.md 中的禁忌表，避免重复已失败方向
- 参考 query_pack.md 中的 claims，利用已验证知识

**输出 idea 示例**：
```yaml
title: "循环展开 + SIMD 手动调度"
strategy_tags: []
mode: open_exploration
generation_lens: open_exploration
thesis: "分析内核循环结构发现：内层循环体 <32B，可通过 4x 展开减少循环开销 ..."
```

---

## Lens 6: workflow_guided — 归族 Workflow 保底

**输入**：
- scenario-routing 归族判定结果
- 族 workflow 文件（convolution.md / matmul.md / flashattention.md / reduction.md）

**使用条件**：
Init 归族命中某族 workflow 的任一分支时必须生成至少 1 个 workflow_guided idea。

**输出 idea 示例**：
```yaml
title: "Conv C-6 分支: P128, P129"
strategy_tags: [P128, P129]
mode: workflow_guided
generation_lens: workflow_guided
workflow_ref:
  family: conv
  workflow_file: convolution.md
  branch: C-6
  branch_condition: conv3d 三维滑窗
  traits: []
  stage_order: ["_common-prerequisites.md C1-C5", "C-6 branch", "GEMM 收尾"]
thesis: "归族命中 conv C-6 分支，按阶段顺序优化"
```

---

## Lens 7: supervisor_sourced — Supervisor 外部复核

Supervisor 的返回格式为了兼容 MCTS 仍使用 `new_nodes[]` 和
`strategy_combination` 字段。KG 编排器通过 `kg_ops.py supervisor` 将每个
node 映射为普通 idea，保留 description/rationale、策略组合、优化类型和
difficulty，并将 `generation_lens` 固定为 `supervisor_sourced`。

该 lens 只表示来源，不赋予 idea 更高的确定性；新 idea 仍必须经过正常的
LLM jury、并行实现、精度和性能验证。编排器同时通过
`parent_experiment`/`parent_idea` 建立 `derived_from`/`inspired_by` 边，保证
Supervisor 建议与触发它的证据可追溯。

---

## 去重和质量门控

### 去重规则

1. **Jaccard 碰撞**：strategy_tags 的 Jaccard 相似度 > 0.6 视为碰撞
2. **禁忌表过滤**：与 banlist 中失败 idea 的 Jaccard > 0.6 直接过滤
3. **批内去重**：同一轮生成的 ideas 之间也做碰撞检测

### 数量保障

- Init 阶段：生成 parallel_num × 2 个 idea（前两轮不缺 proposed ideas）
- Refine 阶段：每个 passed experiment 生成 2-3 个新 idea
- 最低保障：若所有 lens 产出不足，用 open_exploration 占位补齐

### 多样性保障

- bandwidth / tiling / algorithm 三类各至少 1 个 idea
- 至少 1 个 open_exploration idea（除非总数 < 4）
- 归族命中时至少 1 个 workflow_guided idea
