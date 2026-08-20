# 知识图谱 Schema 定义

本文件定义了演化知识图谱的完整数据结构。知识图谱以文件形式持久化于磁盘：

```
output/{op_name}_evo_{timestamp}/
  knowledge_graph.json    # 注册表 + 计数器 + session 锚定
  ideas/                  # 每个 idea 一个 .md 文件
  experiments/            # 每个 experiment 一个 .md 文件
  claims/                 # 每个 claim 一个 .md 文件
  graph/
    edges.jsonl           # typed edges（关系图谱）
  gap_map.md              # 知识缺口地图
  query_pack.md           # 压缩上下文（≤8000 chars）
  log.md                  # 追加式审计时间线
```

---

## 顶层结构（knowledge_graph.json）

```json
{
  "kernel_summary": "<字符串> 算子功能、计算模式、已知约束的一段描述（1-3句话）",

  "session": {
    "session_id": "ai_infra_sparse_flash_attention_gqa_ops-evo_20260430_202933",
    "start_time": "2026-04-30T20:29:33+0800",
    "requested_rounds": 5,
    "actual_rounds_completed": 3,
    "evo_dir": "output/ai_infra_sparse_flash_attention_gqa_ops-evo_20260430_202933",
    "op_name": "ai_infra_sparse_flash_attention_gqa"
  },

  "baseline_performance": {
    "speedup": 1.0,
    "time_ms": null
  },

  "hw_params": null,
  // 目标芯片硬件规格（由 npu-arch skill 在步骤3.5-HW 写入）
  // 非 null 时结构：{ chip_model, ub_size_bytes, core_num, peak_bw_gbps, ... }

  "baseline_evidence": null,
  // 基线 profiling 的硬映射证据（可选，根级字段）
  // 结构与 experiment 的 profiling_evidence 完全一致

  "best_score": 1.0,
  "stagnation_count": 0,
  "stagnation_count_vs_base": 0,
  "supervisor_used_count": 0,
  "last_supervisor_round": -2,
  "open_questions": [],
  "discovered_strategies": [],

  "idea_registry": {
    "<idea_slug>": {
      "stage": "proposed",
      "outcome": "pending",
      "utility_score": 14.5,
      "generation_lens": "profiling_driven",
      "round_created": 1,
      "round_tested": null
    }
  },
  // idea_registry: 轻量注册表，与 ideas/*.md 保持同步
  //   stage: proposed | active | piloted | archived
  //   outcome: unknown | pending | negative | mixed | positive
  //   utility_score: 最近一次 select 时计算的效用分
  //   generation_lens: 生成此 idea 的镜头
  //   round_created: 创建轮次
  //   round_tested: 被选中测试的轮次（null = 尚未测试）

  // KG Supervisor 运行时状态：与 legacy MCTS 使用相同的初始值和
  // cooldown/cap 语义；通过 kg_ops.py supervisor 读写，不由 Agent 手改。

  "counters": {
    "total_ideas": 12,
    "active_ideas": 4,
    "total_experiments": 8,
    "total_claims": 3,
    "rounds_completed": 2
  }
}
```

---

## Idea 实体结构（ideas/<slug>.md frontmatter）

```yaml
---
type: idea
node_id: idea:<slug>
title: "双缓冲 + 512B 对齐消除 MTE2 停顿"
stage: proposed        # proposed → active → piloted → archived
outcome: pending       # unknown → pending → negative → mixed → positive
strategy_tags: [P1, P7]
optimization_type: bandwidth
difficulty: 3
mode: strategy_guided  # strategy_guided | open_exploration | profiling_driven | workflow_guided
workflow_ref: null      # workflow_guided 时必填，结构见下文
parent_idea: null       # 启发此 idea 的前序 idea node_id
parent_experiment: null # 触发此 idea 的实验结果 node_id
generation_lens: profiling_driven  # 生成此 idea 的镜头
generation_round: 1
target_gaps: [G1, G3]   # 此 idea 针对的知识缺口
tags: [memory_bound, double_buffer]
added: 2026-08-19T...
---
```

### Idea 字段约束

| 字段 | 类型 | 约束 |
|------|------|------|
| `stage` | string | proposed / active / piloted / archived |
| `outcome` | string | unknown / pending / negative / mixed / positive |
| `strategy_tags` | string[] | P/D/A/R/X 系列策略 ID；open_exploration 为空数组 |
| `optimization_type` | string | bandwidth / tiling / algorithm / register_opt / vf_fusion / instruction_sched |
| `difficulty` | int | 1-5，失败后 +1，初始不用 5 |
| `mode` | string | strategy_guided / open_exploration / profiling_driven / workflow_guided |
| `workflow_ref` | object 或 null | workflow_guided 必填：{family, workflow_file, branch, branch_condition, traits, stage_order} |
| `generation_lens` | string | profiling_driven / method_transfer / gap_analysis / continuation / open_exploration / workflow_guided / supervisor_sourced |
| `target_gaps` | string[] | gap_map.md 中的 G1, G2, ... 引用 |

### mode 取值说明

| mode | 含义 |
|------|------|
| `strategy_guided` | 默认：使用 strategy_tags 中的策略 |
| `open_exploration` | 开放探索：禁止读取策略库，LLM 自主推理 |
| `profiling_driven` | Profiling 驱动：基于 profiling 证据针对性设计 |
| `workflow_guided` | 归族 workflow 分支保底：按 workflow_ref 阶段顺序优化 |

### generation_lens 取值说明

| lens | 驱动方式 | 说明 |
|------|---------|------|
| `profiling_driven` | 确定性 + LLM | BOTTLENECK_STRATEGY_MAP 映射 + LLM 细化 |
| `method_transfer` | 确定性 + LLM | 策略索引 × 算子特征匹配 |
| `gap_analysis` | gap_map 驱动 | 针对开放知识缺口生成 |
| `continuation` | 实验结果驱动 | 在成功实验基础上叠加正交优化 |
| `open_exploration` | 纯 LLM | 第一原理自由推理 |
| `workflow_guided` | workflow 文件 | 归族命中时按分支阶段优化 |
| `supervisor_sourced` | Supervisor 外部复核 | Supervisor 返回的 MCTS 形状建议映射成 KG idea |

---

## Experiment 实体结构（experiments/<slug>.md frontmatter）

```yaml
---
type: experiment
node_id: exp:<slug>
title: "Round 1 Parallel 0: P1+P7"
idea_id: idea:<slug>
verdict: yes           # yes | partial | no
confidence: high       # high | medium | low
speedup: 2.3
solution_ref: "round_1/parallel_0"
round: 1
parallel_index: 0
profiling_insight: null  # CSV 级快速诊断
profiling_evidence: null # 指令级深度空泡分析
facts: null              # Stage 1 纯事实提取
diagnosis: null          # Stage 2 LLM 诊断
failure_type: null       # impl_error | strategy_infeasible
failure_reason: null
retry_count: 0
tags: []
added: 2026-08-19T...
---
```

### Experiment 字段约束

| 字段 | 类型 | 约束 |
|------|------|------|
| `verdict` | string | yes / partial / no |
| `confidence` | string | high / medium / low |
| `speedup` | float 或 null | 通过后为正数，失败为 null |
| `failure_type` | string 或 null | 仅 verdict=no 时有值：impl_error / strategy_infeasible |
| `retry_count` | int | ≥0，impl_error 修复 idea 继承 parent+1；≥2 时不再重试 |
| `profiling_evidence` | object 或 null | 与旧 schema 的 profiling_evidence 结构完全一致 |

### profiling_evidence 子结构（不变）

```json
{
  "bottleneck_type": "mte2_stall",
  "d_class_pct": 62.5,
  "c_class_pct": 15.3,
  "imbalance_ratio": 1.12,
  "primary_bottleneck": "MTE2",
  "suggested_strategies": ["P1", "P10"],
  "anti_strategies": ["P3"],
  "description": "MTE2 数据加载等待是主瓶颈...",
  "top_recommendation": "启用双缓冲隐藏MTE2搬运延迟",
  "pattern_type": "periodic",
  "overlap_status": "no_overlap",
  "dominant_subtype": "D_MTE2_WAIT",
  "dma_efficiency": { "mte2_short_pct": 35.0, "mte3_short_pct": 20.0 }
}
```

---

## Claim 实体结构（claims/<slug>.md frontmatter）

```yaml
---
type: claim
node_id: claim:<slug>
name: "双缓冲对 MTE2 瓶颈算子至关重要"
status: verified         # drafted → unproven → verified → refuted
supporting_experiments: [exp:001, exp:003]
refuting_experiments: []
strategy_implications: [P1]
bottleneck_labels: [mte2_stall]
confidence: 0.92
added: 2026-08-19T...
---
```

### Claim 字段约束

| 字段 | 类型 | 约束 |
|------|------|------|
| `status` | string | drafted / unproven / verified / refuted |
| `confidence` | float | 0.0-1.0 |
| `strategy_implications` | string[] | 此 claim 验证/否定的策略 ID |
| `bottleneck_labels` | string[] | 相关瓶颈类型 |

### Claim 生成条件

| 条件 | Claim 类型 | 状态 |
|------|-----------|------|
| 3+ 实验确认某策略有效 | strategy_verified | verified (confidence ≥ 0.75) |
| 2+ 实验确认某策略不可行（且 0 成功） | strategy_refuted | verified (confidence ≥ 0.75) |
| 2+ 成功实验共享 ≥60% 相同策略 | bottleneck_pattern | unproven (需更多证据) |

---

## 关系图谱（graph/edges.jsonl）

每行一个 JSON 对象：

```json
{"from": "idea:slug", "to": "exp:slug", "type": "tested_by", "evidence": "...", "added": "..."}
```

### Edge 类型

| Edge 类型 | From → To | 说明 |
|-----------|-----------|------|
| `tested_by` | idea → exp | Idea 被某实验测试 |
| `supports` | exp → claim/idea | 实验支持 claim 或 idea |
| `invalidates` | exp → claim/idea | 实验否定 claim 或 idea |
| `addresses_gap` | idea/claim → gap | 针对某知识缺口 |
| `inspired_by` | idea → idea | 跨 idea 血统 |
| `derived_from` | idea → exp | 新 idea 来源于实验结果 |
| `depends_on` | claim → claim | Claim 依赖链 |
| `refutes` | claim → claim | Claim 否定链 |

---

## Gap Map（gap_map.md）

```markdown
# Gap Map

## Active Gaps

### G1: MTE2 停顿缓解——双缓冲之外还有什么
- **Status**: open
- **Key**: bottleneck_primary_unexplored:mte2_stall
- **Kind**: bottleneck_primary_unexplored
- **Subject**: mte2_stall
- **Evidence**: exp:001 达到 2.3x (P1+P7)，但残余 MTE2 停顿 35%
- **Suggested lenses**: profiling_driven, method_transfer
- **Priority**: 1.8

## Closed Gaps

### G0: 基线瓶颈识别
- **Status**: closed
- **Evidence**: 由 claim:baseline-mte2-bound 关闭
- **Closure reason**: closed by claim:baseline-mte2-bound
```

Gap ID 格式：G1, G2, ...（递增，不复用）

`Key` 由稳定的语义 kind + invariant subject 派生，不能把易变化的计数或
示例列表放入身份。旧 gap_map 没有 `Key` 时先按已知描述规则推断，无法推断
时回退到描述匹配；命中同一 key 只更新原 G<n> 的描述/证据。`closed` gap
仍保留在地图中，但不计入 open gap 统计，也不进入 query pack。关闭原因
单独存放在 `Closure reason`；旧文件若把 `closed by ...` 追加在 Evidence 中，
解析时会恢复到该字段。

---

## Query Pack（query_pack.md，≤8000 chars）

自动生成的压缩摘要，用于 LLM idea 生成的上下文：

| 段落 | 预算 | 来源 | 裁剪优先级 |
|------|------|------|-----------|
| 算子摘要 + 基线 | 800 chars | kernel_summary, baseline_evidence | 不裁剪 |
| Top 5 Gaps + untried coverage | 1200 chars | gap_map.md（按 priority 排序，并保留 canonical `untried_strategies`） | **不优先裁剪** |
| 禁忌表（失败 ideas） | 1400 chars | ideas/ 中 outcome=negative | **不优先裁剪** |
| 关键实验 | 1800 chars | experiments/ 中 score 前 N | 可裁剪 |
| 已验证 Claims | 900 chars | claims/ 中 status=verified | 可裁剪 |
| 最近链路 | 500 chars | edges.jsonl 最近 10 条 | 优先裁剪 |

**裁剪规则**：gaps 和禁忌表不优先裁剪（anti-repetition 价值最高）。

---

## 与旧 world_model.json 的对应关系

| 旧字段 | 新位置 | 说明 |
|--------|--------|------|
| `decision_tree.nodes` | `ideas/*.md` + `experiments/*.md` | 树结构 → 扁平实体 + edges |
| `nodes[id].status=open` | idea `stage=proposed` | |
| `nodes[id].status=passed` | experiment `verdict=yes` | |
| `nodes[id].status=failed` | experiment `verdict=no` | |
| `nodes[id].score` | experiment `speedup` | |
| `nodes[id].difficulty` | idea `difficulty` | |
| `nodes[id].strategy_combination` | idea `strategy_tags` | |
| `nodes[id].mode` | idea `mode` | |
| `nodes[id].profiling_evidence` | experiment `profiling_evidence` | |
| `nodes[id].children` | 通过 edges `inspired_by`/`derived_from` 表达 | |
| `nodes[id].parent_id` | idea `parent_idea` / `parent_experiment` | |
| `open_questions` | `gap_map.md` + `claims/` | 结构化替代 |
| `stagnation_count` | `knowledge_graph.json` 中保留不变 | |
| `best_score` | `knowledge_graph.json` 中保留不变 | |
| `hw_params` | `knowledge_graph.json` 中保留不变 | |
| `baseline_evidence` | `knowledge_graph.json` 中保留不变 | |

---

## 停滞阈值配置（不变）

```
stagnation_window = 2   // 默认值
                        // max_rounds <= 3 → 1
                        // max_rounds == 4 → 2
                        // max_rounds >= 5 → 3

// 双计数器共享同一阈值，任意一个触发即停止（OR 逻辑）：
//   stagnation_count ≥ stagnation_window
//   stagnation_count_vs_base ≥ stagnation_window
```
