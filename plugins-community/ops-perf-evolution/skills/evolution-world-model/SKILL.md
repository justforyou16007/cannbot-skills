---
name: evolution-world-model
description: "知识图谱驱动的算子演化工具和参考文档。支持 Idea-Exp-Claim 知识图谱模式（多镜头 idea 生成、gap 分析、claim 验证、query_pack 上下文压缩）和旧 MCTS 决策树模式。提供统一 kg_ops 入口、状态路由、完整性验证、摘要生成和报告适配。"
---

# Evolution World Model

知识图谱驱动的算子演化工具和参考文档，用于证据驱动的 AscendC 内核进化优化。

## 两种运行模式

| 模式 | 数据格式 | 适用场景 |
|------|---------|---------|
| **知识图谱模式（KG）** | `knowledge_graph.json` + `ideas/` + `experiments/` + `claims/` | 推荐。多镜头 idea 生成、gap 分析、claim 验证 |
| **旧 MCTS 模式** | `world_model.json` + `decision_tree.nodes` | 向后兼容 |

唯一模式判定：仅已有 `world_model.json` 且无 `knowledge_graph.json` 的存量会话使用 MCTS；新目录（两个证据文件都不存在）默认 KG，初始化后以 `knowledge_graph.json` 固化模式。两个证据文件同时存在属于冲突，必须停止并恢复单一证据源。`ideas/` 目录单独存在不参与判定；Python 使用 `kg_ops.detect_mode()`，shell hook 使用 `detect_evo_mode()`。

## 文件布局

```
references/
  schema.md                   # knowledge_graph.json 完整 Schema 定义
  operations.md               # 操作指南：Init / Select / Refine / Analyze
  state_schema.md             # state.json 状态机定义
  idea_generation_lenses.md   # 多镜头 idea 生成指南

scripts/
  # === 知识图谱模式（新） ===
  kg_wiki.py                  # 知识图谱实体 CRUD（ideas/experiments/claims/edges）
  kg_ops.py                   # KG session init/refine/summary/validate/stats 编排入口
  idea_generation.py          # 多镜头 idea 生成引擎
  gap_analysis.py             # Gap 检测和优先级排序
  claim_engine.py             # Claim 生成和管理
  query_pack.py               # 上下文压缩（≤8000 chars）

  # === 核心（两种模式共用） ===
  wm_ops.py                   # MCTS CLI（select/validate/summary/refine/diagnose）
  profiling_evidence.py       # 瓶颈分析 → 策略映射（BOTTLENECK_STRATEGY_MAP）
  state_ops.py                # 运行时状态机管理
  session_anchor.py           # Session 身份锚定
  wm_ledger.py                # 血统追踪 + lineage.jsonl
  wm_audit.py                 # 审计（诊断/验证）
  solution_db.py              # JSONL 谱系追踪
  check_round_artifacts.py    # 产物完整性检查
  transcript_audit.py         # 子 Agent 审计
```

## CLI 工具

### 知识图谱模式

```bash
# KG 会话编排入口（推荐由 evo Agent 调用）
python3 .../scripts/kg_ops.py init --evo-dir <evo_dir> \
    --op-name <op> --requested-rounds 3 --parallel-num 3 \
    --baseline-eval <baseline_evaluation.json>

# 每轮写回 experiments、ideas、claims、gaps 和 query_pack
python3 .../scripts/kg_ops.py refine --evo-dir <evo_dir> --round 1 \
    --results-dir <evo_dir>/round_1 --parallel-map '{"0":"idea:<slug>"}'

# KG 完整性 gate、摘要和统计
python3 .../scripts/kg_ops.py validate --evo-dir <evo_dir>
python3 .../scripts/kg_ops.py summary --evo-dir <evo_dir>
python3 .../scripts/kg_ops.py stats --evo-dir <evo_dir>

# 初始化知识图谱
python3 .../scripts/kg_wiki.py init <evo_dir>

# 创建 idea
python3 .../scripts/kg_wiki.py upsert_idea <evo_dir> \
    --slug <slug> --title "..." --strategy-tags P1,P7 \
    --optimization-type bandwidth --difficulty 3 \
    --generation-lens profiling_driven --generation-round 1

# 创建 experiment
python3 .../scripts/kg_wiki.py add_experiment <evo_dir> \
    --slug <slug> --title "..." --idea-id idea:<slug> \
    --verdict yes --speedup 2.3 --round 1

# 创建 claim
python3 .../scripts/kg_wiki.py add_claim <evo_dir> \
    --slug <slug> --name "..." --status verified \
    --polarity positive --strategy-implications P1 --confidence 0.92

# 多镜头 idea 生成（Init 阶段）
python3 .../scripts/idea_generation.py generate <evo_dir> \
    --baseline-evidence <json> --parallel-num 3

# 从实验结果生成新 ideas（Refine 阶段）
python3 .../scripts/idea_generation.py from_experiment <evo_dir> \
    --experiment-slug <slug> --parallel-num 3

# Gap 分析
python3 .../scripts/gap_analysis.py analyze <evo_dir> \
    [--strategy-index <path>] [--stagnation-count <int>]

# Claim 生成
python3 .../scripts/claim_engine.py generate <evo_dir>

# Query Pack 构建
python3 .../scripts/query_pack.py build <evo_dir> [--max-chars 8000]

# 统计
python3 .../scripts/kg_wiki.py stats <evo_dir>
```

### 旧 MCTS 模式（保留兼容）

```bash
# 节点选择
python3 .../scripts/wm_ops.py select --path {world_model.json} --n {parallel_num}

# 不变量验证
python3 .../scripts/wm_ops.py validate --path {world_model.json}

# Refine
python3 .../scripts/wm_ops.py refine --wm-path {world_model.json} --round {r} ...

# 其他子命令：summary, deep-profiling, diagnose, session 等
```

### Idea 评审方式（知识图谱模式）

知识图谱模式不使用确定性的 idea 排序器。每轮：
1. 读 query_pack.md（gaps + banlist + claims + 实验）作为 LLM 上下文
2. 多镜头生成 ideas（profiling_driven + method_transfer + gap_analysis + continuation + open_exploration + workflow_guided）
3. LLM jury 结合证据、可行性、创新性、workflow 匹配度和预期收益做质量判断
4. 选排名前 parallel_num 个 ideas 进入实验

## 参考文档

### Schema (references/schema.md)

定义知识图谱完整结构：
- `knowledge_graph.json`: session, baseline, hw_params, idea_registry, counters
- `ideas/*.md`: stage, outcome, strategy_tags, mode, generation_lens, target_gaps
- `experiments/*.md`: verdict, speedup, profiling_evidence, facts, diagnosis
- `claims/*.md`: status, strategy_implications, bottleneck_labels, confidence
- `graph/edges.jsonl`: typed edges (tested_by, supports, invalidates, ...)
- `gap_map.md`: 知识缺口（G1, G2, ...）+ priority + suggested_lenses
- `query_pack.md`: ≤8000 chars 压缩上下文

### Operations (references/operations.md)

KG 每轮的十步流程详见 operations.md：读取 query_pack → 多镜头生成 → LLM jury 评审 → 选 top-N → 并行实验 → 写入 experiments/更新 idea outcome → 失败诊断 → claim 生成 → gap 分析 → 重建 query_pack。`kg_ops.py` 将 Init 和 Refine 的实体写回、计数器和 best/stagnation 状态集中管理。

### Idea Generation Lenses (references/idea_generation_lenses.md)

7 种 idea 生成镜头：
- `profiling_driven`: BOTTLENECK_STRATEGY_MAP 确定性映射 + LLM 细化
- `method_transfer`: 策略索引 × 算子特征匹配
- `gap_analysis`: gap_map 驱动
- `continuation`: 成功实验深化
- `open_exploration`: LLM 自由推理
- `workflow_guided`: 归族 workflow 分支保底
- `supervisor_sourced`: Supervisor 外部复核建议映射成 KG idea

## 模块依赖

```
idea_generation.py → kg_wiki.py, profiling_evidence.py
gap_analysis.py    → kg_wiki.py, profiling_evidence.py
claim_engine.py    → kg_wiki.py
query_pack.py      → kg_wiki.py
kg_ops.py          → kg_wiki.py, idea_generation.py, gap_analysis.py, claim_engine.py, query_pack.py
wm_ops.py          → MCTS world_model.json, profiling_evidence.py
state_ops.py       → 独立运行
session_anchor.py  → 独立运行
```
