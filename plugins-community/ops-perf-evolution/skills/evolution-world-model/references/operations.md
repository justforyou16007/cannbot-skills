# 知识图谱操作指南

本文件定义了 `lingxi-evo` / `ops-evo` Agent 在进化过程中对知识图谱的操作流程。

**核心设计原则**（来自 research-wiki 的 idea-creator 范式）：
- **每轮重新生成 ideas**：不维护"待选池"，而是基于最新 query_pack（gaps + banlist + claims + 实验结果）每轮现生成
- **LLM 评审决定选择**：不用确定性 utility 公式排序，由 LLM（跨模型评审）做质量评判
- **失败经验是最有价值的记忆**：failed ideas 以禁忌表形式压入 query_pack，防止重复探索

---

## 每轮进化流程

```
每轮:
  1. 读 query_pack.md → LLM 获得全局上下文
  2. 多镜头 idea 生成（LLM 为主 + 确定性兜底）
  3. LLM 评审 & 打分（跨模型 devil's advocate）
  4. 选排名前 parallel_num 个 ideas
  5. 并行实验（子 agent 实现 + 评测）
  6. 结果 → experiments/*.md + 更新 ideas outcome
  7. 失败诊断（impl_error vs strategy_infeasible）
  8. claim 生成（从累积证据自动产生）
  9. gap 分析 + 更新 gap_map.md
  10. 重建 query_pack.md → 下一轮回到 1
```

---

## 操作一：Init（初始化）

**时机**：步骤3（共享文件准备）完成后，进化轮次开始前执行一次。

```
1. 算子特性分析（不变）

2. 初始化知识图谱：
   python3 kg_wiki.py init <evo_dir>

3. 基线评测 + profiling（不变）

4. 从基线 profiling_evidence 生成初始 gap_map.md：
   python3 gap_analysis.py analyze <evo_dir>

5. 多镜头 idea 生成（参见下文 "Idea 生成流程"）

6. 生成 query_pack.md：
   python3 query_pack.py build <evo_dir>

7. 写入 knowledge_graph.json（session + baseline + counters）
```

---

## 操作二：Idea 生成（每轮）

**参考**：research-wiki `/idea-creator` Phase 0-7 流程

### Step 1: 加载 query_pack（Phase 0）

```
读 query_pack.md，作为 LLM 生成 ideas 的全局上下文：
- gaps → 优先搜索方向（搜索种子）
- 禁忌表 → 不得重新生成的方向（仅 strategy_infeasible 失败）
- 关键实验 → 已知有效/无效的策略组合
- 已验证 claims → 累积知识（可直接复用的结论）
```

### Step 2: 确定性基础生成

```bash
python3 idea_generation.py generate <evo_dir> \
  --baseline-evidence <evidence.json> --parallel-num {N} --round {r}
```

产出基于 profiling 证据和策略索引的候选 ideas。

### Step 3: LLM 多镜头补充生成（Agent 推理）

Agent 从以下镜头生成补充 ideas：

| 镜头 | 输入 | 输出 |
|------|------|------|
| profiling_driven | baseline/experiment profiling_evidence | 针对具体瓶颈的策略组合 |
| method_transfer | strategy-index.md + 算子特征 | 跨领域策略迁移 |
| gap_analysis | gap_map.md open gaps | 针对每个知识缺口的方案 |
| continuation | 上轮成功实验的 profiling | 在已成功方案上叠加正交优化 |
| open_exploration | 内核代码 + profiling 数据 | 第一原理自由推理（不读策略库） |
| workflow_guided | 归族判定结果 | 族 workflow 分支保底 |
| supervisor_sourced | Supervisor JSON new_nodes[] | 外部复核建议映射为 KG idea |

**关键规则**：
- open_exploration 镜头**禁止读取策略库**，必须从代码和 profiling 出发
- 所有镜头的输出只做注释不做淘汰（质量判断留给 Step 4）
- 确保 bandwidth / tiling / algorithm 三类各至少 1 个 idea

### Step 4: LLM 评审 & 打分（Phase 4 跨模型评审）

**参考 idea-creator Phase 4: Cross-Model Jury**

将全部候选 ideas（含注释：prior_work, so_what, effort_note）提交给跨模型评审：

```
评审要求：
1. 对每个 idea 给出最强反对意见和最可能失败模式
2. prior_work 注释是否构成真正的新颖性问题
3. 按 top venue 标准排序
4. 推荐最值得实施的 2-3 个

评审模型：使用 gpt-5.6-sol（xhigh reasoning）
不使用 Claude（执行者不能评审自己的输出）
```

评审排名是**权威的质量判决**——Agent 不得基于自己的偏好淘汰候选。

### Step 5: 选取 top-N

从评审排名中取前 `parallel_num` 个 idea。

### Step 6: 写入 ideas/*.md

```bash
# 入选 ideas → stage=proposed (后续实验将更新为 active/piloted)
# 淘汰 ideas → stage=archived
python3 kg_wiki.py upsert_idea <evo_dir> --slug <slug> --title "..." \
  --strategy-tags "..." --generation-lens "..." --stage proposed ...
```

**注意**：`--outcome` 始终设为 `pending`——实验结果由 Refine 阶段设置，此处不预判。

---

## 操作三：Generate（并行生成，不变）

并行子 agent 实现内核变体，每个子 agent 对应一个选中的 idea。
子 agent 接收：idea 的 strategy_tags + thesis + query_pack 上下文。

---

## 操作四：Refine（结果更新）

**参考**：research-wiki `/result-to-claim` Step 1-5

### Step 1: 创建 experiment 实体

对每个完成的并行变体：

```bash
python3 kg_wiki.py add_experiment <evo_dir> \
  --slug "r{round}-p{idx}" --title "Round {r} Parallel {idx}: {tags}" \
  --idea-id "idea:{slug}" --verdict {yes|no} \
  --speedup {speedup} --solution-ref "round_{r}/parallel_{idx}" \
  --round {r} --parallel-index {idx}
```

**profiling 数据通过 sidecar JSON 持久化**：
`add_experiment` 会自动将 profiling_insight/profiling_evidence/facts/diagnosis
写入 `experiments/<slug>.json`（sidecar），确保 `load_experiments()` 能完整加载。

### Step 2: 更新 idea 状态

```bash
# passed → outcome=positive, stage=piloted
python3 kg_wiki.py update_idea_outcome <evo_dir> \
  --slug {slug} --outcome positive --stage piloted

# failed → outcome=negative, stage=archived
python3 kg_wiki.py update_idea_outcome <evo_dir> \
  --slug {slug} --outcome negative --stage archived
```

### Step 3: 失败诊断（Agent LLM 推理）

对 verdict=no 的 experiment，Agent 判定 failure_type：
- **impl_error**：策略方向可行，实现有 bug → 不进入策略禁忌表，进入修复队列
- **strategy_infeasible**：策略本身与算子不兼容 → 进入禁忌表
- **profiling_lost**：编译/精度证据有效但 profiling 不完整 → 不进入禁忌表或反证，必须补采 profiling

### Step 4: Claim 生成

```bash
python3 claim_engine.py generate <evo_dir>
```

生成条件（含极性标记）：
| 条件 | Claim 类型 | polarity |
|------|-----------|----------|
| 3+ 实验确认某策略有效 | strategy_verified | positive |
| 2+ strategy_infeasible 失败（排除 impl_error） | strategy_refuted | negative |
| 2+ 成功实验共享 ≥60% 相同策略 | bottleneck_pattern | positive |

**关键修复**：
- impl_error 不用于否定策略（只有 strategy_infeasible 算数）
- claim 带 polarity 字段（positive/negative），下游可区分

### Step 5: Gap 分析

```bash
python3 gap_analysis.py analyze <evo_dir> \
  --stagnation-count {count} --stagnation-count-vs-base {count}
```

Gap 检测源（仅在有真实瓶颈证据时生成）：
- **已见瓶颈的未探索策略**：profiling 中看到了瓶颈 A，但 A 的推荐策略还没试
- **瓶颈迁移**：优化解决了瓶颈 A 但暴露了瓶颈 B → 新 gap
- **claim 驱动组合**：已验证的 claim 暗示可以与新策略组合

**关键修复**：不在空 KG 上对 BOTTLENECK_STRATEGY_MAP 全表扫描生成噪声 gap。

### Step 6: 停滞检测（不变）

```
本轮最佳 vs best_score × 1.02 → stagnation_count
本轮最佳 vs 父实验得分 → stagnation_count_vs_base
```

### Step 7: 重建 query_pack

```bash
python3 query_pack.py build <evo_dir>
```

---

## 操作五：Analyze（每轮末尾）

```
1. 读 claims/, experiments/, gap_map.md

2. 结构化 gap 分析（替代旧 open_questions）：
   - 关闭已被实验/claim 解决的 gap
   - 新增瓶颈迁移、未探索方向、claim 暗示的 gap
   - 按 stagnation_count 调整 gap 优先级

3. 如果 stagnation_count > 0：
   - 下轮多镜头生成中提高 open_exploration 占比
   - gap_analysis 更激进地探索未试方向

4. 重建 query_pack.md
```

---

## 操作时序图

```
Step 3.5:  [Init] → 初始化知识图谱 + gap_map
                   → 多镜头生成 ideas + LLM 评审选 top
                   → 生成 query_pack

Round 1:
  [Idea 生成]    → 读 query_pack → 多镜头生成 → LLM 评审 → 选 top-N
  [并行生成]     → lingxi-partial × parallel_num
  [收集结果]
  [Refine]      → 创建 experiments + 更新 idea 状态
                → 失败诊断（impl_error vs strategy_infeasible）
                → Claim 生成
                → Gap 分析 + 重建 query_pack
  [CSV Profile] → 每个 passed experiment 做 CSV 级瓶颈诊断
  [Deep Profile] → 条件触发的指令级空泡分析
  [Analyze]     → Gap 审计 + 知识图谱完整性检查

Round 2:
  [Idea 生成]    → 读更新后的 query_pack → 基于新证据重新生成 ideas
  ...

终止条件满足 → 输出最优解 + 保存 knowledge_graph.json
```

---

## 禁忌表语义

| 失败类型 | 进入禁忌表？ | 处理方式 |
|---------|------------|---------|
| `strategy_infeasible` | ✅ 是 | 策略方向不可行 → 禁忌表 → query_pack banlist 段 |
| `impl_error` | ❌ 否 | 实现有 bug → 修复队列 → 可重试（difficulty+1） |

---

## 与 research-wiki 范式的对应

| research-wiki 概念 | 演化知识图谱对应 |
|-------------------|-----------------|
| `/idea-creator` Phase 0 (load query_pack) | 每轮读 query_pack.md |
| `/idea-creator` Phase 1.5 (multi-lens fan-out) | 多镜头 idea 生成 |
| `/idea-creator` Phase 3 (feasibility gate) | 确定性去重 + 禁忌表过滤 |
| `/idea-creator` Phase 4 (cross-model jury) | LLM 评审 & 打分 |
| `/idea-creator` Phase 7 (write to wiki) | 写入 ideas/*.md |
| `/result-to-claim` (evidence → experiment → edges) | Refine Step 1-2 |
| `/proof-checker` (birth claims) | claim_engine.py generate |
| `gap_map.md` | gap_analysis.py analyze |
| `query_pack.md` (compressed context) | query_pack.py build |
| failed ideas → banlist | 仅 strategy_infeasible → 禁忌表 |
