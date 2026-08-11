# OKF 自演进知识库：沉淀与调用全流程

> 适用范围：`plugins-community/ops-perf-evolution`（lingxi-evo / ops-evo 两条进化路径）
> 配套阅读：X 系列原始生命周期见 [`docs/x-series-lifecycle.md`](./x-series-lifecycle.md)（本文是其 OKF 双轨增强版）
> 本插件知识库分两层：**compat 兼容层**（`discovered/`，只读）+ **OKF 增强层**（`okf/`，首选，带结构化字段）

---

## 一、这是什么

进化优化过程中，算法上会反复出现两类可复用的经验：

- **成功经验**：某算子用某手法突破 ≥10%，换成同类算子可能同样有效（X 系列策略、proven solutions）
- **失败教训**：某策略组合在某种 shape/dtype 下必然失败（failure patterns）

OKF（Open Knowledge Format）把这套经验落成**带结构化 frontmatter 的 markdown 卡片**，挂在进化主循环上**每轮自动沉淀、每轮自动读取**，形成闭环：

```
进化第 N 轮 ──4.4.2/4.5.3──▶ 写 OKF 卡 ──generate_index_json──▶ 重建 INDEX.json
     ▲                                                              │
     │                                            evolution-knowledge（Init/4.4.2/4.5.3）
     │                                                              ▼
   第 N+1 轮 ◀──prepare_strategy_resources.py 注入── 同族算子/同瓶颈 命中卡片
```

**一句话**：第 N 轮发现的成功经验与失败教训，喂给第 N+1 轮的同族算子。

---

## 二、知识库物理结构

```
skills/evolution-knowledge/references/
├── a3/                          # 经典知识库（硬件/算法洞察/API 陷阱/proven_solutions 原文）
│   └── proven_solutions/INDEX.md   # compat 层方案登记表（与 OKF proven 双轨）
└── okf/                         # ★ OKF 增强层（首选，带 frontmatter）
    ├── index.md
    ├── strategies/discovered/       # 21 条 disc_X*.md（discovered_strategy）
    ├── runbooks/proven-solutions/   # 6 条 跨算子方案（proven_solution）
    └── runbooks/failure-patterns/   # 1 条 失败模式（failure_pattern）

skills/evolution-strategies/references/
├── discovered/disc_X*.md        # compat 层（只读，21 条，与 OKF 双轨保留）
└── INDEX.json                   # ★ 机器可读索引（cards=174 含 okf=28）

skills/evolution-knowledge/SKILL.md + scripts/query.py       # 读取入口
skills/evolution-knowledge/SKILL.md + scripts/migrate_*.py  # 写入入口 + 迁移
```

**双轨原则**：新发现的 X 必须**同时**写入 compat `discovered/` 和 OKF `strategies/discovered/`；读取优先 OKF，OKF 缺失时回退 compat。compat 层保证旧逻辑（只读 `discovered` 数组）不破坏，OKF 层提供结构化检索。

---

## 三、数据模型（一张卡长什么样）

三类卡，全部带 YAML frontmatter（由 `generate_index_json.py` 解析进 `INDEX.json`）：

### discovered_strategy（X 系列策略）

```yaml
schema_version: okf.v1
kind: discovered_strategy
id: X5                          # 查重键（kind + id）
origin: discovered
title: "GQA-Decode GEMV->GEMM"
confidence: single_op           # single_op → cross_op_verified（≥2 算子验证后提升）
status: active                  # stub / active / deprecated / retracted
op_families: [attention, flash_attention]   # 来自 INDEX.json vocabulary
bottlenecks: [compute_bound, mte2_stall]    # 来自 INDEX.json vocabulary
tags: [decode, gqa]
discovered_round: 2
discovered_from: "round_2/parallel_2"
base_speedup: 3.527
```

正文：`## 核心思路 / ## 适用场景 / ## 实现要点 / ## 与既有策略的关系 / ## 来源`

### proven_solution（跨算子方案）

`kind: proven_solution`、`op_families`、`strategy_ids`、`confidence`、`status`；
正文沿用 `a3/proven_solutions/INDEX.md` 模板：Source / Speedup / Technique description / Key code change / Applicability conditions。

### failure_pattern（失败模式）

```yaml
schema_version: okf.v1
kind: failure_pattern
key: "P22+A1:reduction:tail_precision"   # 查重键（kind + key）
phenomenon: precision_failure            # compile_error / precision_failure / perf_regression
severity: major
status: stub                             # occurrences ≥2 且跨 ≥2 算子 → active
failed_combination: [P22, A1]
operator_type: [reduction]
occurrences: 1
```

正文：`## Symptom / ## Root cause class / ## Failed combination / ## Resolution / ## Evidence / ## Related`

### 状态机与证据强度

| 维度 | 取值 | 提升规则 |
|---|---|---|
| `status` | stub → active → deprecated/retracted | failure_pattern：`occurrences ≥2` 且跨 ≥2 算子 → active；矛盾/过期不删，标 `deprecated` + 原因 |
| `confidence` | single_op → cross_op_verified | `reuse_stats.ops` 出现 ≥2 个不同算子且 `passed>0` → 提升 |
| 证据门槛（sufficiency rule） | 查询结果须 ≥1 条 `cross_op_verified` **或** ≥2 条 `single_op` 互相支撑 | **`stub` 不能单独支撑结论，只能当调查线索** |

---

## 四、写入链路（沉淀：每轮增量，非结束后一次性）

两步沉淀都嵌在**主循环内部**（`agents/lingxi-evo.md:326-346`），**每轮 REFINE/REACT 阶段检查、当轮命中当轮落卡**，不需要等全部轮数结束：

```
while should_continue and r <= max_rounds:
    # 4.1 GATE → 4.2 SELECT → 4.3 GENERATE
    # 4.4 REFINE ──▶ 4.4.2 失败诊断（写 failure_pattern）
    # 4.5 REACT  ──▶ 4.5.3 策略提炼（写 disc_X + proven）← 每轮最多触发一个 REACT 分支
    # 4.6 CHECKPOINT
```

### 沉淀 ①：failure_pattern —— 门槛最低（4.4.2 失败诊断）

触发条件（全满足）：
1. 本轮有失败节点（`pending_diagnosis.json` 存在且非空）
2. 该失败节点 `strategy_combination` **未命中** OKF failure-pattern 库（完全匹配才命中；库内现有组合少，新失败基本都算未命中）

动作：按 `evolution-knowledge` 协议写 `okf/runbooks/failure-patterns/f{n}-*.md`（`status: stub`、`occurrences: 1`）→ 重建索引。

### 沉淀 ②：disc_X + proven —— 门槛最高（4.5.3 策略提炼）

触发条件（**全部**满足）：
1. 本轮有 `mode="open_exploration"` 的节点
2. 该节点 `status="passed"`（编译 + 精度都过）
3. `score > best_score_before_this_round × 1.10`（**≥10%，相对本轮更新前的最优分**，不是基线；该基准由 4.4.1 refine 脚本保存）
4. LLM 判定"新颖"：超出现有所有策略范畴（非组合）+ OKF 查重无命中

**注意**：只有 `open_exploration` 节点算数，strategy_guided 跟到的大提升不产生新 X 卡（保守设计，防噪音）。

动作（确认新颖后多连写）：
| 步骤 | 目标 |
|---|---|
| ① compat `discovered/disc_X{n}.md` | 策略卡片全文 |
| ② **OKF `okf/strategies/discovered/disc_X{n}.md`** | 增强版，带 `kind/confidence/status/op_families/bottlenecks/tags` |
| ③ `strategy-index.md` "探索发现策略"表 | 追加一行 |
| ④ `world_model.json.discovered_strategies` | 追加 `"X{n}"` |
| ⑤ 跨算子通用时：compat `proven_solutions/INDEX.md` + **OKF `okf/runbooks/proven-solutions/`** | 完整方案条目 |
| ⑥ 重建 `generate_index_json.py` | INDEX.json 落库 |

### 脚本辅助（不走 OKF）

`skills/evolution-world-model/scripts/discover_strategies.py`（Phase D3）：扫 `lineage.jsonl`，
找 `precision_passed + speedup≥1.10×baseline + 组合不在 known` 的节点，生成 compat `disc_X` **草稿模板**。
⚠️ **当前未接进 agent 主流程，且只写 compat、不写 OKF**——正式沉淀仍以 4.5.3 LLM 四连写为准。

---

## 五、读取链路（调用：四个触发点）

### 1. Init / SELECT 阶段 —— 按算子族海选

位置：`agents/lingxi-evo.md:250`（知识库查询协议）、`agents/ops-evo.md:308`

```bash
python3 skills/evolution-knowledge/scripts/query.py \
    --op-family attention --bottleneck compute_bound --status active
```

命中即读卡片全文，纳入节点设计（尤其核对 X 系列适用场景）。

### 2. 4.4.2 失败诊断 —— 按失败组合反查

诊断前先查 failure-pattern：命中 → 复用 Resolution；未命中 → 沉淀（见沉淀①）。

```bash
python3 skills/evolution-knowledge/scripts/query.py \
    --kind failure_pattern --failed-combination P22,A1 --status all
```

### 3. 4.5.3 新颖性判断 —— 防重复注册

```bash
python3 skills/evolution-knowledge/scripts/query.py \
    --kind discovered_strategy --keyword "{手法关键词}"
```

### 4. 子 agent 启动前注入 —— 脚本层自动（prepare_strategy_resources.py）

每个 partial 启动前必跑 `prepare_strategy_resources.py`（`agents/lingxi-evo.md:511`），生成 `{strategy_resources_block}`：
- **`_build_discovered_md()`**：`strategy_combination` 含 X 系列时，**优先读 OKF 版** `okf/strategies/discovered/`（fallback compat），以 `<!-- BEGIN/END -->` 注入全文
- **`_build_failure_patterns_anti_md()`**：当前组合的 `failed_combination` **完全覆盖**某已知失败模式时，注入 `⚠️ Failure Pattern Alerts`（含 Resolution 指引）

> 设计意图：X 全文 + 失败反例都在 partial 动手前给到，子代理不用自己猜。

---

## 六、索引与工具

### INDEX.json（machine-parseable 权威清单）

由 `skills/evolution-strategies/scripts/generate_index_json.py` 生成：
- `discover_okf()` 扫描 `okf/` 三类卡 → 合并进 `cards` 数组，`source_key = evolution-knowledge#okf/...`
- `discovered` 数组 = compat 层 X 系列
- 当前规模：`cards=174`（含 `okf=28`：21 disc + 6 proven + 1 failure）、`discovered=21`

**每次写入 OKF 后必须重建**（maintain 协议强制）：
```bash
python3 skills/evolution-strategies/scripts/generate_index_json.py
```

### 查询 CLI（evolution-knowledge）

| 参数 | 说明 |
|---|---|
| `--kind` | discovered_strategy / proven_solution / failure_pattern |
| `--op-family` / `--bottleneck` | 结构化筛选 |
| `--phenomenon` | 失败现象 |
| `--failed-combination` | 逗号分隔策略 ID，子集匹配 |
| `--status` | 默认 `active`（**stub 默认排除**，需显式 `--status all` 才查得到） |
| `--keyword` | 全文轻量 grep |

输出带 `sufficiency` 布尔（证据门槛是否满足）。

### 迁移脚本（一次性历史数据搬运）

- `migrate_discovered_to_okf.py`：21 条 compat disc → OKF，关键词推断 `op_families/bottlenecks`
- `migrate_proven_to_okf.py`：INDEX.md 6 条 proven → OKF 单文件

---

## 七、端到端示例（以 X5 为例）

**沉淀**：`mla` 进化第 2 轮，`round_2/parallel_2` open_exploration 节点 geomean 3.527x（父 0.5745x，>10%）→ 4.5.3 判新颖 → 写 compat `disc_X5.md` + OKF `disc_X5.md`（`op_families:[attention,flash_attention,matmul]`、`bottlenecks:[compute_bound,mte2_stall,...]`）→ 重建索引。

**调用**：后续 `attention` 算子 Init 时 `query --op-family attention` 命中 X5 → 读全文核对适用场景 → 若进 `strategy_combination`，`prepare_strategy_resources.py` 从 OKF 注入 X5 全文给 partial。

**失败教训对照**：若某轮 `strategy_combination=[P22,A1]`（reduction），partial 启动前会收到 Failure Pattern Alert（P22+A1 → precision_failure），4.4.2 也会反查到该模式复用 Resolution。

---

## 八、快速验证命令（新人自检）

```bash
# 1. 索引健康
python3 skills/evolution-strategies/scripts/generate_index_json.py
python3 -c "import json; d=json.load(open('skills/evolution-strategies/references/INDEX.json')); print(d['stats'])"
# → 期望 cards=174, okf=28, discovered=21

# 2. 查询命中
python3 skills/evolution-knowledge/scripts/query.py --kind discovered_strategy --op-family attention --bottleneck compute_bound
# → 期望返回 X4/X5/X6/X7/X8，sufficiency=true

# 3. 失败反查
python3 skills/evolution-knowledge/scripts/query.py --kind failure_pattern --failed-combination P22,A1 --status all
# → 期望命中 F1（P22+A1:reduction:tail_precision）

# 4. 资源注入（含 X 全文 + 失败 anti）
python3 skills/evolution-world-model/scripts/prepare_strategy_resources.py \
    --strategy-ids "P22,A1,X5" --kernel-dir <某算子 kernel 目录> --evo-dir <evo 输出目录>
# → srblock 含 "Discovered Strategies"（X5 走 OKF 路径）+ "Failure Pattern Alerts"（P22+A1）
```

---

## 九、常见问题排查

### Q1：跑了几轮，OKF 一条没涨？

先分清"链路没接"还是"没到触发点"：
1. **会话早于接线**：OKF 触发描述是 2026-08-06 落库的。**长驻 `--resume` 的旧会话加载的是旧 agent prompt，不会自动升级**——要沉淀必须**新开会话**（或新 spawn lingxi-evo/ops-evo 子代理）。
2. **没到触发点**：
   - 无未命中已知模式的失败节点 → 沉淀①不触发（最常发生）
   - 无 `open_exploration` 突破 ≥10% 且新颖 → 沉淀②不触发（门槛最高，是设计意图）
   - 4.5 REACT 每轮最多触发一个分支，走了 Supervisor 则该轮跳过 4.5.3
3. **自查**：`ls skills/evolution-knowledge/references/okf/strategies/discovered/` 看 disc 数；`grep discovered_strategies output/*/world_model.json` 看有没有候选；对照第八节命令验证链路本身是通的。

### Q2：为什么 `query.py --failed-combination P22,A1` 查不到？

f1 的 `status: stub`，query 默认 `--status active` 会排除。加 `--status all` 即可。这也是设计：**stub 卡只当线索，不当决策依据**。

### Q3：新发现的 X 要写几个地方？

5 处（见沉淀②动作表）：compat disc + OKF disc + strategy-index 表 + world_model.discovered_strategies + （跨算子时）compat & OKF proven。**漏了任何一处，读取链路都会断**（尤其别漏 OKF 版和索引重建）。

### Q4：如何把历史的一条"待验证经验"（如 `references/pending/` 里的笔记）正式纳入？

按 `evolution-knowledge` 协议手工落卡：判断它属于三类卡的哪类 → 写 stub 卡（`status: stub`）→ 重建索引。满足提升条件后（跨算子复现）自然升 active。

---

## 十、关键文件地图

| 文件 | 作用 | 被谁引用 |
|---|---|---|
| `agents/lingxi-evo.md` | lingxi 路径 agent 触发描述（知识库协议 / Init / 4.4.2 / 4.5.3） | lingxi-evo 子代理 |
| `agents/ops-evo.md` | ops 路径同上 | ops-evo 子代理 |
| `skills/evolution-knowledge/SKILL.md` | 写入协议：查重 / frontmatter / 置信度 / 索引重建 | 4.4.2、4.5.3 写卡时 |
| `skills/evolution-knowledge/SKILL.md` + `query.py` | 结构化查询 + sufficiency rule | Init / 4.4.2 / 4.5.3 |
| `skills/evolution-strategies/scripts/generate_index_json.py` | `discover_okf()` 扫描 OKF → INDEX.json | 每次写卡后 |
| `skills/evolution-world-model/scripts/prepare_strategy_resources.py` | OKF 优先注入 X 全文 + 失败 anti | partial 启动前 |
| `skills/evolution-knowledge/references/okf/` | OKF 卡片仓库（首选读取/写入目标） | 上述全部 |
| `skills/evolution-strategies/references/INDEX.json` | 唯一机器索引 | query.py / prepare_strategy_resources.py / profiling_evidence.py / wm_ledger.py |
