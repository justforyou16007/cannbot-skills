# X 系列（进化发现策略）生命周期：生成与调用全流程

> 适用范围：`plugins-community/ops-perf-evolution` 插件（lingxi-evo / ops-evo 两条进化路径）
> X 系列 = 进化过程中自动发现并注册的策略，实体文件位于
> `skills/evolution-strategies/references/discovered/disc_X{n}.md`

---

## 一、生成流程（写入侧）

X 系列只在一个地方诞生：**每轮进化的 REACT 阶段，4.5.3 策略提炼**。

### 1. 触发判定

位置：`agents/lingxi-evo.md:771-773`（ops 仓库路径对应 `agents/ops-evo.md:802-804`）

> 触发条件：本轮有 `mode="open_exploration"` 且 `status="passed"` 且 `score > best_score_before_this_round × 1.10` 的节点（提升 ≥ 10%）

三个硬条件缺一不可：

- **开放探索节点**（strategy_guided 节点表现再好也不提炼）
- **编译 + 精度通过**（`status="passed"`）
- **比本轮更新前的全局最优再高 10%**

基准值 `best_score_before_this_round` 由 4.4 refine 脚本保存（`agents/lingxi-evo.md:560`）。

### 2. 新颖性判断

位置：`agents/lingxi-evo.md:778`

读 `strategy-index.md` 判断"是否超出现有所有策略的范畴（不是组合，而是新方法论）"；
同时查 `proven_solutions/INDEX.md` 避免重复登记（`agents/lingxi-evo.md:65`）。
非新颖 → 输出 `[INFO] open_exploration 手法与策略 {匹配ID} 相似，跳过提炼`（`agents/lingxi-evo.md:780`）。

### 3. 四连写（确认新颖后）

位置：`agents/lingxi-evo.md:779`（ops-evo 对应 `agents/ops-evo.md:810-834`）

| 步骤 | 写入目标 | 内容 |
|---|---|---|
| ① | `skills/evolution-strategies/references/discovered/disc_X{n}.md` | 策略卡片全文。格式固定在 `agents/lingxi-evo.md:782-805`：frontmatter（`id / origin / discovered_round / discovered_from / base_speedup`）+ 核心思路 / 适用场景 / 实现要点 / 来源 |
| ② | `strategy-index.md` 末尾"探索发现策略"表 | 一行：`\| X{n} \| {简洁名称} \| {一句话描述} \|` |
| ③ | `world_model.json` 的 `discovered_strategies` 列表 | 追加 `"X{n}"` |
| ④ | `skills/evolution-knowledge/references/a3/proven_solutions/INDEX.md` | **仅当手法跨算子通用时**追加完整方案条目（Source / Speedup / 前后代码对比 / 适用条件 / Strategy IDs）。格式见 `agents/lingxi-evo.md:807-819`、`agents/ops-evo.md:835-839` |

辅助路径：`skills/evolution-world-model/scripts/discover_strategies.py` 可自动生成 disc_X 草稿模板
（`:216` "LLM 需在下一轮 refine 时补全 placeholder"），并用 `load_known_combos`（`:109-114`）
读取已有 disc 文件去重；`next_disc_id`（`:125-133`）负责编号递增。

---

## 二、调用流程（读取侧）

### 1. 索引层海选（Init / Refine 阶段）

位置：`agents/lingxi-evo.md:247`

主 agent 读 `strategy-index.md` 识别最相关策略——X 系列以"探索发现策略"表与 P/D/A 系列同台呈现
（`skills/evolution-strategies/references/strategy-index.md:268` 起，每行含名称、适用场景、
一句话描述、`discovered/disc_X{n}.md` 路径）。

分层协议定义在 `skills/evolution-strategies/SKILL.md:46-48`：X 系列为 **L2 (Discovered)** 层
（L0 = 通用基础策略 D1-D5/P1-P13/A1-A8；L1 = 进阶 P14+；L2 = 进化发现 X 系列）。

### 2. 节点设计（X 进入 strategy_combination）

位置：`agents/lingxi-evo.md:252-253`、`agents/ops-evo.md:320`

> **X 系列策略必读全文**：strategy_combination 中包含 X 系列时，必须先读取对应的
> `discovered/disc_X{n}.md` 全文，确认其适用场景与当前算子匹配后再纳入组合

通过后 `X{n}` 与其他系列混搭进节点的 `strategy_combination`（示例：`X18/P53/P83/A1`）。

### 3. 子 agent 启动前注入（脚本层）

位置：`skills/evolution-world-model/scripts/prepare_strategy_resources.py`
（调用协议：`agents/lingxi-evo.md:498-508`，每个 partial 启动前必须执行）

- **Preconditions 过滤直接放行**：X 系列无 preconditions 文件，
  `skills/evolution-strategies/scripts/check_preconditions.py:325-327`
  "无 Preconditions 文件 → 默认通过（向后兼容）"
- **全文注入**：`_build_discovered_md()` 经 INDEX.json 的 `discovered` 数组
  （由 `skills/evolution-strategies/scripts/generate_index_json.py:185-197` 生成，
  条目格式 `{id, doc_path, source_key}`）解析 X ID → 读 `disc_X{n}.md` 全文 →
  以 `<!-- BEGIN/END -->` 包裹注入资源块的 "Discovered Strategies" 段
- X 系列无 Playbook；组合中的 P/D/A 部分照常经 `load_playbook.py` 注入 SOP，
  两者在同一资源块中并存

### 4. 跨轮继承与追踪

- **继承**：X{n} 作为父组合的一部分参与子节点派生——地基传承 / 兄弟 offset 差异化 / 滚动槽，
  规则在 `skills/evolution-world-model/scripts/profiling_evidence.py:345-385`
  （`merge_strategies_with_evidence`）
- **偏置**：LLM 可通过父节点诊断的 `next_round_hint.prefer/avoid` 显式推举或封杀 X
  （`agents/lingxi-evo.md:612`，约束：|prefer|+|avoid| ≤ 3，ID 须在 INDEX.json）
- **追踪**：`lineage.jsonl` 每条记录含 `mutation = {added, removed}`，
  即子组合相对父组合的策略 diff（`skills/evolution-world-model/scripts/wm_ledger.py:304-313`）

---

## 三、闭环总览

```
生成：open_exploration 节点提升 ≥10%（lingxi-evo.md:773）
   → 新颖性判断（:778）
   → 四连写：disc_X{n}.md + strategy-index.md + world_model.json + proven_solutions（:779, :807）
                                    │
                                    ▼
调用：读 strategy-index.md X 表（:247）
   → 必读 disc 全文（:253）
   → X{n} 进 strategy_combination
   → prepare_strategy_resources.py 注入全文 → partial 实现
   → 下一轮 merge_strategies_with_evidence 继承/变异（profiling_evidence.py:345）
   → 若 open_exploration 再出显著新招 → 回到生成
```

**一句话**：生成端只在 open_exploration 显著突破时触发（4.5.3 单一入口）；
调用端走"索引海选 → 必读全文 → 组合引用 → 脚本注入 → 跨轮继承"五级链路。
