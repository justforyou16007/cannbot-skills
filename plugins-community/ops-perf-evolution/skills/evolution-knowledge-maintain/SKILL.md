---
name: evolution-knowledge-maintain
description: ops-perf-evolution OKF 知识库摄入与维护 skill。负责把进化过程中产生的策略、方案、失败模式按 OKF v1 规范写入 skills/evolution-knowledge/references/okf/，执行查重、frontmatter 校验、置信度提升与索引重建。
---

# evolution-knowledge-maintain

ops-perf-evolution 的 OKF 知识库摄入与维护入口。

## 职责

- 把进化产物（X 系列策略、proven solutions、failure patterns）写入 OKF
- 查重：同 kind + key 命中则 update，否则 new
- 维护 frontmatter 字段：`confidence`、`status`、`occurrences`、`reuse_stats` 等
- 调用 `generate_index_json.py` 重建 `skills/evolution-strategies/references/INDEX.json`

## 输入格式

### X 系列策略摄入

```json
{
  "schema_version": 1,
  "entries": [{
    "kind": "discovered_strategy",
    "id": "X{n}",
    "title": "策略名称",
    "description": "一句话描述",
    "content": "## 核心思路\n...\n## 适用场景\n...",
    "op_families": ["attention", "flash_attention"],
    "bottlenecks": ["compute_bound", "mte2_stall"],
    "base_speedup": 3.527,
    "source": {"op": "mla", "round": 2, "parallel": "round_2/parallel_2"}
  }]
}
```

### 失败模式摄入

```json
{
  "schema_version": 1,
  "entries": [{
    "kind": "failure_pattern",
    "key": "P22+A1:reduction:tail_precision",
    "phenomenon": "precision_failure",
    "severity": "major",
    "failed_combination": ["P22", "A1"],
    "operator_type": ["reduction"],
    "symptom": "ulp 误差集中在最后一个 tile",
    "root_cause_class": "dtype/tiling 交互",
    "resolution": "改用 A3 尾块单独标量路径",
    "evidence": {"op": "conv2d", "round": 2, "node": "n3"}
  }]
}
```

## 写入协议

1. **查重**
   - discovered_strategy 按 `id` 查
   - proven_solution 按 `title` / `key` 查
   - failure_pattern 按 `key` 查
   - 命中同概念条目：增量 update，保留已有内容，补充新来源/证据
   - 已有同手法但证据不同：update Evidence 字段，不新建

2. **新建条目**
   - discovered_strategy：`okf/strategies/discovered/disc_X{n}.md`
   - proven_solution：`okf/runbooks/proven-solutions/{kebab-title}.md`
   - failure_pattern：`okf/runbooks/failure-patterns/F{n}-{kebab-key}.md`

3. **frontmatter 规范**

   discovered_strategy：
   ```yaml
   schema_version: okf.v1
   kind: discovered_strategy
   id: X5
   origin: discovered
   title: "GQA-Decode GEMV->GEMM"
   description: "..."
   confidence: single_op          # 由验证次数自动提升
   status: active
   op_families: [attention, flash_attention]
   bottlenecks: [compute_bound, mte2_stall]
   tags: [decode, gqa]
   discovered_round: 2
   discovered_from: "round_2/parallel_2"
   base_speedup: 3.527
   ```

   failure_pattern：
   ```yaml
   schema_version: okf.v1
   kind: failure_pattern
   key: "P22+A1:reduction:tail_precision"
   phenomenon: precision_failure
   severity: major
   status: active
   failed_combination: [P22, A1]
   operator_type: [reduction]
   occurrences: 1
   created_at: '2026-08-05T10:00:00Z'
   updated_at: '2026-08-05T10:00:00Z'
   ```

4. **置信度提升规则**
   - `discovered_strategy`:
     - 出生：`confidence: single_op`
     - `reuse_stats.ops` 中出现 ≥2 个不同算子且 `passed > 0` → `cross_op_verified`
   - `failure_pattern`:
     - `occurrences ≥ 2` 且跨 ≥2 个不同算子 → `active`（stub 提升）
   - `proven_solution`:
     - `Source` 中列出 ≥2 个不同算子 evolution → `cross_op_verified`

5. **矛盾处理**
   - 不静默覆盖
   - 在条目正文追加 `> ⚠️ 矛盾标记：新来源 XXX 在 Y 场景下结论相反，待进一步验证`
   - 严重冲突时新建 `status: stub` 候选条目，原条目保持不变

6. **索引重建**
   - 每次写入/更新后调用：
     ```bash
     python3 skills/evolution-strategies/scripts/generate_index_json.py
     ```
   - 该脚本会扫描 `skills/evolution-knowledge/references/okf/` 并合并进 INDEX.json

## 使用方式

一般由 `lingxi-evo` / `ops-evo` agent 在以下时机调用：
- 4.5.3 策略提炼：生成新的 disc_X 时
- 4.5.3 通用方案沉淀：生成 proven_solution 时
- 4.4.2 失败诊断：生成/更新 failure_pattern 时
- 每轮 refine 后：调用更新 X 系列 reuse_stats / failure occurrences

不要让人工直接写 OKF 文件；所有写入应通过本 skill 或等价的 agent 流程完成，以保证 frontmatter 一致性和索引同步。
