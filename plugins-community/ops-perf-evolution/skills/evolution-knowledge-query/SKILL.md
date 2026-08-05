---
name: evolution-knowledge-query
description: ops-perf-evolution OKF 知识库查询 skill。按算子族、瓶颈类型、现象、标签等结构化检索 skills/evolution-knowledge/references/okf/ 中的策略、方案与失败模式。
---

# evolution-knowledge-query

ops-perf-evolution 的 OKF 知识库检索入口。

## 数据源

扫描 `skills/evolution-knowledge/references/okf/` 下的所有 `.md` 文件（排除 `index.md`），frontmatter 由 `skills/evolution-strategies/scripts/generate_index_json.py` 解析并写入 `skills/evolution-strategies/references/INDEX.json`。

## 检索维度

INDEX.json 中为 OKF 卡片提供以下结构化字段：
- `kind`: `discovered_strategy` | `proven_solution` | `failure_pattern`
- `op_families`: 算子族列表（来自 vocabulary，如 attention / reduction / matmul）
- `bottlenecks`: 瓶颈标签列表（如 compute_bound / mte2_stall）
- `phenomenon`: 失败现象（compile_error / precision_failure / perf_regression）
- `tags`: 自由标签
- `confidence`: single_op | cross_op_verified
- `status`: stub | active | deprecated | retracted

## 命令

### 按算子族 + 瓶颈检索

```bash
python3 skills/evolution-knowledge-query/scripts/query.py \
    --kind discovered_strategy \
    --op-family attention \
    --bottleneck compute_bound \
    --status active
```

### 按失败现象检索

```bash
python3 skills/evolution-knowledge-query/scripts/query.py \
    --kind failure_pattern \
    --phenomenon precision_failure \
    --failed-combination P22,A1
```

### 全文关键词检索（轻量 grep）

```bash
python3 skills/evolution-knowledge-query/scripts/query.py \
    --keyword "GQA decode"
```

## 证据强度规则（sufficiency rule）

查询结果必须满足以下任一条件才能作为决策依据：
- 至少 1 条 `confidence: cross_op_verified` 且 `status: active` 的条目
- 或至少 2 条 `confidence: single_op` / `status: active` 的条目互相支撑
- `status: stub` 的条目不能单独支撑结论，只能作为调查线索

## 输出格式

脚本输出 JSON：

```json
{
  "query": {"kind": "discovered_strategy", "op_family": "attention", "bottleneck": "compute_bound"},
  "results": [
    {
      "id": "X5",
      "title": "GQA-Decode GEMV->GEMM",
      "path": "skills/evolution-knowledge/references/okf/strategies/discovered/disc_X5.md",
      "confidence": "cross_op_verified",
      "status": "active",
      "op_families": ["attention", "flash_attention"],
      "bottlenecks": ["compute_bound", "mte2_stall"]
    }
  ],
  "sufficiency": true
}
```

## 在 agent 中的使用位置

### Init / SELECT 阶段

主 agent 在分析算子特性后，调用本 skill 查询：
- 同算子族的 `discovered_strategy`
- 同算子族的 `proven_solution`
- 同算子族 + 相关瓶颈的 `failure_pattern`

### 4.4.2 失败诊断

对每个失败节点，先按 `phenomenon` + `failed_combination` 查询 failure_pattern：
- 命中 → 复用 Resolution，生成修复子节点
- 未命中 → 诊断后调用 `evolution-knowledge-maintain` 新建条目

### 4.5.3 策略提炼

确认新颖前，按 `op_families` + `bottlenecks` 查询 discovered_strategy 和 proven_solution，避免重复登记。

## 与 X 系列原 discovered/ 的关系

- 优先检索 `okf/strategies/discovered/`（增强版，带结构化字段）
- 若 OKF 中不存在某 X ID，回退到 `skills/evolution-strategies/references/discovered/`（只读兼容层）
- 新发现的 X 必须同时写入 OKF 和原 discovered/（双轨保留）
