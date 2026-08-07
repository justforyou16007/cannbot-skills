---
name: evolution-knowledge
description: Domain knowledge base for AscendC evolution optimization covering hardware architecture, algorithm insights, API pitfalls, optimization patterns, proven solutions, and the OKF self-evolution knowledge base (discovered strategies / proven solutions / failure patterns) for A3 (910B) architecture. 当进行 AscendC 进化优化需要查询硬件架构、算法洞察、API 陷阱、优化模式、已验证方案或 OKF 自演进知识库（X 系列策略 / 跨算子方案 / 失败模式）时使用。
---

# Evolution Knowledge Base

AscendC 进化优化领域知识库，分两层：

- **经典层 `references/a3/`**：人工策展的硬件架构、算法洞察、API 陷阱、优化模式与 proven solutions（只读为主）
- **OKF 自演进层 `references/okf/`**：进化过程中自动沉淀的策略/方案/失败模式卡（结构化 frontmatter，运行期首选读写目标）

当前支持 A3 (Ascend 910B) 架构。

## 文件布局

```
references/
  a3/                           # 经典知识库（Ascend 910B / A3 架构，人工策展）
    guide.md                    # 顶层快速入口
    hardware/
      guide.md                  # 瓶颈诊断启发式、tiling 公式
      ascend910b_arch.md        # 架构详情：UB=196KB, 40核, DMA 模型
    algorithm_insights/
      guide.md                  # 算法优化原则
      attention_family.md       # MHA/GQA/MQA 优化
      reduction_ops.md          # Norm/Softmax: two-pass, Welford
      elementwise_fusion.md     # 算子融合
    ascendc_api/
      guide.md                  # Top 5 致命陷阱（快速诊断）
      common_pitfalls.md        # 15 个常见陷阱详解
    optimization_patterns/
      guide.md                  # 决策树：bandwidth vs algorithm vs register
      double_buffering.md       # 20-80% 增益
      tiling_strategies.md      # 10-50% 增益
      causal_block_skip.md      # 20-50% (attention)
      pipeline_overlap.md       # 5-30% 增益
      memory_coalescing.md      # 10-40% (strided access)
    proven_solutions/
      INDEX.md                  # 已验证方案索引（compat 层，可追加写入）
  okf/                          # OKF 自演进知识库（进化沉淀，带结构化 frontmatter）
    index.md
    strategies/discovered/      # X 系列策略（discovered_strategy，21 条，与 a3 上层 discovered/ 双轨）
    runbooks/proven-solutions/  # 跨算子已验证方案（proven_solution，6 条）
    runbooks/failure-patterns/  # 失败模式（failure_pattern，用于 anti 提示）
```

> OKF 卡片的机器索引在 `skills/evolution-strategies/references/INDEX.json`
> （`cards` 中 `source_key` 前缀 `evolution-knowledge#okf/`，由 `generate_index_json.py` 的 `discover_okf()` 扫描生成）。

## 渐进式查询协议

知识库采用**渐进式披露**（progressive disclosure）：先读 guide.md 快速参考，按需深入。

### OKF 层优先（推荐）

运行期读取**先走 OKF**（结构化可检索），未命中再回落经典 a3：

1. **按算子族/瓶颈查 OKF**：用 `evolution-knowledge-query` skill
   ```bash
   python3 skills/evolution-knowledge-query/scripts/query.py \
       --op-family attention --bottleneck compute_bound --status active
   ```
   - `--kind` 可限定 `discovered_strategy` / `proven_solution` / `failure_pattern`
   - 命中即读对应 OKF 卡全文，纳入策略选择
   - **证据门槛（sufficiency rule）**：至少 1 条 `cross_op_verified` 或 2 条 `single_op` 支撑结论；`stub` 卡只能当线索
2. **失败反查**：诊断失败时按 `--failed-combination` 查 failure_pattern，命中复用 Resolution

### 经典层按工作流阶段查询

| 阶段 | 必读 | 按需 |
|---|---|---|
| Init (世界模型初始化) | `a3/hardware/guide.md`, `a3/optimization_patterns/guide.md` | `a3/algorithm_insights/{family}.md` |
| Sub-agent 代码生成前 | `a3/ascendc_api/guide.md` (Top 5 致命陷阱) | `a3/optimization_patterns/*.md` (按策略类型) |
| Refine (新颖性评估) | `a3/proven_solutions/INDEX.md` | — |

### 按 Agent 角色查询

| Agent | 必读 | 可选 |
|---|---|---|
| ops-evo / lingxi-evo | OKF（按 op_family 查）+ hardware + optimization_patterns guide.md | algorithm_insights (匹配算子族) |
| ops-partial / lingxi-partial | ascendc_api/guide.md | optimization_patterns (匹配策略类型) |

## 检索优先级链

**OKF 知识库 → a3 经典知识库 → evolution-strategies → proven_solutions → WebSearch**

（OKF 为进化沉淀的结构化卡，优先级最高；a3 为人工策展参考；strategy 库为 P/D/A/R 系列策略卡片。）

## 写入协议

- **OKF 层（首选）**：进化产生的策略/方案/失败模式一律经 `evolution-knowledge-maintain` skill 写入
  `references/okf/`（查重 → frontmatter 校验 → 置信度提升 → **重建 INDEX.json**）。
  触发点：lingxi-evo/ops-evo 的 **4.4.2 失败诊断**（新 failure_pattern）与 **4.5.3 策略提炼**（新 disc_X + proven，双轨写 compat + OKF）。
- **经典层**：`a3/proven_solutions/INDEX.md` 在策略提炼出跨算子通用方案时追加条目（compat 层）；其他文件为只读参考。
- **新发现的 X 必须双轨**：compat `discovered/disc_X{n}.md` **和** OKF `okf/strategies/discovered/disc_X{n}.md` 都写，并重建索引。

## 架构选择

- 目标芯片 Ascend 910B → 使用 `references/a3/`
- 进化沉淀 → `references/okf/`（结构化卡）+ 写入 `references/a3/proven_solutions/INDEX.md`（compat 登记）
