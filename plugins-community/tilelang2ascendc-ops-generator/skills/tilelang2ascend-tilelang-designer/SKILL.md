---
name: tilelang2ascend-tilelang-designer
description: >
  TileLang kernel 设计调度 Skill。协调 tilelang-op-design（设计方法论）和 tilelang-op-develop（编码规范），
  完成 block-level 设计、tile-level 设计和 model_new_tilelang.py 生成。
  产物对接下游 AscendC translator 算子生成流程。
argument-hint: >
  输入：output_dir 目录路径（包含 model.py）。
  输出：block_level/ 设计、tile_level/ 设计、model_new_tilelang.py 实现。
---

# TileLang Kernel 设计调度 Skill

你是一名 TileLang kernel 设计调度专家。你的目标是协调 `tilelang-op-design`（设计方法论）和 `tilelang-op-develop`（编码规范）两个 Skill 的能力，为 `{output_dir}/model.py` 中的**复杂算子** PyTorch Model 完成 TileLang kernel 设计，产出符合下游 AscendC translator 要求的产物。

## 角色定位

本 Skill 是**调度协调器**，不是端到端开发者：

- **设计方法论** → 来自 `ops/tilelang-op-design` 的 references（技术约束检测、编程模式决策树、BlockLevelDesign、信息收集、质量自检）
- **编码实现规范** → 来自 `ops/tilelang-op-develop` 的 references（编码规范、GEMM/CV 融合、V 核并行化、检查清单、疑难解答）
- **本 Skill 负责**：流程编排、Attention 模式路由、产物格式适配（输出 block_level/ + tile_level/ + model_new_tilelang.py）、下游 AscendC translator 对接

## 适用场景

本 skill 仅用于**复杂算子**路径（见 CLAUDE.md 路由规则）：
- Attention: FlashAttention, SparseAttention, GQA 等
- MatMul 变体: 带 fuse 的 MatMul (matmul+leakyrelu, quant_matmul 等)
- Norm 变体: RMSNorm, LayerNorm (多 strategy)
- Sort: Sort, TopK
- 多输入融合: Concat, multi-tensor fused ops

**简单算子**（Index, IndexPut, Gather, Scatter, Nonzero, RepeatInterleave, EmbeddingDenseBackward）走 ops-direct-invoke 工作流（Architect 设计 → Developer 实现 → Reviewer 审查），不使用本 skill。

## 关键限制
- 必须将核心计算融合成单个算子实现，不要拆分成多个独立算子。
- `model_new_tilelang.py` 中禁止使用 torch 算子；只允许进行张量创建，张量变换以及调用你实现的自定义算子。
- 在 TileLang 实现中应尽可能避免标量逐元素写法，优先使用 `T.copy`、`T.tile.*`、矩阵/向量原语等块级或向量化操作；只有在确实无法避免时才使用标量逻辑。
- 只允许修改或新增 `{output_dir}/` 目录中的文件，不要改动其他目录中的文件。
- 只允许读取当前工作区目录结构内的文件与子目录；禁止读取当前工作区之外的任何路径，包括父目录、兄弟目录、用户目录、绝对路径以及系统其他目录。
- 禁止读取 `asc-devkit/docs/` 目录及其下任何文件；该目录仅供 AscendC 阶段使用，与本阶段无关。

## 任务目录结构
```text
.
├── {output_dir}/         # 当前活跃任务目录
│   ├── model.py          # 参考 PyTorch 模型，禁止修改
│   ├── <op_name>.json    # 原始测试用例文件（备份保留）
│   ├── <op_name>.json.bak# 原始 .json 备份
│   ├── design/           # TileLang DSL 用于表达 kernel 设计
│   │   ├── block_level/  # TileLang block-level 设计
│   │   └── tile_level/   # TileLang tile-level 设计，用于表达完整 kernel 设计
│   ├── kernel/           # AscendC kernel（本阶段不涉及）
│   └── model_new_tilelang.py # 你的 TileLang 优化实现，调用 tile_level/ 下的 TileLang kernel
└── <other_tasks>/        # 其他历史任务，可作为参考实现
```

## Skill 参考资料

### 本 skill 自带脚本

- `.claude/skills/tilelang2ascend-tilelang-designer/scripts/evaluate_tilelang.sh` — TileLang 评测脚本（远程/本地执行，当前仅供可选调试，不作为流程 gate）
- `.claude/skills/tilelang2ascend-tilelang-designer/scripts/validate_tilelang_impl.py` — TileLang 实现退化检测（检测 PyTorch 回退）
- `.claude/skills/tilelang2ascend-tilelang-designer/scripts/verification_tilelang.py` — TileLang 精度验证

### 设计方法论（ops/tilelang-op-design 贡献，不复制、只引用）

> 路径以项目根目录为基准（即 `cannbot-skills/` 下的相对路径）。

| 文档 | 用途 | 使用阶段 |
|------|------|----------|
| `ops/tilelang-op-design/references/BlockLevelDesign.md` | Block 级任务划分、流水骨架、workspace 与同步关系设计 | Step 1d |
| `ops/tilelang-op-design/references/attention-patterns/AttentionPatternIndex.md` | Attention / FlashAttention 类算子的模式路由索引（TND、paged KV cache、mask/causal、GQA/MQA、MLA、topk sparse KV、sink attention） | Step 0 |
| `ops/tilelang-op-design/references/ascend-constraints.md` | NPU 技术约束清单（三维 Kernel/threads/动态边界/L0C溢出）与强制检测规则 | Step 1a |
| `ops/tilelang-op-design/references/decision-tree.md` | 编程模式决策树 + API 映射规则 + NPU 硬件约束 | Step 1b |
| `ops/tilelang-op-design/references/info-sources.md` | 信息收集步骤与优先级（搜索 examples/、确认 API 可用性） | Step 1c |
| `ops/tilelang-op-design/references/quality-checklist.md` | 设计质量 20 项自检清单 | Step 3a |

### 编码方法论（ops/tilelang-op-develop 贡献，不复制、只引用）
- **API 用法**：查阅 [tilelang-api-best-practices SKILL.md](../tilelang-api-best-practices/SKILL.md) 及其 references 目录
- **编程模式和 pass_configs**：查阅 [tilelang-programming-model-guide SKILL.md](../tilelang-programming-model-guide/SKILL.md) 及其 references 目录
> 路径以项目根目录为基准（即 `cannbot-skills/` 下的相对路径）。

| 文档 | 用途 | 使用阶段 |
|------|------|----------|
| `ops/tilelang-op-develop/references/coding-conventions.md` | Buffer 分配 / 索引 / 同步 / 广播 / 测试模板 | Step 2a |
| `ops/tilelang-op-develop/references/gemm-cv-fusion.md` | GEMM + CV 融合 pass_configs、NPU 分形限制 | Step 2b |
| `ops/tilelang-op-develop/references/vector-parallelism.md` | V 核并行化（Developer threads=2 / Expert 手动 vid 切分） | Step 2c |
| `ops/tilelang-op-develop/references/checklist.md` | 22 项上库前检查清单 | Step 3b |
| `ops/tilelang-op-develop/references/troubleshooting.md` | 编译 / 运行 / 精度错误排查手册 | Step 3d |

### 性能方法论（ops/tilelang-perf-optimization 贡献，不复制、只引用）

> 路径以项目根目录为基准（即 `cannbot-skills/` 下的相对路径）。
> 以下文档用于**设计期性能避坑**——在 block/tile 设计阶段就避免引入常见性能劣化模式，
> 而非等算子完成后才回头调优。完整性能调优流程见 `ops/tilelang-perf-optimization/SKILL.md`。

| 文档 | 用途 | 使用阶段 |
|------|------|----------|
| `ops/tilelang-perf-optimization/references/performance-antipatterns.md` | 常见性能劣化模式清单：launch core 数不当、逐元素 for loop、冗余全局同步、指令未融合、tile size 过小、CV 未 overlap、AIV 未流水/双 buffer、多行 tile 循环内归约 | Step 1e, 2e |
| `ops/tilelang-perf-optimization/references/optimization-guide.md` | 性能优化指南：Split-K、Double Buffer、Fixed Core、多行 Tile 粒度扩展、MTE2 预取、指令向量化/融合 | Step 1e（§一 算子类型对应）, 2e |
| `ops/tilelang-perf-optimization/references/cube_advanced_strategies.md` | Cube 高级策略路由表：pingpong / swat / streamk / fullload / scale_coalescing / mte2_preload | Step 1e（仅 Cube/CV 融合型） |
| `ops/tilelang-perf-optimization/references/cv_sync_debug.md` | C/V 融合 lockstep 诊断与避免（设计期预防 hang/死锁） | Step 1e（仅 CV 融合型） |

---

## 流程

🛑 **执行以下各步骤前，必须先完成步骤 0 的强制查阅。未完成步骤 0 全部 checklist 前，禁止 Edit/Write 任何 `{output_dir}/design/` 下的代码文件。**

---

### 🛑 步骤 0: Attention 算子模式路由（Attention / FlashAttention 类算子强制执行）

```
⚠️ 本步骤是硬性门禁。如果 model.py 是 Attention / FlashAttention 类算子，
   必须逐个完成以下 checklist 后才能进入步骤 1。禁止跳过。
```

**触发条件**：`{output_dir}/model.py` 的 forward() 中包含以下任一特征：
- `softmax(Q @ K^T / sqrt(d)) @ V` 或等价 attention 模式
- `scaled_dot_product_attention` / `F.scaled_dot_product_attention`
- 文件名包含 `Attention` / `FlashAttention` / `SparseAttention`
- 类名包含 `Attention` / `SDPA` / `Flash`

**强制执行清单**：

```
0.1 🛑 读取 AttentionPatternIndex.md（必须，不可跳过）:
    Read ops/tilelang-op-design/references/attention-patterns/AttentionPatternIndex.md
    
0.2 🛑 逐条回答"生成前问题"中的 7 个诊断问题，记录命中的模式:
    1. 输入是标准 [B,H,S,D] 还是 (T,H,D) 拼接布局？
    2. K/V 是连续 tensor 还是 paged cache？
    3. Hq 和 Hkv 是否相等？
    4. Dqk 和 Dv 是否相等？
    5. 是否有 sink_k/sink_v？
    6. 是否有 indices/topk？
    7. 是否有 causal、padding、显式 mask？
    
    如果 7 个问题全部否定 → 命中"标准 Attention" → 下一步 0.3 读 archive 模板
    如果任一命中 → 下一步 0.3 读对应的 pattern 文档（可组合）

0.3 🛑 只读取命中的文档（渐进式披露，只读需要的）:
    - 命中模式 → Read 对应文档顶部的"先读这个"部分
    - 7 项全否定 → Read workflows/templates/archive_tasks/flash_attention/ 中的
      block_level/flash_attention.py 和 tile_level/flash_attention.py
      重点理解: online softmax rescale、Q 分块循环、O 分块循环、C/V split 流水线

0.4 🛑 在思考中确认:
    - 已读的 pattern 文档列表及其关键规则
    - 组合顺序（多模式命中时按 TND → Head Sharing → MLA → Sink → Sparse → Paged → Mask 顺序理解）
    - 本算子的 block-level 流水骨架应与命中的 pattern 对齐
```

**门禁规则**：
- 如果触发条件满足但 0.1-0.4 未完成 → **禁止**进入步骤 1，**禁止**生成任何 design/ 下的代码
- 如果触发条件不满足 → 跳过步骤 0，直接进入步骤 1
- 禁止凭记忆或经验跳过模式文档直接设计

---

### 步骤 1: Block 层级设计

生成 `{output_dir}/design/block_level/` 下的 block-level 设计，并同步生成 `{output_dir}/model_new_tilelang.py` 骨架。在这一步只确定 block 级任务划分、流水骨架、workspace 与同步关系，具体计算细节先标记为 `TODO(tile-level)`。

🛑 **在生成 block_level/ 代码前，必须先完成以下技术分析。**

#### 1a. 技术约束检测（强制）

Read `ops/tilelang-op-design/references/ascend-constraints.md`，对当前算子逐项检测 6 项：

| 检测项 | 触发条件 |
|--------|----------|
| 三维 Kernel | 算子是否需要 >1 维 block 并行 |
| threads 参数 | 是否需要 threads > 2 |
| 动态循环边界 | 循环次数是否依赖 tensor 值 |
| GPU 专用 API | 用户参考实现是否含 CUDA API |
| GEMM 非整除 | M/N 是否不被 block size 整除 |
| L0C 溢出 | block_M × block_N × 4 > 128KB |

检测到违反 → 输出警告 + Ascend 替代方案后再继续。

#### 1b. 编程模式决策（强制）

Read `ops/tilelang-op-design/references/decision-tree.md`，按决策树判定：
- 计算类型：纯 Vector / 纯 Cube / 混合（CV 融合）
- 编程模式：Developer（推荐，threads=2 消 workspace/vid）/ Expert / 混合
- API 选型：T.gemm_v0 vs T.mma、alloc_L1 vs alloc_shared

#### 1c. 信息收集（强制）

Read `ops/tilelang-op-design/references/info-sources.md`：
- 搜索 examples/ 同类实现
- 确认 API 可用性

#### 1d. 生成 block_level/ + model_new_tilelang.py 骨架

参考 `ops/tilelang-op-design/references/BlockLevelDesign.md`：
- 任务划分：确定 block 间分工、block 内遍历顺序
- 流水设计：C/V 分工与交错执行方案
- 模版设计：必要时定义多个 `T.prim_func` 按 shape 选择
- block 级设计必须与步骤 0 命中的 pattern 文档中的地址公式、循环结构、数据流对齐
- `model_new_tilelang.py` 负责输入布局整理 + kernel 调用

#### 1e. 设计期性能初检（推荐）

在 block 级设计确定后、进入 tile 级实现前，快速对照性能反模式清单检查设计骨架是否存在明显性能隐患。此步骤**推荐执行**，非硬性门禁，但可避免将性能问题固化到 tile 级代码中。

Read `ops/tilelang-perf-optimization/references/performance-antipatterns.md`，重点关注以下与 block 级设计直接相关的项目：

| 检查项 | 触发条件 | 设计期修正方向 |
|--------|----------|---------------|
| launch core 数 | `block_num` >> 24 物理核 → 按任务数 launch | 改用 Fixed Core 模式，按物理核数 launch，核内 `T.serial` 分配多任务 |
| launch core 数 | `block_num` < 24 仍按 24 launch | 按 `min(block_num, 24)` launch |
| CV 未 overlap | CV 融合型但主循环串行（Cube 写 workspace → Vector 读） | 改用 `T.Pipelined` + AUTO_CV_COMBINE/AUTO_CV_SYNC |
| tile size 过小 | L0C/UB 实际使用量远小于容量 | 在不超限前提下成倍扩大 block_M/block_N |
| AIV 未流水 | 纯 AIV 算子 GM→UB→Vector→GM 串行 | 改用 `T.Pipelined` 或手动双 buffer |

Read `ops/tilelang-perf-optimization/references/optimization-guide.md` §一（算子类型对应），确认本算子的优化范围与优先级顺序。

**Cube/CV 融合型补充**：Read `ops/tilelang-perf-optimization/references/cube_advanced_strategies.md` 的路由表，判断本算子是否需要预留 pingpong/swat/streamk 等高级策略的扩展空间（如在 block 级设计中预留 Split-K 切分维度）。

**CV 融合型补充**：Read `ops/tilelang-perf-optimization/references/cv_sync_debug.md` §4（锁步诊断），确认设计的流水结构不会落入 Cube/Vector 串行乒乓（各闲一半）的 lockstep 调度。

---

### 步骤 2: Tile 层级设计

在第一步基础上继续生成 `{output_dir}/design/tile_level/`。直接以 block-level 设计为骨架，在 tile-level 中补全各处 `TODO(tile-level)`，完成用于表达设计意图的 TileLang 设计与实现。

#### 2a. 编码规范检查

Read `ops/tilelang-op-develop/references/coding-conventions.md`：
- Buffer 分配 shape 一致性
- 数据搬运索引正确性（读/写/workspace 必须一致）
- 同步策略匹配编程模式（Developer 自动/AUTO_SYNC，Expert 手动/barrier_all）

#### 2b. GEMM/CV 融合专项（仅含 GEMM 或 CV 融合时）

Read `ops/tilelang-op-develop/references/gemm-cv-fusion.md`：
- gemm_v0 / T.mma 首次调用 init=True（语义上首次累加需初始化）
- 4 pass_configs 完整（AUTO_SYNC + AUTO_CV_COMBINE + AUTO_CV_SYNC + MEMORY_PLANNING）
- Developer 默认 threads=2 + 片上直连（消 workspace/vid）

#### 2c. V 核并行化

Read `ops/tilelang-op-develop/references/vector-parallelism.md`：
- Developer：threads=2 自动并行，无 vid 偏移
- Expert/回退：手动 vid 切分 + 索引一致性

#### 2d. 生成 tile_level/ 代码

- **API 用法**：查阅 [tilelang-api-best-practices SKILL.md](../tilelang-api-best-practices/SKILL.md) 及其 references 目录
- **编程模式和 pass_configs**：查阅 [tilelang-programming-model-guide SKILL.md](../tilelang-programming-model-guide/SKILL.md) 及其 references 目录


#### 2e. Tile 级性能反模式终检（推荐）

在 tile 级代码完成后，对照性能反模式清单做最终扫描，确认 tile 级实现未引入新的性能劣化。此步骤**推荐执行**。

Read `ops/tilelang-perf-optimization/references/performance-antipatterns.md`，重点扫描以下 tile 级容易引入的问题：

| 检查项 | 识别特征 | 修正方向 |
|--------|----------|---------|
| 逐元素/逐行 for loop | Vector 侧 `for row in range(block_M): T.tile.sub(...)` | broadcast 到同 shape 后一次性向量化 |
| 冗余全局同步 | 循环内频繁 `T.barrier_all()` / `T.sync_all()` | 改用 `T.set_flag`/`T.wait_flag` 精确依赖，或交给 AUTO_SYNC |
| 指令未融合 | `mul + add` / `max(x,0)` / `sqrt + div` 连续出现 | 替换为 `mul_add_dst` / `relu` / `rsqrt` 等复合指令 |
| 多行 tile 循环内归约 | 多行 tile 循环内每块调用 `reduce_sum` | 循环内累积到完整 buffer，循环外一次性归约 |
| AIV 未流水/双 buffer | 纯 AIV 搬入→计算→搬出完全串行 | 改用 `T.Pipelined` 或手动双 buffer |

Read `ops/tilelang-perf-optimization/references/optimization-guide.md`：
- Cube 型：确认 Split-K 切分 + Double Buffer + Fixed Core 是否已纳入设计
- Vector 型：确认 Double Buffer + 指令向量化/融合 + Fixed Core 是否已考虑
- CV 融合型：确认核内优化后再检查核间流水（`T.Pipelined` num_stages、cross_interval）

---

### 步骤 3: 自检（可选）

如用户明确要求，或为了排查 DSL 语法 / 编译问题，可执行以下自检。

#### 3a. 设计质量自检

Read `ops/tilelang-op-design/references/quality-checklist.md`，按 20 项逐项检查设计产物。
必须项（1,2,3,7,8,9,13,19,20）全部通过，推荐项 ≥ 4/9。

#### 3b. 编码规范自检

Read `ops/tilelang-op-develop/references/checklist.md`，按 22 项逐项检查代码产物。

#### 3c. 可选 TileLang 评测

调用 `.claude/skills/tilelang2ascend-tilelang-designer/scripts/evaluate_tilelang.sh {output_dir}` 做辅助检查；但 TileLang 结果当前不作为 correctness gate，也不作为性能测试输入。若遇到框架语义缺陷、尾块处理异常或其他 TileLang 自身 bug，应保留设计表达并在最终说明中明确记录，不要为了通过 TileLang 验证而扭曲设计。

调试时如需排查 DSL 表达问题，参考 `ops/tilelang-op-develop/references/TileLangDebug.md`（T.printf / T.dump_tensor）。

#### 3d. 疑难解答

如遇编译/运行/精度错误，Read `ops/tilelang-op-develop/references/troubleshooting.md` 按错误类型排查：

| 错误类型 | 排查方向 |
|---------|---------|
| 编译错误 | buffer 大小、API 参数、对齐 |
| 运行错误 | 索引越界、同步缺失 |
| 精度错误 | Golden 实现、输出形状 |
