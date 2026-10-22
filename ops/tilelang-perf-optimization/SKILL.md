---
name: tilelang-perf-optimization
description: TileLang 算子性能调优与潜在性能劣化模式检查。提供性能数据采集、瓶颈诊断、优化实施、效果验证能力；也用于生成或评审算子时对照常见性能劣化模式示例检查当前 kernel 代码。触发：算子精度通过后需要优化性能、性能不及预期时。
---

# TileLang 性能优化

## 工作流程

```
Step 1: 基线采集（性能 + 精度）
  → Step 2: 算子类型判断
  → Step 3: 阅读参考文档并识别优化点（输出到 optimization_log.md）
  → Step 4: 逐项实施优化点
  → Step 5: 效果验证（性能 + 精度）
```

## 核心约束

- **逐项实施**：每次 Edit 只改一个优化点，改完立即验证
- **精度优先**：精度未通过禁止性能优化
- **性能验证**：必须使用 `msprof op`，禁止用 Python/Torch 计时
- **Host 轻量化**：禁止 host 侧全量数据搬运（`F.pad`、`.contiguous()`、`.to(dtype)` 等），必须移入 kernel

## 参考文档

- **优化指南**：[optimization-guide.md](references/optimization-guide.md)
- **反模式清单**：[performance-antipatterns.md](references/performance-antipatterns.md)
- **测量纪律**：[measurement_discipline.md](references/measurement_discipline.md)（A/B 口径、环境排除、bound 分类、上限分析）
- **C/V 场景卡片**：[CvPatternIndex.md](references/cv-patterns/CvPatternIndex.md)（串行基线、核内流水、纯 V gate、稀疏取数、在线归约、局部 epilogue；按触发条件选读，源码机制与性能证据分开）
- **C/V 同步调试**：[cv_sync_debug.md](references/cv_sync_debug.md)（C/V 融合 kernel hang / 死锁 / 间歇性错误 / 锁步诊断）
- **cube 高级策略**：[cube_advanced_strategies.md](references/cube_advanced_strategies.md)（pingpong/swat/streamk/fullload/scale_coalescing/mte2_preload 路由）
- **API 用法**：[tilelang-api-best-practices](../tilelang-api-best-practices/SKILL.md)
- **编程模式**：[tilelang-programming-model-guide](../tilelang-programming-model-guide/SKILL.md)
- **cube最佳实践**：[cube_optimization_path.md](references/best-practices/cube_optimization_path.md)
---

## 执行步骤

### Step 1: 基线采集

在 `examples/{op_name}/` 下查找含 `@tilelang.jit` 的脚本，运行：

```bash
msprof op --kernel-name="main_kernel" --output=./msprof_output python ./examples/{op_name}/<script_name>.py
```

精度未通过 → 禁止后续步骤。

**基线测量纪律（强制）**：按 [measurement_discipline.md](references/measurement_discipline.md)
§1-§3 执行——采集前 `npu-smi info` 确认目标卡空闲；≥20 次取中位数；记录 NPU ID /
时间戳 / shape / dtype；kernel time（msprof task duration）与 wall time 分开记录。

### Step 2: 算子类型判断

**生成翻译后的 Ascend C 代码**：

在算子脚本中，JIT 编译返回的函数对象调用 `get_kernel_source()` 可获取翻译后的 Ascend C 代码：

```python
func = jit_func(batch=B, seq_len=S, ...)
print(func.get_kernel_source())
```

运行脚本后，从输出中搜索关键字判断算子类型：

| 判断依据 | 类型 | 典型算子 |
|---------|------|---------|
| `IS_ASCEND_AIC` 出现 | Cube 型 | GEMM、MatMul、Linear |
| `IS_ASCEND_AIV` 出现 | Vector 型 | RoPE、Softmax、Add |
| 两者均出现 | 混合型 | FlashAttention、SparseFlashAttention |

### Step 3: 识别优化点（强制，禁止与 Step 4 合并）

根据算子类型阅读 `optimization-guide.md` 对应章节 + `performance-antipatterns.md`，如果是 cube 核额外参考最佳实践 `best-practices/cube_optimization_path.md`；基线优化已完成但仍不达标时按 [cube_advanced_strategies.md](references/cube_advanced_strategies.md) 的路由表选下一层策略（先做 bound 分类，[measurement_discipline.md](references/measurement_discipline.md) §4）。

**C/V 候选与相关 V 链的补充动作**：

- 涉及 C 后处理、流水、gate、稀疏取数、在线归约或局部 epilogue 时，先读 [CvPatternIndex.md](references/cv-patterns/CvPatternIndex.md) 并只加载命中卡片；记录合同、实际 dispatch 和待验证项，不由源码形态推断 Tv/Tc 或收益

- 优化前检查锁步调度——诊断方法见 [cv_sync_debug.md](references/cv_sync_debug.md) §4；确认存在可并行的独立块、槽位预算和正确同步后，把 overlap 列为候选，与串行基线实测比较。锁步不代表 C/V 耗时必然各占一半
- 调优中如出现 hang / 间歇性错误结果，立即停止性能迭代，转 [cv_sync_debug.md](references/cv_sync_debug.md) 的诊断协议，排除同步 bug 后再回来

在 `optimization_log.md` 中输出：

**Part A 优化点清单**：逐条标注适用/不适用 + 原因 + 参考文件行号。`pass_configs` 不是独立优化点，是伴随修改。

```
[#1] [名称]（参考: optimization-guide.md L445-L650 §2.13）：[适用/不适用] — [原因]
```

**Part B `[ORDER-PLAN]`**：分析依赖关系，排出实施顺序链。依赖分析三条规则：
1. **布局依赖**：改变 layout 的优化排在依赖此 layout 的优化之前
2. **数量依赖**：涉及预算的优化排在改变 buffer 数量的优化之后
3. **配置依赖**：涉及 pass_configs 的优化在相关功能实施后才改动

```
[ORDER-PLAN] 实施顺序：
1. [#N] [名称] — 前置依赖: [无] — 理由: [...]
2. [#M] [名称] — 前置依赖: [#N] — 理由: [...]
```

### Step 4: 逐项实施

**固定优先级**：先静态分析（对照 `performance-antipatterns.md`），再 P0 Host 侧优化（`optimization-guide.md` §2.12）。P0 完成后 Host 侧只允许零拷贝形状变换。

**后续优化点**按 `[ORDER-PLAN]` 逐个实施，每个走 6 子步骤：

```
0: ORDER-CHECK → A: Read 文档 → B: Edit 代码 → C: msprof op 验证 → D: 记录结果 → (失败) E: 重读文档修复
```

**门禁**：`[ORDER-CHECK]` 未写禁止 Read；`[IMPL-#N]` 未写禁止 Edit；`[RESULT-#N]` 未写禁止下一个。

**日志格式**：
```
[ORDER-CHECK] 准备实施: [#N] [名称] | 前置依赖: [#1 ✅ / #2 ❌] | 结论: [✅/❌]
[IMPL-#N] 已阅读 <文件> L行号（§X.X），关键约束: ...
[SELF-CHECK] 本次 Edit 只涉及 [#N]
[RESULT-#N] 优化点: [名称] | 精度: [pass/fail] | 性能: [X us] | 对比: [+/-X%]
```

**Double Buffer 特殊要求**：实施前必须完成 `[DB-ANALYSIS]`（Q1: 循环内有 MTE3？Q2: 有跨迭代累加器？Q3: 选同步方式），未完成禁止写代码。

**最佳实践参考**：

| 算子类型 | 文档 | 核心优化技术 |
|---------|------|-------------|
| Vector 型 | [RoPE 优化](references/best-practices/rope-developer-mode.md) | NPU 内动态生成 Mask、Tile API 向量化、参数简化 |
| Cube 型 | [GEMM Intrinsic](references/best-practices/gemm_intrinsic_optimize.md) | 多缓冲流水线、细粒度 Flag 同步、MMA intrinsic、L0 分块、负载均衡 |
| Cube 型（进阶） | [cube 高级策略路由](references/cube_advanced_strategies.md) | pingpong/swat/streamk/fullload/scale_coalescing/mte2_preload 按 bound 分类选用 |
| CV 融合型 | [Flash Attention](references/best-practices/flash_attn_optimize.md) | num_stages 流水线、批量 Softmax、Cross-core Semaphore、数据布局优化；**多 shape 适配**（BSND 免转置、Sq==1 decode 窄块、加性 mask 屏蔽变长 Skv） |
| CV 融合型（GMM 族） | [GMM+融合 epilogue](references/best-practices/gmm_cv_fusion_optimize.md) | 单向数据流 + 背压深度调优、epilogue 三段式单 pass、多变体编译期分发、分组负载均衡 |
| CV 融合型（排障） | [C/V 同步调试](references/cv_sync_debug.md) | hang/死锁分类树、组屏障对称参与、pipe 配对、GM 可见性、锁步诊断 |

### Step 5: 效果验证

每个优化点后执行：精度验证 → `msprof op` → 记录 → 对比基线。精度失败时保持优化调试，不撤销。

**对比口径（强制）**：按 [measurement_discipline.md](references/measurement_discipline.md)
执行——同卡同 session 背靠背 A/B；kernel time 与 wall time 都报（不一致时先分解 wall
再归因，禁止只看 wall 就回滚 kernel 层面真实有效的改动）；每组数字附 NPU ID /
次数 / 中位数。

调试手段：`T.printf`、`T.dump_tensor`、`get_kernel_source()`，详见 [Programming Guide](../../../docs/TileLang-Ascend%20Programming%20Guide.md)。

迭代终止：达到目标或连续 3 次无提升则中断上报。**终止前必须先做上限分析**
（[measurement_discipline.md](references/measurement_discipline.md) §5：算术强度拆解 +
Amdahl 上限 + roofline 判定），确认剩余空间不足以支撑继续迭代；终止上报必须附带
上限分析结论，不允许只报"连续无提升"。

---

## 优化记录

保存在 `examples/{op_name}/perf_tuning/`：
- `baseline.json` - 基线性能
- `optimization_log.md` - 优化记录
- `final_report.md` - 最终报告
