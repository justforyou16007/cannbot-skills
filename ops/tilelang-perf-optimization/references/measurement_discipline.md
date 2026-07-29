# 性能测量纪律（采集 → 归因 → 上限）

> **适用场景**：所有性能调优轮次的基线采集、候选对比、结论报告。
> 测量口径错了，后续所有"优化"都是在噪声上做拟合。本文档是 Stage 3 每一轮的
> **强制前置**，不是可选参考。

---

## 目录

- [1. A/B 对比纪律（同一条件的定义）](#1-ab-对比纪律同一条件的定义)
- [2. 环境前置检查（排除假性快慢）](#2-环境前置检查排除假性快慢)
- [3. Kernel time vs Wall time（两个口径都要报）](#3-kernel-time-vs-wall-time两个口径都要报)
- [4. msprof 指标 → Bound 分类 → 优化章节路由](#4-msprof-指标--bound-分类--优化章节路由)
- [5. 上限分析（到顶就收手）](#5-上限分析到顶就收手)
- [6. 结论报告模板](#6-结论报告模板)

---

## 1. A/B 对比纪律（同一条件的定义）

性能声明（"提升 X%" / "无回归"）只在以下条件**全部满足**时成立：

| 条件 | 原因 |
|------|------|
| **同一物理 NPU** | 不同卡存在硬件个体差异、温度差异 |
| **同一 session、背靠背运行** | 容器重启、驱动状态变化、其他任务干扰都会污染对比 |
| **基线与候选交替/邻接测量** | 先跑完基线 50 次再跑候选 50 次，期间环境可能已漂移 |
| **同一 shape、同一 dtype、同一测试脚本** | 显而易见的可比性前提，但换脚本重测是最常见的违规 |
| **次数 ≥ 20 取中位数**（不是平均数） | 均值对离群值敏感；中位数对间歇性系统抖动稳健 |

**禁止的对比**：

- 跨 NPU（卡 A 跑基线、卡 B 跑候选）
- 跨 session（今天上午的基线 vs 下午的候选）
- 跨精度状态（基线是旧精度 PASS 版本，候选是新逻辑但精度未复验就先比性能）

**记录要求**：每组数字必须附带 NPU ID、时间戳、shape、dtype、迭代次数、
中位数 + min/max。缺这些字段的性能数字不可用于采纳/回滚判定。

## 2. 环境前置检查（排除假性快慢）

共享 NPU 环境下，**采集前必查**：

```bash
npu-smi info | grep -E "AICore|HBM"
```

- 目标卡 AICore 利用率 > 50%（有其他任务）→ **换卡或等待**，不要在争用状态下采集
- 争用状态下采到的"性能下降 30%"与 kernel 无关，写入 perf_log 会污染整轮结论

其他环境陷阱：

- **前一次 hang 的 kernel 会污染设备状态**（残留任务未清理）：hang 之后换卡或等待
  超时完全清除，再采下一轮
- 连续多 case 验证时，**一个 case 的 hang 会让同进程后续 case 出现连锁错误**——
  多 case 场景每 case 独立子进程 + 超时隔离

## 3. Kernel time vs Wall time（两个口径都要报）

| 口径 | 测的是什么 | 工具 | 适用结论 |
|------|-----------|------|---------|
| **Kernel device time** | 设备上 kernel 本身的执行时长 | `msprof op`（task duration） | "这个 kernel 快了多少"——优化归因用 |
| **E2E wall time** | Python 侧看到的端到端时长（含 launch、数据搬运、wrapper 开销） | 测试脚本计时 | "用户看到的快了多少"——交付结论用 |

**为什么必须分开**：kernel 优化收益可能被 wrapper 开销淹没。

> 典型模式：kernel 优化 −24%（msprof 确认），但 e2e wall 中性——因为 wall 中
> kernel 只占 1/3，其余是 launch + host 侧数据准备 + wrapper。此时：
>
> - 报 "kernel −24%" 是真的，报 "端到端无收益" 也是真的
> - 下一步优化方向在 **host 侧**（`optimization-guide.md` §2.12 Host 侧预处理内化），
>   而不是继续改 kernel

**归因规则**：wall 的变化与 kernel 的变化不一致时，先分解 wall（host 前处理 /
launch / kernel / 回拷），再把 gap 归到正确的组件上。**不允许**只看 wall 就得出
"优化无效"的结论回滚一个 kernel 层面真实有效的改动。

## 4. msprof 指标 → Bound 分类 → 优化章节路由

采集 `msprof op` 后，按以下路由表定位优化方向（不要凭感觉选优化点）：

| msprof 现象 | Bound 类型 | 优先优化方向 | 参考章节 |
|------------|-----------|-------------|---------|
| Cube 利用率高、MTE2 busy 高 | MTE2 bound（GM→L1 带宽） | pingpong / mte2_preload / fullload | [cube_advanced_strategies.md](cube_advanced_strategies.md) |
| MTE2 busy 高但**带宽利用率 <70%** | **假 MTE2 bound**（小块多次搬运） | scale_coalescing | [cube_advanced_strategies.md](cube_advanced_strategies.md) |
| 各 pipe busy 都 ≤70%，但流水图有间隙 | 准无 bound（流水气泡） | mte2_preload / num_stages 加深 | [cube_advanced_strategies.md](cube_advanced_strategies.md) / `optimization-guide.md` §3.1 |
| Cube 忙、Vector 闲（或反之）且 C/V 融合 | 核间气泡（锁步） | C/V overlap / T.Pipelined | `performance-antipatterns.md` CV overlap 节 / [cv_sync_debug.md](cv_sync_debug.md) §4 |
| aiv_scalar（标量）占比高 | 标量 bound | 指令向量化、批量 API | `optimization-guide.md` §2.6 |
| Vector 内逐元素循环明显 | 向量化不足 | Tile API 替换 | `performance-antipatterns.md` 逐元素 for loop 节 |
| GM 带宽未打满且搬运量大 | 搬运效率 | L1 常驻 / Double Buffer / 增大 tile | `optimization-guide.md` §2.2 / §2.11 |
| 任务块数 < 物理核数 | 核利用率不足 | 减小 tile 增块数 / Split-K / streamk | [cube_advanced_strategies.md](cube_advanced_strategies.md) |
| 同步-only 耗时占比 >50% | 同步开销 | cross_interval / 减少握手次数 | `optimization-guide.md` §3.2 / [cv_sync_debug.md](cv_sync_debug.md) §1 |

**"假 bound"提醒**：busy 高 ≠ 带宽打满。MTE2 busy 90% 但带宽利用率 50% 说明
搬运的是大量小块（每次搬运有固定开销），优化方向是**合并搬运**而不是
"加大带宽"。判定真假 bound 是选错优化方向的最大来源。

## 5. 上限分析（到顶就收手）

连续迭代无提升时，先回答"**理论上还剩多少空间**"，再决定继续还是收手。

### 5.1 算术强度拆解

把算子拆成阶段，估算每阶段的耗时占比（可用注释法：注释掉某阶段测差值，或
msprof 分 pipe 占比推算）：

```
最大可达加速比 ≈ 1 / (瓶颈阶段优化后的占比 + 不可优化阶段占比)   （Amdahl）
```

> 例：某融合算子拆分为 GEMM 阶段（85%）+ epilogue 阶段（15%）。epilogue 即使
> 优化到零耗时，总加速上限也只有 1/0.85 ≈ 1.18×。已实测 1.12× → **剩余空间
> ~5%，收手并写明结论**，不要再开 3 轮迭代挤 2%。

### 5.2 对照硬件 roofline

- 算一算算术强度（FLOP/byte）：低于硬件拐点的算子是 **memory bound**，其上限是
  带宽，不是算力——在 memory bound 算子上做 compute 侧优化（指令融合、MMA 调优）
  是无效方向
- 搬运量理论下限（输入+输出+必要中间量）× 带宽倒数 = 时间下限；实测已接近下限
  → 收手

### 5.3 收手结论的写法

收手不是"放弃了"，要写成可审计的结论：

```
已达上限判断：
- 阶段拆解：GEMM 85% / epilogue 15%
- 理论上限：1.18×（epilogue 归零）
- 当前实测：1.12×
- 剩余空间：~5%，低于继续迭代的期望收益
- 结论：终止 Stage 3，建议方向转为 XXX（若未来要做）
```

## 6. 结论报告模板

每轮 perf_log 的性能数字部分必须满足以下格式（缺字段的数字无效）：

```
- npu_id / 时间戳 / shape / dtype
- baseline: N 次中位数 X ms（min a / max b）
- candidate: N 次中位数 Y ms（min c / max d）
- kernel time（msprof task duration）：baseline Xk → candidate Yk
- wall time：baseline Xw → candidate Yw
- 环境：采集时 npu-smi 目标卡空闲（或其他状态说明）
- 口径不一致说明：wall 与 kernel 变化方向/幅度不一致时的分解结论
```

---

## 参考

- bound 分类后的具体优化手段：`optimization-guide.md`、
  [cube_advanced_strategies.md](cube_advanced_strategies.md)
- C/V 同步开销测量（同步-only 测试）：[cv_sync_debug.md](cv_sync_debug.md) §1
- profiling 工具字段细节：`ops-profiling` skill（msprof-guide / csv_fields_reference）
