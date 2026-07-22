# C/V 融合算子同步调试指南（Hang / 死锁 / 间歇性错误结果）

> **适用场景**：C/V 融合（AIC+AIV 混合）kernel 出现以下任一症状时使用本文档：
>
> - kernel **挂死**（运行超时、无任何输出、进程卡在 sync）
> - **间歇性**错误结果（同一输入多次运行结果不一致，或换 shape 后才错）
> - 精度正确但**性能异常低**（疑似锁步调度）
>
> 纯 Cube / 纯 Vector 算子不使用本文档。
>
> **与调优文档的分工**：`optimization-guide.md` §3 和 `best-practices/flash_attn_optimize.md`
> 告诉你"怎么同步是对的"；本文档告诉你"同步出问题后怎么定位是哪一类问题"。

---

## 目录

- [1. 诊断协议（先隔离变量，再下结论）](#1-诊断协议先隔离变量再下结论)
- [2. 死锁分类树（按命中率排序）](#2-死锁分类树按命中率排序)
  - [2.1 组屏障参与不对称（最高频）](#21-组屏障参与不对称最高频)
  - [2.2 set/wait 与数据 pipe 错配](#22-setwait-与数据-pipe-错配)
  - [2.3 GM 数据可见性缺失（flag 对了数据没到）](#23-gm-数据可见性缺失flag-对了数据没到)
  - [2.4 Flag / Semaphore ID 冲突](#24-flag--semaphore-id-冲突)
  - [2.5 自动 CV 同步失效场景](#25-自动-cv-同步失效场景)
- [3. 间歇性错误结果（不是 hang 但更难查）](#3-间歇性错误结果不是-hang-但更难查)
- [4. "正确但慢"：锁步调度诊断](#4-正确但慢锁步调度诊断)
- [5. 调试手段速查](#5-调试手段速查)

---

## 1. 诊断协议（先隔离变量，再下结论）

C/V kernel 的 hang 和间歇性错误，**直接读代码找 bug 的成功率很低**（状态空间是
"核数 × 子块数 × 流水级数 × shape"的笛卡尔积）。必须按以下顺序逐层缩小范围：

### Step 1：单核先行

把 launch 核数降到 **1 个 AIC + 其配对的 AIV**（最小 shape），确认单核语义正确：

- 单核就 hang → 问题在**单核内的同步逻辑**（pipe 配对、flag 初始化），与多核无关
- 单核正确、多核 hang → 问题在**核间/子块间协议**（组屏障、任务分配、zero-work core）

> 经验：单核先行可以把搜索空间砍掉一半以上。不要跳过多核直接调试。

### Step 2：最小化复现

把 shape 缩到能复现问题的最小规模（单 tile、单 K/V 块、loop=1~2），并确认：

- **稳定复现**（每次都 hang/错）→ 结构性问题，走第 2 节分类树
- **概率性复现**（10 次里错 1~2 次）→ 竞态，走第 3 节
- **无法复现** → 先怀疑环境问题（见 [measurement_discipline.md](measurement_discipline.md)
  的 NPU 争用检查），不要在不可复现的 bug 上建调查链

### Step 3：同步-only 测试（隔离同步开销与正确性）

**注释掉所有计算和搬运代码，只保留 C/V 同步原语**（set_cross_flag / wait_cross_flag），
其余用 dummy 数据：

- 同步-only 版仍然 hang → 同步协议本身错误（分类树 2.1 / 2.4 / 2.5）
- 同步-only 版正常 → 同步协议骨架正确，问题在数据通路（2.2 / 2.3）

这个测试同时给出**同步开销占比**（见 §4）——一鱼两吃。

---

## 2. 死锁分类树（按命中率排序）

### 2.1 组屏障参与不对称（最高频）

**机制**：C/V 融合 kernel 中，AIC 与 AIV 之间的 cross-core flag 是**组屏障**语义——
一个 AIC 配对的**所有** AIV 子块（MIX 单元通常 1 Cube : 2 Vector）都发出 set 之后，
AIC 侧的 wait 才完成（AND 归约）。任何一个子块不 set，AIC **永远等待**。

**典型错误写法**：

```python
# 错误 1：只有 0 号子块发信号 —— 1 号子块永远不到，AIC 死等
if vid == 0:
    T.set_cross_flag("FIX", SEM_ID)

# 错误 2：某个 shape 下 1 号子块没有任务（zero-work），早退跳过了 set
if my_rows == 0:
    return  # ← 早退分支没有补 set，AIC 死等
```

**识别特征**：

- 小 shape（只够填满一个子块）必现 hang，大 shape 正常 —— zero-work 子块型
- 改核数后 hang 与否变化 —— 子块参与数变化型

**修复原则**：**每个握手，所有配对子块必须对称参与**。生产者工作天然是单子块的
（数据不可行拆）时，要把生产工作**按行/块范围拆给两个子块**，每个子块写自己的
不相交切片再各自 set——而不是让一个子块空跑或跳过。

```python
# 正确：两个子块各自处理自己的切片，各自 set
process_my_slice(vid)          # vid=0 处理前半，vid=1 处理后半（无任务时也是空切片）
T.set_cross_flag("FIX", SEM_ID)  # 所有子块无条件到达
```

**注意**：这条规则对 `T.Pipelined` + `TL_ASCEND_AUTO_CV_SYNC` 自动同步同样适用——
自动同步只在**所有分支都到达同步点**时才能生成正确代码。循环内有核间早退
（不同子块循环次数不同）时，自动同步会静默生成不对称协议，必须改写为
先归一化迭代次数、再进流水。

### 2.2 set/wait 与数据 pipe 错配

**机制**：`set_cross_flag(pipe, id)` 中的 pipe 声明的是"等哪个 pipe 上的操作退休后再
置旗"。如果声明的 pipe 与实际搬运数据的 pipe 不一致，flag 会**先于数据就绪**置位，
消费者读到旧数据（错误结果）或触发后续连锁 hang。

**检查清单**：

| 数据通路 | 正确的 set pipe |
|---------|----------------|
| Cube 结果经 Fixpipe 到 workspace | `"FIX"` |
| AIV 经 DataCopy（UB→GM workspace） | MTE3 对应的 pipe |
| GM→L1/L0 供 Cube 消费 | MTE2/MTE1 对应 pipe |

- set 侧：声明的 pipe 必须是**实际写出数据的那个 pipe**
- wait 侧：声明的 pipe 必须是**消费者接下来要用的 pipe**（消费者用它读数据，不是生产者的 pipe）

> 编译能通过 ≠ pipe 配对正确。这类错配**编译期零告警**，是"编译干净但结果错"的
> 头号来源。用 `get_kernel_source()` 检查生成代码中每个 set/wait 的 pipe 参数
> 与其保护的数据通路是否一致。

### 2.3 GM 数据可见性缺失（flag 对了数据没到）

**机制**：cross-core flag 建立的是**控制依赖**（"你可以继续了"），不自动保证消费者
随后的 **GM 读**能看到生产者写的数据。当生产者的结果经 **GM workspace** 路由、
且消费者拿到 flag 后第一次接触数据就是 GM 读时，可能读到未刷新的旧值。

**症状**：间歇性错误（时序敏感），且**同一二进制换台机器/换个时段结果不同**。

**修复（二选一）**：

1. **结果经 UB 路由**：消费者从 UB 拿数据（UB 访问天然在 flag 保护的 pipe 链上）
2. **显式可见性 fence**：生产者在 set 前确保 GM 写完成（写 pipe 与 set pipe 一致，
   见 2.2），必要时减小 workspace 使其命中 L2（A2/A3 L2 约 201MB，连续且按核复用的
   workspace 交互更可靠也更快）

### 2.4 Flag / Semaphore ID 冲突

**机制**：flag/semaphore ID 是有限资源，且部分 ID 被框架/硬件保留。冲突表现为
"协议看起来完全正确但随机 hang 或结果乱"。

**检查清单**：

- 同一 kernel 内两个不同用途的握手**复用了同一个 ID**（复制粘贴高发）
- ID 与 TileLang 自动同步（`TL_ASCEND_AUTO_CV_SYNC`）内部分配的 ID 冲突——
  手写 cross flag 与自动同步混用时尤其要查 `get_kernel_source()` 里实际分配的 ID
- 双缓冲场景中，ping/pong 两侧的 ID 没有按 side 区分
- 查阅 `best-practices/flash_attn_optimize.md` 的 "Flag 与 Semaphore ID 分配指南"
  作为分配基线；新 kernel 从该表的扩展开始，不要随机选号

### 2.5 自动 CV 同步失效场景

`TL_ASCEND_AUTO_CV_COMBINE` / `TL_ASCEND_AUTO_CV_SYNC` 在以下场景**推断不出正确
协议**（不报错，静默生成错误/低效同步）：

| 失效场景 | 后果 | 处理 |
|---------|------|------|
| 同步点在条件分支内（不同核走不同分支） | 组屏障不对称 → hang（2.1） | 归一化分支，同步点提出分支外 |
| 循环携带的跨迭代依赖（累加器跨迭代复用） | 自动流水级数推断错误 | 手写 flat pattern，不用嵌套 `T.Pipelined` |
| 嵌套 `T.Pipelined` | 内层同步语义混乱 | 核间只留一层 `T.Pipelined`，核内 double buffer 手写 |
| 生产者/消费者粒度不一致（一次 cube 产出供多次 vec 消费） | 握手次数不匹配 | 改成显式 semaphore 计数或调整粒度对齐 |

**判定方法**：先关自动同步、手写最小握手跑通；再开自动同步对比 `get_kernel_source()`
的生成差异。生成代码与自己预期的协议不一致时，以手写为准。

---

## 3. 间歇性错误结果（不是 hang 但更难查）

间歇性 = **竞态**。按命中率排序：

1. **2.3 GM 可见性**（时序窗口型）——最高频
2. **2.2 pipe 错配**（flag 先于数据）——第二高频
3. **消费者还没读完、生产者就覆盖**（反向依赖缺失）：workspace 双缓冲的"释放"信号
   （V→C 方向）缺失或过晚。检查每个 C→V 的"就绪"信号是否都有配对的 V→C"已消费"
   信号（背压），单向信号的系统必然在流水加深后崩
4. **多 AIV 重复执行同一工作**：Mix 类型算子结果**数值加倍**是特征签名——
   缺 `if vid == 0:` 类 guard（见 `optimization-guide.md` §4 速查表）

**放大竞态的手段**（把低概率问题变成高概率再定位）：

- 加深流水级数 / 减小 tile（更多次握手 = 更多竞态机会）
- 同一 case 连跑 ≥20 次统计失败率
- 在关键路径插 `T.printf` 会改变时序（printf 本身是同步点），**printf 后问题消失
  是竞态的强证据**——改用写诊断值到 GM workspace 再读回的方式观察

---

## 4. "正确但慢"：锁步调度诊断

**机制**：C/V 融合算子精度正确但性能远低于预期，第一嫌疑不是 tile 参数，而是
**流水线塌陷**——Cube 算完 → Vector 算 → 下一个 Cube……串行乒乓，两个引擎各闲一半。

**量化确认**（两步）：

1. 用 §1 Step 3 的**同步-only 测试**测同步开销占比：注释全部计算后同步本身耗时
   占比 >50% → 同步粒度过细（用 `cross_interval=2` 降频，见
   `optimization-guide.md` §3.2）
2. 用 `get_kernel_source()` 检查生成的主循环：如果 AIC 段和 AIV 段在同一迭代内
   顺序出现、且没有 `T.Pipelined(num_stages≥2)` 的跨迭代交错 → 锁步确认

**修复**：`T.Pipelined` + `num_stages` 调优（`optimization-guide.md` §3.1）+
开启自动 CV combine/sync（`performance-antipatterns.md` "AIC/AIV 混合算子未开启
CV overlap"节）。

**判定上限**：如果 Cube 阶段耗时占比 ~90%（GEMM-bound 融合算子常见），锁步的损失
上限只有 ~10%，此时流水化收益有限，应把精力转回 Cube 侧
（[cube_optimization_path.md](best-practices/cube_optimization_path.md) /
[cube_advanced_strategies.md](cube_advanced_strategies.md)）——上限分析方法见
[measurement_discipline.md](measurement_discipline.md) §5。

---

## 5. 调试手段速查

| 手段 | 用途 | 注意 |
|------|------|------|
| `get_kernel_source()` | 检查生成的 set/wait pipe 参数、flag ID 分配、主循环是否锁步 | 每次改同步逻辑后必看 |
| 同步-only 测试 | 隔离同步协议 vs 数据通路；测同步开销占比 | §1 Step 3 |
| 单核先行 | 砍半搜索空间 | §1 Step 1 |
| `T.printf` | 观察执行顺序 | 本身是同步点，会掩盖竞态 |
| 诊断值写 GM workspace | 观察中间状态（不改变时序） | 用真实输出 tensor 槽位带回，别用 raw-byte workspace 读回（host 侧可能读不到设备直写的数据） |
| 连跑 ≥20 次 | 统计间歇性问题的失败率 | 概率 <5% 的竞态几乎无法靠单次运行定位 |
| `npu-smi info` | 排除环境争用导致的假性 hang/超时 | 共享机器必查，见 [measurement_discipline.md](measurement_discipline.md) |

---

## 参考

- 同步正确性写法基线：`optimization-guide.md` §3（核间优化）、
  `best-practices/flash_attn_optimize.md`（Flag/Semaphore ID 分配指南）
- 测量与环境排除：[measurement_discipline.md](measurement_discipline.md)
- C/V overlap 开启方法：`performance-antipatterns.md` "AIC/AIV 混合算子未开启 CV overlap"
- **下沉到原生 AscendC 层时**：`ascendc-crash-debug` skill 的
  `references/mix-hang-debug.md`（910B MIX_AIC_1_1/1_2 的 R1-R6 根因分类：
  错核执行守卫、`SyncAll<false>` 语义、InitBuffer 核趟、Mmad 队列事件、
  设备 wedge 判别、四阶段开发流程）。分工约定：本文档覆盖 **TileLang DSL 层**
  的同步问题（DSL 原语、自动同步失效、调优期的锁步/竞态）；当
  `get_kernel_source()` 显示生成的 AscendC 代码本身有守卫/barrier 结构错误、
  或 hang 需在原生 kernel 层复现排查时，以 mix-hang-debug.md 为准。
