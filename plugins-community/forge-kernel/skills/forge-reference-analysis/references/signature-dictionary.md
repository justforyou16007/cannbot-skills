# 签名字典：观测量 → 架构决策

本字典把 msprof 与配对评测的可观测量，翻译成**可以成立的架构决策假设**，
同时划出**不能由该观测量支撑的推断**。

两栏都重要。后一栏是防止过度解读用的：一个观测量能支撑的结论是有限的，
超出那个限度的断言会把 milestone 建在沙子上，而且因为它看起来有数据支撑，
出错时特别难查。

> 所有阈值均引自 [`ops/ops-profiling/references/csv_fields_reference.md`](../../../../../ops/ops-profiling/references/csv_fields_reference.md)。
> 架构事实引自 [`ops/npu-arch`](../../../../../ops/npu-arch/SKILL.md)。
> 字段实际存在性以该文档和 `evaluate_ops_direct.py` 的输出为准。

---

## 零、先排除测量不可信的情况

**这一步必须在任何架构推断之前做。** 下面任何一条成立时，所有占比数据都
不能用来推架构 —— 你看到的分布反映的是测量环境，不是参考实现的设计。

| 观测量 | 判据 | 含义 |
|---|---|---|
| `Current Freq` vs `Rated Freq` | Current < Rated | DVFS 降频。时间被拉长，各段占比失真 |
| `cv_pct`（评测输出） | 大于实测噪声底 | 读数本身不稳，先解决环境再谈分析 |
| `aiv_icache_miss_rate` | > 15% | 取指开销混进了各段占比 |

降频尤其隐蔽：它会让一个本来计算受限的 kernel 看起来像搬运受限，
因为计算段被拉长的比例和搬运段不同。

---

## 一、核间分工（对应论文的 warp 角色专门化）

### 观测量：`Block Dim` 与 `Mix Block Dim`

| 观测 | 可支撑的架构决策 |
|---|---|
| `Mix Block Dim` 为 `N/A` | 参考是**纯 Vector 或纯 Cube 算子**，没有 AIC/AIV 分工 |
| `Mix Block Dim` 有数值 | 参考是 **MIX 算子**，Cube 与 Vector 分工协作；该数值是从核 blockDim |
| `Block Dim` 远小于可用核数 | 存在**有意的核数限制**，通常因为单核要吃下某个常驻结构；这是一个架构决策，不是疏漏 |
| `Block Dim` 等于可用核数 | 切分策略以铺满核为目标 |

**不能由此推断**：
- 具体每个核分到多少数据（需要结合 shape 与搬运量）
- Cube 与 Vector 之间的流水编排方式（谁等谁、几级）

> `Mix Block Dim` 仅 A2/A3 有效，见字段文档说明。

### 观测量：`aic_time(us)` 与 `aiv_time(us)`

| 观测 | 可支撑的架构决策 |
|---|---|
| 两者量级接近 | Cube 与 Vector 负载被**刻意配平**，是一个编排决策 |
| 一侧远大于另一侧 | 另一侧是**附属角色**，主干在占用高的那侧 |
| `aic_cube_ratio ≈ 0` 但 `aic_time` 不为 0 | Cube 核被占用但没做矩阵运算，通常是**搬运或同步角色** |

---

## 二、搬运与数据复用（对应论文的 TMA / 操作数 staging）

### 观测量：`ai*_mte2_ratio`（GM → AICORE）

| 观测 | 可支撑的架构决策 |
|---|---|
| > 50%（MTE2 Bound） | 架构的**数据复用不足**：同一份数据被反复从 GM 取。可能的决策方向是提高驻留层级或加大复用粒度 |
| 低且 `aiv_vec_ratio` 高 | 参考已把搬运藏在计算后面，存在**有效的重叠编排** |

### 观测量：`aic_mte1_ratio`（L1 → L0A/L0B）

| 观测 | 可支撑的架构决策 |
|---|---|
| 显著非零 | 参考使用了 **L1 作为中间驻留层**，而不是 GM 直达 L0 |
| 接近零但 `aic_cube_ratio` 高 | 矩阵操作数走了别的通路，或 L1 驻留做得极少 |

### 观测量：`ai*_mte2_active_bw(GB/s)`

活跃带宽把「搬了多久」换算成「搬得多快」。

| 观测 | 可支撑的架构决策 |
|---|---|
| 接近该芯片的 HBM 带宽上限 | 搬运已达硬件极限，**再优化搬运方式无收益**，只能减少搬运总量 |
| 远低于上限但 MTE2 占比高 | 搬运效率有问题（如非对齐、碎片化），属**战术层**可改 |

这一对判断的区分很关键：前者要改架构（减少搬运量），后者只要调战术
（改搬运方式）。搞反了会在错误的层级上耗掉整轮迭代。

**不能由此推断**：具体的 tile 形状。带宽和占比只给出总量级，
同一搬运量可由多种 tile 划分产生。

---

## 三、输出布局

### 观测量：`aic_fixpipe_ratio`（L0C → GM/L1）

| 观测 | 可支撑的架构决策 |
|---|---|
| > 15% | 字段文档明确判为**地址未对齐**。这通常由输出张量布局决策导致，属架构层 |
| 正常范围 | 输出布局与硬件对齐要求相容 |

---

## 四、计算饱和度

### 观测量：`aic_cube_ratio`、`aiv_vec_ratio`、`aic_cube_fops`

| 观测 | 可支撑的架构决策 |
|---|---|
| `aic_cube_ratio` > 40%（MatMul 类） | 字段文档判为正常，Cube 利用合理 |
| VEC 算子的 `aic_cube_ratio` 明显非零 | 存在**意料之外的 Cube 使用**，值得追问参考为什么要用 |
| `aiv_vec_ratio` > 50% | VEC Bound。计算是瓶颈，搬运优化无用 |
| `aic_cube_fops` 折算利用率接近峰值 | 参考已接近算力上限，**超越它需要换算法**，不是换实现 |

最后一条是停止条件的重要输入：如果参考已经贴着 roofline，那么还想更快
就必须减少运算量本身 —— 那是算法契约层的决策，属于 M0。

---

## 五、标量开销

### 观测量：`ai*_scalar_ratio`

| 观测 | 可支撑的架构决策 |
|---|---|
| > 30%（SCALAR Bound） | 标量计算挤占了核时间。可能是**地址计算或分支逻辑进了内层循环**，属架构层的循环结构决策 |

> 标量瓶颈的具体改法见 `ascendc-performance-best-practices` 的
> `reference/scalar/guide.md`，那是战术层。

---

## 五之二、停顿分解（`ResourceConflictRatio.csv`）

这一份对应论文 NCU pass 里的 **stall decomposition**，是判断「谁在等谁」
的唯一直接依据。**它是上板独有数据，仿真拿不到** —— 所以用仿真数据做
参考分析时这一整段缺失，`signature_probe.py` 会因此拒绝输出架构假设。

### 等待占比

| 字段 | 含义 | 可支撑的架构决策 |
|---|---|---|
| `aic_cube_wait_ratio` | Cube 被阻塞占比 | Cube 在等操作数，供给侧（MTE1/MTE2）跟不上 |
| `aiv_vec_wait_ratio` | Vector 被阻塞占比 | Vector 在等数据或在等 Cube 交接 |
| `ai*_mte2_wait_ratio` | MTE2 被阻塞占比 | 搬入通道本身被堵，通常是带宽或冲突 |
| `ai*_mte3_wait_ratio` | MTE3 被阻塞占比 | 写回被堵，可能是输出未对齐或目标繁忙 |

**等待占比最高的单元通常是受害者，不是瓶颈。** 它在等谁，要结合各段占比
一起看。把受害者当瓶颈去优化，是这一步最常见的误判。

### 冲突细分

| 字段 | 阈值 | 可支撑的架构决策 |
|---|---|---|
| `aiv_vec_total_cflt_ratio` | <5% 良好，>15% 严重 | 总体冲突水平 |
| `aiv_vec_bankgroup_cflt_ratio` | <3% | **block stride 设置不合理**，多次访问落入同一 bank group |
| `aiv_vec_bank_cflt_ratio` | <3% | **UB 地址分配不合理**，多个操作数命中同一 bank |
| `aiv_vec_resc_cflt_ratio` | <5% | 多个计算单元争用同一执行单元，需错开流水编排 |
| `aiv_vec_mte_cflt_ratio` | <3% | 搬运与计算争用 |

**这意味着 buffer 布局不是完全不可观测的**：具体布局读不出来，
但「布局有没有造成冲突」是直接可测的，而且能区分是 bank 冲突
（地址分配）还是 bankgroup 冲突（步长）。

---

## 五之三、搬运量（`Memory.csv`）

占比只说「搬运占了多少时间」，搬运量说「到底搬了多少字节」。后者对
反推 tiling 策略更有力。

| 字段 | 用途 |
|---|---|
| `GM_to_UB_datas(KB)` / `UB_to_GM_datas(KB)` | Vector 侧实际搬运量 |
| `GM_to_L1_datas(KB)` / `L0C_to_GM_datas(KB)` | Cube 侧数据路径 |
| `read/write_main_memory_datas(KB)` | 主存读写总量 |

### 数据复用倍数

算出算子语义要求的**理论最小读取字节数**，与实测搬运量相除：

```
复用倍数 = GM_to_UB_datas(KB) × 1024 ÷ 理论最小字节数
```

| 结果 | 含义 |
|---|---|
| 接近 1 | 几乎没有冗余读取，参考的复用做得很好 |
| 明显大于 1 | 同一份数据被反复搬入，提高驻留层级或加大复用粒度有空间 |

这个数直接量化了「还有多少冗余可以消除」，是对参考 tiling 策略最有力的
一个约束 —— 虽然它仍不唯一确定 tile 形状。

`signature_probe.py` 的 `--min-gm-bytes` 参数就是用来传理论下界的。

---

## 六、两个数据源，别混用

本字典里的占比类指标**全部来自 msprof**，不来自配对评测。这一点容易搞错，
而搞错的后果是写出一个永远取不到值的 trigger 通道。

| 数据源 | 怎么拿 | 给出什么 | 不给什么 |
|---|---|---|---|
| 配对评测 `evaluate_ops_direct.py` | `--baseline-path` / `--evolved-path` | `time_us`（多次取中位）、`speedup`、`precision_passed`、`cv_pct`、`shape_results`、`gating` | **不给流水线占比** |
| profiler `ops-profiling` | `msprof_profile_run.sh` + `msprof_perf_summary.py` | `PipeUtilization.csv` 等各段占比、`OpBasicInfo.csv` 的 Block Dim 与频率 | 不做配对加速比判定 |

**配对评测不产出流水线数据是有意为之**，不是缺陷：它在源码里写明了理由 ——
`acl.prof.init` 每进程只能调一次，多 session 并发采集会互相抢资源导致计时不稳。
所以它改用 `torch.npu.Event` 事件计时，保证 baseline 与 candidate 的计时方法
完全一致、可比。代价就是没有流水线分解。

实际表现是：`evaluation_results.json` 里确实有 `pipeline` 和 `bottleneck` 两个键，
但它们**恒为 `{}` 和 `"unknown"`**。

### 对 milestone 的直接影响

trigger 的 `channel` 字段写法，按指标来源分两类：

```
性能类（增益、时间、精度）→ 写评测产物
  "channel": "evaluation_results.json:comparison.speedup"
  "channel": "evaluation_results.json:aggregate.target.precision_passed"

架构签名类（各段占比、核数、频率）→ 写 msprof 产物
  "channel": "msprof:PipeUtilization.csv:aiv_mte2_ratio"
  "channel": "msprof:OpBasicInfo.csv:Block Dim"
```

把架构签名的通道写成 `evaluation_results.json:*.pipeline.*` 是个常见错误，
它不会报错，只会永远读到空值，于是 mechanism line 永远无法证实 —— 而这
恰恰是最难发现的一类问题：门禁看起来在跑，实际什么都没验证。

### 噪声底直接取 `cv_pct`

配对评测对每个 shape 跑 `num_trials` 次事件计时，取中位数作为结果，
同时算出变异系数 `cv_pct`。这个数就是该次测量的噪声水平，不需要另外估计。

`noise_floor.py` 优先读它；读不到时才回退到「多次评测结果之间的离散度」。

---

## 七、开放问题：这些 msprof 分辨不了

把下列内容写成「观测到的架构决策」是**无依据的断言**，
`milestone_validate.py` 的 V2 规则会因为给不出 trigger 而拒掉它们。

| 分辨不了的 | 为什么 | 可以怎么办 |
|---|---|---|
| 具体 tile 形状 | 多种划分可产生相同的搬运总量与核数 | 用复用倍数缩小范围，再实现候选形状比接近度 |
| Buffer 的**精确布局** | 没有字段直接给出地址分配方案 | 布局本身读不出，但**它造成的冲突是可测的**（见五之二）；可据冲突类型反推布局缺陷 |
| double buffer 的具体深度 | 重叠的效果可观测，深度不可观测 | 把「存在重叠」写成决策，深度留给战术层扫 |
| 同步原语的具体用法 | 无对应字段 | 不写进架构决策，属战术层 |
| 重算与存储的边界位置 | 只能从搬运量变化间接感知 | 做偏重算、偏存储两版实现，比搬运量接近度 |
| 暴露的时间花在哪一段 | msprof 算子级无时间线视图 | 用 `lever_probe.py` 逐项消融定位 |

> **一处更正**：早期版本曾把「bank 分配」整体列为不可观测。这是错的 ——
> `ResourceConflictRatio.csv` 明确区分 bank 冲突与 bankgroup 冲突，
> 并给出各自阈值。不可观测的是**具体布局方案**，不是**布局有没有问题**。

> 论文的原话是：仪器分辨不了的，留作显式的开放问题。这不是退让 ——
> 一个诚实标注的开放问题可以被后续实验回答，一个伪装成结论的猜测不能。
