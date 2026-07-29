# Cube 核算子高级优化策略路由（pingpong / swat / streamk / fullload / scale_coalescing / mte2_preload）

> **适用场景**：Cube 型（GEMM/MatMul/Linear 族）或 C/V 融合算子的 **Cube 阶段**
> 已完成基线优化（`best-practices/cube_optimization_path.md` / `gemm_intrinsic_optimize.md`）
> 但性能仍不达标时，按本文档的路由表选择下一层策略。
>
> **使用前提**：先按 [measurement_discipline.md](measurement_discipline.md) §4
> 完成 bound 分类——**选错策略的第一原因是没有先判 bound 类型**。
>
> 本文档的策略机制蒸馏自 `ascendc-performance-best-practices` skill 的
> `reference/matmul/` 系列设计文档（更完整的推导与容量公式见该 skill），
> 此处补充各策略在 **TileLang DSL 中的表达方式**。

---

## 目录

- [1. 策略路由总表](#1-策略路由总表)
- [2. pingpong（基线，其他策略的前置）](#2-pingpong基线其他策略的前置)
- [3. swat（尾碎片负载不均）](#3-swat尾碎片负载不均)
- [4. streamk（欠并行 + 长 K）](#4-streamk欠并行--长-k)
- [5. fullload（小侧矩阵驻留 L1）](#5-fullload小侧矩阵驻留-l1)
- [6. scale_coalescing（假 MTE2 bound）](#6-scale_coalescing假-mte2-bound)
- [7. mte2_preload（准无 bound 的最后一口）](#7-mte2_preload准无-bound-的最后一口)
- [8. 策略组合与冲突](#8-策略组合与冲突)

---

## 1. 策略路由总表

| 策略 | 激活条件（定量） | 核心手段 | TileLang 表达层 |
|------|----------------|---------|----------------|
| **pingpong** | 所有 matmul（默认前置） | L1/L0 双缓冲 + 事件流水 | Developer 模式多缓冲 / Expert T.mma 手工 |
| **swat** | `totalBlockCnt > aicNum × 数倍` 但尾碎片负载不均 | Serpentine 行窗口 + 尾块合并 + 末轮二维再切分 | 任务分配层（`T.ceildiv` 映射/swizzle） |
| **streamk** | `totalBlockCnt < aicNum`（欠并行）+ `K ≥ 2048` | 切 K 到空闲核，AIV 从 workspace 归约 | 需 Expert 模式 + workspace 归约 |
| **fullload** | 一侧矩阵 ≤ L1/2 + 对侧循环次数 ≥ 2 + 真 MTE2 bound | 小侧一次性驻留 L1，消除重复 GM→L1 | tiling 层（L1 buffer 规划） |
| **scale_coalescing** | **假 MTE2 bound**（busy 高、带宽利用率 <70%）+ 存在 <20KB 小块 | K 向小块合并成一次大 MTE2 | copy 粒度层 |
| **mte2_preload** | pingpong 已开 + 各 pipe busy ≤70% + 流水图 MTE2 PING/PONG 有 gap + `kL1TileNum ≥ 2` | 主循环改三段结构（首轮 PING / 预取 PONG / 消费） | 循环结构层 |

**决策顺序**：先 pingpong → 按 bound 分类选一个主攻策略 → 剩余气泡用 mte2_preload 收尾。

---

## 2. pingpong（基线，其他策略的前置）

**机制**：L1 和 L0 双缓冲，MTE2（GM→L1）、MTE1（L1→L0）、M（MMA）、FIX（L0C→GM）
四级 pipe 事件同步流水化，当前计算用 buffer 0 时下一轮搬运到 buffer 1。

**关键参数**：`L1_BUFFER_NUM` 取满足 L1 容量预算的**最大整数**（L1 预算 ÷ 单份
buffer 大小，向下取整）。

**TileLang 表达**：

- Developer 模式（`T.gemm_v0`）：通过 buffer 声明的份数 + 自动流水实现，
  可调的是 L1/L0 tile 尺寸与缓冲份数
- Expert 模式（`T.mma`）：手工分配 ping/pong 两份 buffer + 显式 flag
  （见 `best-practices/gemm_intrinsic_optimize.md` §1-2）

**判定已生效**：msprof 流水图上 MTE2 与 M 段重叠，无长串行交替。

## 3. swat（尾碎片负载不均）

**机制**：解决 non-full-load 路径的三个负载问题——

| 子机制 | 解决什么 |
|--------|---------|
| A. Serpentine 行窗口遍历（`WINDOW_LEN=4`） | B 侧流式加载的跨核复用：相邻核按蛇形推进，B 片段在 L2 被复用而不是反复失效 |
| B. 尾块合并 | `M % baseM` / `N % baseN` 产生的薄片与满块同行时，薄片收益被满块拖没——重排尺寸 |
| C. 末轮二维再切分 | `totalBlockCnt % blockNum ≠ 0` 时末轮空闲核——把末轮块再切小摊给空闲核 |

**激活条件**：`totalBlockCnt > aicNum × 数倍`（并行度本身够，问题在不均）。
欠并行时不要用它，走 streamk。

**TileLang 表达**：属于**任务分配/调度层**，通过 block index → tile 的映射函数
（swizzle 顺序）表达；Expert 模式可完全手工，Developer 模式看 DSL 是否暴露
调度原语，不暴露时此项只能改 host 侧任务划分。

## 4. streamk（欠并行 + 长 K）

**机制**：`totalBlockCnt < aicNum` 时有核空闲。把 K 维切开分给空闲核，每个核算
部分 K，**AIV 从 GM workspace 做跨核归约**（这是天然的 C/V 协作点：Cube 算部分积，
Vector 归约）。

**两个子模式**：

- **SK（纯 streamk）**：所有块都切 K
- **DP+SK**：能整分的块走正常数据并行，余数块切 K——通常优于纯 SK

**激活条件**（同时满足）：

- `totalBlockCnt < aicNum`
- `K ≥ 2048`（K 够长，切开后每份仍有足够计算量摊薄归约开销）
- 或 baseM/baseN 已到 CUBE 粒度下限的 MN 不整除情形

**TileLang 表达**：需要 Expert 模式——workspace 声明、部分积写出、归约 kernel
段都要手工；归约段的正确性依赖跨核写后读的同步（见
[cv_sync_debug.md](cv_sync_debug.md) §2.3 GM 可见性）。

**注意**：streamk 的归约引入额外 GM 流量，K 不够长时归约开销会吃掉并行收益——
这就是为什么激活条件里有 `K ≥ 2048`。

## 5. fullload（小侧矩阵驻留 L1）

**机制**：GEMM 有一侧矩阵足够小（≤ L1 容量的一半）时，把它**一次性完整搬入 L1
驻留**，对侧循环中不再重复 GM→L1 搬运它，MTE2 流量直接砍半量级。

**激活条件**（三个同时）：

1. 一侧矩阵 ≤ L1/2（要留另一半给对侧的 pingpong）
2. 对侧循环次数 `T ≥ 2`（只循环一次时没有重复搬运可省）
3. **真** MTE2 bound（带宽利用率高，见 §6 的假 bound 鉴别）

**TileLang 表达**：tiling 层的 L1 buffer 规划问题——把常驻侧声明为单份大 buffer、
循环外一次搬入；注意与双缓冲的 L1 预算冲突（`L1_BUFFER_NUM` 要重新核算）。

## 6. scale_coalescing（假 MTE2 bound）

**机制**：量化 matmul 的 scale/bias/LUT 常是 <20KB 的小块，按 `baseK` 逐块搬运时
每次搬运有固定开销——MTE2 **busy 很高但带宽利用率 <70%**（假 bound）。把 K 方向
相邻的小块**合并成一次大搬运**（`scaleKL1 = SCALE_L1_BUFFER_NUM × kL1`），
搬运次数下降一个量级。

**假 bound 鉴别**（使用本策略前的必做检查）：

```
MTE2 busy 高（如 90%）+ 带宽利用率 < 70%  →  假 bound → 合并搬运
MTE2 busy 高 + 带宽利用率 > 70%           →  真 bound → fullload / 减小搬运量
```

**TileLang 表达**：copy 粒度层——把循环内多次小 copy 改成循环外/少次大 copy +
视图切片；注意合并后单次搬运要 ≥16KB 量级才能接近带宽上限（小块永远打不满）。

## 7. mte2_preload（准无 bound 的最后一口）

**机制**：pingpong 已开、各 pipe busy ≤70%（无明确 bound），但流水图上 MTE2 的
PING/PONG 段之间仍有 gap。把主循环改造成**三段结构**：

```
段 1：首轮 PING 加载
段 2：预取 PONG（在消费 PING 之前先发出下一轮搬运）
段 3：消费
```

零 tiling 数值改动，纯循环结构调整，把 MTE2 提前一拍发出。

**激活条件**（全部满足）：

- pingpong 已开且已确认生效
- 各 pipe busy ≤ 70%（有 bound 时先治 bound，不要先做这项）
- 流水图可见 MTE2 PING/PONG 间 gap
- `kL1TileNum ≥ 2`（只有一轮搬运时无预取对象）

**TileLang 表达**：循环结构层——Expert 模式手工排布 load/compute 顺序；
Developer 模式受 DSL 调度限制可能无法表达，此时标注 `[DSL-LIMIT]` 并记录
预期收益，不要在 DSL 里硬凑。

## 8. 策略组合与冲突

| 组合 | 是否兼容 | 说明 |
|------|---------|------|
| pingpong + 任何其他 | ✅ 必须 | pingpong 是所有策略的前置 |
| swat + streamk | ❌ 二选一 | 前者治"并行够但不均"，后者治"并行不够"，前提互斥 |
| fullload + scale_coalescing | ⚠️ 算预算 | 两者都占 L1，合并后的 L1 总占用必须重算 |
| streamk + C/V 融合 epilogue | ⚠️ 注意归约点 | streamk 的 AIV 归约与融合 epilogue 的顺序要显式设计，归约未完成前 epilogue 不能消费 |
| mte2_preload + 任何 | ✅ 最后做 | 它是榨取剩余气泡的收尾手段 |

**ORDER-PLAN 补充规则**（接入 SKILL.md Step 3 的依赖分析）：

1. bound 分类（measurement_discipline §4）永远排最前
2. fullload / scale_coalescing 改变 L1 预算 → 排在依赖 buffer 数量的优化之前
3. swat / streamk 改变任务分配 → 排在所有核内优化之前
4. mte2_preload 永远最后

---

## 参考

- 策略机制的完整推导与容量公式：`ascendc-performance-best-practices` skill
  `reference/matmul/`（guide.md 路由表 + 各策略 design 文档）
- 基线 cube 优化（双缓冲/tile 推导/陷阱）：
  [best-practices/cube_optimization_path.md](best-practices/cube_optimization_path.md)、
  [best-practices/gemm_intrinsic_optimize.md](best-practices/gemm_intrinsic_optimize.md)
- bound 分类方法：[measurement_discipline.md](measurement_discipline.md) §4
- streamk 归约的同步正确性：[cv_sync_debug.md](cv_sync_debug.md)
