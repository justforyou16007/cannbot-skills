# GMM + 融合 Epilogue（分组矩阵乘 + 激活/量化）C/V 融合优化最佳实践

> **适用算子族**：Grouped Matmul（GMM）+ 融合 epilogue——dequant → activation
> （SwiGLU/SiLU/GELU）→ requant 三段式尾部，典型如 MoE 的 gating/up/down projection、
> A8W8 量化 GMM。
>
> **为什么单独成篇**：Flash Attention 的 C/V 融合是**双向**数据流
> （S: C→V、P: V→C、O: C→V 三个 workspace），而 GMM+epilogue 是**单向**数据流
> （Cube 产出 → Vector 消费加工），同步的主要维度从"双向协议"变成
> "**背压（backpressure）**"。两者的调优手法不同，FA 的经验不能直接搬。

---

## 目录

- [1. 数据流形态与 FA 的对照](#1-数据流形态与-fa-的对照)
- [2. 核心优化 1：背压深度调优（本族特有的杠杆）](#2-核心优化-1背压深度调优本族特有的杠杆)
- [3. 核心优化 2：epilogue 三段式的 Vector 侧实现](#3-核心优化-2epilogue-三段式的-vector-侧实现)
- [4. 核心优化 3：多变体分发（dtype × 格式 × transpose）](#4-核心优化-3多变体分发dtype--格式--transpose)
- [5. Host 侧 tiling 与 workspace](#5-host-侧-tiling-与-workspace)
- [6. 负载均衡：分组长度不均](#6-负载均衡分组长度不均)
- [7. 调优顺序与检查清单](#7-调优顺序与检查清单)

---

新增算子先对照 [串行基线](../cv-patterns/serial-baseline.md) 与 [槽位生命周期](../cv-patterns/pipeline-lifetime.md)；下面是调优候选，不预设负载必然 Cube-bound。

## 1. 数据流形态与 FA 的对照

| 维度 | Flash Attention | GMM + 融合 epilogue |
|------|----------------|---------------------|
| C/V 数据流方向 | 双向 3 通道（S/P/O） | 单向 1 通道（Cube 结果 → Vec epilogue） |
| 同步协议核心 | 双向握手、状态携带（online softmax 的 max/sum） | 就绪信号 + **背压** |
| 主要死锁模式 | 双向协议配错（见 [../cv_sync_debug.md](../cv_sync_debug.md)） | 无背压 → Cube 覆盖未消费数据；复用等待过早 → 可能限制 overlap |
| Cube/Vec 耗时比 | 随 shape、mask、tile 与平台变化，需测量 | 随 groupList、量化与激活变化，需测量 |
| 流水目标 | C、V 都持续忙 | 按实测瓶颈提高整体吞吐，控制复用等待 |

仅在测量确认 Cube-bound 时，以 Vector 跟上 Cube 为主要目标；Vector 或访存主导时要重新排序优化候选。分类方法见 [measurement_discipline.md](../measurement_discipline.md) §5。

## 2. 核心优化 1：背压深度调优（本族特有的杠杆）

**机制**：单向数据流下，Cube 生产 workspace 块，Vector 消费。必须有一条
**V→C 的"已消费"信号**（背压），否则流水加深后 Cube 会覆盖 Vector 还没读的
workspace。释放粒度取决于槽位协议：可以逐槽释放，也可在明确整组槽位都已消费后成组释放。槽位数不等于每 N 块才发一次信号；先画 ready/reuse 依赖，再讨论同步频率。

**背压深度的权衡**：

| 深度 N | 效果 |
|--------|------|
| 太小 | 可能因槽位不足产生等待；两个槽位也可实现有效 overlap，取决于阶段时长与协议 |
| 太大 | workspace 增长且可能降低缓存命中；更多槽位不能补足较慢阶段的吞吐 |
| 合适 | Cube 几乎不等，Vector 稳定消费，workspace 不超预算 |

**调优方法（扫描法）**：

1. 先以已验证的保守槽位协议跑通，再在内存预算内选择少量深度候选
2. 每档保持数学路径不变，测设备时间线、整段时间和 workspace 峰值
3. 按目标 case 的实测选择深度；存在缓存/带宽竞争时收益不一定单调
4. 同步-only 仅用于隔离协议问题，不据此设定通用的 <10% 验收线

**TileLang 表达**：显式区分 C→V ready 和 V→C release，ID 数量/复用时机依据槽位、参与核与目标 API 语义确定；不能只加入 `if (i+1) % N == 0` 就视为完整协议。ID 设计可参考 [flash_attn_optimize.md](flash_attn_optimize.md)，并检查实际生成代码。

**正确性红线**：背压信号缺失或晚于 Cube 的下一轮写 → 间歇性错误（竞态，
见 [../cv_sync_debug.md](../cv_sync_debug.md) §3 第 3 条）。**先对后快**：
先把每块一次的保守背压跑正确，再扫描加深。

## 3. 核心优化 2：epilogue 三段式的 Vector 侧实现

典型 epilogue = **dequant（int8×scale→fp16/bf16）→ activation（SwiGLU 等）→
requant（fp→int8 + per-token scale）**。

优化要点（按收益排序）：

1. **融合成单 pass**：三段不在 UB 和 GM 之间往返，全部在 UB 内链式完成。
   反模式：dequant 写 GM → 读回 → activation 写 GM → 读回 → requant
2. **activation 用向量化原语**：SiLU(x) = x·sigmoid(x) 手写 5 个 vec op
   （sigmoid 可用除法或查表近似，视精度要求）比逐元素 scalar 快一个量级——
   见 `../performance-antipatterns.md` "Vector Core 内逐元素 for loop"
3. **requant 的 per-token max 归约**：用 `WholeReduceMax` 类整行归约 API，
   不要逐元素比较循环；量化乘 1/127 而不是除 127（乘法更快且精度一致）
4. **dequant 的 scale 搬运合并**：scale 是小块，按
   [../cube_advanced_strategies.md](../cube_advanced_strategies.md) §6
   scale_coalescing 合并搬运

**UB 预算检查**：三段链式中间 buffer（dequant 输出 fp32 + activation 中间量 +
requant 输入）同时驻留 UB，tile 尺寸按最坏情况核算（`optimization-guide.md` §2.11）。

## 4. 核心优化 3：多变体分发（dtype × 格式 × transpose）

GMM 族的变体空间典型为 `dequantDtype(3) × weightFormat(2) × transB(2) = 12` 种组合。

**原则**：

- **编译期分发，不是运行期分支**：每个变体是独立的模板实例（TileLang 中是
  独立的 JIT 特化），运行期 if-else 会让所有变体都变慢
- **变体间共享骨架**：12 个变体的差异只在 load 格式和 dequant 常数，
  Cube 主循环和 epilogue 骨架必须保持同一份代码（参数化），不允许复制 12 份
  再分别改——维护灾难且容易改漏
- **host 侧 dispatch 表**： tiling key → 变体的映射集中在 host 一处，
  kernel 内部不再判断

**验证要求**：每个变体至少一个精度 case；变体间做**同 shape 对照**（同输入下
两个等价变体应 bit 一致或误差在量化噪声内），能快速发现某个变体实例改错。

## 5. Host 侧 tiling 与 workspace

- **BasicBlock 选取**：baseM/baseN/baseK 按 cube 侧标准推导
  （[cube_optimization_path.md](cube_optimization_path.md) §2.3），GMM 无特殊；
  但**每组（group）的行数不同**，tiling 必须按"各组独立切分 + 全局块索引连续"
  组织，kernel 内通过每组的起始偏移定位自己的块
- **workspace 预算**：列出实际活跃的全部 allocation（每槽尺寸 × 槽数 × 实例数 × dtype 字节，加对齐及 API scratch）。仅实现确实保留全量 accumulator 时计入 `Σ_groups(M_g × N) × sizeof(accum)`；流式槽位不能再无条件叠加一个全量矩阵。按 host tiling 与 kernel 地址一致核算
- **host 预计算**：各组的行偏移前缀和（`cum_out[G+1]`）在 host 算好传入，
  kernel 内不做分组长度的串行扫描

## 6. 负载均衡：分组长度不均

GMM 特有的问题：各组行数差异大（MoE 专家负载不均），按组静态分配核会导致
忙的核很忙、闲的核很闲。

**对策**（按实现成本排序）：

1. **全局块队列**：所有组的所有 BasicBlock 排成连续全局队列，核按
   `blockIdx` 动态领取——空闲核自然领走剩余块（最简单，通常够用）
2. **swat 式末轮处理**：末轮不均衡见
   [../cube_advanced_strategies.md](../cube_advanced_strategies.md) §3
3. **超长组 streamk**：单组就大到欠并行时，对该组切 K
   （[../cube_advanced_strategies.md](../cube_advanced_strategies.md) §4）

**报告要求**：负载不均造成的慢 case（个别组特别大）要单独报告该 case 的
耗时与占比，不允许用全体 case 的中位数掩盖（慢 case 可能是真实业务的典型输入）。

## 7. 调优顺序与检查清单

```
[ ] 1. 对齐完整合同，建立串行基线，测量 bound（measurement_discipline §4）
[ ] 2. 核对 ready/reuse 与 workspace；融合候选先通过精度和重复复用检查
[ ] 3. 依测量排序：Cube/tiling、槽位深度、epilogue 向量化、分组负载均衡
[ ] 4. 每轮只改变一个优化点，对比设备时间、整段时间与 workspace
[ ] 5. 上限分析（measurement_discipline §5）并记录实际收益与剩余瓶颈
```

**常见误区**：

- ❌ 未测量瓶颈就固定先调背压或先调 epilogue——先保证协议正确，再按数据排序
- ❌ 背压一次到位设很深——没有先验证保守背压的正确性，竞态 bug 混入性能调优轮
- ❌ 变体复制 12 份分别调——一个 bug 修 11 遍漏 1 遍
- ❌ workspace 用拍脑袋常数——shape 一变就溢出或浪费

---

## 参考

- C/V 同步正确性与调试：[../cv_sync_debug.md](../cv_sync_debug.md)
- Cube 策略路由（streamk/swat/fullload）：[../cube_advanced_strategies.md](../cube_advanced_strategies.md)
- 背压的 semaphore ID 分配基线：[flash_attn_optimize.md](flash_attn_optimize.md)
  "Flag 与 Semaphore ID 分配指南"
- A8W8 量化 matmul 的完整开发场景：`ascendc-blaze-best-practice` skill
  `references/scenarios/a8w8-quant-matmul-development.md`
- 测量与上限分析：[../measurement_discipline.md](../measurement_discipline.md)
