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

## 1. 数据流形态与 FA 的对照

| 维度 | Flash Attention | GMM + 融合 epilogue |
|------|----------------|---------------------|
| C/V 数据流方向 | 双向 3 通道（S/P/O） | 单向 1 通道（Cube 结果 → Vec epilogue） |
| 同步协议核心 | 双向握手、状态携带（online softmax 的 max/sum） | 就绪信号 + **背压** |
| 主要死锁模式 | 双向协议配错（见 [../cv_sync_debug.md](../cv_sync_debug.md)） | 无背压 → Cube 覆盖未消费数据；背压过紧 → 锁步 |
| Cube/Vec 耗时比 | 接近（需要精细交错） | Cube 通常远重于 Vec（GEMM-bound） |
| 流水目标 | C、V 都持续忙 | Cube 不等 Vec 即可（Vec 跟上就行） |

**推论**：GMM+epilogue 的性能天花板基本由 Cube 阶段决定（算术强度拆解见
[../measurement_discipline.md](../measurement_discipline.md) §5），Vector epilogue
优化的目标是"**不拖住 Cube**"，而不是"自身多快"。

## 2. 核心优化 1：背压深度调优（本族特有的杠杆）

**机制**：单向数据流下，Cube 生产 workspace 块，Vector 消费。必须有一条
**V→C 的"已消费"信号**（背压），否则流水加深后 Cube 会覆盖 Vector 还没读的
workspace。背压按**每 N 块一次**（而不是每块一次）发出，N 即背压深度。

**背压深度的权衡**：

| 深度 N | 效果 |
|--------|------|
| 太小（如 1~2） | 接近锁步：Cube 每写几块就等 Vector 确认，核间气泡大 |
| 太大 | workspace 需求随 N 线性增长；且 Vector 掉队太多时背压失去意义 |
| 合适 | Cube 几乎不等，Vector 稳定消费，workspace 不超预算 |

**调优方法（扫描法）**：

1. 从 N = workspace 预算允许的最大值开始（先保证正确性和最小 Cube 等待）
2. 逐步减半 N，每档测性能
3. 性能拐点（Cube 开始等待）的前一档为最优
4. 同步-only 测试（[../cv_sync_debug.md](../cv_sync_debug.md) §1）确认同步开销占比
   已 <10%

**TileLang 表达**：就绪/背压两个 semaphore（C→V ready + V→C release）+ 计数条件
（`if (i+1) % N == 0` 发背压）。ID 分配参照
[flash_attn_optimize.md](flash_attn_optimize.md) 的分配指南，两个方向各一个 ID。

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
- **workspace 公式**：`Σ_groups(M_g × N) × sizeof(accum) + 背压深度 × 块大小 +
  对齐预留`。**不要用拍脑袋常数**（如 "+20MB"），按公式算——shape 变化时
  拍脑袋常数要么浪费要么溢出
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
[ ] 1. bound 分类（measurement_discipline §4）——GMM 族通常是 Cube bound 或真 MTE2 bound
[ ] 2. Cube 基线：pingpong + tiling 推导（cube_optimization_path §2.3）
[ ] 3. C/V 单向流水 + 保守背压（每块一次）跑通正确性
[ ] 4. 背压深度扫描（§2）——确认 Cube 等待已消除
[ ] 5. epilogue 单 pass 融合 + 向量化（§3）——确认 Vector 不再拖住 Cube
[ ] 6. 负载均衡（§6 全局块队列）——分组不均时必做
[ ] 7. 上限分析（measurement_discipline §5）——Cube 阶段占比即天花板，到顶收手
```

**常见误区**：

- ❌ 先优化 epilogue 的向量指令，没做背压——Vector 再快，Cube 在等背压也白搭
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
