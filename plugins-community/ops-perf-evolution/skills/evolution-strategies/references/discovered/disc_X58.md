---
id: X58
origin: discovered
discovered_round: 5
discovered_from: "57_ParallelPolarizedSelfAttention evolution round_5/parallel_0 (r5a)"
base_speedup: "0.3225x → 1.1148x vs PyTorch（全量 50-case，49/50 valid，精度 PASS；相对起点 3.46×）"
---

# Strategy X58: N=1 matvec 链归约突破（多相位拆分 + RA 二叉树 ReduceSum + Broadcast 消标量 + PipeBarrier 松弛）

## 核心思路

通道/空间自注意力等 **N=1 matvec 链** 算子（每输出元素 = 权重的逐元素乘加链，本质纯 Vector 管道、Cube 因 M=1/N=1 不可用）曾长期被判"1.0x 物理不可达"。本次进化以四杠杆组合彻底推翻该判断：**① 多相位链拆分**（把长 matvec 链拆成 n 个阶段，消除 gN 倍冗余链重算，每块只读自己 slice 消冗余全 X 读）；**② RA 二叉树 ReduceSum**（把串行 Add 依赖链 O(S) 换成对半 log-depth 树归约，消灭串行 V-pipe 依赖）；**③ Broadcast 消灭 GetValue 标量广播**（S-pipe 标量逐元素读换 Brcb/Duplicate 向量广播，消除 scalar_pipe 瓶颈）；**④ PipeBarrier 松弛**（精确到 PIPE_V 而非 PIPE_ALL，让 MTE2 数据预取与 V 计算重叠）。叠加 fp16 X 数据路径（读流量减半 + UB 精确 upcast）与链 weight tile 双缓冲，把最优从 0.3225x 推到 **1.1148x**。

## 适用场景

- **N=1 matvec 链注意力/归约算子**（通道注意力、空间注意力、polarized attention、逐位置乘加链），Vector 管道 bound、Cube 不可用
- profiling 显示 `vector_pipe` 主导（aiv_vec_ratio 高）+ `scalar_loading/scalar_compute` 次高、mte2 有气泡但未饱和
- 已被"不可达"判断覆盖、需重新评估的纯 Vector 路径算子

## 实现要点

1. **7-phase 链拆分**：把通道+空间注意力长链拆成 7 个独立相位（Pass A/B/C 等），消除 gN 倍冗余链重算；n-split 使每块只读自己 n-slice。
2. **RA 二叉树 ReduceSum**：对 span 归约用对半 log-depth 树替代串行 Add。⚠️ 注意：vector-bound kernel 里带 WholeReduceSum 的 `ColReduceAllRA` 变体反而 +0.9% 回归（多一次全量归约），优先保留原 Level-2/对半树形态，勿盲目套全归约 API。
3. **Broadcast 替代 GetValue**：Pass B/C 的标量广播用 `BroadCast`/`Duplicate` 向量广播，消除 S-pipe 标量读串行（P99）。
4. **PipeBarrier 精确化**：被提升的 BroadTile 自带 PIPE_V 时，外层循环用 `PipeBarrier<PIPE_V>` 而非 `PIPE_ALL`，释放 MTE2 X-prefetch 与 V 计算重叠（r5a 关键，+~8-11%）。
5. **fp16 X 路径**：X 读走 fp16/bf16 减半流量，UB 内精确 upcast 回 fp32 计算。
6. **链 weight tile 双缓冲**：weight 链 tile 用 TQue 双缓冲，消除 MTE2/V 串行化。

**失败/回退方向**：host transpose 权重链（host 预处理计入 asc 时间 + case7/10 精度失败）；64-granularity c2-split（13 case 系统精度错误）；运行时自适应 X-tile（大-n UB 溢出）。

## 来源

自动发现于 57_ParallelPolarizedSelfAttention 第 5 轮进化（r5a，open_exploration 链 x0→x0b→r3b→r4b→r5a）。全量 50-case 权威验证 1.1148x（49/50 valid），精度 PASS。突破链每轮：0.6625→0.7049→0.872→1.031→1.1148x。
