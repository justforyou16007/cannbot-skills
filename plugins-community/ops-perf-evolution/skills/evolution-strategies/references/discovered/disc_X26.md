---
id: X26
origin: discovered
discovered_round: 1
discovered_from: round_1/parallel_2
base_speedup: 1.28x (vs baseline 0.417x → 0.535x vs PyTorch)
op_family: conv (ConvTranspose/deconv col2im 散射类)
bottlenecks: [mte2_stall, undersize_transfer]
---

# Strategy X26: Col2im Tap-Batch 事件化（散射类 per-tap 流水重构）

## 核心思路

col2im/gather 散射类算子（ConvTranspose 的 col2im、deconv scatter）逐输出行
按 (kh,kw) tap 循环时，基线每 tap 一次 `PipeBarrier<PIPE_ALL>` 会同时排空
MTE2/MTE3/V 三管道，把 MTE2 加载与 V 计算完全串行化（~16 个 PIPE_ALL/行 →
~260us 全是 barrier 开销）。重构为 **tap-batch**：

- 一次循环内把该行所有有效 (kh,kw) tap 的 G 行加载全部发出（MTE2 loads 排队重叠），
  仅 1 次 `PipeBarrier`（等全部加载就绪）；
- 随后所有 Gather+Add 在 V 管道上 in-order 背靠背执行（V 管道天然保序，无 barrier），
  仅 1 次 `PipeBarrier` 保护下一行复用槽位（WAR）；
- barrier 从 ~16/行降到 2/行。

## 适用场景

- 逐输出行循环内有多个独立 tap/源片段、每 tap 一次全管道 barrier 的散射算子
  （col2im、deconv、scatter-add 类）
- 判定特征：`mte2_instructions` 高 + 单次搬运 <16KB（`undersize_transfer`）+ 各 tap 无
  真实数据依赖（同一累加器的串行 Add 也可在 V 管道 in-order 完成）

## 实现要点

- 对齐 zero slot（`src[wiA]`）必须在 **V 管道**一次性初始化，规避 S-pipe `SetValue`
  与 V-pipe `Gather` 的顺序问题（S/V 管道间无保序，跨管道操作必须显式 barrier）。
- 行级 B1/B4 保守保留 `PIPE_ALL`：实测换成 `PIPE_MTE3/PIPE_V` 均破坏正确性
  （MTE3 copyout 与下一行 V 填充竞争同一 UB 区）。
- 逐 tap 事件化（`MTE2_V`/`V_MTE2` HardEvent）可替代行级 PIPE_ALL 的前半段
  （加载等待），后半段（copyout WAR）仍需保留行级 barrier。
- 禁忌：strided 多块 DataCopyPad（blockCount>1 合并多个 G 行）在 dav-2201 挂死硬件，
  勿用（需先最小探针验证）。

## 来源

自动发现于第 1 轮进化，算子 10_ConvTranspose2d，col2im 316us → 277us；
Cube 化 GEMM 后 col2im 成为新瓶颈（最差 274us vs GEMM 9us）。
