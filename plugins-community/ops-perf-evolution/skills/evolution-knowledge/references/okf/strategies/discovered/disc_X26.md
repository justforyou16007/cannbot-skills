---
schema_version: okf.v1
kind: discovered_strategy
id: X26
status: active
confidence: single_op
op_families: [conv, deconv, scatter, gather]
bottlenecks: [mte2_stall, undersize_transfer]
tags: [col2im, tap-batch, pipeline-barrier, event-sync, dav-2201]
discovered_from: 10_ConvTranspose2d evolution round_1/parallel_2
---

# X26: Col2im Tap-Batch 事件化

## 手法本质
散射类算子（col2im/deconv）逐输出行按 tap 循环时，把每 tap 一次
`PipeBarrier<PIPE_ALL>`（~16/行）重构为 tap-batch：一次发出该行所有 tap 的
MTE2 加载 + 1 次 barrier，随后 V 管道 in-order 背靠背 Gather+Add + 1 次 barrier。
barrier 降到 2/行。

## 适用条件
- 逐行多 tap 散射，各 tap 加载无真实数据依赖
- 特征：mte2_instructions 高 + 单次搬运 <16KB（undersize_transfer）
- 前提：同一累加器的串行 Add 可在 V 管道 in-order 完成（V 保序）

## 关键纪律
1. 对齐 zero slot 在 V 管道一次性初始化（S/V 跨管道无保序）
2. 行级 B1/B4 保守保留 PIPE_ALL（MTE3 copyout WAR）
3. 逐 tap 事件化（MTE2_V/V_MTE2）替代加载等待段
4. 禁忌：strided 多块 DataCopyPad 在 dav-2201 挂死

## Evidence
10_ConvTranspose2d round_1/parallel_2：col2im 316us → 277us（Cube 化 GEMM 后
col2im 为新瓶颈，最差 274us vs GEMM 9us）；full-op geomean 0.417x → 0.535x。
