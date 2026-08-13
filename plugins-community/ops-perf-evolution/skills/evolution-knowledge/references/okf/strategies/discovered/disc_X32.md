---
schema_version: okf.v1
kind: discovered_strategy
id: X32
origin: discovered
discovered_round: 4
discovered_from: "21_GroupedQueryAttention evolution round 2-4 (n1_cont, x0_x1_x1, n4_proj_comb/stageA, x0_4)"
base_speedup: "大 fp32 case 早期 2-3x 观测为 msprof 采样伪影；round-4 全量复核：fp16 hi/lo 投影净收益在噪声带内（0.60 vs 0.68 parent）"
confidence: single_op
status: active
op_families: [matmul, attention, flash_attention]
bottlenecks: [mte2_stall, compute_bound, no_overlap]
tags: [fp32, projection-gemm, fp16-hilo, precision, cube, gqa, prefill, 3xTF32, mte2-bound, l1-double-buffer]
strategy_ids: [X32, P128]
---

# X32: fp32 投影 GEMM 的精度安全替代与真实瓶颈（MTE2-bound 修正）

## 核心发现（round-4 修正）
910B2C 上 GQA prefill 的 fp32 投影 GEMM 是 dominant 成本（case44 投影占 kernel 55%），
但 **profiling 证明它是 MTE2-bound（内存受限），不是 compute-bound**：
mac_ratio 0.245 / mte2_ratio 0.709 / scalar 0.612。64x64 固定块的 GM→L1 搬运是瓶颈。

**关键负结果**：P128 bf16 hi/lo（attention 精度通过）与 fp16 hi/lo 3-GEMM（投影）的
cube 吞吐翻倍**没有转化为端到端收益**——因为投影已内存受限。round-3 观测到的大 fp32 case
2-3x 提升（c44 0.097→0.254 等）经 round-4 全量复核确认是 **msprof 采样伪影**
（c44 1.69x / c47 0.97x 是采样假象，真实约 0.25-0.3x）。

## 仍然成立的结论
1. **精度路径**：P128 bf16 hi/lo 仅对 attention 过容差；投影近零输出 MERE 超 2^-13。
   fp16 hi/lo（~2^-21，K≤1472 MERE≈9e-6）是投影的精度安全替代（如需保精度）。
2. **投影 GEMM L1 双缓冲**（round-4 x0_4）：GM→L1 与 mmad 重叠，10 大 case +3-14%
   （c47 2201→1923us, c44 1997→1854us）——这是针对 MTE2-bound 的正确方向（隐藏搬运）。
3. **KV padding 消减**（round-4 x0_4）：fp32 nChunk 阈值自适应，中型 S 减 QK^T/PV 冗余列。

## 修正后的优化方向
投影 GEMM 既然 MTE2-bound，正确杠杆是**内存访问**而非 cube 精度：
- 增大投影 GEMM 的搬运块（64x64→更大 tile），减少 undersize transfer
- L1 双缓冲/预取隐藏 GM→L1 延迟（已验证 +3-14%）
- 消除 split/投影的 GM 往返（pack+split 融合，但仅小/中 shape 有效）

## 边界（实测否定项）
- **pack+split 全 GM 融合在大 fp32 shape 不可行**（慢 1.4-2.5x，已证回退）
- **attention K/V 双缓冲无收益**（真实瓶颈是 AIC/AIV workspace GM 往返）
- **fp16 hi/lo 投影单维度无净收益**（投影 MTE2-bound，cube 吞吐翻倍被内存瓶颈吞没）
- L0C/L0B 须按 max(nChunk, dimPad) 分配防越界（dimPad>nChunk 时越界写坏 c20/c41）

## 来源
21_GroupedQueryAttention 进化 round 2-4：
- n1_cont (P128 fp16 hi/lo 投影 split)：round_2/parallel_0
- x0_x1_x1 (pack+split 输出投影)：round_3/parallel_1
- n4_proj_comb (P128 bf16 attention + fp16 投影，全量 0.5994x, MTE2-bound 洞察)：round_4/parallel_0
- n4_proj_stageA (Stage A fp16 投影，全量 0.610x)：round_4/parallel_1
- x0_4 (L1 双缓冲 + KV padding + L0C/L0B 修复，10 大 case +7%)：round_4/parallel_2
