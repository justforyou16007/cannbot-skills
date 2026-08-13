---
schema_version: okf.v1
kind: proven_solution
id: x32
title: fp16 hi/lo 3-GEMM 用于 fp32 投影 GEMM + 投影 GEMM L1 双缓冲
source: GroupedQueryAttention evolution round 2-4
speedup: "大 fp32 case 2-3x; round4 L1 双缓冲再 +3-14%"
status: active
op_families: [matmul, attention, flash_attention]
strategy_ids: [X32, P128]
bottlenecks: [compute_bound, no_overlap, mte2_stall]
---

# X32: fp16 hi/lo 3-GEMM 用于 fp32 投影 GEMM + 投影 GEMM L1 双缓冲

fp32 投影 GEMM 3xTF32 是 GQA prefill dominant 瓶颈（case44 投影占 kernel 55%）。
P128 bf16 hi/lo 对投影近零输出 MERE 超 2^-13；fp16 hi/lo（~2^-21，K≤1472 MERE≈9e-6）
过容差 + fp16 cube 2x tf32，大 fp32 case 2-3x。round-4 投影 GEMM L1 双缓冲再 +3-14%。
pack+split 全 GM 融合仅小/中 shape 有效；attention K/V 双缓冲无收益（真实瓶颈是 workspace GM 往返）。
详见 evolution-strategies/references/discovered/disc_X32.md。
