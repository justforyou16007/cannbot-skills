---
schema_version: okf.v1
kind: proven_solution
id: x33
title: "attention workspace-ring 握手延迟 bound 的 Q-tile 放大 + 投影 GEMM group A-reuse"
source: GroupedQueryAttention evolution round 3
speedup: "全量 50-case 0.598->0.640-0.657x（vs round-4 prior-art +7-10%，vs 原始基线 0.4997 +28-31%）；case2 fp32 1.21x"
status: active
op_families: [attention, flash_attention, matmul]
strategy_ids: [X33, X24, P1]
bottlenecks: [no_overlap, mte2_stall]
---

# X33: attention workspace-ring 握手延迟 bound 的 Q-tile 放大 + 投影 GEMM group A-reuse

GQA prefill 的 attention 是跨核 AIC↔AIV workspace-ring 握手延迟 bound（各管道利用率<15%），
per-task Q tile 放大（32→64→128）每次把握手次数减半直接压延迟（case47 +24%、case35 +19%，
attention 318→265us）。fp32 投影 GEMM 是 MTE2(GM→L1) 管道饱和，用 by-group A-reuse
（GROUP_BY=6，A over-read 63x→11x，Stage-A 781→450us，mte2_ratio 0.746→0.693）。
配 X24（AIC_ONLY 纯 cube / AIV_ONLY 纯搬移）+ P1（L1 双缓冲）是最优组合
（全量 0.598→0.657x，case2 fp32 1.21x）。详见 evolution-strategies/references/discovered/disc_X33.md。
