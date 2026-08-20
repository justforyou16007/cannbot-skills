---
schema_version: okf.v1
kind: discovered_strategy
id: X33
origin: discovered
discovered_round: 3
discovered_from: "21_GroupedQueryAttention evolution round 3 (x0_x1_x1 Q-tile 128 / m2 X24+AIV_ONLY / n1_cont X24+P1)"
base_speedup: "全量 50-case 0.598->0.640-0.657x（vs round-4 prior-art +7-10%，vs 原始基线 0.4997 +28-31%）；attention Q-tile 32->64 单步 case47 +24%、case35 +19%；group A-reuse aic_mte2_ratio 0.746->0.693"
confidence: single_op
status: active
op_families: [attention, flash_attention, matmul]
bottlenecks: [no_overlap, mte2_stall]
tags: [attention, gqa, prefill, workspace-ring, handshake-latency, q-tile, mte2-bound, a-reuse, l1-double-buffer, x24, aiv-only]
strategy_ids: [X33, X24, P1]
---

# X33: attention workspace-ring 握手延迟 bound 的 Q-tile 放大 + 投影 GEMM group A-reuse

## 核心发现（round 3）
910B2C 上 GQA prefill 的 attention 是**跨核 AIC<->AIV workspace-ring 握手延迟 bound**
（aic/aiv 各管道利用率均 <15%，非带宽/计算 bound）。fp32 投影 GEMM 是
**MTE2(GM->L1) 管道饱和**（aic_mte2_ratio=0.693-0.746, cube 空置）。

两个有效杠杆：
1. **attention per-task Q tile 放大**（GQA_BLOCK_Q 32->64->128）：每次放大把
   AIC<->AIV workspace-ring 握手次数减半，直接压 attention 延迟 bound。
   32->64 单步：case47 0.320->0.398（+24%）、case35 0.387->0.460（+19%）、
   attention 设备时间 318us->265us（case44）。64->128 再减半握手，配合
   PV row-chunked（GQA_PV_CHUNK=64）控制 P L1 缓存宽度，AIV padding-row skip
   修复小 seq 的 padding 回归。
2. **投影 GEMM by-group A-reuse**（GQA_GEMM_GROUP_BY=6）：每个 task 算
   [64,6] 输出 slab，A tile 每 k-step 只载一次被 6 列 B 复用，A GM over-read
   从 ~63x 降到 ~11x，Stage-A QKV GEMM 781us->450us，aic_mte2_ratio 0.746->0.693。
   GROUP_BY=8（128KB L0C 满）在设备上 fault，用 6（96KB）。

## 关键负结果（延续 X32）
- fp16 hi/lo 投影是精度安全替代（~2^-21 过 MERE），非提速手段（MTE2-bound 吞掉 cube 增益）。
- P87 L0 重叠（L1->L0 与 mmad 并发）在 910B2C 上 AICore 死锁（f3 验证）；P104 块缩小回归；
  P66 K-batch 放大回归；P92 A-reuse group=2 中性（本变体 GROUP_BY=6 因网格足够大才有效）。
- 投影 GEMM 残余瓶颈是 B 侧权重搬运（case44 B:A≈12.7:1），A 侧优化已近极限。

## 适用场景
- attention/FA/GQA 类跨核 workspace-ring 流水 kernel，profiling 显示各管道利用率低（<15-30%）
  但耗时高 → Q-tile 放大减握手次数。
- MTE2-bound 投影/GEMM，A 侧 over-read 严重 → by-group A-reuse。
- **前提**：by-group A-reuse 只在投影 GEMM 是 MTE2-bound（aic_mte2_ratio 高、cube 管道明显空置）时有效；若瓶颈是 cube 计算或输出带宽，GROUP_BY 只会增加 L0C 压力并可能回退。
- 与 X24（AIC_ONLY 纯 cube / AIV_ONLY 纯搬移 kernel）叠加：X24 压 launch 地板，
  P1 L1 双缓冲压 GM->L1 延迟，三组合是 GQA 投影+attention 的有效配方。

## 实现要点
- Q-tile 放大需同步 AIV 每任务处理的 HQ-row 组数（GQA_HALF_Q / softmax tile），
  并处理小 seq 的 AIV padding 浪费（padding-row skip）。
- GROUP_BY 上限由 L0C 容量决定（GROUP_BY * BLK * BLK * 4B ≤ L0C），设备 fault 需降级。
- **先判瓶颈再开 A-reuse**：用 msprof 确认 aic_mte2_ratio 高且 cube 管道明显空置；若 cube 已饱和则不要增加 GROUP_BY。
- 与 P92（A-reuse）区分：P92 是 L1 cache 复用（中性），X33 是显式 task 级 group 复用。

## 来源
21_GroupedQueryAttention 进化 round 3：
- x0_x1_x1 (Q-tile 64->128 + PV chunk + padding-row skip，全量 0.640x)：round_3/parallel_2
- m2 (X24 AIC_ONLY 4 kernel + AIV_ONLY 8 kernel，全量 0.657x)：round_3/parallel_0
- n1_cont (X24+P1+AIV_ONLY，3-case spot 0.709x，case2 fp32 1.21x)：round_3/parallel_1
- 父证据：round_2/parallel_2 (x0_x1 Q-tile 32->64 + GROUP_BY=6，同子集 +10.6%)
