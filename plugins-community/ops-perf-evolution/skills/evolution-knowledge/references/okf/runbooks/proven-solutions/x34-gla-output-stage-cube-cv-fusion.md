---
schema_version: okf.v1
kind: proven_solution
id: x34
title: GLA/SSM 输出阶段 Cube CV 融合重写（P102 MIX_AIC_1_1 + hilo 精度分裂 + host 零布局预处理 + v_ws 尾块填充）
source: ChunkGlaFwdKernelO evolution round 1-3 (lingxi-evo)
speedup: "geomean 0.293x → 10.34x vs PyTorch（相对基线 ~35x）；round1 2.918x → round2 NVM 7.594x → round3 v_ws 10.34x；50/50 精度"
status: active
op_families: [ssm, gla, linear_attention, chunk-scan, cv_fusion]
strategy_ids: [P102, P133, P134, P25, P66, X24]
bottlenecks: [compute_bound, no_overlap, mte2_stall, undersize_transfer, launch_floor]
---

# X34: GLA 输出阶段 Cube CV 融合重写（MIX_AIC_1_1 + hilo 精度分裂）

GLA/SSM 系 chunked 输出阶段算子（`o[t] = tril(A)@v + scale*(q*exp(g))@h`，每 (batch,chunk,head) 独立任务）的
纯 AIV Vector 外积累加实现（count-8 Mul 模拟矩阵乘，每 BlockOuterAdd 带 PIPE_ALL）是 P102 标准猎物：
重写为 `KERNEL_TYPE_MIX_AIC_1_1`，Cube 三次 Mmad 融合到单 L0C（`qg_hi@h` init=1 + `qg_lo@h` + `A_m@v` init=0），
Fixpipe(F322F16) 直接写回 o；AIV 承担 qg=exp/mul 预处理与 tril mask；任务间 SyncAll 流水 + workspace 双缓冲。

## 关键经验（3 条）

1. **hilo 高低位分裂是 fp16 Cube 路径保精度的必备件**：qg = q*scale*exp(g) 的 fp32 中间量 cast fp16 进 Cube
   后，case0 MERE 超标；qg_hi/qg_lo 双分量两次 Mmad 累加后 50/50 全过。fp16 输入 + fp32 L0C 累加不够，预处理中间量的
   低精度部分（exp 后小数值）仍需 hilo 补偿（与 X17/X32 的 hi/lo 思想同源，此处用于 GLA 输出投影）。

2. **小 gemm MIX 内核的真实瓶颈迁移到 AIV MTE2**：两处 gemm 维度 ≤128（K 单块），Cube 完成极快（aic_mac_ratio 0.0127
   几乎空闲），瓶颈变成 AIV 预处理数据的 GM workspace 往返与短搬运（aiv_mte2_ratio ~0.41，Vector 计算仅 0.007）。
   下一杠杆 = P1 任务间双缓冲 + P66 大块 DMA 合并 + 减少 workspace 写入量（如评估 qg_lo 搬运可否精简），而非再加大 Cube。

3. **纯 AIV 向量路径天花板明显（对照组）**：同轮 P134+P1 任务流水（0.51x）与 BFR1A 全宽 count-32 向量化 + 批处理
   rank-1 累加（0.55x）均只到 ~2x 相对基线；Cube 化（2.92x）是唯一数量级突破。后续小 shape launch 地板可考虑 X24 AIC_ONLY。

## round2/3 演进：NVM + v_ws（7.594x → 10.34x，+36%）

> 在同算子后续轮次（lingxi-evo round 2/3）发现：**host 布局预处理是最大的隐藏固定开销**，比 kernel 内优化收益更大。

- **NVM（round2, 7.594x）— host 零布局预处理**：父 MIX 有 8 个 host 设备 op（4×pad + 4×permute + OnesLike/Tril）约 120µs/call，
  占大 case 时间 60%+。把 q/g/A 改原生 `[B,T,H,LD]` 布局直传 kernel，kernel 内 strided DataCopy/Nd2Nz 读取（L2 缓存 gather），
  host 设备 op 从 8 → 1（仅 v 的 constant_pad_nd）。kernel 为 latency-bound（非 throughput-bound），砍 host 固定成本是主杠杆。
- **v_ws 混合（round3, 10.34x）— 消除最后一个 host pad**：host 设备 op → 0。tail chunk（L<BT）的 v 零填充由 AIV 在 workspace 内完成
  （Duplicate 零填 BT*V + strided 读 L 行 + 写 workspace），full chunk（L==BT）AIC 直接从 GM 原生 strided 读 v（AIV 零额外开销）。
  AIV 仅写 L 行到 workspace（qg_hi/qg_lo 写 L*K、A_m 写 L*BT，Fixpipe 只写 L 行 + 下三角掩码使脏尾行无依赖）。
  小 case 近翻倍（case0 34→15.7us 5.43x），大 case +10-15%；min/max 3.42x/32.9x，无 case <3.4x。
- **AIC_ONLY 方向（X24）**：round2 n4 4.755x（host 去 expand8 + 三次 Mmad 融合 + AIC_ONLY 压 launch 地板），round3 n4_s1 4.79x
  （host op 16→12 + h 复用；L0/L1 双缓冲致中 case 回归已回退）。AIC_ONLY 下瓶颈变为 host 预处理 + launch 地板。
- **h 复用（候选）**：gemm2/3 共享一次 h GM 加载（省每 task K*V 读 + Nd2Nz），功能验证通过但因系统负载未完成测量。

## 适用条件

- GLA/GDN/SSM 系 chunked 输出阶段（任务 = batch×chunk×head，无跨核状态依赖，h 为每 chunk 独立初始状态）
- gemm 维度 K/V ∈ {32,64,128}，chunk_size 32/64
- fp16 输入 + fp32 中间精度要求（hilo 分裂需要）
- 910B2C（MIX_AIC_1_1、FixpipeParamsV220、F322F16 已验证）

## 来源

ChunkGlaFwdKernelO evolution round 1-3（lingxi-evo）：
- round_1/parallel_0：MIX_AIC_1_1 初版（2.918x）
- round_2/parallel_2：NVM（7.594x）；round_2/parallel_1：AIC_ONLY（4.755x）
- round_3/parallel_2：v_ws（10.34x，全局最优）
对照组：round_1 parallel_1（P134+P1 0.51x）、parallel_2（BFR1A 0.55x）；round_2/3 MIX 流水变体（P1/P10/P66/P133）因系统 OOM 未完成评估且不具竞争力。
