---
id: X49
origin: discovered
discovered_round: 2
discovered_from: 2_FFN_evo/round_2/parallel_2 (x0_x1 open_exploration)
base_speedup: 0.275x → 0.475x (部分测量, +44% vs round1 最优 0.331x; case0 +94% vs 父、case1 +218%)
---

# Strategy X49: 两阶段 GEMM 跨核生产/消费切分（N-split stage-split + GM-volatile flag）

## 核心思路
两阶段 GEMM 链（FFN 的 GEMM1→act→GEMM2、QK^T+PV 等）若按 M-block 归属（X48）或 (mi,ni) tile 分任务，阶段一（GEMM1）的并行度受 mBlocks 限制（小 M 大 N2 时仅 4-8 核工作，其余空转，cube_utilization 仅 8%）。**N-split stage-split**：把"h 的生产"与"h 的消费"拆到不同核——
- **生产者**：K2 切成 P 片，核 0..P-1 各产一片。AIC: GEMM1(x[mi]@W1[:,slice]) → AIV: act → h[mi][slice] 写 GM；写完后 AIV 用 **GM volatile flag 广播** flag[mi][p]=1（release）。
- **消费者**：所有 G=min(24,nBlocks) 核各领 N2 列块 R_c。AIV 轮询 flag[mi][0..P) 全 1（GM volatile 非破坏读，acquire）→ H_READY；AIC: GEMM2(h[mi]@W2[:,R_c])。
- **模式选择**：mode1(N-split) 当且仅当 nBlocks > mBlocks（N-split 并行度 G 高于 M-split 的 mBlocks），否则 mode0(M-block 归属保底)。G=min(24,nBlocks) 保证每核 ≥1 输出 tile 规避 tile-less flag 死锁。

## 适用场景
- 两阶段 GEMM 链中阶段一为单核/低并行瓶颈（mBlocks < 核数，如小 M 大 N2 的 FFN / QK^T+PV）
- 已有 M-block 归属（X48）但并行度仍受 mBlocks 限制时
- 与带宽路径（P92 baseN 放大 / P93 K 轴错峰）正交可组合

## 实现要点
1. **跨核同步协议（规避竞态）**：生产者写 h（MTE3）后 `PipeBarrier<PIPE_ALL>` 再写 flag（release）；消费者 MTE2 读 flag（acquire）后才让 AIC 读 h。flag 槽 32B 对齐、每 M-block 独立、无需 reset。
2. **flag 缓冲必须零初始化**：用 `at::empty`（垃圾残留）会把垃圾 1 误判为"已就绪"，导致 h 未写完就消费 → 必须 `at::zeros` 初始化 workspace。
3. 生产者与消费者的 AIC/AIV 角色分离；P=min(G, K2/128)。
4. 跨 mi 循环串行（produce[mi]→wait→consume[mi]）未做软件流水加深——后续用三缓冲思想（P20）隐藏。
5. 910B2C 无 GM 原子（SetAtomicAdd 不支持），用 GM volatile flag 轮询替代原子计数。

## Trade-off
- flag 轮询有 GM 读开销（P×log 次/mi）；mode1 仅覆盖 nBlocks>mBlocks 的 case
- 共享 h 的跨核写→读需严格 release/acquire 协议，否则间歇性竞态（X48 曾遇 NaN）

## 来源
自动发现于第 2 轮进化，算子 FFN（910B2C），speedup 0.475x（部分测量），case0 [128,1024] 0.626x（父 0.323）、case1 [256,2048] 0.321x（父 0.101）。与 X14（GM-flag 跨核同步原语）同源，本策略聚焦两阶段 GEMM 的 stage-split 拓扑与模式选择。
