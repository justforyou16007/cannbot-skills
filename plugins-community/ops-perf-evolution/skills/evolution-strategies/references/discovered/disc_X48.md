---
id: X48
origin: discovered
discovered_round: 1
discovered_from: 2_FFN_evo/round_1/parallel_2 (x0 open_exploration)
base_speedup: 0.153x → 0.275x (geomean 22 case, +80%；大 shape case4 [1024,1536] 0.074→0.252 3.4x / case18 6.7x / case21 6.5x)
---

# Strategy X48: 链式 GEMM M-block 归属调度（第一阶段输出复用）

## 核心思路
两阶段 GEMM 链（FFN 的 GEMM1→激活→GEMM2、QK^T→softmax→PV、SwiGLU 等）在朴素实现中，若按 `(mi, ni)` 输出 tile 划分任务，每个 tile 独立执行完整串行链，则 **GEMM1（第一阶段）会被重复计算 nBlocks 次**（同一 M-block 的每个 N 输出 tile 各算一遍）。nBlocks 越大病态越慢（FFN case23 nBlocks=32 → 0.018x，nBlocks=20 → 0.037x）。

修复：**M-block 归属调度**——把 mBlocks 个 M-block round-robin 分给各核 group，每个 group 独占其 M-block 的全部 nBlocks 个 N 输出 tile，**GEMM1 每个 M-block 只算一次**，中间结果（激活后的 h）在 group 内复用，GEMM2 循环遍历本 group 的 ni tile。

## 适用场景
- 两阶段 GEMM 链：FFN (GEMM1→act→GEMM2)、QK^T+PV、SwiGLU、MoE FFN 等，第一阶段输出被第二阶段多个 N-tile 消费
- 大 N1（hidden/输出维大，nBlocks≥16）时收益最大（冗余倍数= nBlocks）
- 小 M（mBlocks < 核数）时并行度受限，需配合其他多核手段（见 trade-off）

## 实现要点
1. `BlockScheduler` 改写为 M-block 归属：`mBlocks` round-robin 分给 `numCores` 个 group，每 group 处理 `(mBlock, 全部 ni)` 的完整列块
2. `ProcessCube(mi)`：GEMM1(全 N1) 每 M-block 只算一次 → workspace；`ProcessVec(mi)`：激活只算一次 → h；GEMM2/epilogue 循环遍历本 group 的 ni
3. `usedCoreNum = min(核数, mBlocks)`（并行度 = M-block 数而非输出 tile 数）
4. 与 host 固定开销消除（X35/X39）正交可叠加；glu 变体注意 gate/up 双段读取逻辑
5. 更强但未攻克的变体：N-split 全核摊薄 B 读（共享 h 的跨核写→读存在间歇性竞态，SyncAll/PipeBarrier 均无法消除，需 GM-volatile flag 协议重构跨核同步）

## Trade-off
- 并行度 ≤ mBlocks，mBlocks < 核数时核利用不足（小 shape 内存带宽受限，可能轻微回退）
- 消除 GEMM1 nBlocks 倍冗余，但未消除第一阶段输出经 GM workspace 的中间往返（进一步优化见 P53/P78 片上链式）

## 来源
自动发现于第 1 轮进化，算子 FFN（910B2C，fp16/bf16 双 GEMM+fastgelu），speedup 0.275x（22 case geomean），大 shape 3.4-6.7x 提速
