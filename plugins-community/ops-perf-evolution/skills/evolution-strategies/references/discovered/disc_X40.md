---
schema_version: okf.v1
kind: discovered_strategy
id: X40
title: "X40: MatmulApiTiling 单核陷阱 + singleN/singleM hint 前置多核化（totalBlock=1 → per-M 自适应 24 blocks）"
origin: discovered
discovered_round: 1
discovered_from: "FluxTimestepGuidanceProjectionEmbedding evolution, round 1 parallel_0 (n1 P104) + round 2 parallel_0 (n6 P2) + round 3 parallel_1 (n9 全链 P2)"
base_speedup: 0.725x → 1.461x vs PyTorch（基线 0.179x，8 倍）
confidence: single_op
status: active
op_families: [matmul, elementwise, reduction]
bottlenecks: [tiling_imbalance, no_overlap, compute_bound]
tags: [matmul, totalBlock-1, single-core-geem, multi-core-expansion, matmulapitiling, hint-before-gettiling, per-m-adaptive, tiny-shape]
strategy_ids: [X40, P104, P2]
---

### X40: MatmulApiTiling 单核陷阱 + singleN/singleM hint 前置多核化

**Source**: FluxTimestepGuidanceProjectionEmbedding evolution（多轮融合算子：embed→GEMM1→SiLU→GEMM2+text GEMM→epilogue）
**Speedup**: 基线 0.179x → round1(n1 P104) 0.725x → round2(n6 P2 GEMM2/GTX) 1.28x → round3(n9 全链 P2) 1.461x vs PyTorch（8 倍）；fp16 1.646x / bf16 1.221x
**Operator type**: 小 batch GEMM 链 + 向量后处理融合（B∈[1,384]，K=768/3072，N=768/3072），固定开销与 GEMM 核利用率主导
**Hardware**: Ascend910B2C (DAV_2201), 24 AIC

**Technique description**:
小 shape GEMM 用 `MatmulApiTiling::GetTiling(TCubeTiling&)` 时，API 会为"整块能塞进单核"的 shape 返回 `singleCoreM=M, singleCoreN=N`（`mTotalCnt=nTotalCnt=1` → **totalBlock=1**），即**所有 GEMM 都只跑在 1 个 AIC 核上**（24 核只用 1 核），这是小 shape GEMM 算子性能杀手。修复：在调用 `GetTiling` **之前**设置 `tilingApi.singleN = 128`（或 per-M 分段的 singleN/singleM）hint，让 API **自洽地**重排 `singleCoreN/baseN/baseK/stepN/depthA1/depthB1`，得到多 N-block（如 N=3072→24 blocks、N=768→6 blocks），吃满 24 AIC。**关键纪律**：hint 必须在 `GetTiling` 之前设置；严禁在 GetTiling 之后手工缩小 singleCoreM/N（不重算 stepN/depthA1/depthB1/baseK 会导致 AIC trap 507014）。per-M 自适应经验规则：M≤64 → N-split singleN=32（24 N-blocks，B 矩阵 1x 重载、A 重载便宜）；64<M≤128 → singleM=64（12 blocks）；M>128 → singleM=M/4 ALIGNED_H=16（24 blocks）。

**Key code change**:
```cpp
// Before: MatmulApiTiling 返回 singleCoreM=M, singleCoreN=N → totalBlock=1（单核 GEMM）
matmul_tiling::MatmulApiTiling tilingApi(*platform);
tilingApi.SetShape(M, N, K);
tilingApi.GetTiling(header.cubeTiling);   // totalBlock = mCnt*nCnt = 1 → 只用 1 AIC 核

// After: GetTiling 前设置 singleN hint → API 自洽重排依赖字段 → 多 N-block 吃满核
matmul_tiling::MatmulApiTiling tilingApi(*platform);
tilingApi.singleN = 128;                    // hint 必须前置
tilingApi.SetShape(M, N, K);
tilingApi.GetTiling(header.cubeTiling);     // singleCoreN=128, baseK=128, depthA1/B1 重算
// totalBlock = ceil(N/128) → GEMM1(N=3072)=24 blocks, GEMM2(N=768)=6 blocks
// per-M 自适应：M≤64 nsplit singleN=32；64<M≤128 singleM=64；M>128 singleM=M/4
```

**Applicability conditions**:
- **小 shape / 单 tile 可装入单核**的 GEMM（M/N/K 小，API 倾向返回 singleCoreM=M, singleCoreN=N）
- 诊断方法：`FLUX_DEBUG_TILING` 类环境变量打印 tiling 头，或 host 读 `totalBlock` 确认是否 =1 / < aicCoreNum
- 与 P54 launch 融合（M-batching：精度 split GEMM 激活 M 拼接一次 launch）叠加收益最大
- 对计算密集大 shape（totalBlock 本已 ≥ 核数）无显著收益
- 必须保持 tiling 字段自洽性（hint 前置，禁 GetTiling 后手改）

**Strategy IDs**: X40（关联 P104 多核切分扩展、P2 自适应分块、P54 launch 融合）
