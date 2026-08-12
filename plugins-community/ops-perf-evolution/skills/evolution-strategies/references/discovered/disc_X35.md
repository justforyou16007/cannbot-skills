---
id: X35
origin: discovered
discovered_round: 1
discovered_from: round_1/parallel_0
base_speedup: 0.272x → 0.713x (geomean, MatmulTransA)
---

# Strategy X35: MatmulImpl Host Fixed-Overhead Elimination + Coverage-Aware Serial Multi-Tile

## 核心思路
MatmulImpl 高阶 API 的每次调用含固定 host/device 开销：`MultiCoreMatmulTiling::GetTiling()` 主机计算、TilingHeader 阻塞 H2D copy、`at::zeros` workspace 清零（~6.5us/次的 aclnnInplaceZero 设备 kernel）、workspace 分配。对小到中 shape（输出规模 ≤ 4×L0C），这些固定开销占运行时的绝大部分（launch-bound），是低加速比的主因。

消除/压缩四类固定开销：
1. **workspace `at::zeros` → `at::empty`**：MatmulImpl 的 KFC workspace 只读 init state，实测 50/50 位精确无需零初始化（CANN 参考路径默认零化但非必需）。
2. **per-shape tiling cache**：host 侧 unordered_map<TilingKey, CachedTiling> 缓存 GetTiling 结果 + TilingHeader，消除每次调用的重算 + H2D copy。
3. **workspace 缓存复用**：跨调用复用同一 workspace buffer。
4. **kernel 核内串行多 tile 遍历**：`for(tile=blockIdx; tile<totalBlock; tile+=blockDim)`，`End()` 移出循环只调一次——修复 totalBlock>blockDim 时的漏算缺陷，并保证全部核忙碌。

配套：**自适应 core re-grid**（host 侧枚举 16 对齐候选 tile，L0C 约束 `sm*sn*sizeof(accum)≤L0C`，评分优先 target totalBlock）：输出规模 ≤ 4×L0C（overhead 受限）时 re-grid 到 24 核；规模更大（compute/MTE2 受限）保留 GetTiling 天然对齐 tile（避免 Cube 掩码浪费）。实测：大 shape 从 16 tiles 拆到 32 tiles(24 核) 反而变慢，故核心目标是 totalBlock ∈ [8,24] 内最大化 tile 面积。

## 适用场景
- MatmulImpl / Blaze 高阶 API 实现的 matmul 系算子（transA/transB/bmm/matmul_bias），shape 中小（M,N ≤ 1024，K 小），launch/host 固定开销主导（speedup << 1）
- 纯 Cube 或 CV 融合的 matmul，多 shape 混合调用（tiling 缓存收益随 shape 数线性增长）
- 需要多核全覆盖且 totalBlock 可能超过 blockDim 的 kernel

## 实现要点
1. workspace 改用 `at::empty` 前必须全量精度验证（MatmulImpl 读 init state 的假设），不可盲目
2. tiling 缓存键：`(M, N, K, dtype)`，注意 tiling 结果与输入 data_ptr 无关（纯 shape 函数）
3. kernel 遍历 `tile=blockIdx; tile<totalBlock; tile+=blockDim` 时，每 tile 用 SetSingleShape(curM,curN,fullK) 后 IterateAll；End() 只调一次（MatmulImpl 契约）
4. re-grid 枚举候选需保持 ALIGNED_H=16、不低于 baseM/baseN、L0C accum 约束；stepM/stepN 同步收紧
5. 与 P104（多核切分扩展）互补：P104 只在 totalBlock<aicCoreNum 时缩小 tile，X22 提供按输出规模自适应的 core-target 重排 + 固定开销消除

## 来源
自动发现于第 1 轮进化，算子 MatmulTransA（C=A^T@B，910B2C，fp32/fp16/bf16，K≤256），speedup 0.272x → 0.713x（2.6x），50/50 精度全通过
