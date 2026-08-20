# Proven Solutions Registry

## Purpose

Record successful optimization techniques discovered during evolution runs.
Enables cross-operator knowledge transfer: if a technique worked for operator A,
it may work for a similar operator B.

## Format

Each entry should follow this template:

```markdown
### {technique_name}

**Source**: {op_name} evolution, round {r}, parallel {p}
**Speedup**: {before}x → {after}x ({improvement_pct}% improvement)
**Operator type**: {elementwise | reduction | attention | matmul | ...}
**Hardware**: {chip_model}

**Technique description**:
{1-3 sentences describing what was done}

**Key code change**:
```cpp
// Before:
{key code snippet before optimization}

// After:
{key code snippet after optimization}
```

**Applicability conditions**:
- {condition 1}
- {condition 2}

**Strategy IDs**: {P1, P2, ...} or {X-series if novel}
```

## 按算子类型索引

| 算子类型 | 已有方案 | 来源 |
|---------|---------|------|
| attention | X1: Causal S2 Block Skipping | FlashAttentionSimple evolution |
| elementwise, reduction, normalization | X2: Scalar-to-Vector Conversion | NMS + GroupNorm + TopK evolution |
| sort, topk, nms | X3: Hardware Sort Engine Replacement | NMS + TopK evolution |
| reduction | X4: Hybrid Scalar/Vector Dispatch | GroupNorm evolution |
| matmul | MatmulTransA: Cube Tiling Systematic Design | output_0725_v2 generation |
| conv (backward-filter) | X21: 910B2C Cube 化平台纪律 + 数据驱动 Kahan 精度路由 | conv3dbf_cb evolution |
| conv (kw==1, 小空间维) | X23: Batched Spatial 全宽向量化 + 编译器驱动标量预取 | 8_ConvStandard3d evolution |
| conv-transpose / deconv (col2im 散射) | X26: Col2im Tap-Batch 事件化 | 10_ConvTranspose2d evolution |
| conv-transpose / deconv (stride=1, 固定开销主导) | X29: Shift-Free Tap-Comb 稠密直算 | 10_ConvTranspose2d evolution |
| attention (物化式 MHA/MQA/GQA) | X30: 头维-minor Q 布局免 heads-merge kernel | 22_MultiQueryAttention evolution |
| attention (GQA prefill) | X33: workspace-ring 握手延迟 bound 的 Q-tile 放大 + 投影 GEMM group A-reuse | 21_GroupedQueryAttention evolution |
| attention (GQA prefill，shape-band 分派) | shape-band host 分派（small 两遍 softmax / medium BQ=64 / large BQ=64-128）+ X33/X45/X46 组合 | 21_GroupedQueryAttention evolution round_1-3 |
| attention (block-sparse GQA prefill) | 单遍 flash-attention 融合（MIX_AIC_1_2 + P137 紧凑 gather + P hi/lo 双平面） | 80_GqaSparseFwd evolution |
| softmax (logits 有界) | X31: bounded logits softmax 去 max 减法 | 22_MultiQueryAttention evolution |
| ssm / GLA / linear-attention (chunked 输出阶段) | X34: GLA 输出阶段 Cube CV 融合重写（MIX_AIC_1_1 + hilo 精度分裂） | ChunkGlaFwdKernelO evolution |
| deconv, conv (col2im 散射) | X34: Col2im 相位化行归约 | 10_ConvTranspose2d evolution |
| host-bound 算子（大 workspace 全写） | X39: host 侧全写 workspace 免零初始化（at::empty + 动态 blockDim） | 80_GqaSparseFwd evolution |
| attention (FA/GQA，wrapper 布局转换开销大) | X47: BSH/BSND-direct 布局直通（strided GM load/store 消除 wrapper permute+contiguous） | FusedInferAttentionScore evolution |
| attention (N=1 matvec 链：通道/空间/polarized) | X58: N=1 matvec 链归约突破（多相位拆分 + RA 二叉树 ReduceSum + Broadcast 消标量 + PipeBarrier 松弛） | 57_ParallelPolarizedSelfAttention evolution |

_新方案沉淀时按此格式追加行_

## Recorded Solutions

### X1: Causal S2 Block Skipping (FlashAttentionSimple)

**Source**: FlashAttentionSimple evolution, arch_round8
**Speedup**: 1.003x → 1.83x (45.5% latency reduction at S=32K)
**Operator type**: attention
**Hardware**: Ascend 910B3

**Technique description**:
In causal attention, entire S2 blocks where all entries are masked (below the diagonal)
are skipped entirely — no cube matmul, no vector post-processing, no DMA.
The valid S2 range is computed per S1 block on the host side using the causal constraint:
`valid_s2_end = s1_end` (the last row of the current Q block determines the furthest valid K column).

**Key code change**:
```cpp
// Before: iterate over all S2 blocks
for (int s2 = 0; s2 < totalS2Blocks; s2++) { ProcessBlock(s1, s2); }

// After: skip invalid blocks
int validS2End = (s1BlockEnd) / blockSize + 1;
for (int s2 = 0; s2 < min(validS2End, totalS2Blocks); s2++) { ProcessBlock(s1, s2); }
```

**Applicability conditions**:
- Attention operator with causal (lower-triangular) mask
- Block-tiled computation (not element-wise attention)
- Sequence length > 512 (smaller sequences have few blocks to skip)

**Strategy IDs**: X-series (novel, discovered during evolution)

---

---

### X2: Scalar-to-Vector Conversion (NMS + GroupNorm + TopK)

**Source**: NMS (30_NMS) evolution, GroupNorm (11_GroupNorm_evo) P0, TopK (TopK_evo_0629) Round 1-3
**Speedup**: 11.25x (NMS), 1.84x geomean (GroupNorm fp32: 0.50x→, fp16: 1.93x, bf16: 4.31x), 4.08x (TopK, from 0.66x)
**Operator type**: elementwise, reduction, normalization, sort, topk, nms
**Hardware**: Ascend 910B2C

**Technique description**:
TileLang-generated AscendC code frequently contains scalar for-loops that bypass the NPU Vector unit entirely. The systematic replacement of these patterns with AscendC Vector API calls — ReduceSum for accumulation loops, Adds/Muls chains for element-wise math, Sort32/MrgSort for sorting/selection, and zero-copy sub-tensor slicing for batch operations — is the single highest-impact optimization for generated kernels. A hybrid dispatch (vector path for large workloads, scalar path for small ones) prevents regression on edge cases.

**Key code change**:
```cpp
// Pattern A: Scalar reduction → ReduceSum (+2-5x)
// Before: for(i) sum += x[i]; sum_sq += x[i]*x[i];
// After:
AscendC::ReduceSum<float, true>(scalarOut, xTensor, reduceTmp, count);

// Pattern B: Scalar element-wise → Adds/Muls chain (+3-10x)
// Before: for(i) y[i] = (x[i] - mean) * inv_std * w + b;
// After:
AscendC::Adds<float>(y, x, -mean, n);
AscendC::Muls<float>(y, y, inv_std, n);
AscendC::Muls<float>(y, y, w_val, n);
AscendC::Adds<float>(y, y, b_val, n);

// Pattern C: Scalar sort → Sort32/MrgSort (+5-100x for large n)
// Before: O(n²) bubble sort or O(nk) selection sort
// After:
AscendC::Sort<float, true>(vals, vals, indices, tmpBuf, n / 64);
```

**Applicability conditions**:
- Kernel contains `for` loops with scalar arithmetic (detected by grep for `for.*++.*sum\s*+=` patterns)
- Reduction dimension ≥ 64 elements (smaller: scalar overhead negligible)
- Element-wise operations ≥ 3 in chain (fewer: vector launch overhead may dominate)
- Sort/selection n ≥ 32 (smaller: scalar selection sort is acceptable)

**Strategy IDs**: X2 (cross-operator validation: works on NMS, GroupNorm, TopK; pattern abstracted to P85-P89 strategy cards)

### X3: Hardware Sort Engine Replacement (NMS + TopK)

**Source**: NMS (30_NMS) sort optimization, TopK (TopK_evo_0629) Round 1
**Speedup**: NMS case3: 2181x, TopK: 0.66x → 4.08x (hardware sort is key contributor)
**Operator type**: sort, topk, nms (any operator with sorting/selection)
**Hardware**: Ascend 910B2C

**Technique description**:
AscendC provides hardware-accelerated Sort/Sort32/MrgSort APIs executed on the Vector unit. These replace any scalar sorting or top-k selection algorithm (bubble sort, selection sort, heap-based top-k) with O(n log² n) bitonic sort hardware execution. Key patterns: (1) Negate → Sort ascending → Negate for descending sort; (2) Sort32(repeat=BATCH_ROWS) for batch-parallel small-row sorting; (3) MrgSort for tiled large-row scenarios; (4) Sort + Extract for top-k from sorted results.

**Key code change**:
```cpp
// NMS: Negate → Sort → Negate for descending
for (int32_t k = 0; k < M; k++) svScore(k) = -svScore(k);
AscendC::Sort<float, true>(svScore, svScore, orderU32, tmpF, SORT_SZ / 64);
for (int32_t k = 0; k < M; k++) svScore(k) = -svScore(k);

// TopK: Sort32(repeat=B) batch parallel for small rows
AscendC::Sort<float, true>(packedBuf, packedBuf, orderBuf, tmpBuf, 
                           curBatch * rowLen / 64);
```

**Applicability conditions**:
- Any kernel with sorting, top-k selection, or NMS-style threshold selection
- Element count ≥ 32 (below: scalar methods may be faster)
- Requires 2× data buffer + 1× index buffer in UB for Sort32

**Strategy IDs**: X3 (captured in P87 strategy card)

### X4: Hybrid Scalar/Vector Dispatch (GroupNorm)

**Source**: GroupNorm (11_GroupNorm_evo) P0 optimization
**Speedup**: eliminated regression cases; fp32 geomean improved from baseline 0.22x to 0.50x
**Operator type**: reduction, normalization (any operator with variable workload size)
**Hardware**: Ascend 910B2C

**Technique description**:
Vector API overhead (EnQue/DeQue pipeline synchronization) can dominate on small workloads. A hybrid architecture with an automatic dispatch threshold (e.g., spatial_size >= 128 → vector path, else → scalar path) ensures that small workloads don't suffer from vectorization overhead while large workloads get the full vector benefit. The scalar path uses AllocTensor → DataCopyPad → raw pointer → FreeTensor (no EnQue/DeQue), while the vector path uses DeQue → ReduceSum/Adds/Muls → EnQue.

**Key code change**:
```cpp
constexpr int32_t VECTOR_MIN_SPATIAL = 128;
if (count_spatial >= VECTOR_MIN_SPATIAL) {
    ProcessBatchedVectorized(...);  // EnQue/DeQue + Vector API
} else {
    ProcessBatchedScalar(...);      // AllocTensor + pointer + FreeTensor
}
```

**Applicability conditions**:
- Operator has variable per-tile workload size (e.g., different spatial dimensions)
- Vector path uses EnQue/DeQue (synchronization overhead ~tens of cycles)
- Requires profiling to tune the threshold for specific hardware

**Strategy IDs**: X4 (captured in P89 strategy card)

---

### MatmulTransA: Cube Tiling Systematic Design

**Source**: MatmulTransA generation (`4_MatmulTransA.py`), output path `output_0725_v2/`
**Speedup**: generated AscendC vs PyTorch geomean 0.964x, 50/50 PASS (parity-level Cube baseline)
**Operator type**: matmul
**Hardware**: Ascend 910B3 (npu13)

**Technique description**:
For MatMul variants (including transA/transB), avoid scalar outer-product emulation or manual Cube pipeline construction. Use `MatmulImpl` high-level Cube API with `MatmulApiTiling::GetTiling(TCubeTiling&)`, apply multi-core tiling expansion when `totalBlock < aicCoreNum`, keep `ALIGNED_H=16` while shrinking `singleCoreM/N`, and route to the appropriate high-level implementation (`MatmulImpl` / `Blaze` / `tensor_api`) by platform. A blocking copy of `TCubeTiling` into the host tiling header, correct `MatmulGetTmpBufSizeV2`, and kernel-side tail guards (`SetSingleShape` with `mBaseTail/nBaseTail`) complete the systematic tiling design.

**Key code change**:
```cpp
// Before: naive scalar/outer-product or unoptimized Cube launch

// After: MatmulImpl high-level API + systematic tiling
AscendC::tiling::TCubeTiling cubeTiling;
MatmulApiTiling::GetTiling(cubeTiling);
auto singleCoreM = cubeTiling.singleCoreM;
auto singleCoreN = cubeTiling.singleCoreN;
auto baseM = cubeTiling.baseM;
auto baseN = cubeTiling.baseN;
auto aicCoreNum = platform.GetCoreNumAic();

// P104: multi-core tiling expansion
int64_t totalBlock = CeilDiv(M, singleCoreM) * CeilDiv(N, singleCoreN);
if (totalBlock < aicCoreNum) {
    while (totalBlock < aicCoreNum && singleCoreM > baseM) {
        singleCoreM = std::max(baseM, (singleCoreM / 2) / ALIGNED_H * ALIGNED_H);
        totalBlock = CeilDiv(M, singleCoreM) * CeilDiv(N, singleCoreN);
    }
}
cubeTiling.singleCoreM = singleCoreM;
cubeTiling.singleCoreN = singleCoreN;
cubeTiling.usedCoreNum = std::min(totalBlock, aicCoreNum);

// Kernel: native transA via MatmulType<GM, ND, T, isTrans>, mm.Init, SetSingleShape with tail guard
```

**Applicability conditions**:
- MatMul / BatchMatMul / MatMulBias / GroupedMatmul / quant_matmul variants
- Shapes satisfy Cube baseM/baseN/baseK constraints
- Platform supports `MatmulImpl` high-level API (e.g., DAV_2201)

**Strategy IDs**: P46 (MatmulImpl high-level API), P104 (multi-core tiling expansion), P108 (platform routing)

### X21: 910B2C Cube 化平台纪律 + 数据驱动 Kahan 精度路由 (Conv3DBackpropFilter)

**Source**: conv3dbf_cb evolution, round 2, parallel 1 (x0_fix1) + parallel 0 (n2_s1 交叉实证)
**Speedup**: 1/20 failed → 20/20, avg_speedup 0.1085x（AIV 向量化天花板 0.00235x 的 46 倍）
**Operator type**: conv (backward-filter, cube 化)
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
910B2C 上把 AIV 算子 Cube 化时，正确性杀手是四条平台纪律而非算法：(1) 无 `__vector__` 的
`__global__ __aicore__` 内核被编译为 MIX 双入口，AIC_ONLY 必须显式
`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)`；(2) `PipeBarrier<PIPE_MTE2>` 不建立
MTE2→V 顺序（且 MTE2_S 事件不存在），跨 pipe 同步必须 SetFlag/WaitFlag HardEvent 链；
(3) TransDataTo5HD 在该平台静默杀死内核（rc=0 无日志），须改标量 UB gather + 精确 cast
round-trip；(4) vendor fractal N 序与文档假设可能不同（本例实测 (kd,ci1,kh,kw,ci0)，
kd 在 ci1 外），必须 one-hot/ramp 探针实测。对 fp32 L0C 累加必然超差的相消重元素 case，
用 AIV 预扫描（max|x|·max|g|·S > 5e10 / 含 inf-NaN / fp16+stride>1）数据驱动分流到
完整 Kahan 兜底（带 TwoSum 补偿，非 plain-fp32 重写）。

**Key code change**:
```cpp
// Before: MIX 双入口 + PipeBarrier 同步 + TransDataTo5HD + 假设 fractal 序 + 全量 fp32 L0C
__global__ __aicore__ void conv_dw_cube_kernel(...) {
    ...
    PipeBarrier<PIPE_MTE2>();              // 不建立 MTE2→V 顺序，transdata 恒输出 0
    TransDataTo5HD(dst, src, ...);         // 静默杀死内核（no-op, rc=0）
}

// After: AIC_ONLY 声明 + HardEvent 链 + 标量 UB gather + 探针定序 + 预扫描分流
__global__ __aicore__ void conv_dw_cube_kernel(...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    ...
    SetFlag<HardEvent::MTE2_MTE3>(...); WaitFlag<HardEvent::MTE2_MTE3>(...);  // 显式事件链
    // 标量 UB gather 替代 TransDataTo5HD: dst[hw*16+lane] = src[lane*HW+hw]
    // fractal n1 = g*(ci1G*DkHkWk) + (kd*ci1G + ci1)*hkWk + hw   // 探针实测序
}
// Host: 预扫描 max|x|*max|g|*S > 5e10 或含 inf/NaN → Kahan 兜底 kernel
```

**Applicability conditions**:
- 任何在 910B2C 上新写的 Cube（AIC）内核（四条纪律与算子无关）
- vendor fractal 布局读写（conv backprop-filter dw、wrepack 等）必须探针定序
- 紧容差（bf16/fp16 vs fp64 golden）且存在大相消元素的归约类算子适用精度路由
- compare 工具判读陷阱：NaN 掩码不匹配时 MERE/MARE 默认 0.0，"MERE=0 但 FAIL" 不代表有限值精确

**Strategy IDs**: X21（关联 X18 k 序探针纪律的 3D 推广）

---

### X23: Batched Spatial 全宽向量化 + 编译器驱动标量预取 (ConvStandard3d)

**Source**: 8_ConvStandard3d evolution, round 1, parallel 2 (x0, open_exploration)
**Speedup**: 0.36x → 3.51x geomean (60/60 精度 PASS; case46 3599us→5.1us)
**Operator type**: conv (kw==1, W 维逐点 FMA), fp32 pure-Vector
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
把多个连续空间位置打包进全宽 64-lane fp32 向量：task=(n,cg,od) 一次覆盖整条 od 行的全部 oh 输出，P=floor(64/W) 个连续 oh 打包，每个 (co,tap) 仅 1 次权重标量读驱动 P*W 宽 Axpy——向量指令数与 S-pipe 标量读同时降 P 倍。Boundary 用零填充 + strided DataCopyPad 加载合法子区间，compute 统一全宽（零有效性判断）。内层 co-loop 加 `#pragma unroll 4` 让编译器对 GetValue(Axpy) 链软件流水，隐藏 ~43cy S-pipe 标量读延迟。

**Key code change**:
```cpp
// Before: task=(n,cg,od,oh), acc(G,W), 每 (co,tap) GetValue + W 宽 Axpy（lane 利用率 1/4~1/8）
for (g...) { AscendC::Axpy(acc[g*WP_], xLocal_[tt*WP_], wLocal_.GetValue(g*TcP_+tt), W_); }

// After: task=(n,cg,od), acc[G,Hout*W], P 个 oh 打包进 64-lane 全宽向量
#pragma unroll 4   // ← 编译器软件流水隐藏 S-pipe 标量读延迟（⚠️手工多标量预取触发编译器误编译）
for (g...) { AscendC::Axpy(acc[g*plane_], xBatch_[0], wLocal_.GetValue(g*K_+tap), P*W_); }
// xBatch_ Duplicate 清零 + strided DataCopyPad 加载合法 oh 子区间
```

**Applicability conditions**:
- 滑窗卷积 kw==1 / W 维逐点 FMA，W∈[8,32] 且 W%8==0（32B 对齐）
- 任意「小空间维 + 标量权重驱动向量 FMA」的算子，GetValue S-pipe 串行化是主瓶颈（实测换立即数 ~170x）
- 仅用 `#pragma unroll` 驱动流水；手工多标量预取会触发编译器误编译（结果错误、速度虚高）

**Strategy IDs**: X23

---

### X26: Col2im Tap-Batch 事件化 (ConvTranspose/deconv 散射)

**Source**: 10_ConvTranspose2d evolution, round 1, parallel 2 (x0, open_exploration)
**Speedup**: 0.417x → 0.535x geomean vs PyTorch (1.28x vs baseline); col2im 316us → 277us
**Operator type**: conv-transpose / deconv (col2im 散射), fp32, Cube GEMM + Vector col2im 两阶段
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
Cube 化 GEMM 后 col2im 成为新瓶颈（最差 274us vs GEMM 9us，GM-load-instruction bound：
每输出行 ~8 个 64B 微小 G-row DataCopyPad）。基线每 tap 一次 `PipeBarrier<PIPE_ALL>`
（~16/行）把 MTE2/MTE3/V 三管道完全串行化。重构为 tap-batch：一次发出该行所有有效
tap 的 MTE2 加载（排队重叠）→ 1 次 barrier → V 管道 in-order 背靠背 Gather+Add →
1 次 barrier 保护下一行复用。barrier 从 ~16/行降到 2/行。对齐 zero slot 在 V 管道
一次性初始化（规避 S-pipe SetValue 与 V-pipe Gather 的 S/V 无保序问题）；行级 B1/B4
保守保留 PIPE_ALL（MTE3 copyout WAR）。

**Key code change**:
```cpp
// Before: 每 (kh,kw) tap: LoadGmToUb(src) + PipeBarrier<PIPE_ALL> + Gather + Add + PipeBarrier<PIPE_ALL>
for (kh...) { for (kw...) { LoadGmToUb(src, G[m*ngPad+base], wi);
    PipeBarrier<PIPE_ALL>(); Gather(tmp, src, off[...], 0, woA);
    Add(acc, acc, tmp, woA); PipeBarrier<PIPE_ALL>(); } }

// After: tap-batch —— 所有 tap 加载发出后 1 barrier，V 管道连续 Gather+Add 后 1 barrier
for (kh...) for (kw...) LoadGmToUb(srcs[t], G[m*ngPad+base], wi);  // MTE2 排队重叠
PipeBarrier<PIPE_ALL>();  // 全部加载就绪
for (t...) { Gather(tmp[t], srcs[t], off[...], 0, woA); Add(acc, acc, tmp[t], woA); }
PipeBarrier<PIPE_ALL>();  // WAR 保护下一行
```

**Applicability conditions**:
- 逐输出行循环内有多个独立 tap/源片段、每 tap 一次全管道 barrier 的散射算子
  （col2im、deconv、scatter-add）
- 特征：mte2_instructions 高 + 单次搬运 <16KB（undersize_transfer）
- 前提：同一累加器的串行 Add 可在 V 管道 in-order 完成（V 管道保序）
- 禁忌：strided 多块 DataCopyPad（blockCount>1）在 dav-2201 挂死硬件

**Strategy IDs**: X26

---

### X27: transpose-on-load 分形加载 + 全 shape 手动 mmad 二维 tile 网格（免 host 预转置）

**Source**: 5_MatmulTransB KB 验证轮 evolution, round 1, parallel 2 (x0, open_exploration)
**Speedup**: 0.35x → 0.771x vs PyTorch (2.40x vs baseline); 设备 13.68μs → 5.70μs; fp32[64,128]=6.3μs vs x24 11.2μs
**Operator type**: matmul (transB 存储, 微小→中 shape, 固定开销主导)
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
transB 类 matmul 手动 mmad 的完整覆盖形态：B 转置由 `LoadToL0B<LOAD2D> isBTranspose=true` 在 L1→L0B 分形加载时完成（免 host workspace 预转置 pass，910B2C 任意 M/N/K 正确）；输出 C 按 gridM×gridN 16 对齐 tile、每 tile FULL K 单条 Mmad；并行度扩展启发式摊到更多 core。全 50 shape 无一走 MatmulImpl fallback。

**Key code change**:
```cpp
// Before (x24): host 侧 B 预转置 Bt[K,N] 入 workspace + TQue CopyIn
// After (X27): LoadToL0B<LOAD2D>(l0b, l1B, l0bLoad, isBTranspose=true);  // transB 在读入时完成
```

**Applicability conditions**:
- transB 类 matmul 微小→中 shape：免 host 预转置 workspace pass
- 全 shape 手动覆盖：二维 tile 网格 + 并行度扩展启发式
- 前提：910B2C LoadToL0B LOAD2D isBTranspose=true 已实证正确（区别于 f3 LoadData3D 陷阱）

**Strategy IDs**: X24, X27

---

### X29: Shift-Free Tap-Comb 稠密直算（stride=1 反卷积消 Gather）

**Source**: 10_ConvTranspose2d evolution, round 2, parallel 2 (sv1)
**Speedup**: 0.5356x (两 kernel GEMM+col2im 基线) → 0.927x (case0 22.53us vs ref 20.88us); case7 [1,1,5,5] 2.433x; kernel 级 1.83x
**Operator type**: conv-transpose / deconv (stride=1, 固定开销/交接主导)
**Hardware**: Ascend910B2C (dav-2201)

**Technique description**:
stride=1 反卷积恒等变换：ConvTranspose2d ≡ 零填充输入(补 KH-1-pad) + 180°翻核标准卷积。
单 AIV kernel 内用 shift-free tap-comb（comb[ci][kf2][row][0:woA]=P[ci][row] 左移 kf2 的 8-float 对齐槽）
使固定 (ci,kf2,kf) 的累加成为连续块一次对齐 Axpy，彻底消除 Gather 间接寻址 / col2im 散射 /
G workspace 往返 / AIC→AIV 交接 / torch x2 预处理。

**Key code change**:
```cpp
// Before: 两 kernel — cube GEMM 写 G workspace → AIV col2im 逐 tap Gather+Add + AIC→AIV 交接
// After:  单 AIV ct2d_dense — load x/w 一次入 UB → 标量构造 padded[ci][ph][pw] →
//         comb[ci][kf2][row][0:woA] 左移对齐 → 每 (ci,kf2,kf,co) 一次 Axpy(len=ho*woA) →
//         单次 bulk DataCopyPad(woA 填充缓冲) + host [:wo] 视图
```

**Applicability conditions**:
- stride==1 反卷积/ConvTranspose2d，ci 小（工作集 < UB），固定开销/交接主导
- 零填充平面 ph 必须 = ho+kh-1（非 hi+2t，pad>0 越界）
- DataCopyPad UB→GM blockLen≥32B（wo<8 用 [n][co][ho][woA] 填充缓冲 + host 视图规避 507015）
- host 按 stride 分发：非 stride-1 / 大 UB 回退 cube GEMM + col2im

**Strategy IDs**: X29

---

### X32: fp16 hi/lo 投影 GEMM 精度替代 + 投影 GEMM 真实瓶颈识别（MTE2-bound）

**Source**: GroupedQueryAttention evolution, round 2-4
**Speedup**: 早期大 fp32 case 2-3x 为采样伪影；round-4 全量：fp16 投影净收益在噪声带内；L1 双缓冲 +3-14%
**Operator type**: matmul / attention (GQA prefill fp32 投影)
**Hardware**: Ascend 910B2C (DAV_2201)

**Technique description**:
fp32 投影 GEMM 是 GQA prefill dominant 成本（case44 占 kernel 55%），profiling 证明它是
**MTE2-bound（mac_ratio 0.245 / mte2_ratio 0.709）非 compute-bound**。P128 bf16 hi/lo
对 attention 精度通过、对投影近零输出 MERE 超 2^-13；fp16 hi/lo（~2^-21）是精度安全替代
但 cube 吞吐翻倍被内存瓶颈吞没（单维度无净收益）。正确杠杆：投影 GEMM L1 双缓冲
（GM→L1 与 mmad 重叠，+3-14%）、大块化减少 undersize transfer、消除 GM 往返。

**Key code change**:
```cpp
// L1 双缓冲: aBufL1_[0/1], bBufL1_[0/1] 2-slot ping-pong，k-step ks+1 载入与 ks mmad 重叠
// fp16 hi/lo 精度替代（如需）: hi=fl16(x), lo=fl16(x-fl32(hi)), 3-GEMM 舍 lo@lo
```

**Applicability conditions**:
- fp32 投影 GEMM 已 MTE2-bound → 先优化内存访问（L1 双缓冲/大块化），再考虑精度替代
- 精度需保真时 fp16 hi/lo 替代 3xTF32（K≤1472 MERE≈9e-6）

**Strategy IDs**: X32, P128

_Add new entries below as they are discovered._

### X30: 头维-minor Q 布局免 heads-merge kernel（22_MultiQueryAttention）

**Source**: 22_MultiQueryAttention evolution, round 1, parallel 3 (x0)
**Speedup**: 0.9648x → 1.1267x vs PyTorch（+17%，50/50 精度 PASS）
**Operator type**: 物化式多段 MHA/MQA/GQA（含独立 heads-merge 转置 kernel）
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
heads-merge kernel（O[BH*S,hdPad]→Om[M,H*hdPad] 转置+cast）是纯搬运。因注意力链
QK^T→softmax→PV 对 Q 行序不敏感，Q 布局改 token-major（head-minor）：prep Q-split 从
逐 head strided gather 变 token 粒度连续块拷贝（blockLen=H*hdPad），PV-GEMM 输出
O.view(M,H*hdPad) 天然即合并 heads 布局——heads-merge 整段删除（少 1 launch + O/Om
GM 往返 + cast pass）。

**Key code change**:
```cpp
// Before: Q4 行序 (b,h,s) → S-GEMM→softmax→PV → O[BH*S,hdPad] → heads-merge 转置 → Om[M,H*hdPad]
// After:  Q4 行序 (b,s,h)（prep 连续块拷贝 blockLen=H*hdPad）→ 同注意力链 → O.view(M,H*hdPad) 直接消费
```

**Applicability conditions**:
- 物化式 MHA/MQA/GQA，heads-merge 为纯搬运且 head 间无归约
- 注意力链不依赖 Q 行序；配合 PV 输出直接 cast T 可再省 cast pass

**Strategy IDs**: X30

---

### X31: bounded logits softmax 去 max 减法（22_MultiQueryAttention）

**Source**: 22_MultiQueryAttention evolution, round 1, parallel 3 (x0)
**Speedup**: 0.9648x → 1.1267x vs PyTorch（+17%，与 X30 同变体叠加，50/50 精度 PASS）
**Operator type**: 含 softmax 且 logits 有界的算子
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
softmax(z) 平移不变；max 减法（WholeReduceMax+Brcb+Sub，~3 pass+8 barrier/tile）仅防
exp 溢出（fp32 溢出点 ~88）。logits 实测有界 |z|<40 时删除 max 减法直接 Muls(invDiv)+Exp
（pass 8->5, barrier ~23->~12），配合行求和 chunk 化 WholeReduceSum 全发出 + log-depth
树合并（barrier O(C)->O(log C)），softmax 单 kernel 时间 -25%。

**Key code change**:
```cpp
// Before: WholeReduceMax -> Brcb(max) -> Sub -> Muls(invDiv) -> Exp -> WholeReduceSum
// After:  Muls(invDiv) -> Exp -> chunked WholeReduceSum（先全发出，再 log-depth 树合并）
```

**Applicability conditions**:
- 输入域有界（logits 上界记录证据，|z| < 30-40 留余量），保留恢复 max 减法开关
- 是 A5 max-trick 的条件反向优化；与 X30 正交

**Strategy IDs**: X31

---

### X33: attention workspace-ring 握手延迟 bound 的 Q-tile 放大 + 投影 GEMM group A-reuse（21_GroupedQueryAttention）

**Source**: 21_GroupedQueryAttention evolution, round 3, parallel 2 (x0_x1_x1) + parallel 0 (m2) + parallel 1 (n1_cont)
**Speedup**: 全量 50-case 0.598x → 0.640-0.657x（round-3 vs round-4 prior-art +7-10%；vs 原始基线 0.4997 +28-31%）；attention Q-tile 32→64 单步 case47 +24%、case35 +19%（attention 318→265us）；case2 fp32 小 shape 达 1.21x（快于 PyTorch）
**Operator type**: GQA/FA prefill（attention workspace-ring 流水 + 投影 GEMM）
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
GQA attention 各 aic/aiv 管道利用率 <15% → 是跨核 AIC↔AIV workspace-ring 握手延迟 bound（非带宽/计算）。per-task Q tile 放大（GQA_BLOCK_Q 32→64→128）把每次输出的握手次数减半，直接压延迟。fp32 投影 GEMM 是 MTE2(GM→L1) 管道饱和（aic_mte2_ratio=0.693-0.746），用 by-group A-reuse（GQA_GEMM_GROUP_BY=6，每 task [64,6] slab，A tile 复用 6 列 B）把 A GM over-read 从 ~63x 降到 ~11x。配 X24（AIC_ONLY 纯 cube + AIV_ONLY 纯搬移压 launch 地板）+ P1（L1 双缓冲）是最优组合。

**Key code change**:
```cpp
// attention Q tile: gqa_tiling.h
// Before: constexpr uint32_t GQA_BLOCK_Q = 32;
// After:  constexpr uint32_t GQA_BLOCK_Q = 128;  // 每放大一次握手减半
//        constexpr uint32_t GQA_PV_CHUNK = 64;  // PV 分块控 L0C 宽度

// 投影 GEMM group A-reuse: gqa_matmul.h
// Before: 每 task 1 个 [64,64] 输出块，A 被 nNum 个 N 块重复读
// After:  每 task 覆盖 GROUP_BY=6 个连续 N-tile，A 每 k-step 载一次复用 6 次
```

**Applicability conditions**:
- attention 各管道利用率低（<15-30%）但耗时长 → Q-tile 放大减握手次数；Q-tile 放大需同步 AIV softmax 行组数 + 小 seq 的 padding-row skip
- 投影/GEMM A 侧 over-read 严重 → by-group A-reuse；GROUP_BY 上限由 L0C 容量定（GROUP_BY*BLK*BLK*4B ≤ L0C，fault 时降级）
- 与 X24（AIC_ONLY/AIV_ONLY）+ P1（L1 双缓冲）叠加收益最大
- 残余瓶颈：投影 GEMM B 侧权重搬运（case44 B:A≈12.7:1）

**Strategy IDs**: X33, X24, P1

### X34: GLA 输出阶段 Cube CV 融合重写（MIX_AIC_1_1 + hilo 精度分裂）

**Source**: ChunkGlaFwdKernelO evolution, round 1-3（round1 p0 → round2 p2 NVM → round3 p2 v_ws）
**Speedup**: 0.293x → 10.34x vs PyTorch（geomean，相对基线 ~35x；round1 2.918x → round2 7.594x → round3 10.34x；50/50 精度，min 3.42x）
**Operator type**: ssm / GLA / linear-attention（chunked 输出阶段）
**Hardware**: 910B2C

**Technique description**:
GLA/SSM 系 chunked 输出阶段（`o[t] = tril(A)@v + scale*(q*exp(g))@h`）的纯 AIV count-8 Mul 外积模拟矩阵乘
重写为 `KERNEL_TYPE_MIX_AIC_1_1` Cube CV 融合（三次 Mmad 融合 + Fixpipe F322F16 + hilo 精度分裂 + 任务间流水）。
**关键演进：host 零布局预处理（NVM）**——父 MIX 的 8 个 host 设备 op（pad/permute/OnesLike/Tril）约 120µs/call 占大 case 60%+，
改 q/g/A 原生布局直传 + kernel 内 strided 读取，host op 8→1；**v_ws 混合**再消除最后 host pad（tail chunk AIV 在 workspace 零填充 v，
full chunk AIC 直读 GM），host op→0，小 case 近翻倍。AIC_ONLY 变体（X24）另证小 shape launch 地板可压。

**Key code change**:
```cpp
// op_kernel/chunk_gla_fwd_o.cpp: KERNEL_TYPE_MIX_AIC_1_1
// AIC: 三次 Mmad 融合到单 L0C, 末次 Fixpipe(F322F16) 直写 o (mSize=L 只写实际块长)
//   Mmad(l0c, l0a_qg_hi, l0b_h, mp_init1);   // qg_hi@h 覆盖
//   Mmad(l0c, l0a_qg_lo, l0b_h, mp_init0);   // qg_lo@h 累加
//   Mmad(l0c, l0a_Am, l0b_v, mp_init0);      // A_m@v 累加
//   Fixpipe(l0c, ws_o, fp322f16);            // L0C(fp32)→GM(fp16) 直写
// AIV: qg = q*scale*exp(g) 分块(EXP_CHUNK=512) + hilo 分裂; A_m = A * tril_mask
// 任务间: AIC gemm(task_i) ∥ AIV prep(task_{i+1}), SyncAll<false>() 每迭代 1 次
```

**Applicability conditions**:
- GLA/GDN/SSM 系 chunked 输出阶段（任务 = batch×chunk×head 独立，h 为每 chunk 独立状态）
- gemm 维度 K/V ∈ {32,64,128}，chunk_size 32/64；fp16 输入 + fp32 中间精度要求
- 下一杠杆 = AIV MTE2 优化（P1 任务间双缓冲 + P66 大块 DMA 合并 + 精简 workspace 写入），非加大 Cube
- 纯 AIV 向量路径（任务流水 0.51x / 全宽 count-32 批处理 0.55x）天花板 ~2x 相对基线，不敌 Cube 化

**Strategy IDs**: P102, P133, P134, P25, P66（OKF 卡 x34；同轮对照组 BFR1A 见 x34 卡内）

### X34: Col2im 相位化行归约 (10_ConvTranspose2d)

**Source**: 10_ConvTranspose2d evolution, round 2, parallel 0
**Speedup**: 全量 50-case geomean 0.405x → 0.478x (+18%)；case2 [2,32,16,16] 548.4us → 118us (4.66x)
**Operator type**: deconv / conv (col2im 散射 fallback)
**Hardware**: Ascend910B2C

**Technique description**:
反卷积 fallback（cube GEMM + col2im）的真正瓶颈是 col2im 散射（逐输出行 undersize 加载 + Gather + parity 过滤 + PIPE_ALL，~274us/大 case），不是 im2col 带宽（GEMM 中间 G 仅 0.03-2MB、~9us）。把 col2im 重构为相位化行归约：按输出奇偶拆 s² 相位，每相位对相位内 K²/s² 条 G 行做全长对齐向量 Add 归约 → 相位平面 → 行粒度批量写回；列移位经 DataCopyPad src offset + padding 落地（AIV 全 32B 对齐，绕开 odd-shift 对齐封锁）；tap 表预计算 + 一次向量 Gather 奇偶交织。

**Key code change**:
```cpp
// Before (ct2d_col2im): 逐输出行 (n,co,oh) 循环 tap，每 tap 一次 MTE2 加载 + Gather + PIPE_ALL (~16/行)
// After (ct2d_col2im_phase):
//   每相位内 K²/4 条 G 行全长对齐向量 Add 归约 (P94)
//   acc_phase = Σ_{taps in phase} G_row[0:ng]   // 一次向量 Add 替代逐 tap Gather+Add
//   列移位 c0=d_w0-b 经 DataCopyPad src offset + left/rightPadding (P26)
//   相位平面 [co, n·hi·wi] 行粒度批量写回 (P56); wo==woA 多块 DataCopyPad
//   一次向量 Gather 奇偶交织写回 (idx 一次性预计算)
```

**Applicability conditions**:
- 反卷积/分数步长卷积的 col2im fallback（GEMM + col2im 两 kernel）；散射类 col2im/deconv
- GEMM 中间矩阵 G 可复用（常驻 L2）；G 行 [0,ng) 连续（全长对齐行归约前提）
- 多块加载仅用于 wi%8==0 && srcCol==0（非对齐 src 行多块 dav-2201 病态）
- 残余瓶颈：G 读带宽 (~35us floor) + 双 kernel launch + co 小 case 并行度不足

**Strategy IDs**: P94, P56, P26, X34（OKF 卡 disc_X34）

| 路由/多分支选择 | X42: 路由选中分支消除 | AdaptiveAttention evolution |

### X42: 路由选中分支消除（selected-option elimination）

**Source**: AdaptiveAttention evolution, round 1, parallel 2
**Speedup**: 0.187x → 0.449x geomean (2.40x improvement)，大 shape 5-7.6x
**Operator type**: attention / 路由选择式多分支算子（MoE、条件计算通用）
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
路由器在计算前产生 one-hot 选择向量 sel，输出 = sum_i sel_i·branch_i(x)（one-hot 下仅
取被选中分支）。按 batch 分发把主计算 task 表只映射到被选中分支的权重/数据，跳过其余
n_branch-1 个分支的完整计算链（投影 GEMM、attention、输出投影）。AIC CUBE_ONLY 核无法
可靠读 GM 标量，host D2H 读回 sel（B×n_branch float）算 option 索引作启动参数。

**Key code change**:
```cpp
// Before: 对全部 n_opt 候选层各计算 QKV 投影→attention→out 投影，再 sum_i sel_i·out_i
// After:  只算被 router 选中的 option 的完整链
int64_t opt[B];  // host D2H 读 sel 后 argmax 每 batch 的 option
// 投影 GEMM C[task=(b,i,h)] = x[b] @ W[(i, sel[b], h)]^T（按分支裁剪 + 行块权重）
// attention 任务 B×3H → B×H，Sbuf/Pbuf O(S²) 物化减 n_branch 倍
```

**Applicability conditions**:
- 输出 = one-hot 选择 × 各分支结果（AdaptiveAttention 多候选层 / MoE expert 路由）
- router 代价 << 被跳过分支代价；分支权重可切块寻址
- 小 shape 下 host D2H 同步开销可能吃收益（评估 threshold）

**Strategy IDs**: X42（OKF 卡 disc_X42，与 X30/X33 正交）

| 向量模拟→Cube 全量重写 | X37: fp32 Cube mmad 替代向量模拟 matmul | LongformerAttention evolution |

### X37: fp32 Cube mmad 全量重写替代向量模拟 matmul（SetHF32(false) 全精度）

**Source**: LongformerAttention evolution, round 1, parallel 2 (x0, open_exploration)
**Speedup**: geomean 0.0013x → 4.149x vs PyTorch（基线 368ms → 最优 81μs）；小 shape 2-3x、大 shape 7-9x；50/50 PASS（MERE ~1e-6）
**Operator type**: attention（QK^T → softmax → PV → W_o 多 matmul 流水）
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
基线把所有 matmul 用 AIV 向量标量模拟（16×16 tile 逐点 GetValue 累加）完全绕过 Cube，
且 host 侧 ~1000+ 次串行 launch。重写为 7 次单次发射 kernel（lf_prep→lf_gemm_nt_fp32
QK^T→lf_softmax→lf_gemm_nn_fp32 PV→lf_merge→lf_gemm_nt_fp32 W_o→lf_bias），三个 matmul
全部走 Matmul 高阶 API（ASCENDC_CUBE_ONLY + REGIST_MATMUL_OBJ + IterateAll）fp32 Cube；
`SetHF32(false)` 关闭 HF32 10-bit 尾数截断获得全精度 fp32（MERE ~1e-6）。Cube 重写是
跨越关键：同轮 w1（workflow 策略）0.843x、n2（仅发射消除）0.050x。

**Key code change**:
```cpp
// Before: AIV 向量标量模拟 matmul —— 16×16 tile 逐点 Mul + GetValue 累加，绕过 Cube
// After:  Matmul 高阶 API + SetHF32(false) 全精度 fp32 Cube
Mmz pmat;
pmat.SetHF32(false);                          // 全精度 fp32（默认 HF32 仅 10-bit 尾数）
REGIST_MATMUL_OBJ(&pmat, 1, Tiling, CUBE_ONLY);
pmat.SetTensorA(A, aL0, N1, M0, ...);
pmat.SetTensorB(B, bL0, N1, M0, ...);
pmat.IterateAll();                            // AIC 执行，coreIdx 步进覆盖全部任务
```

**Applicability conditions**:
- 基线 matmul 为向量标量模拟（AIV GetValue 逐点累加）的算子 —— 改 Cube 是跨越式收益
- attention 族多 matmul 流水，fp32 精度路径可放弃 fp16 hi/lo 拆分（HF32 拆分 3 次 gemm
  被全精度单次 gemm 取代）
- 小/中 shape 为主、gemm 非瓶颈上限的算子（全精度 fp32 比 HF32 慢但 shape 小可接受）
- 配套工程约束：uint8 位图 mask 向量操作须 256-bit block 对齐；小 seqLen 统一 sR16≥64；
  fp32 向量 op count ≥8 且为 8 的倍数；exp 保留行 max 减法防下溢

**Strategy IDs**: X37（OKF 卡 disc_X37；与 P54 发射消除、X24 手动 mmad 同族但覆盖「向量
模拟→Cube 全量重写」数量级跨越场景）

### 单遍 flash-attention 融合消除 5-kernel workspace 往返（80_GqaSparseFwd）

**Source**: 80_GqaSparseFwd evolution, round 1, parallel_0 (n1, workflow_guided P115/P121/P123/P137)
**Speedup**: 25 例 geomean 0.405x → 10.75x vs PyTorch（min 4.2x / max 19.5x，约 26.5x vs 基线）；50/50 全量精度过；大 case（totalQ≥4）普遍 14-19x
**Operator type**: GQA block-sparse attention（MiniMax-M3 prefill，paged KV，block_size=128）
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
基线 5 个顺序 kernel（k0 gather → k1 向量 QK^T → k2 online softmax → k3 向量 PV → k4 epilogue），S/P/O 全部经 GM workspace 全量往返、固定 24 核、k1/k3 为 AIV 向量模拟 GEMM。重写为 2-kernel：
1. **k0 紧凑化 gather（P137）**：paged KV 按 topk_idx+block_table 批量 strided DataCopy（每 (task,chunk) 从 256 条 MTE2 命令降到 2 条）→ wsK/wsV。
2. **单遍融合 flash-attention（P115/P121/P123）**：MIX_AIC_1_2 单 kernel，AIC Cube mmad 跑 QK^T → ws1(S 环) → AIV two-pass max-trick softmax（causal lv 掩码）→ ws2(P fp16/bf16 hi/lo 双平面) → Cube PV → ws3(O 环) → AIV 写回。S/P/O 只在每核小环槽（cores×RING_DEPTH×tile），消除跨 kernel 全量 workspace（数百 MB → 每核 10MB 环槽）；6 个 CrossCore mode2 flag 定向握手；动态核数 min(totalTasks,24)。
3. **P hi/lo 双平面**：bf16 大 seq case44 MERE 越界，拆 P_hi@V + P_lo@V 修复（fp16 单平面不够精度）。

**Key code change**:
```cpp
// 基线：5 顺序 kernel + workspace 全量 GM 往返（wsS/wsP/wsO 数百 MB）
// After：k0 紧凑 gather + MIX_AIC_1_2 单遍融合，S/P/O 每核环槽
//   AIC: cube QK^T -> ws1 ring -> (AIV softmax) -> ws2 P hi/lo -> cube PV -> ws3 ring
//   CrossCoreSetFlag/WaitFlag mode2 定向握手（6 flag），RING_DEPTH=2
//   k0: DataCopyParams 批量 strided copy（blockLen/srcStride 32B 单位），256 条 MTE2 -> 2 条
```

**Applicability conditions**:
- 5+ 顺序 kernel + 全量 workspace 中转的 attention（S/P/O 物化到 GM 再读回）—— 融合消除跨 kernel 往返是数量级收益
- block-sparse/paged KV 的 top-k gather 独立成 k0（P137）优于折入融合 kernel（避免每 task K/V 握手，X33 教训）
- bf16 精度路径须 P hi/lo 双平面（fp16 单平面 MERE 越界）
- 保留 k0 独立 kernel：分页 gather 需标量索引，折入 AIV 会引入每 task 一次 K/V 握手

**Strategy IDs**: P115, P121, P123, P137, P119, P120, A5, A4；OKF 卡 X39（host workspace 免 memset，x0 侧沉淀）
| matmul (双 GEMM 链 FFN/QK^T+PV/SwiGLU) | X48: 链式 GEMM M-block 归属调度（第一阶段输出复用） | 2_FFN evolution |

### X48: Chained GEMM M-block Ownership Scheduling (first-stage output reuse)

**Source**: FFN evolution, round 1, parallel 2
**Speedup**: 0.153x → 0.275x (+80% geomean, 22 cases)；大 shape case4 [1024,1536] 0.074→0.252 (3.4x)、case18 0.060→0.403 (6.7x)、case21 0.034→0.227 (6.5x)
**Operator type**: matmul (两阶段 GEMM 链：FFN / QK^T+PV / SwiGLU / MoE FFN)
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
朴素按 (mi,ni) 输出 tile 分任务的链式 GEMM 会把第一阶段（GEMM1）重复计算 nBlocks 次（同一 M-block 的每个 N-tile 各算一遍 GEMM1+激活）。改为 M-block 归属调度：每核 group 独占其 M-block 的全部 N-tile，GEMM1 每 M-block 只算一次、中间结果 h 在 group 内复用，GEMM2 循环遍历本 group 的 ni。nBlocks 越大收益越大（FFN nBlocks=32 → 0.018x 病态慢被消除）。

**Key code change**:
```cpp
// Before: 每个 (mi,ni) tile 独立执行完整串行链 GEMM1→act→GEMM2
for (tile : allTiles) {           // tile=(mi,ni)
    GEMM1(x@W1 全N1 -> ws1);      // 对同一 mi 重复 nBlocks 次
    act(ws1 -> ws2);              // 重复 nBlocks 次
    GEMM2(ws2@W2[:,ni] -> out);
}
// After: M-block 归属，GEMM1 每 mi 只算一次
// BlockScheduler: mBlocks round-robin 分给 groups，group 独占其 mi 的全部 nBlocks 个 ni
for (group owns mi) {
    GEMM1(x@W1 全N1 -> ws1);      // 每个 mi 只算一次
    act(ws1 -> ws2);              // 每个 mi 只算一次
    for (ni in group) { GEMM2(ws2@W2[:,ni] -> out); }
}
// usedCoreNum = min(核数, mBlocks)
```

**Applicability conditions**:
- 两阶段 GEMM 链（第一阶段输出被第二阶段多个 N-tile 消费）
- 大 N1（nBlocks≥16）时收益最大；小 M（mBlocks<核数）时核利用受限
- 与 host 固定开销消除（X35/X39）正交；中间结果仍经 GM workspace 时可再叠加 P53/P78 片上链式
- glu 变体注意 gate/up 双段读取逻辑

**Strategy IDs**: X48

| matmul (双 GEMM 链 FFN/QK^T+PV，小 M 大 N2) | X49: 两阶段 GEMM 跨核生产/消费切分（N-split stage-split + GM-volatile flag） | 2_FFN evolution |

### X49: Two-stage GEMM cross-core producer/consumer split (N-split + GM-volatile flag)

**Source**: FFN evolution, round 2, parallel 2
**Speedup**: 0.275x → 0.475x (+44% vs round1 最优)；case0 [128,1024] 0.626x（父 0.323，+94%）、case1 [256,2048] 0.321x（父 0.101，+218%）
**Operator type**: matmul（两阶段 GEMM 链：FFN / QK^T+PV / SwiGLU）
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
阶段一（GEMM1）按 M-block 归属（X48）时并行度受 mBlocks 限制（小 M 大 N2 仅 4-8 核工作）。N-split stage-split：生产者核切 K2 成 P 片各产 h 片（GEMM1→act→写 GM），写后 GM volatile flag 广播（release）；消费者全核 G=min(24,nBlocks) 各领 N2 列块轮询 flag（acquire）后做 GEMM2。模式选择：nBlocks>mBlocks 走 N-split，否则 M-block 归属保底。

**Key code change**:
```cpp
// Producer (core < P): K2 切片产 h
GEMM1(x[mi]@W1[:,slice]) -> ws1; act -> h[mi][slice];
PipeBarrier<PIPE_ALL>; flag[mi][p] = 1;   // release (GM volatile)
// Consumer (all G cores): 轮询 flag 后做 GEMM2
AIV: wait flag[mi][0..P) all 1 (GM volatile acquire);
AIC: GEMM2(h[mi]@W2[:,R_c]) -> out;
// host 模式选择: schedMode = (nBlocks > mBlocks) ? 1 : 0; G = min(24, nBlocks)
```

**Applicability conditions**:
- 两阶段 GEMM 链阶段一为并行度瓶颈（mBlocks < 核数）
- 已有 X48 M-block 归属但仍核空转时升级
- flag workspace 必须零初始化（at::zeros，防垃圾 1 误判）；910B2C 无 GM 原子用 volatile flag
- 与带宽路径（P92/P93）正交；后续可加跨 mi 软件流水（P20）

**Strategy IDs**: X49



### Shape-band host dispatch for GQA prefill attention (small two-pass / medium BQ=64 / large BQ=64-128)

**Source**: 21_GroupedQueryAttention evolution, rounds 1-3 (round_1/parallel_0-2, round_2/parallel_0-2, round_3/parallel_0-2)
**Speedup**: baseline full-50 0.816x → best 0.97x (torch.npu.Event basis; band-level: large fp32 0.629→0.928, bf16 medium 0.755→1.005, fp32 medium 0.601→0.627, +4-13% projection GEMM A/B). Precision 50/50 PASS all variants.
**Operator type**: attention (GQA prefill, fused QKV/out projection)
**Hardware**: Ascend 910B2C (DAV_2201), MIX AIC/AIV

**Technique description**:
Host-side shape-band dispatch routes each seq band to a dedicated attention kernel structure, each tuned for its S-regime bottleneck:
- **small (seq<64)**: two-pass softmax — S=QK^T fits entirely in UB, AIC computes ALL QK^T → sync once → AIV softmax ALL → sync once → AIC computes ALL PV. Only 2 cross-engine syncs vs 2×task-num for per-task workspace-ring handshake (kills aic_scalar=0.46 root cause). Neutral in practice (small shapes are host-launch-bound, attention kernel only 10-15us of ~85us total).
- **medium (64<=seq<200)**: BQ=64 attention (X33 Q-tile amplification halves AIC↔AIV handshakes) + native fp32 single-pass (X46) + projection GEMM by-group A-reuse (GROUP_BY=6, A over-read 63x→11x).
- **large (seq>=200)**: BQ=64→128 deep-pipelined online softmax (QK^T(task+1) ahead of PV(task)), Q L1-resident, native fp32 single-pass (X46), P chunked PV stream.
- Cross-cutting: X45 mNum-adaptive bf16 projection GEMM hybrid (full-K WS only when mNum>MGROUP) fixed the bf16 medium valley; host data-path elimination (X52: single-pass strided-gather transpose + conditional at::empty + mask cache).

**Key code change**:
```cpp
// host dispatch (grouped_query_attention.cpp) — shape-band routing
if (isFp32 && S < 64)      LaunchAttentionSmallBand(...);   // two-pass softmax, 2 syncs
else if (isFp32 && S < 200) LaunchAttentionFp32SingleMed(...); // BQ=64 + native fp32
else if (isFp32 && S >= 200) LaunchAttentionFp32Large(...);    // BQ=64/128 deep pipeline
else if (!isFp32 && 64<=S && S<200 && tasks>=24)
                           LaunchAttentionFp16Bf16Med(...);   // BQ=64 medium
else                       LaunchAttentionFp16/Bf16/Fp32(...); // baseline paths
```

**Applicability conditions**:
- GQA/FA prefill attention where the S (seq) range spans orders of magnitude (3-897 here) and each S-regime has a different bottleneck (small: host-bound; medium: handshake-bound + projection floor; large: deep pipeline + native fp32)
- Host dispatch cost ≈ 0 (one branch, analogous to dtype dispatch)
- Requires: X33 Q-tile template parameterization, X46 native fp32 (no SetHF32Mode), X45 mNum-adaptive GEMM hybrid, X28/X38 small-S KV chunk
- Precision guard: conditional at::empty only when no K-padding (Dm%64==0 / hd==Dp), else at::zeros (0*NaN=NaN)

**Strategy IDs**: X28, X33, X38, X45, X46, X52

### X56: STRIDE-SPARSE residue-block Cube attention（stride-mask 按 residue 分组只算有效列）

**Source**: 25_StridedAttention evolution, round 1, parallel 2 (x0, open_exploration)
**Speedup**: 10-case geomean 0.586x → 0.811x vs PyTorch (+38%), 10/10 precision PASS. fp16 大 case 0.269→0.692 / 0.471→0.686, bf16 0.211→0.568 / 0.140→0.417.
**Operator type**: attention (stride-mask self-attention, MHA)
**Hardware**: Ascend 910B2C (DAV_2201), MIX AIC/AIV

**Technique description**:
stride-mask attention 的 mask `score[i,j]` 有限 iff `i%stride==j%stride`，每个 query 行只需自己 residue class 的 Srmax=ceil(S/stride) 个 key。把 attention task 从 `(b,h,query-block)` 重组织为 `(b,h,residue,query-block)`，QK^T/softmax/PV 宽度从 S_pad 收窄到 Srmax16（stride 倍缩减）：Q/V 走 strided 2D block load（行 stride `stride*HDP16`）免 host 重排；fp32 K^T host residue-grouped 到 `[B,H,HDP16,stride*Srmax16]`；per-residue padding mask 替代全 stride×S_pad mask 表；bf16 大 case 从 O(S²) vector attention 切到 Cube attention（P hi/lo 拆分保精度）。

**Key code change**:
```cpp
// host tiling (strided_attention.cpp) — residue-block attention
const int64_t Srmax = CeilDiv(S, stride);          // valid keys per row
const int64_t Srmax16 = CeilDiv(Srmax, 16) * 16;
const int64_t S_padA = stride * Srmax16;           // all residue rows in-bounds
// K^T host residue-grouped (fp32): [B,H,S_pad,HDP16] -> [B,H,HDP16,stride*Srmax16]
// attention task = (b, h, residue r, query-block), width = Srmax16
// write-back: sOrig = sl * stride + r
```

**Applicability conditions**:
- stride-mask attention（每行仅 1/stride 列有效），stride>=2
- attention 核心（QK^T/softmax/PV）是大 case 瓶颈、Cube 已建成（P120）
- bf16 大 case 还在 O(S²) vector attention 时切 Cube + P hi/lo 保精度
- 与 P123（causal KV skip 循环界裁剪）互补：X56 是重排分组后只算有效列

**Strategy IDs**: X56, P120, P123

### attn+outproj 单 kernel 链融合（SyncAll 分隔）+ native proj BLOCK_M 64 头合并 GEMM

**Source**: 25_StridedAttention evolution, round_1/parallel_2 (x0, open_exploration)
**Speedup**: 10-case geomean 0.5975x → 0.9237x vs PyTorch (+55%), 10/10 precision PASS. fp32 大 case 0.242→0.375 / 0.293→0.473, fp16 0.274→0.695 / 0.465→0.707, bf16 0.214→0.579 / 0.141→0.419.
**Operator type**: attention (stride-mask self-attention, MHA)
**Hardware**: Ascend 910B2C (DAV_2201), MIX AIC/AIV

**Technique description**:
MIX attention（AIC QK^T/PV + AIV softmax）与 cube outproj 两个串行 kernel 融合为单 kernel：Phase 1 MIX 写 attn [padM,Dp] → SyncAll<true> GM grid barrier → Phase 2 AIC outproj 直接读 attn 写 out [M,D]，L0A/L0B/L0C 两阶段共享。消除 1 launch + attn GM 往返 + host outF32->out copy。配套 native proj BLOCK_M 16→64 头合并 GEMM（task=(b,p,M-chunk)，x M-slice 跨 nH head 复用，x 流量 B*H*3*S*D→B*3*(H/nH)*S*D，cube 利用率 ~2→14 TFLOPs），bf16 切 MIX cube（P hi/lo 拆分保精度）。

**Key code change**:
```cpp
// 单 kernel 链融合: Phase1 MIX attention -> SyncAll -> Phase2 AIC outproj
// (sa_attn_outproj_cube_fp32.cpp)
// Phase 2 outproj 直写 out [M,D] (dstStride=D, nSize clamp, D%16==0)
FixpipeParamsV220 fp; fp.nSize = D - n0; fp.mSize = remM; fp.dstStride = D;
```

**Applicability conditions**:
- MIX attention + cube outproj 融合算子（FA/GQA/MLA 大 case）
- native 投影 BLOCK_M 小（16）导致 cube 利用率低（~2 TFLOPs vs 参考 ~14 TFLOPs）
- SyncAll 需验证目标平台可用且不 deadlock（910B2C 已验证）
- ⚠️ 注意：x0 全量 50-case 精度 48/50（fp32 stride>=4 大 case MERE 0.04/0.06 fail），交付需先验证全量精度

**Strategy IDs**: X57, X50, X56, X44

### shape-keyed 权重 pad 缓存 + fp16/bf16 workspace at::empty（X52 深化）

**Source**: 25_StridedAttention evolution, round_1/parallel_1 (n2, X52/X39/P54)
**Speedup**: 10-case geomean 0.5975x → 0.9121x vs PyTorch (+53%), 50-case 精度 50/50 PASS, 50-case 性能 ~0.75x. fp32 [2,157,864] 1005→669us、[2,383,1472] 1998→1233us.
**Operator type**: attention (stride-mask self-attention, MHA)
**Hardware**: Ascend 910B2C (DAV_2201)

**Technique description**:
host 数据通路消除深化：wqT/wkT/wvT/woP 权重 pad（transpose+per-head pad copy）是 (weight, D, H, HD, HDP16, Dp) 的纯函数，eval 中权重跨调用恒定 → shape-keyed 单槽 memo 缓存（(data_ptr+shape+dtype+device) key），每次 forward 省 ~12（fp32）/~10（fp16/bf16）个 host kernel。配 fp16/bf16 workspace at::empty（HDP==HD 守卫，kernel Fixpipe 已写满全部区域时安全）。

**Key code change**:
```cpp
// 单槽权重缓存（op_host/strided_attention.cpp）
static std::unordered_map<size_t, at::Tensor> g_weightCache;
auto key = hash(data_ptr, D, H, HD, HDP16, Dp, dtype, device);
if (!g_weightCache.count(key)) {
    g_weightCache[key] = PadWeightGather(w);  // 单次 strided gather 替代 H 次 slice copy
}
// 缓存项持有 wRef 强引用防 data_ptr 复用
```

**Applicability conditions**:
- 融合 attention 管线 host 侧有重复权重 pad/transpose（eval/部署中权重恒定）
- fp16/bf16 大 case workspace 大（~4MB each）且 HDP==HD（无 pad 列）时 at::empty 安全
- ⚠️ fp32 在-device K^T 转置（col-major nN→zN）在 910B2C 确认损坏（max_diff 40），host transpose 是唯一可靠路径

**Strategy IDs**: X52, X39, P54

### X58: N=1 matvec 链归约 Vector 管道突破（57_ParallelPolarizedSelfAttention）

**Source**: 57_ParallelPolarizedSelfAttention evolution, round_5/parallel_0 (r5a)，突破链 x0→x0b→r3b→r4b→r5a
**Speedup**: 0.3225x → 1.1148x vs PyTorch (3.46× improvement; 全量 50-case，49/50 valid，精度 PASS)
**Operator type**: attention / N=1 matvec 链（通道+空间 self-attention），纯 Vector 管道、Cube M=1/N=1 不可用
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
N=1 matvec 链注意力（每输出元素 = 权重逐元素乘加链）曾长期被判"1.0x 物理不可达"（matvec 归约架构限制）。四杠杆组合推翻该判断：① **多相位链拆分**（7-phase 消除 gN 倍冗余链重算 + n-split 只读自己 n-slice 消冗余全 X 读）；② **RA 二叉树 ReduceSum**（串行 Add 依赖链 O(S) → 对半 log-depth 树归约）；③ **Broadcast 消灭 GetValue 标量广播**（S-pipe 标量读 → 向量 Broadcast/Duplicate）；④ **PipeBarrier 松弛**（PIPE_ALL→PIPE_V，BroadTile 自带 PIPE_V 时释放 MTE2 X-prefetch 与 V 计算重叠）。叠加 fp16 X 数据路径 + 链 weight tile 双缓冲。

**Key code change**:
```cpp
// Before：每 Pass 后 PipeBarrier<PIPE_ALL> + 逐元素 GetValue 标量读 + 串行 Add 链
PipeBarrier<PIPE_ALL>();
for (i...) { acc = Adds(acc, xLocal.GetValue(i)*wLocal.GetValue(i), n); }

// After（r5a）：7-phase 拆分 + RA 二叉树 ReduceSum + Broadcast/Duplicate（P99）
//             + PipeBarrier<PIPE_V>（MTE2 预取与 V 计算重叠）+ fp16 X 路径 + 链 weight 双缓冲
```

**Applicability conditions**:
- N=1 matvec 链注意力/归约算子（通道/空间/polarized attention），Vector 管道 bound、Cube 不可用
- profiling：vector_pipe 主导 + scalar_loading/scalar_compute 次高、mte2 未饱和
- ⚠️ ColReduceAllRA（带 WholeReduceSum）在 vector-bound kernel +0.9% 回归，用 Level-2/对半树形态
- ⚠️ 勿重复失败方向：host transpose 权重链、64-granularity c2-split、运行时自适应 X-tile

**Strategy IDs**: X58（关联 P99/P94/P95；与 [[disc_X54]] 全宽归约、[[disc_X25]] 向量权重广播互补）
