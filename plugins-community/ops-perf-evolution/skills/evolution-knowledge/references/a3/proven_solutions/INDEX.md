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

_Add new entries below as they are discovered._
