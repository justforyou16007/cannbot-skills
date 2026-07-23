---
id: P98
bottlenecks: [scalar_compute, compute_bound, near_optimal]
op_families: [reduction, normalization, elementwise, softmax]
complexity: L1
conflicts_with: []
synergizes_with: [P94, P95, P67]
requires: []
has_preconditions: true
has_playbook: true
---

# P98: 混合标量/向量路径自适应调度 (Hybrid Scalar/Vector Dispatch)

## 核心思想
向量化并非在所有 workload 大小上都优于标量。对于极小 workload，Vector API 的 EnQue/DeQue 流水线同步和 PipeBarrier 开销可能超过标量循环的计算开销，导致向量化后的性能反而不如原始标量版本（"负优化"）。该策略根据 per-tile workload 大小自动选择：大 workload 走向量化路径（EnQue/DeQue + ReduceSum/Adds/Muls），小 workload 走轻量标量路径（AllocTensor → DataCopyPad → raw pointer → FreeTensor，无队列同步开销）。阈值基于 profiling 数据设定，初始值 128 元素（910B），可随 profiling 反馈自适应调整。

## 代码骨架

```cpp
// === 改造前（基线）：单一标量或单一路径 ===
void ProcessBatched(LocalTensor<float> xGm, LocalTensor<float> yGm, ...) {
    // 要么全标量（大 workload 慢），要么全向量（小 workload 有开销）
    for (int32_t i = 0; i < count_spatial; ++i) {
        sum += xPtr[i];
        sum_sq += xPtr[i] * xPtr[i];
    }
}

// === 改造后：混合调度架构 ===
constexpr int32_t VECTOR_MIN_SPATIAL = 128;  // 阈值（可基于 profiling 调优）

void ProcessBatched(...) {
    if (count_spatial >= VECTOR_MIN_SPATIAL) {
        ProcessBatchedVectorized(xGm, yGm, weightGm, biasGm, count_spatial);
    } else {
        ProcessBatchedScalar(xGm, yGm, weightGm, biasGm, count_spatial);
    }
}

// 标量路径：无 EnQue/DeQue，AllocTensor → DataCopyPad → raw pointer → FreeTensor
__aicore__ void ProcessBatchedScalar(__gm__ float* xGm, __gm__ float* yGm, 
    __gm__ float* wGm, __gm__ float* bGm, int32_t count) {
    LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    DataCopyPad(xLocal, xGm + offset, count);
    float* xPtr = reinterpret_cast<float*>(xLocal.GetPhyAddr());
    float sum = 0.0f, sum_sq = 0.0f;
    for (int32_t i = 0; i < count; ++i) {
        float val = xPtr[i];
        sum += val;
        sum_sq += val * val;
    }
    // ... 标量处理 ...
    inQueueX.FreeTensor(xLocal);
}

// 向量路径：EnQue/DeQue + ReduceSum/Adds/Muls
__aicore__ void ProcessBatchedVectorized(
    __gm__ float* xGm, __gm__ float* yGm,
    __gm__ float* wGm, __gm__ float* bGm, int32_t count) 
{
    LocalTensor<float> xIn = inQueueX.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    
    AscendC::ReduceSum<float, true>(scalarLocal, xIn, reduceTmp, count);
    float mean = scalarLocal.GetValue(0) / count;
    
    AscendC::Adds<float>(yLocal, xIn, -mean, count);
    AscendC::Muls<float>(yLocal, yLocal, inv_std, count);
    AscendC::Muls<float>(yLocal, yLocal, w_val, count);
    AscendC::Adds<float>(yLocal, yLocal, b_val, count);
    
    outQueueY.EnQue<float>(yLocal);
    inQueueX.FreeTensor(xIn);
}

// 多级 dispatch（用于 TopK 等更复杂的分支场景）
void Compute(const TopKParams& params) {
    if (rowLength <= 32) {
        ComputeBatchSmall(...);    // Sort32(repeat=B) 批量并行
    } else if (rowLength <= 128 && is32BAligned(rowLength)) {
        ComputeBatchAligned(...);  // 单 MTE2/MTE3 burst
    } else if (rowLength <= 512) {
        ComputeBatchStaged(...);   // slot staging 批量
    } else {
        ComputeRow(...);           // 单行 tiled running-topk
    }
}
```

## 关键修改点

1. **阈值设定**：初始值 `VECTOR_MIN_SPATIAL = 128`（910B）；应基于 `per_tile_elements × sizeof(T)` 而非仅 `spatial_size`，且 910B/950 阈值不同需 per-arch 配置
2. **标量路径设计**：不使用 EnQue/DeQue（直接 AllocTensor/FreeTensor），避免流水线同步开销；适合每 tile 数据量 < 512B 的场景
3. **多级 dispatch**：TopK 等复杂算子用 3-5 级 dispatch 比单一阈值更有效（按 rowLength、对齐性、dtype 综合决策）
4. **Profiling 驱动调优**：初始阈值部署后，通过 profiling 对比各 case 在两种路径下的性能，自动调整阈值（±16 步长）
5. **预期收益**：消除小 workload 退化 case（避免负优化），总体 geomean +10-25%

## 常见陷阱

⚠️ 阈值过高 → 大部分 case 走标量，向量化收益未充分发挥
⚠️ 阈值过低 → 小 case 走向量路径，EnQue/DeQue 开销导致退化（GroupNorm spatial≈1512 的 case 曾从 2x 退化到 0.027x）
⚠️ 多级 dispatch 增加代码复杂度（分支数 × 每分支代码量），需权衡收益
⚠️ 阈值基于 `spatial_size` 而非 `per_tile_workload` 可能误判（batch 内 tile 数不同的场景）

## 代码搜索关键词

```bash
grep -nE "VECTOR_MIN_SPATIAL|ScalarPath|VectorizedPath|dispatch|ProcessBatched" op_kernel/*.cpp op_host/*.cpp
```

## 来源

- GroupNorm 进化 (11_GroupNorm_evo) P0: 混合调度架构，spatial≥128→向量，<128→标量，消除小 case 退化
- TopK 进化 (TopK_evo_0629): 5 级 dispatch（BatchSmall/BatchMed/BatchAligned/BatchStaged/Row），geomean 4.08x
