---
id: X27
origin: discovered
discovered_round: 1
discovered_from: round_1/parallel_2 (x0, open_exploration)
base_speedup: 2.40x vs 基线 / 0.771x vs PyTorch
confidence: single_op
status: active
op_families: [matmul]
bottlenecks: [fixed_overhead_dominated, scalar_compute, compute_bound]
tags: [manual-mmad, transpose-on-load, full-shape-grid, aic-only, no-host-transpose, LoadToL0B]
strategy_ids: [X27]
---

### X27: transpose-on-load 分形加载 + 全 shape 手动 mmad 二维 tile 网格（免 host 预转置）

**Source**: 5_MatmulTransB evolution（KB 验证轮），round 1, parallel 2（x0, open_exploration）
**Speedup**: geomean 0.35x → 0.771x vs PyTorch（设备 13.68μs → 5.70μs，2.40x vs 基线，50/50 精度 PASS）；21 个 case ≥1.0x（最高 1.18x）；fp32[64,128]=6.3μs（vs x24 host-transpose 版 11.2μs）、fp16[128,256]=5.1μs（vs x24 10.9μs）

**Operator type**: matmul（微小/中 shape，固定开销主导），C[M,N]=A[M,K]@B[N,K]^T（B 转置存储）
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
在 x24（手动 mmad + host B 预转置 + MatmulImpl fallback）基础上做两个关键替代，把「手动 mmad 可覆盖范围」从单 tile 微小 shape 扩展到**全部 50 shape 无一走 fallback**：

1. **transpose-on-load 替代 host B 预转置**：B 的 transB 由 `LoadToL0B<LOAD2D>` 的 `isBTranspose=true` 在 L1→L0B 分形加载时完成，**不需要额外的 host workspace 转置 pass**。关键实证：这不是 LoadData3D（f3 禁用的 16x16-only）也不是坏掉的 LoadData2D ifTranspose，而是 **L1→L0B 的 LoadToL0B LOAD2D isBTranspose=true 形态，910B2C 上对任意 M/N/K 正确**。免去 host 转置的内存 pass 和 workspace 管理。
2. **全 shape 二维 tile 网格**：输出 C 按 gridM×gridN 切 16 对齐 tile，每 tile 保留 FULL K（cube MMAD 内部吞 K），单条 Mmad 完成整 tile；按 `tid=blockIdx+i*blockDim` 走 tile 网格。把 x24 落入 MatmulImpl fallback 的中大 shape（fp32[64,128]/[128,256]/[100,200]、fp16[128,256]）也用手动路径跑，实测比 MatmulImpl 快 ~1.7-2x。配合「并行度扩展」启发式：per-tile 操作数 >96KB 且 grid<24 时拆分大维摊到更多 core。
3. 叠加 x24 已有手段：AIC_ONLY 去 AIV 空转、紧凑 tiling 只读 9 个 int32、单 tile 整块装入 L0、fp32 Fixpipe NoQuant。

**Key code change**:
```cpp
// device 侧（matmul_trans_b_manual.h）：
// 1) B 分形加载时完成转置（免 host 预转置 pass）
LocalTensor<half> l0b = l0bQue.AllocTensor<TPosition::B1>();
LoadToL0B<LOAD2D>(l0b, l1B, l0bLoad, isBTranspose=true);  // L1→L0B, transB 在读入时完成
// 2) 单条 Mmad（FULL K，cube 内部吞 K）
Mmad(l0c, l0a, l0b, mmadParams);   // unpadded m/n/k
// 3) fp32 Fixpipe NoQuant
FixpipeParamsV220 fp; fp.quantMode = QuantMode_t::NoQuant;
Fixpipe(l0c, cGm, ..., fp);
// 4) AIC_ONLY（*_fp16/fp32/bf16.cpp）
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
```

**Applicability conditions**:
- transB 类 matmul（B 转置存储）微小→中 shape：transpose-on-load 免 host 预转置 workspace pass，比 x24 少一次内存往返
- 需要「全 shape 手动覆盖」（不依赖 MatmulImpl fallback）时：二维 tile 网格 + 并行度扩展启发式，中大 shape 也手动化
- 前提：910B2C 上 LoadToL0B LOAD2D isBTranspose=true 已实证正确（区别于 f3 的 LoadData3D 陷阱、LoadData2D ifTranspose 坏路径）
- 上限：1x1 等极微 shape 受 ~4.5μs 设备启动地板限制（~0.3x，硬件不可达）；fp32[256,512]（K 超大）需 K-slicing 但收益仅 ~0.01x geomean

**Strategy IDs**: X27（关联 X24 AIC_ONLY、x24 手动 mmad proven_solution、f3 LoadData3D 陷阱规避）
