---
schema_version: okf.v1
kind: proven_solution
title: "MatmulTransA: Cube Tiling Systematic Design"
source: "MatmulTransA generation (`4_MatmulTransA.py`), output path `output_0725_v2/`"
speedup: "generated AscendC vs PyTorch geomean 0.964x, 50/50 PASS (parity-level Cube baseline)"
operator_type: "matmul"
hardware: "Ascend 910B3 (npu13)"
confidence: cross_op_verified
status: active
op_families: ['matmul']
strategy_ids: P46 (MatmulImpl high-level API), P104 (multi-core tiling expansion), P108 (platform routing)
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

---
