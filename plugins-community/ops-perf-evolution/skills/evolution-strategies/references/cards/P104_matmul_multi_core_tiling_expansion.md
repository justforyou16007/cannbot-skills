---
source_key: evolution-strategies#card/P104_matmul_multi_core_tiling_expansion
id: P104
title: Matmul Multi-Core Tiling Expansion
doc_path: cards/P104_matmul_multi_core_tiling_expansion.md
bottlenecks: [parallelism_underutil, tiling_imbalance]
op_families: [matmul]
complexity: L1
triggers:
  - MatmulApiTiling::GetTiling returns totalBlock < aicCoreNum
  - singleCoreM == M or singleCoreN == N causing blockDim == 1
  - M/N large enough to split across cores while keeping ALIGNED_H=16
conflicts_with: [P72]
synergizes_with: [P4, P105, P107]
requires: []
has_preconditions: true
has_playbook: true
references:
  - ops/ascendc-tiling-design/references/matmul/ascendc-api-matmul-tiling.md
---

# P104: Matmul 多核切分扩展（Multi-Core Tiling Expansion）

## 核心思想

`MatmulApiTiling::GetTiling(TCubeTiling&)` 自动给出的 `singleCoreM / singleCoreN` 可能直接将整行/整列矩阵分配给一个核，导致 `totalBlock = 1`、`blockDim = 1`，多核完全未生效。P104 在 Host 侧**手动缩小 `singleCoreM / singleCoreN`**，使 `totalBlock ≥ aicCoreNum`，同时保持 16 字节对齐且不低于 `baseM / baseN`，从而把任务扩散到所有 AIC 核。

## 代码骨架

```cpp
// === 改造前（基线）：GetTiling 后直接序列化，可能只用到 1 个核 ===
MatmulApiTiling::GetTiling(cubeTiling);
// cubeTiling.singleCoreM 可能 == M，totalBlock == 1

// === 改造后：多核切分扩展 ===
constexpr uint32_t ALIGNED_H = 16;
auto singleCoreM = cubeTiling.singleCoreM;
auto singleCoreN = cubeTiling.singleCoreN;
auto baseM = cubeTiling.baseM;
auto baseN = cubeTiling.baseN;
auto aicCoreNum = platform.GetCoreNumAic();

int64_t mTotalCnt = CeilDiv(M, static_cast<int64_t>(singleCoreM));
int64_t nTotalCnt = CeilDiv(N, static_cast<int64_t>(singleCoreN));
int64_t totalBlock = mTotalCnt * nTotalCnt;

if (totalBlock < aicCoreNum) {
    while (totalBlock < aicCoreNum && singleCoreM > baseM) {
        singleCoreM = std::max(baseM, (singleCoreM / 2) / ALIGNED_H * ALIGNED_H);
        mTotalCnt = CeilDiv(M, static_cast<int64_t>(singleCoreM));
        totalBlock = mTotalCnt * nTotalCnt;
    }
    while (totalBlock < aicCoreNum && singleCoreN > baseN) {
        singleCoreN = std::max(baseN, (singleCoreN / 2) / ALIGNED_H * ALIGNED_H);
        nTotalCnt = CeilDiv(N, static_cast<int64_t>(singleCoreN));
        totalBlock = mTotalCnt * nTotalCnt;
    }
}

cubeTiling.singleCoreM = singleCoreM;
cubeTiling.singleCoreN = singleCoreN;
cubeTiling.usedCoreNum = std::min(totalBlock, aicCoreNum);

int32_t mBaseTail = M - (mTotalCnt - 1) * singleCoreM;
int32_t nBaseTail = N - (nTotalCnt - 1) * singleCoreN;
```

## 关键修改点

1. **强制对齐**：缩小后的 `singleCoreM / singleCoreN` 必须是 `ALIGNED_H = 16` 的整数倍。
2. **下限保护**：不得小于 `baseM / baseN`，否则 cube 一次发射的尺寸无法保证。
3. **同步更新派生字段**：`mTotalCnt / nTotalCnt / totalBlock / usedCoreNum / mBaseTail / nBaseTail` 全部重新计算。
4. **Kernel 尾块守卫**：Kernel 端按 `(mIdx == mTotalCnt - 1) ? mBaseTail : singleCoreM` 调用 `SetSingleShape`。

## 常见陷阱

⚠️ 缩小后未重新计算尾块，导致最后一块越界写 GM。
⚠️ `usedCoreNum` 未取 `min(totalBlock, aicCoreNum)`，富余核无任务空转。
⚠️ Kernel 入口缺少 `if (GetBlockIdx() >= totalBlock) return;`，导致越界。
⚠️ 将 `singleCoreM / N` 缩小到低于 `baseM / N`，Cube 发射异常或性能崩溃。

## 代码搜索关键词

```bash
grep -nE "MatmulApiTiling|GetTiling|singleCoreM|singleCoreN|baseM|baseN|totalBlock|usedCoreNum|GetCoreNumAic" \
    op_host/*_tiling.cpp op_kernel/*.cpp
```

## 来源

- `ops/ascendc-tiling-design/references/matmul/ascendc-api-matmul-tiling.md` §多核切分扩展
