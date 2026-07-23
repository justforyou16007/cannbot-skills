---
id: X7
origin: discovered
discovered_from: best_decode deep profiling (NPU 12, msprof compare)
base_speedup: 实测有效（2026-07-21，best_decode2 round_3/parallel_3 x1）：同窗口 mean_asc -4.2%（最净窗 -8.5%），case9 -42.6%/case10 -30.4%/case0 -21.1%；x1 最终 114.71us，为 merge_final 基座
---

# Strategy X7: Token Block Size (TB) Tuning for Pure-Vector Attention

## 核心思路

在纯 AIV（Vector）实现的 FlashDecode / TurboQuant decode 中，kernel 内部以
**token block（TB）** 为单位处理 KV cache：每个 chunk 加载 `TB` 个 token 的
slot，解量化后做 `q·K`、softmax、weighted-V 累加。TB 的选择直接决定：

1. **标量控制流占比**：TB 越小，循环迭代次数越多，`for` 循环、地址计算、
   online-softmax 标量更新占比越高（容易形成 Scalar Bound）。
2. **向量指令利用率**：TB 越大，每次 Vector op 处理的元素数越多（`TB * D`），
   向量单元更饱满。
3. **UB 容量压力**：TB 增大时，`slotBuf_`、`kBlkBuf_`、`vBlkBuf_`、
   `rawBlkBuf_` 等 buffer 线性增大，可能触顶。

当前实现中 FP8 路径 `TB=16`、MSE 路径 `TB=8`。分析显示 MSE 路径的
`aiv_scalar_ratio` 显著高于 FP8 路径，一个重要原因是其 TB 只有 8。
通过动态选择 TB（尤其是 MSE 路径），可以在 UB 容量允许范围内尽量增大 TB，
降低标量循环占比。

## 适用场景

- **算子类型**：纯 Vector 实现的 decode attention、quantized KV attention。
- **触发条件**：
  1. profiling 显示 `aiv_scalar_ratio` 高（>40%）且 `aiv_vec_ratio` 不高；
  2. 当前 TB 固定为较小值（如 8），而 UB 仍有剩余空间；
  3. MSE 路径比 FP8 路径慢的比例高于理论值（MSE 多一次 centroids gather）。
- **限制条件**：必须保证 `TB * slotPad_ * sizeof(T)` 不超过可用 UB。

## 前提条件

1. Kernel 的 buffer 分配支持动态 TB（当前用 `tb_` 成员变量控制，已支持）。
2. Host 侧能够将 TB 作为 Tiling 参数传入 kernel。
3. 已验证不同 TB 下的精度和边界行为（尤其是尾块 `nTok < TB`）。

## 实现要点

### 动态 TB 计算

Host 侧根据路径和 UB 预算选择 TB：

```cpp
int64_t ComputeTb(bool keyFp8, int64_t D, int64_t slotLen, int64_t slotPad) {
    // 基础约束
    const int64_t UB_SIZE = 128 * 1024;  // 910B AIV UB, 需按实际平台查询
    const int64_t RESERVED = 20 * 1024;  // 栈、队列、中间 buffer 预留
    const int64_t FP8_TB = 16;
    const int64_t MSE_TB_MIN = 8;

    if (keyFp8) {
        return FP8_TB;  // FP8 无 pitBuf_，UB 压力小
    }

    // MSE 路径：有 pitBuf_ (D*D fp32) + centBuf_ + 各种 offset/blk buffer
    // 计算当前配置下允许的最大 TB
    int64_t pitBuf = D * D * sizeof(float);
    int64_t centBuf = 16 * sizeof(float);
    int64_t qBuf = D * sizeof(half);  // 或 bfloat16
    int64_t fixed = pitBuf + centBuf + qBuf + 4096;  // 其他固定 buffer 估算

    int64_t available = UB_SIZE - RESERVED - fixed;
    // 每个 token 的主要可变 buffer：slot + kBlk + vBlk + raw + i32a + i32b + f32a + f32b
    int64_t bytesPerTok = slotPad * sizeof(int32_t)   // streamBuf_/streamHalfBuf_
                        + D * sizeof(float) * 2       // kBlkBuf_ + vBlkBuf_
                        + D * sizeof(int32_t)         // rawBlkBuf_
                        + D * sizeof(int32_t) * 2     // i32aBuf_ + i32bBuf_
                        + D * sizeof(float) * 2;      // f32aBuf_ + f32bBuf_

    int64_t maxTb = available / bytesPerTok;
    if (maxTb < MSE_TB_MIN) { maxTb = MSE_TB_MIN; }
    if (maxTb > 16) { maxTb = 16; }
    // 对齐到 8，保持 gather/offset 模式简洁
    maxTb = (maxTb / 8) * 8;
    if (maxTb < MSE_TB_MIN) { maxTb = MSE_TB_MIN; }
    return maxTb;
}
```

### 去掉 pitBuf_ 以进一步增大 TB

MSE 路径中 `pitBuf_`（`D*D` fp32）是最大的固定开销。若 Host 侧预计算 qRot
（即 `useQrot=1`），kernel 不再需要从 `pitGm_` 加载 Pi 并做 matvec，可以：

1. 去掉 `pitBuf_`，释放 64KB UB；
2. MSE 路径 TB 从 8 提升到 16；
3. 预期 scalar ratio 大幅下降。

这是本策略与 X4（Host 预计算）的典型组合。

## 预期收益

- MSE 路径 `aiv_scalar_ratio` 预计从 43% 降至 30% 以下；
- 每个 chunk 的向量指令数不变，但 dispatch 次数减少一半；
- 综合收益：MSE 路径耗时降低 15%-25%（取决于 shape）。

## 常见陷阱

⚠️ **UB 溢出**：增大 TB 前必须精确计算 UB 使用量，尤其是 `slotPad_` 已按 32B
对齐，实际占用可能大于 `slotLen * TB`。

⚠️ **尾块处理**：`nTok < TB` 时当前代码用 mask/stale-lane 方式处理，TB 改变后
需重新验证 softmax 正确性。

⚠️ **Gather offset 表**：TB 增大后，`kOffBlkBuf_`、`vOffBlkBuf_` 等大小变为
`TB * D`，Init 阶段的 offset 预计算循环变长，但这是一次性开销。

## 代码搜索关键词

```bash
grep -n "tb_\|TB_MAX_\|token.*block\|TB=\|tb =" op_kernel/*.cpp op_host/*.cpp

grep -n "InitBuffer.*slotBuf\|InitBuffer.*kBlkBuf\|InitBuffer.*vBlkBuf\|InitBuffer.*pitBuf" \
    op_kernel/*.cpp

grep -n "useQrot\|qrotGm_\|pitBuf_" op_kernel/*.cpp
```

## 来源

发现于 best_decode 深度性能分析：FP8 路径 TB=16 的 scalar ratio 为 52%，
而 MSE 路径 TB=8 的 scalar ratio 为 43%（且 vec ratio 明显更高）。
MSE 路径若去掉 `pitBuf_` 并将 TB 提升到 16，有望接近 FP8 路径的标量占比。
