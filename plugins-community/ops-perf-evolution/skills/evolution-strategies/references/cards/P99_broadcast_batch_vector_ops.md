---
id: P99
bottlenecks: [scalar_compute, scalar_loading, compute_bound]
op_families: [normalization, elementwise, reduction, softmax]
complexity: L1
conflicts_with: []
synergizes_with: [P94, P95, P69]
requires: []
has_preconditions: true
has_playbook: true
---

# P99: BroadCast + 批量向量运算替代逐行标量循环 (BroadCast + Batch Vector Ops)

## 核心思想
当小向量（如 `rstd[gRows]`）需要广播到大矩阵（`[gRows, D]`）做逐元素运算时，优化前用 `for` 循环逐行执行 `Muls`/`Adds`/`Subs`（gRows 条 Vec 指令，每条仅处理 D 个元素），指令 dispatch 开销被放大 gRows 倍。优化后用 `BroadCast` 先将小向量展开为与大矩阵同形的中间 tensor，再用 1 条 `Mul`/`Add` 指令覆盖全量数据。从 gRows 条 Vec 指令减少到 2 条（BroadCast + 批量运算），指令 dispatch 开销下降约 gRows/2 倍。

## 代码骨架

```cpp
// === 改造前（基线）：逐行标量-向量 Muls ===
for (int32_t i = 0; i < gRows; i++) {
    int32_t off = (gStart + i) * UD;
    float rstd = rstdLocal.GetValue(i);    // 标量取值
    Muls(xF[off], xF[off], rstd, UD);      // 逐行 Muls，gRows 条 Vec 指令
}
// 每条 Muls 仅处理 UD(=128) 个元素 → Vec 指令开销 / 有效计算比极差

// === 改造后：BroadCast 展开 + 1 条 Mul 覆盖全量 ===
{
    const uint32_t src_shape[2] = {static_cast<uint32_t>(gRows), 1};
    const uint32_t dst_shape[2] = {static_cast<uint32_t>(gRows), UD};
    BroadCast<float, 2, 1>(xSq, rstdLocal, dst_shape, src_shape, bcast_tmp);
    //  ↑ rstdLocal [gRows, 1] → broadcast dim=1 → xSq [gRows, UD]
    PipeBarrier<PIPE_V>();
    Mul(xF[gStart * UD], xF[gStart * UD], xSq, gRows * UD);
    //  ↑ 1 条 Mul 处理 gRows × UD 个元素
}
```

## BroadCast API 参数速查

```cpp
BroadCast<T, NDIM, BROADCAST_DIM>(dst, src, dstShape, srcShape, tmpBuf);
// dstShape/srcShape 为 const uint32_t[NDIM] 数组；srcShape[BROADCAST_DIM] 必须为 1
// 示例: rstd [3, 1] → [3, 128]，在 dim=1 上广播
BroadCast<float, 2, 1>(dst, src, {3, 128}, {3, 1}, tmp);
```

## 关键修改点

1. **shape 数组**：必须 `const uint32_t[N]` 格式；`NDIM` 为维度数，`BROADCAST_DIM` 为被广播维度（0-based），且 src_shape 中该维必须为 1
2. **Buffer 分配**：输出 buffer（`xSq`）与临时 buffer（`bcast_tmp`）均需 ≥ `gRows × UD × sizeof(T)` 字节
3. **PipeBarrier 放置**：BroadCast 写 xSq 后、Mul 读 xSq 前加 `PipeBarrier<PIPE_V>()`；若两者写不同 buffer（xSq ≠ xF）则天然无冲突
4. **matvec 变体**：标量驱动的 `q @ Pi` 类 matvec 也可用 BroadCast + Mul + ReduceSum 实现，与 P94 形态 ε（复制-Mul-批量归约）互为替代方案，见 Playbook Step 3C
5. **预期收益**：gRows 条 Vec 指令 → 2 条，约 gRows/2 倍指令效率提升（gRows=32 时约 16x）

## 常见陷阱

⚠️ `NDIM`/`BROADCAST_DIM` 模板参数必须与 shape 数组维度严格匹配
⚠️ src_shape 中 BROADCAST_DIM 维必须为 1（否则不是广播而是拷贝）
⚠️ 广播后的中间 tensor 占用 `gRows × UD × sizeof(T)` 字节 UB，需确保容量足够
⚠️ gRows < 4 时 BroadCast + PipeBarrier 开销可能超过逐行 Muls，无需优化

## 代码搜索关键词

```bash
grep -nE "BroadCast|bcast_tmp|broadcast.*shape" op_kernel/*.cpp
```

## 来源

- tmp.md 优化点1：BroadCast rstd + 批量缩放替代逐行 Muls
- best_decode qrot kernel：标量驱动 matvec 场景，与 P94 形态 ε 互为替代实现
