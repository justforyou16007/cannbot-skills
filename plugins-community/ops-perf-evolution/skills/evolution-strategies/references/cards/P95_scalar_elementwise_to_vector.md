---
id: P95
bottlenecks: [scalar_compute, compute_bound]
op_families: [elementwise, normalization, reduction, broadcast_mask]
complexity: L1
conflicts_with: []
synergizes_with: [P67, P69, P84, P94]
requires: []
has_preconditions: true
has_playbook: true
---

# P95: 标量逐元素运算向量化 (Scalar Element-wise → Adds/Muls Vector Chain)

## 核心思想
检测 kernel 中逐元素标量算术循环（`for(i) y[i] = f(x[i])`），用 AscendC 标量-向量指令（`Adds`/`Muls`/`Subs`）和向量-向量指令（`Add`/`Mul`/`Div`/`Sub`）链替代。`Adds`/`Muls` 是标量-向量混合指令，每条指令对 tensor 所有元素同时执行 `y = x OP scalar`，无需 `Duplicate` 广播 buffer，4 条指令可替代 O(4N) 次标量循环。910B Vector 单元 256-bit 宽度下可同时处理 8×FP32 或 16×FP16，标量循环每次仅 1 个元素，理论加速比 8-16x。

## 代码骨架

```cpp
// === 改造前（基线）：标量逐元素归一化 ===
float* xPtr = reinterpret_cast<float*>(xLocal.GetPhyAddr());
float* yPtr = reinterpret_cast<float*>(yLocal.GetPhyAddr());
for (int32_t i = 0; i < count_spatial; ++i) {
    float val = xPtr[i];
    float normalized = (val - mean) * inv_std;
    yPtr[i] = normalized * w_val + b_val;
}

// === 改造后：Adds/Muls 标量-向量指令链（4 条指令） ===
// Adds/Muls 直接接收标量参数，无需 Duplicate 广播；中间结果全程留在 UB
AscendC::Adds<float>(yLocal, xIn, -mean, count_spatial);       // y = x - mean
AscendC::Muls<float>(yLocal, yLocal, inv_std, count_spatial);  // y = y * inv_std
AscendC::Muls<float>(yLocal, yLocal, w_val, count_spatial);    // y = y * weight
AscendC::Adds<float>(yLocal, yLocal, b_val, count_spatial);    // y = y + bias
```

## 可用标量-向量指令速查

| 指令 | 功能 | 替代的标量模式 |
|------|------|---------------|
| `Adds<T>(dst, src, scalar, len)` | dst[i] = src[i] + scalar | `y[i] = x[i] + c` |
| `Muls<T>(dst, src, scalar, len)` | dst[i] = src[i] * scalar | `y[i] = x[i] * c` |
| `Subs<T>(dst, src, scalar, len)` | dst[i] = src[i] - scalar | `y[i] = x[i] - c` |

## 关键修改点

1. **识别可融合链**：如果标量循环每次迭代计算 `y[i] = f_n(...f_1(x[i]))` 且中间结果不被他用，整条链可映射为 N 条向量指令，中间结果保留 UB 直通消费（n 次计算从 2n 次 GM 搬运减到 2 次）
2. **指令选择**：优先标量-向量指令（`Adds`/`Muls`/`Subs`），避免 `Duplicate` 广播→`Add`/`Mul` 两步操作
3. **UB 容量**：链中每个中间 tensor 占 `count × sizeof(T)` 字节，链越长生命周期越长
4. **隐蔽变体**：标量循环包裹向量 BLAS-like 调用（`for` 内 `Axpy`/`Muls` + `GetValue` 标量 scale），看到 `Axpy` 不代表已向量化，改造见 Playbook Step 3C 形态 ε
5. **预期收益**：单次标量-向量转换 3-10x；融合链消除 GM 往返额外 20-50%

## 常见陷阱

⚠️ `Adds`/`Muls` 的 scalar 参数类型必须与 tensor 元素类型一致（FP32 tensor + float scalar）
⚠️ 向量链中间结果不可超出 UB 容量（每个中间 tensor = `count × sizeof(T)` 字节）；若后续接 P1 双缓冲（num=2），CopyIn/CopyOut 队列占用 ×2，tileSize 需重新核算减半
⚠️ 连续 Vector 写同一 buffer 后如需 MTE3 搬运，必须加 `PipeBarrier<PIPE_V>()`
⚠️ 极小 count（< 128）时向量化开销可能超过标量，考虑混合调度（见 P98）

## 代码搜索关键词

```bash
grep -nE "Adds|Muls|Subs|Duplicate|SubRelu|vector.chain|localTensor" op_kernel/*.cpp
```

## 来源

- GroupNorm 进化 (11_GroupNorm_evo) P0: Adds/Muls 链替代归一化标量循环，bf16 4.31x geomean
- NMS 进化 (30_NMS): 15 条向量指令链替代逐候选框标量 IoU 计算，11.25x geomean
