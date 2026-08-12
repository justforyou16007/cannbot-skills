---
id: P135
bottlenecks: [compute_bound, scalar_compute]
op_families: [ssm, linear_attention, recurrent, matmul, reduction]
complexity: L2
conflicts_with: []
synergizes_with: [P132, P133]
requires: [P132]
has_preconditions: true
has_playbook: true
---

# P135: delta-rule 矩阵求逆分治（对角向量前代 + 非对角 cube MM）

## 核心思想

Delta Rule（GDN/KDA）类算子块内预计算要解一个 `C×C` 下三角线性系统（GDR：`attn_1 = (I + KKT·attn)^{-1}`），其中 `A = -(KKT·attn).tril(-1)` 是严格下三角。朴素求逆是 O(C³) 串行逐行回代，C=64 时是一个明显的计算瓶颈，且 cube 完全闲着。

**分治**把求逆拆成"对角子块向量前代 + 非对角 cube MM"：

```
把 X = (I-A)^{-1} 按对角子块（INVERSE_SHAPE）分块：
· 对角子块 X11：向量 forward substitution（逐行回代，串行但块小）→ InverseCompute
· 非对角子块 X10 = X11 · M10 · X00：cube 一次 MM 完成 → AttnInverseMMCompute
```

```
X = | X00  0  |^{-1}        X00、X11 = 对角子块求逆（向量前代）
    | X10 X11 |             X10 = -X11 · A10 · X00（cube MM）
```

**收益**：O(C³) 串行求逆的绝大部分（非对角）交给 cube 并行算，向量只留小的对角子块前代；同时为 P133 的 CV 流水提供可握手的 C2a 阶段（AIV 前代完 → SetFlag 0x7 → AIC 算 off-diag）。

**GLA 差异**：GLA（门控线性注意力）用 `cumprod(exp(g))` 累积衰减，**没有求逆**——本卡仅适用于 GDN/KDA（含 corr 修正项的 delta rule）。

## 代码骨架

```cpp
// === 改造前：整块求逆（逐行回代 / solve_triangular，O(C³) 串行，cube 空闲）===
// attn_1 = solve_triangular(I + KKT*attn, I, upper=False)   // 全在向量/标量

// === 改造后：对角前代 + 非对角 cube MM ===
// AIV：对角子块向量前代（InverseCompute）
for (sub = 0; sub < chunkSize / INVERSE_SHAPE; sub++) {
    Mul(attnUb, attnUb, gammaUb[sub*C]);            // 乘衰减
    InverseCompute(attnWs, gammaUb);                // 前代：X11 = solve(...)
}
CrossCoreSetFlag<0x2, PIPE_MTE3>(0x7);              // 通知 AIC off-diag 可算

// AIC：非对角子块 cube MM（AttnInverseMMCompute）
for (i...) {
    AttnInverseMMCompute(i * ccOffset);             // X10 = X11 · M10 · X00
}
```

## 关键修改点

1. **识别求逆来源**：delta rule 的 `(I + KKT·β·decay)^{-1}`；确认是 delta rule（有 corr/β 修正）而非 GLA（cumprod，无求逆）。
2. **分块**：按 `INVERSE_SHAPE` 子块分对角/非对角；非对角块数 = `(C/INVERSE_SHAPE)^2 - (C/INVERSE_SHAPE)`。
3. **对角向量前代**：逐行 forward substitution，只处理对角子块，串行但规模 O(C²) 小。
4. **非对角 cube MM**：`X10 = X11 · M10 · X00` 一次 mmad，cube 吞吐；依赖对角前代完成（flag 0x7）。
5. **接 P133 流水**：前代（AIV）与 off-diag（AIC）用定向 flag 握手，融入块内预计算 CV 链。

## 常见陷阱

- ❌ 对 GLA 用本卡 → GLA 无求逆，用 cumprod 衰减；误用会找不存在的求逆。
- ❌ 整块 `solve_triangular` 放向量 → O(C³) 串行，cube 空闲；必须分治。
- ❌ 非对角也用向量前代 → 非对角是大量独立 MM，cube 明显更快。
- ❌ 对角前代没写完就发 flag → AIC 读到未完成的 X11，结果错；flag 0x7 必须在前代全部结束后。
- ❌ 分块粒度与 mmad 形状不匹配 → 非对角 MM 的 M/N/K 要对齐 16；`INVERSE_SHAPE` 按 cube 形状选。

## 代码搜索关键词

```bash
grep -nE "solve_triangular|Inverse|inverse|triangular|tril|forward|前代|substitution|AttnInverseMMCompute|InverseCompute|INVERSE_SHAPE|M10" op_kernel/*.{cpp,h} model.py
```
