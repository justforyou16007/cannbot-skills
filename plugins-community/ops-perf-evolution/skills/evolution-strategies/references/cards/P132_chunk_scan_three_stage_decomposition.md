---
id: P132
bottlenecks: [compute_bound, scalar_compute]
op_families: [ssm, linear_attention, recurrent, matmul, cv_fusion, reduction]
complexity: L2
conflicts_with: []
synergizes_with: [P133, P134, P135, P136]
requires: []
has_preconditions: true
has_playbook: true
---

# P132: 线性递归 chunk-scan 三阶段分解（块内预计算 / 块间状态扫描 / 输出投影）

## 核心思想

线性递归 / 状态空间类算子（Gated Delta Rule、Gated Linear Attention、Mamba 等，**无 QK^T、无 softmax**）的核心是沿序列维递推一个状态矩阵 S：

```
逐 token 串行：S_t = f(S_{t-1}, x_t)     # 每个 token 依赖前一个，O(seq) 串行依赖
```

逐 token 串行导致核内循环无法并行、每个 token 一次小计算 + 一次状态访存，性能被串行链和启动开销封顶。**三阶段分解**把序列切成 `C` 大小的 chunk（参考实现 C=64），把串行依赖压缩到块间：

```
Stage1 块内预计算（全并行）：本 chunk 内部的可并行量
  KKT = k @ k^T · β          # C×C×Dk 外积聚合成矩阵（GDR）
  QKT = q @ k^T              # C×C×Dk
  g_cum = cumsum(g)；decay = exp(g[i]-g[j])     # 门控衰减累积
  attn_1 = (I + KKT·decay)^-1                    # 矩阵求逆（GLA 用 cumprod 衰减替代）
  k_cumdecay = attn_1 @ kcd；v_inner = attn_1 @ v_beta；q_prime = q·scale·g_cum_exp

Stage2 块间状态扫描（仅小状态串行）：state (Dv, Dk) 跨 chunk 递推
  state ← state·decay + v_new^T @ kg            # C 个 token 压缩成一次状态更新

Stage3 输出投影：out = attn_inter + masked_qkt @ v_new
```

**收益**：块内所有 `C×C` / `C×D` matmul 全并行（吃满 cube）；块间只剩 64 个 token 的状态递推，串行依赖从 O(seq) 压到 O(seq/C)。

与 FA 的关系：FA 是 softmax 注意力（块内是 KKT/softmax/PV），本卡是**无 softmax 的线性递归**（块内是 KKT+求逆/衰减）。两者共享"分块 + 块内预计算 + 块间扫描"的骨架，但块内数学完全不同。

## 代码骨架

```python
# === 改造前：逐 token 串行递推 ===
state = initial_state                       # (Dv, Dk)
for t in range(seq_len):
    k_t, v_t, g_t = key[t], value[t], g[t]  # 每个 token 一次状态访存 + 小计算
    state = state * exp(g_t) + outer(v_t, k_t)   # 串行链，无法并行

# === 改造后：三阶段 chunk 分解（C=64）===
# Stage1 块内预计算（对每个 chunk，全并行）
for cg in chunk_groups:                     # totalChunk = Nv × n_chunks 按核并行
    kkt   = KKT(cg)                         # cube: k@k^T·β
    gcum  = cumsum(g[cg])                   # vector
    attn1 = invert(I + kkt·exp_decay(gcum)) # GDR: solve_triangular；GLA: cumprod 替代
    kcd   = attn1 @ gBK(cg)                 # cube: k_cumdecay
    v_in  = attn1 @ vBeta(cg)               # cube: v_inner
    qpr   = q[cg]·scale·exp(gcum)           # vector: q_prime

# Stage2 块间状态扫描（串行，每 chunk 一次状态更新）
state = initial_state
for cg in chunk_groups:                     # O(seq/C) 次
    attn_inter[cg] = qpr @ state^T          # cube
    v_new[cg]     = v_in + kcd @ state^T
    state = state·decay + v_new[cg]^T @ kg[cg]

# Stage3 输出投影
out = attn_inter + masked_qkt @ v_new       # cube + vector 融合
```

## 关键修改点

1. **切 chunk（C=64）**：序列按 C 对齐 pad；`n_chunks = padded // C`。C 过小 → 块间串行多、启动多；C 过大 → 块内 C×C matmul 超 UB。
2. **块内预计算全并行**：Stage1 的任务是 `Nv × n_chunks` 个独立 chunk，按核均分，无跨核依赖。
3. **块间只递推小状态 (Dv, Dk)**：Stage2 每 chunk 只做几次 `(C,Dv)@(Dv,Dk)` / `(Dv,Dk)` 小 matmul + 一次状态衰减更新，把串行代价降到最低。
4. **数值对齐参考**：块内中间量（KKT/attn_1/k_cumdecay/v_inner/q_prime/kg）按参考实现全程 bf16，避免 UB 翻倍（见 P136）。
5. **GLA 差异**：无求逆，Stage1 用 `cumprod(exp(g))` 计算跨 chunk 衰减，Stage2 状态更新为 `state ← state·cumdecay + ...`（无 corr 修正项）。

## 常见陷阱

- ❌ 没切 chunk 直接逐 token 循环 → 串行链 + 启动开销封顶，进化轮次浪费在通用循环优化上。
- ❌ 块间状态 (Dv,Dk) 每次整块搬 GM → 状态应 UB 常驻，跨 chunk 复用（见 P136）。
- ❌ 把块内求逆/衰减也做成串行 → 块内是全并行的，只有块间是串行。
- ❌ 对无 QK^T 的递归算子误走 FA workflow（P119/P120 online-softmax）→ 数学不适用，必须确认无 softmax 后走 ssm-scan。
- ❌ C=64 但尾 chunk 没做变长处理 → 尾块 pad 的伪计算吃满 64，用 `lastChunkSize = seq % C` 特化。

## 代码搜索关键词

```bash
grep -nE "for .*in range\(seq|chunkSize|CHUNK_SIZE|n_chunks|cumsum|solve_triangular|cumprod|state.*state|k_cumdecay|v_inner|g_cum" op_kernel/*.cpp op_kernel/*.h model.py
```
