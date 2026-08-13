# Attention Family Optimization Insights

## Operator Classification（线性 / 稀疏 / 压缩 大类别）

Attention 按**计算特性**分三大类：线性、稀疏、压缩。标准 Softmax Attention（MHA / cross / self）是基础形态，三大类是它的变体方向；**通道注意力（SE/SK 等）不是 attention 计算**，不归本类。

- **基础 Softmax Attention**（默认）：MHA / cross / self / SDPA / FlashAttention —— `QK^T → scale → mask → softmax → PV`，online-softmax 状态机
- **稀疏 Attention**：SparseFlashAttention / BlockSparse / Strided / Window / Local / BigBird / Longformer —— softmax 保留 + 稀疏 mask，大部分 score 无效
- **压缩 Attention**：MLA（latent c 压缩）/ GQA / MQA（head 共享）/ Paged（kv-cache）—— KV 经压缩表示
- **线性 Attention**：GLA / Mamba2（SSD）/ GDN-KDA / LinearAttention / Nyström / Linformer —— 无 QK^T、无 softmax，状态递推 + chunk-scan
- **通道注意力**（排除）：SE / SK / Triplet / CBAM / CoordAtt —— `global pool → FC → sigmoid/softmax → 通道 scale`，无 QK^T、无 head 结构 → 归 reduction / elementwise

### 标准 Softmax Attention（head 结构）
- **MHA** (Multi-Head Attention): num_q_heads == num_kv_heads
- **GQA** (Grouped Query Attention): num_q_heads = G × num_kv_heads
- **MQA** (Multi-Query Attention): num_kv_heads == 1（GQA 特例）

### 稀疏 Attention（softmax + 稀疏 mask）
- BlockSparse / Strided / Window / BigBird / Longformer / SparseFlashAttention / GqaSparse
- 判据：softmax 注意力 + 稀疏 pattern（block_table / band / strided / top-k）
- 优化：flashattention.md **sparse trait** —— P123（skip + 边界 mask）、**P137**（紧凑化走 Cube，top-k 不再回退 Vector）、P138/P139（L0 守卫 / 多槽零填充稳定性）、P76/P77（sparse KV）

### 压缩 Attention（KV 压缩表示）
- MLA（latent c 压缩：`c → W^U → [nope|rope] → K/V`）/ GQA / MQA（head 共享）/ Paged（block table）
- 判据：KV 经压缩表示（latent 向量 / `num_kv_heads < num_q_heads` / kv-cache）
- 优化：flashattention.md **mla 分支**（P124 nope/rope 拼 K + P125 跨任务预取 + P126 任务表 + **P140** M-tile 放大）、**gqa 分支**（P116 G→M 合并 + X5 decode GEMV→GEMM + X4）

### 线性 Attention（无 QK^T、无 softmax）
- GLA（门控衰减 `S = α⊙S + k⊗v`）/ Mamba2（SSD）/ GDN-KDA（delta rule `S ← β⊙(k⊗v − corr) + S`，含 corr 修正）/ LinearAttention / Nyström / Linformer
- 判据：沿序列维状态递推 + chunk-scan；有 `cumprod` 衰减 / `solve_triangular` 求逆
- 优化：ssm-scan.md —— P132 三阶段分解（GLA 用 cumprod 替代求逆）+ P133 块内 CV 流水 + P134 任务划分 + P135 求逆分治（仅 GDN-KDA）+ P136 bf16 量化

## Key Algorithmic Optimizations

### 1. Flash Attention (Tiled Online Softmax)
```
Standard attention: O = softmax(QK^T / √d) × V
  Problem: QK^T is S×S matrix — O(S²) memory, doesn't fit in SRAM

Flash Attention: Process Q×K^T in tiles, maintain running max and sum
  - Tile K,V along S2 dimension into blocks of size B
  - For each Q block, iterate over K,V blocks:
    score_ij = Q_i × K_j^T / √d
    m_new = max(m_old, rowmax(score_ij))
    P_ij = exp(score_ij - m_new)
    l_new = l_old × exp(m_old - m_new) + rowsum(P_ij)
    O_i = O_i × (l_old × exp(m_old - m_new) / l_new) + P_ij × V_j / l_new

  Benefit: O(S) memory, O(S²) compute (same), but fits in SRAM
```

### 2. Causal Masking Optimization
```
Causal mask: entry (i,j) is -inf if j > i

Block-level optimization:
  For S2 block [s2_start, s2_end] and S1 block [s1_start, s1_end]:
    if s2_start > s1_end: SKIP entire block (all masked)
    if s2_end ≤ s1_start: FULL block (no masking needed, all valid)
    else: PARTIAL block (needs element-level masking)

  Theoretical skip rate for causal: (n-1)/(2n) → 50% for large n
  Practical: 45% at S=32K, 37% at S=2K, 22% at S=512
```

### 3. GQA KV Sharing
```
When num_q_heads > num_kv_heads:
  KV head k serves Q heads [k×G, (k+1)×G - 1] where G = num_q_heads/num_kv_heads

  Implementation approaches:
  a) Expand KV: repeat_interleave K,V to match Q head count (wastes memory)
  b) Shared indexing: Map Q head to KV head via kv_idx = q_head_idx / G
     (saves memory, same compute)

  CRITICAL: When computing causal token for GQA:
    token = s1_row_within_block  (correct: based on row position)
    NOT: token = global_q_head_offset + s1_row  (WRONG for grouped layout)
```

### 4. KV-Cache (Inference Optimization)
```
During inference, past K,V are cached and only new tokens are computed.

Prefill phase: Process full sequence (like training)
Decode phase: Q has length 1, K,V have length seq_len_kv

Optimization for decode:
  - No tiling needed for Q dimension (single row)
  - Tile only along KV sequence dimension
  - Memory-bound (tiny compute per token)
```

### 5. Sliding Window Attention
```
Instead of full causal mask, only attend to last W tokens:
  valid if: i - W < j ≤ i

Block-level skip: skip blocks where s2_end < s1_start - W
Additional skip vs. standard causal: blocks far behind the window
```

## Industry Reference Implementations

### FlashAttention-2 (Tri Dao)
- Key innovation: Parallelize over sequence length, not batch/head
- Reduces non-matmul FLOPs by optimizing the online softmax rescaling
- 2× speedup over FlashAttention-1 on A100

### FlashAttention-3 (Tri Dao, 2024)
- Asynchronous execution with warp specialization
- FP8 quantization support
- Intra-warp pipelining for better hardware utilization

### PagedAttention (vLLM)
- Block-level KV cache management for inference
- Reduces memory fragmentation in serving scenarios

## Ascend-Specific Considerations

1. **Cube-Vector split**: On Ascend 910B, QK^T runs on Cube core, softmax+P×V runs on Vector core. Optimization must balance both.

2. **Workspace for online softmax**: Need GM workspace to store intermediate m, l, O across S2 tiles. MTE3 writes to workspace are significant overhead.

3. **Block size selection**: Must align with Cube unit's matmul granularity (typically 16×16 or 32×32 tiles).

4. **Pre/next token parameters**: Ascend attention APIs use `preTokens` and `nextTokens` to define the attention window, equivalent to a band mask.

