# Attention Family Optimization Insights

## Operator Classification（线性 / 稀疏 / 压缩 大类别 + 具体小类）

Attention 按**计算特性**分三大类：线性、稀疏、压缩。标准 Softmax Attention（MHA / cross / self）是基础形态，三大类是它的变体方向；**视觉注意力（CV-Attn）**与**通道注意力**不是序列 attention 计算，单独归类；**名字带 Attention 但不是注意力的**排除。小类划分依据 level4 分析（计算模式 CP1-CP13），每个算子可跨多个 CP（如 MLA 横跨 CP3+CP4+CP11）。

### 标准 Softmax Attention（CP1 / CP2）
| 小类 | CP | 机制定义 | 代表算子 |
|---|---|---|---|
| **MHA** | CP1 | 每 Q head 配独立 KV head，`[S,S]` 分数矩阵完整物化（两次 GEMM + 一次行归约） | 13,17,18,19,26,45,70 |
| **GQA** | CP1+CP3 | query 分组共享 KV head，`kv_head = q_head // group` | 21,33,80,81,83,84 |
| **MQA** | CP1 | GQA group=1 退化：全部 Q 共享单份 KV | 22 |
| **FA** | CP2 | 分块流式遍历 KV，维护 running max/l，不物化 `[S,S]`（与 MHA 数学等价、kernel 骨架不同） | 20,27,28,29,54 |

### 稀疏 Attention（softmax + 稀疏 mask）
| 小类 | CP | 机制定义 | 代表算子 |
|---|---|---|---|
| **Sparse-掩码型** | CP1 | 先全量算 QK^T 再 masked_fill(-inf)，**不省计算**仅保语义 | 14,23,24,25,35,37,51 |
| **Sparse-块跳过型** | CP3 | block-mask/top-k 确定 KV 子集后流式累加，**真正减少 GEMM**（O(N²)→O(N·k)） | 93,99,100,101 |
| **Sparse-展开型** | CP9 | unfold 抽取块滑窗 + halo 邻域作为 KV，权重仍由 QK^T 产生 | 50 |

优化（flashattention.md **sparse trait**）：掩码型 → P123（skip + 边界 mask）+ P118；块跳过型 → **P137**（紧凑化走 Cube，top-k 不再回退 Vector）+ P138/P139（L0 守卫 / 多槽零填充稳定性）+ P76/P77。

### 压缩 Attention（KV 压缩表示）
| 小类 | CP | 机制定义 | 代表算子 |
|---|---|---|---|
| **MLA** | CP3+CP4+CP11 | K/V 低秩联合压缩进 latent 维再投影回原维（nope/rope 解耦），KV cache 降至 4%-14% | 31,67,75,76,77,78 |
| **Paged** | CP4 | KV cache 切定长物理块，block_table 页表映射，只解决内存分配不改变公式 | 30,32,69 |

优化（flashattention.md）：MLA 分支（P124 nope/rope 拼 K + P125 跨任务预取 + P126 任务表 + **P140** M-tile 放大）；Paged → paged/decode trait（P60/P61/P126）。

### 线性 Attention（无 QK^T、无 softmax，状态递推 + chunk-scan）
| 小类 | CP | 机制定义 | 代表算子 |
|---|---|---|---|
| **Linear** | CP5+CP6 | 低秩投影(15)/Nyström 分解(34)/核映射 `φ(Q)(φ(K)^T V)`(66)；**无 chunk 状态递推** | 15,34,66 |
| **KDA** | CP7 | 门控 delta rule `S←S·exp(g)+β(k⊗v)`，逐通道向量门控 + beta 误差修正 | 39,40,41,71,72,73,79,88,89,90,96 |
| **GDN** | CP7 | 门控 DeltaNet，同 delta rule 族，对标 fla/gdn2 + OlmoHybrid | 68,82,85,87,91,92,94,95 |
| **GLA** | CP7 | 门控线性注意力，**标量衰减**（每 head 一个 gamma） | 74,86 |
| **Mamba2** | CP7 | SSD 状态空间：`S_t=a_t·S_{t-1}+dt'_t·(x_t⊗B_t)`，dt 预处理链 + D 跳连 | 97,98 |

优化（ssm-scan.md）：P132 三阶段分解（GLA/Mamba2 用 cumprod 替代求逆）+ P133 块内 CV 流水 + P134 任务划分 + P135 求逆分治（仅 KDA/GDN）+ P136 bf16 量化。区分：GLA/Mamba2 标量衰减、KDA/GDN 逐通道门控 + 求逆。

### 变体（量化 / 去 softmax）
| 小类 | CP | 机制定义 | 代表算子 |
|---|---|---|---|
| **SageAttention** | CP10 | 量化注意力：QK 走 int8、PV 走 fp16，流式结构套量化/反量化 | 102 |
| **HSTU** | CP12 | `QK^T → SiLU → PV`，**不做行归一化**（去 softmax，排序场景） | 103 |

### 视觉注意力 CV-Attn（非序列 attention）
| 小类 | CP | 机制定义 | 代表算子 |
|---|---|---|---|
| **空间自注意力** | CP1 | (B,C,H,W) 展平 HW token 做标准 QK^T-softmax-PV，常与卷积分支组合 | 16,44,46,48,52,53 |
| **通道** | CP8 | 每通道一个标量权重再广播相乘，无逐对打分 | 38,56,60,62 |
| **空间门控** | CP8 | 每空间位置权重再广播，通道维共享 | 43,63 |
| **双分支** | CP8 | 通道 + 空间两条注意力分支并行/串行 + 融合门控 | 36,57,61,64 |
| **聚合分发** | CP8 | 先聚合全局/上下文描述子再分发回各位置 | 42,47,59 |
| **邻域展开** | CP9 | unfold 局部邻域 + linear 生成权重（非 QK^T）→ softmax → fold | 55 |

路由：**CV-Attn-通道/空间门控/双分支/聚合分发**（CP8，无 `[S,S]`）→ 归 reduction / elementwise；**空间自注意力**（CP1，真 QK^T）→ 归 attention 或按视觉语义处理。

### 非 Attention（排除）
| 小类 | CP | 机制定义 | 代表算子 |
|---|---|---|---|
| **Non-Attn** | CP13 | 名字带 Attention 或形似，实际不是（FFT 频谱/专家路由/线性投影加权） | 49,58,65 |

→ 标记为误归类，不按注意力算子处理。

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

