---
schema_version: okf.v1
kind: discovered_strategy
id: X50
origin: discovered
discovered_round: 1
discovered_from: 78_TrtllmBatchDecodeSparseMlaDsv4 evolution, round_1/parallel_2 (x0, open_exploration)
base_speedup: 0.87x → 5.58x geomean vs PyTorch（mean 6.09x，total 7.55x；min 2.62x / max 12.90x，45/50 valid，50/50 精度 PASS）
confidence: single_op
status: active
op_families: [attention, flash_attention, mla]
bottlenecks: [launch_overhead, no_overlap, tiling_imbalance, mte2_stall]
tags: [kernel-fusion, decode, head-block-parallel, kv-tile-reuse, ub-residency, mla, sparse-attention]
strategy_ids: [P54, P57, P58]
---
# X50: 融合 decode MLA kernel — 5→2 kernel + (j,head-block) 并行 + KV tile 跨 head 复用

## 核心思路
decode 稀疏 MLA（QK^T→softmax→PV→epilogue 链）基线为 5 个串行 AIV kernel，
每个 kernel 把中间结果（wsScores/wsProbs/wsOut）物化到 GM，且并行粒度仅 j
（decode 单 token sumQ=1 时只有 1/48 AIV 核工作）。三重组合拳：

1. **kernel 融合**：qk_vec+softmax+pv_vec+epilogue 4 个 kernel 合成 1 个
   `sparse_mla_fused.cpp`。scores/probs/out 全部驻留 UB（head 块 nH×capPad 与
   512 维输出均远小于 192KB），消除 3 个 GM workspace 往返 + 3 次 launch。
2. **(j, head-block) 扁平并行**：每工作项处理 (j, hb) 一个 head 块，核数自适应
   （sumQ>=48 按 head 分块，否则 numHB=round(48/sumQ) 上限 hPad），decode 单 token
   高 H 场景从 1/48 核 → 吃满 48 AIV 核。
3. **KV tile 跨 head 复用**：循环序反转为 `[kv_tile][head]` —— 每个 16 行 kv tile
   只从 wsKv 读一次，在 UB 内被该 head 块所有 head 复用，消除 H 倍冗余 GM 读
   （单 token 高 H 场景 QK+PV 各搬 wsKv H 遍是 memory-bound 主因）。

数值语义与 reference 完全一致（两遍法 softmax：scores 存 UB 原地转 probs，
probs=exp(s-max)/(sum+exp(sink-max))；无效列 wsKv 零填充 + validF32 在 exp 后乘 0；
max 为常数平移，softmax 平移不变性保证全无效行输出 0）。

## 适用场景
- decode 稀疏注意力（MLA/MQA/GQA，q_len=1，head_dim 大如 512，head 数 H 大如 64）
- 多 kernel 串行物化中间结果的注意力链（GM 往返成为 launch-latency/mte2 主成本）
- 单 token 时核利用率极低（按 j 并行只能上 1 核）的 decode 形态

## 实现要点
- UB 预算：head 块 nH≤8 × capPad≤384 fp32 scores + probs + out 行，须 ≤192KB，
  超出则缩 head 块大小（numHB 自适应）。
- sink 路径：DataCopy(attnSinkGm_[hStart], nH) 中 nH 非 16 倍数违反 MTE2 32B 对齐，
  必须整行 [hPad] 对齐搬运 + 标量 gather 到连续缓冲再向量 Sub/Exp/Add。
- 数值稳定：无效列必须 -1e30 mask 注入（仅靠 wsKv 零填充会让某 head 全零概率在
  fp32 exp 下溢时出现除 0/NaN），exp 前保留行 max 减法。
- V→S 时序：softmax 行归约结果被标量读取时需 PipeBarrier<PIPE_ALL> 或 TQue 事件同步。

## 来源
自动发现于第 1 轮进化（78_TrtllmBatchDecodeSparseMlaDsv4，open_exploration 从第一性
原理推理）。5 kernel 串行 AIV 向量 GEMM 基线（geomean 0.87x，单 token case 0.14-0.5x）
→ 2 kernel 融合实现（geomean 5.58x）。最慢单 token case 27 (H=64) 0.14x→2.81x
（相对基线 18.6x 更快），case 38 (sumQ=32) 9.16x→12.04x。与 X47（布局直通）互补。
