---
id: X30
origin: discovered
discovered_round: 1
discovered_from: 22_MultiQueryAttention round_1/parallel_3 (x0, open_exploration)
base_speedup: 1.1267x vs PyTorch（基线 n1_s1 0.9648x，+17%）
confidence: single_op
status: active
op_families: [attention, flash_attention, matmul]
bottlenecks: [no_overlap, mte2_stall, scalar_compute]
tags: [attention, mqa, gqa, heads-merge, layout, token-major, head-minor, kernel-elimination]
strategy_ids: [X30]
---

# Strategy X30: 头维-minor Q 布局免 heads-merge kernel（token-major 折叠 head 合并）

## 核心思路
多头注意力的 heads-merge kernel（O[BH*S, hdPad] → Om[M, H*hdPad] 转置+cast）本质是
行序 (b,h,s) → (b,s,h) 的 shuffle，纯搬运。由于 QK^T → softmax → PV 整条注意力链对
Q 的行序不敏感（输出 Y=sum_h O[(b,h,s),d]*wot[h*hdPad+d,n] 与行序无关，逐位一致），
把 Q 的布局从「逐 head 连续 (b,h,s)」改为「token 连续 (b,s,h)（head-minor）」，
则 prep 的 Q-split 从逐 head strided gather 变成 token 粒度连续块拷贝（blockLen=H*hdPad，
srcGap 跳过 K/V 段，dst 连续），且 PV-GEMM 输出 O[M, H*hdPad] 天然就是合并后的 heads
布局——heads-merge kernel 整段删除（少 1 次 launch + O/Om GM 往返 + cast pass）。

## 适用场景
- 物化式多段 kernel 的 MHA/MQA/GQA（含独立 heads-merge 或 head-layout-shuffle kernel）
- Q/K/V 投影已融合或 Q 单独物化，且注意力链不依赖特定 Q 行序
- heads-merge 是纯搬运（无跨 head 归约）时收益最大；若 head 间需归约（如 group norm over heads）不适用

## 实现要点
1. Q4 行序改为 token-major：行索引 = b*S + s（每 token 内 H 个 head 连续），即 Q4[B*S*H, hdPad] 中行序 (b,s,h)。
2. prep 的 Q-split：从逐 head strided gather（srcGap 跳过非本 head 列）改为 token 粒度连续块拷贝（blockLen=H*hdPad，srcGap 跳过 K/V 段，dst 连续写）。
3. 注意力链（S-GEMM、softmax、PV-GEMM）不感知行序，直接复用；PV-GEMM 输出 O[M, H*hdPad] 即合并后布局，out-proj 直接消费。
4. 若 head 数为 1（MQA）或 hdPad 已对齐，blockLen=H*hdPad 天然 512B 对齐，连续拷贝高效。
5. 配合 PV-GEMM 输出直接 cast T（L0C fp32 累加后 cast 到 T 存储，与旧 merge CAST_ROUND 等价）可再省一个 cast pass。

## 来源
自动发现于第 1 轮进化（22_MultiQueryAttention, x0 open_exploration），speedup 1.1267x
（基线 0.9648x，+17%）。与 X4（Concat-Fused Strided DMA）思路互补：X4 消除 concat 的
strided DMA，X30 消除 head 合并的转置 kernel。
