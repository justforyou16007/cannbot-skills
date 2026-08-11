---
schema_version: okf.v1
kind: discovered_strategy
id: X30
origin: discovered
discovered_round: 1
discovered_from: "22_MultiQueryAttention evolution, round 1, parallel 3 (x0, open_exploration)"
base_speedup: 1.1267x vs PyTorch（基线 n1_s1 0.9648x，+17%）
confidence: single_op
status: active
op_families: [attention, flash_attention, matmul]
bottlenecks: [no_overlap, mte2_stall, scalar_compute]
tags: [attention, mqa, gqa, heads-merge, layout, token-major, head-minor, kernel-elimination]
strategy_ids: [X30]
---

# X30: 头维-minor Q 布局免 heads-merge kernel（token-major 折叠 head 合并）

**Source**: 22_MultiQueryAttention evolution，round 1, parallel 3（x0, open_exploration）
**Speedup**: 1.1267x vs PyTorch（基线 0.9648x，+17%），50/50 精度 PASS
**Operator type**: 物化式多段 MQA/MHA/GQA（含独立 heads-merge 转置 kernel）
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
多头注意力的 heads-merge kernel（O[BH*S, hdPad] → Om[M, H*hdPad] 转置+cast）本质是行序
(b,h,s) → (b,s,h) 的 shuffle，纯搬运。因 QK^T→softmax→PV 整条注意力链对 Q 行序不敏感
（Y=sum_h O[(b,h,s),d]*wot[h*hdPad+d,n] 与行序无关，逐位一致），把 Q 布局改为 token 连续
(b,s,h)（head-minor）：prep 的 Q-split 从逐 head strided gather 变成 token 粒度连续块拷贝
（blockLen=H*hdPad, srcGap 跳过 K/V 段, dst 连续），且 PV-GEMM 输出 O[M, H*hdPad] 天然即
合并后 heads 布局——heads-merge kernel 整段删除（少 1 次 launch + O/Om GM 往返 + cast pass）。

**Key code change**:
```cpp
// prep Q-split：行序从 (b,h,s) 逐 head strided 改为 (b,s,h) token 连续块拷贝
//   dst 行索引 = b*S + s（每 token 内 H 个 head 连续），blockLen = H*hdPad
DataCopyPad(dst[t], src[b*S*Hhd + s*Hhd], blockLen=H*hdPad, srcGap 跳过 K/V 段, dst 连续);
// heads-merge kernel 整段删除：PV-GEMM 输出 O.view(M, H*hdPad) 直接供 out-proj 消费
```

**Applicability conditions**:
- 物化式多段 MHA/MQA/GQA，含独立 heads-merge/head-layout-shuffle kernel 且为纯搬运
- Q/K/V 投影已融合或 Q 单独物化；注意力链不依赖特定 Q 行序
- head 间无归约（如 head 上做 group norm）时收益最大
- 配合 PV-GEMM 输出直接 cast T（L0C fp32 累加后 cast 存储）可再省一个 cast pass

**Strategy IDs**: X30（与 X4 Concat-Fused Strided DMA 思路互补）
