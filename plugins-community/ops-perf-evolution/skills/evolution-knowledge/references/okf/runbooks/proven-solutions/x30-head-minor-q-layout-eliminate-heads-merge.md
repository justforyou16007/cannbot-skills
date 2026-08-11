---
schema_version: okf.v1
kind: proven_solution
id: x30-head-minor-q-layout
status: active
op_families: [attention, flash_attention, matmul]
strategy_ids: [X30]
tags: [attention, mqa, gqa, heads-merge, layout, kernel-elimination]
---
# x30: 头维-minor Q 布局免 heads-merge kernel（token-major 折叠 head 合并）

**Source**: 22_MultiQueryAttention evolution, round 1, parallel 3 (x0)
**Speedup**: 0.9648x → 1.1267x vs PyTorch（+17%，50/50 精度 PASS）
**Operator type**: 物化式多段 MHA/MQA/GQA（含独立 heads-merge 转置 kernel）
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
heads-merge kernel（O[BH*S,hdPad]→Om[M,H*hdPad] 转置+cast）是纯搬运。因注意力链
QK^T→softmax→PV 对 Q 行序不敏感，Q 布局改 token-major（head-minor）：prep Q-split 从逐
head strided gather 变 token 粒度连续块拷贝（blockLen=H*hdPad, srcGap 跳过 K/V 段），
PV-GEMM 输出 O.view(M,H*hdPad) 天然即合并 heads 布局——heads-merge 整段删除（少 1 launch
+ O/Om GM 往返 + cast pass）。配合 PV 输出直接 cast T 再省 cast pass。

**Key code change**: Q4 行序 (b,h,s)→(b,s,h)；heads-merge kernel 删除，out-proj 直接消费 O.view(M,H*hdPad)。

**Applicability conditions**: 物化式 MHA/MQA/GQA；heads-merge 纯搬运且 head 间无归约；注意力链不依赖 Q 行序。
