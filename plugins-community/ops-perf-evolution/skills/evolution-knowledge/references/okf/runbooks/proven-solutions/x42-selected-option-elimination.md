---
schema_version: okf.v1
kind: proven_solution
id: x42-selected-option-elimination
status: active
op_families: [attention, flash_attention, matmul, moe, conditional-compute]
strategy_ids: [X42]
tags: [attention, routing, candidate-layer, conditional-compute, one-hot, branch-elimination]
---
# x42: 路由选中分支消除（selected-option elimination）

**Source**: AdaptiveAttention evolution, round 1, parallel 2 (x0)
**Speedup**: 0.187x → 0.449x geomean（2.40x），大 shape 5-7.6x
**Operator type**: 路由/门控选择式多分支算子（AdaptiveAttention 多候选层 / MoE expert 路由）
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
router 在计算前产生 one-hot sel，输出 = sum_i sel_i·branch_i(x)。按 batch 分发把主计算
task 表只映射到被选中分支的权重/数据，跳过其余分支的完整计算链。AIC CUBE_ONLY 核无法
可靠读 GM sel（实测恒 0），host D2H 读回（B×n_branch float）算 option 索引作启动参数。

**Key code change**: 投影 GEMM 按分支裁剪（行块权重寻址）；attention 任务 B×3H→B×H；
Sbuf/Pbuf 物化减 3x；out 投影 K 维 3D→D。

**Applicability conditions**: 输出 = one-hot 选择×各分支结果；router 代价<<被跳分支代价；
小 shape D2H 同步可能吃收益。

**Strategy IDs**: X35
