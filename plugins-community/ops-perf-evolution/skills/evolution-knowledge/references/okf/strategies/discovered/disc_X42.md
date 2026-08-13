---
schema_version: okf.v1
kind: discovered_strategy
id: X42
origin: discovered
discovered_round: 1
discovered_from: "33_AdaptiveAttention evolution, round 1, parallel 2 (x0, open_exploration)"
base_speedup: "0.4494x vs PyTorch（基线 geomean 0.187x，2.40x；大 shape 5-7.6x）"
confidence: single_op
status: active
op_families: [attention, flash_attention, matmul, moe, conditional-compute]
bottlenecks: [compute_bound, no_overlap]
tags: [attention, routing, candidate-layer, conditional-compute, one-hot, branch-elimination, task-dispatch]
strategy_ids: [X35]
---

# X35: 路由选中分支消除（one-hot selector 预计算 → 只算被选中分支）

**Source**: 33_AdaptiveAttention evolution，round 1, parallel 2（x0, open_exploration）
**Speedup**: 0.4494x vs PyTorch（基线 geomean 0.187x，2.40x），47/50 case 精度 PASS
**Operator type**: 路由/门控选择式多分支算子（AdaptiveAttention 多候选层注意力 / MoE expert 路由 / 条件计算）
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
当算子在主计算之前已由一个廉价 router 产生 one-hot 选择向量（sel），且输出 =
sum_i sel_i·branch_i(x)（one-hot 下即只取被选中分支），则全部未选中分支的计算都是
纯浪费。按 batch 分发，把主计算（QKV 投影、attention、out 投影）的 task 表只映射到
被选中分支的权重/数据，跳过其余 n_branch-1 个分支的完整计算链。本算子：attention
任务 B×3H→B×H、Sbuf/Pbuf O(S²) 物化减 3x、out 投影 K 维 3D→D。

**Key code change**:
```cpp
// host 在 router 后 D2H 读回 sel（AIC CUBE_ONLY 核内 DataCopy+GetValue 恒返回 0），
// 计算每 batch 的 option 索引，作为 int64 启动参数 opt0..opt3 传入 cube 核
int64_t opt[B];  // host 从 sel[b] argmax 计算
// 投影 GEMM 按分支裁剪：C[task=(b,i,h)] = x[b] @ W[(i,sel[b],h)]^T，只取被选中分支权重行块
// 权重重排为 [3*D_pad, D_pad] 行块布局（原 [D_pad, 3*D_pad] 列块布局按行块寻址会 stride-3 错误）
// attention 任务从 B*3H 减为 B*H，Sbuf/Pbuf 物化减 n_branch 倍
```

**Applicability conditions**:
- 路由/门控选择式多分支算子：输出 = one-hot 选择 × 各分支结果
- router 代价 << 被跳过分支的代价；分支间权重可按分支切块寻址
- AIC CUBE_ONLY 核无法可靠读 GM 标量 → host D2H 读回 selector（量极小）
- 小 shape 下 D2H 同步开销可能吃掉收益（需评估 threshold），大 shape 收益远大于开销

**Strategy IDs**: X35（与 X30 kernel elimination、X33 group A-reuse 正交可组合）


## 实现变体 B（round_2 精化，x0_x1）：AIV 侧选择免 host D2H
host D2H 读 sel 对小 shape 有 ~10-50μs 同步回退。改为**保留全宽投影**，把 option 选择搬到
AIV kernel：新增 AaUnpackSelKernel/AaScatterSelKernel（AIV 读 sel→argmax 可靠），从全宽
投影中抽取被选中 option 的 Q/K/V 并把 KV head 重复到 H，attention 任务仍 B*3H→B*H；
scatter 把被选中 option 的 PV 输出写入零初始化全宽 Ocat（未选中块贡献 0）。无 D2H、无
opt 启动参数。AdaptiveAttention 22/50 geomean 0.449→0.998（case1 1.47→2.67x）。局限：
全宽投影对 fp32 大 shape 是 3x 立方成本（可叠加 P128 hi/lo 3-pair 缓解）。
