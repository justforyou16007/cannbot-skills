---
id: X42
origin: discovered
discovered_round: 1
discovered_from: 33_AdaptiveAttention round_1/parallel_2 (x0, open_exploration)
base_speedup: 0.4494x vs PyTorch（基线 geomean 0.187x，2.40x；大 shape 5-7.6x）
confidence: single_op
status: active
op_families: [attention, flash_attention, matmul, moe, conditional-compute]
bottlenecks: [compute_bound, no_overlap]
tags: [attention, routing, candidate-layer, conditional-compute, one-hot, branch-elimination, task-dispatch]
strategy_ids: [X42]
---

# Strategy X42: 路由选中分支消除（one-hot selector 预计算 → 只算被选中分支）

## 核心思路
当算子在主计算**之前**已由一个廉价 router/selector 产生 **one-hot 选择向量**（如
AdaptiveAttention 的 router argmax→sel、MoE 的 expert router），而输出 = sum_i sel_i ·
branch_i(x)（one-hot 下即只取被选中分支），则**全部未选中分支的计算都是纯浪费**。
按 batch/token 分发，把主计算（投影 GEMM、attention、输出投影）的 task 表只映射到
被选中分支的权重/数据，跳过其余 n_branch-1 个分支的完整计算链。

## 适用场景
- 路由/门控选择式多分支算子：输出 = one-hot 选择 × 各分支计算结果（AdaptiveAttention
  多候选层、MoE expert 路由、条件计算）
- router 计算代价远小于被跳过分支的代价（本算子 router 是 B 个 GEMV，被跳的是
  n_opt-1 条完整 QKV+attention+out 链）
- 分支间权重布局可按分支切块寻址（行块或列块），task→(batch, branch) 映射可 host 拍平

## 实现要点
1. **host 读回 selector**：AIC (ASCENDC_CUBE_ONLY) 核内 DataCopy+GetValue 读 GM 的 sel
   实测恒返回 0（CUBE_ONLY 核无可靠 GM 标量读），需在 router kernel 后 host D2H 读回
   sel（仅 B×n_branch 个 float，量极小），计算每 batch 的 option 索引作为启动参数传入。
2. **按分支裁剪投影**：QKV/out 投影从全分支宽度（3*D_pad / 3*kv*hd_pad）裁剪为单分支
   宽度（D_pad / kv*hd_pad），每个 batch 通过 sel 选中自己的权重行块。若权重是
   [D_pad, 3*D_pad] 列块布局，需重排为 [3*D_pad, D_pad] 行块布局供按行块寻址。
3. **任务缩减**：attention 任务从 B×n_branch×H 减为 B×H，Sbuf/Pbuf 的 O(S²) 物化减
   n_branch 倍；out 投影 K 维从 3*D_pad 裁为 D_pad。
4. **压缩布局**：unpack/scatter 只搬移被选中分支的 Q/K/V 与输出。
5. **代价权衡**：小 shape 下 host D2H 同步（每次调用 ~1-5μs）可能吃掉收益，需评估
   threshold；大 shape 收益远大于同步开销。

## 收益
AdaptiveAttention 上 47/50 case geomean 0.187→0.449（2.40x），fp16/bf16/fp32 分别
2.38/2.32/2.50x；大 shape 5-7.6x（case35 0.007→0.050、case36 0.040→0.223）。n_opt=2/3
时理论上限 ~2-3x（跳过 50%/67% 分支）。

## 来源
自动发现于第 1 轮进化（33_AdaptiveAttention, x0 open_exploration），speedup 0.4494x
（基线 0.187x）。与 X33（group A-reuse）、X30（kernel elimination）正交可组合。


## 实现变体 B（round_2 精化，x0_x1）：AIV 侧选择免 host D2H
host D2H 读 sel 对小 shape 有 ~10-50μs 同步回退。改为**保留全宽投影**，把 option 选择搬到
AIV kernel：新增 AaUnpackSelKernel/AaScatterSelKernel（AIV 读 sel→argmax 可靠），从全宽
投影中抽取被选中 option 的 Q/K/V 并把 KV head 重复到 H，attention 任务仍 B*3H→B*H；
scatter 把被选中 option 的 PV 输出写入零初始化全宽 Ocat（未选中块贡献 0）。无 D2H、无
opt 启动参数。AdaptiveAttention 22/50 geomean 0.449→0.998（case1 1.47→2.67x）。局限：
全宽投影对 fp32 大 shape 是 3x 立方成本（可叠加 P128 hi/lo 3-pair 缓解）。
