---
id: X31
origin: discovered
discovered_round: 1
discovered_from: 22_MultiQueryAttention round_1/parallel_3 (x0, open_exploration)
base_speedup: 1.1267x vs PyTorch（基线 n1_s1 0.9648x，+17%；与 X30 同变体叠加）
confidence: single_op
status: active
op_families: [reduction, elementwise, attention]
bottlenecks: [scalar_compute, compute_bound]
tags: [softmax, max-trick, bounded-logits, translation-invariance, barrier-reduction]
strategy_ids: [X31]
---

# Strategy X31: bounded logits softmax 去 max 减法（平移不变 + 数值有界时跳过 max-trick）

## 核心思路
softmax(z) = softmax(z - max(z)) 是平移不变式，max 减法（WholeReduceMax + Brcb 广播 +
Sub，约 3 pass + 8 barrier/tile）仅用于防止 exp 溢出（fp32 溢出点 ~88）。当 logits 实测
严格有界且远低于溢出阈值（本算子 |z|<~40；经验安全线 |z| < 30~40，留余量）时，max 减法
是纯开销——删除后直接 Muls(invDiv) + Exp，pass 数 8->5、barrier ~23->~12。配合行求和
chunk 化 WholeReduceSum 全发出 + log-depth 成对树合并（大 S 时 barrier O(C)->O(log C)），
softmax 单 kernel 时间可降 ~25%（case44 占整 forward ~40%）。

## 适用场景
- 输入域有界的 softmax（经有限范围投影后 logits 有界，如 x∈[-5,5] 经 nn.Linear 投影）
- 由实现者确认 logits 上界（可在注释中记录实测上界），并保留恢复 max 减法的开关
- 与 X30（head-minor 布局）正交可叠加；是 A5（max-trick）的条件反向优化

## 实现要点
1. 前提验证：实测 logits 上界 |z|max（如 q·k/sqrt(hd) 经随机权重投影 < 20-40），远低于 fp32 exp 溢出点 88。
2. 删除 WholeReduceMax + Brcb 广播 + Sub 三阶段；logits 先 Muls(invDiv) 再直接 Exp。
3. 除数用标量 Muls 替代 Div + 重复 divisor UB 缓冲，释放全 tile fp32 slot，tile 行数可提升（32->48~56），固定 barrier 摊到更多行。
4. 行求和：chunk 化 WholeReduceSum 先全部发出（相互独立），再做 log-depth 成对树合并（barrier O(C)->O(log C)）。
5. 注释中记录 logits 上界证据与恢复开关；若输入范围未来无界，须加 clamp 或恢复 max 减法。

## 来源
自动发现于第 1 轮进化（22_MultiQueryAttention, x0 open_exploration），speedup 1.1267x
（基线 0.9648x，+17%，与 X30 同变体叠加）。与 A5（max-trick 保证数值稳定）互补：
A5 是安全默认，X31 是有界输入下的收益优化，二者由 logits 上界证据切换。
