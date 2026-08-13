---
id: X25
origin: discovered
discovered_round: 1
discovered_from: "8_ConvStandard3d evolution, round 1, parallel 2 (x0, open_exploration)"
base_speedup: 0.802x（保守 full-60 估计 vs PyTorch；39/60 实测 geomean 0.5516 vs 同用例基线 0.4057，+1.36x）
---

# Strategy X25: 向量权重广播（Brcb + MulAddDst）替代 S-pipe 标量读

## 核心思路
滑窗卷积/小 GEMM 的瓶颈常是每 (co,tap) 一次权重 `GetValue()`（S-pipe UB→标量寄存器读，~43cy，依赖链串行）。X25 用 **V-pipe `vbrcb`（Brcb）向量广播** 替代：权重在 GM 预展开为 `[Co,K,8]` 布局（每权重 8 份连续副本），每 (g,tap) 一次 `Brcb` 把 8 个 b32 广播成 64-lane 权重向量，驱动 `numBatches` 条 `MulAddDst`（acc += x*w）——**全程不经过 S-pipe，标量读彻底消失**，数值语义不变。权重跨 oh-batch 不变，batch 循环内移使一次广播覆盖一个权重全部输出位置的复用。

## 适用场景
- **算子类型**: 滑窗卷积 kw==1 / 小 GEMM / 任意「标量权重驱动向量 FMA」且权重跨 batch 复用的算子
- **瓶颈类型**: `scalar_loading` / `scalar_compute`（GetValue S-pipe 串行化为主瓶颈）
- **预期收益**: 多 oh-batch / 多输出位置复用（numBatches≥2）时最大——case28 0.05→0.30（5.9x）、case27 0.16→0.74（4.6x）、case47/56 2.4x；numBatches=1 高 K case 收益被 8x 权重载入抵消

## 实现要点
1. 权重 GM 布局 `[Co,K,8]`：model_new_ascendc.py 侧把 wt1d 展开为每权重 8 份连续副本（fp32 PAR=64 → 64/8 副本）
2. x 按 tap 整行加载（rowLen=numBatches*P*W），边界行先 Duplicate 清零再按合法子区间加载，compute 统一全宽
3. `wvec` 用 2×64-lane ping-pong 打破 Brcb→MulAddDst 写后读依赖链
4. ⚠️ 局部细化保守：unroll 4 + 4 槽 ping-pong 在部分 case 调度劣化 ~5x，需实测调优
5. 与 X23 互补：X23 保留 GetValue 但减少数量（hoist）；X25 用 V-pipe 广播彻底消除标量读

## 约束
- numBatches=1 高 K case（如 K=1152, 单 batch）无跨 batch 摊销，Brcb 的 2-op 代价 + 8x 权重 GM 载入抵消收益
- W%8==0 满足 32B 对齐门控
- 测量受并发 msprof 争用污染，geomean 未达蒸馏阈值，但机制新颖且 per-case 实证有效

## 来源
自动发现于 8_ConvStandard3d evolution round 1/parallel_2（x0, open_exploration），60/60 精度 PASS，多 batch case 5.9x 提升
