---
schema_version: okf.v1
kind: discovered_strategy
id: X37
title: "FP32-via-FP16 Hi/Lo 4-GEMM Cube Decomposition"
origin: discovered
confidence: single_op
status: active
op_families: ['flash_attention', 'matmul', 'conv']
bottlenecks: ['scalar_compute', 'compute_bound']
tags: ['fp32', 'fp16', 'cube', 'hi-lo', 'decomposition', 'precision']
discovered_round: 1
discovered_from: round_1/parallel_2
base_speedup: "0.47x -> 0.98x geomean (WindowAttention, fp32 geomean 0.1x->0.956x)"
---

# Strategy X37: FP32-via-FP16 Hi/Lo 4-GEMM（fp32 精度要求的 Cube 重构路径）

## 核心思路
Cube 的 Mmad 不直接消费 fp32 输入（910B2C fp32 cube MMA 实测 NaN）。X17/P128 用 bf16 hi/lo 3-GEMM（~1e-4 相对误差）解决 fp32 GEMM 上 Cube 的问题，但当应用要求 fp32 allclose（atol≈2e-5，如 torch 算子精度校验）时，bf16 3-GEMM 的 ~1e-4 误差不够。

本策略改用 **fp16 hi/lo 拆分 + 4 次 fp16 Mmad**：
```
x = hi + lo + delta，hi = fp16(x)，lo = fp16(x − fp32(hi))，|delta| ≤ 2^-24|x|
A@B ≈ Ah@Bh + Ah@Bl + Al@Bh + Al@Bl   （4 次 fp16 Mmad，fp32 L0C 累加）
```
fp16 有 11 位尾数（vs bf16 的 7 位），hi+lo 捕获 ~23 bit，逼近 fp32 的 24 bit → 最大绝对误差 ~9e-6，落在 fp32 allclose atol=2e-5 内。**4-GEMM 必须保留 lo@lo 项**（fp16 hi/lo 的 lo 分量贡献不可忽略，丢该项误差回到 ~1e-4 量级）。

## 适用场景
- fp32 输入的 GEMM 类计算（matmul / conv / attention 的 QK^T/PV/K1/K3 投影），Cube 仅支持 fp16/bf16，且精度要求是 fp32 allclose（atol ~2e-5）
- compute-domain，fp32 被困在 Vector/标量路径（标量 GetValue 内层循环）
- 容差允许 ~1e-5 相对误差；fp16 动态范围覆盖输入值域

## 实现要点
1. 独立 split 预 pass：纯 AIV kernel 把 fp32 拆成 fp16 hi/lo（Q 可折叠 scale）。UB 用深 ring（8-deep）规避 MTE3→V WAR 竞争
2. K1/K3 GEMM 与 K2 注意力均可套用（bulk-sync 或跨核 ping-pong）
3. 每 k-chunk 的 `cmatrixInitVal` 必须为 true（每个 chunk 是独立输出 tile，AIV 分别累加 O）
4. 4x cube FLOPs 是固有代价；小 shape 建议按 M*C 阈值路由回向量 FMA

## 与 X17/P128 的关系
| 维度 | X17/P128 (bf16 3-GEMM) | X24 (fp16 4-GEMM) |
|---|---|---|
| dtype | bf16（7 尾数位） | fp16（11 尾数位） |
| GEMM 数 | 3（无 lo·lo） | 4（含 lo·lo） |
| 相对误差 | ~1e-4 | ~9e-6 |
| 通过 fp32 allclose(atol=2e-5) | 否 | 是 |

## 来源
自动发现于第 1 轮进化，算子 WindowAttention（FA 家族），speedup 0.47x → 0.98x（fp32 geomean 0.1x → 0.956x，case48 20ms→0.59ms）。
