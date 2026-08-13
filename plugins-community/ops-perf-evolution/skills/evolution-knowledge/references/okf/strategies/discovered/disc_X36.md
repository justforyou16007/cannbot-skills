---
schema_version: okf.v1
kind: discovered_strategy
id: X36
title: "Zero-Host-Padding AIV Small-Matmul"
origin: discovered
confidence: single_op
status: active
op_families: ['matmul']
bottlenecks: ['undersize_transfer', 'fake_mte2_bound', 'scalar_loading']
tags: ['aiv', 'vector', 'host-padding', 'small-shape', 'zero-pad']
discovered_round: 1
discovered_from: round_1/parallel_3
base_speedup: "0.375x -> 0.61x geomean (BatchMatmul, 小 shape 0.15x->0.98x)"
---

# Strategy X36: Zero-Host-Padding AIV Small-Matmul（消除 host 零填充的 Vector 精确小 matmul）

## 核心思路
matmul 系算子在 host 侧常把 M/N/K 零填充到 Cube 对齐粒度（如 16），`constant_pad_nd` 会发射独立的 pad kernel（2 个），其时间被 msprof 计入算子总时长。对小 shape（如 [1,2,4]→[16,16,16]），pad kernel 占 17-30us（83%），完全主导官方指标——优化 kernel 本身（MatmulImpl 初始化、mmad 时序）对官方 geomean 无济于事。

本策略：
1. **host 不 pad，直接下发原始 (B,M,K)/(B,K,N)**，彻底消除 pad kernel；
2. **AIV 核内精确计算** C[m][n] = Σ_k A[m][k]·B[k][n]，全程 fp32 累加（fp16/bf16 输入先向量 Cast 到 fp32，输出 CAST_ROUND/CAST_RINT 回 T）；
3. **32B 对齐行展开**：B/A 在 UB 中按 32B 对齐 row stride 展开（DataCopyPad 逐行零填充），使 Muls/Add/Cast 全部满足对齐与 count 约束；
4. **多核切分**：host 按 batch × M 行切分（blockDim = batch × mChunks），摊薄 AIV launch 地板；
5. **紧凑 VectorTiling**（几个 int，不含 TCubeTiling 的 ~104 int），降低 LoadTilingFromGM 开销。

## 适用场景
- matmul 系算子的**非 16 对齐小 shape**（M,N,K ≤ 64，M*K ≤ 256）：[1,2,4]、[1,8,16]、[3,6,12] 等，官方评估将 host pad kernel 计入总时长时
- 纯 Cube / MatmulImpl 路径对 tiny shape 的 init/launch 地板 ≥ 4-8us，而 AIV 精确路径可到 3-8us
- 与 X13（双引擎分流）互补：X13 在 M 阈值路由，本策略强调消除 pad + AIV 精确计算

## 实现要点
1. **准入判据 M*K ≤ 256**：AIV 标量读 UB（GetValue ~30ns/次）是硬约束，超阈值则回退 Cube MatmulImpl
2. B/A 行展开时 `dstStride` 单位 = 32B DataBlock，`rightPadding` 单位 = 元素；对齐行（N*sizeof(T)%32==0）退化为单块 DataCopy
3. fp16→fp32 用 `CAST_ROUND`，bf32→bf16 用 `CAST_RINT`（fp32 输入跳过 cast）
4. A/B 先 Cast 到 fp32 缓冲，热循环只做 fp32 Muls/Add（规避 AIV 后端 bf16 标量→fp32 转的编译错误）
5. 全部 M 行结果先累积到 UB 缓冲（M*brow fp32），一次 barrier 后统一写回
6. 精度阈值需按 dtype 单独配（fp32 atol=1e-3；fp16 atol=9e-2；bf16 atol=1e-1）

## 来源
自动发现于第 1 轮进化，算子 BatchMatmul，speedup 0.375x → 0.61x（小 shape 0.15x→0.98x，[1,2,4] 最高 1.56x）。与 n1（SetFixSplit 多核 Cube）正交可组合。
