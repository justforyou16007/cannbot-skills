---
schema_version: okf.v1
kind: discovered_strategy
id: X47
origin: discovered
discovered_round: 10
discovered_from: "34_NystromAttention evolution rounds 8-10 (n21/n22/n25/n26)"
base_speedup: "geomean 0.487→1.0254x vs PyTorch（round8-10 累计，相对基线 0.054 达 19x）；case9 0.142→1.005、case8 0.251→0.691"
confidence: single_op
status: active
op_families: [attention, matmul, flash_attention]
bottlenecks: [scalar_compute, compute_bound, launch_overhead]
tags: [wqkvT-elimination, dense-per-head, no-prep, set-hf32-false, core-softmax-block-vectorize, launch-merge, attention-core]
strategy_ids: [X47]
---

# X47: 融合 attention 的结构消除 + core_softmax 块向量化 + launch 合并

## 核心发现（round 8-10）
在 NystromAttention 达到 1x 的三个正交结构杠杆（可跨算子复用）：

1. **消除从未被消费的中间 GEMM**（n21）：投影同时产出 feature-major `wqkvT` 与 row-major `wqkv`，
   但 cube-core attention 只读 `wqkv`——`wqkvT` 的投影 GEMM 从未被消费，消除后投影 Cube 工作量减半。
   **先查中间产物是否真被下游读取**，未被读的就是纯浪费。

2. **稠密 per-head 投影 + no-prep 路由**（n21）：Q/K/V 直接写稠密 `wqkvd[3,BH,S,DHP]`
   （每 head 一个 64-col N-tile），消除 ~40MB re-pack pass；proj 直读 `x`（transA=false）、
   outproj 直读 `Wo`（transB=true），整个 prep 的 `X^T`/`Wo^T` 转置 pass 跳过。

3. **fp32 投影用 SetHF32(false) 单 GEMM 替代 hi/lo 3-GEMM**（n20/n22）：910B2C 上 SetHF32(false)
   全精度 fp32 Cube MERE~1e-6，单 GEMM 的 Cube 工作量约 hi/lo 的 1.5x 少（省 6 个 split_hilo AIV pass），
   case9 proj/outproj 1.8x、case2 4.1x。

4. **core_softmax 块式向量化**（n26）：原 P1/P3 每行 1 次 DataCopyPad + 逐元素 SetValue 掩码 +
   chunked WholeReduce，case9 每 task 383 行 × 35k 次 per-row PIPE_ALL GM barrier。改为 ROW BLOCK
   处理（P1 整块连续 DataCopy、P3 per-row DataCopyPad 批处理共享单次 barrier）、maskRow 向量 Add
   替代逐行 SetValue、长行用 Level-2 ReduceMax/Sum 单次 GetValue——case9 core_softmax 268→104us。

5. **launch 合并**（n23/n25）：round_od 消除（core epilogue 直写 dtype）+ split_hilo_multi
   （4 组 split 合并 1 launch），launch 数减 30-40%，小 shape 显著收益。

## 适用场景
- 融合 attention 投影（QKV/out）与 attention core 的算子
- 投影同时产出 feature-major 与 row-major 中间结果的（检查消费方）
- 有大量 per-row barrier / 标量掩码的 softmax kernel
- fp32 投影用 hi/lo 3-GEMM 的（改 SetHF32(false) 单 GEMM）

## 实现要点
1. 审计所有中间 GM tensor 的消费方，消除未被读取的（wqkvT 模式）。
2. 投影直写稠密 per-head 布局，跳过 re-pack 与 prep 转置 pass。
3. SetHF32(false) 单 fp32 GEMM（MatmulImpl），精度 MERE~1e-6。
4. softmax 按行块处理，消除 per-row PIPE_ALL barrier；长行用 Level-2 归约。
5. launch 合并：dtype 转换折入 core epilogue、多 split 合并为 multi-slot。

## 来源
34_NystromAttention 进化 round 8-10：
- n21 (wqkvT 消除 + dense per-head + no-prep)：round_8/parallel_2
- n20/n22 (fp32 SetHF32 单 GEMM)：round_8/parallel_1, round_9/parallel_0
- n25 (launch 合并)：round_10/parallel_0
- n26 (core_softmax 块向量化)：round_10/parallel_1
