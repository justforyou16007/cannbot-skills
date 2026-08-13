---
id: X44
origin: discovered
discovered_round: 1
discovered_from: round_1/parallel_2 (x0, open_exploration)
base_speedup: 0.0013x → 4.149x geomean vs PyTorch（基线 geomean 81μs → 336μs ref；小 shape 2-3x、大 shape 7-9x）
confidence: single_op
status: active
op_families: [attention, matmul]
bottlenecks: [scalar_compute, launch_overhead, compute_bound]
tags: [cube-mmad, vector-simulated-matmul, set-hf32-false, fp32-cube, full-precision, attention, multi-matmul, matmul-highlevel-api]
strategy_ids: [X37]
---

# Strategy X37: fp32 Cube mmad 全量重写替代向量模拟 matmul（SetHF32(false) 全精度）

## 核心思路
基线内核用 AIV 向量标量模拟 matmul（16×16 tile 逐点 GetValue 累加）完全绕过 Cube 矩阵
单元，是数量级性能杀手。把算子内全部 matmul（attention 的 QK^T/PV/W_o）改用 Matmul
高阶 API（`ASCENDC_CUBE_ONLY` + `REGIST_MATMUL_OBJ` + `IterateAll`）直接走 fp32 Cube；
910B2C 上 `SetHF32(false)` 关闭 HF32 精度截断获得全精度（MERE ~1e-6）。

## 适用场景
- 任何 matmul 被实现为向量标量模拟（AIV GetValue 逐点累加）的算子
- attention 族多 matmul 流水（QK^T→softmax→PV→W_o），fp32 精度路径不再需 fp16
  hi/lo 拆分（HF32 拆分 3 次 gemm 被 SetHF32(false) 全精度单次 gemm 取代）
- 小/中 shape 为主、gemm 非瓶颈上限的算子：全精度比 HF32 慢但 shape 小可接受

## 实现要点
1. `SetHF32(false)` 全精度：HF32(true) 仅 10-bit 尾数不够 fp32 rtol=1.2e-4；关闭后
   MERE ~1e-6，50/50 PASS。
2. Matmul 高阶 API：`ASCENDC_CUBE_ONLY` + `REGIST_MATMUL_OBJ` + `IterateAll`，AIC 执行。
3. fp32 向量 op count 必须 ≥8 且为 8 的倍数；手写 halving 归约在 nPad 非 2 幂时越界，
   改用 WholeReduceMax/Sum。
4. uint8 位图 mask 向量操作必须 256-bit block 对齐（count 是 256 的倍数），否则 mask
   写入不完整 → softmax 输出均匀 1/32。
5. 小 seqLen 统一 sR16=max(64, ceil(S/16)*16)，防 mask 对齐问题。
6. exp 必须保留行 max 减法防下溢 → NaN。

## 来源
自动发现于第 1 轮进化（LongformerAttention），round_1/parallel_2（x0，open_exploration）。
7 次发射流水 lf_prep→lf_gemm_nt_fp32(QK^T)→lf_softmax→lf_gemm_nn_fp32(PV)→lf_merge→
lf_gemm_nt_fp32(W_o)→lf_bias，三个 matmul 全部 Cube 重写。geomean 0.0013x→4.149x
（基线 368ms → 最优 81μs），50/50 PASS，MERE ~1e-6。
