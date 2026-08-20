---
schema_version: okf.v1
kind: discovered_strategy
id: X46
origin: discovered
title: 原生 fp32 Cube 单 pass 投影/attention 替代 fp16 hi/lo 3-mmad（对齐 PyTorch fp32 参考路径）
description: "融合 GQA prefill 的 fp32 投影 GEMM 原用 fp16 hi/lo 3-mmad（3 次 fp16 mmad/k-step + split GM 往返），PyTorch fp32 matmul 在 910B 默认即单 pass 全精度 fp32。改为原生 fp32 cube 单 pass（GqaMatmulKernel<float,float,false>, 1 mmad/k-step, 手动 Mmad 已全精度 MERE~1e-6 无需 SetHF32Mode(false)）后大 fp32 case +54-117%。关键：SetHF32Mode(false) 是 per-core 持久寄存器跨 kernel launch 破坏后续 MIX attention kernel，不可调用；手动 Mmad 已是全精度。"
confidence: single_op
status: active
op_families: [matmul, attention, flash_attention, gqa]
bottlenecks: [mte2_stall, scalar_compute, compute_bound]
tags: [fp32, native-fp32-cube, single-pass, set-hf32-false, projection-gemm, attention, gqa, prefill, gm-roundtrip]
strategy_ids: [X44, X37, X46]
discovered_round: 7
discovered_from: "21_GroupedQueryAttention evolution round 5-7 (sv1/sv2/sv3 原生 fp32 系列)"
base_speedup: "大 fp32 case +54-117% (c20 +54%, c35 +98%, c47 +117%, sv1); fp32 16-case +24-38%; 全量 clean 0.8054x (sv3), 合并内核估算 0.808x"
---

# X46: 原生 fp32 Cube 单 pass 替代 fp16 hi/lo 3-mmad

## 核心发现
PyTorch fp32 matmul 在 910B 默认 HF32 关闭即**单 pass 全精度 fp32**。融合 GQA 的 fp32 路径此前用 fp16 hi/lo 3-mmad（3 次 fp16 mmad/k-step + split 双平面 GM 往返）——反比参考多做 3 倍 mmad + split 数据通路，是大 fp32 case 慢 2-3 倍的根因。

改为**原生 fp32 cube 单 pass**（`GqaMatmulKernel<float,float,false>`, SPLIT3=false, IS_FP32=true, 1 mmad/k-step）后，大 fp32 投影 case +54-117%（c20 +54%, c35 +98%, c47 +117%），fp32 16-case geomean +24-38%。同时消除 gqa_split_fp16 xHi/xLo GM 往返与权重 hi/lo 双平面。

## 关键工程发现
1. **SetHF32Mode(false) 不可调用**：它是 per-core 标量控制寄存器，**跨 kernel launch 持久**，会破坏后续 MIX attention kernel（实测启用后特定 head 输出错乱）。**手动 Mmad 已是全精度**（MERE~1e-6 << 1.22e-4），无需 HF32 开关。
   - **补充（3_FusionAttention 实测，消解与 X44 的表面矛盾）**：危险的是**全局** `SetHF32Mode(false)`；**Matmul 对象级** `mmQK_.SetHF32(false, 0)`（X44/3_FusionAttention 用法）是 per-matmul 实例配置，随对象析构失效、不跨 launch 污染，**安全可用**。两条经验不冲突：手动 Mmad 路线不要碰全局开关，Matmul 高阶 API 路线用对象级开关。
2. **native attention 单 pass**（sv2）：QK^T 3-mmad→1-mmad, PV 2-mmad→1-mmad, 单 fp32 P 平面（ws2 槽减半），大 fp32 +30-80%。
3. **小 fp32 M=1 回归**：mNum==1 时原生单 mmad 无 M 并行 → 强制 by=1 得最大 N 列拆分（c2/c5/c8 修复）；极小 shape 路由回 fp16-hi/lo attention 路径避开 host K^T transpose ~20us launch。

## 适用场景
- fp32 输入的 GEMM（matmul/conv/attention 投影），当前用 fp16 hi/lo 3-mmad 且投影 MTE2-bound。
- 对齐 PyTorch/CANN 的 fp32 matmul 参考路径（torch_npu 默认 HF32 关闭）。
- 融合管线中 split GM 往返可消除的算子。

## 实现要点
- 复用 `GqaMatmulKernel` 模板的 SPLIT3=false + IS_FP32=true 分支（1 mmad/k-step）。
- 手动 Mmad 已是全精度，**不要**调 SetHF32Mode(false)。
- host 直喂 fp32 平面（x2d/attn2d），删除 gqa_split_fp16 与权重 hi/lo 双平面。
- M=1 小 shape 强制 by=1（N 列拆分）或路由回 fp16-hi/lo 路径。
- fp32 cube B-operand 必须 nZ L0B（zZ L1→nZ L0B LoadDataWithTranspose），nN→zN 是坏路径（Mmad 按错误格式解释）。

## 来源
21_GroupedQueryAttention 进化 round 5-7：
- sv1 (原生 fp32 投影, 大 fp32 +54-117%)：round_5/parallel_1
- sv2 (native fp32 attention QK^T/PV, 大 fp32 +30-80%)：round_6/parallel_1
- sv3 (原生 fp32 投影+全K B-stationary+M=1修复, clean 0.8054x)：round_7/parallel_2
- 合并 (sv3+sv2, 全量估算 0.808x)：round_7/parallel_0
