---
schema_version: okf.v1
kind: failure_pattern
key: "P128:matmul:projection_near_zero_precision"
phenomenon: precision_failure
severity: major
status: active
failed_combination: [P128]
operator_type: [attention, flash_attention, matmul]
occurrences: 2
created_at: '2026-08-11T14:00:00Z'
updated_at: '2026-08-13T10:00:00Z'
tags: [precision, bf16, hilo, fp32, projection, near-zero, mer_e, 3-gemm, 4-gemm]
---

## Symptom
P128（fp32 经 bf16 hi/lo 拆分跑原生 bf16 cube）应用到 **投影 GEMM（out_proj）路径**时精度失败：近零输出 case 的 MERE 超过 2^-13（1.22e-4）容差。具体地，bf16 hi/lo **3-GEMM（丢弃 lo@lo）**在 fp32 判据下 16 个 case 有 8 个超限（MERE 1.3e-4~2.3e-4）；**4-GEMM（保留 lo@lo）**后仍有 3 个大 Dm 投影 case 边缘超限（MERE 1.23e-4/1.42e-4/1.52e-4）。而同样的 bf16 hi/lo 3-GEMM 应用到 **attention 路径（QK^T/PV）**则通过容差（fp32 geomean 0.384→0.514）。

## Root cause class
dtype 精度路由 —— bf16 hi/lo 拆分的总精度 ~2^-16（bf16 8-bit mantissa × hi/lo 两项），不足以覆盖**近零输出**路径的相对误差要求。attention 路径的 softmax 权重有界（0~1，logits 经 max-trick 后幅度受限），近零项少；而**投影 GEMM 输出可接近零**（如 GQA out_proj 的某些输出列），近零值上被丢弃的 lo@lo 项的相对误差被放大，MERE 超 2^-13。

## Failed combination
- P128（fp32-via-bf16 hi/lo 3-GEMM）应用于**投影 GEMM**（近零输出路径）
- 注意：同策略应用于 **attention 路径**不失败——失败的是「应用对象」而非策略本身

## Resolution
- **投影 GEMM 不适用 bf16 hi/lo**：保留 3xTF32（fp32 plane）或改用 **fp16 hi/lo**（~2^-21，比 bf16 hi/lo 高 8 倍余量）——投影近零输出需要更高精度。
- 若坚持 bf16 hi/lo：**必须保留 lo@lo 做 4-GEMM**，且对投影路径仍需预检近零输出占比。
- 通用判据：hi/lo 拆分的总精度须满足目标判据的 ~4 倍余量（Playbook 的"~1e-4 贴容差边缘"假设在实践中不成立——3-GEMM 8/16 case 超限即是证明）。

## Evidence
- 来源1：21_GroupedQueryAttention 第二轮进化（EVO_DIR=output/21_GroupedQueryAttention_evo/GroupedQueryAttention_evo_20260810_212700）
- n1（P128 bf16 hi/lo **attention**）通过：50/50 精度，fp32 geomean 0.384→0.514，case44 0.285→1.693x
- R3 n1_cont_s1（投影 split 融合）fp32 段 geomean 0.397，投影 GEMM 仍 3xTF32（case44 占 kernel 时间 55%）
- 3-GEMM vs 4-GEMM：丢 lo@lo 8/16 超限 → 保留 lo@lo 仅 3 个大 Dm 边缘超限（impl_note，round_1/parallel_1）

- 来源2：14_WindowAttention 进化（round_7/parallel_1 = r7a，2026-08）
- r7a 实测：bf16 hi/lo 3-GEMM 在 fp32 容差（rel_thr=2^-13=1.22e-4）下 MERE~2e-4，
  17 个 fp32 case 中 11 个失败；改 **fp16 hi/lo**（10-bit 尾数，~2^-20 精度）后 MERE~6e-6，
  17/17 全过。再次印证「bf16 hi/lo 精度不足，转 fp16 hi/lo」为通用判据。

## Related
- X28（小 S prefill FA 自适应 KV chunk）——同算子族同 dtype 问题的正面对策
- a3/algorithm_insights/attention.md（FA 精度路径设计）
