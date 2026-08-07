---
schema_version: okf.v1
kind: failure_pattern
key: "manual-mmad:loaddata3d-16x16-only"
phenomenon: precision_failure
severity: blocking
status: active
failed_combination: [P87]
operator_type: [matmul]
occurrences: 2
created_at: '2026-08-07T16:30:00Z'
updated_at: '2026-08-07T16:30:00Z'
tags: [manual-mmad, loaddata3d, fractal-loading, transpose, l0a, l0b]
---

## Symptom
手动 mmad kernel 用 `LoadData3D`（L1→L0A/L0B 的 tile 加载辅助函数）做分形搬运时，**只对 16x16x16 identity 输出正确**；对真实 shape（32x128、32x48、13x29 等）输出错误。无编译错误、无报错，静默精度失败。为保精度被迫 `useManual=0` 全退回 MatmulImpl fallback，手动 kernel 从未真正 dispatch（5_MatmulTransB 上 P87 两次尝试均因此失败）。

## Root cause class
API 误用/实现缺陷——`LoadData3D` 的参数模型只对单个 16×16 cube tile（identity）正确，**无法泛化到多 16-tile、非 16 对齐的任意 M/N/K 分形布局**（L0A 的 Nz、L0B 的 Zn）。这是「手动 mmad 分形加载」的经典坑，不是硬件限制。

## Failed combination
- P87（手动 mmad 流水线：手写 L0A/L0B/L0C 搬运 + mmad + fixpipe）两次尝试均倒在分形加载：
  1. 5_MatmulTransB round_2/parallel_2（x0_x1）——「L1→L0A/L0B LoadData3D 参数无法泛化到多 16-tile，真实 shape 输出错误」
  2. 5_MatmulTransB round_3/parallel_0（n1_cont_cont）——同因，host 强制 useManual=0 走 fallback
- 伴随限制（同一算子，独立于本 pattern）：
  1. 910B 设备转置原语（LoadData3D enTranspose / LoadData2D ifTranspose / LoadDataWithTranspose）不产生干净转置
  2. AIV `vtranspose` >16×16 挂起（NPU wedge）

## Resolution
- **弃用 LoadData3D 做分形加载**；改用 `TQue<TPosition::A1/B1>` + `CopyIn`，或按官方 manual mmad/gemm 模板做 ND→Nz（L0A）/ ND→Zn（L0B）转换，对任意 M/N/K 正确。
- **transB 类算子**：设备转置不可用 → host 侧把 B 预转置成 Bt[K,N] 入 workspace，kernel 走无转置的 L0B 加载。
- 先写「16x16 identity → 真实 shape」的最小探针验证分形加载正确性，再展开全 shape；不要直接对全部 50 shape 上手动路径。
- 参照 X24 proven_solution（已完整跑通任意 shape 的正确实现形态）。

## Evidence
5_MatmulTransB evolution：round_2/parallel_2（x0_x1，P87 首败）、round_3/parallel_0（n1_cont_cont，P87 再败）。实现笔记原文：「实测 manual kernel 仅 16x16x16 identity 正确；对 32x128/32x48 等真实 shape 输出错误（L1→L0A/L0B 的 LoadData3D 参数无法泛化到多 16-tile）」。后续手动 mmad 攻坚轮用 TQue CopyIn 方案 50/50 全 shape 跑通（见 x24 proven_solution），实证本坑可绕。

## Related
- X24（手动 mmad 打破 MatmulImpl 固定开销地板，正确分形加载形态）
- X24/AIC_ONLY（设备地板压降）
- 910B 平台纪律（设备转置原语不可用、vtranspose>16x16 挂起）
