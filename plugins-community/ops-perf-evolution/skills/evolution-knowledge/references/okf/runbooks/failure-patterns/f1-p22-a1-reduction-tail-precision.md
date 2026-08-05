---
schema_version: okf.v1
kind: failure_pattern
key: "P22+A1:reduction:tail_precision"
phenomenon: precision_failure
severity: major
status: stub
failed_combination: [P22, A1]
operator_type: [reduction]
occurrences: 0
created_at: '2026-08-05T10:00:00Z'
updated_at: '2026-08-05T10:00:00Z'
tags: [precision, dtype, tail]
---

## Symptom
精度失败，ulp 误差集中在最后一个 tile（尾块未对齐 32B）。

## Root cause class
dtype/tiling 交互 —— A1（fp32 中间累加）与 P22 的尾块路径不兼容。

## Failed combination
- P22（向量化）
- A1（fp32 中间累加）

## Resolution
待验证：改用 A3 尾块单独标量路径，或避免 A1 与 P22 同时作用于尾块。

## Evidence
（待填充）

## Related
- A3: 尾块标量兜底
