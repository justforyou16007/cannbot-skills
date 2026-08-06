---
schema_version: okf.v1
kind: failure_pattern
key: "multi-kernel:cube_registration"
phenomenon: build_failure
severity: blocking
status: active
failed_combination: [P102, P128]
operator_type: [conv, matmul, cv_fusion]
occurrences: 1
created_at: '2026-08-06T19:30:00Z'
updated_at: '2026-08-06T19:30:00Z'
tags: [multi-kernel, kernel-registration, cube, mix, environment]
---

## Symptom
910B2C 上做多 kernel Cube 流水线（im2col AIV + wrepack AIV + GEMM AIC + epilogue AIV，5 个 `extern "C" __global__` kernel 写在一个 `op_kernel/*.cpp`）时，**单文件多 kernel 只注册一个 binary**——所有 kernel（包括已验证可用的原始 vector kernel）输出全零，无编译错误、无运行时报错（静默失败 rc=0）。

## Root cause class
toolchain/build —— 该 lingxi-evo 构建环境下 `op_kernel/*.cpp` 内超过一个 `extern "C" __global__` kernel 时，host stub / kernel 注册只保留最后一个（或第一个），其余 kernel 的 device 二进制未注册。

## Failed combination
- P102（MIX CV 融合重写，AIC 承担 mmad）
- P128（FP32-via-BF16 hi/lo 3-GEMM）
- 伴随阻塞（同一环境，独立于注册问题）：
  1. 拆成单文件单 kernel 后，AIV im2col kernel 仍输出全零（对 byte-identical 的 Duplicate+DataCopyPad 原语），根因未定
  2. 跨 kernel AIV→AIC L2 一致性需 host 侧 D2H aclrtMemcpy flush（RunOpApi 不排序 AIV/AIC 队列）
  3. `KERNEL_TYPE_AIV_ONLY` 在 910B2C 挂死设备（NPU wedge, R6）
  4. `__vector__` kernel 被 host-stub generator 静默跳过（不生成 aclrtlaunch header）

## Resolution
- 单文件**只放一个** `extern "C" __global__` kernel；多 kernel 必须拆到独立 `*.cpp` 文件 + 共享头
- 在重新尝试 P102/P128 多 kernel Cube 流水线前，先最小探针验证：单文件双 kernel 是否注册失败（写两个 trivial kernel 看第二个是否输出）
- AIV→AIC 数据交接需要 host 侧 D2H flush 或全局 barrier（`SyncAll<false>`）保证 L2 可见性
- 避免 `KERNEL_TYPE_AIV_ONLY`（改用 MIX 默认）；`__vector__` 内核在此构建环境不可用
- 建议先修复 kernel 注册问题（可能是构建/toolchain bug），再重试 Cube 路径

## Evidence
8_ConvStandard3d evolution, round 1, parallel 3 (n7 workflow 分支，C-2/C-6 → P128/P102 Cube GEMM)。Cube 流水线完整实现但未达正确性，回退恢复原始 vector kernel（60/60 精度 PASS，speedup=0.565x ≈ baseline）。实现笔记记录了逐条实证验证过程。

## Related
- X21（910B2C Cube 化平台纪律：AIC_ONLY 显式声明、HardEvent 链、禁 TransDataTo5HD、one-hot 探针）
- P102（MIX CV 融合重写 SOP：Phase 0 纯 AIC 单路径 derisk）
- P128（FP32-via-BF16 hi/lo 3-GEMM）
