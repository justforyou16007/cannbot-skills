---
schema_version: okf.v1
kind: failure_pattern
key: "P128:attention:crosscore_hilo_workspace_l2_visibility"
phenomenon: precision_failure
severity: major
status: active
failed_combination: [P128]
operator_type: [attention, flash_attention]
occurrences: 1
created_at: '2026-08-13T10:00:00Z'
updated_at: '2026-08-13T10:00:00Z'
tags: [fp32, hilo, crosscore, workspace, l2-visibility, race, mmad, barrier, cube]
---

## Symptom

P128（fp32 经 bf16/fp16 hi/lo 拆分跑 Cube 3-GEMM）在多核 MIX_AIC_1_2 kernel 中，
由 AIV 拆分写 hi/lo workspace → AIC 读 workspace 做 3-GEMM 时，**同输入多次运行结果
非确定**：workspace 中某 task 的 kc=0 槽位有时含 kc=1 列的数据（列错位），输出时对时错。
CANN 不报错，属静默错误。

## Root cause class

跨核 GM workspace 可见性缺陷 —— AIV（MTE3 写）→ AIC（MTE2 读）的大体积 hi/lo
workspace（~32KB/task 一次 flag）在 SetFlag 之前无法全局可见（L2 缓存一致性不保证）。
不同于小体积 fp16/bf16 原生 cube 路径（用相同 CrossCore flag 机制却 17/17 通过），
差异在 P128 的 hi/lo 双份大 workspace 写入规模。

## Failed combination

- P128（fp32-via-bf16/fp16 hi/lo 3-GEMM）在多核 kernel 内 **AIV 拆分写 GM workspace →
  AIC 读** 的跨核交换形态
- 注意：同策略的「AIC 计算本身」精确（dump 重算逐位吻合），「AIV split 单点」正确；
  失败的是「跨核 workspace 交换」这一环节

## Resolution

- **host 预拆分（kernel 边界同步）**：把 fp32→hi/lo 拆分移到 host（torch cast），在
  kernel launch 边界完成。kernel launch = 全同步，AIC 读预拆分 fp16/bf16 面板与读原生
  数据完全等价，彻底规避跨核 L2 可见性缺陷。这是本次 WindowAttention 的落地解。
- 若必须 kernel 内跨核交换，改 **per-kt 小切片 + per-kt flag** 流水（fp16 路径 wsP 的
  已验证模式，~8KB/subcore/kt），避免一次性大 workspace 单 flag。
- 已证无效的修复（勿重试）：MTE3_V WAR 屏障、DCCI 逐行刷 L2、DSB、per-subcore 刷半区。

## Evidence

- 来源：14_WindowAttention 进化（round_4/parallel_0 = r4a，round_6/parallel_0 = r6a）
- r6a 证据链：AIC 3-GEMM fp64 重算逐位吻合（max diff 0.0）；AIV split 单点正确（hi 与
  torch bf16(x_pad) 逐元素一致）；wsA 同输入多次运行非确定（有时 kc=0 槽含 kc=1 列）
- r7a 落地：host 预拆分 fp16 hi/lo，FP32 17/17 全过（MERE 8e-6~6e-5），fp32 geomean
  0.272→0.866x
- 交叉验证：FP16/BF16 原生 cube 路径用相同 CrossCore flag 机制 17/17+16/16 通过

## Related

- X17 / P128（fp32-via-bf16 hi/lo 3-GEMM 算法本体）
- X44（SetHF32(false) 全精度 fp32 Cube，无需 hi/lo 拆分——更优方案，见 proven-solutions）
- x38（WindowAttention FP32-on-Cube host 预拆分落地）
- f4（bf16 hi/lo 对近零输出精度不足，转 fp16 hi/lo）
