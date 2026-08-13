---
schema_version: okf.v1
kind: failure_pattern
key: "P1+P5+P8:elementwise:tiny_shape_launch_dominated_no_gain"
phenomenon: performance_no_gain
severity: minor
status: active
failed_combination: [P1, P5, P8]
operator_type: [elementwise, broadcast]
occurrences: 1
created_at: '2026-08-12T00:30:00Z'
updated_at: '2026-08-12T00:30:00Z'
tags: [double-buffer, tiny-shape, launch-overhead, elementwise, no-gain, pipebarrier, hardevent]
---

## Symptom
对 **tiny-shape、launch 固定开销主导** 的 elementwise 算子应用 P1 双缓冲 + P5 细粒度 HardEvent 同步 + P8 UB 分区后，性能与基线**基本持平**（FusedKdaGate：双缓冲变体 geomean 0.4555x vs 基线 0.4609x，噪声范围内持平）。尽管实现完整（gBuf BUFFER_NUM=2、subRows 子 tile Prologue/Steady/Epilogue 三阶段流水、全部 PipeBarrier<PIPE_ALL> 替换为 MTE2_V/V_MTE3/MTE3_MTE2/V_S 细粒度事件、UB 按 subElems 分区）。

## Root cause class
**瓶颈不在流水重叠**。该算子 heads≤4、rows≤178、hk_pad≤128，每核单 task 数据 ≤32KB，kernel 总耗时（10-80us）几乎全部是 **launch/固定开销**（任务调度、tiling 加载、MTE 启动），双缓冲能优化的「CopyIn/Compute/CopyOut 重叠」部分 <1us。双缓冲还引入额外调度与事件同步开销，反而可能轻微劣化。判定依据：不同 shape 的 kernel 时间几乎相同、cube_util 极低、算术强度 <1 FLOP/byte。

## Failed combination
- P1（双缓冲）+ P5（细粒度同步）+ P8（UB 分区）应用于 tiny-shape launch 主导 elementwise
- 注意：同一策略组合对**大 shape 内存密集**算子仍是基本盘（P1 是 memory_bound 的首选），失败的是「应用对象」——tiny-shape 下流水重叠收益趋零

## Resolution
- **tiny-shape launch 主导算子**：不要优先投资双缓冲。先做结构重排（见 X33）——消除 S-pipe 标量读、精简向量链 pass、对齐广播、减少每 task 调度开销，把 kernel 时间压向 launch 地板；launch 地板之上再考虑任务类型/紧凑 tiling。
- 判定前置：先看 profiling 是否 `no_overlap` + 大 shape；若 kernel 时间与 shape 几乎无关 → launch 主导，P1 不是关键路径。
- 双缓冲引入的额外事件同步（每 task 多个 HardEvent 配对）在 tiny-shape 下是负收益，应避免。

## Evidence
- 来源：FusedKdaGate evolution round 1（EVO_DIR=output/FusedKdaGate_evo）
- n1（P1/P5/P8 双缓冲变体）：geomean 0.4555x vs PyTorch，50/50 精度 PASS，1 轮 Local Refinement（tasksPerCore==1 跳过冗余 MTE3_MTE2 同步）后仍持平
- 同轮 workflow_guided 变体（X33 技术：单 kernel 融合 + 结构重排）0.49x → 2.31x，证明结构重排才是关键路径
- 附带发现：非 64 倍数向量计数的非确定性错误（curRows<hk 的 partial slab cnt=8/16/48 时向量单元随机错）需 `cntAligned=ceil(cnt/64)*64` + Duplicate 尾部清零修复——基线恒 64 倍数故未暴露

## Related
- X33（Elementwise 工作流全流程优化）——本失败模式的正面替代方案
- X24（AIC_ONLY 任务类型压 launch 地板）——launch 主导算子的另一条正面路径
