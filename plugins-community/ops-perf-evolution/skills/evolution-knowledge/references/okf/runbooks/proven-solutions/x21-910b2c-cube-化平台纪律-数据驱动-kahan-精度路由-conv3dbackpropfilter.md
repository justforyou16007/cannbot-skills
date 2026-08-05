---
schema_version: okf.v1
kind: proven_solution
title: "X21: 910B2C Cube 化平台纪律 + 数据驱动 Kahan 精度路由 (Conv3DBackpropFilter)"
source: "conv3dbf_cb evolution, round 2, parallel 1 (x0_fix1) + parallel 0 (n2_s1 交叉实证)"
speedup: "1/20 failed → 20/20, avg_speedup 0.1085x（AIV 向量化天花板 0.00235x 的 46 倍）"
operator_type: "conv (backward-filter, cube 化)"
hardware: "Ascend910B2C (DAV_2201)"
confidence: single_op
status: active
op_families: [cv_fusion, matmul]
strategy_ids: X21（关联 X18 k 序探针纪律的 3D 推广）
---

### X21: 910B2C Cube 化平台纪律 + 数据驱动 Kahan 精度路由 (Conv3DBackpropFilter)

**Source**: conv3dbf_cb evolution, round 2, parallel 1 (x0_fix1) + parallel 0 (n2_s1 交叉实证)
**Speedup**: 1/20 failed → 20/20, avg_speedup 0.1085x（AIV 向量化天花板 0.00235x 的 46 倍）
**Operator type**: conv (backward-filter, cube 化)
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
910B2C 上把 AIV 算子 Cube 化时，正确性杀手是四条平台纪律而非算法：(1) 无 `__vector__` 的
`__global__ __aicore__` 内核被编译为 MIX 双入口，AIC_ONLY 必须显式
`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)`；(2) `PipeBarrier<PIPE_MTE2>` 不建立
MTE2→V 顺序（且 MTE2_S 事件不存在），跨 pipe 同步必须 SetFlag/WaitFlag HardEvent 链；
(3) TransDataTo5HD 在该平台静默杀死内核（rc=0 无日志），须改标量 UB gather + 精确 cast
round-trip；(4) vendor fractal N 序与文档假设可能不同（本例实测 (kd,ci1,kh,kw,ci0)，
kd 在 ci1 外），必须 one-hot/ramp 探针实测。对 fp32 L0C 累加必然超差的相消重元素 case，
用 AIV 预扫描（max|x|·max|g|·S > 5e10 / 含 inf-NaN / fp16+stride>1）数据驱动分流到
完整 Kahan 兜底（带 TwoSum 补偿，非 plain-fp32 重写）。

**Key code change**:
```cpp
// Before: MIX 双入口 + PipeBarrier 同步 + TransDataTo5HD + 假设 fractal 序 + 全量 fp32 L0C
__global__ __aicore__ void conv_dw_cube_kernel(...) {
    ...
    PipeBarrier<PIPE_MTE2>();              // 不建立 MTE2→V 顺序，transdata 恒输出 0
    TransDataTo5HD(dst, src, ...);         // 静默杀死内核（no-op, rc=0）
}

// After: AIC_ONLY 声明 + HardEvent 链 + 标量 UB gather + 探针定序 + 预扫描分流
__global__ __aicore__ void conv_dw_cube_kernel(...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    ...
    SetFlag<HardEvent::MTE2_MTE3>(...); WaitFlag<HardEvent::MTE2_MTE3>(...);  // 显式事件链
    // 标量 UB gather 替代 TransDataTo5HD: dst[hw*16+lane] = src[lane*HW+hw]
    // fractal n1 = g*(ci1G*DkHkWk) + (kd*ci1G + ci1)*hkWk + hw   // 探针实测序
}
// Host: 预扫描 max|x|*max|g|*S > 5e10 或含 inf/NaN → Kahan 兜底 kernel
```

**Applicability conditions**:
- 任何在 910B2C 上新写的 Cube（AIC）内核（四条纪律与算子无关）
- vendor fractal 布局读写（conv backprop-filter dw、wrepack 等）必须探针定序
- 紧容差（bf16/fp16 vs fp64 golden）且存在大相消元素的归约类算子适用精度路由
- compare 工具判读陷阱：NaN 掩码不匹配时 MERE/MARE 默认 0.0，"MERE=0 但 FAIL" 不代表有限值精确

**Strategy IDs**: X21（关联 X18 k 序探针纪律的 3D 推广）

---

_Add new entries below as they are discovered._
