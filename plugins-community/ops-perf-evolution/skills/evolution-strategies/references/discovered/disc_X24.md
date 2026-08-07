---
id: X24
origin: discovered
discovered_round: 2
discovered_from: round_2/parallel_2
base_speedup: 1.757x (vs 基线) / 0.616x (vs PyTorch)
---

# Strategy X24: AIC_ONLY 任务类型压设备 launch 地板（固定开销主导算子）

## 核心思路
微小 shape、固定开销主导的算子（kernel 时间与 shape 几乎无关，cube_util 极低、标量占比高）其设备时间大部分是「每 AIC 调度启动」的固定开销，不是计算。X24 把任务类型从默认 `KERNEL_TYPE_MIX_AIC_1_2`（1 AIC + 2 AIV）改为 **`KERNEL_TYPE_AIC_ONLY`**（纯 cube 核，不启动 AIV 向量核），去掉每 AIC 调度的 2 个空转 AIV 任务启动，实测**设备固定地板从 ~7-8μs 压到 ~4.3-4.5μs**。叠加紧凑 tiling 加载（GM 标量读取 65→4 int32）后逼近 910B 的 kernel launch 物理地板。

## 适用场景
- **算子类型**: 微小 shape matmul / elementwise / reduction 等固定开销主导算子；纯 cube 计算、无 AIV 侧必做工作
- **瓶颈类型**: `scalar_compute` / `no_overlap` / `compute_bound`（固定开销而非计算量主导）、cube_util ≪ 标量占比
- **预期收益**: 5_MatmulTransB 设备侧 12.09μs→6.88μs（1.757x vs 基线 / 0.616x vs PyTorch，50/50 精度 PASS）；与手动 mmad / 低开销 tiling 叠加收益最大

## 实现要点
1. kernel 注册处声明 `KERNEL_TYPE_AIC_ONLY`（去掉 AIV 任务启动，每 launch 省 ~1.5-2us）
2. 配 compact tiling 头：manual 路径只读前 4-12 个 int32（替代全量 ~65 个逐标量加载）
3. 固定开销主导判定：不同 shape kernel 时间几乎相同 + cube_util 远低于标量占比（如 cube_util 3.95% / aic_scalar_ratio 0.661）
4. 与手动 mmad 分形加载（TQue CopyIn，非 LoadData3D）叠加，参考 X24 proven_solution

## 约束
- 纯 cube 算子收益最大；有 AIV 侧 epilogue/向量必做工作时不能盲目 AIC_ONLY
- 上限认知：910B 每 kernel launch 设备地板 ~4.3-5.6μs；目标加速比低于硬件地板时物理不可达，先确认目标可行性

## 来源
自动发现于 5_MatmulTransB 手动 mmad 攻坚轮 round 2/parallel_2（x0_x1, open_exploration），设备侧 12.09μs→6.88μs，1.757x vs 基线 / 0.616x vs PyTorch，50/50 精度 PASS
