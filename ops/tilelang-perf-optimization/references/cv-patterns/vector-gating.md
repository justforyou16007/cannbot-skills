# 纯 Vector 链：融合计算，保留公式分支

## 何时读

gate/softplus/exp/sigmoid 等逐元素链，源码无矩阵乘。先构建 V 侧候选；没有 C 的单核本身不能证明 `Tv≫Tc`。

## 可迁移规则与案例

- **S3-01 / FusedGdnGatingKernel**：`fused_gdn_gating_kernel` 在单 Triton kernel 内计算 `g=-exp(A_log)*softplus(a+dt_bias)` 和 `beta=sigmoid(b)`；按 head 的参数广播复用，保留两个输出。
- **S3-02 / FusedKdaGate**：`kda_gate_fwd_kernel` 将 head 级 `-exp(A)` 提升到 token/D tile 外复用，提供 tile autotune。迁移的是公式和复用机制，不能直接复制 Triton 调参值到另一平台。
- `tl.where` 的结果选择不保证未选分支的 exp/log 不执行；指令与 SFU 收益须看编译产物及 profiling。
- 纯 V 仍可用内部搬运/计算双缓冲；CANN SwiGLU 的 `isDoubleBuffer` 选择 1/2 buffer，不能写“纯 V 没有双缓冲”。

## 合同反例：两种 KDA gate

```text
unbounded: -exp(A) * softplus(g+bias)
bounded:   lower_bound * sigmoid(exp(A)*(g+bias))
A=0, g+bias=0, lower_bound=-5 → 分别为 -ln(2) 与 -2.5
```

所读 Triton fused gate 对应 softplus；`apply_kda_gate` 的 safe_gate 路径为 PyTorch 展开。两者不是精度阈值内的替换，也不能因为位于 triton 目录就把后者称为 Triton kernel。记录实际调用者，避免优化一个未 dispatch 的实现。

## 最小验证

分开 use_lower_bound/use_bias、softplus beta/threshold；覆盖阈值两侧、极端值、尾 tile，核对输出 dtype。比较展开链与融合核的设备和整段时间，再判断瓶颈是 launch、访存还是超越函数。

## 固定版本证据

- [GDN-GATE](https://github.com/vllm-project/vllm-ascend/blob/5f8a1286a2d04d35b94ebc8a961057c48a7b82d2/vllm_ascend/ops/triton/fused_gdn_gating.py)
- [KDA-GATE](https://github.com/vllm-project/vllm-ascend/blob/5f8a1286a2d04d35b94ebc8a961057c48a7b82d2/vllm_ascend/ops/triton/kda/kda.py#L1130-L1360)
- [KDA-GATE-PY](https://github.com/vllm-project/vllm-ascend/blob/5f8a1286a2d04d35b94ebc8a961057c48a7b82d2/vllm_ascend/ops/triton/kda/gate.py#L1-L190)
- [SWIGLU-ENTRY](https://github.com/hicann/ops-nn/blob/0026707d7ecb24d258e030d831c8121ac2d055bd/activation/swi_glu/op_kernel/swi_glu.cpp#L1-L110)
- [L4-FusedGdnGatingKernel](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/FusedGdnGatingKernel/definition.json)
- [L4-FusedKdaGate](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/FusedKdaGate/definition.json)
