# 局部 epilogue：保留最终输出，减少额外中间体

## 何时读

dot 后接 gate/beta/mask，或多份矩阵结果在局部累加后只写最终输出。检查局部活跃集和数值合同，不能从“源码无中间 Tensor”推出“编译器无 spill”。

## 源码案例与限制

- **S6-01 / ChunkScaledDotKktFwd**：`chunk_scaled_dot_kkt_fwd_kernel` 在 dot 后做 safe_exp gate、beta、严格下三角 mask，最后写 A。最终 A 仍是返回值，不能删除；safe_exp clamp、H/Hg 映射、g=None dispatch、output_dtype 和尾块需对齐。
- **S6-02 / ChunkGlaFwdKernelO**：`chunk_gla_fwd_kernel_o` 合并 gated query×state 与 A×V 到 `b_o`，只写 O。局部 epilogue 可借鉴，但整体是含 V→C 及多个 C 的图，不是简单单 C→V；不要与 ChunkO 近名条目混用。

## L4 数值和布局合同

1. L4 使用 `exp(g)`，目标 KDA kernel 使用 `exp2(g)`；其余条件相同才可用 `g_kernel=g_L4/ln(2)`。若上游已转换，不要重复转换。
2. L4 h 为 K×V，目标指针按 V×K 读再转置；核对真实 stride 后适配存储，不是仅改 shape。
3. 目标 `q*scale` 后 cast 回 q.dtype；L4 fp32 展开可能有不同舍入。检查 A 中是否已经包含 scale，避免重复乘。
4. KKT 的严格下三角应排除对角线；safe_exp 和 beta 的位置不随意交换。

## 字节例子与最小验证

L4 KKT：k `[1,80,8,128]`，beta/g `[1,80,16]`，chunk64，尾16；fp32 输出 `[1,80,16,64]` 为320 KiB。假想同等 padded 中间体写+读640 KiB；这是展开模型，不证明每个 unfused 实现都必然产生这些流量，也不证明融合必要。

验证 chunk 边界、H/Hg 分组、极端 gate/clamp、mask、cast 和 dtype，先对齐数值域与布局再验精度。编译后检查局部 buffer、寄存器/UB 压力和 GM 搬运，再与同合同基线做设备及整段 profiling。

## 固定版本证据

- [KKT](https://github.com/vllm-project/vllm-ascend/blob/5f8a1286a2d04d35b94ebc8a961057c48a7b82d2/vllm_ascend/ops/triton/fla/chunk_scaled_dot_kkt.py#L1-L200)
- [KDA-OUT](https://github.com/vllm-project/vllm-ascend/blob/5f8a1286a2d04d35b94ebc8a961057c48a7b82d2/vllm_ascend/ops/triton/kda/kda.py#L910-L1135)
- [L4-ChunkScaledDotKktFwd](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/ChunkScaledDotKktFwd/definition.json)
- [WL-ChunkScaledDotKktFwd](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/ChunkScaledDotKktFwd/workload.jsonl#L1-L3)
- [L4-ChunkGlaFwdKernelO](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/ChunkGlaFwdKernelO/definition.json)
- [L4-ChunkO](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/ChunkO/definition.json)
