# 串行基线：成熟 C 后接独立 V

## 何时读

投影后接归一化、GMM 后接激活/量化；正在决定保留独立算子还是开发融合。先建立可运行、可维护的合同基线，不根据算子名字预设 `Tv≪Tc`。

## 可迁移规则

- 把融合候选与串行路径放在同一完整合同下比较，保留正确基线作为对照。
- 分开测 C、V、整段；独立 C/V 时间之和不自动等于图模式总耗时，launch、搬运和并发竞争也参与总成本。
- “fused projection”只说明某些投影合并，不说明后续 RMSNorm 已融合。先追 call site、能力开关和实际 dispatch。

## 源码案例与限制

| 研究编号 | 源码路径/符号与机制 | 不能据此推断 |
|---|---|---|
| S1-01 / MlaPreprocessOperation | `_mla_preprocess` 的非融合分支：`fused_qkv_a_proj → split → q_a_layernorm`；`AscendRMSNorm.forward_oot` 无 residual 时调用 `npu_rms_norm` | 不是完整 MLA ABI 等价；RoPE、eps、cache 写入、量化和融合 prolog 路由需另核对；没有测出 Tv/Tc |
| S1-02 / FFN | `apply_moe_mlp` 根据 `supports_fused_activation` 选择合并路径，或 `apply_gmm1 → activation → apply_act_quant → apply_gmm2` | 这是局部 CV/CVC；量化可能再 launch，不能写成恰好 C+一个 V；FFN vendor reference 不是独立数学 golden |

## 最小验证

- MLA 按 token、Q/KV rank、dtype、prefill/decode 分组；先对齐投影→RMSNorm，再验证 GPT-J 风格 RoPE 和 cache 副作用。
- FFN 固定一种 quant_method、激活及 gate/up 排列，核对 scale 定义和各阶段 cast；记录融合与回退的 kernel 数。
- 同设备、同数据和图模式比较串行/候选；无实测时只输出待验证方案，不声称“串行最优”或“必须融合”。

## 固定版本证据

- [MLA](https://github.com/vllm-project/vllm-ascend/blob/5f8a1286a2d04d35b94ebc8a961057c48a7b82d2/vllm_ascend/attention/mla_v1.py)
- [RMS-WRAPPER](https://github.com/vllm-project/vllm-ascend/blob/5f8a1286a2d04d35b94ebc8a961057c48a7b82d2/vllm_ascend/ops/layernorm.py#L1-L165)
- [RMS-ENTRY](https://github.com/hicann/ops-nn/blob/0026707d7ecb24d258e030d831c8121ac2d055bd/norm/rms_norm/op_kernel/rms_norm.cpp#L1-L140)
- [MOE](https://github.com/vllm-project/vllm-ascend/blob/5f8a1286a2d04d35b94ebc8a961057c48a7b82d2/vllm_ascend/ops/fused_moe/moe_mlp.py#L1-L300)
- [SWIGLU-ENTRY](https://github.com/hicann/ops-nn/blob/0026707d7ecb24d258e030d831c8121ac2d055bd/activation/swi_glu/op_kernel/swi_glu.cpp#L1-L110)
- [L4-MlaPreprocessOperation](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/MlaPreprocessOperation/definition.json)
- [L4-FFN](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/FFN/definition.json#L1-L100)
