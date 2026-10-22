# 在线消费：区分缩小状态与消除上游物化

## 何时读

Matmul 后沿 batch 归约、分块 Top-K、online softmax。先问最终输出是否必须保留完整中间矩阵，再选择在线状态。

## 三种机制与边界

| 研究编号 | 源码与可迁移机制 | 保留下来的成本/合同 |
|---|---|---|
| S5-01 / QuantMatmulReduceSum | `MMCompute/VectorCompute/Dequant/DataCopyOut`；`PIPELINE_NUM=4` GM ring，最终 Y 初始化后 `SetAtomicAdd` 累加 batch tile | GM partial ring 仍存在；不是“零中间体”；核对 NZ、scale 顺序、输出 dtype、原子累加误差与初始化先于累加 |
| S5-02 / LightningIndexer | `ProcessTopK` 将 `scoreOutLocal` 幸存者与下一 trunk 合并到 `mrgValueLocal` | `ProcessVec1` 先写 scoreGm，只有消费端 online；合并容量约 `topkCountAlign256+trunkLen`，不是固定 k；L4 还按 index 排序并处理 values |
| S5-03 / GQA | 维护每行最大值、归一化状态和输出累加器，而不保存全量 score/probability | 状态仍随 query tile 和 D 变化；exp2/log2 数值域、全 mask/空输入必须单独处理 |

## 不变量

Top-K 在固定总排序（含 tie 的 index 次序）下满足：

```text
TopK(A ∪ B) = TopK(TopK(A) ∪ B)
```

同时携带 value/index，内存约 O(k+chunk)。如果合同不保证 tie 顺序，应按集合及容许规则验证，不能擅自要求或假定稳定次序。

Online softmax 对新块 x，在相同指数域内更新：

```text
m' = max(m, max(x))
alpha = exp(m-m')
p = exp(x-m')
l' = alpha*l + sum(p)
a' = alpha*a + p@V
output = a'/l'
```

这是数学解释，不要求源码保存原始 l（也可能保存 log-sum-exp）。max 改变时 l 和 a 都要 rescale。全 mask 时 `-inf - -inf` 会出 NaN，空集合分支与输出约定不可省略。

## 字节例子与验收

L4 QMRS 的 x1 `[4,256,2048]`、x2 `[4,2048,4096]` 为 int8。假设展开 fp32 batch 结果 `[4,256,4096]`，大小16 MiB；最终 Y `[256,4096]` 为4 MiB，假想中间体写+读32 MiB。ring partial 另按 `4*coreNum*baseM*baseN*sizeof(int32)` 计（其余 workspace/对齐另计）。这不是实测 HBM 节省或4倍加速。

验证分块/尾块、ties、k 大于有效元素数、归约顺序误差和全 mask。LightningIndexer vendor reference 不是独立数学 golden；kernel 与 wrapper 后处理分开计时。不要因为消费者在线就删除上游仍被其他消费者使用的 scoreGm。

## 固定版本证据

- [QMRS-MIX](https://github.com/hicann/ops-nn/blob/0026707d7ecb24d258e030d831c8121ac2d055bd/matmul/quant_matmul_reduce_sum/op_kernel/quant_matmul_reduce_sum_mixcore.h)
- [LI-VEC](https://github.com/hicann/ops-transformer/blob/bbb4dafe296119266a0a64133f17bc42b6597ed3/attention/lightning_indexer/op_kernel/arch35/lightning_indexer_service_vector_arch35.h)
- [GQA](https://github.com/vllm-project/vllm-ascend/blob/5f8a1286a2d04d35b94ebc8a961057c48a7b82d2/vllm_ascend/models/minimax_m3/ops/msa_m3_triton_a5.py)
- [L4-QuantMatmulReduceSum](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/QuantMatmulReduceSum/definition.json)
- [WL-QuantMatmulReduceSum](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/QuantMatmulReduceSum/workload.jsonl#L1-L3)
- [L4-LightningIndexer](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/LightningIndexer/definition.json#L1-L130)
- [L4-GqaSparseFwd](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/GqaSparseFwd/definition.json)
