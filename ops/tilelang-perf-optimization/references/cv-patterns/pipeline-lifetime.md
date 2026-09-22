# 核内流水：先证明槽位生命周期

## 何时读

C/V 跨块 overlap、双缓冲、GM ring buffer，或异步 Matmul 与等待之间有可提前执行的工作。图级多 stream 需要另外的执行与同步证据。

## 可迁移规则

两个槽位的关键依赖（i 表示 chunk 序号，不是核编号）：

```mermaid
flowchart LR
    Ci["C(i) 写槽 i%2"] -->|ready| Vi["V(i) 消费"]
    Vi -->|reuse| Ci2["C(i+2) 覆写同槽"]
```

- ready 保护消费前的数据完成；reuse 保护旧数据消费结束后才能覆写。两条边不可合并为“每轮同步一下”。
- 双缓冲是两份槽位及所有权，不能只复制循环；检查启动、排空、尾块、空任务参与和实际搬运 pipe。
- workspace 按实际同时存活的槽位、每槽尺寸、核数、dtype 与对齐计算；不是所有 GMM 都要保存全量输出中间体。
- 提前执行的工作不能依赖未就绪结果。Vector Pre/Post 共用 Vector 资源，不能把三阶段默认当三台独立引擎。

## 源码案例与限制

- **S2-01 / GMSQ A8W8**：`GMMSwigluSplitWorkSpaceCompute::Process` 以 `parallelNum=2` 交替 `mmOutGM1/mmOutGM2`；`GmmsqAllCoreCubeVecStageBarrier` 与 `GmmsqAllCoreWorkspaceReuseBarrier` 分别保护阶段就绪/复用。groupList 是累计边界，循环跳过空专家。GM 中间工作区仍存在。
- **同编号的 A8W4 MSD 扩展**：`GMMSwigluQuantPipelineSchedule::Process` 交错 `Pre(n+1)/Mid(n)/Post(n-1)`。这是不同量化变体，不能直接替代 L4 默认 int8×int8 合同。
- **S2-02 / FFN AntiQuant**：`SplitMM1` 先转换 W1、提交 MM1，再转换不依赖 MM1 输出的 W2，之后 `WaitIterateAll/End`，最后做激活。转换后的权重仍写 GM；能否覆盖等待要测量。

## 数值模型的使用范围

理想两阶段、单 C 资源和单 V 资源、每块耗时 c/v、n≥1：

```text
串行 = n*(c+v)
理想流水 = c+v+(n-1)*max(c,v)
n=4, c=v=10 us → 80 us / 50 us = 1.6x
```

这是教学模型，忽略同步/带宽竞争/搬运成本且假设分块独立；不是源码实测，更不是图级双流收益。

## 最小验证

覆盖空专家、1/2/3+ chunk、尾块跨专家、重复复用；核对 ready/reuse 的次数、槽位和参与者。正确后比较只改变调度的版本，采集实际 overlap、等待与整段耗时；不要同时改变 tile 和数学路径。

模板调优参见 [GMM 最佳实践](../best-practices/gmm_cv_fusion_optimize.md)，同步异常参见 [同步调试](../cv_sync_debug.md)。

## 固定版本证据

- [GMSQ-SPLIT](https://github.com/hicann/ops-transformer/blob/bbb4dafe296119266a0a64133f17bc42b6597ed3/gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_split_ws.h#L190-L310)
- [GMSQ-MSD](https://github.com/hicann/ops-transformer/blob/bbb4dafe296119266a0a64133f17bc42b6597ed3/gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_pipeline.h#L1-L240)
- [FFN-ANTI](https://github.com/hicann/ops-transformer/blob/bbb4dafe296119266a0a64133f17bc42b6597ed3/ffn/ffn/op_kernel/ffn_antiquant.h#L470-L620)
- [L4-GroupedMatmulSwigluQuant](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/GroupedMatmulSwigluQuant/definition.json)
- [L4-FFN](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/FFN/definition.json#L1-L100)
