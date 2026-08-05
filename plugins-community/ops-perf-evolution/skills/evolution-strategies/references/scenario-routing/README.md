# Scenario Routing — 算子归族与工作流入口

> 进化优化的第一步：**先归族，再进对应族的工作流（workflow），按工作流的阶段顺序调用策略卡**。
> 不要直接在 strategy-index.md 平铺选卡——那里只是注册表，没有主次顺序。

## 使用协议

```
Step 0 — 归族：用下方判定表确定算子族（结构判据，不靠算子名）
Step 1 — 通用前置：打开 _common-prerequisites.md，逐项判定（命中用卡，未命中跳过）
Step 2+ — 族内阶段：打开对应族的 workflow 文件，按阶段顺序推进
Refine 阶段 — profiling 证据可用时，strategy-index.md 的瓶颈查表优先于工作流顺序
```

## 族判定表（按特异性从高到低判定，命中即停）

**不要靠算子名字判族**（EMSA / BAM / CoT / CrissCross / Window / Axial / BigBird / Linformer 等变体名字不含标准词）。判据来自三处可观察证据：`model.py` 数学结构 / kernel 与 host 代码特征 / shape 特征。

| 顺序 | 族 | 结构判据（命中任一即归族） | 易混淆点 | 工作流 |
|---|---|---|---|---|
| 1 | **FlashAttention** | ① 计算主干 `Q@K^T → (scale+mask) → softmax → @V`（含 online softmax 变体）；② kernel 含 online softmax 状态机（running max/sum/rescale）或 FA 特征参数（`preTokens`/`nextTokens`/sink/paged KV cache）；③ head 数关系命中 MHA/MQA/GQA（`num_q_heads` vs `num_kv_heads`，判据见 [../../../evolution-knowledge/references/a3/algorithm_insights/attention_family.md](../../../evolution-knowledge/references/a3/algorithm_insights/attention_family.md)） | softmax 夹在两个 matmul 中间是 FA 的指纹，**必须先于 matmul/conv 判** | [flashattention.md](flashattention.md) |
| 2 | **Convolution** | ① 滑窗局部加权和（im2col+GEMM 形态）；② kernel/host 含 `kH`/`kW`/`stride`/`pad`/`dilation` 卷积参数或 `im2col`/`LoadData3D` 证据；③ 变体：depthwise（`groups==Cin`）、ConvTranspose（col2im/分数步长形态） | im2col 之后就是 GEMM——**先于 matmul 判**，滑窗参数是 conv 指纹 | [convolution.md](convolution.md) |
| 3 | **Matmul 系**（cv_fusion / moe） | ① 主干为（批量）矩阵乘 `A[M,K]@B[K,N]`，或 matmul + 向量后处理融合；② kernel 用 `MatmulImpl`/`Mmad`/L0A/L0B/L0C 或 host 侧 `MatmulApiTiling`；③ MoE 变体加判 expert 分组/门控（GroupedMatmul、SwiGLU） | 纯向量/标量模拟的 matmul 也算（正是 P102 的猎物） | [matmul.md](matmul.md) |
| 4 | **Quantization** | 存在 scale/zero_point、quant/dequant 算子对、int8/fp8 张量 | **量化通常不是独立族而是 trait**：命中后回主干族 workflow 的量化支路；仅当算子本体就是 quant/dequant 时才进本族 | [quantization.md](quantization.md) |
| 5 | **Reduction 系**（normalization/softmax） | ① 沿轴归约（sum/mean/max/min）；② normalization = 减均值除方差 + scale/shift；③ kernel 含 `ReduceSum`/`WholeReduceSum`/Welford 状态 | 独立的 softmax 归此；夹在 QK^T 与 PV 之间的归 FA | [reduction.md](reduction.md) |
| 6 | **Sort** | 比较-交换/归并结构、TopK、argsort 索引跟踪 | — | [misc-families.md](misc-families.md) |
| 7 | **Conversion** | **输出元素集合 = 输入元素集合，值不变只变位置/布局**（transpose/permute/split/concat）——纯搬运 + 地址重排，无算术运算 | vs index_scatter：conversion 地址映射编译期静态，gather/scatter 索引是运行时数据 | [conversion.md](conversion.md) |
| 8 | **小族** | index_scatter（gather/scatter 运行时索引）、pooling（滑窗池化）、broadcast（输入 shape 不同需广播对齐）、optimizer（参数+梯度+状态多路更新） | pooling 的滑窗是池化（max/avg），不是 conv 的加权和 | [misc-families.md](misc-families.md) |
| 9 | **Elementwise** | **排除法兜底**：输入输出 shape 相同、逐元素独立、无跨元素依赖，以上均不命中 | 放最后判——它是"无结构"的族 | [elementwise.md](elementwise.md) |

## 兜底规则

1. **多族特征同时命中**（如量化 GQA、conv+bias+激活融合）→ 拆"主干族 + trait"：主干决定进哪个 workflow，trait（量化/稀疏/融合后处理）走 workflow 内的对应支路。与 `ops/ascendc-tiling-design` 的"基础形态 + 正交 trait"同构。
2. **全部不命中** → 只做 [_common-prerequisites.md](_common-prerequisites.md) 通用前置 + strategy-index.md 瓶颈查表，workflow 缺失不妨碍进化。
3. **workflow 内标注"覆盖缺口"的阶段** → 该场景尚无沉淀卡，走通用前置 + 瓶颈驱动，优化完成后按 X 系列流程沉淀新卡。
