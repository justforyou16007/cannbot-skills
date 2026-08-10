# Pattern: Attention / FA 反量化（anti-quant）

## 先读这个

- FA 量化扩展契约（scale 沿 K-mmad、量化 dtype 宏、V2 末块量化链、清单 Q1-Q5）：`ops/ascendc-tiling-design/references/flashattention/subfamilies/quantization_design.md`。
- Attention 模式路由（TND/paged/mask 等）：`@references/attention-patterns/AttentionPatternIndex.md`。

## 1. 识别条件

- `npu_sparse_flash_attention_antiquant` / attention 算子消费 int8/int4 KV cache。
- 参数含 `key_quant_mode` / `value_quant_mode` / `query_quant_mode` / `quant_scale_repo_mode` / `cache_mode`。
- 输入含量化 cache + 对应 scale（`key_dequant_scale` / `value_dequant_scale` / scale repo）。
- MLA/KVCache C8 场景（`npu_mla_prolog_v3` 的 weight/kv/query 量化参数）。

## 2. 决策轴选择

| 轴 | 取值 |
|----|------|
| 对象 | attention 的 Q/K/V/P（cache）|
| 粒度 | per-group（K 方向，MXFP8 G=32）或 per-token |
| 对称性 | 对称 |
| scale 来源 | static（cache 写时预存 scale / scale repo）|
| dtype | int8 / fp8 e4m3 / MX（e8m0 scale）|

## 3. 核心规则

- **scale 必须沿被消费 matmul 的 K-mmad 轴量化**：

| 张量 | 被消费 matmul | K-mmad 轴 | scale 量化轴 |
|------|------------|---------|-------------|
| Q | Q·K^T | D | D |
| K | Q·K^T | D | D |
| P | P·V | S_k | S_k |
| V | P·V | S_k | **S_k**（不是 BSND innermost 的 D）|

> 反直觉陷阱：V 在 BSND 布局下 D 是 innermost，但 V scale 必须沿 S_k。

- MX 量化：`MmxMmad`（ISASI）/ `MxMatmul`（高阶），fp8 类型 `fp8_e4m3fn_t`，scale 是 `fp8_e8m0_t`，块大小 32 元素，沿 K-mmad 轴。
- V2 末块量化链：`ReduceMax → scale=amax/quant_max → Cast<quant_dtype,float>(O/scale)`，禁止 cast 溢出（RINT→NaN）。

## 4. 实现要点（AscendC）

- Kernel 类型与基础 FA 一致（典型 `__mix__(1,2)`）；量化路径不改变 kernel 类型。
- scale staging：Q/K/V 的 scale 从 GM 加载到 L1 供 MX Mmad 消费；P 在 V1 末块量化后的 P-scale(e8m0) 从 AIV 写回 L1P 段供 AIC BMM2 读取（容量按运行时 Sk 动态分配，不静态预留 Sk_max）。
- MXFP8 的 BMM1 引入 K 维子循环（c1v1Loop）：按 K 切片（典型 256），子循环内 L1 双缓冲。
- BMM2 数据量减半 → UB 中间段可退单缓冲。
- 编译宏：量化 dtype 类型（MXFP8/INT8/AntiQ）与量化 vs 非量化都必须分离（影响 typedef 与 Matmul 模板参数）。

## 5. 精度要点 + 检查点

- [ ] **清单 Q1（I5 落地校核）**：所有 scale 张量内存连续轴对齐其消费 matmul 的 K-mmad 方向？V scale 沿 S_k 已显式声明？
- [ ] **Q2**：数据载体 vs scale 载体的 L1 head offset 公式分别独立推导？
- [ ] **Q3**：V2 末块量化链各 API（DataCopy/Reduce/Cast）对齐要求显式校核？
- [ ] **Q4**：cast 数值公式保证 `amax/scale` 不超量化 dtype max（防 RINT→NaN）？
- [ ] **Q5**：Load2DMX 系列 API 字段单位（kStep/kStartPosition）与量化 dtype 匹配？
- [ ] scale repo / dequant scale 与量化 cache 的 layout（cache/scale scatter、CP 对齐）一致。

## 6. 参考实现

- `ops/ascendc-tiling-design/references/flashattention/subfamilies/quantization_design.md` — 契约与清单。
- `model/model-infer-quantization/references/quantization-structure-cards.md` 结构卡 C — MLA/KVCache C8 模型级边界。
