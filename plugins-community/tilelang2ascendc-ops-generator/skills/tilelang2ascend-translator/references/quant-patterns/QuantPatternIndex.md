# Quant Pattern Index

> 量化/反量化类算子（quantize / dequantize / quant_matmul / dynamic_quant / weight-group-quant / kv-cache-quant / fused dequant+activation+quant / anti-quant）统一路由入口。

生成量化类算子前先读本文件，用它决定需要看的模式文档。模式可以组合，但不要临时发明新模式。

渐进式披露原则：不要一上来读完所有模式文档。先用本索引定位命中的模式，再只读对应文档顶部的 `先读这个`；只有实现细节不够时，再继续读该文档后面的 `决策轴选择 / 核心规则 / 实现要点 / 精度要点`。

## 核心模式

| 如果看到 | 读这个文档 |
| --- | --- |
| `torch_npu.npu_quant_matmul` / 算子名含 `QuantMatmul` / int8×int8 输入 + `scale`/`pertoken_scale`/`offset`/`bias` 输出高精度 | `@references/quant-patterns/patterns/quant-matmul.md` |
| `torch_npu.npu_dynamic_quant` / 逐行 reduce → `max(abs)/127` → 输出 int8 + 每行 scale | `@references/quant-patterns/patterns/quantize-dequantize.md` |
| 纯 `quantize` / `dequantize`（fp→int+scale 或 int+scale→fp，无 matmul） | `@references/quant-patterns/patterns/quantize-dequantize.md` |
| a8w4 / int4 权重 / `group_size` / per-group 权重 scale / 权重拆 int4 | `@references/quant-patterns/patterns/weight-group-quant.md` |
| KV cache 量化（int8/int4 cache + scale cache / `k_proj.output_scale` / `v_proj.output_scale` / `get_cache_scale`） | `@references/quant-patterns/patterns/kv-cache-quant.md` |
| dequant 后接 activation 再 quant（`npu_dequant_swiglu_quant` / dequant→SiLU·GELU·SwiGLU→requant） | `@references/quant-patterns/patterns/fused-quant.md` |
| sparse FA / attention 反量化（`key_quant_mode` / `value_quant_mode` / `quant_scale_repo_mode` / `npu_sparse_flash_attention_antiquant`） | `@references/quant-patterns/patterns/anti-quant.md` |
| 以上按对象更细分：仅激活量化（无权重） | `patterns/quantize-dequantize.md` |
| MXFP8/MXFP4（per-group 32/64 scale，`fp8_e8m0`） | `patterns/weight-group-quant.md`（MX 小节） |

## 组合顺序

多个量化维度同时出现时，按这个顺序理解：

```text
1. 量化对象：先分清 activation / weight / KV-cache / output 各量化哪一路。
2. 粒度：per-tensor / per-channel / per-token / per-group(K-block) 各是什么粒度。
3. scale 来源：static（预设 scale）还是 dynamic（运行时 absmax）。
4. 对称性：非对称时有 offset/zero-point（int4 常用 scale+offset，分母 15/255）。
5. 融合：反量化/量化是否与 activation 或 matmul 融合在同一个算子内。
```

命中多个模式时按需阅读，先读每个命中文档的 `先读这个`。

## 生成前问题

1. 量化对象是 activation / weight / KV-cache / output？
2. scale 粒度是 per-tensor / per-channel / per-token / per-group？
3. 对称量化还是非对称（有 offset / zero-point）？
4. scale 是预设（static）还是运行时计算（dynamic，`max(abs)`）？
5. dtype 是 int8 / int4 / fp8(e4m3·e5m2) / MX(fp8·fp4)？
6. 输出是高精度（dequant，`int→fp`）还是再量化（`fp→int + scale`）？
7. 是否与激活（SiLU/SwiGLU/GELU）或 matmul 融合？

根据答案选择上面的文档。若 7 个问题全部否定（无任何量化特征）→ 本索引不适用，走通用 matmul / elementwise 流程。

## 共享参考

量化实现时同时对照以下共享文档：

- `@references/quant-patterns/quant-decision-table.md` — 决策轴 A-E 组合 → 公式 / API / 平台 / 精度档
- `@references/quant-patterns/quant-formulas.md` — 量化公式族（对称 / 非对称 / 动态 range / FP19 / Int9）
- `@references/quant-patterns/quant-cast-round.md` — Cast 能力矩阵 + RoundMode 语义 + fp32→int8 两步
- `@references/quant-patterns/quant-precision-rules.md` — 精度铁律（乘法顺序 / bias 位置 / cast / scale 截断 / 标量同步）
- `@references/quant-patterns/quant-verification.md` — 量化算子验证标准（MERE/MARE + ±1 LSB）

## 平台备注

- 910B2C / CANN 9.0：量化 matmul 必须走纯 Cube 高阶 Matmul API + 纯 AIV dequant 双 kernel（见 `patterns/quant-matmul.md`）；手写 Mmad 不可用。
- A2/A3（DAV_2201）：MatmulImpl + AIV 反量化；A5（DAV_3510）：Blaze MIX 路径（`a8w8`/`mx` 场景）。
