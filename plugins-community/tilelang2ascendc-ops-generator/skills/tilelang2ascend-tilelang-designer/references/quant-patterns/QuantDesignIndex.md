# Quant 设计路由索引（TileLang 设计阶段）

> 量化/反量化类算子进入 Block/Tile 设计前，先读本文件定位量化形态与设计决策。
> 本文件是**薄设计层**：实现细节（AscendC API / Cast / 精度铁律）不在本目录复制，统一到主库
> `skills/tilelang2ascend-translator/references/quant-patterns/` 查阅（同插件内允许读取）。

## 使用方式

1. 读本文件定位命中的量化形态与设计决策。
2. 按需读主库对应 pattern 的「核心规则 / 实现要点」段，获取公式与结构细节。
3. 设计期产出 `design/PERF_DESIGN.md` 时，追加第 5 节的 quant 待验证清单。

## 1. 识别条件（看 model.py）

| 特征 | 量化形态 | 主库 pattern |
|------|---------|-------------|
| `npu_quant_matmul` / int8×int8 + scale/pertoken/offset/bias | quant_matmul | `quant-patterns/patterns/quant-matmul.md` |
| `npu_dynamic_quant` / 逐行 reduce → int8 + 每行 scale | 纯量化 | `patterns/quantize-dequantize.md` |
| int4 权重 / `group_size` / per-group scale | 权重量化 | `patterns/weight-group-quant.md` |
| KV cache 量化（int8/int4 cache + scale cache） | KV-cache | `patterns/kv-cache-quant.md` |
| dequant 后接激活再量化（dequant_swiglu_quant） | 融合 | `patterns/fused-quant.md` |
| sparse FA / attention 反量化 | anti-quant | `patterns/anti-quant.md` |

主库 `QuantPatternIndex.md` 含完整路由表与 7 个生成前问题，设计阶段同样适用。

## 2. 设计决策（公式 / 粒度 / scale 来源）

| 决策 | 选择依据 | 主库参考 |
|------|---------|---------|
| 量化公式 | 对称 127 / 非对称 255·15 / 动态 range / FP19 | `quant-patterns/quant-formulas.md` |
| scale 粒度 | per-tensor / per-channel `[N]` / per-token `[M]` / per-group | `quant-patterns/quant-decision-table.md` |
| scale 来源 | static（预设）/ dynamic（in-kernel reduce） | `quant-patterns/quant-decision-table.md` |
| 输出 cast | fp16 CAST_NONE / bf16·int32 CAST_RINT / int8 两步 | `quant-patterns/quant-cast-round.md` |

## 3. Block / Tile 结构设计要点

按量化形态给出 block/tile 级结构决策（TileLang 表达）：

### 3.1 quant_matmul（CV 融合：Cube int8 GEMM + Vector dequant epilogue）
- Cube：`T.gemm_v0(a_l1, b_l1, c_l0, init=(bk==0))`，int32 累加。
- C→V 交接：int32 acc `T.copy(c_l0, workspace[...])` → V 核读回 dequant。
- V 核 dequant epilogue：`T.copy(workspace[...], acc_i32_ub)` → cast → `T.tile.cast`。
- 决策：任务划分 `T.Kernel(B*m_num*n_num)` + `vid` 切分 M；或纯 Cube 写 workspace + 顺序 AIV kernel（910B2C 平台配方见主库）。
- 参考：`output_test_830_goodcase_L4/6_QuantMatmul/design/tile_level/quant_matmul.py`。

### 3.2 纯量化（per-token dynamic quant）
- 纯 AIV：每行 `T.reduce.max` → scale → `T.tile.cast` 两步 → int8。
- 决策：行划分 `blockM`、每核 `blockM/GetSubBlockNum()` 行；N 大时分 chunk（kBlockN 4096）。

### 3.3 权重量化（a8w4 MSD）
- pre(AIV 拆分 int8→2×int4) → mid(AIC int4 matmul + SetQuantVector) → post(AIV 合并/反量化/融合/重量化)。
- 软流水：`pre[n+1] ∥ mid[n] ∥ post[n-1]`，workspace 双缓冲 + `SyncAll`。

### 3.4 KV-cache quant
- 纯 AIV：每行 ReduceMin/ReduceMax → scale → 4-bit 量化 + 打包 → 写 cache + scale。

### 3.5 融合 quant（dequant+act+quant）
- dequant → activation（软件 fp32 路径）→ requant 三段全部在 UB 内链式完成（单 pass）。
- SwiGLU 跨 N-half：按 `[M, H]` 调度，每输出块产出左右两路 C tile。

## 4. 反量化顺序设计（设计期就要定对）

```text
int32 acc → [int32 bias] → cast fp32 → ×scale[col] → ×pertoken[row] → [float bias] → [offset] → cast out
```

- 必须是 `(acc × scale) × pertoken`（scale 先、pertoken 后）。
- int32 bias 在 scale 前（int32 域）；float bias 在 scale 后。
- 设计期就在 tile_level 表达里排对顺序，避免 Phase 4 D 类返工。

## 5. PERF_DESIGN.md quant 待验证清单（追加模板）

设计期 1e/2e 产出 `PERF_DESIGN.md` 时，quant 算子追加以下候选待验证项（按命中情况勾选）：

| # | 待验证项 | 假设 | 验证方法 | 命中时修改点 |
|---|---------|------|---------|-------------|
| QP1 | dequant 在 AIV 是否拖住 Cube | GEMM-bound 时 epilogue 只需跟上 | msprof Cube/Vec 占比 | 背压深度 / 单 pass 融合 |
| QP2 | scale 列向量每列块只 load 一次常驻 UB | 避免每 tile 重载 | UB 流量对比 | 列向量缓存 |
| QP3 | 行广播 `Broadcast` vs 逐行 `Mul` | 大 tile 用 Broadcast 更快 | 两种实现对比 | 选型 |
| QP4 | 双 kernel vs 单 kernel 软流水 | 单 kernel 有 CV 并行但复杂 | geomean 对比 | 结构选型 |
| QP5 | N>2048 列分块 PIPE_ALL 屏障开销 | 正确性优先，性能可优化 | 列分块数扫描 | 屏障级别 |
| QP6 | int32 acc C→V workspace 交接带宽 | 避免额外拷贝 | 带宽估算 | workspace 布局 |
| QP7 | Split-K / 大 K | K≫M×N 时可拆 | roofline | 任务划分 |

3.5 性能迭代以该清单为强制种子（与 `tilelang-perf-optimization` 流程一致）。

## 6. 边界

- 本文件不做 AscendC 实现细节（Phase 4 用主库 `quant-patterns/`）。
- 量化形态可组合（如 quant_matmul 同时带融合 activation），按主导形态设计，组合按主库 `QuantPatternIndex.md` 顺序理解。
