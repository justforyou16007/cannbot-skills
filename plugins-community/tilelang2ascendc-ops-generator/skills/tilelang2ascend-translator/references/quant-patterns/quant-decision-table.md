# Quant 决策表：决策轴 A-E → 实现路径

> 任何量化算子先按 5 个决策轴定位，再决定公式 / API / 精度档。每个 pattern 文档内部也会给出该形态的决策选择，本文是跨形态的汇总表。

## 决策轴定义

| 轴 | 取值 | 影响 |
|----|------|------|
| A. 量化对象 | activation / weight / KV-cache / output | 决定算子形态与 scale 来源（output 多为 dynamic，weight 多为 static） |
| B. 粒度 | per-tensor / per-channel / per-channel(col) / per-token(row) / per-group(K-block) | 决定 scale 形状与广播方式 |
| C. 对称性 | symmetric(scale) / asymmetric(scale+offset) | 决定公式分母（127 vs 255/15）与是否要 offset |
| D. scale 来源 | static(预设) / dynamic(运行时 absmax) | 决定是否要 in-kernel reduce + 标量读 |
| E. dtype | int8 / int4 / fp8(e4m3·e5m2) / MX(fp8·fp4) / bf16 | 决定 Cast 链路与硬件量化指令 |

## 决策表

### A × 对象 → 形态

| 量化对象 | 典型算子形态 | 参考 pattern |
|---------|-------------|-------------|
| activation（逐 token 动态） | `dynamic_quant` / quantize | `quantize-dequantize.md` |
| output（matmul 输出再量化） | `quant_matmul`（int8 输出） / fused-quant 收尾 | `quant-matmul.md` / `fused-quant.md` |
| weight | 权重 per-group 量化 | `weight-group-quant.md` |
| KV-cache | KV 量化存储 | `kv-cache-quant.md` |
| attention Q/K/V/P | anti-quant / FA 量化 | `anti-quant.md` |

### B × C × D → 公式与 scale 形状

| 组合 | 公式 | scale 形状 | 广播方式 |
|------|------|-----------|---------|
| per-tensor, symmetric, static | `q = round(x/scale)`，`scale=预设标量` | 标量 | `Muls` |
| per-tensor, asymmetric, static | `q = round((x-offset)/scale)` | 标量 × 2 | `Muls` + `Adds` |
| per-channel, symmetric, static | `q = round(x/scale[col])` | `[N]` | 逐行 `Mul`（列广播） |
| per-token, symmetric, dynamic | `scale[i]=max(abs(x[i,:]))/127`，`q=round(x/scale[i])` | `[M]` | 行广播 `Broadcast` 或逐行 `Muls` |
| per-group, symmetric, static | K 方向每 G 元素一个 scale | `[M, K/G]` 或 `[N, K/G]` | 硬件折叠（`SetQuantVector`）或显式分段 |
| per-token, asymmetric, dynamic | `scale[i]=(max-min)/255`，`offset[i]` | `[M]` × 2 | 行广播 |

### D × 实现方式

| scale 来源 | 实现 | 关键 API |
|-----------|------|---------|
| static | 直接 `DataCopy` scale 到 UB | `DataCopyPad` |
| dynamic | in-kernel reduce：`Abs` → `ReduceMax` → `GetValue` | 需 `V_S`/`S_V` 事件同步（见 `quant-precision-rules.md` §标量同步） |

### E × Cast 链路

| dtype | 量化 cast | 反量化 cast |
|-------|----------|------------|
| int8 | `f32 → f16 → s8` 两步 RINT（无 f32→s8 直接指令） | `i32 → f32` CAST_NONE；输出 fp16 CAST_NONE / bf16·i32 CAST_RINT |
| int4 | int8 拆高/低 4 位（`(x&0x0F)-8`）或 f32→int4 两步 | 合并 `high×16+low` |
| fp8 | `f32 → fp8_e4m3fn` 等 | `fp8 → f32` |
| MX | 硬件 MX Mmad + e8m0 scale（K 方向每 32/64） | 硬件内折叠 |

## 精度档（与验证标准对应）

| 输出 dtype | 判定 |
|-----------|------|
| fp16 / bf16 / fp32（dequant 输出） | MERE/MARE 阈值 |
| int8 / int16（量化输出） | 元素级 `max_abs_diff ≤ 1`（±1 LSB） |
| int32 / bool | 严格相等 |

详见 `quant-verification.md`。

## 选型提示

- **scale 先乘、pertoken 后乘**：dequant 顺序必须是 `(acc × scale) × pertoken`，与 torch_npu 对拍 bit-exact（见 `quant-precision-rules.md`）。
- **int32 bias 在 scale 前、float bias 在 scale 后**：保证精度（见 `quant-precision-rules.md`）。
- **per-token dynamic 需要 AIV reduce + Scalar 读**：`KERNEL_TYPE_MIX_AIC_1_2` 即可。
- **per-group 权重**：优先用高阶 Matmul API 的 `SetQuantVector` 硬件折叠，避免手动反量化。
- **910B2C 量化 matmul**：必须纯 Cube 高阶 API + 纯 AIV dequant 双 kernel（见 `quant-matmul.md`）。
