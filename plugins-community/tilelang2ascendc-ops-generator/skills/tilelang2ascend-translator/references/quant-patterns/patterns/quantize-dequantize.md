# Pattern: 纯量化 / 反量化（quantize / dequantize / dynamic_quant）

## 先读这个

- 逐行（per-token/per-row）动态量化的完整实现，见 `@references/ascendc_dynamic_quant_kb.md`。
- dequant 类（dequant + activation + quant）的精度档与骨架，见 `@references/dequant_kernel_patterns.md`。
- 本文件只补这两份文档的定位与组合：**无 matmul、无权重** 的纯 quantize/dequantize 算子走本 pattern。

## 1. 识别条件

- `torch_npu.npu_dynamic_quant(x)` 类：逐行 reduce → 输出 int8 + 每行 scale。
- 纯 `quantize`：fp → int + scale（激活量化）。
- 纯 `dequantize`：int + scale → fp。
- 输入无 matmul、无 per-group 权重。
- 输出可能同时含 `quantized(int8)` + `scale(fp32)`（混合输出，分开判定，见 `quant-verification.md`）。

## 2. 决策轴选择

| 轴 | 取值 | 说明 |
|----|------|------|
| 对象 | activation / output | 无权重 |
| 粒度 | 通常 per-token / per-row | scale 形状 `[M]` 或去最后一维 |
| 对称性 | 多数对称（`max(abs)/127`）；非对称有 offset | 见 `quant-formulas.md` §2 |
| scale 来源 | 动态（in-kernel reduce）为主 | 需标量同步 |
| dtype | int8 常见；int4 需打包 | 见 `quant-cast-round.md` |

## 3. 核心规则（per-token dynamic quant）

```text
row_max_abs = max(|x|) per row
scale[i]    = max(row_max_abs[i] / 127.0, 1e-10)   # fp32
quantized   = round(x / scale[i]).clamp(-128, 127)  # int8
```

语义：每个 token / 每行独立。

## 4. 实现要点（AscendC）

```
per row m:
  // dequant 侧（若 int+scale→fp）
  load x[m,:] int32 → UB → Cast fp32 (CAST_NONE)
  Mul by scale[col] / Muls by scale[m]

  // quant 侧（fp→int8）
  Abs(x[m,:]) → tmp
  ReduceMax(absmax, tmp, work, N)          // fp32 域
  V_S sync → scale = absmax / 127
  (写 scale 输出，S_V sync)
  Muls(x[m,:], x[m,:], 1/scale)            // 或 Div 精确匹配 reference
  Cast(half, x, CAST_RINT) → Cast(i8, half, CAST_RINT)   // 两步
  DataCopy 写 int8 + scale
```

**API 流程（关键顺序）**：
1. `Abs` → `ReduceMax` → `GetValue`（需 `V_S` 同步）。
2. 写 scale 用 UB + `DataCopyPad`（不用 `GlobalTensor::SetValue` 直写）。
3. `Muls` 用 `1/scale`（标量）；需精确匹配除法语义时用 `Div`（scaleVec 由 `Duplicate` 填充）。
4. 输出 cast 走 f32→f16→s8 两步 RINT。

## 5. 精度要点 + 检查点

- [ ] scale 计算在 fp32 域（`ReduceMax<float>`），不用输入 dtype。
- [ ] 标量读有 `V_S` 同步、写回有 `S_V` 同步。
- [ ] int8 输出 `max_abs_diff ≤ 1`；scale 输出过 fp32 MERE/MARE。
- [ ] 输入 dtype 转 fp32 无损（bf16/fp16→fp32 用 CAST_NONE）。
- [ ] N 大时分 chunk（kBlockN，典型 4096）：reduce 阶段累计 chunk-max，quant 阶段用确定 scale 整行划分。

## 6. 参考实现

- `skills/tilelang2ascend-translator/references/ascendc_dynamic_quant_kb.md` — 完整实现 + TQue/Cast 约束。
- rowwise int8 quant 收尾类实现（注意其 reference 若是 vendor kernel 则属 Tier 2；新算子若 CPU oracle 需 Tier 1）。
