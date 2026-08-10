# Quant 量化算子验证标准

> 量化/反量化算子的精度判定。按输出 dtype 决定判定方式。

## 1. 五类决策矩阵

由 `verification_ascendc.py` 自动判定，量化相关三类：

| 类别 | 触发条件 | 判定标准 |
|------|---------|---------|
| 整数计算类 | 输入最高精度为 int 且输出为 int | `|actual − golden| == 0` 严格相等 |
| **量化计算类** | 输入最高精度为 float 且输出为 int | `|actual − golden| <= 1`（±1 LSB） |
| 浮点计算类 | 输出为 float 类型 | 三项 AND：max_error_cap + matched_ratio ≥ 0.9 + MERE < rel_threshold |

输入类型自动推断：从实际输入 tensor 取最高精度 dtype 分类为 float / int / no_tensor。

## 2. 量化输出（int8 / int16）判定

**元素级 max abs diff ≤ 1**，不是平均值——哪怕只有一个元素差 2，整张张量判 fail：

```python
INT_LSB_TOLERANCE = {torch.int8: 1, torch.int16: 1}
diff = (rhs.to(torch.int32) - lhs.to(torch.int32)).abs()
max_abs = diff.max().item()
is_pass = max_abs <= tol
```

## 3. 浮点输出（dequant 结果 / scale）判定

MERE / MARE 阈值（按输出 dtype）：

| dtype | MERE threshold | MARE threshold |
|-------|---------------|---------------|
| float16 | 9.77e-4（2⁻¹⁰） | 9.77e-3 |
| bfloat16 | 7.81e-3（2⁻⁷） | 7.81e-2 |
| float32 | 1.22e-4（2⁻¹³） | 1.22e-3 |

判定：`MERE < T` 且 `MARE < 10×T`（`ops-precision-standard/reference/quantization.md`）。

FP8 输出：e4m3 用 2⁻³≈0.125，e5m2 用 2⁻²≈0.25。

## 4. 混合输出算子

- **dynamic quant**：scale 输出（fp32）走 MERE/MARE，量化输出（int8）走 ±1 LSB——两个输出分别判定。
- **quant_matmul**：dequant 结果（fp16/bf16）走 MERE/MARE；int8 输出走 ±1 LSB。
- 分开判，不能混用。

## 5. 量化算子精度档与阈值匹配（重要）

| 计算链 | 实际精度 | 阈值行为 |
|--------|---------|---------|
| 只含 bit-exact op（Tier 1） | ~fp32 ulp | 过 fp32 阈值 |
| 含 hw 超越函数（Tier 2） | ~fp16 量级 | fp32 不可达；fp16 阈值可过 |
| 输出 int8/int16（Tier 3） | 量化阶梯离散 | 严格相等 fail；±1 LSB 可过 |

- 判定方法：机械扫 kernel 源码看是否调用 hw 超越函数（Exp/Sigmoid/Reciprocal/Sqrt…）。
- 带 fp32 输出（如 dynamic quant 的 scale）+ CPU fp32 reference → 必须走 Tier 1 软件 fp32 路径（不用 hw 超越函数），否则 MARE 锁死在 ~2⁻⁹。

## 6. 验证脚本

- `.claude/skills/tilelang2ascend-translator/scripts/verification_ascendc.py`
- 单标杆比对：NPU 输出 vs CPU fp32 golden（`scripts/mare_mere_threshold.py`）
- 禁止改验证脚本放宽阈值；禁止把 vendor kernel 当 oracle。
