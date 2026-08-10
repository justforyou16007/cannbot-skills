# Quant 量化公式族

> 所有量化/反量化算子的公式选择。分母（127/255/15）取决于量化格式与数据分布，**不是通用的 127**，必须按 model.py 语义选择。

## 1. 对称量化（symmetric）

```text
scale = max(|x|) / Qmax            # Qmax: int8=127, int4=7, fp8 e4m3=448...
q     = round(x / scale)            # clamp 到 [Qmin, Qmax] 防溢出
```

- 反量化：`x = q * scale`。
- 典型：int8 per-token 动态量化（`dynamic_quant`）、int8 matmul 输出重量化。

## 2. 非对称量化（asymmetric，带 offset / zero-point）

```text
int8:  scale  = (max - min) / 255
       offset = -128 - round(min / scale)
       q      = round(x / scale + offset)
int4:  scale  = max((max - min) / 15, eps)     # int4 表示范围 [-8,7]，分母 15 非 7
       offset = 7 - max / scale
       q      = trunc((x - min) / scale + 0.5)，clamp [0,15]
```

- 反量化：`x = (q - offset) * scale` 或 `x = q*scale + zero_point`（不同 API 的 offset 定义相反，以 model.py 为准）。
- `eps`（如 1e-12）：避免 `max == min` 时除零；用 `SafeDiv(a, b)`（`|b| < eps` 时返回 a）。
- 典型：KV-cache 量化（`TurboQuantStore` 用 `(v_max - v_min)/15` + clamp [0,15]）、int4 weight。

## 3. 动态 range 量化（dynamic）

```text
per-token:  scale[i] = max(abs(x[i,:])) / Qmax      # 每行独立
per-row 最大时用 reduce:  Abs → ReduceMax → 标量读
```

- 与静态量化唯一区别是 scale 在 kernel 内计算（`Abs` + `ReduceMax` + `GetValue`），不是预设。
- 标量读必须 `SetFlag/WaitFlag<HardEvent::V_S>`（见 `quant-precision-rules.md`）。

## 4. FP19 / FP9 硬件格式（NPU 量化指令专用）

- NPU 量化指令用 **FP19**（1 符号 + 9 指数 + 9 尾数）表示 scale，**Int9**（[-256,255]）表示 offset，两者打包进一个 `uint64`。
- `scale` 截断：fp32 scale 按 `0xFFFFE000` 截断到 19 位有效位，模拟 `npu_trans_quant_param`（条件：fp32 scale、无 pertoken、输出非 bf16/int32）。
- offset 必须限制在 `[-256, 255]`：`Cast(offsetI32, offsetF32, CAST_RINT)` → `Maxs(MIN_INT9)` → `Mins(MAX_INT9)`。
- 打包：`quantPre = (fp19Scale << 9) | QUANT_SCALE`；offset 拼 `(int9 << 37)`。
- 参考实现：`cake-evo/.../acc_08_quant_precision.md` Variant E/F/G。

## 5. per-group 权重量化

```text
权重 [N, K] 按 K 方向每 G 元素一组，每组一个 scale [N, K/G]。
matmul 时逐 K-group 乘对应 scale。
```

- **推荐**：用高阶 Matmul API `SetQuantVector(perGroupScale)` 硬件内折叠（见 `weight-group-quant.md`）。
- 手动实现：把 scale 沿 K 维展开成 `[N, K]` 再逐块乘，或拆成 per-group 的 GEMM 循环。

## 6. 反量化（dequant，int→fp）

```text
int32 acc
  → [int32 bias]  acc += bias        (int32 域, scale 之前)
  → Cast int32 → fp32                (CAST_NONE)
  →               acc *= scale[col]  (列广播)
  → [pertoken]    acc *= pertoken[row] (行广播, scale 之后)
  → [float bias]  acc += bias        (fp32 域, scale 之后)
  → [offset]      acc += offset
  → Cast fp32 → out
       fp16:   CAST_NONE
       bf16/int32: CAST_RINT
       int8:   clamp[-128,127] 后 f32→f16→s8 两步 RINT
```

**乘法顺序必须是 `(acc × scale) × pertoken`**（scale 先、pertoken 后）；反序有 1-ULP 偏差（对拍 torch_npu 实证）。

## 公式选择要点

| 场景 | 用 |
|------|----|
| 对称 int8 输出（matmul 重量化 / dynamic_quant） | §1，`Qmax=127` |
| 非对称 int4（KV-cache / weight） | §2，分母 15 |
| 非对称 int8 | §2，分母 255 |
| NPU 量化指令参数预处理 | §4（FP19 + Int9） |
| 权重 per-group | §5，优先 `SetQuantVector` |
| dequant 输出高精度 | §6，顺序与 cast 铁律 |
