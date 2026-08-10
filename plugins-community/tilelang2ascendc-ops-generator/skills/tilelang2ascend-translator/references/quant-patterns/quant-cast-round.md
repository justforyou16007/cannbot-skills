# Quant Cast 能力矩阵与 RoundMode

> Cast 是量化算子的核心操作。本表整理 fp32↔整数、fp32↔浮点的能力与 RoundMode 语义。以 Atlas A2 / CANN 实测为准，不同 SoC/CANN 以官方 API 文档为准。

## 1. Cast 能力矩阵

### fp32 → 整数

| 目的类型 | 支持的 RoundMode |
|---|---|
| int32 / int64 / int16 | RINT / FLOOR / CEIL / ROUND / TRUNC |
| **int8** | **不支持直接 fp32 → int8 cast，必须走 fp16 中转** |

### fp32 → 浮点

| 目的类型 | 支持的 RoundMode |
|---|---|
| half | NONE / ODD（NONE 有损时 = RINT） |
| bfloat16 | RINT / FLOOR / CEIL / ROUND / TRUNC |

### half / bfloat16 → 整数

| 源 → 目的 | 支持的 RoundMode |
|---|---|
| half → int32 | RINT / FLOOR / CEIL / ROUND / TRUNC |
| half → int16 | RINT |
| half → int8 | FLOOR / CEIL / ROUND / TRUNC / NONE |
| bf16 → int32 | RINT / FLOOR / CEIL / ROUND / TRUNC |

### 不存在的 Cast 重载

| 形式 | 状态 |
|------|------|
| `Cast<int8_t, int32_t>` / `Cast<int8_t, fp32>` / `Cast<int16_t, int32_t>` | 不存在（编译失败） |

需要时组合可用 cast（fp16 中转、int16 中转）。

## 2. RoundMode 语义

| Mode | 含义 |
|------|------|
| `CAST_NONE` | 有精度损失时 = `CAST_RINT`；无损失时不舍入 |
| `CAST_RINT` | 四舍六入五成双（IEEE round-half-to-even / banker's） |
| `CAST_FLOOR` | 向 -∞ 舍入 |
| `CAST_CEIL` | 向 +∞ 舍入 |
| `CAST_ROUND` | 四舍五入（远离零） |
| `CAST_TRUNC` | 向零舍入 |
| `CAST_ODD` | Von Neumann，最近邻奇数舍入 |

## 3. fp32 → int8 两条路径

### 路径 A：fp32 → fp16 → int8（推荐，与硬件 quant 单元更接近）

```cpp
Cast(yFp16, yFp32, RoundMode::CAST_NONE, count);  // fp16 lossy → 自动 RINT
Cast(yI8,   yFp16, RoundMode::CAST_NONE, count);  // half→int8 → RINT
```

精度注意：fp16 的 ulp 会把接近整数 .5 的 fp32 值吸到可表示的半步上，影响最终 int8 舍入。需要更精确时先用 `Mins/Maxs` clamp 到 [-128,127] 再两步 RINT。

### 路径 B：fp32 → int16 → host narrow（精度最高但 GM 翻倍）

```cpp
Cast(yI16, yFp32, RoundMode::CAST_RINT, count);   // 直接 banker's 入整
```
host：`y_int16.to(torch.int8)`（低字节即 int8）。中间精度全程 fp32，无 fp16 ulp 干扰，但 int16 GM 占用 2×。

## 4. 量化常用 Cast 决策速查

| 用途 | Cast | RoundMode |
|------|------|-----------|
| int32 acc → fp32（dequant） | `Cast(fp32, i32, CAST_NONE)` | CAST_NONE |
| fp32 → fp16（dequant 输出） | `Cast(half, fp32, CAST_NONE)` | CAST_NONE |
| fp32 → bf16（dequant 输出） | `Cast(bf16, fp32, CAST_RINT)` | **必须 CAST_RINT**（CAST_NONE 出垃圾） |
| fp32 → int8（quant 输出） | `Cast(half, fp32, CAST_RINT)` → `Cast(i8, half, CAST_RINT)` | RINT 两步 |
| fp32 → int16（intermediate） | `Cast(i16, fp32, CAST_RINT)` | CAST_RINT |
| fp32 → int32（dequant 输出） | `Cast(i32, fp32, CAST_RINT)` | CAST_RINT |
| 动态 range 边界归约 | `ReduceMax`（fp32 域） | — |

## 5. Cast 常见坑

- **Cast 起始地址必须 32B 对齐**：`AllocTensor` 默认满足；`TBuf::Get<T>()` 从 buffer 起始对齐。
- **src/dst 不能同一物理 buffer 不同元素类型**（in-place dtype conversion 不稳定，会出 garbage）：让 dst 落到当前未使用的 buffer 上的新 view。
- **int8 输出必须 clamp 后再 cast**：`Mins(127.0f)` + `Maxs(-128.0f)`，否则溢出（cast 溢出 → RINT 产出 NaN）。
- **bf16 输出用 CAST_RINT**：CAST_NONE 对 bf16 会产生 garbage（6_QuantMatmul 实证）。
