# Quant 精度铁律

> 量化/反量化算子 D 类（精度不匹配）修复时的优先排查清单。大部分规则来自对拍 torch_npu / CPU fp32 的实证，**不要照抄 model.py 注释里的 native 参考实现**，以对拍实验为准。

## 排查顺序（D 类 quant 失败按序检查）

### ① 乘法顺序（最高频根因）

```text
必须是 (acc × scale) × pertoken     # 列 scale 先、行 pertoken 后
```

- torch_npu.npu_quant_matmul 实测该顺序 bit-exact；反过来（pertoken 先）有 1-ULP 偏差。
- model.py 注释中 `acc × pertoken × scale` 的 native 参考是错的。
- 印证：`api-gmm.md` §3 的 per-token dequant 同为 `Mul(scale 列) → Mul(pertoken 行)`。

### ② bias 位置

- **int32 bias**：必须在 scale **前**加（int32 域），`Add` 后 `Cast` 到 fp32。
- **float bias**：必须在 scale **后**加（fp32 域）。
- 若 float bias 以"加完再 cast"的 post-add 实现，在 bias 与 acc×scale 相消处相对误差可达 MERE 4~10，远超容差（6_QuantMatmul 全量验证实证）。

### ③ 输出 cast RoundMode

- fp16 → `CAST_NONE`；**bf16 / int32 → 必须 `CAST_RINT`**（CAST_NONE 出垃圾）。
- **int8 无 f32→s8 直接指令**：先 `Mins(127.0f)` + `Maxs(-128.0f)` clamp，再走 **f32→f16→s8 两步 RINT**（round-half-to-even）。
- `Cast` 溢出（amax/scale 超量化 dtype max）→ RINT 产出 NaN，禁止该反模式。

### ④ scale 预处理（匹配 NPU 硬件）

- fp32 scale 需按 `0xFFFFE000` 截断到 19 位有效位模拟 `npu_trans_quant_param`（条件：fp32 scale、无 pertoken、输出非 bf16/int32）。
- 否则 dequant 结果与 torch_npu 有系统性偏差。

### ⑤ 标量读/写同步（dynamic quant）

- **读 reduce 结果**：`ReduceMax` 后必须 `SetFlag/WaitFlag<HardEvent::V_S>` 再 `GetValue(0)`，否则拿到过期 max。
- **写回标量后回 V 核**：同样要 `S_V` 同步。
- 标量写 GM 建议走 UB + `DataCopyPad`（`GlobalTensor::SetValue` 直写指令与 vector pipeline 不同通道，host 读回不保证落盘）。

### ⑥ V→MTE2 WAR 屏障

- dequant 中 scale/bias/offset 等 staging buffer 复用前，前一轮 vector 可能仍在读 → **需 `PipeBarrier<PIPE_ALL>`**（910B2C / CANN 9.0 实测 PIPE_V 不足，恰好在 rowsPerCore 边界 tile 确定性用错列向量）。
- 触发场景：N>2048 列分块、per-token 行边界。

### ⑦ 归约与除法精度

- `max(|x|)` 应在 **fp32 域**用 `Abs` + `ReduceMax<float>` 完成；用输入 dtype 直接 reduce 丢 1 ulp，导致 `scale=max/127` 偏差。
- `Div`（向量除）与 `Muls`（标量乘 1/scale）在 fp32 ulp 有 1-2 位漂移；需严格匹配 reference 除法语义时用 `Div`（scaleVec 由 `Duplicate` 填充）。
- bf16→fp32 / fp16→fp32 均无损，用 `CAST_NONE`。

### ⑧ int4 拆分偏移

- a8w4 MSD 拆分：`high4 = floor(x/16)`，`low4 = (x & 0x0F) - 8`；重构 `x = high×16 + low + 8`。
- **±8 偏移必须用辅助矩阵（weightAuxiliaryMatrix）修正**，在 post 阶段 `Add` 加回；漏修则输出系统性偏 `-8·Σ(w·scale)`。

## 反 hacking 清单（禁止行为）

- ❌ 改验证脚本放宽 fp32 阈值 / 加 op 白名单。
- ❌ 改 `model.py`（reference）适应 NPU 精度。
- ❌ 用形状特化、magic constant、benchmark-aware 输入让 case 过。
- ❌ 用 `torch_npu.npu_*` vendor kernel 当 oracle 让两端互相 cancel（vendor kernel 自身可能带 bug）。
- ❌ kernel 里用 hw 超越函数后声称是 Tier 1 用 fp32 阈值评判。
- 诚实路径：**输出 dtype + reference 精度档 + 计算链分类** 三者一致（详见 `dequant_kernel_patterns.md`）。

## 量化算子的精度档（判定基准）

| 档 | 计算链特征 | 判定 |
|----|-----------|------|
| Tier 1 | 只用 bit-exact op（Mul/Add/Cast/ReduceMax/Mins/Maxs…） | fp32 阈值可达 |
| Tier 2 | 含 hw 超越函数（Exp/Sigmoid/Reciprocal…） | fp16 阈值评判 |
| Tier 3 | 输出 int8/int16 | ±1 LSB（`max_abs_diff ≤ 1`） |

带 fp32 输出 + CPU oracle 的量化算子（如 dynamic quant 的 scale 输出），只能走 Tier 1 软件 fp32 路径。
