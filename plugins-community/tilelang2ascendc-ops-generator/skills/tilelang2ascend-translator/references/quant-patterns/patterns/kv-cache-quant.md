# Pattern: KV Cache 量化（int4/int8 存储 + scale cache）

## 先读这个

- KV cache 量化的 kernel 侧实现（int4 量化 + 打包 + scale 存储）：本文件。
- FA 侧反量化契约（scale 沿 K-mmad、量化 dtype 编译宏、V2 末块量化链）：`ops/ascendc-tiling-design/references/flashattention/subfamilies/quantization_design.md`。
- 模型级 KV cache 量化边界（scale 来源、post-load、cache scale 传递）：`model/model-infer-quantization/references/quantization-structure-cards.md` 结构卡 C。

## 1. 识别条件

- 输出写 KV cache（int8/int4）+ 每 token/每 head 的 scale（或 scale cache）。
- `k_proj.output_scale` / `v_proj.output_scale` / `get_cache_scale()` / `kv_cache_scheme`。
- FA / sparse FA 消费量化 cache + scale（`key_quant_mode` / `value_quant_mode` / `quant_scale_repo_mode`）。
- 独立算子形态：KV-cache store 算子（如 TurboQuantStore）或 FA 量化路径。

## 2. 决策轴选择

| 轴 | 取值 |
|----|------|
| 对象 | **KV-cache**（activation 视角） |
| 粒度 | per-token / per-head / per-block |
| 对称性 | 对称（int8 `max/127`）或非对称（int4 `(max-min)/15` + min） |
| scale 来源 | dynamic（写入时算）为主；FA 侧读 scale cache |
| dtype | int8 常见；int4 需 4-bit 打包（2 元素/byte） |

## 3. 核心规则（int4 KV-cache store，非对称动态 range）

```text
per (token, head) row:
  v_min = min(v), v_max = max(v)                        # scalar reductions
  v_scale = max((v_max - v_min) / 15.0, 1e-8)           # int4 分母 15
  q = trunc((v - v_min) / v_scale + 0.5), clamp [0,15]  # int4
  pack q 成 4-bit pairs (2 元素/byte)                    # packed_val [D/2] uint8
  store v_scale, v_min -> fp16 LE（scale cache）
```

- 对称 int8 变体：`scale = max(abs(v))/127`，无 min。
- 写回路径常含 `slot_mapping` / `block/offset` 的 scatter 寻址。

## 4. 实现要点（AscendC）

### 4.1 纯 KV-store 算子（纯 AIV，如 TurboQuantStore）

```
per row（每核处理连续 NH 行）:
  load v[D] fp32
  ReduceMin + ReduceMax（标量，V_S 同步）
  scale = (max-min)/15.0 clamp ≥ 1e-8；min 存 fp16
  Sub(min) → Div/Muls(1/scale) → Add(0.5) → trunc → clamp[0,15] → int4
  pack：每 2 个 4-bit 拼 1 byte（位操作 / And+Shift）
  write packed_val + v_scale + v_min 到 kv_cache[blk, off, head, :D+6]
```

- 4-bit 打包：`packed[i] = (q[2i] & 0xF) | ((q[2i+1] & 0xF) << 4)`，用向量 And/Shift/Or 而非逐元素循环。
- 整算子是 per-row 独立的 vector 任务，block 分连续行块。

### 4.2 FA 反量化侧（消费量化 cache）

- 读 int8/int4 cache + scale cache，进 FA 前 dequant（或 FA 内反量化）。
- scale 必须沿被消费 matmul 的 **K-mmad 轴**（V 在 BSND 下 D 是 innermost，但 scale 沿 S_k，反直觉陷阱）。
- 量化 FA 编译宏分离（MXFP8/INT8/AntiQuant typedef 与 Matmul 模板参数）。
- V2 末块量化链：`ReduceMax → scale=amax/quant_max → Cast<quant_dtype,float>(O/scale)`，禁止 cast 溢出（RINT→NaN）。

## 5. 精度要点 + 检查点

- [ ] int4 用分母 15 + clamp[0,15]（非 127）。
- [ ] `min` 用 fp16 存（2B），与 scale 一起写入 scale cache。
- [ ] 4-bit 打包用向量位操作，无逐元素循环。
- [ ] scale 与 cache 的 scatter/reorder（CP/offload）正确。
- [ ] 量化 cache 是否真实被 FA 消费（不能只凭 dtype 判定 C8 生效）。

## 6. 参考实现

- KV-cache int4 量化 store（TurboQuantStore 类，4 个变体）已有完整验证产物。
- `ops/ascendc-tiling-design/references/flashattention/subfamilies/quantization_design.md` — FA 反量化契约。
- `model/model-infer-quantization/references/quantization-structure-cards.md` 结构卡 C — 模型级 cache scale 边界。
