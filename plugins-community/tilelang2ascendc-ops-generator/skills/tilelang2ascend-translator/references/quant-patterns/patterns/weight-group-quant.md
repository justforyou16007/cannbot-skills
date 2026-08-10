# Pattern: per-group 权重量化（a8w4 / SetQuantVector / MX）

## 先读这个

- 权重 per-group 量化：权重沿 K 方向每 G 元素一组一个 scale。**优先用高阶 Matmul API 的 `SetQuantVector` 硬件折叠**，避免手动反量化。
- int4 权重 + int8 激活（MSD 拆分）的实现见本文件 §4。
- MX（mxfp8/mxfp4）见 §5。

## 1. 识别条件

- 权重为 int4 / fp8 / MX（packed `fp4x2`），或权重 scale 是 per-group（`group_size` / `quantGroupNum`）。
- 算子名含 `a8w4` / `quant_matmul` + int4 权重 / grouped quant matmul。
- 激活可能是 int8（拆 int4）或保持 fp16。

## 2. 决策轴选择

| 轴 | 取值 |
|----|------|
| 对象 | **weight**（per-group scale）+ activation |
| 粒度 | per-group（K 方向每 G 元素，典型 G=32/64/128） |
| 对称性 | 对称（scale）为主；int4 非对称用 §2 公式 |
| scale 来源 | static（权重 scale 预设输入） |
| dtype | int4 / fp8_e4m3 / MX(fp8·fp4) |

## 3. 核心规则

```text
权重 W[N, K] 按 K 方向每 G 元素一组，scale 形状 [N, K/G]。
matmul 逐 K-group：C += A[:, g*G:(g+1)*G] @ (W[:, g*G:(g+1)*G] * scale[g])
```

**推荐实现**：硬件内折叠——`mm.SetQuantVector(perGroupScale)` 把权重 scale 直接喂给 cube，无需 AIV 手动乘。

## 4. 实现要点：a8w4 MSD（int8 激活拆两个 int4 + int4 权重）

### 4.1 激活拆分（AIV 前处理）

```cpp
// 高 4 位 = floor(x/16)
Cast(xHighHalf, xTensor, CAST_NONE, vK);        // int8→half
Muls(xHighHalf, xHighHalf, 0.0625f, vK);        // ÷16
Cast(xHighI4, xHighHalf, CAST_FLOOR, vK);       // floor → int4

// 低 4 位 = (x & 0x0F) - 8
Duplicate(xLowI16, 0x0F0F, 128);                // 掩码（int16 对）
And(xLowHalf, xTensor, xLowI16, ...);           // 提取低 nibble
Cast(xLowHalf2, xLowHalf, CAST_NONE, vK);
Adds(xHighHalf, xLowHalf2, -8.0f, vK);          // 映射到 signed [-8,7]
Cast(xLowI4, xHighHalf, CAST_NONE, vK);
```

- 拆分结果 `x = high×16 + low + 8`。
- 两半存入 workspace（M 轴 ×2），matmul 对 high/low 两半各算一次。

### 4.2 中处理（AIC，SetQuantVector）

```cpp
mm.SetSingleShape(curM, curN, quantGroupSize);   // 按 K 分组
mm.SetTensorA(xGM[...]);
mm.SetTensorB(weightSlice);
mm.SetQuantVector(weightScaleGM[group*N*qGroupNum + loopK*N + tailN]);  // per-group scale 硬件折叠
mm.Iterate();
mm.GetTensorC(mmOutGM[offset], loopK == 0 ? 0 : 1);
```

### 4.3 后处理（AIV）

```cpp
// 合并 high/low：high×16 + low
Muls(highF32, highF32, 16.0f, n);
Add(combined, highF32, lowF32, n);
// ±8 偏移修正：加辅助矩阵（预计算 -8·Σ(w·scale)）
Add(mmLocal, mmLocal, weightAuxiliaryMatrixLocal, n);
// × per-token scale → activation → requant（若融合）
```

**关键**：±8 偏移必须用辅助矩阵修正，漏修输出系统性偏差。

### 4.4 完整流水（MIX_AIC_1_2 软流水）

pre(AIV 拆分) ∥ mid(AIC matmul) ∥ post(AIV 合并/反量化/融合/重量化)，workspace 双缓冲 + `SyncAll`。

## 5. MX per-group（mxfp8/mxfp4）

```text
c[i,j] = Σ_g scaleA[i,g] × scaleB[g,j] × Σ_k' a[i,g·G+k'] · b[g·G+k', j]   # G=32
```

| 维度 | MXFP8 | MXFP4 |
|------|-------|-------|
| C++ 类型 | `fp8_e4m3fn_t` | `fp4x2_e2m1_t`（packed） |
| 硬件 C0 | 32 | 64 |
| scale | fp8_e8m0（K 方向每 32） | 同 |

- 用 `MmxMmad`/`MxMatmul` 或 Blaze `MatmulWithScaleMx` 路径。
- scale 必须沿被消费 matmul 的 **K-mmad 轴**量化；V 在 BSND 布局下 D 是 innermost 但 scale 沿 S_k（反直觉陷阱）。
- cast 数值公式保证 `amax/scale` 不超量化 dtype max，否则 RINT 出 NaN。
- 参考：`ops/ascendc-blaze-best-practice/.../mx-matmul-development.md`、`ops/ascendc-tiling-design/.../flashattention/subfamilies/quantization_design.md`。

## 6. 精度要点 + 检查点

- [ ] 权重 scale 沿 K-mmad 轴对齐（MX）。
- [ ] int4 拆分 ±8 偏移已修正。
- [ ] `SetQuantVector` 索引正确（`group*N*qGroupNum + loopK*N + tailN`）。
- [ ] 激活拆分高/低位写入 workspace 的偏移正确（M 轴 ×2）。
- [ ] cast 数值不溢出（`amax/scale` ≤ dtype max）。

## 7. 参考实现

- `gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_a8w4_msd_{pre,mid,post}.h` — 完整 MSD 三阶段。
- `ops/ascendc-blaze-best-practice/references/scenarios/mx-matmul-development.md` — MX。
