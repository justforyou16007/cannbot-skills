# Pattern: 量化 MatMul（int8/int4 GEMM + dequant epilogue）

## 先读这个

- 反量化核心规则（顺序 / bias / cast）→ `@references/quant-patterns/quant-precision-rules.md`。
- 公式与 scale 粒度 → `@references/quant-patterns/quant-formulas.md` §6。
- 平台配方：**910B2C / CANN 9.0 必须走"纯 Cube 高阶 Matmul API + 纯 AIV dequant 双 kernel"**，手写 Mmad（LoadData3D/Mmad/Fixpipe + CrossCore ring）在本工具链输出垃圾（one-hot probe max_diff=129 实证）。

## 1. 识别条件

- `torch_npu.npu_quant_matmul(x1, x2, scale, offset, pertoken_scale, bias, output_dtype, group_sizes)`。
- 输入 x1/x2 为 int8（已量化），输出高精度（fp16/bf16/int32）或再量化（int8）。
- 多输入：x1, x2, scale（必要）+ offset / pertoken_scale / bias（可选）。
- 注意：本算子**输入已量化**，算子内只做 int8 GEMM + dequant，**不做 fp→int8 输入量化**。

## 2. 决策轴选择

| 轴 | 取值 |
|----|------|
| 对象 | 激活 × 激活（matmul），输出 dequant |
| 粒度 | scale 通常 per-channel `[N]`；pertoken_scale per-token `[M]`；bias 可选 |
| 对称性 | 对称为主；非对称有 offset |
| scale 来源 | static（scale 输入已给）；pertoken static |
| dtype | int8×int8 → int32 acc → 反量化 |

## 3. 核心规则（反量化顺序，bit-exact 铁律）

```text
int32 acc
  → [biasMode==int32]  acc += bias      (int32 域, scale 之前)
  → Cast int32 → fp32                   (CAST_NONE)
  →                     acc *= scale[col]     (列广播)
  → [hasPertoken]       acc *= pertoken[row] (行广播, scale 之后)
  → [biasMode==float]   acc += bias      (fp32 域, scale 之后)
  → [hasOffset]         acc += offset[col]
  → Cast fp32 → out
       fp16:   CAST_NONE
       bf16/int32: CAST_RINT
       int8:   clamp[-128,127] 后 f32→f16→s8 两步 RINT
```

**必须 `(acc × scale) × pertoken`**（scale 先、pertoken 后），反序 1-ULP 偏差。

## 4. 实现要点（AscendC）

### 平台选型

| 平台 | 路径 | 说明 |
|------|------|------|
| 910B2C / CANN 9.0 | **纯 Cube 高阶 API + 纯 AIV dequant 双 kernel** | 正确性唯一可用路径（见下） |
| A2/A3 (DAV_2201) | MatmulImpl + AIV 反量化 | 见 `ascendc-api-best-practices/references/api-gmm.md` §3 |
| A5 (DAV_3510) | Blaze MIX 路径（`a8w8` 场景） | `ops/ascendc-blaze-best-practice/.../a8w8-quant-matmul-development.md` |

### 910B2C 纯 Cube + 纯 AIV 双 kernel 配方（已验证）

**Cube kernel**（`quant_matmul_mm.cpp`，int8×int8→int32 写 GM workspace）：
- `#define ASCENDC_CUBE_ONLY`；`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`。
- AIV 提前 return；AIC prologue：`SetAtomicNone/SetMaskNorm/SetLoadDataBoundary(0)/SetLoadDataPaddingValue(0)`。
- `SetSysWorkspaceForce(kfcWorkspace)`（host `at::zeros` 分配 KFC workspace）。
- `Matmul<A,B,C,BIAS,CFG_NORM>` + `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mmObj, &cubeTiling)`。
- 多核切分：`blockIdx → (mIdx,nIdx)`，`SetSingleShape(curM,curN,K)`，`IterateAll(cGM)`。
- 每 kernel 入口独立 `ascendc_library` target（4 dtype 拆 4 个 .cpp）。

**Dequant kernel**（纯 AIV，`dequant_epilogue.h`）：
- 顺序启动在 Cube kernel 之后（同 stream 保序），读 GM int32 workspace。
- 列向量（scale/bias/offset/pertoken）每列块只 load 一次常驻 UB。
- `Cast` 链 + 广播：列广播逐行 `Mul`；行广播 `Broadcast<2,1>` 后 `Mul`。
- **staging 前 `PipeBarrier<PIPE_ALL>`**（N>2048 列分块时 V→MTE2 WAR，PIPE_V 不足）。

### host/tiling

- `MatmulApiTiling::GetTiling` 或 `MultiCoreMatmulTiling`；scale 需 19-bit 截断预处理（见 precision-rules ④）。
- 输出 int8 时 clamp 后再两步 cast。

## 5. 精度要点 + 检查点

- [ ] 乘法顺序 `(acc×scale)×pertoken`（对拍 torch_npu）。
- [ ] int32 bias 在 scale 前 / float bias 在 scale 后。
- [ ] bf16 输出 CAST_RINT；int8 输出 clamp + f32→f16→s8 两步。
- [ ] fp32 scale 19-bit 截断（`0xFFFFE000`，无 pertoken 且输出非 bf16/int32 时）。
- [ ] 列向量常驻 UB + `PipeBarrier<PIPE_ALL>`（N>2048）。
- [ ] Cube 手写 Mmad 反模式：**禁止**在 910B2C 使用。
- [ ] 多 kernel 入口：每 dtype 独立 ascendc_library target，禁止多入口合单 .cpp（`rtRegisterAllKernel` 失败）。

## 6. 优化点（性能）

| 优化点 | 假设 | 验证 |
|--------|------|------|
| 双 kernel 无 CV 并行 | 性能天花板低（6_QuantMatmul geomean 0.34x） | msprof |
| 单 kernel 软流水（MIX_AIC_1_2 + workspace 双缓冲） | 有 CV 重叠但复杂 | 参照 `gmm/grouped_matmul_swiglu_quant` |
| Split-K / 大 K | K≫M×N 时 | roofline |
| 自适应 tile（bm/bn/bk 按 grid/pad 打分） | 小 shape 利用率低 | geomean 对比 |

## 7. 参考实现

- `output_test_830_goodcase_L4/6_QuantMatmul/` — 完整可用实现（Cube kernel + dequant_epilogue.h + host tiling + 4 dtype 入口）。
- `ops/ascendc-blaze-best-practice/references/scenarios/a8w8-quant-matmul-development.md` — A5/Blaze MIX 路径与 4 量化模式。
- `ops/ascendc-api-best-practices/references/api-gmm.md` §3 — GMM A8W8O16 per-token dequant 代码模板。
