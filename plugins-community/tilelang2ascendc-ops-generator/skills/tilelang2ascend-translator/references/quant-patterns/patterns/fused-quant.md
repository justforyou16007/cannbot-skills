# Pattern: 融合反量化 + 激活 + 量化（dequant → activation → quant）

## 先读这个

- dequant 精度档（Tier 1/2/3）与软件 fp32 路径：`@references/dequant_kernel_patterns.md`。
- 三段式 C/V 融合优化（背压、单 pass）：`ops/tilelang-perf-optimization/references/best-practices/gmm_cv_fusion_optimize.md`。
- 精度铁律：`@references/quant-patterns/quant-precision-rules.md`。

## 1. 识别条件

- `npu_dequant_swiglu_quant` / `npu_dequant_*` 类：dequant → activation（SiLU/SwiGLU/GELU）→ requant。
- MoE 的 gating/up/down projection 量化链路（`gate_up_proj` 输出 int32 + pertoken scale → `npu_dequant_swiglu_quant` → `down_proj`）。
- 输出 int8 + 每 token scale（供下一量化层）。

## 2. 决策轴选择

| 轴 | 取值 |
|----|------|
| 对象 | output（matmul 输出）|
| 粒度 | per-channel（weight scale）+ per-token（activation scale）|
| 对称性 | 对称 |
| scale 来源 | matmul 输出 pertoken scale + 权重 scale（static）+ requant scale（dynamic）|
| dtype | int8 |

## 3. 核心规则（三段式）

```text
[dequant]   acc(int32) × weightScale[col] × pertokenScale[row]
            （或 int8 matmul 直接输出反量化结果）
[activation] SiLU(x) = x·sigmoid(x) / SwiGLU(a,g) = Swish(a)·g
[requant]   per-token: scale = max(abs)/127；out = round(S/scale) → int8 + scale
```

**SwiGLU 特殊约束**：跨 N-half 门控（`+H` 配对列）时，必须按输出形状 `[M, H]` 调度、每输出块产出左右两路 C tile；stock QuantMatmulMultiStageWorkspace 每 slot 单 tile 无法跨 N-block 取 `+H`。

## 4. 实现要点（AscendC）

```
per row m:
  // dequant
  load acc int32 → Cast fp32 → Mul(weightScale[col]) → Muls(pertokenScale[m])
  // activation（Tier 1 软件 fp32 路径，若输出含 fp32 + CPU oracle）
  sigmoid = 1/(1+exp(-x))   // 软件实现：clamp[-50,50] → exp(-x) 7阶Horner → Newton 1/x
  // requant（per-token dynamic int8）
  Abs → ReduceMax → scale=max/127 → Muls(1/scale) → Cast f32→f16→s8
  // write
  DataCopy y_i8 + per-row scale
```

**软件 sigmoid（Tier 1 标准）**：
1. clamp x to [-50, 50]。
2. `k = round(-x·log2e)`（Cast fp32→int32 RINT）；`r = -x - k·ln2`。
3. `exp(-x) = poly(r)·2^k`（7 阶 Horner；`2^k` 用 `(k+127)<<23` ReinterpretCast）。
4. `1/(1+exp(-x))`：拆 mantissa `m∈[1,2)`，`y0=1.5-0.5·m`，Newton 3 步收敛。
- 禁止在带 fp32 输出的 Tier 1 路径用 hw `Exp/Sigmoid/Reciprocal/Sqrt/GeluV2/Div`（piecewise 近似锁死 MARE ~2⁻⁹）。

## 5. 优化点（C/V 融合）

| 优化点 | 说明 |
|--------|------|
| 三段单 pass | 全部在 UB 内链式完成，禁止 dequant→GM→读回→act→GM→读回→requant |
| 背压深度 | Cube 生产 workspace → Vector 消费，V→C"已消费"信号每 N 块发一次（N 用扫描法调优）|
| GEMM-bound | 天花板由 Cube 决定；epilogue 目标"不拖住 Cube"而非自身多快 |

## 6. 精度要点 + 检查点

- [ ] activation 用软件 fp32（Tier 1）时无 hw 超越函数。
- [ ] SwiGLU 跨 N-half 按 `[M,H]` 调度。
- [ ] requant 标量读/写同步（V_S / S_V）。
- [ ] `quant_scale` dtype 在 host 归一化（fp16/bf16→fp32）。
- [ ] mode-1 even/odd permute 时 weight_scale 必须用同一 permute 重排。

## 7. 参考实现

- `output/4_GroupedMatmulSwigluQuant/` — 完整 fused 产物（dequant+swiglu+quant）。
- `ops/tilelang-perf-optimization/references/best-practices/gmm_cv_fusion_optimize.md` — 性能优化。
- `gmm/grouped_matmul_swiglu_quant/` — a8w4 融合版本。
