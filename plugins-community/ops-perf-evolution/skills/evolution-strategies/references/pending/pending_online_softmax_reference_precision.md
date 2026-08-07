# [待验证经验] Online Softmax 可用性取决于参考实现的 softmax 精度路径

> **状态**: ⚠️ PENDING — 尚未确认是否为普遍现象，暂不落地到 P119 / flashattention.md
> **验证条件**: 出现第 2 个"参考是 FP16/BF16 softmax 且 online 精度失败"的 attention 算子时，即可确认并落地
> **来源**: 14_WindowAttention 进化（2026-08），w0 / w0_1 / 手写全 FP32 online 验证

---

## 现象

对 **WindowAttention**（Swin-style，含 relative position bias），尝试把 FP16/BF16 路径改造成 FlashAttention online softmax（P119）时，**无论怎么实现精度都失败**：

| 尝试 | 实现方式 | 结果 |
|---|---|---|
| w0 (Round 1) | SoftmaxFlashV2 官方 API + SplitFuse CV 流水 | 精度失败 → 回退基线 |
| w0_1 (Round 3) | 按 S 阈值 dispatch（S>384 走 online），FP32 running 状态 | 大 S case 仍失败 → 回退 full-row |
| 手写验证 | 完全手写 online（零官方 API）+ 全 FP32 中间操作 | 5 个大 S case 4 个失败，MERE 1e-3~2e-2 |

**对照**: 同一时期 **15_LinformerAttention** 用**手写 A5+P119 online softmax** 精度通过（仅继承一个 baseline 就有的 fp16 case-9 边界失败）。

## 根因

### 参考实现（model.py）的 softmax 精度路径不同

```python
# ① WindowAttention 型：参考在 FP16/BF16 上做 softmax → online 过不了
attn = q @ k.transpose(-2,-1)          # FP16 matmul
attn = F.softmax(attn, dim=-1)         # softmax 作用在 FP16 上（sub/P 有 T 舍入）

# ② Linformer 型：参考显式提成 FP32 softmax → online 能过
scores = torch.matmul(q.float(), k.float().transpose(-2,-1))  # 转 FP32
weights = F.softmax(scores / sqrt(d), dim=-1)                  # softmax 全程 FP32
output = torch.matmul(weights, v.float()).to(dtype=x.dtype)    # 最后才转回 T
```

### 舍入点对比（为什么 online 抹掉参考的舍入 → 精度失败）

参考 FP16 softmax 有 **4 个舍入点**：
```
S   = fp16(Q@Kᵀ·scale + bias)     ★R1 入 fp16
sub = fp16(S − m*)                ★R2 入 fp16，减整行 max
P   = fp16(p / l)                 ★R3 入 fp16
out = fp16(O)                     ★R4 入 fp16
```

Online 全 FP32 只剩 **2 个**：
```
S   = fp16(...)                   ★R1' 入 fp16
out = fp16(acc / l)               ★R4' 入 fp16
```

**差异集中在 R2、R3**：
1. **R2（sub 舍入）**: 参考在 fp16 下做 `S − m*`（减整行 max，舍入一次）；online 在 fp32 下做 `fp32(S_tile) − m_new`（减 running max，且不 round 到 fp16）。即使 online 强行 fp16 sub，减的 max 是 running 而非最终 max，舍入位置也不同。
2. **R3（P 舍入）**: 参考把 P 舍入回 fp16 再进 PV matmul；online 让 P 保持 fp32 进 PV，累加链不同。

全 FP32 online 是"更精确但与参考不相等" → MERE 1e-3~2e-2，超过 9.8e-4 阈值。

### 为什么 Linformer 能过

Linformer 的 model.py 参考本来就是 FP32 softmax（`q.float()` / `k.float()`），没有 R2/R3 这两处 fp16 舍入，所以 online 全 FP32 恰好和参考一致。

## 附带的独立发现

**官方 SoftmaxFlashV2 高阶 API 在 WindowAttention 上 MERE ≈ 1.9（bug）**；手写 A5+P119（ReduceMax→Max→Sub→Exp→ReduceSum→Broadcast→Div）修复到 MERE 1e-3~2e-2（算法正确但仍过不了参考阈值）。→ 手写手法本身是正确的，且优于官方 API。

---

## 拟修改位置（确认普遍现象后落地）

### 1. `references/scenario-routing/flashattention.md`
在 S1 阶段 + Online Softmax 实现要点中新增"第 0 步：参考实现精度路径判断"：
- 读 model.py，判断 softmax 输入是 fp32 还是 T
- 参考显式 `.float()` 提成 fp32 → online 可用
- 参考直接对 T 做 softmax → online 不可用，走 full-row safe softmax

### 2. `references/cards/P119_two_pass_to_online_softmax.md`
在"何时启用"一节前新增"参考实现精度路径"前置判断 + 判定表。

### 3. `references/playbooks/P119_two_pass_to_online_softmax.md`
在 Step 1.5（S 阈值判断）之前插入 **Step 1.4 参考实现精度路径判断**：
```
读 model.py：softmax 输入 scores/attn 是 fp32 还是 T？
- 参考 .float() 提成 fp32 → online 可用，继续 Step 1.5
- 参考直接对 T 做 softmax → online 不可用，停止 P119，
  改用 full-row safe softmax（逐 tile 复刻参考的 T 舍入）
```

## 判定表（拟写入）

| 参考 softmax 精度 | 舍入点 | Online 全 FP32 能否过 |
|---|---|---|
| FP32（参考显式 `.float()`） | 无 T 舍入 | ✅ 能过（与参考一致） |
| FP16/BF16（参考直接对 T softmax） | sub 入 T、P 入 T | ❌ 过不了 |

## 验证记录（证据链）

- 手写验证输出目录: `output/14_WindowAttention_evo2/test_handwritten_online/`
- 根因诊断脚本: `debug_kernel_vs_sim.py`（kernel vs 同算法 torch FP32 仿真 MERE ≈ 6e-5~1.7e-4，证明 kernel 无 bug）
- 对照算子: `output/15_LinformerAttention_evo/15_LinformerAttention_evo_20260804_120034/round_4/parallel_0/`
