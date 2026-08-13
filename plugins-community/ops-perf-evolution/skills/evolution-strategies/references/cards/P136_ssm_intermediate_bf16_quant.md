---
id: P136
bottlenecks: [ub_memory_pressure, mte2_stall]
op_families: [ssm, linear_attention, recurrent, matmul]
complexity: L1
conflicts_with: []
synergizes_with: [P132, P133, P134, P135]
requires: [P132]
has_preconditions: true
has_playbook: true
---

# P136: SSM 中间量 bf16 量化 + 状态 UB 常驻（head grouping）

## 核心思想

三阶段 chunk-scan（P132）会物化大量中间量：`v_beta / q_prime / attn_1 / k_cumdecay / kg / v_new / attn_inter`，全部是 `(C×Dv)` / `(C×Dk)` / `(C×C)` 张量。若都按 fp32 存 GM/UB，UB 被中间量挤爆，每阶段都要多级搬进搬出，mte2 带宽成为瓶颈。**两个配套手段**：

1. **中间量全程 bf16**：所有块间传递的中间量（v_beta/q_prime/attn_1/k_cumdecay/kg/v_new/attn_inter）按 bf16 量化（对齐参考实现），只有求逆/衰减等**串行依赖的数学核心**保留 fp32 中间计算。UB 流量减半。
2. **状态 (Dv, Dk) UB 常驻跨 chunk 复用**：Stage2 的 state 只在小矩阵维度（Dv×Dk），一旦载入 UB 就在整个序列的 chunk 循环里复用，不反复搬 GM。
3. **head grouping**：`Nv % Nk == 0` 时每个 value head 读自己的 key head（`Nv//Nk` 分组），等价 `repeat_interleave`，消除 KV 冗余复制。

```
GDR 参考的量化边界（model.py）：
· bf16：v_beta, q_prime, attn_1, k_cumdecay, kg, v_new, attn_inter, final_state, out
· fp32：KKT 前 k/k^T 提升、求逆前代、decay 计算（串行依赖的数学核心保留 fp32）
```

**收益**：UB 占用减半 → 更多双缓冲槽位 → CV 流水（P133）重叠更充分；状态常驻消除每 chunk 的状态搬移；head grouping 减少 GM 读。

## 代码骨架

```cpp
// === 改造前：中间量全 fp32 / 状态每 chunk 搬 GM ===
// float v_beta[C*Dv]; float q_prime[C*Dk]; float state[Dv*Dk];
// 每 chunk: CopyIn(state, GM); 计算; CopyOut(state, GM);

// === 改造后：bf16 中间量 + 状态 UB 常驻 ===
using bhalf_t = bfloat16_t;
bhalf_t vBetaWs[C*Dv];   bhalf_t qPrimeWs[C*Dk];
bhalf_t attn1Ws[C*C];    bhalf_t kCumdecayWs[C*Dk]; bhalf_t kgWs[C*Dk];
bhalf_t vNewWs[C*Dv];    bhalf_t attnInterWs[C*Dv];
bhalf_t curState[Dv*Dk];                     // UB 常驻，跨 chunk 复用

for (cId = 0; cId < chunkNum; cId++) {       // Stage2：state 只在 UB 里递推
    // attn_inter = q_prime @ state^T；v_new = v_in + k_cumdecay @ state^T
    // state = state*decay + v_new^T @ kg    —— 全程 UB，不搬 GM
}
```

## 关键修改点

1. **bf16 边界划分**：把中间量按"块间传递 → bf16、串行依赖数学核心 → fp32"划分。GDR 参考：块间全 bf16，求逆/衰减的 fp32 中间计算保留。
2. **cast 时机**：cube 输出（L0C fp32）→ `DownCast` bf16 写 GM/进下一级；bf16 用 `CAST_RINT` 保留舍入。
3. **状态常驻**：`curState` UB 常驻，只在 batch 边界或序列切换时更新；确认状态规模 `Dv×Dk×2B` 不超 UB 预算。
4. **head grouping**：`Nv//Nk` 分组读取 KV（`repeat_interleave` 等价），消除冗余；host 侧算好 group 索引。
5. **尾块**：`lastChunkSize = seq % C` 特化，避免 pad 的伪计算吃满 C。

## 常见陷阱

- ❌ 中间量全 fp32 → UB 挤爆、mte2 翻倍，三阶段收益被带宽吃掉；必须按边界 bf16。
- ❌ 状态每 chunk 搬 GM → 状态 (Dv,Dk) 是最该常驻的，搬一次 GM 一次握手；常驻 UB。
- ❌ bf16 cast 用错舍入 → 精度退化；用 CAST_RINT。
- ❌ 把求逆/衰减核心也 bf16 → 串行依赖的数值稳定性崩；这些保留 fp32。
- ❌ 忽略 head grouping → Nv//Nk 冗余 KV 复制，GM 读翻倍。

## 代码搜索关键词

```bash
grep -nE "bfloat16|bf16|DownCast|CAST_RINT|v_beta|q_prime|attn_1|k_cumdecay|kg|v_new|attn_inter|curState|repeat_interleave|Nv.*Nk|head.*group" op_kernel/*.{cpp,h} model.py
```
