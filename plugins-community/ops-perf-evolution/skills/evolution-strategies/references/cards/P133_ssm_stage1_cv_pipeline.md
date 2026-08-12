---
id: P133
bottlenecks: [no_overlap, partial_overlap, compute_bound]
op_families: [ssm, linear_attention, recurrent, cv_fusion, matmul]
complexity: L2
conflicts_with: []
synergizes_with: [P132, P134, P135, P136]
requires: [P132]
has_preconditions: true
has_playbook: true
---

# P133: 块内预计算 AIC/AIV CV 流水（cube 算 KKT/求逆、vector 算 cumsum/exp/前代，6-flag 定向握手）

## 核心思想

P132 三阶段分解后，Stage1 块内预计算是**多级 matmul + 向量预处理交错**的序列：prep → KKT/QKT → cumsum/exp → 求逆 → gBK/VBeta → k_cumdecay/v_inner → q_prime。若每级之间用 `SyncAll` 全局屏障，AIC（cube）和 AIV（vector）会互相停等，cube 利用率腰斩——这正是 FA 拆分前的问题，在 SSM 块内预计算同样出现。

**C/V 分工 + 定向 flag 握手**把块内预计算改成双核流水：

```
V0a prep (vector) ──0x9──▶ C1 KKT/QKT (cube) ──0x8/0xA──▶ V1a 求逆对角前代 (vector)
                            │                        └──▶ V3 q_prime (vector)
V1a ──0x7──▶ C2a off-diag 求逆 (cube) ──▶ V2 gBK/VBeta (vector)
V2 ──0x6/0x5──▶ C2b k_cumdecay / v_inner (cube)
```

- **cube（AIC）干矩阵运算**：KKT/QKT（C1）、off-diag 求逆 MM（C2a）、k_cumdecay/v_inner（C2b）——全是大 matmul，cube 吞吐。
- **vector（AIV）干标量/逐元素**：prep（V0a）、cumsum+exp（V0b）、求逆对角前代（V1a）、gBK/VBeta（V2）、q_prime（V3）——全是串行依赖/逐元素，vector 擅长。
- **6 个 `CrossCoreSetFlag/WaitFlag` 定向握手**（事件号 0x9/0x8/0xA/0x7/0x6/0x5）替代 `SyncAll`：AIC 只在真需要时才等 AIV 的产物，反之亦然，重叠最大化。

**收益**：块内预计算的串行 matmul-向量链从"每级全核屏障"变成"定向握手"，AIC/AIV 同时工作，Stage1 时间接近 max(cube, vector) 而非两者之和。

与 P121（FA SplitFuse）的关系：P121 是 softmax 注意力 QK/softmax/PV 四阶段的 CV 流水；本卡是**线性递归块内预计算**（KKT/求逆/kcumdecay 等）的 CV 流水——同样的握手拓扑，不同的阶段内容。无求逆的 GLA 用 cumprod 衰减替代 C2a，流水骨架不变。

## 代码骨架

```cpp
// === 改造前：块内预计算每级 SyncAll（AIC/AIV 互相停等）===
// 每算一级都要等全核，cube 空转在向量级上

// === 改造后：C/V 分工 + 定向 flag 握手 ===
void ParaChunkAIC() {                        // cube 侧
    CrossCoreWaitFlag(0x9);                  // 等 V0a prep 写完
    for (i...) AICProcess(key, key, kktWs);  // C1: KKT
    for (i...) AICProcess(q, key, qktWs);    // C1: QKT
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);    // 通知 V1a 可算前代
    CrossCoreSetFlag<0x2, PIPE_FIX>(0xA);    // 通知 V3 可算 q_prime
    CrossCoreWaitFlag(0x7);                  // 等 V1a 对角前代写完
    for (i...) AttnInverseMMCompute(...);    // C2a: off-diag 求逆（cube）
    CrossCoreWaitFlag(0x6); CrossCoreWaitFlag(0x5);  // 等 V2 gBK/VBeta
    for (i...) AICProcess(attn, gBK, k_cumdecay);    // C2b
    for (i...) AICProcess(attn, vBeta, v_inner);      // C2b
}

void ParaChunkAIV() {                        // vector 侧
    QKPreProcess(q, ws); QKPreProcess(k, ws);   // V0a prep
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9);      // 通知 C1 可算
    GCumExpCompute(g, g_cum);                   // V0b: cumsum+exp
    CrossCoreWaitFlag(0x8);                     // 等 C1 KKT
    InverseCompute(attnWs, gamma);              // V1a: 对角前代
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x7);      // 通知 C2a 可算 off-diag
    GBKCompute(gBKWs, kg, beta, gCum, ...);     // V2
    VBetaCompute(vBetaWs, v, beta, ...);        // V2
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6);      // 通知 C2b 可算 k_cumdecay
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);      // 通知 C2b 可算 v_inner
    CrossCoreWaitFlag(0xA);                     // 等 C1 QKT（stage1 完成）
}
```

## 关键修改点

1. **识别块内预计算的 C/V 依赖链**：列出每级的产物和消费方，标出谁等谁。GDR 的链是 `V0a→C1→{V1a→C2a→V2→C2b} , V3`（依赖见上文）。
2. **flag 事件号分配**：每对 (生产者, 消费者) 一个事件号，编号避开其他 stage 用的号（GDR：stage1 用 0x9/0x8/0xA/0x7/0x6/0x5，stage2 用 6/8/10，stage3 用 9/7，避免冲突）。
3. **`SetFlag` 带 PIPE 参数**：`CrossCoreSetFlag<0x2, PIPE_FIX/MTE3>` 指定在哪个流水阶段后发信号，保证数据已落 GM。
4. **phase=bwd 时链条不同**：反向块内预计算是 dQ/dK/dV 链，同样适用 C/V 分工 + 握手（见 flashattention.md phase trait）。
5. **GLA 差异**：无 C2a 求逆，chain 为 `V0a→C1(KKT)→V1a(衰减累积)→V2→C2b`，去掉 0x7 那条握手。

## 常见陷阱

- ❌ 保留 `SyncAll` 做每级同步 → AIC/AIV 互相停等，cube 利用率腰斩；必须改成定向握手。
- ❌ flag 事件号与 stage2/3 冲突 → 握手错乱；每个 stage 用独立事件号区间。
- ❌ `SetFlag` 不带 PIPE 参数 → 信号可能在数据还没落 GM 时就发，消费方读到脏数据。
- ❌ 把求逆前代（串行依赖）塞进 cube → 前代是逐行 forward substitution，vector 更快；cube 只做 off-diag MM（见 P135）。
- ❌ 无求逆的 GLA 还保留 C2a 阶段 → 白等 0x7；GLA 链要去掉求逆节点。

## 代码搜索关键词

```bash
grep -nE "SyncAll|sync_all|CrossCoreSetFlag|CrossCoreWaitFlag|SetFlag|WaitFlag|ParaChunk|AICProcess|InverseCompute|AttnInverseMMCompute|GCumExpCompute|QKPreProcess|GBKCompute|VBetaCompute" op_kernel/*.{cpp,h}
```
