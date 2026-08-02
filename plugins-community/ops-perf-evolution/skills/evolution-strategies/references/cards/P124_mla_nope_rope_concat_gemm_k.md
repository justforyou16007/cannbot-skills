---
id: P124
bottlenecks: [cube_bound, mte2_stall]
op_families: [attention, flash_attention]
complexity: L1
conflicts_with: []
synergizes_with: [P120, P122, P116, P121]
requires: [P120]
has_preconditions: true
has_playbook: true
---

# P124: MLA nope+rope 拼接进单 GEMM K 维（embed-split K 维流水，一次 L0C 累加）

## 核心思想

MLA 的 QK^T 中 Q = [Q_nope | Q_rope]、K = [K_nope | K_rope] 是两个独立 GM 张量。朴素实现把 score 拆成两段分别计算再相加（两次点积循环 / 两次独立 GEMM），每段都要独立的搬运-计算-累加-合并开销，且 K 维被砍半后 mmad 启动开销占比翻倍。

把两段沿 K 维拼成**一次 GEMM 的 K 维**（`embedCat = d_nope + d_rope`，典型 512+64=576）：

1. **Q 拼接常驻 L1**：任务开头把 Q_nope 搬到 L1 `[0, d_nope)`、Q_rope 搬到 L1 紧接的 `[d_nope, d_nope+d_rope)`，一次搬运、整个 KV 循环复用（`nIdx==0` 时执行）。
2. **K 按 embed-split 分段流水**：nope 段按 128 步长分 `ceil(d_nope/128)` 段，rope 段作最后一段；L1B 按 2 拍（256 宽）双缓冲，MTE2 搬下一段与 M 算当前段重叠；L1→L0 只搬当前 128 段，L0A/L0B 双缓冲。
3. **单 L0C 累加器贯穿全部 K 段**：`mmad(..., initL0C = (embedSplitIdx==0))`，所有分段累加进同一 L0C，整段 K 维结束后**一次 fixpipe** 出 S——没有中间合并、没有第二次写回。

M 维取 qHeads（MLA 单 KV 头，全部 q 头共享同一 latent KV，是 P116 头合并的极限形式），rowNum=128 喂满 mmad M 维。

## 代码骨架

```cpp
// === 改造前（两段独立点积再相加）===
for (j) { sum  = dot(qNope, kNope[j]);      // 第一遍 Dqk_nope
          sum += dot(qRope, kRope[j]);      // 第二遍 d_rope
          sc[j] = sum * scale; }

// === 改造后（单 GEMM，K 维 = embedCat）===
using L1TileShape = GemmShape<128, 128, 576>;        // K = 512 nope + 64 rope
if (nIdx == 0) {                                      // Q 拼接常驻 L1，仅一次
    copyGmToL1A(l1A,             gQ,     layoutAInL1, layoutQ);       // [0, 512)
    copyGmToL1A(l1A[row*embed],  gQRope, layoutRopeInL1, layoutQRope);// [512, 576)
}
for (embedSplitIdx = 0; embedSplitIdx < nopeSplitLoop + 1; embedSplitIdx++) {
    bool isRope = (embedSplitIdx == nopeSplitLoop);   // 最后一段是 rope
    // L0A: 从 L1 的对应偏移取 Q 段（rope 段偏移 = rowNumRound * embed）
    copyL1ToL0A(l0A[pp], l1A[qL1Offset], ...);
    // L1B/L0B: nope 段 2 拍双缓冲搬 K，rope 段独立小 buffer
    copyL1ToL0B(l0B[pp], isRope ? l1BRope[pp] : l1B[pp][inner*128*seqR], ...);
    Mmad(l0C[pp2], l0A[pp], l0B[pp], {m, n, kSplit, /*initL0C=*/(embedSplitIdx==0)});
}
copyL0CToGm(gS, l0C[pp2], ...);                       // 全程唯一一次写回
```

## 关键修改点

1. **删除 nope/rope 两段独立 score 计算与相加逻辑**，合并为 K 维循环内的分段 mmad；`initL0C` 仅在首段为 true。
2. **L1 布局重排**：Q 两段在 L1 连续拼接（rope 偏移 = `rowNumRound * d_nope` 元素）；K_nope 双拍缓冲 + K_rope 独立双缓冲小 buffer。
3. **GM→L1 搬运分段化**：nope 每 2 个 embed-split（256 元素）搬一拍，`MTE1_MTE2` 事件在两段消费完后 SetFlag 释放；rope 用独立事件槽（如 EVENT_ID+2）避免与 nope 缓冲互踩。
4. **scale 时机不变**：softmax 前对 S 乘 scale（tor），不在 K 维循环内做。

## 常见陷阱

- ❌ rope 段走 nope 的 L1B 双拍缓冲 → 事件槽冲突数据被覆盖；rope 必须独立 buffer + 独立事件 ID。
- ❌ `initL0C` 每段都置 true → 只保留最后一段（rope）的贡献，score 全错。
- ❌ Q_rope 的 L1 偏移用 `rowNum`（未对齐）而非 `rowNumRound` → zN 分形下行错位，score 部分错乱。
- ❌ embed-split 步长不按 128（C0 对齐倍数）切 → L0B 分形不对齐，搬运报错或结果错。
- ❌ 拼接后 K 维超 L0A/L0B 单次容量仍不分段 → 编译期 L0 溢出；embed-split 是必需的，不是可选的。

## 代码搜索关键词

```bash
grep -nE "rope|Rope|nope|Nope|latent|embedSplit|initL0C|Mmad" op_kernel/*.cpp op_kernel/*.h op_kernel/**/*.hpp
```
