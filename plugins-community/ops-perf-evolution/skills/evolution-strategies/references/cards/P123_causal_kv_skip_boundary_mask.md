---
id: P123
bottlenecks: [compute_bound, cube_bound]
op_families: [attention, flash_attention]
complexity: L1
conflicts_with: []
synergizes_with: [D2, P118, P119, P120]
requires: []
has_preconditions: true
has_playbook: true
---

# P123: Causal/Sparse KV 范围跳过 + 边界 Tile Mask 特化 + dtype×mask 编译期分发

## 核心思想

**识别信号**：causal 的实现是"全量 KV 循环 + 逐元素兜底"——KV 循环永远跑满 `kvLen/BN` 轮，每轮对 score 逐元素判断可见性并填 `-inf`。下三角以外的 tile 被完整地"算一遍再扔掉"，causal 场景近一半的 QK^T/PV 计算和 K/V 搬运是纯浪费。

三级优化：

1. **循环界跳过（tile 级）**：causal 时直接把 KV 循环上界裁到可见范围——
   `noSkipKvS = min(kvSeqlen, (qSBlockIdx+1)*qSTile + max(kvSeqlen-qSeqlen, 0))`，
   全被 mask 的尾部 tile **不进循环、不搬运、不计算**。长 KV decode 场景收益接近 `(1 - qSeqlen/2kvSeqlen)`。
2. **边界 tile 特化（块级二分）**：循环内按 `triUp/triDown` 判断当前 tile 是否与对角线相交——相交才走带 mask 的 epilogue 重载（读 mask、上三角置 -inf）；不相交走 **nomask 快速重载**（同名 epilogue 的无 mask 参数版本），mask 搬运和比较全部消除。 sparse mode 4（pre/next token 带状窗口）同理按 `preToken/nextToken` 裁剪 `startIdx/noSkipKvS`。
3. **dtype × mask 编译期分发**：`MaskType` 作为模板参数 + `if constexpr` 编译期裁剪，实例化为 4 个独立 kernel 入口（half/bf16 × causal/nomask，各自单独 .cpp 缩短编译时间）；nomask 入口的二进制里**完全没有 mask 代码**；host 按属性直接 launch 对应入口（等价于 D2 在 FA 的落地）。

辅助机制：可见范围为空（`kvSLoopNumTotal <= 0`）的 task 走 `EpilogueInitOutWhenZero` 直接写 0 输出，不进主循环；causal mask 本身由独立 prepare kernel 预生成（见 P118），主 kernel 只消费。

## 代码骨架

```cpp
// === 改造前：全量循环 + 元素级 mask 兜底 ===
for (int t = 0; t < kvLen/BN; t++) {          // 全被 mask 的 tile 照算
    computeScores(...);
    if (caus) for (qi) for (kj)               // 逐元素判断
        if (ks+kj > gq+off) sp[...] = NEG;
}

// === 改造后：循环界裁剪 + 边界特化 + 编译期分发 ===
int64_t noSkipKvS = kvSeqlen;
if constexpr (MASK_TYPE == MaskType::MASK_CAUSAL) {   // 编译期
    int64_t diffS = max(kvSeqlen - qSeqlen, 0);
    noSkipKvS = min(kvSeqlen, (qSBlockIdx+1)*qSTile + diffS);
}
kvEnd = CeilDiv(noSkipKvS, KV_STACK);
for (kvSIdx = 0; kvSIdx < kvEnd; kvSIdx++) {
    uint32_t triUp = noSkipKvS - qSBlockSize, triDown = noSkipKvS;
    if (triUp < kvSEndIdx - 1)                // 与对角线相交：mask 路径
        epilogueOnlineSoftmax(..., gMask, ..., triUp, triDown, ...);
    else                                      // 纯下三角：nomask 快速路径
        epilogueOnlineSoftmax(..., /*无 mask 参数*/...);
}
// host 分发：isCausal ? mha_kernel_half_causal<<<>>> : mha_kernel_half_nomask<<<>>>
```

## 关键修改点

1. **KV 循环上界公式化**：`noSkipKvS` 必须含 `diffS = max(kvSeqlen-qSeqlen, 0)` 对齐项（right-down causal），漏掉会算错对齐。
2. **边界判定下放到 epilogue 重载**：mask/nomask 两个同名重载，按 tile 位置二选一；不要在逐元素层做。
3. **空范围兜底**：`kvEnd <= 0` 或 `startIdx >= kvEnd` 的 task 必须走 InitOutWhenZero 写 0，否则输出残留脏数据。
4. **编译期分发四入口**：模板参数 `<InputDtype, MaskType>` + 独立 .cpp 实例化；host `isCausal` 直接选入口，runtime 零分支。
5. **与 P122 联动**：尾块 `stackSeqTile = noSkipKvS - idx*512` 取余，跳过逻辑与堆叠块边界必须一致。

## 常见陷阱

- ❌ `noSkipKvS` 漏加 `diffS` → kvSeqlen > qSeqlen（decode/增量推理）时对角线错位，结果全错。
- ❌ 只裁循环界、不处理空范围 task → 部分 (b, h, q块) 输出未初始化脏数据。
- ❌ mask/nomask 用 runtime if 共用一个函数体 → nomask 路径仍背 mask 分支与寄存器压力；用模板重载/`if constexpr`。
- ❌ 边界 tile 判定用 `<=`/`>=` 写错一格 → 对角线元素被误 mask，精度挂在前几行。
- ❌ 四入口共用一个 .cpp → 编译时间翻倍且 nomask 二进制携带死代码。

## 代码搜索关键词

```bash
grep -nE "noSkipKvS|triUp|triDown|causal|sparseMode|preToken|nextToken|MaskType|InitOutWhenZero|isCausal" op_kernel/*.cpp op_kernel/*.h
```
