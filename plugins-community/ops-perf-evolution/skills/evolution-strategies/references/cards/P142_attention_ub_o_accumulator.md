---
id: P142
bottlenecks: [mte3_stall, no_overlap, partial_overlap]
op_families: [attention, flash_attention]
complexity: L2
conflicts_with: []
synergizes_with: [P81, P38, P119, P120, P121]
requires: []
has_preconditions: true
has_playbook: true
---

# P142: Attention UB 常驻 O 累加器

## 核心思想
Flash Attention 每处理一个 KV tile 都要更新输出 O：
`O_new = rescale(m_old, m_new, l_old, l_new) * O_old + P_tile @ V_tile`。
如果每次把 `O_old` 写回 GM/UB 再读回，每个 KV tile 都会引入 MTE2/MTE3 往返和同步；将 `O_tile` 常驻 UB，跨 KV tile 流式更新，可彻底砍掉这部分搬运。

## 代码骨架

```cpp
// === 改造前（per-tile GM 往返）===
for (int kv = 0; kv < kvTiles; ++kv) {
    ComputePTile(...);                 // P_tile = softmax(S_tile)
    O_old = LoadFromGM(O_Gm);          // MTE2/MTE3 往返
    O_new = Rescale(O_old) + P_tile @ V_tile;
    WriteToGM(O_Gm, O_new);            // 写回 GM
}

// === 改造后（UB 常驻 O）===
LocalTensor<float> O_ub = ub.AllocTensor<float>(M_tile * d);
SetZero(O_ub);
for (int kv = 0; kv < kvTiles; ++kv) {
    ComputePTile(...);
    // L0C 算 P_tile @ V_tile，结果 rescale 后加到 UB O
    OUpdateInUB(O_ub, P_tile, V_tile, m_old, m_new, l_old, l_new);
}
DataCopy(O_Gm, O_ub, M_tile * d);      // 循环结束一次性写回
ub.FreeTensor(O_ub);
```

## 关键修改点

1. **UB 分配 O_tile**：循环外为当前 `M_tile × d` 输出 tile 分配 UB buffer 并清零。
2. **流式更新**：每 KV tile 在 L0C 算 `P_tile @ V_tile`（fp32 累加），按 running max/sum 对 UB 中 `O_old` 做 rescale 后相加，**不写出 GM**。
3. **一次性写回**：全部 KV tile 处理完后，UB 中的 O_tile 一次性 DataCopy 到 GM。
4. **容量边界**：若 `M_tile × d` 超过 UB 预算，拆小 M_tile 或 fallback 到 L1/GM ring；不要为硬塞 UB 牺牲 Q-tile 并行度。
5. **精度**：rescaling 在 UB 中按 fp32 进行；最终写回前才 cast 到目标 dtype。

## 常见陷阱

- ❌ UB 容量估算不足导致 UB 溢出或主数据 tile 被挤占。
- ❌ 循环内对 O_ub 做 in-place 修改后未同步就进入下一轮 KV tile。
- ❌ 与 GM workspace-ring（X38）混用：已用 X38 放大 kvN 时，UB 常驻是替代方案，不要叠加。
- ❌ M_tile 过大导致 UB 放不下，又不拆分，直接崩溃。

## 代码搜索关键词

```bash
grep -nE "O_Gm|O_.*GM|DataCopy.*O|WriteO|RescaleO|UpdateO|O_tile" op_kernel/*.cpp op_kernel/*.h
```
