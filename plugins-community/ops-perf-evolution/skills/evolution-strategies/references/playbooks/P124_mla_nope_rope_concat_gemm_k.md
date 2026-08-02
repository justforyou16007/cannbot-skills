# P124 Playbook: MLA nope+rope 拼接进单 GEMM K 维

> **强制流程**。本 Playbook 指导子 agent 把 MLA QK^T 的 nope/rope 两段独立计算合并为单 GEMM 的 K 维（embed-split 分段流水 + 单 L0C 累加 + 一次写回）。前置：P120 已实施（QK^T 已在 Cube 上）。必须按 Step 1-5 逐步执行，严禁跳步。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p124_locations.txt`：

```bash
# 1.1 rope/nope 双张量通路（GM 指针、layout、搬运调用）
grep -nE "rope|Rope|nope|Nope" \
    shared/original/op_kernel/*.{cpp,h} shared/original/op_kernel/**/*.hpp > /tmp/p124_locations.txt
# 1.2 两段独立 score 计算点（两次 mmad / 两次点积 + 相加）
grep -nE "Mmad|mmad|ReduceSum|Add\(|Adds\(" \
    shared/original/op_kernel/*.{cpp,h} shared/original/op_kernel/**/*.hpp >> /tmp/p124_locations.txt
# 1.3 当前 K 维 tile 常量（embed、d_nope、d_rope、L1TileShape）
grep -nE "embed|EMBED|GemmShape|L1TileShape|d_nope|d_rope|512|576" \
    shared/original/op_kernel/*.{cpp,h} shared/original/op_kernel/**/*.hpp >> /tmp/p124_locations.txt
# 1.4 Q 的 L1 装载点（是否常驻、装载次数）
grep -nE "copyGmToL1A|loadQ|l1A" \
    shared/original/op_kernel/*.{cpp,h} shared/original/op_kernel/**/*.hpp >> /tmp/p124_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，明确：rope 张量在哪几个 GM 通路出现、score 当前分几段计算、每段 K 维长度、Q 装载是否每 tile 重复。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| score 计算 | nope/rope 两段独立 + 相加 | 单 K 维循环（embedSplit），单 L0C 累加 | op_kernel/*:L? |
| GEMM K 维 | d_nope（或分段独立） | embedCat = d_nope + d_rope | op_kernel/*:L? |
| Q 的 L1 布局 | 单段 / 重复装载 | nope+rope 连续拼接常驻（nIdx==0 一次） | op_kernel/*:L? |
| K 的 L1 缓冲 | 单缓冲整段 | nope 2 拍（256 宽）双缓冲 + rope 独立小缓冲 | op_kernel/*:L? |
| L0C 写回 | 每段一次 + 合并 | 全程一次 fixpipe | op_kernel/*:L? |
| 事件槽 | 复用同一组 | rope 用独立 EVENT_ID（+2 偏移） | op_kernel/*:L? |

## Step 3: 代码改造

### 3A. 形态识别

- **Canonical**：两次独立 mmad（Q_nope×K_nope 与 Q_rope×K_rope）+ L0C/GM 上相加合并。
- **变体 A**：向量侧两段点积相加（尚未 cube 化）→ 先回 P120，本卡只做拼接部分设计。
- **变体 B**：Q/K 在 host 侧已拼接成单张量 → 跳过 L1 拼接，只需确认 K 维 embed-split 分段流水已存在，否则只补分段。

### 3B. Canonical Template

```cpp
// 改造前：两段独立
blockMmadNope(gQ, gK, gS1, ...);            // K = d_nope
blockMmadRope(gQRope, gKRope, gS2, ...);    // K = d_rope
Add(gS, gS1, gS2);                          // 合并

// 改造后：单 K 维循环
if (nIdx == 0) {  // Q 拼接常驻 L1
    copyGmToL1A(l1A, gQ, layoutAInL1, layoutQ);
    copyGmToL1A(l1A[rowNumRound * embed], gARope, layoutRopeInL1, layoutQRope);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0); WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
}
for (embedSplitIdx = 0; embedSplitIdx < nopeSplitLoop + 1; embedSplitIdx++) {
    bool isRope = (embedSplitIdx == nopeSplitLoop);
    uint32_t kSplit = isRope ? embedRope : min(EMBED_SPLIT, embed - embedSplitIdx * EMBED_SPLIT);
    uint64_t qOff = isRope ? rowNumRound * embed : embedSplitIdx * rowNumRound * EMBED_SPLIT;
    copyL1ToL0A(l0A[pp], l1A[qOff], ...);
    // K: nope 每 2 段一拍 GM→L1 双缓冲；rope 独立 l1BRope + 事件 ID+2
    copyL1ToL0B(l0B[pp], isRope ? l1BRope[pp] : l1B[pp][inner * EMBED_SPLIT * seqRound], ...);
    Mmad(l0C[pp2], l0A[pp], l0B[pp], {m, n, kSplit, /*initL0C=*/(embedSplitIdx == 0)});
}
copyL0CToGm(gS, l0C[pp2], ...);  // 唯一一次写回
```

### 3C. Variant Notes

- d_nope 非 128 整数倍 → 末段 kSplit 取余量，L0 分形按实际 kSplit MakeLayout（RoundUp 到 C0）。
- d_rope ≠ 64 时 rope 段同样按 EMBED_SPLIT 继续细分，不改框架。
- L1 预算紧张 → 优先缩 L1B 拍宽（256→128），不动 Q 常驻区。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P124 约束核查：
  1. L1 预算：Q常驻(rowNumRound*embedCat*2B) + K双拍(2*seqRound*256*2B) + rope(2*seqRound*64*2B) ≤ L1_MAX？ 计算式 = ___
  2. initL0C 仅在 embedSplitIdx==0 为 true？ yes/no
  3. rope 段事件槽与 nope 段不共享？ yes/no
  4. Q_rope 的 L1 偏移用 rowNumRound（对齐后）？ yes/no
  5. 两段独立 mmad + Add 合并代码已删除？ yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: K 维拼接常量/循环存在
(grep -cE "embedSplit|embedCat|isRope" modified_files/op_kernel/*.{cpp,h} modified_files/op_kernel/**/*.hpp; true)
# 期望: >= 1

# 检查 2: initL0C 分段控制存在
(grep -cE "initL0C|embedSplitIdx == 0" modified_files/op_kernel/*.{cpp,h} modified_files/op_kernel/**/*.hpp; true)
# 期望: >= 1

# 检查 3: Q 常驻装载仅一次（nIdx==0 守卫）
(grep -cE "nIdx == 0" modified_files/op_kernel/*.{cpp,h} modified_files/op_kernel/**/*.hpp; true)
# 期望: >= 1

# 检查 4: 分段间无中间合并（无 S1+S2 式 Add）
(grep -cE "Add\(.*[sS](core)?1.*[sS](core)?2|Add\(gS" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: == 0

# 检查 5: L0C 写回次数未增加（copyL0CToGm 每 KV 块仍 1 次）
(grep -cE "copyL0CToGm|CopyL0CToGm" modified_files/op_kernel/*.{cpp,h} modified_files/op_kernel/**/*.hpp; true)
# 期望: 与改造前每 KV 块写回次数相同（通常 1）
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P124 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: nope+rope concat applied (single_k_loop=yes, q_l1_resident=yes, single_l0c_writeback=yes)
Step 4: constraints checked (l1_budget=___, rope_event_slot=___)
Step 5: all 5 grep checks passed
```

## Step 6: Known Pitfalls

- score 只剩 rope 段贡献 → `initL0C` 每段都置 true 了，改回仅首段。
- score 部分行/列错乱 → Q_rope 的 L1 偏移用了未对齐的 rowNum，改 rowNumRound。
- 偶发数据竞争/结果随机错 → rope 与 nope 复用了同一事件 ID，rope 改独立槽（+2）。
- 编译报 L0 溢出 → embed-split 步长过大，保持 128 并检查 L0A/L0B 分形尺寸。
- 性能无提升 → K 维循环内 MTE2_MTE1 事件粒度太细（每段 SetFlag/WaitFlag 串行化），确认 GM→L1 双拍预取已打开。
