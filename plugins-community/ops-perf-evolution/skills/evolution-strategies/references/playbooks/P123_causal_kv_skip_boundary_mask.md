# P123 Playbook: Causal/Sparse KV 范围跳过 + 边界 Tile Mask 特化

> **强制流程**。本 Playbook 指导子 agent 把 attention 算子的 causal/sparse mask 从"全量循环 + 逐元素兜底"改写为"循环界裁剪 + 边界 tile 特化 + 编译期分发"。必须按 Step 1-5 逐步执行，严禁跳步。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p123_locations.txt`：

```bash
# 1.1 causal 判定与 mask 应用位置
grep -nE "causal|caus|isCausal|sparseMode|mask" \
    shared/original/op_kernel/*.{cpp,h} > /tmp/p123_locations.txt
# 1.2 逐元素 mask 填充（NEG/-1e30 兜底）
grep -nE "Duplicate\(.*NEG|Duplicate\(.*-1e30|-3e38|SetValue" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p123_locations.txt
# 1.3 KV 循环界（当前是否跑满全量）
grep -nE "for.*kv|for.*s2|kvLen|kl|kvEnd|kvSLoopNum" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p123_locations.txt
# 1.4 kernel 入口与模板参数（dtype/mask 分发现状）
grep -nE "launch_|template|MaskType|__global__" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p123_locations.txt
# 1.5 q/kv 序列长度差（diffS 对齐项依据）
grep -nE "kvSeqlen|qSeqlen|Sk\b|off\b" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p123_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，明确：causal 当前如何实现、KV 循环上界公式、逐元素 mask 有几处、kernel 入口如何分发。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| KV 循环上界 | 全量 kvLen | `CeilDiv(noSkipKvS, KV_STACK)` | op_kernel/*.cpp:L? |
| diffS 对齐项 | 无 | `max(kvSeqlen - qSeqlen, 0)` | op_kernel/*.cpp:L? |
| 全 mask 尾部 tile | 照算再扔 | 不进循环（裁剪掉） | op_kernel/*.cpp:L? |
| 边界 tile mask | 逐元素判断 | triUp/triDown 判定 + mask epilogue 重载 | op_kernel/*.cpp:L? |
| 非边界 tile | 同走 mask 路径 | nomask 快速重载（无 mask 参数） | op_kernel/*.cpp:L? |
| mask 分支 | runtime if | `if constexpr (MASK_TYPE == ...)` | op_kernel/*.cpp:L? |
| kernel 入口 | 单一入口 | dtype × causal 4 入口（独立 .cpp） | op_kernel/*.cpp:L? |
| 空范围 task | 未处理 | InitOutWhenZero 写 0 输出 | op_kernel/*.cpp:L? |

## Step 3: 代码改造

### 3A. 形态识别

- **Canonical**：right-down causal（sparseMode=3），kvSeqlen ≥ qSeqlen。
- **变体 A**：banded sparse（sparseMode=4，pre/next token）→ 同时裁剪 `startIdx`（左界）与 `noSkipKvS`（右界）。
- **变体 B**：top-left causal（qSeqlen == kvSeqlen 且 diffS=0）→ 公式自然退化，无需特判。

### 3B. Canonical Template

```cpp
// 改造前：for (t = 0; t < kvLen/BN; t++) { compute; if (caus) 逐元素填 NEG; }

// 改造后：循环界裁剪 + 边界特化
int64_t noSkipKvS = kvSeqlen;
if constexpr (MASK_TYPE == MaskType::MASK_CAUSAL) {
    int64_t diffS = std::max(kvSeqlen - qSeqlen, 0L);      // right-down 对齐项
    noSkipKvS = std::min(kvSeqlen, (qSBlockIdx+1)*qSTile + diffS);
}
uint32_t kvEnd = CeilDiv((uint32_t)noSkipKvS, KV_STACK);
if (kvEnd == 0) { epilogueInitOut(gO, ...); return; }      // 空范围兜底写 0
for (uint32_t kvSIdx = 0; kvSIdx < kvEnd; kvSIdx++) {
    uint32_t kvSEndIdx = kvSIdx * KV_STACK + stackSeqTile;
    uint32_t triUp = noSkipKvS - qSBlockSize;
    if (triUp < kvSEndIdx - 1)                              // 与对角线相交
        epilogueOnlineSoftmax(..., gMask, triUp, noSkipKvS, ...);   // mask 重载
    else                                                    // 纯下三角
        epilogueOnlineSoftmax(... /* 无 mask 参数 */);              // nomask 重载
}
```

### 3C. Variant Notes

- 若 kvSeqlen < qSeqlen（罕见）→ diffS 截 0，noSkipKvS 退化为 `(qSBlockIdx+1)*qSTile`，仍正确。
- 若 mask 尚未预生成 → 先落地 P118（独立 AIV kernel 预生成），本卡只消费。
- 若入口已按 dtype 分发 → 只加 MaskType 一维，.cpp 数量 ×2。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P123 约束核查：
  1. noSkipKvS 含 diffS = max(kvSeqlen-qSeqlen, 0)？ 公式 = ___
  2. 空范围 task（kvEnd<=0）已走 InitOutWhenZero？ yes/no
  3. 边界判定 triUp < kvSEndIdx-1 的 off-by-one 已用小 shape 验证？ yes/no
  4. nomask 路径二进制无 mask 代码（if constexpr / 模板重载）？ yes/no
  5. 4 入口各自独立 .cpp？ yes/no
  6. 与 P122 联动：尾块 stackSeqTile = noSkipKvS - idx*STACK？ yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 循环界裁剪
(grep -cE "noSkipKvS|kvSLoopNum" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 2: diffS 对齐项
(grep -cE "diffS|kvSeqlen *- *qSeqlen|Sk *- *S\b" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 3: 边界 tile 判定
(grep -cE "triUp|triDown|doTriUMask" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 4: 编译期分发
(grep -cE "MaskType|MASK_CAUSAL|if constexpr" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 5: 空范围兜底
(grep -cE "InitOut|initOut|Duplicate\(.*0.*0f.*embed" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P123 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: causal kv skip applied (noSkipKvS=yes, boundary_specialized=yes, entries=___)
Step 4: constraints checked (diffS=___, zero_guard=yes, offbyone_verified=yes)
Step 5: all 5 grep checks passed
```

## Step 6: Known Pitfalls

- decode（kvSeqlen≫qSeqlen）精度全错 → noSkipKvS 漏加 diffS，对角线整体左移。
- 对角线元素被误 mask（前几行精度挂）→ triUp 边界 `<=`/`>=` 写错一格；用 qSBlockSize=1 的小 shape 单测。
- 部分输出块是脏数据 → 空范围 task 没走 InitOutWhenZero。
- nomask 入口性能没变化 → mask 分支是 runtime if 而非编译期，死代码仍在二进制里。
- 裁剪后反而更慢 → KV_STACK 太大导致尾块浪费（配 P122 核对 stack 基准与取余）。
