# P122 Playbook: KV 堆叠大块化 + L1 容量自适应 Q/KV 划分

> **强制流程**。本 Playbook 指导子 agent 把 FA 算子的 KV tile 从小块（≤128）堆叠为大块（512 基准），并把 L1 划分从硬编码改为容量公式自适应。前置：P120（cube 路径已建立）。必须按 Step 1-5 逐步执行，严禁跳步。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p122_locations.txt`：

```bash
# 1.1 KV tile 常量（BN / kvTile / s2Base 等）
grep -nE "constexpr.*BN|constexpr.*KV|kvTile|s2Base|S2_BASE|blockSize" \
    shared/original/op_kernel/*.{cpp,h} > /tmp/p122_locations.txt
# 1.2 KV 循环与尾块处理
grep -nE "for.*kv|for.*s2|kvSIdx|stackSeqTile" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p122_locations.txt
# 1.3 L1 buffer 分配（硬编码尺寸）
grep -nE "InitBuffer|L1|l1A|l1B|A1|B1" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p122_locations.txt
# 1.4 paged cache 物理块（如有）
grep -nE "paged|blockTable|BlockTable" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p122_locations.txt
# 1.5 workspace 槽尺寸（堆叠后必须联动）
grep -nE "workspace|Workspace|WS_BLOCK|OutSize" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p122_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，明确：当前 KV tile 尺寸、L1 各 buffer 的硬编码值、是否有 paged 物理块、workspace 槽尺寸公式。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| KV 逻辑块 | ≤128 | 512（或按预算重算） | op_kernel/*.h:L? |
| K 维块 kDyn | 硬编码 | `max(RoundUp(embed,128), 256)` | op_kernel/*.cpp:L? |
| N 维块 nDyn | 硬编码 | L1 预算公式推出，对齐 32、上限 128 | op_kernel/*.cpp:L? |
| Q 常驻区 | 每轮重搬 | `Q_TILE(128) × kDyn × dtype` 常驻 L1 | op_kernel/*.cpp:L? |
| V 常驻区 | 每轮重搬 | `embedV × KV_STACK × dtype` 常驻 L1 | op_kernel/*.cpp:L? |
| 双缓冲 | 未计入预算 | DOUBLE_BUFFER=2 写入 nDyn 公式 | op_kernel/*.cpp:L? |
| 尾块 | 满载假设 | `min(STACK, noSkipKvS - idx*STACK)` 取余 | op_kernel/*.cpp:L? |
| workspace 槽 | 旧 tile 尺寸 | `Q_TILE × KV_STACK` 联动放大 | host/kernel:L? |
| paged 堆叠 | — | `blockStackNum = ceil(STACK/blockSize)` | op_kernel/*.cpp:L? |

## Step 3: 代码改造

### 3A. 形态识别

- **Canonical**：非 paged KV，连续 GM 布局，KV tile 32~128 硬编码。
- **变体 A**：paged cache → 按物理块 stack 成 512 逻辑块，跨块边界用乒乓装载（`l1KvPingPongFlag`）。
- **变体 B**：embed > 256 → kDyn 取 RoundUp(embed,128)，L1 预算自动收紧 nDyn，禁止截断 embed。

### 3B. Canonical Template

```cpp
// 改造前：constexpr int BN = 32; for (t = 0; t < kvLen; t += BN) { ... }

// 改造后：堆叠 + 自适应
constexpr uint32_t MAX_KV_STACK_LEN = 512;
constexpr uint32_t Q_TILE_CEIL = 128;
uint32_t kDynNum = RoundUp(embed, 128);  kDynNum = std::max(kDynNum, 256u);
uint32_t maxQKPL1 = L1_MAX_SIZE - embedV * MAX_KV_STACK_LEN * sizeof(T);   // 扣 V 常驻
uint32_t maxQL1   = Q_TILE_CEIL * kDynNum * sizeof(T);                     // 扣 Q 常驻
uint32_t nDynNum  = (maxQKPL1 - maxQL1) / kDynNum / sizeof(T) / DOUBLE_BUFFER;
nDynNum = std::min(RoundDown(nDynNum, 32), 128u);                          // 对齐+上限
blockMmadQK.init(resource, nDynNum, kDynNum, MAX_KV_STACK_LEN);
for (kvSIdx = 0; kvSIdx < kvEnd; kvSIdx++) {
    uint32_t stackSeqTile = std::min(MAX_KV_STACK_LEN,
                                     noSkipKvS - kvSIdx * MAX_KV_STACK_LEN); // 尾块取余
    blockMmadQK(..., GemmCoord{rowNum, stackSeqTile, embed}, ...);
}
```

### 3C. Variant Notes

- 若 kvSeqlen < 512 → 堆叠退化为单块，`stackSeqTile = kvSeqlen`，结构无需特判。
- 若 P123 已落地 → `noSkipKvS` 直接复用其裁剪值，两个策略共用同一份循环界。
- 若 PV 与 QK 共享 K/V 常驻区 → PV init 传 `L1_QK_SIZE` 偏移，`kPV = nDyn*kDyn/PV_L1_M`。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P122 约束核查：
  1. L1 预算恒等式：Q常驻 + V常驻 + nDyn*kDyn*dtype*DOUBLE_BUFFER ≤ L1_MAX？
     计算式 = ___
  2. nDyn 已 RoundDown 32 且 ≤128？ kDyn 已 RoundUp 128 且 ≥256？ yes/no
  3. workspace 槽尺寸已联动改为 Q_TILE × KV_STACK？ yes/no
  4. 尾块取余与 P123 的 noSkipKvS 一致？ yes/no
  5. embed>256 时 kDyn 未被截断？ yes/no
  6. paged 场景 blockStackNum 公式正确（ceil）？ yes/no/na
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 堆叠常量
(grep -cE "KV_STACK|STACK_LEN" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 2: 自适应公式
(grep -cE "L1_MAX|nDyn|kDyn" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 2

# 检查 3: 双缓冲计入预算
(grep -cE "DOUBLE_BUFFER|STAGES" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 4: 尾块取余
(grep -cE "stackSeqTile|min.*STACK|%.*STACK" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 5: 对齐函数
(grep -cE "RoundUp|RoundDown" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 2
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P122 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: kv stacking applied (stack_len=___, kDyn=___, nDyn=___)
Step 4: constraints checked (l1_budget=___, workspace_resized=yes)
Step 5: all 5 grep checks passed
```

## Step 6: Known Pitfalls

- 运行时随机踩内存/结果错乱 → nDyn 公式漏除 DOUBLE_BUFFER 或未对齐 32，L1 分配越界。
- 堆叠后精度挂 → workspace 槽尺寸没联动放大，S/P 写穿到相邻区域。
- causal 场景多读多算 → 尾块仍按 512 满载，未接 P123 的 noSkipKvS。
- nDyn 算出天文数字 → L1 预算被减成负数后无符号回绕；embed>256 时检查 kDyn 是否取 max(...,256)。
- paged 跨块数据错位 → blockStackNum 用了整除而非 ceil，最后一个物理块被丢弃。
