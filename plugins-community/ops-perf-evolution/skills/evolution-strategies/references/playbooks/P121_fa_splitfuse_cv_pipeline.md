# P121 Playbook: FA SplitFuse CV 流水拓扑（GM workspace 握手槽 + PRE_LAUNCH 预发射）

> **强制流程**。本 Playbook 指导子 agent 把 FA 算子的 QK→softmax→PV→rescaleO 四阶段从串行/SyncAll 结构改写为 AIC/AIV 双核生产者-消费者流水。前置：P120（QK/PV 已上 cube）。必须按 Step 1-5 逐步执行，严禁跳步。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p121_locations.txt`：

```bash
# 1.1 全局同步点（消除对象）
grep -nE "SyncAll|sync_all" \
    shared/original/op_kernel/*.{cpp,h} > /tmp/p121_locations.txt
# 1.2 四阶段位置：QK / softmax / PV / rescale
grep -nE "Mmad|mmad|softmax|Softmax|Exp\(|rescale|Rescale" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p121_locations.txt
# 1.3 现有中间结果通路（UB 还是 GM workspace）
grep -nE "workspace|workSpace|WorkSpace|gS|gP|gOTmp" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p121_locations.txt
# 1.4 核角色划分（mix kernel 的 AIC/AIV 分支）
grep -nE "__DAV_C220_CUBE__|__DAV_C220_VEC__|GetSubBlockNum|GetBlockIdx" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p121_locations.txt
# 1.5 现有跨核/事件机制
grep -nE "CrossCoreSetFlag|CrossCoreWaitFlag|SetFlag|WaitFlag" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p121_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，明确：四阶段当前执行顺序、SyncAll 有几处、中间结果经 UB 还是 GM、AIC/AIV 是否已分工。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 阶段间同步 | SyncAll 全局屏障 | qkReady/softmaxReady/pvReady 三 flag 定向握手 | op_kernel/*.cpp:L? |
| S/P 通路 | UB（或串行复用） | GM workspace，按 `coreIdx*SLOTS + slot%SLOTS` 分槽 | op_kernel/*.cpp:L? |
| 槽数 | 1（串行） | PRE_LAUNCH+1（推荐 PRE_LAUNCH=2，3 槽） | op_kernel/*.cpp:L? |
| KV 循环上界 | kvEnd | kvEnd + preKVNum（尾部排空） | op_kernel/*.cpp:L? |
| 阶段错位 | 无 | 当前 i 做 QK+softmax，i-PRE_LAUNCH 做 PV+rescaleO | op_kernel/*.cpp:L? |
| 初始化 | 无 | 预置全部 HardEvent SetFlag 暖机 | op_kernel/*.cpp:L? |
| AIV coreIdx | 未校正 | `GetBlockIdx()/GetSubBlockNum()` | op_kernel/*.cpp:L? |
| flag id | — | 常量 QK_READY=1 / SOFTMAX_READY=2 / PV_READY=3 | op_kernel/*.h:L? |

## Step 3: 代码改造

### 3A. 形态识别

- **Canonical**：mix kernel，`__DAV_C220_CUBE__`/`__DAV_C220_VEC__` 双分支，主循环内 SyncAll 串接四阶段。
- **变体 A**：单核向量实现尚无 CV 分工 → 先回 P120 建立 cube 路径，本卡只做循环骨架预留。
- **变体 B**：已有 CrossCoreFlag 但单槽 workspace → 只加槽位维度与 PRE_LAUNCH 错位（工作量减半）。

### 3B. Canonical Template

```cpp
// 改造前：for (kv) { QK; SyncAll; softmax; SyncAll; PV; SyncAll; rescale; SyncAll; }

// 改造后：三 flag 握手 + 槽位环 + 预发射
for (uint32_t kvSIdx = kvStart; kvSIdx < kvEnd + preKVNum; kvSIdx++) {
    uint32_t slot = stackSeqCount % (PRE_LAUNCH + 1);
    uint64_t ws = coreIdx * WS_BLOCK * (PRE_LAUNCH+1) + slot * WS_BLOCK;
    if (kvSIdx < kvEnd) {                        // 当前 tile：QK + softmax
        blockMmadQK(..., gS[ws], ...);           // AIC
        CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
        CrossCoreWaitFlag(qkReady);              // AIV
        epilogueOnlineSoftmax(gP[ws], gS[ws], ...);   // P 复用 S 同槽偏移
        CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
    }
    if (kvSIdx >= kvStart + preKVNum) {          // 滞后 tile：PV + rescaleO
        uint64_t wsL = /* (stackSeqCount-PRE_LAUNCH) % SLOTS 的同式寻址 */;
        blockMmadPV(gP[wsL], ..., softmaxReady); // AIC 等 softmax
        CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
        CrossCoreWaitFlag(pvReady);              // AIV
        epilogueRescaleO(gO, gOTmp[wsL], gOUpdate[coreIdx*WS_BLOCK], ...);
    }
    stackSeqCount++;
}
```

### 3C. Variant Notes

- 若 AIV 有 2 子核 → 每个 AIV 子核分担一半行，`coreIdx` 必须先除 `GetSubBlockNum()` 再寻址 workspace。
- 若 KV 循环存在跳过（P123 的 kvStart>0）→ 槽位序号仍从 `stackSeqCount=0` 起算，与 kvStart 解耦。
- 若 workspace 预算不足 3 槽 → PRE_LAUNCH 降为 1（2 槽），收益略减但结构不变。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P121 约束核查：
  1. workspace 总量 = cores × Q_TILE × KV_STACK × (4B S + 2B P + 4B OTmp + 4B Update) × SLOTS
     计算式 = ___， ≤ 申请预算？ yes/no
  2. 槽数 ≥ PRE_LAUNCH+1？ yes/no
  3. 每个 WaitFlag 都有同 id 的初始化预置 SetFlag？ 列出对应表 = ___
  4. 循环上界 +preKVNum 且滞后段条件 kvSIdx >= kvStart+preKVNum？ yes/no
  5. 主循环内 SyncAll 已全部删除？ yes/no
  6. AIV 侧 coreIdx 已除 GetSubBlockNum()？ yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 跨核握手已引入
(grep -cE "CrossCoreSetFlag|CrossCoreWaitFlag" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 3

# 检查 2: SyncAll 已删除
(grep -cE "SyncAll" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: == 0

# 检查 3: 预发射错位
(grep -cE "PRE_LAUNCH|preKVNum" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 2

# 检查 4: 槽位寻址
(grep -cE "%.*PRE_LAUNCH|% *SLOTS|stackSeqCount" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 5: 初始化暖机
(grep -cE "SetFlag<.*HardEvent" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 4
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P121 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: splitfuse cv pipeline applied (slots=___, pre_launch=___, syncall_removed=yes)
Step 4: constraints checked (workspace_bytes=___, warmup_flags=___)
Step 5: all 5 grep checks passed
```

## Step 6: Known Pitfalls

- kernel hang 在第一个 WaitFlag → 初始化预置 SetFlag 漏了对应事件 id；逐项对照 Step 4 第 3 条。
- 输出偶发缺块/错位 → 槽数 < PRE_LAUNCH+1，或滞后段 wsL 寻址用了 `kvSIdx` 而非 `stackSeqCount-PRE_LAUNCH`。
- 结果偶发错误且难复现 → workspace 槽被相邻核踩：寻址公式漏乘 `coreIdx` 或 AIV 未除子核数。
- 性能无提升 → 主循环仍有 SyncAll 或 PipeBarrier<PIPE_ALL> 残留，流水被打断。
- 尾部输出为旧数据 → 循环上界没加 preKVNum，最后 PRE_LAUNCH 个 tile 的 PV/rescale 没跑。
