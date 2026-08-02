---
id: P121
bottlenecks: [no_overlap, partial_overlap]
op_families: [attention, flash_attention, cv_fusion]
complexity: L2
conflicts_with: []
synergizes_with: [P14, P73, P120, P122, P38]
requires: []
has_preconditions: true
has_playbook: true
---

# P121: FA SplitFuse CV 流水拓扑（AIC QK+PV / AIV softmax+rescaleO，GM workspace 握手槽 + PRE_LAUNCH 预发射）

## 核心思想

FlashAttention 单核内的四个阶段天然分属两种核：`QK^T`(cube)、`softmax update`(vector)、`PV`(cube)、`O rescale`(vector)。串行实现（或 `SyncAll` 全局屏障实现）里 Cube 等 Vector、Vector 等 Cube，两类核各有一半时间在空转。

FIA SplitFuse 拓扑把四个阶段铺成 **AIC/AIV 双核生产者-消费者流水**：

```
AIC:  QK[i] ──qkReady──► ... ──softmaxReady──► PV[i] ──pvReady──► ...
AIV:  ... ──wait qkReady──► Softmax[i] ──wait...──► RescaleO[i] ──...
```

三个关键机制：

1. **GM workspace 多槽环 = CV 间数据通路**：S/P/OTmp/Update 全部经 GM workspace 交接，按 `coreIdx * SLOTS + slot % SLOTS` 寻址（`SLOTS = PRE_LAUNCH + 1`，FIA 取 PRE_LAUNCH=2 即 3 槽）。槽数 ≥2 才能让"写 i+1"与"读 i"并行；workspace 预算 = `cores × Q_TILE(128) × KV_STACK(512) × (4B S + 2B P + 4B OTmp + 4B Update) × SLOTS`。
2. **CrossCoreFlag 三握手**：`qkReady(1) / softmaxReady(2) / pvReady(3)` 三个 flag，`CrossCoreSetFlag<0x2, PIPE_FIX>` / `CrossCoreWaitFlag` 做 AIC↔AIV 定向通知（0x2 = 跨核生产者-消费者模式），**不用 SyncAll**。
3. **PRE_LAUNCH 预发射错位**：KV 循环上界写成 `kvEnd + preKVNum`，循环体内"当前 i 做 QK+softmax，i-PRE_LAUNCH 做 PV+rescaleO"——Cube 永远领先 Vector 两个阶段，尾部靠多跑的 preKVNum 次迭代排空。配合**初始化时把所有 HardEvent 预置 SetFlag**（M_MTE1/MTE1_MTE2/FIX_M/MTE3_V/...），第一次 Wait 直接通过，流水无冷启动。

与 P14 的关系：P14 是通用"环形任务缓存 CV 预发射"模式；本卡是 FA 场景的具体拓扑——双 cube 阶段夹双 epilogue、GM workspace 槽位寻址公式、三 flag 协议、预置 SetFlag 暖机清单。与 P73 的关系：P73 调 AIC:AIV 核数比例；本卡决定流水结构本身。

## 代码骨架

```cpp
// 初始化暖机：预置全部事件，首次 WaitFlag 直通（AIC/AIV 各自一套）
SetFlag<HardEvent::M_MTE1>(EVENT_ID0..7); SetFlag<HardEvent::FIX_M>(EVENT_ID0/1); ...

for (uint32_t kvSIdx = kvStart; kvSIdx < kvEnd + preKVNum; kvSIdx++) {
    uint32_t slot = stackSeqCount % (PRE_LAUNCH + 1);
    uint64_t wsOff = coreIdx * WS_BLOCK * (PRE_LAUNCH+1) + slot * WS_BLOCK;
    if (kvSIdx < kvEnd) {                       // 阶段 1+2：当前 tile
#ifdef CUBE
        blockMmadQK(gQ, gK, gS[wsOff], ...);    // AIC: S=QK^T → workspace 槽
        CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#else
        CrossCoreWaitFlag(qkReady);             // AIV: softmax(S)→P 回写同槽
        epilogueOnlineSoftmax(gP[wsOff], gS[wsOff], ...);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
    }
    if (kvSIdx >= kvStart + preKVNum) {         // 阶段 3+4：滞后 tile
#ifdef CUBE
        blockMmadPV(gP[wsOffL], gV, gOTmp[wsOffL], ..., softmaxReady);
        CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#else
        CrossCoreWaitFlag(pvReady);             // AIV: O rescale/累加
        epilogueRescaleO(gO, gOTmp[wsOffL], gOUpdate[...], ...);
#endif
    }
    stackSeqCount++;
}
// 任务级负载均衡：taskIdx = coreIdx; taskIdx += coreNum 跨步领取
```

## 关键修改点

1. **删除主循环内全部 `SyncAll`**，替换为三 flag 定向握手；flag id 用常量（1/2/3）防冲突。
2. **workspace 按核分槽**：每核独占 `SLOTS` 个槽，槽内 S/P/OTmp/Update 分区；寻址公式必须核间不重叠。
3. **循环上界 `+ preKVNum`** 并拆分"当前阶段/滞后阶段"两个 if，保证尾部 PV/rescale 被排空。
4. **初始化预置 SetFlag 清单**必须与主循环 Wait 的事件一一对应，漏一个就死锁或首拍停等。
5. **softmax 输出 P 与 QK 输出 S 复用同一槽位偏移**（读完 S 写 P），省一半 workspace。

## 常见陷阱

- ❌ 槽位数 < PRE_LAUNCH+1 → 写 i+1 覆盖未读完的 i，数据竞争且结果偶发错误（难复现）。
- ❌ 用 `SyncAll` 替代 flag 握手 → 退化为串行，本策略收益清零。
- ❌ 忘记初始化预置 SetFlag → 第一次 WaitFlag 永久阻塞（kernel hang）。
- ❌ AIV 侧 `coreIdx` 忘记除以 `GetSubBlockNum()` → 多 AIV 子核写同一 workspace 槽。
- ❌ 循环上界不加 preKVNum → 尾部 PRE_LAUNCH 个 tile 的 PV/rescale 丢失，输出缺块。

## 代码搜索关键词

```bash
grep -nE "CrossCoreSetFlag|CrossCoreWaitFlag|SyncAll|PRE_LAUNCH|workspace|WORKSPACE|GetSubBlockNum|qkReady|softmaxReady|pvReady" op_kernel/*.cpp op_kernel/*.h
```
