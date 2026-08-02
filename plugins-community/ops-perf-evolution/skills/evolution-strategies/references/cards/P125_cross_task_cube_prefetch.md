---
id: P125
bottlenecks: [no_overlap, partial_overlap]
op_families: [attention, flash_attention, cv_fusion]
complexity: L2
conflicts_with: []
synergizes_with: [P121, P14, P107, P122]
requires: [P121]
has_preconditions: true
has_playbook: true
---

# P125: 跨任务 Cube 预取（任务边界提前发射下一任务首个 QK^T，消除切换空泡）

## 核心思想

CV 流水（P121）的 PRE_LAUNCH 预发射只覆盖**任务内** KV 维；任务边界处流水照样断——当前任务最后一个 PV 还没算完时 Cube 已经无活可干，下一任务的"读 tiling 参数 → Q preload 进 L1 → 首块 K 搬运 → 首次 mmad"全部串行暴露在关键路径上。decode 场景任务小（kvSeqlen 短）、任务多（B×Sq 个），切换空泡占比被任务数线性放大。

把预发射从任务内延伸到**任务间**：KV 主循环的最后一拍（`nIdx` 越过本任务上界）检测 `nextProcess = process + coreNum < taskNum`，**提前读下一任务的 GM tiling 参数并发射其首个 QK^T**（含 Q 常驻 L1 装载），随后才回头补当前任务的尾部 PV。Cube 视角下任务切换完全透明，QK 流水永不断。

三个关键决策：

1. **预取时机 = 主循环尾部一拍**：循环上界写成 `nLoop + UNIT_BLOCK_STACK_NUM`，最后一拍 `nIdx >= nLoop` 时本任务已无 QK 可算，这一拍用来发射下一任务首块。
2. **GM 偏移预计算**：nextProcess 的 `(batch, tokenIdx, kvSeqlen)` 从 GM tiling 表读（P126），`gmOffsetQ/K/V` 当拍算出直接喂 mmad——下一任务正式进入主循环时 `nIdx==0` 分支被跳过（首块已算）。
3. **握手协议不变**：预取的 QK^T 照常 `CrossCoreSetFlag(qkReady)`，Vector 侧无需区分"本任务第 0 块"还是"下任务第 0 块"——ping-pong 槽位由全局 `pingpongIdx` 统一推进。

与 P121 的关系：P121 建任务内流水拓扑；本卡堵上任务边界的最后一段空泡。与 P107 的关系：任务遍历顺序用蛇形（`isForward` 往返）保持 L2 亲和，预取目标和遍历顺序必须一致（预取的必须是**真正下一个**被执行的任务）。

## 代码骨架

```cpp
// === 改造前（任务边界断流）===
for (process = coreIdx; process < taskNum; process += coreNum) {
    readTiling(process); loadQtoL1();            // 串行暴露
    for (nIdx = 0; nIdx < nLoop; nIdx += STACK) {
        QK[nIdx]; Softmax[nIdx]; PV[nIdx - STACK];   // 任务内流水
    }
    PV 尾部排空;                                   // Cube 空转
}

// === 改造后（边界预取）===
for (process = coreIdx; process < taskNum; process += coreNum) {
    for (nIdx = 0; nIdx < nLoop + STACK; nIdx += STACK) {
        if (nIdx < nLoop) { QK(process, nIdx); }     // 本任务 QK
        else if (process + coreNum < taskNum) {      // 尾部一拍：
            readTiling(nextProcess);                 //   预读下一任务参数
            QK(nextProcess, /*nIdx=*/0);             //   预发射首块 QK^T（含 Q→L1）
        }
        if (nIdx >= STACK) PV(process, nIdx - STACK);// 本任务 PV 照常
    }
    // 下一任务进入主循环时 nIdx 从 STACK 起步（首块已算完）
}
```

## 关键修改点

1. **KV 主循环上界 +STACK**：`nIdx < nLoop + UNIT_BLOCK_STACK_NUM`，多跑的一拍专用于预取/排空。
2. **预取分支**：`nIdx >= nLoop && nextProcess < taskNum` 时读下一任务 tiling、算 GM 偏移、调用 QK mmad（其内部 `nIdx==0` 分支顺带完成 Q→L1）。
3. **下一任务主循环跳过首块**：`process == coreIdx` 的特判首块改为"仅当未被上一任务预取时执行"，或统一改为循环从 `nIdx = STACK` 起步、首块永远由上一任务尾部（或 process 循环开头的冷启动特判）发射。
4. **ping-pong 槽位连续性**：`pingpongIdx` 跨任务单调推进，禁止在任务边界清零——否则预取块与正式块的 GM workspace 槽位对不上。

## 常见陷阱

- ❌ 预取了"错误顺序"的下一任务（蛇形遍历下 `nextProcess` 不是简单 +coreNum）→ 预取块的 S 被消费到错误的输出上；`nextRealProcess` 必须复用与正式遍历完全相同的蛇形换算。
- ❌ 任务边界清零 pingpongIdx / 事件 ID → 预取块与后续块槽位冲突，结果随机错。
- ❌ 预取分支忘了 `nextKvSeqlen == 0` 的空任务跳过 → 对空任务发射 mmad，读到非法 GM。
- ❌ 冷启动（第一个任务）也走预取路径 → 首个任务的首块没人发射，整个任务丢失；保留 `process == coreIdx` 冷启动特判。
- ❌ Q preload 依赖的 L1 区被上一任务 PV 的 L1A（P）占用 → 确认 P 的 L1 区与 Q 常驻区不重叠（P122 的 L1 预算划分已保证）。

## 代码搜索关键词

```bash
grep -nE "process|taskNum|formerTaskNum|nextProcess|nLoop|pingpongIdx|CrossCoreSetFlag" op_kernel/*.cpp op_kernel/*.h
```
