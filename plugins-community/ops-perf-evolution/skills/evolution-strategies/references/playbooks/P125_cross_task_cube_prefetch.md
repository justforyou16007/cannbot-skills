# P125 Playbook: 跨任务 Cube 预取

> **强制流程**。本 Playbook 指导子 agent 在 CV 流水（P121 已实施）的多任务 kernel 上，把 QK 预发射从任务内延伸到任务间，消除任务切换的 Cube 空泡。必须按 Step 1-5 逐步执行，严禁跳步。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p125_locations.txt`：

```bash
# 1.1 任务循环结构（process 循环、任务数、任务参数读取点）
grep -nE "process|taskNum|formerTaskNum|coreNum" \
    shared/original/op_kernel/*.{cpp,h} > /tmp/p125_locations.txt
# 1.2 任务参数 / GM 偏移计算点（每任务开头重复执行的部分）
grep -nE "gmOffset|curBatch|curTokenIdx|kvSeqlen|GetValue" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p125_locations.txt
# 1.3 KV 主循环边界与 QK/PV 调用点
grep -nE "nIdx|nLoop|blockMmad|BlockMmad" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p125_locations.txt
# 1.4 ping-pong 槽位推进（是否任务边界清零）
grep -nE "pingpongIdx|PingPongFlag|taskPingPong" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p125_locations.txt
# 1.5 遍历顺序（蛇形/顺序）
grep -nE "isForward|bigProcess|realProcess" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p125_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，明确：任务循环形式、每任务开头的串行初始化有哪些（tiling 读/Q preload/首块 K 搬运）、KV 主循环上界、pingpongIdx 是否跨任务连续、遍历顺序。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| KV 主循环上界 | nLoop | nLoop + STACK（多一拍尾部） | op_kernel/*:L? |
| 下一任务参数读取 | 下一任务开头串行 | 本任务尾部一拍预读 | op_kernel/*:L? |
| 下一任务首块 QK | 下一任务 nIdx==0 发射 | 本任务尾部预发射（含 Q→L1） | op_kernel/*:L? |
| 下一任务循环起点 | nIdx=0 特判首块 | 跳过首块（已被预取） | op_kernel/*:L? |
| pingpongIdx | 任务边界清零 / 连续 | 跨任务单调推进 | op_kernel/*:L? |
| 冷启动 | 无 | process==coreIdx 保留首块发射特判 | op_kernel/*:L? |

## Step 3: 代码改造

### 3A. 形态识别

- **Canonical**：process 循环 + 每任务开头"读参数→loadQ→首块 QK"串行 + 任务内 KV 流水（P121 拓扑）。
- **变体 A**：无 CV 流水的串行多任务（先回 P121）。
- **变体 B**：单任务多 KV 块（无任务边界）→ **不适用**。

### 3B. Canonical Template

```cpp
// 改造前：任务边界断流
for (process = coreIdx; process < taskNum; process += coreNum) {
    readTiling(process); loadQtoL1();
    for (nIdx = 0; nIdx < nLoop; nIdx += STACK) { QK(nIdx); PV(nIdx - STACK); }
}

// 改造后：尾部一拍预取下一任务
for (process = coreIdx; process < taskNum; process += coreNum) {
    realProcess = serpentine(process);           // 与正式遍历同一换算
    if (process == coreIdx) {                    // 冷启动：首任务首块照常发射
        QK(realProcess, /*nIdx=*/0);
    }
    for (nIdx = STACK; nIdx < nLoop + STACK; nIdx += STACK) {
        if (nIdx < nLoop) QK(realProcess, nIdx);
        uint32_t nextProcess = process + coreNum;
        if (nIdx + STACK > nLoop + STACK - 1 && nextProcess < taskNum) {
            nextReal = serpentine(nextProcess);  // 预取目标必须走同一蛇形换算
            readTiling(nextReal);                // 预读 (batch, token, kvSeqlen)
            if (nextKvSeqlen > 0) QK(nextReal, /*nIdx=*/0);   // 预发射（含 Q→L1）
        }
        if (nIdx >= STACK) PV(realProcess, nIdx - STACK);     // 本任务 PV 照常
    }
    pingpongIdx 保持单调推进;                     // 任务边界禁止清零
}
```

### 3C. Variant Notes

- 蛇形遍历：`nextRealProcess` 必须复用正式遍历的 `bigProcess/isForward` 换算，禁止简化为 `process + coreNum` 直接索引。
- 空任务（kvSeqlen==0）：预取分支照常跳过 mmad，但 tiling 读取无副作用可保留。
- 首任务冷启动的 QK 与尾部预取的 QK 共用同一调用路径（nIdx==0 分支含 Q→L1），避免两份代码漂移。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P125 约束核查：
  1. 预取目标任务 = 正式遍历的下一任务（蛇形换算一致）？ yes/no
  2. pingpongIdx 跨任务单调推进、边界无清零？ yes/no
  3. 冷启动（process==coreIdx）首块发射保留？ yes/no
  4. 预取分支含 nextKvSeqlen==0 空任务保护？ yes/no
  5. 预取 QK 的 GM workspace 槽位与 Vector 侧消费槽位公式一致？ yes/no
  6. 下一任务主循环不再重复发射首块（无双重发射）？ yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 跨任务预取变量存在
(grep -cE "nextProcess|nextRealProcess|nextGMOffset" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 2: 主循环上界含尾部一拍
(grep -cE "nLoop \+ .*STACK|nLoop \+ UNIT_BLOCK" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 3: 冷启动特判保留
(grep -cE "process == coreIdx" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 4: 跨核握手未删除
(grep -cE "CrossCoreSetFlag|CrossCoreWaitFlag" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 2

# 检查 5: pingpongIdx 无任务边界清零（process 循环体内无 =0 赋值）
(grep -cE "pingpongIdx = 0|pingpongIdx=0" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: 仅允许出现在 process 循环体外的初始化处（0 或 1 处，且在循环前）
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P125 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: cross-task prefetch applied (tail_slot=yes, serpentine_consistent=yes, cold_start_kept=yes)
Step 4: constraints checked (pingpong_continuous=___, empty_task_guard=___)
Step 5: all 5 grep checks passed
```

## Step 6: Known Pitfalls

- 输出按 batch 错位/串台 → 预取目标没用蛇形换算，修 `nextRealProcess`。
- 结果随机错、偶发 NaN → pingpongIdx 任务边界被清零，预取块与正式块抢同一 GM 槽。
- 首 batch 输出全 0 / 丢失 → 冷启动特判被删，首任务首块无人发射。
- 最后一个 batch 后多算一块 → 预取分支缺 `nextProcess < taskNum` 上界判断。
- 性能无提升 → 预取分支被放在 `nIdx < nLoop` 内每拍都跑（应只在尾部一拍），或 Q preload 的 L1 区与 P 区重叠互相驱逐。
