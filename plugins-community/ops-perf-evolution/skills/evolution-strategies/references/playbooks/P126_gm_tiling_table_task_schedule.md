# P126 Playbook: GM Tiling 表驱动变长任务调度

> **强制流程**。本 Playbook 指导子 agent 把 kernel 的"均匀切分 + 除法反解坐标"调度改为"host 生成 GM 任务表 + kernel 轮询领取"，支持变长 kvSeqlen decode 一次 launch。涉及 host tiling 与 kernel 两侧改动，必须按 Step 1-5 逐步执行，严禁跳步。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p126_locations.txt`：

```bash
# 1.1 kernel 内均匀切分与坐标反解
grep -nE "blockLength|blkLen|GetBlockIdx|/ \(|% \(" \
    shared/original/op_kernel/*.{cpp,h} > /tmp/p126_locations.txt
# 1.2 kernel 内 GM 偏移计算（依赖 b/h/s 反解结果的位置）
grep -nE "gmOffset|offQn|offQr|offKn|offV|offOut" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p126_locations.txt
# 1.3 host tiling 结构与入口
grep -nE "TilingData|RegisterTiling|tiling" \
    shared/original/op_host/*.{cpp,h} 2>/dev/null >> /tmp/p126_locations.txt || \
grep -rnE "launch_|tiling" shared/original/op_kernel/*launch* >> /tmp/p126_locations.txt
# 1.4 变长相关输入（seqLens / actualSeqLen）
grep -nE "seqLen|actualSeq|kvSeq" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p126_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，明确：当前切分方式、坐标反解有几处、host tiling 在哪个文件、kvSeqlen 从哪个输入/属性获得。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 任务切分 | totalQ/coreNum 均匀切 | host 任务表 + process 轮询 | op_host/*:L? / op_kernel/*:L? |
| 坐标获取 | qi/(N*S) 除法反解 | 表项 (batch, token, kvSeqlen) 直读 | op_kernel/*:L? |
| KV 循环上界 | 固定 Skv / 最大值 | nLoop = ceil(kvSeqlen/blockSize) | op_kernel/*:L? |
| 任务顺序 | 顺序 | kvSeqlen 降序（host 排序） | op_host/*:L? |
| 表寻址 | 无 | tilingHeadSize + tilingParaSize × process | op_kernel/*:L? |
| 空任务 | 无保护 | kvSeqlen==0 → continue | op_kernel/*:L? |

## Step 3: 代码改造

### 3A. 形态识别

- **Canonical**：per-query 均匀切分 + 三重除法反解 (b, h, s)。
- **变体 A**：已按 (b, s) 切分但 kvSeqlen 固定取最大值 → 只加表与变长循环界。
- **变体 B**：无 host tiling 文件（launch 直传标量）→ 表通过新增 GM tiling buffer 传入，host 侧 launch wrapper 填表。

### 3B. Canonical Template

```cpp
// Host 侧（tiling 或 launch wrapper）
uint32_t taskNum = 0;
for (b, s in batch × sq) { tasks[taskNum++] = {b, s, seqLens[b]}; }
sort(tasks, by kvSeqlen DESC);                    // 长任务先领
tilingBuf[HEAD + PARA_SIZE*i + {0,1,2}] = tasks[i];
tilingBuf[TILING_TASKNUM] = taskNum;

// Kernel 侧
// 改造前
int64_t blkLen = (totalQ + numBlocks - 1) / numBlocks;
for (qi = bid*blkLen; ...) { b = qi/(Nq*S); h = ...; s = ...; }

// 改造后
uint32_t taskNum = gTiling.GetValue(TILING_TASKNUM);
for (uint32_t process = coreIdx; process < taskNum; process += coreNum) {
    uint32_t off = tilingHeadSize + tilingParaSize * process;
    uint32_t curBatch = gTiling.GetValue(off), curTokenIdx = gTiling.GetValue(off+1);
    uint32_t kvSeqlen = gTiling.GetValue(off+2);
    if (kvSeqlen == 0) continue;
    uint32_t nLoop = (kvSeqlen + blockSize - 1) / blockSize;
    // GM 偏移由表项直算；inputLayout(BSH/BNSD) 分支保留
}
```

### 3C. Variant Notes

- 变体 B（无 host tiling）：GM tiling buffer 用 `uint32_t` 数组，表头字段与 kernel 共享常量头文件（如 kernel_common.hpp），防止两侧字段漂移。
- 表容量：host 按 `maxBatch × maxSq` 预留，taskNum 超容量时 host 侧报错而非截断。
- workspace 寻址与任务表解耦：S/P/O 中间 buffer 一律 `coreIdx * TMP_SIZE` 按核分配。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P126 约束核查：
  1. host/kernel 的 tilingParaSize、字段偏移常量同源（共享头文件）？ yes/no
  2. 任务表已按 kvSeqlen 降序？ yes/no
  3. 除法/取模坐标反解已全部删除？ yes/no
  4. kvSeqlen==0 空任务保护存在？ yes/no
  5. inputLayout（BSH/BNSD）分支在 GM 偏移计算中保留？ yes/no
  6. 表容量 ≥ maxBatch × maxSq 且 host 有溢出断言？ yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 任务表寻址存在
(grep -cE "tilingParaSize|offsetTiling|tilingHeadSize" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 2: process 轮询循环存在
(grep -cE "process .*coreNum|process \+= " modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 3: 除法反解已删除
(grep -cE "rem / |rem % |/ \(Nq|blkLen" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: == 0

# 检查 4: 空任务保护存在
(grep -cE "kvSeqlen == 0" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 5: 变长循环界存在
(grep -cE "nLoop = .*kvSeqlen" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P126 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: gm task table applied (sorted_desc=yes, div_decode_removed=yes, empty_guard=yes)
Step 4: constraints checked (field_sync=___, capacity=___)
Step 5: all 5 grep checks passed
```

## Step 6: Known Pitfalls

- 输出串台（batch i 的结果写到 batch j）→ host/kernel 字段偏移不一致，统一到共享常量头。
- 尾核拖尾依旧 → 任务表忘排序，确认 host 侧按 kvSeqlen 降序。
- 部分 batch 丢失 → 表容量不足被静默截断，host 加断言。
- 变长 batch 结果错 → KV 循环上界仍用固定 Skv，改 ceil(kvSeqlen/blockSize)。
- 精度回归但调度正确 → 删除反解时把 output 偏移的 inputLayout 分支一并删了，恢复分支。
