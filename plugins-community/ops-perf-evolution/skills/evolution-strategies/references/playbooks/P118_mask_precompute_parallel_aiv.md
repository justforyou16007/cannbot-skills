# P118 Playbook: Causal Mask 独立 AIV Kernel 并行预生成

> **强制流程**。本 Playbook 指导子 agent 把 causal mask 从 attention 主路径剥离为独立 AIV 预生成 kernel，并用按需 dense mask 替代通用压缩表。必须按 Step 1-5 逐步执行。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p118_locations.txt`：

```bash
# 1.1 mask 相关参数与压缩模式
grep -nE "causal|attenMask|maskCompress|COMPRESS|maskS2|2048" \
    shared/original/op_host/*.{cpp,h} > /tmp/p118_locations.txt
# 1.2 主 kernel 内 mask 构造/分支证据
grep -nE "causal|mask|continue" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p118_locations.txt
# 1.3 mask 消费方式（DataCopy 读 mask 的位置）
grep -nE "attenMask|maskGm|DataCopy.*mask" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p118_locations.txt
# 1.4 workspace / GM 分配（mask buffer 落点）
grep -nE "workspace|Workspace|GM_ADDR" \
    shared/original/op_host/*.{cpp,h} >> /tmp/p118_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，列出 mask 语义（0/1 含义）、当前生成位置、消费位置、skv 最大值。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| mask 语义 | ? | 0=保留 / 1=屏蔽（或记录反向） | implementation_note.txt |
| 生成位置 | 主 kernel / host | 独立 AIV kernel | op_kernel/*_mask_kernel.cpp（新增） |
| 压缩模式 | COMPRESS / 2048 表 | NO_COMPRESS_MODE | op_host/*_tiling.cpp:L? |
| mask 行数 | ? | sq * maskGroupSize（配 P116） | op_host/*_tiling.cpp:L? |
| 行字节数 | ? | AlignUp(skv, 4) | op_host/*_tiling.cpp:L? |
| launch 方式 | — | 同 stream 先于 attention kernel | op_host / launch glue:L? |
| UB 上限 | — | kMaxMaskColumns=8192 分行 | op_kernel/*_mask_kernel.cpp |

## Step 3: 代码改造

### 3A. 新增 mask kernel（word 级行生成）

按卡片 `## 代码骨架` 落地，要点：
- `wordsPerRow = (columns + 3) / 4`，UB buffer 按 `kMaxMaskColumns / 4` 个 int32 预分配；
- 行分配 `for (row = GetBlockIdx(); row < rows; row += numBlocks)`；
- 每行：`Duplicate(0x01010101)` → 全保留段 `Duplicate(0, allowed/4)` → `PipeBarrier<PIPE_V>` → 边界 word `SetValue(allowed/4, boundary[(allowed&3)-1])`；
- `allowed = queryRow + (columns - queryRows) + 1`，clamp 到 `[0, columns]`；
- 写出用 `DataCopyPad`（`blockLen = columns` 字节）。

### 3B. Host tiling 切换

```cpp
input.set_attenMaskCompressMode(NO_COMPRESS_MODE);   // 关掉通用 2048x2048 压缩表
input.set_attenMaskS2Size(AlignUpHost(skv, 4));
// mask GM 区域挂进 workspace 布局：maskOffset / maskBytes 规划 + 512B 对齐
```

### 3C. Launch 时序

同 stream 先 launch mask kernel（numBlocks = AIV 核数），再 launch attention kernel；跨 stream 必须 event 同步。

### 3D. 消费端确认

attention kernel 按 dense 行读取（每行 `attenMaskS2Size` 字节），删除主路径内的逐元素 mask 构造 / `causal continue` 分支。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P118 约束核查：
  1. mask 字节语义 = ___（0/1 含义），与消费端一致？ yes/no
  2. right-down 对齐公式 allowed = queryRow + (skv - sq) + 1？ yes/no
  3. 边界余数表 {0x01010100, 0x01010000, 0x01000000} 顺序正确？ yes/no
  4. GQA 合并 M 时 mask 行数 ×G（groupSize 除法）？ yes/no / 未合并说明
  5. mask kernel 全 AIV 核分摊（row += numBlocks）？ yes/no
  6. 生成→消费同 stream 或有 event 同步？ yes/no
  7. skv > 8192 分行处理？ yes/no / skv 上限 = ___
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 独立 mask kernel 存在
(ls modified_files/op_kernel/ | grep -cE "mask"; true)
# 期望: >= 1

# 检查 2: word 级生成技巧
(grep -cE "0x01010101|0x01010100|0x01010000|0x01000000" modified_files/op_kernel/*.cpp; true)
# 期望: >= 3

# 检查 3: 全核行分摊
(grep -cE "row\s*\+=\s*numBlocks|GetBlockIdx\(\).*rows" modified_files/op_kernel/*.cpp; true)
# 期望: >= 1

# 检查 4: 压缩模式关闭
(grep -cE "NO_COMPRESS_MODE|attenMaskCompressMode" modified_files/op_host/*.cpp; true)
# 期望: >= 1

# 检查 5: mask launch 先于 attention
(grep -cE "launch.*mask" modified_files/op_host/*.cpp modified_files/*.cpp; true)
# 期望: >= 1
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P118 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: standalone mask kernel applied (rows=___, no_compress=yes)
Step 4: constraints checked (semantics=___, sync=___)
Step 5: all 5 grep checks passed
```

## 常见陷阱

- 继续用 2048×2048 通用压缩表 → 短序列性能反噬。
- mask kernel 单核生成 → 行数大时生成耗时反超 attention。
- 边界余数表顺序颠倒 → 因果边界偏移，精度难查。
- 跨 stream 无同步 → attention 读到半成品 mask。
- 删主路径 mask 分支时漏删 `causal continue` → 双重 mask 或逻辑冲突。
