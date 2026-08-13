---
id: X39
origin: discovered
discovered_round: 1
discovered_from: round_1/parallel_2 (GqaSparseFwd evolution, open_exploration)
base_speedup: 全量 50-case 精度过；大 case case39 2.297x / case40 1.592x / case41 1.353x（vs PyTorch），geomean 0.405->0.447x（+10%）
---

# Strategy X39: host 侧全写 workspace 免零初始化（at::zeros → at::empty）

## 核心思路
算子 host 侧每次 forward 调用分配大块 workspace（及输出 buffer）时，若 kernel 会**写满全部区域**（无读前未写），
用 `at::empty` 替代 `at::zeros`，消除每次调用对数百 MB~GB 级内存的 memset（清零写）。
大 case 下 memset 是 memory-bound 地板，消除后直接降低每次调用的固定成本。

## 适用场景
- host 分配大 workspace / 输出 buffer，且 kernel 顺序写满所有区域（无读前未写）
- 大 shape case 的 memory-bound 地板（实测 1.35-2.3x 提升）
- 与"动态核数"（blockDim = min(24, totalTasks)）配合：小 case 免拉满核、大 case 免 memset

## 实现要点
1. 审查 workspace 各分区（如 wsK/wsV/wsS/wsP/wsO/wsMeta）是否全部被 kernel 写入后才被读——确认无读前未写
2. `at::zeros({totalBytes})` → `at::empty({totalBytes})`；输出 buffer 同理（model_new 只读实际写的行时安全）
3. 动态核数 `blockDim = min(coreNum, totalTasks)`，小任务不拉起空转核
4. 注意：**不能**对部分写/依赖零初始化的区域用 empty（会引入垃圾值 NaN/未定义行为）——必须逐区域审计

## 约束 / 红线
- 仅当 kernel 写满全部分配区域时安全；有 padded/未写行被后续读取时禁用
- 这是 host 侧分配优化，不改 kernel 计算逻辑；与 P54（launch 合并）互补但不同（P54 减 launch 次数，本策略减 memset 成本）

## 来源
自动发现于 GqaSparseFwd 进化 round_1/parallel_2（open_exploration 节点 x0）：
workspace at::empty + output at::empty + 动态核数 + tiling.dtype 正确性修复，50/50 精度过，大 case 1.35-2.3x。
