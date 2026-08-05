# P127 Playbook: Conversion 模式特化路由落地 SOP

> **强制流程**。把 layout-transform 算子从"通用 gather 主路径"改造为 R1→R5 模式特化路由。按 Step 1-4 逐步执行，严禁跳步。

## Step 1: 现状取证

```bash
# 1.1 确认 layout-transform 语义与当前实现形态
grep -nE "permute|transpose|Permute|Transpose" op_kernel/*.{cpp,h} op_host/*.cpp
# 1.2 逐元素索引证据（generic gather 主路径标志）
grep -nE "GetValue|SetValue|/ *%|% *[^=]" op_kernel/*.{cpp,h}
# 1.3 现有搬运调用（是否已 tile-based）
grep -nE "DataCopyPad|DataCopyExtParams|Gather" op_kernel/*.{cpp,h}
```

**交付物**：写入 `implementation_note.txt` "P127 Step 1"：当前是哪种 permutation、是否单一 generic kernel、逐元素索引有几处。

## Step 2: R1+R2 静态判定（零拷贝机会，最先做）

| 检查 | 方法 | 命中动作 |
|---|---|---|
| identity permutation | dims == 0..n-1？ | 直接 view，不启动 kernel |
| 仅移动 size-1 维 | 被移动的轴 size 全为 1？ | 直接 view |
| 连续维合并 | 置换后仍相邻的轴合并 | 输出 mergedShape/mergedDims，降 rank 后进 Step 3 |

**交付物**：记录 R1/R2 判定结果；若 R1 命中，本算子优化结束（返回 view 即为最优）。

## Step 3: R3+R4 模式分发设计

填模式分发表（不填完不得进入 Step 4）：

| 置换模式 | 判定条件（基于 mergedDims/mergedShape） | 专用 kernel | 搬运方式 |
|---|---|---|---|
| 2D transpose | mergedDims == {1,0} | transpose_2d | P26 stride 零拷贝（srcStride=0 连读 + dstStride 跳写） |
| batch transpose | 批内两轴交换；batch==1 降级 2D（R3） | batch_transpose | 2D tile + batch 外循环 |
| swap-adjacent | 相邻两维交换 | swap_adjacent | tile 内转置 |
| small-channel | C≤16 通道重排 | 走 P114 路径 | TransDataTo5HD+vnchwconv+Gather |
| reverse dims | 轴整体逆序 | reverse_dims | 倒序 tile 搬运 |
| fallback | 以上都不命中 | generic_permute | 1D gather，正确性优先 |

约束：每个专用 kernel grid ≤ 物理核数，kernel 内循环多 block；tile 参数（BLOCK_M/BLOCK_N）列入调优空间。

## Step 4: 落地与验证

1. 按 Step 3 表格实现分发逻辑与专用 kernel
2. **两条禁令自查**：分发逻辑无单一通用主路径；每个专用 kernel 内无逐元素 div/mod、无 GetValue/SetValue
3. 正确性：每个模式分支至少一个 case；fallback 分支必须有一个罕见 permutation case
4. 性能：对比改造前逐元素实现的搬运耗时；tile 参数做一轮 autotune 扫参
