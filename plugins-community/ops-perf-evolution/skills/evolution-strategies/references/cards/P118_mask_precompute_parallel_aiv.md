---
id: P118
bottlenecks: [no_overlap, mte2_stall]
op_families: [attention, flash_attention, broadcast_mask]
complexity: L1
conflicts_with: []
synergizes_with: [P14, P116]
requires: []
has_preconditions: true
has_playbook: true
---

# P118: Causal Mask 独立 AIV Kernel 并行预生成（按需 Dense 替代通用压缩表）

## 核心思想

Causal mask 不该占用 attention 主路径的计算/搬运时间。两个正交要点：

1. **按需 dense 替代通用压缩表**：只生成当前 shape 需要的 `s1 × s2` dense right-down mask（byte 级 0/1），而不是预生成一张通用 2048×2048 压缩表再解压——短序列场景下压缩表的生成 + 读取开销占主导地位，dense 直达更省。
2. **mask 生成下沉独立 vector kernel**：单独的轻量 kernel 用**全部 AIV 核**并行生成 mask 到 GM workspace，attention cube kernel（AIC）直接 DataCopy 消费。mask kernel 计算量极小（每行一次 Duplicate + 边界 word 修正），与主 kernel 串行 launch 也只占零头；异构分工（AIV 生成 / AIC 消费）避免 mask 逻辑挤占 cube kernel 的 UB 与流水。

行生成用 word 级技巧：每行先 `Duplicate(0x01010101)` 填满（全屏蔽），再对允许区间的前 `allowed/4` 个 word 写 0（全保留），边界 word 按余数写 `0x01010100 / 0x01010000 / 0x01000000`——纯 vector 两次 Duplicate + 一次 SetValue，无逐元素循环。

## 代码骨架

```cpp
// === 独立 mask kernel（全 AIV 并行，行 stride 分配）===
__global__ __aicore__ void causal_mask_kernel(
    GM_ADDR mask, int64_t rows, int64_t columns, int64_t groupSize, int64_t numBlocks)
{
    // 每行 wordsPerRow 个 int32，byte 语义：0=保留，1=屏蔽
    const int64_t wordsPerRow = (columns + 3) / 4;
    for (int64_t row = GetBlockIdx(); row < rows; row += numBlocks) {
        auto local = outQueue.AllocTensor<int32_t>();
        Duplicate(local, 0x01010101, wordsPerRow);            // 默认全屏蔽
        const int64_t queryRow = row / groupSize;             // GQA 合并 M 时 G 行共享
        int64_t allowed = queryRow + (columns - queryRows) + 1;  // right-down 对齐
        allowed = min(max(allowed, 0), columns);
        if (allowed / 4 > 0) Duplicate(local, 0, allowed / 4); // 全保留段
        PipeBarrier<PIPE_V>();
        const int32_t boundary[3] = {0x01010100, 0x01010000, 0x01000000};
        if ((allowed & 3) != 0) local.SetValue(allowed / 4, boundary[(allowed & 3) - 1]);
        // EnQue → DeQue → DataCopyPad 写出
    }
}

// === Host tiling：NO_COMPRESS + 行字节数对齐 4 ===
input.set_attenMaskCompressMode(NO_COMPRESS_MODE);   // 禁用通用 2048x2048 压缩表
input.set_attenMaskS2Size(AlignUp(skv, 4));          // 行字节数 4B 对齐（word 语义）
maskRows = isCausal ? sq * maskGroupSize : 0;        // 配合 P116：合并 M 后 ×G
```

## 关键修改点

1. **mask 语义先确认**：本配方 `0=保留 / 1=屏蔽`；若 kernel 消费端语义相反，Duplicate 初值与边界字全部取反。
2. **right-down 对齐**：`allowed = queryRow + (skv - sq) + 1`，Sq≠Skv 时因果边界整体右移，别写成左上对齐。
3. **groupSize 行共享**：GQA grouped-M（P116）后 mask 行数 = `sq * G`，同一 query row 的 G 个 mask 行内容相同（`queryRow = row / groupSize`）。
4. **压缩模式关闭**：`attenMaskCompressMode = NO_COMPRESS_MODE`，并保证消费端按 dense 行读取（`attenMaskS2Size = AlignUp(skv, 4)`）。
5. **launch 时序**：mask kernel 与 attention kernel 同 stream 串行即安全；跨 stream 并行需显式 event 同步。
6. **超长 skv 分行**：单行 word 数受 UB buffer 限制（参考常量 `kMaxMaskColumns = 8192`），超出需按行分段生成。

## 常见陷阱

- ❌ 继续生成 2048×2048 通用压缩表：短序列（skv≪2048）下表生成 + 解压读取 dominated，性能反噬。
- ❌ mask kernel 用单核生成：行数 `sq*G` 可达数千，必须 `row += numBlocks` 全核分摊。
- ❌ 边界 word 余数表写错：`allowed & 3` ∈ {1,2,3} 分别对应 `0x01010100 / 0x01010000 / 0x01000000`，顺序颠倒会使因果边界偏移 1~2 列，精度挂得很难查。
- ❌ mask kernel 与 attention 跨 stream 无同步：attention 读到未写完的 mask。

## 代码搜索关键词

```bash
# 检测 causal mask 现状（P118 适用信号）
grep -nE "causal|attenMask|maskCompress|2048" op_host/*.cpp op_kernel/*.cpp
# 检测 mask 是否在主 kernel 内逐元素生成（下沉信号）
grep -nE "if.*causal.*continue|mask.*SetValue|ComputeMask|CreateMask" op_kernel/*.cpp
# 检测是否已独立 mask kernel（已下沉则不适用）
grep -nE "mask_kernel|MaskKernel|maskOffset|launch.*mask" op_kernel/*.cpp op_host/*.cpp
```
