---
id: P134
bottlenecks: [tiling_imbalance, no_overlap, mte2_stall]
op_families: [ssm, linear_attention, recurrent, matmul, cv_fusion]
complexity: L1
conflicts_with: []
synergizes_with: [P132, P133, P136]
requires: [P132]
has_preconditions: true
has_playbook: true
---

# P134: chunk-group 任务划分（group 间串行 + group 内 (nv, chunk) 多核并行）

## 核心思想

P132 三阶段后，块间状态递推（Stage2）强制 **chunk 之间串行**——第 k+1 个 chunk 的状态依赖第 k 个。若把整个序列一次性分给所有核，每核都要等前核的状态，退化成单核串行。**chunk-group 任务划分**把序列切成多个 group，满足两个约束：

```
· group 之间串行（状态递推依赖）      —— 每组算完才进下一组
· group 内并行（(nv, chunk) 独立任务） —— 组内所有 chunk 按核均分
```

参考实现的任务模型（GDR）：

```
blockDim = aiCoreNum                       # 核数 = 物理核数
maxGroupLength = P_NUM × aiCoreNum × C     # P_NUM=2, C=64 → 一组覆盖 2×核数×64 行

for bid in batches:
    for pos in seqStart..seqEnd step maxGroupLength:    # chunk-group 循环（串行）
        cg = {startPos: pos, length: min(maxGroupLength, seqEnd-pos)}
        RunStage1(cg)   # totalChunk = nv × num_chunk 按 coreIdx 均分（former/tail）
        SyncAll
        RunStage2(cg)   # nvPerCore = ceil(Nv/core) 切核，核内串行 cId
        SyncAll
        RunStage3(cg)   # totalChunks = nv × chunkNum 融合均分
        SyncAll
```

**收益**：核数吃满靠 group 内 (nv, chunk) 并行；状态串行只发生在 64 维块间，单核负担最小。三阶段各按自己的任务粒度分配（Stage1/3 全并行、Stage2 串行但只递推小状态）。

## 代码骨架

```cpp
// === 改造前：全序列一次分核（核间状态依赖 → 退化成串行）===
// 每核处理一段序列，但 chunk 状态跨核依赖，实际串行

// === 改造后：chunk-group 任务划分 ===
void Process() {
    for (int32_t bid = 0; bid < batchNum; bid++) {
        seqStart = actualSeqLengths.cumsum[bid]; seqEnd = ...;
        for (int32_t pos = seqStart; pos < seqEnd; pos += maxGroupLength) {
            cg = {pos, min(maxGroupLength, seqEnd - pos)};
            Stage1::Run(cg);    // totalChunk = nv * chunkNum，按 coreIdx 均分
            SyncAll();
            Stage2::Run(cg);    // nvPerCore = ceil(Nv / coreNum)，核内串行 cId
            SyncAll();
            Stage3::Run(cg);    // totalChunks = nv * chunkNum，融合均分
            SyncAll();
        }
    }
}
// maxGroupLength = 2 × aiCoreNum × 64   （P_NUM=2，C=64）
```

## 关键修改点

1. **maxGroupLength 计算**：`P_NUM × aiCoreNum × C`。组内行数 = 核数 × 每核可并行 chunk 数 × chunk 大小；P_NUM 是流水并行度（GDR=2）。
2. **Stage1/3 全并行**：任务 = `nv × num_chunk`（组内所有 chunk × 所有 head），按 `coreIdx` former/tail 均分；`taskId` 反解 `(nvId, cgId)`。
3. **Stage2 串行但轻**：`nvPerCore = ceil(Nv / coreNum)` 切核，核内串行遍历 `cId`；每 chunk 只做 (Dv,Dk) 小矩阵递推。
4. **batch 维度外层循环**：每个 batch 的序列独立状态，外层按 batch 循环，内层 chunk-group。
5. **尾组变长**：最后一个 group 可能不满 `maxGroupLength`，Stage 内按 `length` 特化（`lastChunkSize = seq % C`）。

## 常见陷阱

- ❌ 全序列一次分核 → 跨核状态依赖把并行度打回单核；必须 chunk-group 化。
- ❌ maxGroupLength 过小 → group 数多、SyncAll 频繁，启动开销吃掉并行收益。
- ❌ Stage2 也按 (nv,chunk) 全并行 → 状态串行依赖被打破，结果错误；Stage2 只能核内串行 cId。
- ❌ 忽略 batch 维度 → 多 batch 时状态初始化错乱；每个 batch 独立 initial_state。
- ❌ 尾组不特化 → 伪计算吃满 maxGroupLength；用实际 length 裁剪。

## 代码搜索关键词

```bash
grep -nE "blockDim|maxGroupLength|chunkGroup|chunk_group|cg|num_chunk|nvPerCore|coreIdx|former|tail|SyncAll|GetBlockIdx" op_kernel/*.{cpp,h} op_host/*.cpp
```
