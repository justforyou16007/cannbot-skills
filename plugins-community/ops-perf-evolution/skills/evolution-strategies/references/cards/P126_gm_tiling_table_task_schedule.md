---
id: P126
bottlenecks: [tiling_imbalance, scalar_compute]
op_families: [attention, flash_attention]
complexity: L1
conflicts_with: []
synergizes_with: [P107, P116, P125, P122]
requires: []
has_preconditions: true
has_playbook: true
---

# P126: GM Tiling 表驱动变长任务调度（任务列表拍平进 GM，一次 launch 跑全 batch decode）

## 核心思想

decode 场景每个 batch 的 kvSeqlen 都不同，老派做法是 kernel 内按 `totalQ / coreNum` 均匀切分、逐 query 用 `b = qi / (N*S)` 反解坐标：① 每核任务量与 kvSeqlen 无关，长序列核拖尾、短序列核空转，负载天然不均；② 任务参数（batch/token/kvSeqlen）在核内反复做整数除法反解；③ 变长只能靠 padding 或多次 launch。

把调度权交给 host：**tiling 阶段把任务列表拍平成 GM 上的定长记录表**——每任务一条 `(batch, tokenIdx, kvSeqlen)`，kernel 以 `process = coreIdx; process < taskNum; process += coreNum` 轮询领取，按 `tilingHeadSize + tilingParaSize * process` 直接索引：

1. **变长原生支持**：kvSeqlen 是每任务记录的一个字段，循环上界 `nLoop = ceil(kvSeqlen / blockSize)` 现场算，零 padding、零除法反解。
2. **任务粒度对齐头合并**：配合 P116 头合并后任务 = `(batch, token)` 而非 `(batch, head, seq)`，任务数从 B×S×Nq 降到 B×S，调度开销同步下降。
3. **负载均衡两件套**：任务按 kvSeqlen 降序排进表（host 侧排序，长任务先被领走）+ 核间 stride 轮询；追求 L2 亲和再叠加蛇形遍历（P107）。
4. **一次 launch**：全 batch 一个 kernel，无 per-batch launch 开销；taskNum 也是表里一个字段。

## 代码骨架

```cpp
// === Host 侧（tiling）===
// tiling[HEAD]  = {numHeads, blockSize, headSize, paraSize, taskNum, ...}
// tiling[HEAD + paraSize*i] = {batch_i, tokenIdx_i, kvSeqlen_i}   // i 按 kvSeqlen 降序

// === Kernel 侧 ===
// 改造前：均匀切分 + 除法反解
int64_t blkLen = (totalQ + numBlocks - 1) / numBlocks;
for (qi = bid * blkLen; qi < min(q0+blkLen, totalQ); qi++) {
    b = qi / (Nq*S); h = (qi % (Nq*S)) / S; s = qi % S;   // 反解坐标
    ...  // 每核任务量与 kvSeqlen 无关 → 负载不均
}

// 改造后：GM 表轮询领取
uint32_t taskNum = gTiling.GetValue(TILING_TASKNUM);
for (uint32_t process = coreIdx; process < taskNum; process += coreNum) {
    uint32_t off = tilingHeadSize + tilingParaSize * process;
    uint32_t curBatch    = gTiling.GetValue(off);
    uint32_t curTokenIdx = gTiling.GetValue(off + 1);
    uint32_t kvSeqlen    = gTiling.GetValue(off + 2);
    if (kvSeqlen == 0) continue;                 // 空任务保护
    uint32_t nLoop = (kvSeqlen + blockSize - 1) / blockSize;
    ...  // 任务量 ∝ kvSeqlen，长任务先排先领 → 均衡
}
```

## 关键修改点

1. **Host tiling 生成任务表**：遍历 `(batch, token)` 收集 kvSeqlen（从 seqLens 数组），按 kvSeqlen 降序写入 GM tiling 区；taskNum、tilingParaSize 写入表头。
2. **Kernel 删除均匀切分与坐标反解**：`blkLen` / `qi/(Nq*S)` 全部删除，改为 process 轮询 + 表项直读。
3. **GM 偏移由表项直接算出**：`gmOffsetK = curBatch * fullKvSeqlen * embed` 等，无需 head 维（头已合并进 M）。
4. **空任务保护**：`kvSeqlen == 0 → continue`（decode 允许 0 长度 KV 的 batch）。
5. **workspace 按核分配**：S/P/OTmp 的 GM workspace 按 `coreIdx * TMP_SIZE` 寻址，与任务表解耦，任何核领任何任务都安全。

## 常见陷阱

- ❌ 任务表仍按 batch 顺序（未按 kvSeqlen 降序）→ 长任务最后被领走，尾核拖尾抵消收益。
- ❌ `tilingParaSize` host/kernel 两侧不一致 → 读到错位记录，batch/token 张冠李戴（结果串台）。
- ❌ 表项用 int32 存 kvSeqlen 但 kernel 按 uint64 读 → 越界读到下一条记录。
- ❌ 删除除法反解时漏掉输出偏移的 layout 分支（BSH/BNSD）→ 输出写错位置；`inputLayout` 分支必须保留。
- ❌ 任务数 > 表容量（tiling GM 区太小）→ host 侧按 `maxBatch * maxSq` 预留并断言。

## 代码搜索关键词

```bash
grep -nE "blockLength|blkLen|totalQ|GetValue\(offset|tilingParaSize|taskNum|formerTaskNum|process" op_kernel/*.cpp op_kernel/*.h ../op_host/*.cpp
```
