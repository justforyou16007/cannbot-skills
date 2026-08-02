---
id: P120
bottlenecks: [compute_bound, scalar_compute]
op_families: [attention, flash_attention, cv_fusion]
complexity: L2
conflicts_with: []
synergizes_with: [P13, P14, P119, P121, P122]
requires: []
has_preconditions: true
has_playbook: true
---

# P120: FA 向量点积模拟 Matmul → Cube mmad 重写（QK^T / PV 上 Cube）

## 核心思想

老版 MHA（纯向量 v46，910B3）用 `Mul + ReduceSum` 逐 `(qi, kj)` 对做 QK^T 点积、用标量 `Muls + Add` 逐元素累加 PV——**用 Vector 单元模拟矩阵运算，性能在架构层面被封顶**（Cube 的 fp16 算力比 Vector 高一个数量级以上）。同时它把 Q/K/V 全部 `Cast` 成 fp32 再算，UB 流量翻倍；tile 只有 BM=16/BN=32。

重写为 Cube mmad 的四个关键决策（新版 FIA SplitFuse 移植路线）：

1. **输入 dtype 直接进 Cube**：Q/K/V 保持 fp16/bf16 进 mmad，**不要先 Cast fp32**；fp32 只出现在 L0C 输出 S、online softmax 状态和 O 累加链。
2. **K 按 ColumnMajor 读取 = 免转置 K^T**：QK^T 的 B 矩阵就是 K 的转置。GM 上 K 以行主序存储，按列主序视图读取，经 `Nd2NzParams` 一次 DMA 转为 nZ 分形进 L1，L1→L0B `ifTranspose=false` 直送——零显式 transpose。
3. **Q preload 常驻 L1**：每个 task 开头 `loadQGM` 把 Q tile（128 行）一次性搬入 L1 并常驻，KV 循环只搬 K/V；L0A/L0B/L0C 全部双缓冲。
4. **P 回写输入 dtype 喂第二级 mmad**：online softmax 输出的 P 从 fp32 `DownCast` 回 fp16/bf16（bf16 用 `CAST_RINT`、fp16 用 `CAST_NONE`）写 GM workspace，PV mmad 直接消费，O 以 fp32 在 L0C/GM 累加。

与 P13 的关系：P13 是"标量/向量模拟矩阵运算 → MIX CV 重写"的**通用判定卡**；本卡是判定通过后在 **attention 场景的具体落地配方**（layout 选型、分形搬运、精度链路、tile 基准值）。

## 代码骨架

```cpp
// === 改造前（老版 v46：向量点积 + 全量 fp32 Cast）===
for (int32_t qi = 0; qi < qr; qi++)
    for (int32_t kj = 0; kj < kr; kj++) {
        Mul(rw, qf[qi*Df], kf[kj*Df], Df);          // 逐对向量乘
        ReduceSum(sp[so+kj*P8], rw, rt, Df);        // 逐对归约 = 一个 score
    }
// PV：标量读 p，逐元素 Muls+Add 累加（同样 Vector 模拟 matmul）

// === 改造后（Cube mmad + 分形 layout）===
using LayoutQ = layout::RowMajor;                    // A=Q:  GM→L1 nd2nz→zN
using LayoutK = layout::ColumnMajor;                 // B=K:  列主序读取→nZ，天然 K^T
using L1TileShapeQK  = GemmShape<128, 128, 128>;     // L1 tile: M/N/K
using L0TileShapeQK  = GemmShape<128, 128, 128>;     // L0 tile
using BlockMmadQK = Gemm::Block::BlockMmad<MmadAtlasA2FAIQK<...>,
                    L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;  // S=fp32
using L1TileShapePV  = GemmShape<128, 128, 256>;     // PV 的 K 维 = KV stack 256
using BlockMmadPV = Gemm::Block::BlockMmad<MmadAtlasA2FAIPV<...>,
                    L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>; // OTmp=fp32

blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQ, rowNum, qNBlockSize, qHeads); // Q 常驻 L1
// KV 循环内：blockMmadQK(...) 只搬 K；softmax(AIV)；blockMmadPV(...) 消费 P(fp16)
```

## 关键修改点

1. **删除全部逐对 `Mul+ReduceSum` / 标量 `Muls+Add` 矩阵模拟代码**，替换为 mmad 调用；score 矩阵 S 由 L0C 以 fp32 fixpipe 到 GM workspace。
2. **精度链路重排**：Q/K/V 进 cube 前不做 fp32 Cast；softmax 在 fp32 的 S 上做；P downcast 回输入 dtype；O 累加 fp32，收尾一次 Cast 输出。
3. **L1/L0 双缓冲与事件流水**：GM→L1（MTE2）、L1→L0（MTE1）、mmad（M）、L0C→GM（FIX）四级用 `SetFlag/WaitFlag<HardEvent::MTE2_MTE1 / M_MTE1 / MTE1_M / M_FIX / FIX_M>` 衔接。
4. **小 tile 加 PIPE_M barrier**：`(m/16)*(n/16) < 10` 时 Mmad 后插 `PipeBarrier<PIPE_M>` 防流水冲突。
5. **nd2nz stride 回退**：GM 行 stride 超过 65535 时按行/列降级为多次单条 DataCopy，避免 DMA 参数非法。

## 常见陷阱

- ❌ 保留"先 Cast fp32 再进 cube"的老习惯 → UB 流量翻倍且 cube 只能吃 fp32（算力减半或不支持）；fp16/bf16 必须直达 L1。
- ❌ 对 K 做显式 transpose 再进 L0B → 多一次 UB/L1 往返；用 ColumnMajor 视图 + nd2nz 一次到位。
- ❌ P 以 fp32 喂 PV mmad → PV 算力减半；必须 downcast 回输入 dtype（注意 bf16 用 CAST_RINT）。
- ❌ L0C 不双缓冲 → FIX 与 M 互相停等，cube 利用率腰斩。
- ❌ 矩阵维 < 64 仍强行上 cube → 启动开销不划算，退回 P42/P39 向量化（判定见 P13）。

## 代码搜索关键词

```bash
grep -nE "ReduceSum|Mmad|BlockMmad|Nd2Nz|ColumnMajor|loadQGM|L0C|Cast\(.*RoundMode" op_kernel/*.cpp op_kernel/*.h
```
