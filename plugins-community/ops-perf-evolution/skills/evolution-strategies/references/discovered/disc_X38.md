---
id: X38
origin: discovered
discovered_round: 1
discovered_from: round_1/parallel_2
base_speedup: 0.579x → 0.856x (mean, FlashAttentionV2; geomean 0.530→0.794)
---

# Strategy X38: GM-Workspace-Ring KV-tile 宽度缩放 + Host 批量权重打包

## 核心思路
FA 算子若采用 **GM workspace ring 队列**（AIC 写 S/P/O → GM → AIV 读）而非 L1 常驻，则每 KV tile 产生固定次数的 GM 往返 + 跨核握手 + 标量开销（AIC scalar_ratio ~0.8, AIV ~0.7，cube MAC 仅 4-6%）。此时放大 KV tile 宽度（kvN）不是为了让 L1 常驻，而是**直接减半 GM ring 往返次数**、翻倍 MM1/MM2 的 cube GEMM 尺寸。同时，融合 QKV/out 投影算子的 host 侧 per-head 权重打包（48+ 次 `narrow().copy_()`）构成独立 copy-storm（fp32 case 占墙钟 ~40%），可批量化为 `cat + pad + reshape + 单次 strided copy`。

## 适用场景
- FA 算子使用 GM workspace ring 握手（AIC↔AIV 经 GM 中转 S/P/O），每 KV tile 固定握手开销主导，而非带宽或 cube 吞吐
- 融合投影 GEMM 且 heads 数多（per-head copy 次数 > 12），host 侧权重打包成为墙钟拖累
- 硬件：Ascend 910B 系列 MIX AIC/AIV

## 实现要点
- **kvN 自适应**：新增 tiling 字段 `kvN`。seq<=64 用 64（避免单宽 tile 的 2x 尾块浪费）；seq>64 用 128。所有 workspace 槽/buffer 仍按 128 max 分配，仅 ring slot stride、MM n/k、fixpipe width、softmax srcK、mask、causal 对角索引（`t = bx*BLOCK_M/kvN`）改为运行时 kvN。
- **sPad 对齐 kvN**：放大 tile 后必须 `sPad = align_up(seq, kvN)`（原 64），否则末 KV tile 越界读 mPad（NaN 根因）。
- **host 批量打包**：heads 多时 `cat` 各 head 权重块 → `constant_pad_nd` 补齐 dhPad → `reshape` → 单次 strided copy；heads 少时保留逐 head copy（constant_pad_nd 有 30-50us 固定开销，小 shape 反而拖累）。输出张量用 `at::empty`（kernel 全量覆盖，padded 行被 mask 丢弃，未初始化安全）。
- 收益来源：KV tile 数减半 → GM ring 往返/握手减半 + cube GEMM 翻倍；host 打包从 O(heads) 次 launch 降到 O(1)。

## 来源
自动发现于第 1 轮进化，算子 FlashAttentionV2，round_1/parallel_2（open_exploration）：mean 0.579→0.856（geomean 0.530→0.794）。代表 case：case 35 bf16 [1,463,1280] 0.423→0.767x，case 20 causal 0.674→1.026x，case 34 fp16 0.860→1.499x。
