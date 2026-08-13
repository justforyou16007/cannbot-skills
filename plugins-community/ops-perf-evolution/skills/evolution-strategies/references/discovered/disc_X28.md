---
id: X28
origin: discovered
discovered_round: 3
discovered_from: round_3/parallel_0 (n2_cont_cont)
base_speedup: 0.976x (P8-canonical), geomean 0.588 vs baseline 0.500
---

# Strategy X28: 小 S Prefill FA 自适应 KV Chunk 粒度

## 核心思路
FA/GQA prefill 的 host 侧常把 `kvPad = align_up(S, nChunk)`，nChunk 取 cube 列分形友好的 256(fp16)/128(fp32) 大块。
当序列长度 S 很小（如 S=3~113）时，kvPad 被大幅上取整（S=3 → kvPad=256，85x 填充浪费），attention 的
QK^T/softmax/PV 全在 mask 覆盖的 pad 列上白算（mask 只是让 exp→0，计算照跑）。

本策略把 nChunk 改为 **按 S 自适应**：小 S（如 S<=256）用 nChunk=64（kvPad=align_up(S,64)），kernel 支持
尾块 partial chunk（QK^T/PV 的 nCols/nKv < nChunk/KS）。对齐粒度只需 cube 列分形 16 元素，无需 256/128。

## 适用场景
- 算子：FA/GQA/MHA **prefill**（S 变化范围大，含大量小 S case）；host 以 nChunk 做序列 padding。
- 瓶颈：小 case 的 **padding 冗余计算**（over-padding on KV/sequence dim），非带宽/计算真瓶颈。
- 硬件：Ascend 910B 系列（dav-c220-vec），cube 列分形 16 对齐即可。

## 实现要点
- host 侧：`nChunk = min(256/128, kvPad)` 或按 S 阈值切换（S<=256 → nChunk=64），`kvPad = align_up(S, nChunk)`。
- kernel 侧：QK^T/PV 的 chunk 循环改为 while + 尾块 partial（nCols/nKv < nChunk 单独 Mmad/fixpipe），
  DoQKStep/AllocL0BKT/AllocL0BV 增补列宽/行宽参数。
- fp32 3xTF32 路径同步支持尾块（zZ→nZ transpose load 列宽参数化）。
- 尾块 nCols 必须 16 对齐（kvPad=align_up(S,16) 保证）。
- 联动：packDataElems 变小使更多 case 落入 host 打包融合路径；但中型 case strided-DMA pack 可能反而慢，
  需按数据量调 pack 路径阈值（本案例 131072→80000）。

## 来源
自动发现于 GroupedQueryAttention 进化第 3 轮，n2_cont_cont（round_3/parallel_0），geomean 0.588（基线 0.500），
P8 规范 score 0.976。小/中 case attention device 时间降 2-4x。与 X7(TB tuning)/X20(weight repack cache) 正交互补。
