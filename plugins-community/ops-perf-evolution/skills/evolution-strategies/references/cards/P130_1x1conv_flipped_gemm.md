---
id: P130
bottlenecks: [compute_bound, mte2_stall]
op_families: [matmul, special]
complexity: L1
conflicts_with: []
synergizes_with: [P129, P131]
requires: []
has_preconditions: true
has_playbook: true
---

# P130: 1×1 Conv 翻转 GEMM（原始布局即操作数，免 wrepack/im2col）

> 晋升自 X19（cann-bench 1×1 case 1.94~1.98× 标杆，为 20 case 中收益最高单点）。详细实战记录见 `discovered/disc_X19.md`。

## 核心思想

1×1 卷积的 im2col 是恒等变换：NCHW 原始布局的 x 本身就是 `[K=Ci, M=H*W]` 的 GEMM B 操作数。把 GEMM **翻转**——A = 原始权重 `[Co, Ci]`（跨图共享，免 wrepack）、B = 原始 x `[Ci, H*W]`（免 im2col 发射）、acc = `[Co, H*W]` fp32——epilogue 按通道行加 bias 后直接写出 NCHW，全程零 transpose、省两级 kernel 发射。

## 适用场景

- 1×1 卷积（kH=kW=1、s=1、p=0、d=1）、pointwise conv、1×1 reduction
- 瓶颈：launch-bound / 小 kernel 发射开销占比大；或大 Ci 1×1 的 im2col 纯搬运浪费
- 前提：Ci%16==0（Nd2Nz 分形约束）且 (H*W)%16==0（B' 行 32B 对齐）

## 关键修改点

1. 全通路保持 Nd2Nz GM→L1（plain 2D DataCopy GM↔L1 在 dav-c220 不可靠）
2. 翻转后 m=Co（通道）、n=H*W（空间）：epilogue 的 acc 行为通道行，整块连续区域一次 load + 行内标量 Adds（V pipe 行间免同步）+ 一次 cast + 一次 store，比逐行 flag pair 快 3~4×
3. 空间尾 n 维非 16 对齐时 LoadData3D Zn 的 channelSize 必须 round16，否则跳部分分形块导致列数据损坏
4. 与模式分发组合：host 按几何判定走本模式还是 normal/fused（P129）

## 常见陷阱

- ⚠️ 用 UB 中转 pad 解除 (H*W)%16 门控已证**净负收益**（xpad 走 DataCopyPad 慢速 dummy-pad 路径）——门控不满足时回 normal 路径，勿强解

## 代码搜索关键词

```bash
grep -nE "kH.*1|kW.*1|kernel.*1|pointwise" op_host/*_tiling.cpp; grep -nE "Nd2Nz|LoadData3D" op_kernel/*.cpp
```
