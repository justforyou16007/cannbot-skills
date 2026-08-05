---
id: X19
origin: discovered
discovered_round: 3
discovered_from: level3_conv_2d-cannbench-web-submit (P2) + conv2d_base (tilelang2ascendc-ops-generator)
base_speedup: cann-bench 1×1 case 1.98× 标杆（54.02us 标杆 vs 54.20/74.28us 实测，kernel 级反超近 2×）
---
> **已晋升为 P130**（card + precondition + playbook 三件套见 `cards/`、`preconditions/`、`playbooks/`）。本文件保留为实战详细记录（含实测数据与否定项依据），卡片内容以 P130 为准。


# Strategy X19: 1×1 Conv 翻转 GEMM（原始布局即操作数，免 wrepack/im2col）

## 核心思路
1×1 卷积的 im2col 是恒等变换：NCHW 原始布局的 x 本身就是 `[K=Ci, M=H*W]` 的
GEMM B 操作数。把 GEMM **翻转**——A = 原始权重 [Co, Ci]（跨图共享，免 wrepack）、
B = 原始 x [Ci, H*W]（免 im2col 发射）、acc = [Co, H*W] fp32——epilogue 按通道行
加 bias 后直接写出 NCHW，全程零 transpose、省两级 kernel 发射。

## 适用场景
- **算子类型**: 1×1 卷积（kH=kW=1、s=1、p=0、d=1），pointwise conv、1×1 reduction
- **瓶颈类型**: launch-bound / 小 kernel 发射开销占比大；或大 Ci 1×1 的
  im2col 纯搬运浪费
- **前提条件**: Ci%16==0（Nd2Nz 分形约束）且 (H*W)%16==0（B' 行 32B 对齐）。
  注意：用 UB 中转 pad 解除 (H*W)%16 门控已证**净负收益**（xpad 自身 ~5.4us
  走 DataCopyPad 慢速 dummy-pad 路径，翻转 GEMM 在窄 padded 行上也更慢），
  门控不满足时回 normal 路径，勿强解
- **预期收益**: 省 wrepack+im2col 两级发射，实测 1×1 case 1.0~1.98× 标杆

## 实现要点
1. 全通路保持 Nd2Nz GM→L1（plain 2D DataCopy GM↔L1 在 dav-c220 不可靠）
2. 翻转后 m 维 = Co（通道）、n 维 = H*W（空间）：epilogue 的 acc 行为通道行，
   整块连续区域一次 load + 行内标量 Adds（V pipe 行间免同步）+ 一次 cast +
   一次 store，比逐行 flag pair 快 3~4×
3. 空间尾 n 维非 16 对齐时 LoadData3D Zn 的 channelSize 必须 round16，
   否则跳部分分形块导致列数据损坏
4. 与 X10 的模式分发组合：host 按几何判定走本模式还是 normal/fused

## 来源
conv2d_base P0-b 首验；cann-bench conv_2d case 6（[2,2048,16,16] bf16 1×1）
实测 1.94~1.98× 标杆，为全部 20 case 中收益最高的单点。
