---
schema_version: okf.v1
kind: discovered_strategy
id: X18
origin: discovered
title: Fused img2col LoadData3D 随路展开（含 band 窗口化突破 L1 容量）
description: 卷积的 im2col GM 物化路径流量为 `特征图→UB→GM→L1→L0`，GM 往返可占整算子 70%。 fused 模式让 **A 侧不落地**：特征图转 NC1HWC0 常驻 L1，用 img2col LoadData3D （pad/stride/dilation 硬件处理）从 L1 直接随路展开进 L0A，B 侧复用 K-major 权重 ws。CopyOut/epilogue/m/n
confidence: single_op
status: active
op_families: ['cv_fusion', 'matmul']
bottlenecks: ['compute_bound', 'mte3_stall', 'scalar_compute']
tags: ['compute_bound', 'conv2d', 'cv_fusion', 'matmul', 'mte3_stall', 'scalar_compute']
discovered_round: 4
discovered_from: level3_conv_2d-cannbench-web-submit (P3/P5) + conv2d_base (tilelang2ascendc-ops-generator)
base_speedup: conv2d_base case0 46→21us（消 70% im2col 物化）；cann-bench 大 case 0.32→0.59
---
> **已晋升为 P129**（card + precondition + playbook 三件套见 `cards/`、`preconditions/`、`playbooks/`）。本文件保留为实战详细记录（含实测数据与否定项依据），卡片内容以 P129 为准。


# Strategy X18: Fused img2col LoadData3D 随路展开（含 band 窗口化突破 L1 容量）

## 核心思路
卷积的 im2col GM 物化路径流量为 `特征图→UB→GM→L1→L0`，GM 往返可占整算子 70%。
fused 模式让 **A 侧不落地**：特征图转 NC1HWC0 常驻 L1，用 img2col LoadData3D
（pad/stride/dilation 硬件处理）从 L1 直接随路展开进 L0A，B 侧复用 K-major
权重 ws。CopyOut/epilogue/m/n 分解与 normal 路径完全复用，改动集中在 A 侧加载。

**band 窗口化推广**（特征图 xNz 超 L1 时）：L1 只常驻输入**行带**而非整图。
band 以独立 fmap 身份喂 img2col LoadData3D：重算 `padTop' = pH + minRow − o0·sH
∈ [0,pH]`、`padBottom' ≤ pH`（均在原 pad 范围内，规避 pad 位宽溢出），mStartPt
用 band 相对值——硬件按 `(l1H, padTop', padBottom', sH, dH, kH)` 重建的输出高度
Ho' = o1−o0 代数恒成立，m→(ho,wo) 光栅与全图一致。块 = (img, band)，n 维在块内
循环（band 载一次供全部 nBlock 复用）。

## 适用场景
- **算子类型**: conv2d（kHkW>1）及任何 im2col+GEMM 形态的算子
- **瓶颈类型**: im2col GM 往返带宽/耗时占比大；或大特征图 B 面板随 m-block
  重复加载（L1 入口带宽瓶颈）
- **前提条件**: Ci%16==0（img2col channelSize 约束，非对齐先通道补零）；
  xNz + 常驻 B 面板 ≤ L1 预算（不满足时用 band 窗口化）
- **预期收益**: 消 im2col 物化 70% 占比（46→21us 级）；band 化使大 case
  从 normal 路径 0.32 → 0.59

## 实现要点
1. **硬件 img2col 展开 k 序 = (ci1, kh, kw, ci0)**——C0 通道块在最外层，与软件
   惯用 (kh,kw,ci) 不同；权重 ws 必须按同序 repack（缓存键带布局标志）。
   两序在 Ci==16 或 kHkW==1 时恰好重合会"蒙对"——**必须用 one-hot 探针**
   （单元素置 1 观察落点）在 Ci>16 且 kHkW>1 的几何上实测确认，勿凭文档假设
2. NCHW 需 xtrans AIV kernel 先生成 xT [H*W, Ci]（TransDataTo5HD repeat 批量
   转置；bf16 按 uint16 位级转置）；NHWC 直接用原始 x，零中转
3. 门控回退（实测坑）：
   - **l1W ≤ ~17 窄特征图 img2col LoadData3D 病理性慢**（~10us/次，为正常 7 倍，
     mExtension 调满无效）→ W<24 回 im2col 物化路径
   - **kExtension > 64 不可靠**（部分几何出垃圾甚至挂死设备）→ 保持 ≤64
   - 留环境变量开关（如 CONV2D_FUSED=0）做 A/B 与回退
4. 每个 k tile 的展开与 Mmad 重叠靠 L0 队列深度（见 X11 要点 3）
5. 尾块纪律不变：m/n 满 tile 计算 + Fixpipe 裁剪；channelSize 非 16 倍数会
   跳分形块，空间尾 round16

## 来源
conv2d_base（tilelang2ascendc-ops-generator）P5 首次落地（k 序探针发现）；
cann-bench conv_2d P3/P5 推广 band 窗口化（xNz 8MB/4MB 超 L1 的 case 2/3）
与非对齐 Ci（xtrans 通道补零后覆盖 Ci=63/47/97/65/31 等）。
