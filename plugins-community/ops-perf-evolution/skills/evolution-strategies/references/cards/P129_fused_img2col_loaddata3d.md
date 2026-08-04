---
id: P129
bottlenecks: [mte2_stall]
op_families: [matmul, special]
complexity: L2
conflicts_with: []
synergizes_with: [P128, P131]
requires: []
has_preconditions: true
has_playbook: true
---

# P129: Fused img2col LoadData3D 随路展开（含 band 窗口化突破 L1 容量）

> 晋升自 X18（conv2d_base case0 46→21us，消 70% im2col 物化；大 case 0.32→0.59）。详细实战记录见 `discovered/disc_X18.md`。

## 核心思想

卷积的 im2col GM 物化路径流量为 `特征图→UB→GM→L1→L0`，GM 往返可占整算子 70%。fused 模式让 **A 侧不落地**：特征图转 NC1HWC0 常驻 L1，用 img2col LoadData3D（pad/stride/dilation 硬件处理）从 L1 直接随路展开进 L0A，B 侧复用 K-major 权重 ws。CopyOut/epilogue/m/n 分解与 normal 路径完全复用，改动集中在 A 侧加载。

**band 窗口化推广**（特征图超 L1 时）：L1 只常驻输入**行带**而非整图。band 以独立 fmap 身份喂 LoadData3D：重算 `padTop' = pH + minRow − o0·sH ∈ [0,pH]`、`padBottom' ≤ pH`（均在原 pad 范围内，规避 pad 位宽溢出），mStartPt 用 band 相对值；硬件重建的输出高度 Ho' = o1−o0 代数恒成立。块 = (img, band)，n 维在块内循环（band 载一次供全部 nBlock 复用）。

## 适用场景

- conv2d（kHkW>1）及任何 im2col+GEMM 形态算子
- 瓶颈：im2col GM 往返带宽/耗时占比大（mte2_stall）；或大特征图 B 面板随 m-block 重复加载
- 前提：Ci%16==0（非对齐先通道补零）；xNz + 常驻 B 面板 ≤ L1（不满足用 band 窗口化）

## 关键修改点

1. **硬件 img2col 展开 k 序 = `(ci1, kh, kw, ci0)`**——C0 通道块在最外层，与软件惯用 `(kh,kw,ci)` 不同；权重 ws 必须按同序 repack（缓存键带布局标志）。Ci==16 或 kHkW==1 时两序重合会"蒙对"——**必须用 one-hot 探针在 Ci>16 且 kHkW>1 的几何上实测确认，勿凭文档假设**
2. NCHW 需 xtrans AIV kernel 先生成 xT `[H*W, Ci]`（TransDataTo5HD repeat 批量转置；bf16 按 uint16 位级转置）；NHWC 直接用原始 x，零中转
3. 门控回退（实测坑）：**W<24 窄特征图 LoadData3D 病理性慢**（~10us/次，为正常 7 倍）→ 回 im2col 物化路径；**kExtension > 64 不可靠**（出垃圾甚至挂死设备）→ 保持 ≤64；留环境变量开关（如 `CONV2D_FUSED=0`）做 A/B
4. 展开与 Mmad 重叠靠 L0 队列深度（见 P128 要点 3）
5. 尾块纪律：m/n 满 tile 计算 + Fixpipe 裁剪；channelSize 非 16 倍数会跳分形块，空间尾 round16

## 代码搜索关键词

```bash
grep -nE "im2col|img2col|LoadData3D|kExtension|mExtension" op_kernel/*.cpp op_host/*_tiling.cpp
```
