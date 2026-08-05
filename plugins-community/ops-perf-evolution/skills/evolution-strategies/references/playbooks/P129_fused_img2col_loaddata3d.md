# P129 Playbook: Fused img2col LoadData3D 落地 SOP

> **强制流程**。完整实战记录见 `references/discovered/disc_X18.md`。

## Step 1: 取证与 k 序探针（勿跳过）

```bash
grep -nE "im2col|img2col|LoadData3D|kExtension|repack" op_kernel/*.{cpp,h} > /tmp/p129_locations.txt
```
**one-hot 探针实测硬件展开 k 序**：单元素置 1 观察落点，在 Ci>16 且 kHkW>1 的几何上确认 k 序 = `(ci1,kh,kw,ci0)`（Ci==16 或 kHkW==1 时两种序重合会"蒙对"，勿凭文档假设）。

## Step 2: A 侧改造（fused 随路展开）

1. 特征图转 NC1HWC0 常驻 L1（NCHW 需 xtrans AIV kernel 生成 xT：TransDataTo5HD repeat 批量转置，bf16 按 uint16 位级转置；NHWC 直接用原始 x）
2. img2col LoadData3D 从 L1 直接展开进 L0A（pad/stride/dilation 硬件处理）
3. 权重 ws 按探针确认的 k 序 repack（缓存键带布局标志，配 P131）

## Step 3: band 窗口化（特征图超 L1 时）

1. L1 只常驻输入行带：`padTop' = pH + minRow − o0·sH ∈ [0,pH]`、`padBottom' ≤ pH`，mStartPt 用 band 相对值
2. 块 = (img, band)，n 维块内循环（band 载一次供全部 nBlock 复用）

## Step 4: 门控与验证

- **W<24 窄特征图 LoadData3D 病理性慢（~7 倍）→ 回 im2col 物化路径**
- **kExtension 保持 ≤64**（>64 出垃圾甚至挂死设备）
- 留环境变量开关（`CONV2D_FUSED=0`）做 A/B
- 尾块：m/n 满 tile 计算 + Fixpipe 裁剪；channelSize 非 16 倍数会跳分形块，空间尾 round16
