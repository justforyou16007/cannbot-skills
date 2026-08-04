# P130 Playbook: 1×1 Conv 翻转 GEMM 落地 SOP

> **强制流程**。完整实战记录见 `references/discovered/disc_X19.md`。

## Step 1: 门控判定（不满足直接回退，勿强解）

| 门控 | 条件 | 不满足时 |
|---|---|---|
| 卷积形态 | kH==kW==1 且 s==1 且 p==0 且 d==1 | 走 P129 或 normal 路径 |
| 分形约束 | Ci%16==0 且 (H*W)%16==0 | 回 normal 路径——**禁止用 UB 中转 pad 强解**（已证净负收益） |

## Step 2: 翻转 GEMM 组织

1. A = 原始权重 `[Co, Ci]`（跨图共享，免 wrepack）；B = 原始 x `[Ci, H*W]`（原始布局即操作数，免 im2col）；acc = `[Co, H*W]` fp32
2. 全通路 Nd2Nz GM→L1（plain 2D DataCopy 在 dav-c220 不可靠）
3. 翻转后 m=Co、n=H*W：epilogue 按通道行加 bias——整块连续 load + 行内标量 Adds（V pipe 行间免同步）+ 一次 cast + 一次 store（比逐行 flag pair 快 3~4×）
4. 空间尾 n 维非 16 对齐：LoadData3D Zn 的 channelSize 必须 round16，否则跳分形块列数据损坏

## Step 3: 验证

- 与 normal 路径对拍（同 case 双跑）；确认 host 模式分发逻辑：几何判定走本模式还是 normal/fused
