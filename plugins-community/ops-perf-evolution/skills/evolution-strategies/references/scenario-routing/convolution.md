# Convolution 优化工作流（Conv1d/2d/3d、Depthwise、ConvTranspose）

> 适用：滑窗局部加权和类算子。conv 的本质是 im2col+GEMM，**先按几何模式分发，再决定用哪张卡**——不同 kernel 形状的最优结构完全不同。

## Step 0 — 族判定确认

命中 [README.md](README.md) 判定表第 2 行（滑窗参数 `kH/kW/stride/pad/dilation`、im2col/LoadData3D 证据）即归本族。族内变体：depthwise（`groups==Cin`，逐通道）、ConvTranspose（col2im/分数步长形态）。

## Step 1 — 通用前置

打开 [_common-prerequisites.md](_common-prerequisites.md) 过 C1-C5。Conv 族常见命中：**C1**（fp32 被困 Vector 路径 → 见 P128）。

## Step 2 — 几何模式分发（先判模式，按序路由）

| 模式 | 判定条件 | 专用路径 | 跳过条件 |
|---|---|---|---|
| C-1 1×1 pointwise | `kH==kW==1 && s==1 && p==0 && d==1` | **P130**（翻转 GEMM：原始布局即操作数，免 wrepack/im2col 两级发射） | 门控不满足（Ci%16≠0 或 (H·W)%16≠0）→ 回 C-3 normal 路径，**勿用 UB pad 强解**（已证净负收益） |
| C-2 fp32 输入 | dtype==fp32 且 Cube 仅支持 fp16/bf16 | **P128**（hi/lo 拆分 3-GEMM：Cube 吞吐 + ~1e-4 精度）；配套单趟 split-xtrans | 容差不允许 ~1e-4 相对误差 |
| C-3 一般 kHkW>1 | 常规卷积核 | **P129**（fused img2col LoadData3D 随路展开，A 侧不落地，消 70% im2col 物化）；特征图超 L1 → band 窗口化支路 | 窄特征图（W<24）LoadData3D 病理性慢 → 回 im2col 物化路径；kExtension 保持 ≤64 |
| C-4 推理重复调用 | 同一权重反复卷积（评测/推理循环） | **P131**（host 侧 repack LRU 缓存，跳过 repack kernel 发射） | 权重每次变化 |
| C-5 depthwise | `groups==Cin` | **覆盖缺口**：GEMM 退化为逐通道向量乘加，走通用前置 + 瓶颈驱动 | — |
| C-6 conv3d | 三维滑窗 | **覆盖缺口**：按 C-3 思路推广（LoadData3D 支持 3D），无沉淀卡 | — |
| C-7 ConvTranspose | 分数步长/col2im 形态 | **覆盖缺口**：col2im+GEMM 结构，结合 matmul workflow 的 CV 支路（[matmul.md](matmul.md) M1/M7：P46/P16/P51） | — |

## Step 3 — GEMM 主干的族内收尾

im2col/fused 展开后本质是 GEMM：多核切分、L2 复用、CV 流水等转入 [matmul.md](matmul.md) 的 M2/M3/M7 阶段（P104/P105/P16/P51），注意权重 ws 的 k 序必须与 img2col 硬件展开序 `(ci1,kh,kw,ci0)` 一致（P129 要点）。

## 门控纪律（继承 X 卡实测教训）

1. 硬件 img2col 展开 k 序 = `(ci1, kh, kw, ci0)`——与软件惯用序不同；Ci==16 或 kHkW==1 时会"蒙对"，**必须用 one-hot 探针实测确认**，勿凭文档假设
2. 门控不满足（对齐、W<24、kExtension>64）一律回退 normal 路径，勿强解
3. 留环境变量开关（如 `CONV2D_FUSED=0`）做 A/B 与回退
4. ±inf 特殊值 case 不能用于验证展开类改动（"双 NaN 巧合"会掩盖错误），用有限值 case
