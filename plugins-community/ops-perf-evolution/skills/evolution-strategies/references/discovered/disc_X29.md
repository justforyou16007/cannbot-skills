---
id: X29
origin: discovered
discovered_round: 2
discovered_from: round_2/parallel_2 (sv1)
base_speedup: 0.927x vs PyTorch (case0 22.53us vs ref 20.88us); case7 2.433x
op_family: conv (ConvTranspose/deconv stride=1)
bottlenecks: [compute_bound, mte2_stall]
---

# Strategy X29: Shift-Free Tap-Comb 稠密直算（stride=1 反卷积消 Gather）

## 核心思路

对 stride==1 的反卷积/ConvTranspose2d，利用数学恒等变换：
**ConvTranspose2d(x, w, stride=1, pad) ≡ 零填充输入 + 180° 翻核的标准 cross-correlation**：
`out[co,oh,ow] = bias[co] + Σ_{ci,kf,kf2} wf[ci,co,kf,kf2] * P[ci,oh+kf,ow+kf2]`
其中 `P[ci] = x[n,ci]` 上下左右各补 `t = KH-1-pad` 个零，`wf[kf][kf2] = at[...,(KH-1-kf)*KW+(KW-1-kf2)]`（翻核）。

单 kernel（Direct-AIV）内实现 **shift-free tap-comb**：
- 每 (ci,kf2) 构造 `comb[ci][kf2][row][0:woA] = P[ci][row]` 左移 kf2 的对齐槽（8-float=32B 对齐）；
- 固定 (ci,kf2,kf) 时 comb 中 `kf..kf+ho-1` 行 CONTIGUOUS，因此每 (ci,kf2,kf,co) 的累加是**一次对齐 Axpy(len=ho*woA)**；
- 稠密 Axpy 链取代逐 tap 的 Gather 间接寻址，**彻底消除 Gather/col2im 散射/G workspace 往返/AIC→AIV 交接**。

相比 Gather-based 直接累加（x0，~10+ cycles/elem V-pipe Gather），X29 是 shift-free 布局，V-pipe 只做对齐稠密 FMA。

## 适用场景

- 小 ci、stride=1 的 ConvTranspose2d / deconv（固定开销/交接主导，Gather 爆炸受限）
- 判定特征：`Gather 指令多 + 单 kernel 化可消除 AIC→AIV 交接 + ci 小（工作集 < UB）`
- host 按 stride 分发：stride==1 走 dense，非 stride-1 / 大 UB 回退 cube GEMM + col2im

## 实现要点

- **零填充平面尺寸必须 `ph = ho+kh-1`（不是 `hi+2t`）**：pad>0 时后者少一行，comb 读取越界（case8 实测 bug）。
- **DataCopyPad UB→GM blockLen 必须 ≥ 32B**：wo<8 时 28B/24B 直接 507015 ADDR_MISALIGN（GM→UB 方向无此限制）。规避：host 分配 `[n][co][ho][woA]` 填充缓冲，kernel 单次 bulk DataCopyPad 写，host 返回 `[:wo]` 视图。
- **DataCopyPad(GM→UB)+left/rightPadding 向量化 comb 构建在 dav-2201 上 507015 fault**：已回退标量 SetValue（残余标量阶段 ~144 次）。
- **host 预处理下沉**：model_new_ascendc.py 不再构造 x2（permute/reshape/zeros/copy 三个 torch kernel 实测 ~17us），直接传原始 NCHW x；kernel 按 x[n][ci] 连续平面 blockCount=1 DataCopyPad 读取。
- 权重预翻核：`wf[kf][kf2]=at[...,(KH-1-kf)*KW+(KW-1-kf2)]` 在 host 预计算或 kernel 标量预翻核。
- 残余开销：标量 comb 放置 + 稠密 Axpy 链 + 单 kernel launch floor（910B ~11us）。

## 来源

自动发现于第 2 轮进化，算子 10_ConvTranspose2d（evo_v3 round_2/parallel_2 sv1）。
case0 [1,3,4,4] 0.927x（22.53us），case7 [1,1,5,5] 2.433x；相对 round_2 两 kernel 基线 41.2us → 22.5us（1.83x kernel 级）。
