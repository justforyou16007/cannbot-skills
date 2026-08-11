---
schema_version: okf.v1
kind: discovered_strategy
id: X29
status: active
confidence: single_op
op_families: [conv, deconv]
bottlenecks: [compute_bound, mte2_stall]
tags: [convtranspose, stride1, shift-free, dense-direct, gather-elimination, dav-2201]
discovered_from: 10_ConvTranspose2d evolution round_2/parallel_2 (sv1)
---

# X29: Shift-Free Tap-Comb 稠密直算

## 手法本质
stride=1 反卷积的恒等变换：ConvTranspose2d ≡ 零填充输入 + 180°翻核标准卷积。单 kernel 内
用 comb[ci][kf2][row][0:woA]=P[ci][row] 左移 kf2 的 8-float 对齐槽构造 shift-free 布局，
固定 (ci,kf2,kf) 的累加是连续块一次对齐 Axpy，彻底消除 Gather 间接寻址/col2im 散射/
G workspace 往返/AIC→AIV 交接。

## 适用条件
- stride=1，ci 小（工作集 < UB），固定开销/交接主导
- 判定特征：Gather 指令多 + 单 kernel 化可消除交接
- host 按 stride 分发：stride==1 走 dense，否则回退 cube GEMM+col2im

## 关键纪律
1. 零填充平面 ph 必须 = ho+kh-1（非 hi+2t，pad>0 时少一行越界）
2. DataCopyPad UB→GM blockLen≥32B（wo<8 用 [n][co][ho][woA] host 填充缓冲 + [:wo] 视图规避 507015）
3. DataCopyPad(GM→UB)+left/rightPadding 向量化 comb 在 dav-2201 507015，回退标量
4. host 预处理下沉：去掉 torch x2 构建（permute/zeros/copy ~17us），直接传 NCHW x
5. 权重预翻核 wf[kf][kf2]=at[...,(KH-1-kf)*KW+(KW-1-kf2)]

## Evidence
10_ConvTranspose2d evo_v3 round_2/parallel_2：case0 [1,3,4,4] 0.927x（22.53us），
case7 [1,1,5,5] 2.433x；相对 round_2 两 kernel 基线 41.2us → 22.5us（1.83x kernel 级）。
单 kernel launch floor ~11us 是进一步上限。
