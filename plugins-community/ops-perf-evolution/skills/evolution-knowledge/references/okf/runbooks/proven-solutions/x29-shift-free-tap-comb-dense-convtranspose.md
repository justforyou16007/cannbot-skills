---
schema_version: okf.v1
kind: proven_solution
id: x29
status: active
op_families: [conv, deconv]
strategy_ids: [X29]
bottlenecks: [compute_bound, mte2_stall]
tags: [convtranspose, stride1, shift-free, dense-direct, gather-elimination, dav-2201]
source: 10_ConvTranspose2d evolution round_2/parallel_2
---

# X29: Shift-Free Tap-Comb 稠密直算（stride=1 反卷积）

## 手法本质
stride=1 反卷积恒等变换为「零填充输入 + 180°翻核标准卷积」，单 AIV kernel 内用
shift-free tap-comb（comb 左移对齐槽）使累加成为连续块一次对齐 Axpy，彻底消除
Gather 间接寻址 / col2im 散射 / G workspace 往返 / AIC→AIV 交接。

## 适用条件
- stride==1 反卷积，ci 小（工作集 < UB），固定开销/交接主导
- 判定特征：Gather 指令多 + 单 kernel 化可消除交接

## 关键纪律
1. 零填充平面 ph = ho+kh-1（非 hi+2t，pad>0 越界）
2. DataCopyPad UB→GM blockLen≥32B（wo<8 用 [n][co][ho][woA] 填充缓冲 + host [:wo] 视图规避 507015）
3. DataCopyPad(GM→UB)+left/rightPadding 向量化 comb 在 dav-2201 507015，回退标量
4. host 预处理下沉（去 torch x2 构建 ~17us）
5. host 按 stride 分发：stride==1 走 dense，否则回退 cube GEMM+col2im

## Evidence
10_ConvTranspose2d round_2/parallel_2：case0 [1,3,4,4] 0.927x（22.53us vs ref 20.88us），
case7 2.433x；相对两 kernel 基线 41.2us → 22.5us（1.83x kernel 级）。单 kernel launch floor ~11us。
