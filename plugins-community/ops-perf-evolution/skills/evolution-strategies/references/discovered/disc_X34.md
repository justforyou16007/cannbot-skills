---
schema_version: okf.v1
kind: discovered_strategy
id: X34
origin: discovered
title: Col2im 相位化行归约（stride-s 反卷积散射重构，GEMM 中间矩阵上向量化）
description: 反卷积 fallback 路径（GEMM + col2im）的真正瓶颈不是 im2col 带宽而是 col2im 散射（逐输出行 undersize 加载 + Gather + parity 过滤 + PIPE_ALL 屏障）。把 col2im 从'逐输出行散射'改为'每相位对相位内 K²/s² 条 G 行做全长对齐向量 Add 归约 → 相位平面 → 行粒度批量写回'。列移位经 DataCopyPad src offset + left/rightPadding 落地（AIV 全 32B 对齐，绕开 odd-shift 对齐封锁）。case2 [2,32,16,16] 548→118us (4.7x)。
confidence: single_op
status: active
op_families: ['deconv', 'conv', 'scatter', 'gather']
bottlenecks: ['mte2_stall', 'tiling_imbalance', 'undersize_transfer', 'no_overlap']
tags: ['col2im', 'phase-decomposition', 'deconv', 'stride2', 'vector-reduction', 'G-matrix-reuse', 'dav-2201']
discovered_round: 2
discovered_from: 10_ConvTranspose2d evolution round_2/parallel_0 (sv2, Supervisor 校正后)
base_speedup: case2 [2,32,16,16] 548.4→118us (4.66x)；全量 50-case geomean 0.405→0.478 (+18%)
---

# X34: Col2im 相位化行归约

## 核心思路
反卷积 fallback（cube GEMM + col2im）中，**GEMM 中间矩阵 G[mg, ng] 仅 0.03-2MB、GEMM ~9us 非瓶颈**；真正瓶颈是 col2im 散射——逐输出行 (n,co,oh) 循环 tap，每 tap 一次 64B undersize GM 加载 + Gather + parity 过滤 + PIPE_ALL（~274us/大 case）。

把 col2im 重构为 **相位化行归约**：
1. 按输出奇偶拆 s² 个相位（stride-s），每相位只消费相位内 K²/s² 条 G 行
2. 对相位内 G 行做**全长对齐向量 Add 归约**（acc_phase = Σ_{taps} G_row[0:ng]），一次向量 Add 替代逐 tap Gather+Add（P94）
3. 列移位 c0=d_w0-b 经 DataCopyPad src offset + left/rightPadding 落地（P26，AIV 全 32B 对齐——绕开 x0 的 odd-shift 对齐封锁）
4. 相位平面 [co, n·hi·wi] 行粒度批量写回（P56）；wo==woA 时多块 DataCopyPad
5. 一次向量 Gather 奇偶交织写回（idx 一次性预计算，固定 SB 布局）

## 适用场景
- **算子类型**: 反卷积/分数步长卷积的 col2im fallback（GEMM + col2im 两 kernel 形态）；任何散射类 col2im/deconv
- **瓶颈类型**: col2im 散射（undersize_transfer + scalar Gather + PIPE_ALL 屏障序列化）主导；mte2_stall
- **前提条件**: GEMM 中间矩阵 G 可复用（常驻 L2）；相位行归约全长对齐（G 行 [0,ng) 连续）
- **预期收益**: col2im 散射从 ~274us 压到 ~110us（case2 4.7x）；全量 geomean +18%

## 实现要点
1. **tap 表预计算**：rTaps/rDh/cTaps/cDw 启动时算好，消除逐行 parity 过滤
2. **多块加载门控**：多块 DataCopyPad 仅用于 `wi%8==0 && srcCol==0`——非对齐 src 行多块在 dav-2201 病态（case29 实测回归后修复）
3. **列移位落地**：DataCopyPad src offset + left/rightPadding（勿用 V-pipe 逐元素移位——dav-2201 无元素级 shift，507015/strided 挂死）
4. **G 读带宽是残余 floor**（case2 ~35us）：2MB G 读取 + 双 kernel launch/AIC-AIV 切换 + co 小 case 并行度不足（unit=(n,co,po) 仅 6-8 核）
5. **差异化定位**：X26 是流水线级 barrier 优化（3→2/行），本策略是算法级相位归约——更高杠杆，两者可叠加
6. **DataCopyPad(isPad=true) 尾部垃圾陷阱（配套 X26 tap-batch 实测发现）**：dav-2201 上 `DataCopyPad(isPad=true)` 在最后一个 32B 块的 `[wi, align8(wi))` 未对齐尾写入 **garbage（非 paddingValue=0）**——若把零 sentinel 放在元素 wi 会被首载 clobber（18/50 wrong）。修复：sentinel 移到 32B 对齐元素 wiA（loads 整 32B 块写 `[0,wiA)`，永不触碰 wiA）+ Gather offset 无效映射从 wi*4 改为 wiA*4 + 一次性 V-pipe `Duplicate(src,0)`

## 关键前提洞察（Supervisor 校正）
"im2col GM 往返占 70%" 的诊断有误导：GEMM 中间 G 很小，col2im 散射才是大头。**col2im 应极薄**（业界共识：ggml/CUTLASS dgrad output-stationary gather）。攻 col2im 自身结构而非 im2col 物化。

## 来源
10_ConvTranspose2d evolution round_2/parallel_0（sv2）：Supervisor 在停滞门控时校正瓶颈诊断 → 相位化行归约实现。case2 4.66x、case9 1.5x、case1 1.4x、全量 +18%。与 round_1 x0（AIV odd-shift comb 相位分解）对比：**同样的相位分解思想，载体从 AIV shifted-comb 改为 GEMM 行归约侧即绕开 32B 对齐封锁**——载体选择是关键。
