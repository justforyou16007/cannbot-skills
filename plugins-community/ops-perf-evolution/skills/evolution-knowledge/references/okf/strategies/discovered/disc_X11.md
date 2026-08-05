---
schema_version: okf.v1
kind: discovered_strategy
id: X11
origin: discovered
title: ND-drain + L0B 喂给率驱动 baseM 扩升（量化 Matmul Cube 利用率修复）
description: "c220 (910B) 上 MatmulImpl 的多 fragment NZ drain 是芯片缺陷（与 orgM 对齐无关，已证伪对齐假说）， 而 ND int32 多行 drain 完全正确——Cube 输出格式应路由到 ND。真正的 AIC 瓶颈不是 B 的 L2 refetch（scM 扩 16x 时间纹丝不动），而是 **L0B-fill:mad 比**：baseM=16 时每个 ba"
confidence: single_op
status: active
op_families: ['matmul']
bottlenecks: ['compute_bound', 'l2_cache_thrash', 'mte3_stall', 'no_overlap', 'scalar_compute']
tags: ['compute_bound', 'l2_cache_thrash', 'matmul', 'mte3_stall', 'no_overlap', 'quantization', 'scalar_compute']
discovered_round: 1
discovered_from: round_1/parallel_1 (level3_quant_matmul_evo_r3)
base_speedup: 0.8274x (vs 官方 QuantBatchMatmulV3, 基线 0.1999x → 4.14x)
---
# Strategy X11: ND-drain + L0B 喂给率驱动 baseM 扩升（量化 Matmul Cube 利用率修复）

## 核心思路
c220 (910B) 上 MatmulImpl 的多 fragment NZ drain 是芯片缺陷（与 orgM 对齐无关，已证伪对齐假说），
而 ND int32 多行 drain 完全正确——Cube 输出格式应路由到 ND。真正的 AIC 瓶颈不是 B 的 L2
refetch（scM 扩 16x 时间纹丝不动），而是 **L0B-fill:mad 比**：baseM=16 时每个 base tile 把
K×baseN 灌入 L0B 只做 16 行 MAC（~15 MAC/B），喂不饱 cube 阵列。把 baseM 扩升到
L0C 容量上限（int32: 128KB → baseM=128 @ baseN=256），AIC 时间可降 4x+，直至撞上
L2→L1 ~1.9TB/s 硬墙。

## 适用场景
- int8/int4 量化 Matmul（Cube 累加 int32），输出需反量化 epilogue 的算子
- 大 M×N 形状下 Cube 利用率低（小 baseM / 单 fragment）且疑似受 drain quirk 限制
- 评估口径为 Σ kernel Duration（AIC/AIV 重叠对此口径无效，不要投入双流方向）

## 实现要点
1. **ND drain 路由**：CType 从 NZ 改 ND（TPosition::GM, CubeFormat::ND, int32），
   消除 NZ 全局分形组 stride 推导；配 epilogue 直接读 ND 连续行，免 NZ→ND 重排。
2. **baseM 扩升**：候选 {64, 128} 实测扫描，L0C=128KB → int32 baseM×baseN≤32K elems
   （128×256 恰好占满）；depthA1/B1=4 后收益饱和（L2 带宽墙，非流水深度问题）。
   硬约束：**baseM ≤ 每个 tile 的 singleM**（singleM<baseM 直接 AIC 崩溃）；
   singleN≤16 尾 tile 崩溃 → **重叠尾 strip**：末 strip 从 N-baseN（M 轴对称 M-scM）
   起算全宽 tile，重叠区两 core 写相同值（幂等，bit-exact 安全）。
3. **AIV 反量化 epilogue strip 化**（750µs→34µs, 22x 的关键）：
   strip 所有权（每 AIV 连续 fragment 区间）+ per-N 向量（scale/bias/offset 展开为
   [16][256] 模式）**每 strip 构建一次**而非每 fragment 重建 + TQue 双缓冲
   （load/cast/store 跨 fragment 流水，稳态零 PipeBarrier<PIPE_ALL>）+
   单条 2D DataCopyPad 完成 16 行加载与存储 + pertoken strip 级预载。
4. **Host 侧隐式开销排查**：torch::empty 在 torch_npu 上隐式零初始化
   （每分配一次 = 一个 aclnnInplaceZero 内核）→ 内部 workspace 改全局 scratch 缓存；
   非对齐 N 触发 impl 全局 B 加载慢路径（orgN%256!=0，与 32B 对齐无关）→ align256 补齐。

## 来源
自动发现于 level3_quant_matmul Round-3 进化第 1 轮（x0 open_exploration），
Ascend910B2C，geomean 0.1999x → 0.8274x vs 官方 QuantBatchMatmulV3（20/20 bit-exact）。
