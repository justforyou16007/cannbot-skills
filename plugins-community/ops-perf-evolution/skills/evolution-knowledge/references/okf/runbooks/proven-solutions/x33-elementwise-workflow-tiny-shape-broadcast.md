---
schema_version: okf.v1
kind: proven_solution
id: x33
title: Elementwise 工作流全流程优化 tiny-shape launch 主导 elementwise（单 kernel 融合 + 多 block 拷贝合并 + 多核并行）
source: FusedKdaGate evolution round 1-3 (n0 → n0_s1 → n2_s1/x0_x1_x1)
speedup: "geomean 0.49x → 2.31x vs PyTorch（约 4.7 倍提升，Asc 均值 33.9us → 7.2us）"
status: active
op_families: [elementwise, broadcast, reduction]
strategy_ids: [P112, P84, P95, P89, P12, P1, P5, P7, P8, X3, D1, D2]
bottlenecks: [no_overlap, scalar_compute, tiling_imbalance, mte2_stall]
tags: [elementwise, tiny-shape, launch-dominance, per-head-broadcast, duplicate, select-64-align, vector-chain, multi-block-datacopypad, multi-core]
---

# X33: Elementwise 工作流全流程优化 tiny-shape launch 主导 elementwise（单 kernel 融合 + 多 block 拷贝合并 + 多核并行）

**Source**: FusedKdaGate evolution（RLHF KDA gate 融合算子），round 1-3（n0 workflow_guided → n0_s1 → n2_s1 2.308x / x0_x1_x1 2.136x）
**Speedup**: geomean 0.49x → **2.31x** vs PyTorch（约 4.7 倍）；Asc 均值 33.9us → 7.2us；单 case 最高 4.10x。50/50 精度 PASS。
**Operator type**: 内存密集 tiny-shape elementwise（per-head 标量 rate=exp(A) + per-head bias 广播；heads≤4、rows≤178、hk 7~384；单核单 task 数据 ≤32KB）
**Hardware**: Ascend910B2C (DAV_2201)

**Technique description**:
对 **tiny-shape + per-head 标量/向量广播 + launch 固定开销主导** 的 elementwise 算子，单纯流水重叠（双缓冲）无效（详见 f5 failure pattern）。**关键根因**：基线的 host 预处理（cast/transpose/head 切分/pad）在 forward 侧发射 ~5 个 NPU kernel，而 msprof 的 `asc_us` 累加所有 AI_CORE kernel 时间——多 kernel 的 launch 固定开销叠加才是基线慢的主因。按 elementwise workflow（E1-E5 阶段）结构重排 + **单 kernel 全融合**即可压到接近 launch 地板：
1. **单 kernel 全融合**：把 host 的 cast/transpose/head 切分/pad 全部下沉进 kernel，forward 只发射 1 个 NPU kernel（评测口径 `asc_us` 是所有 AI_CORE kernel 累加，融合后从 ~5 kernel 降到 1，launch 开销 ~5 倍下降）。
2. **flat 布局免转置**：直接处理 `[rows, heads*hk]`，rate/bias 按 head 广播，避免 host 侧 transpose。
3. **per-head 广播向量化**：rate（每 head 1 个标量）与 bias（每 head hk 宽）用 `Duplicate` / `DataCopyPad` 构建对齐向量广播（逐 head 对齐段布局 `[nRows, heads*hkPad]`，head 段起始 32B 对齐），替代逐 head 循环 + 每 task 一次 `GetValue` 标量读（消除 S-pipe 标量读串行）。
4. **CompareScalar/Select 的 count 必须 64 对齐**：`CompareScalar`/`Select` 的向量 count 非 64 倍数时 mask 错乱（NPU 6 稳定复现），用 `cntPad = ceil(cnt/64)*64` + 尾部清零修复。
5. **对齐 hk fast path**：`hkPad == hk` 时逐行连续拷贝（CopyIn/CopyOut 循环次数从 nRows*heads 降到 nRows），减少 MTE 指令数。
6. **DataCopyPad 读 A 必须 32B blockLen**：heads*4 子 32B 读偶发陈旧值，需按 32B blockLen 对齐。
7. **前置 C4**（无双缓冲+非对齐 → P1/P5/P7/P8）与 **E1-E2**（P112 tile、P84/P95 向量链）组合，vector 指令链保持高效。

**Round 3 追加（从 1.96x → 2.31x 的关键，x0_x1_x1 与 n2_s1 验证）**:
8. **多 block DataCopyPad 拷贝合并（最大收益）**：aligned hk 时 CopyIn/CopyOut 从「每行 1 条」合并为「每 slab 1 条单 block 拷贝」（rows 在 GM/UB 连续）；unaligned hk 时从「nRows*heads 条 per-head 拷贝」合并为「1 条 multi-block 拷贝」（blockCount=nRows*heads, blockLen=hk*T, srcStride=0, dstStride=0；GM 源连续，UB 侧 per-head 32B-padded block 背靠背，硬件逐 block pad）。unaligned CopyOut 无法用 multi-block 合并（UB→GM 的 srcStride 32B dataBlock 粒度无法表达 <32B 的 (hkPad-hk)*T 间隙）。
9. **rate2d/bias2d 2D Broadcast 一次性构建**：setup 时构建 2D Broadcast，per-slab 的 nRows 次 per-row Add/Mul 变为单条 1D 向量 op（配合 row-replicated [tileRows, lPad2] 布局，tail slab 正确）。
10. **Host tiling 多核并行**：MAX_TILE_ELEMS 6144→2048（avg AIV cores 3.1→7.2），并去掉过时的 slab-start 32B 约束（multi-block DataCopyPad GM src 是 1B 对齐；aligned path 自动 32B 对齐）——tiny-shape 也吃多核。
11. **X3 dtype 直传 + 结构优化组合**：纯 X3 dtype 直传（n2）仅 1.709x（tiny-shape 带宽非瓶颈），但把 n0_s1 的 7 项结构优化同步到 fp32 路径后（n2_s1）达 2.308x——dtype 直传对低精度 case（bf16 最高）有收益，需叠加结构优化才整体达标。

**Key code change**（要点）:
```cpp
// Before: 每 task 一次 rateLocal.GetValue(h)（S-pipe 标量读）+ 逐 head 循环
// After: 用 Duplicate 构建 ratePad[h*hkPad] 对齐向量广播 + 64 对齐 Select
Duplicate(ratePadLocal[h * hkPad], rate, hkPad);      // 对齐偏移广播
DataCopyPad(biasPadLocal, biasGm_[h * hkPad], ...);   // bias 对齐装载
// CompareScalar/Select 用 cntPad（64 对齐），尾部用 Duplicate 清零
```

**Applicability conditions**:
- 适用：内存密集 tiny-shape elementwise / 带 per-head 广播的 gate/激活类算子（heads 小、每核单 task、launch 固定开销占比高）
- 收益主源：消除 S-pipe 标量读 + 向量链 pass 精简 + 对齐广播，把 kernel 时间压向 launch 地板
- 不适用：计算密集大 shape（双缓冲/流水仍是主导）；per-head 广播已向量化且 hk 恒 64 对齐的算子
- 注意：Select/CompareScalar 的 64 对齐要求是 Ascend910B 向量单元的硬约束（非 64 倍数 count 导致 mask 错乱）

**Strategy IDs**: X33（及关联 P112/P84/P95/P89/P12）
