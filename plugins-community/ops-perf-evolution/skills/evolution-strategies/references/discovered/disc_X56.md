---
id: X56
origin: discovered
discovered_round: 1
discovered_from: "25_StridedAttention evolution round_1/parallel_2 (x0, open_exploration)"
base_speedup: "10-case msprof quick geomean 0.586x -> 0.811x vs PyTorch (+38%)，10/10 精度 PASS；fp16 大 case 0.269->0.692 / 0.471->0.686，bf16 0.211->0.568 / 0.140->0.417；fp32 大 case 未变（瓶颈在投影/outproj GEMM）"
---

# Strategy X56: STRIDE-SPARSE residue-block Cube attention（stride-mask 按 residue 分组只算有效列）

## 核心思路
strided-mask self-attention 的 mask `score[i,j]` 有限 iff `i%stride == j%stride`，因此每个 query 行只需要**自己 residue class 的 Srmax=ceil(S/stride) 个 key**，而非全 S_pad 宽。父内核算稠密 S_pad 宽 QK^T/softmax/PV，浪费 stride 倍 Cube+AIV 工作量。本策略把 attention task 从 `(b,h,query-block)` 重组织为 `(b,h,residue,query-block)`，attention 宽度从 S_pad 收窄到 Srmax16（stride 倍缩减）：

1. **Q/V 走 strided 2D block load**（行 stride `stride*HDP16`）从自然 `[B,H,S_padA,HDP16]` workspace 读（`S_padA=stride*Srmax16` 保持所有 residue 行 in-bounds）——**无需 host Q/V 重排**。
2. **fp32 K^T host residue-grouped** 到 `[B,H,HDP16,stride*Srmax16]`（view+permute+contiguous）；fp16/bf16 K/V 直接 strided native 读。
3. **per-residue padding mask**（真实 key 0、tail -2^30）替代全 stride×S_pad mask 表——mask 内存与 softmax 宽度同步收窄。
4. **bf16 大 case 切到 Cube attention**（P hi/lo 拆分恢复 fp32 精度）替代 O(S²) AIV vector attention。

## 适用场景
- stride-mask（每行仅 1/stride 列有效）的 self/cross attention，stride>=2
- attention 核心（QK^T/softmax/PV）是大 case 瓶颈、Cube 已建成（P120）的算子
- 与 P123（causal KV skip 循环界裁剪）互补：P123 是 skip 循环界，X56 是**重排分组后只算有效列**，实现机制不同（residue-grouped 布局 vs 循环界裁剪）

## 实现要点
- `S_padA=stride*Srmax16`（Srmax16=ceil(S/stride) 对齐 16）使所有 residue 行在 Q/V workspace in-bounds
- fp32 K^T 必须 host transpose+residue-group；fp16/bf16 可 strided native 读（sz=2 支持 strided 2D load）
- per-residue tail mask `[CntFor(r), SPR)` 处理尾部不完整 residue
- 写回映射 `sOrig = sl*stride + r` 恢复原序列位置
- bf16 切 Cube 需 P hi/lo 拆分（fp32 L0C 累加 + hi/lo 两平面）保精度

## 来源
自动发现于 25_StridedAttention 第 1 轮进化（x0，open_exploration 从第一原理推理）。
10-case geomean 0.586x → 0.811x（+38%），10/10 精度 PASS。fp16 大 case 0.269→0.692、
0.471→0.686；bf16 大 case 0.211→0.568、0.140→0.417。fp32 大 case 未变（瓶颈在投影/
outproj GEMM，非 attention）。与同轮 n5（P123 residue-grouped，0.728x）殊途同归但幅度更高。
