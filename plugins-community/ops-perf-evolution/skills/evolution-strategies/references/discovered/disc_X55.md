---
id: X55
origin: discovered
discovered_round: 2
discovered_from: "46_DAModule evolution round_2/parallel_1 (x0_x1, open_exploration 延续)"
base_speedup: "10-case 子集 msprof quick geomean 0.41x vs PyTorch（父 0.20x -> 0.41x，2x）；大 C case 大幅改善：case3 [1,512,4,4] 0.051->0.345x、case9 [1,1024,7,7] 0.034->0.30x、case2 [1,256,4,4] 0.098->0.549x；全量 50 精度 PASS"
---

# Strategy X55: Shared-im2col 双分支 Conv-on-Cube（一份 im2col 供两分支 + HF32-for-fp32 conv 路由）

## 核心思路
融合算子含**两个共享同一输入 x 的 3x3 conv 分支**（如 Dual Attention 的位置/通道注意力分支）时：
1. **一份 im2col 矩阵供两分支复用**：纯 AIV kernel 构建一次 im2col Ic[B,9C,N]（fp32），两个分支的 conv GEMM 都读它——im2col 构建成本从 2 次摊销为 1 次。
2. **两个 fp32 Cube GEMM（transA）**：`yp = wpF^T @ Ic`、`yc = wcF^T @ Ic`（[C,N] = [C,9C] @ [9C,N]），wpF/wcF 直接用原始 [C,9,C] 权重当 [9C,C]（TRANS_A），零 host permute/repack。
3. **HF32-for-fp32 conv 路由**：fp32 输入启用 HF32（匹配 NPU 参考 conv2d 的 cube 舍入），fp16/bf16 走全 fp32（上cast 乘积精确匹配参考）。
4. **Shape 自适应路由**：C>=128 走 Cube conv；小 C 保留父节点已证高效的 Vector conv（0.9-1.0x）。

## 适用场景
- 多分支共享输入的 conv（Dual Attention / 并行 attention 结构）
- conv 是 Vector 外积累加、大 C case（C=256..1024）compute-bound 的融合算子
- 有多个 bmm/attention 已上 Cube、conv 成为剩余瓶颈的算子

## 实现要点
- im2col 布局 (B,C,9,N) 连续，K 行 = ci*9+k；权重 (ci,k,co) 行主 → transA GEMM 零 repack
- 32B 对齐：im2col 每行 slot 布局（slotElems=ceil(W*sizeof(T)/32)*32/sizeof(T)），DataCopyPad dst/src stride 按 slot
- 边界行（kh=0 的 h=0 与 kh=2 的 h=H-1）host 用 at::zeros 预清零，kernel skip 边界 task
- 9N store stride 溢出（fp16 N>3640）回退 Vector conv
- HF32 仅 fp32 conv 启用（匹配参考）；bmm 恒 SetHF32(false) 全精度

## 来源
自动发现于第 2 轮进化，算子 46_DAModule，x0_x1 节点（open_exploration），10-case 子集 speedup 0.41x vs PyTorch
