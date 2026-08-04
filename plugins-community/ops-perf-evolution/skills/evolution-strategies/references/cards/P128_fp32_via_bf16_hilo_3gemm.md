---
id: P128
bottlenecks: [compute_bound]
op_families: [matmul, special]
complexity: L2
conflicts_with: []
synergizes_with: [P129, P130, D1]
requires: []
has_preconditions: true
has_playbook: true
---

# P128: FP32-via-BF16 Hi/Lo 拆分 3-GEMM（Cube 不吃 fp32 时的保精度路径）

> 晋升自 X17（discovered，conv_2d fp32 实测：标量兜底 ~21000x 慢 → 0.5x 标杆）。详细实战记录见 `discovered/disc_X17.md`。

## 核心思想

910B Cube 的 Mmad 不直接消费 fp32 输入。将 fp32 操作数拆分为 bf16 双分量 `x ≈ hi + lo`（`hi = cast_bf16(x)`，`lo = cast_bf16(x − hi)`），用 3 个 bf16 GEMM `hi·hi + hi·lo + lo·hi`（fp32 累加）逼近 fp32 GEMM——获得 Cube 级吞吐 + ~1e-4 相对误差（远小于 torch fp32 conv 自身的 ~1e-3）。

关键配套：**单趟 split-xtrans**——一个 AIV kernel 从 fp32 源直接产出 hi/lo 双份目标布局（如 bf16 xT），省掉独立 split kernel 和第二趟 im2col/转置物化。

## 适用场景

- fp32 输入的 GEMM 类计算（conv、matmul），Cube 仅支持 fp16/bf16
- 瓶颈：fp32 被困在 Vector/标量路径（compute_bound）
- 前提：容差允许 ~1e-4 相对误差；**±inf/NaN 特殊值必须验证**（hi=±inf 时 lo=NaN，与 golden 的 NaN 图样天然逐位匹配，无需显式 NaN 掩码）

## 关键修改点

1. 拆分：`hi = Cast_bf16(x, CAST_RINT)`，`lo = Cast_bf16(x − hi, CAST_RINT)`，两级 Cast + Sub，**无 NaN fixup**（fixup 反而破坏 inf 语义）
2. **3-GEMM 是精度安全下限**：2-GEMM（舍 lo·hi 项）~2e-3 误差贴容差边缘且有误差抵消风险，已否决
3. GEMM 组织：hi/lo 双 L1 面板 + 双 ws，每 k tile 2 次展开 + 3 Mmad；**A2/B2 队列深度按每 chunk tile 数定**：双 tile 需 depth-4（4×16KB L0 占满）让展开与 Mmad 重叠（实测 conv 段再提速 ~30%）；单 tile 用 depth-2（depth-4 在单面板几何上挂死设备）
4. 4 加载变体（Ahi/Alo/Bhi/Blo 各载一次）已证**回退**——失去跨面板预取的损失 > 省 33% 流量的收益

## 常见陷阱

- ⚠️ ±inf case 不能用于验证 kExtension 类展开改动（"双 NaN 巧合"掩盖错误），必须用有限值 case
- ⚠️ 预期上限：3 倍 cube 工作量是固有下限，相对标杆约 0.3→0.5×

## 代码搜索关键词

```bash
grep -nE "float|fp32|Cast\(|Mmad" op_kernel/*.cpp op_host/*_tiling.cpp
```
