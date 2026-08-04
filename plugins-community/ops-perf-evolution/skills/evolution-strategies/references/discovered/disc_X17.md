---
id: X17
origin: discovered
discovered_round: 4
discovered_from: level3_conv_2d-cannbench-web-submit (P4/P5)
base_speedup: fp32 geomean 0.000 → 0.50（单 case 最高 0.60，标量兜底 ~21000x 慢 → 0.5x）
---
> **已晋升为 P128**（card + precondition + playbook 三件套见 `cards/`、`preconditions/`、`playbooks/`）。本文件保留为实战详细记录（含实测数据与否定项依据），卡片内容以 P128 为准。


# Strategy X17: FP32-via-BF16 Hi/Lo 拆分 3-GEMM（Cube 不吃 fp32 时的保精度路径）

## 核心思路
910B Cube 的 Mmad 不直接消费 fp32 输入。将 fp32 操作数拆分为 bf16 双分量
`x ≈ hi + lo`（hi = cast_bf16(x)，lo = cast_bf16(x − hi)），用 3 个 bf16 GEMM
`hi·hi + hi·lo + lo·hi`（fp32 累加）逼近 fp32 GEMM，获得 Cube 级吞吐 + 接近
fp32 的精度（~1e-4 相对误差，远小于 torch fp32 conv 自身的 ~1e-3）。

关键配套：**单趟 split-xtrans**——一个 AIV kernel 从 fp32 源直接产出 hi/lo 双份
目标布局（如 bf16 xT），省掉独立的 split kernel 和第二趟 im2col/转置物化。

## 适用场景
- **算子类型**: fp32 输入的 GEMM 类计算（conv、matmul），Cube 仅支持 fp16/bf16
- **瓶颈类型**: compute-domain，fp32 被困在 Vector/标量路径
- **前提条件**: 容差允许 ~1e-4 相对误差（cann-bench 级别通常可过）；
  **±inf/NaN 特殊值必须验证**——hi=±inf 时 lo=NaN，感受野含 inf 则累加 NaN，
  与 golden 的 NaN 图样天然逐位匹配（无需显式 NaN 掩码逻辑）
- **预期收益**: 相对标量 fp32 兜底 3~4 个数量级；相对标杆 0.3→0.5×
  （3 倍 cube 工作量是固有下限）

## 实现要点
1. 拆分：`hi = Cast_bf16(x, CAST_RINT)`，`lo = Cast_bf16(x − hi, CAST_RINT)`，
   两级 Cast + Sub，无 NaN fixup（fixup 反而破坏 inf 语义）
2. **2-GEMM（舍 lo·hi 项）经精度分析否决**：~2e-3 相对误差贴容差边缘且存在
   误差抵消风险；3-GEMM 是精度安全下限
3. GEMM 组织：hi/lo 双 L1 面板 + hi/lo 双 ws，每 k tile 2 次展开 + 3 Mmad；
   **A2/B2 队列深度按每 chunk 的 tile 数定**：双 tile（hi+lo）需 depth-4
   （4×16KB=64KB L0 占满）让展开与 Mmad 重叠，实测 conv 段再提速 ~30%；
   单 tile 用 depth-2 即可（depth-4 在单面板几何上挂死设备）
4. 4 加载变体（Ahi/Alo/Bhi/Blo 各载一次省流量）已证**回退**——失去跨面板
   预取的损失 > 省 33% 流量的收益；保留 6 虚拟面板 + 预取为默认
5. 验证纪律：±inf case 不能用于验证 kExtension 类展开改动（"双 NaN 巧合"
   会掩盖错误），必须用实数有限值 case

## 来源
发现于 cann-bench conv_2d fp32 7 case（含 ±inf 特殊值 case）。
fp32 geomean 0.363（直接 3-GEMM）→ 0.50（单趟 split-xtrans + depth-4 L0）。
