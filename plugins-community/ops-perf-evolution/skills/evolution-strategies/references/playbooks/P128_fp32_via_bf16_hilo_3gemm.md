# P128 Playbook: FP32-via-BF16 Hi/Lo 拆分 3-GEMM 落地 SOP

> **强制流程**。完整实战记录与参数依据见 `references/discovered/disc_X17.md`。

## Step 1: 取证

```bash
grep -nE "float|Cast\(|Mmad|ReduceSum" op_kernel/*.{cpp,h} > /tmp/p128_locations.txt
```
确认：fp32 数据通路位置、当前 GEMM 由什么实现（标量/向量模拟？）、容差是否允许 ~1e-4。

## Step 2: 拆分与布局（单趟 split-xtrans）

1. AIV kernel 从 fp32 源一次产出 hi/lo 双份目标布局：`hi = Cast_bf16(x, CAST_RINT)`、`lo = Cast_bf16(x − hi, CAST_RINT)`
2. **禁止加 NaN fixup**（会破坏 inf 语义；hi=±inf 时 lo=NaN 与 golden 图样天然匹配）

## Step 3: 3-GEMM 组织

1. hi/lo 双 L1 面板 + 双 workspace；每 k tile：2 次展开 + 3 Mmad（hi·hi + hi·lo + lo·hi，fp32 累加）
2. L0 队列深度：双 tile chunk → depth-4（4×16KB 占满 L0，展开与 Mmad 重叠）；单 tile → depth-2（depth-4 在单面板几何挂死设备）
3. 不要舍 lo·hi 项做 2-GEMM（~2e-3 误差贴容差边缘，已否决）
4. 不要做 4 加载变体（省 33% 流量 < 失去跨面板预取的损失，已证回退）

## Step 4: 验证纪律

- ±inf 特殊值 case **不能**用于验证 kExtension 类展开改动（"双 NaN 巧合"掩盖错误），必须用有限值 case
- 精度预期：~1e-4 相对误差；性能预期：相对标杆 0.3→0.5×（3 倍 cube 工作量是固有下限）
