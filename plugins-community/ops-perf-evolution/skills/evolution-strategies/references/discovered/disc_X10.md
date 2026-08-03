---
id: X10
origin: discovered
discovered_round: 3
discovered_from: level3_conv_2d-cannbench-web-submit (P0-P2)
base_speedup: 4600x (geomean 0.00015 → 0.69 全程；本轮 0.00015 → 0.031)
---

# Strategy X10: Scalar-Fallback → Cube GEMM 多级流水重构（含几何模式分发）

## 核心思路
当 kernel 是"正确性回退产物"——纯 AIV 标量循环（逐元素 `GetValue`/`SetValue` 点积）、
Cube 完全闲置——任何局部向量化都只是 10~100× 的过渡，必须做**架构性重构**：
把计算搬到 Cube，按 host-zero-prep 多级 kernel 流水组织（布局变换 AIV kernel +
GEMM AIC kernel + epilogue AIV kernel），host 只发射 kernel 与分配 workspace
（`torch::empty`），零 at:: 计算。同时按**几何特征在 host 侧做模式分发**
（mode dispatch），不同 case 类走不同 GEMM 组织，而非单一通用路径。

关键执行策略：**保正确性增量迁移**——新路径按 dtype/几何逐个启用，未覆盖的
case 保留旧 kernel 兜底，全程保持 N/N 正确性不回退；每步带环境变量开关可 A/B。

## 适用场景
- **算子类型**: 有可映射到 GEMM 的计算（conv/im2col 类、matmul 变体、einsum 类），
  当前实现为标量/纯 Vector 回退路径，与标杆差距 ≥2 个数量级
- **瓶颈类型**: compute-domain 错配（该用 Cube 却在 Vector 上标量算）
- **前提条件**: 平台有 Cube 单元且输入 dtype 可被 Cube 消费（fp32 见 X11）；
  存在同平台已验证的参考实现时优先移植而非重写
- **预期收益**: 10³~10⁵×（标量 conv2d → Cube GEMM 实测 0.00015→0.031，最终 0.69×）

## 实现要点
1. 流水拆分：wrepack（权重→K-major ws）/ im2col 或 xtrans（特征图布局）/
   conv（Nd2Nz→L1→LoadData3D→L0→Mmad fp32 acc→Fixpipe）/ epilogue
   （bias Adds + cast + 布局写出，跨行大块处理）
2. 模式分发（conv 实例）：
   - 1×1、s=1、p=0、对齐 → 翻转 GEMM（X13），省两级发射
   - Ci%16==0 且 xNz≤L1 → fused img2col LoadData3D 随路展开（X12），免物化
   - 其余 → normal im2col 物化 + GEMM（通用回退）
3. **对齐补零是必选项不是优化项**：Ci/Co 非 16 倍数时 x 与 filter 同步补零
   （pad 通道对点积贡献恒 0，语义安全；标杆实现的 MemSet 也在做同样的事）
4. 数据通路纪律（dav-c220）：GM→L1 只走 Nd2Nz 随路转 NZ（plain 2D DataCopy
   GM↔L1 不可靠）；非 32B 倍数传输走 DataCopyPad（plain DataCopy 静默截断）；
   m/n 尾块满 tile 计算 + Fixpipe 裁剪（dstStride=完整行宽）
5. 评估门控：全量精度 PASS（含 ±inf/NaN 特殊值 case）→ 性能对比 → 失败回滚快照

## 来源
发现于 cann-bench conv_2d 提交优化（20 case，fp16/fp32/bf16 混合）。
原始提交为保 20/20 正确性从错误的 im2col 实现退到标量直接卷积（76ms 级），
重构后 fp16/bf16 全部上 Cube，3 个 case 反超标杆（最高 1.98×）。
