---
id: X60
origin: discovered
discovered_round: 1
discovered_from: round_1/parallel_1
base_speedup: 1.40x (T2/T3; 310P W4A8 nibble 抽取链, 相对已含 X59 的 baseline)
---

# Strategy X60: 向量链发射最小化（代数重写 + calcount 合并）

## 核心思路
当 profiling 显示 Vector busy 高但数据吞吐极低（每条向量指令平均仅 ~4-5ns ≈ 发射率墙，
Scalar busy 与 Vector busy 同高且来自指令发射）时，优化目标应从"数据通路带宽"切换为
"指令/发射单元数"：(1) 用**位代数恒等式**重写数值链——如用伴随掩码 And（b&0x08）替代
floor 取整技巧链（x*k+c 后 cast），一条恒等式可证明的位运算等价式把 21 call/13 屏障的链压到
9 call/5 屏障；(2) 把 N 次小 calcount 的**同类 API call 合并为 1 个大 calcount call**——
底层微指令流完全相同（wrapper 按 repeat 拆分发射），但每 call 省 N-1 个 scalar call site
的发射/簿记开销。

## 适用场景
- 向量链密集、每个 op 后跟 PipeBarrier、Scalar busy 占比 20%+ 的内核（发射开销主导型）
- 无 vector ShiftLeft/ShiftRight 的老架构（310P/m200 等）上用 half 域乘加技巧模拟位操作的链
  （nibble 提取、位域拆装）——代数重写收益最大
- 症状识别：改动前后 MTE2/MTE3 不变而 Scalar busy 大幅下降 = 发射开销被压缩的直接证据
- Gather 类 API：多次 128 元素 call 合并为单个大 calcount call（注意 repeatTime 上限，
  m200 Gather 每 repeat 128 half、wrapper 拆分不改变微指令流）

## 实现要点
1. **恒等式必须代数可证**（不是数值近似）：例 valLo = t - 16*floor(t/8) ≡ t - 2*(b&0x08)；
   valHi = floor(byte/16) ≡ (int8 重解释 b&0xF0)*0.0625（精确 2 的幂除）。逐位一致红线保持
2. **call 合并的硬件约束先查头文件**：Gatherb 在 m200 显式 NOT_SUPPORT；Gather 仅 2/4 字节
   类型、每 repeat 上限 128 half——先排除死路再设计（本策略与 X59 的批放大正交：X59 放大
   数据粒度，X60 压缩指令粒度）
3. **屏障精简逐条分析**：dst 不相交的相邻 op 背靠背发射免屏障；真 WAR/RAW 边保留
   （同 pipe 指令按序发射是前提）
4. **收益判据**：每窗口（2048B）发射单元数下降比例 ≈ Scalar busy 下降比例；
   Vector busy 下降幅度 = 被删 call 数 × ~4.5ns × 窗口数

## 来源
自动发现于 quant_batch_matmul_v4 第二轮进化 round_1/parallel_1（ops-nn，Ascend310P3，
W4A8 W4Trans 段），相对已含 X59 的 baseline（= 仓 HEAD 78a8081b4）：T1 1.29x / T2 1.40x /
T3 1.40x / G1 1.36x / G2 1.13x，精度 5/5 bit-exact。流水线证据：Vector 848→616μs、
Scalar 310→83μs（MTE2/Mmad/MTE3 不变）。
