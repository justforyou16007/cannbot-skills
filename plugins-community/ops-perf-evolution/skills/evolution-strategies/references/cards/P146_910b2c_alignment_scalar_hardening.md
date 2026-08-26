---
id: P146
bottlenecks: [mte2_stall, scalar_loading, ub_memory_pressure]
op_families: [attention, elementwise, normalization, broadcast_mask]
complexity: L1
conflicts_with: []
synergizes_with: [P144, P145, P10, P66]
requires: []
has_preconditions: true
has_playbook: false
quantified_gain:
  - shape: "64_TripletAttention 50-case（910B2C，小 H×W 7×11~21×23 全覆盖）"
    baseline_us: 4966.7
    optimized_us: 380.9
    speedup: 1.28
    source: "ops-perf-evolution output/64_TripletAttention_v2 round_3/parallel_2（Wpad=RoundUp8、Hpad8 列读、2-block 读、广播 8× 缓冲）"
---

# P146: 910B2C 非对齐 DMA 与标量通道加固套件

## 核心思想

910B2C 上小空间维（H×W ≤ 23×23）算子的搬运/标量行为与常规假设差异很大，违反即
**静默丢数据 / UB 越界 / npuSynchronize 崩溃**。以下六条是本族算子反复踩坑后验证的硬约束，
命中 CV-Attn 或任何小张量纯 AIV 算子时**无条件套用**（不必等 profiling 证据）：

1. **DataCopy/DataCopyPad 的 `blockLen` 必须 32B 对齐**：行宽用 `Wpad = RoundUp8(W)` 补到 32B 倍数。
   4B/非对齐 blockLen 会**静默失败**（无报错、读到垃圾）。`srcStride` 同样须 32B 对齐。
2. **单 block 读不可靠**：逐通道 slot 至少要按 `max(hsL, 2)` 行读（`hsL==1` 时读 2 行），
   否则 `1×Wpad` 单行读偶发丢数据。
3. **列读用 strided blockLen=32B**：`a_ch` 这类按列消费的图，stage 成 `(B, C, Hpad8)` 行布局
   （`Hpad8 = RoundUp8(H)`），列读用 `blockLen=32B` strided；`+8` 尾浮标保证末通道 8-float 读不越界。
4. **标量通道异步于向量流水**：热路径**禁用 scalar `GetValue`**；所有要进向量计算的标量用
   **1-elem 向量 op + 8-elem 对齐槽**承载。唯一的标量读允许在 Init 一次性载权重。
5. **Broadcast 工作缓冲按 `8 × dst` 容量**：`2×` 启发式会低估 `{1,Wpad}→{1,Wpad}` 广播的临时需求，
   溢出相邻 UB 缓冲（间歇性错误，非 shape 决定）。
6. **`UB→GM DataCopyPad` 偶发丢写**：zpool 这类「写后要被同/异核再读」的中间量，
   要么挪 host（P144），要么分 launch 读（P145），不要在 UB 写 GM 后立刻信任其值。

## 代码骨架

```cpp
constexpr int32_t Wpad = RoundUp8(W);          // (1) blockLen 32B 对齐
constexpr int32_t Hpad8 = RoundUp8(H);         // (3) 列读 strided 布局
// x 读到 UB：逐通道 slot 按 max(hsL,2) 行预留（2-block 读）(2)
pipe.InitBuffer(ubXout, (cb * MaxI(hs, 2) * Wpad) * sizeof(T));
// a_ch stage 为 (B, C, Hpad8) 行，列读 blockLen=32B (3)
pipe.InitBuffer(ubBwork, (8 * MaxI(Wpad, RoundUp8(hs))) * sizeof(T));  // (5) 广播 8×
// 标量用 1-elem 向量 op 写 8 对齐槽，而非 scalar GetValue (4)
ubAcc.SetValue<uint8_t>(0, 0); // 占位：实际用 Duplicate/Mul 等向量 op 携带标量
```

## 关键修改点

1. **先补对齐再谈性能**：`64_TripletAttention` round_1 的 2-launch 方案每个 case 崩溃，
   根因正是 `GatherAndAdd` 用非 32B 对齐 stride 的 2D DataCopy + `DataCopyPad blockLen=srcCols*sizeof(T)`
   （`W2p=22` 浮标=88B 非 32B 倍数）。修复后（round_3）立即可跑。
2. **标量通道是隐蔽瓶颈**：baseline scalar pipe 占比高（36_BAM 观测 40%+）时，先查热路径
   是否大量 `GetValue`；改 1-elem 向量 op 承载可同时降 scalar 与 icache miss。
3. **与 P66/P10 的关系**：P66 管 GM 512B 对齐（Cube 侧）；P146 管 **UB/MTE2 32B 对齐**（向量侧）。
   小空间行（<512B）无法上 512B 粒度，32B 是向量路径的现实地板。

## 前置条件

- 算子行宽/中间图不是 32B 倍数（小 H×W、奇偶 C/W）。
- 目标 910B2C（A2 系列 910B 的向量 MTE2 对齐要求一致；A3 需另行确认）。

## 实测验证

- `64_TripletAttention_v2` round_3/parallel_2：全部 50 case 有效、50/50 精度、geomean 1.28x。
  对齐修复是「能否跑通」的分水岭（round_1 崩溃 → round_3 通过）。
- `36_BAM_evo`（0.98x）同样受益于 32B per-row DataCopyPad 模式。
