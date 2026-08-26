---
id: P145
bottlenecks: [no_overlap, mte3_stall]
op_families: [attention, elementwise, normalization]
complexity: L1
conflicts_with: []
synergizes_with: [P144, P146, P1]
requires: []
has_preconditions: true
has_playbook: false
quantified_gain:
  - shape: "64_TripletAttention 50-case（910B2C，geomean vs PyTorch）"
    baseline_us: 4966.7
    optimized_us: 380.9
    speedup: 1.28
    source: "ops-perf-evolution output/64_TripletAttention_v2 round_3/parallel_2（4-launch：zpool→stats→staging→output）"
---

# P145: 多 launch 纯 AIV 流水架构（避免同 launch GM 写后读）

## 核心思想

**910B2C 同一次 kernel launch 内，MTE3 写 GM → MTE2 读 GM 的可见性不可靠**：
同一 launch 里先写 workspace 再读回会读到中间值/丢写，`npuSynchronize error` 或数值错乱。
可靠的跨阶段可见性由 **stream 顺序保证的多次 launch** 提供（两次 launch 之间 stream 隐含 barrier）。

因此对「多相位、有 GM 中间量」的纯 AIV 算子（典型：CV-Attn 的
`zpool → BN-stats → staging(conv+BN+激活) → output(门控广播)`），采用**分段 launch 流水**：

```
launch#0 mode=zpool   : 计算全部 z workspace（可用 P144 挪 host 后禁用）
launch#1 mode=stats   : per-core 部分和 → stats_ws（列主序 [stat][core]）
launch#2 mode=staging : 归约 stats + conv/BN/ReLU/Sigmoid → 三个 attention 图 stage 到 GM
launch#3 mode=output  : 读 x + staged 图（MTE2 only），门控广播相乘写 out
```

`64_TripletAttention` 基线的 4-launch 纯 AIV 即此架构；round_3/parallel_2 保留该骨架
但把 zpool（launch#0）与 stats 归约（launch#2 前半）改 host（P144），净收益最大化。

## 代码骨架

```cpp
// ---- host: 分段 launch，stream 顺序保证 GM 可见性 ----
EXEC_KERNEL_CMD(kernel, nCore, xPad, out, /* ..., */ mode0);  // zpool（或 P144 挪 host 跳过）
c10_npu::getCurrentNPUStream().synchronize();
EXEC_KERNEL_CMD(kernel, nCore, /* ..., */ mode1);             // stats: per-core 部分和
c10_npu::getCurrentNPUStream().synchronize();
// （可选）host 归约 stats_ws → 6 标量（P144）
EXEC_KERNEL_CMD(kernel, nCore, /* ..., */ mode2, sc0,sh0,...); // staging: conv+BN+激活 → 图 stage
EXEC_KERNEL_CMD(kernel, nCore, /* ..., */ mode3);             // output: 读图+广播相乘
```

## 关键修改点

1. **禁止同 launch 内「写 A 再读 A」**：跨相位数据一律走 GM workspace + 独立 launch。
   910B2C 上 MTE3→MTE2 事件计数有限（15 次），`DataSyncBarrier/PipeBarrier` 无法可靠替代 launch 边界。
2. **launch 数不是越少越好**：多 launch 有固定开销；若相位间数据量小且可被 host 算完（P144），
   合并/删除该 launch 的收益 > 保留。基线 3-launch → 最优 4-launch 的取舍按「每个 launch 是否
   有独立可并行工作」判定。
3. **与 P1 双缓冲协同**：launch 内部用 P1 隐藏 GM 搬运；跨 launch 的重叠靠多 stream/多 NPU 队列，
   非本策略范围。
4. **小 launch-bound 场景**（H×W 极小，如 7×11）不要机械套多 launch——优先 host 侧（P144）削 launch。

## 前置条件

- 存在跨相位的 GM 中间量（workspace），且同一 launch 内 MTE2/MTE3 往返在 910B2C 不可靠。
- 算子可拆为「无跨核同步」的多个纯 AIV 相位（每相位按 (b, cb, h) 切核，见 P104）。

## 实测验证

- `64_TripletAttention_v2` round_3/parallel_2：4-launch（zpool 挪 host 后实际 3 个 launch），
  50/50 精度，geomean 1.28x。round_1 尝试的「2-launch 纯设备」方案（同 launch 写读）
  在**每个 case** 都 `npuSynchronize error` 崩溃——同 launch GM 往返不可靠的直接证据。
- 同族参考：`36_BAM_evo` 多 launch + host zpool（0.98x），架构一致。
