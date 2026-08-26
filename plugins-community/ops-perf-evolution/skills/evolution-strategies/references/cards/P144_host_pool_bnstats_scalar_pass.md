---
id: P144
bottlenecks: [no_overlap, scalar_loading]
op_families: [attention, elementwise, normalization]
complexity: L1
conflicts_with: []
synergizes_with: [P145, P146, P147, A2]
requires: [P147]
has_preconditions: true
has_playbook: false
quantified_gain:
  - shape: "64_TripletAttention 50-case（910B2C，geomean vs PyTorch）"
    baseline_us: 4966.7
    optimized_us: 380.9
    speedup: 1.28
    source: "ops-perf-evolution output/64_TripletAttention_v2 round_3/parallel_2（x0_fix1_fix2）"
---

# P144: host 端计算全局池化与 BN 训练统计，标量传参替代设备归约/往返同步

## 核心思想

CV-Attn（CP8 门控广播）与含 `BN(training)` 的算子，其数学链常是：
`zpool(mean/max) → conv → 激活 → 全局归约 E[y]/E[y²] → BN scale/shift → 门控广播`。
其中 **zpool 与 BN 统计都是纯归约**，与逐元素门控不同——它们**不必**在设备上做。
把这两段挪到 **host CPU（fp64）** 计算，只把最终 **6 个 BN scale/shift 标量**（或加 zpool 结果）
通过 kernel 参数 / 小 GM 缓冲传入，即可一次性消除：

1. **D2H 统计回读 + stream sync**（host 原本就要等 stats launch 完才能算 scale/shift）；
2. **设备端 RA ReduceSum**——910B2C 上 `{n,48}` 的 RA ReduceSum **只可靠归约前 16 列**
   （col16/24/32/40 间歇不归约，保持首行原始值），BN 需要 6×2 列，必踩；
3. **一个完整 launch 的固定开销**（staging pass 不再需要 device reduce 相位）。

基线 `64_TripletAttention`（0.158x）的主瓶颈正是 host-CPU 往返 + 多 launch 同步；
最优变体（round_3/parallel_2）靠本策略把 geomean 推到 **1.28x**（50/50 精度）。

## 代码骨架

```cpp
// ---- host: 一次 stats launch 后 CPU 归约，产出 6 个标量 ----
at::Tensor statsWs = at::zeros({used, 48}, opts);   // 每核 48 列（8-float slot 步长）
launch_stats_kernel(/* mode=1 ... */);              // 设备只写 per-core 部分和
c10_npu::getCurrentNPUStream().synchronize();
{
    auto statsCpu = statsWs.to(at::kCPU);           // D2H（仅一次，small）
    const float *sp = statsCpu.data_ptr<float>();
    double s[6] = {0};
    for (int c = 0; c < used; ++c)
        for (int k = 0; k < 6; ++k) s[k] += (double)sp[c * 48 + k * 8];
    for (int k = 0; k < 3; ++k) {
        double mean = s[2*k]   / Ns[k];             // E[y]
        double ey2  = s[2*k+1] / Ns[k];             // E[y²]
        double var  = ey2 - mean*mean; if (var < 0) var = 0;  // fp32 抵消护栏
        sc[k] = (float)(gamma[k] / std::sqrt(var + 1e-5));
        sh[k] = (float)(beta[k]  - mean * sc[k]);
    }
}
// ---- 后续 launch 直接把 6 标量作 kernel 参数传入，不再 reduce ----
EXEC_KERNEL_CMD(kernel, nCore, /* ..., */ sc0, sh0, sc1, sh1, sc2, sh2);
```

## 关键修改点

1. **zpool 也挪 host**：910B2C 的 `UB→GM DataCopyPad` 偶发丢写（zpool 写后读损坏 conv 输入）。
   zpool 是纯 mean/max，host 一次算完更稳（此时 `mode-0 zpool launch` 直接禁用）。
2. **方差用 fp64**：`E[y²]-mean²` 在 fp32 会严重抵消（负方差），host fp64 规避；仍保留 `var<0→0` 护栏。
3. **标量 vs GM 缓冲**：6 标量直接作 kernel 参数最简单；若 kernel 侧需要，也可 stage 到 `scaleWs[64]`
   8-float 对齐槽（见 P146 标量通道规则）。
4. **与 A2 Welford 的关系**：本策略是「把归约挪 host」，不改变归约算法本身；若归约必须在设备完成，
   用 A2 Welford + 逐列 ReduceSum（仍受 16 列上限约束，必要时分 2 次归约）。

## 前置条件（命中才用，否则走 P145/设备归约）

- 归约输入对 host 可访问（`to(cpu)` 的 workspace 较小，一次 D2H 可接受）；或算子结构允许 host 直接算 zpool。
- **数值一致性**：zpool 数值须复现 NPU conv 输入转换（12-sig-bit RNE，见 P147）——host 侧必须对
  zpool 与卷积核应用 `RoundTo12Sig`，否则 K=3 大 case 相对误差超 verifier 桶被拒。
- 归约域是「全局/通道级」小域（B×C×H 量级），非逐元素大域。

## 实测验证

- `64_TripletAttention_v2` round_3/parallel_2（x0_fix1_fix2）：host 算 zpool + host fp64 归约 BN 统计
  + 6 标量传参，50/50 精度，geomean **1.28x** vs PyTorch（基线 0.158x，提升约 8×），
  相对 `_v2` 前序变体也消除了 staging 的 device-reduce 相位。
- 同族参考：`36_BAM_evo` round_8/parallel_2（0.98x）使用相近架构（host 侧 zpool + 多 launch），
  印证该路由对 CV-Attn 双分支类普适。
