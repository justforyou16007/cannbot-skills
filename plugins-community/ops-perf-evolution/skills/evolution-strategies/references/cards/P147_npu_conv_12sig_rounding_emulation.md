---
id: P147
bottlenecks: [compute_bound, scalar_compute]
op_families: [attention, cv_fusion, elementwise]
complexity: L1
conflicts_with: []
synergizes_with: [P144, P129, A1]
requires: []
has_preconditions: true
has_playbook: false
quantified_gain:
  - shape: "64_TripletAttention 50-case（910B2C，K=3/5/7 全覆盖）"
    baseline_us: 0
    optimized_us: 0
    speedup: 1.0
    source: "ops-perf-evolution output/64_TripletAttention_v2：K=3 大 case 相对误差超 verifier 桶 → RoundTo12Sig 后 50/50 通过（使 1.28x 结果成立的前提）"
---

# P147: NPU 卷积输入 12-sig-bit 舍入模拟（host 侧复现 F.conv2d 数值）

## 核心思想

910B2C 的 `F.conv2d` 参考实现会先把**激活与卷积核**都转成 **12 有效位（12-sig-bit, RNE）**
格式，再做 fp32 乘累加。当你的 kernel 用向量路径复算同一卷积（或用 host 预计算 conv 输入）时，
若不做同样的输入转换，会留下 ~1e-3 量级的系统性偏差——**K=3 等特定 kernel 尺寸的 case 相对误差
超出 verifier 的相对误差桶而被判精度失败**。

解决：在 host 侧对进入卷积的激活（zpool 结果）与卷积核应用 `RoundTo12Sig`（round-to-nearest-even），
逐位复现 NPU conv 的输入转换：

```cpp
static float RoundTo12Sig(float v) {
    // 保留 12 个有效位（11 个小数位），RNE
    float s = std::copysign(1.0f, v);
    float a = std::fabs(v);
    int   e = std::ilogb(a);
    if (e <= -127) return v;                       // 次正规/零原样返回（RoundTo12Sig(0)==0）
    float m = a * std::pow(2.0f, 11 - e);          // 缩放到整数域
    m = std::round(m);                             // RNE 到整数
    return s * m * std::pow(2.0f, e - 11);
}
```

## 关键修改点

1. **只对「conv 乘数」做 12-sig**：zpool 激活 + 三个卷积核（`wch/wcw/whw`）转换；
   **16 个尾部标量（conv bias、BN gamma/beta）不是 conv 乘数，保持 fp32 精确**。
2. **halo 零不受影响**：`RoundTo12Sig(0) == 0`，padding 区域无需特判。
3. **与 P144 强绑定**：一旦把 zpool 挪到 host 计算，就必须在 host 侧对 zpool 结果套 12-sig，
   否则设备/参考 conv 的数值行为不匹配（K=3 大 case 是敏感桶）。
4. **识别信号**：kernel 编译通过、shape 正确、但 K=3 大 C 的 case 相对误差 ~1e-3 稳定超桶 →
   优先怀疑 conv 输入转换未复现，而非累加顺序（A1/A6 解决的是累加精度，不是输入量化）。
5. **设备向量路径等效**：若 zpool/激活在设备上生成，让设备把值按 12-sig 规整后再喂 conv
   （或依赖设备 conv 已做输入转换——仅当 conv 本身走 NPU 算子的 reference 而非自写向量卷积时成立）。

## 前置条件

- 自写向量卷积或 host 预计算 conv 输入，且目标是对齐 NPU `F.conv2d` 的数值输出。
- verifier 用相对误差桶且容忍度 ~1e-3 量级。

## 实测验证

- `64_TripletAttention_v2`：修复前 K=3 大 case（如 `[2,512,7,11]` 类）max_abs_diff 5.98~21.4、
  MERE 0.5~7.1，部分 NaN；对 zpool + 卷积核套 `RoundTo12Sig` 后 **50/50 通过**，
  使 P144/P145/P146 的 1.28x 结果具备交付前提（精度不达标则性能数据仅供参考）。
