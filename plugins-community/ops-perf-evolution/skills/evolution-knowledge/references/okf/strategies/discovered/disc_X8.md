---
schema_version: okf.v1
kind: discovered_strategy
id: X8
origin: discovered
title: Host-Side Precomputation of Constant Offset Tables
description: 许多 AscendC kernel 在 `Init` 阶段会计算一些**仅由超参数决定、与输入数据无关** 的常量表，例如： - 量化/反量化的 bit offset 表 - gather 索引表 - stride/alignment 修正表 - RoPE 频率表 - softmax 的 stride offset 表 这些表目前在 device 端通过标量循环计算（`SetValue`），每个
confidence: single_op
status: active
op_families: ['attention', 'flash_attention']
bottlenecks: ['compute_bound', 'mte2_stall', 'mte3_stall', 'no_overlap', 'scalar_compute', 'tiling_imbalance']
tags: ['attention', 'compute_bound', 'decode', 'flash_attention', 'mte2_stall', 'mte3_stall', 'no_overlap', 'scalar_compute', 'tiling_imbalance']
discovered_from: best_decode deep profiling (NPU 12, msprof compare)
base_speedup: 实测有效（2026-07-20，best_decode2 round_1 n2 host Pi 转置缓存）：mean_asc 149.58→125.64us（-16.0%，同窗口），R1 主线，"零风险最大单项"
---
# Strategy X8: Host-Side Precomputation of Constant Offset Tables

## 核心思路

许多 AscendC kernel 在 `Init` 阶段会计算一些**仅由超参数决定、与输入数据无关**
的常量表，例如：

- 量化/反量化的 bit offset 表
- gather 索引表
- stride/alignment 修正表
- RoPE 频率表
- softmax 的 stride offset 表

这些表目前在 device 端通过标量循环计算（`SetValue`），每个 kernel launch
都要重复执行，带来以下开销：

1. **标量指令开销**：`for` 循环 + 位运算 + `SetValue` 占用 Scalar Pipeline；
2. **初始化延迟**：必须在任何 Vector 计算前完成，拉长头开销；
3. **UB 占用**：需要为这些表分配 UB buffer（如 `kOffBaseBuf_`、`kInvBaseBuf_` 等）。

若将这些表的计算移到 **Host 侧**（在 Tiling 阶段完成，或通过 Python 预计算后
作为 GM buffer / TilingData 传入），device 端只需一次 `DataCopy` 加载，可以：

- 消除 device 端标量循环；
- 减少 kernel 启动延迟；
- 降低 UB 占用（部分表可改用 GM 常量或 TilingData 存储）。

## 适用场景

- **算子类型**：量化 attention、FlashDecode、RoPE、Embedding、任何 `Init` 阶段
  有大量标量预计算的 kernel。
- **触发条件**（满足任一）：
  1. `Init` 阶段存在 `for` 循环填充 offset/index/scale 表；
  2. 这些表完全由 `D_`、`bits_`、`slotLen_` 等 Host 已知参数决定；
  3. profiling 显示头开销中 Scalar 占比较高，但表本身计算量很小；
  4. 常量表占用较大 UB（如 `D*D` 或 `D*TB` 级别），挤压数据 buffer 空间。

## 前提条件

1. Host 侧能够访问生成这些表所需的全部参数（通常来自 TilingData 或 op 属性）。
2. 常量表尺寸可控，不会导致 TilingData 过大（通常 < 1KB 可直接放 TilingData，
   更大则放 GM workspace）。
3. 不引入新的 GM 往返：若表已经需要从 GM 加载（如 centroids），Host 预计算
   应仅改变计算位置，不改变数据流向。

## 实现要点

### 典型改造：反量化 offset 表

**Device 端基线**：

```cpp
__aicore__ inline void Init(...) {
    LocalTensor<int32_t> kOffB = kOffBaseBuf_.Get<int32_t>();
    LocalTensor<int32_t> kInvB = kInvBaseBuf_.Get<int32_t>();
    for (int64_t i = 0; i < D_; i++) {
        int64_t kb = i * mseBits_;
        int64_t kbyi = kb >> 3;
        int64_t ksh = kb & 7;
        kOffB.SetValue(i, (int32_t)(kbyi * 4));
        kInvB.SetValue(i, (int32_t)((127 - ksh) << 23));
    }
    // 同样还有 vOffB、vInvB、strideOff 等
}
```

**Host 侧预计算**：

```cpp
// op_host/*.cpp
void CalcOffsetTables(int64_t D, int64_t mseBits, int64_t vqb,
                      std::vector<int32_t>& kOff,
                      std::vector<int32_t>& kInv,
                      std::vector<int32_t>& vOff,
                      std::vector<int32_t>& vInv) {
    kOff.resize(D); kInv.resize(D);
    vOff.resize(D); vInv.resize(D);
    for (int64_t i = 0; i < D; i++) {
        int64_t kb = i * mseBits;
        kOff[i] = (int32_t)((kb >> 3) * 4);
        kInv[i] = (int32_t)((127 - (kb & 7)) << 23);
        int64_t vb = i * vqb;
        vOff[i] = (int32_t)(((vb >> 3)) * 4);
        vInv[i] = (int32_t)((127 - (vb & 7)) << 23);
    }
}
```

**Device 端优化后**：

```cpp
// 方案 1：通过 TilingData 传入（表很小，如 D=128 时 4*128*4B = 2KB）
__aicore__ inline void Init(GM_ADDR kOffGm, GM_ADDR kInvGm, ...) {
    LocalTensor<int32_t> kOffB = kOffBaseBuf_.Get<int32_t>();
    LocalTensor<int32_t> kInvB = kInvBaseBuf_.Get<int32_t>();
    DataCopy(kOffB, kOffGm_, D_);
    DataCopy(kInvB, kInvGm_, D_);
    // 无需标量循环
}

// 方案 2：直接作为 kernel 标量参数传入（若表极小，如 D=64 且只需几个值）
// 但通常 offset 表较大，建议用 GM buffer
```

### 最小可行步骤

若不想大改 Host-Driver 接口，可先预计算那些**计算略复杂但尺寸小**的表：

1. `strideOffBuf_`（当前 16 个 `uint32_t`，完全线性）
2. `kInvBaseBuf_` / `vInvBaseBuf_`（D 个 `int32_t`）
3. `kOffBaseBuf_` / `vOffBaseBuf_`（D 个 `int32_t`）

### 组合收益

- 单独使用 X4：减少 Init 阶段标量指令，略微降低头开销；
- 与 X3 组合：释放的 UB 可用于增大 TB。

## 预期收益

- 消除 `Init` 中 D-loop 级别的标量计算（约 D 次 `SetValue` + 位运算）；
- 头开销降低 2-5 us（取决于表的数量和 D 的大小）；
- 释放的 UB 可支持 X3 的 TB 提升，带来额外 10%-20% 收益。

## 常见陷阱

⚠️ **TilingData 过大**：若 offset 表很大（如 D=4096），不宜放入 TilingData，
应使用 GM buffer 并通过 `DataCopy` 加载。

⚠️ **GM 搬运增加**：Host 预计算后，device 端从"计算"变成"搬运"，若表很大可能
引入 MTE2 压力。需权衡计算与搬运代价。

⚠️ **动态参数变化**：若 offset 表依赖运行时才能知道的参数（如 `seqLen`），
则不能 Host 预计算。本策略仅适用于纯编译期/Host 已知参数。

⚠️ **精度无关性**：必须确保 Host 与 device 计算结果完全一致（尤其是移位、
取整、浮点常量），否则会导致 bit-exact 不一致。

## 代码搜索关键词

```bash
# 检测 Init 中的标量循环填表
grep -n "SetValue.*i\|for.*i.*SetValue\|InitBuffer.*Off\|InitBuffer.*Inv" \
    op_kernel/*.cpp

# 检测完全由超参数决定的表
grep -n "mseBits_\|vqb_\|D_\|slotLen_" op_kernel/*.cpp | grep -E "SetValue|for"

# 检测 TilingData 结构
grep -n "struct.*Tiling\|TilingData\|BEGIN_TILING_DATA_DEF" \
    op_host/*.h op_host/*.cpp
```

## 来源

发现于 best_decode 深度性能分析：
`Init` 阶段存在多个 `for (int64_t i = 0; i < D_; i++)` 循环填充
`kOffBaseBuf_`、`kInvBaseBuf_`、`vOffBaseBuf_`、`vInvBaseBuf_`、`strideOffBuf_`。
这些表完全由 `D_`、`mseBits_`、`vqb_` 决定，本可在 Host 预计算。
