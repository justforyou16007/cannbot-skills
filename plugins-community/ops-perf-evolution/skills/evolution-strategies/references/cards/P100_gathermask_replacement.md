---
id: P100
bottlenecks: [scalar_compute, scalar_loading, compute_bound]
op_families: [elementwise, special, index_scatter, attention]
complexity: L1
conflicts_with: []
synergizes_with: [P97, P95, P101]
requires: []
has_preconditions: true
has_playbook: true
---

# P100: GatherMask 替代逐元素 SetValue/GetValue (GatherMask Replacement)

## 核心思想
当需要从连续 tensor 中按固定间隔取元素重组数据时（如 RoPE 奇偶分拆、解交织、矩阵转置等），优化前用 `for` 循环逐元素 `src.GetValue(i)` → `dst.SetValue(j, val)` 操作，每条调用仅处理 1 个元素（O(n) 次 Scalar ↔ Vector 数据通路切换）。优化后用 AscendC `GatherMask` API，在 Vector 单元上一次完成间隔采集，O(1) 条向量指令替代 O(n) 次标量操作，且零拷贝（直接从源 tensor 的偏移视图上操作）。

## 代码骨架

```cpp
// === 改造前（基线）：逐元素 SetValue/GetValue 解交织 ===
// 从 kvF32[rms_size_:] 中分离偶数位和奇数位
int half = rope_size_ / 2;
LocalTensor<float> kF32 = ropeBuf_.Get<float>();
for (int i = 0; i < half; ++i) {
    kF32.SetValue(i, kvF32.GetValue(rms_size_ + 2 * i));            // 偶数位
}
for (int i = 0; i < half; ++i) {
    kF32.SetValue(half + i, kvF32.GetValue(rms_size_ + 2 * i + 1)); // 奇数位
}
// 2 × half 次 SetValue + GetValue Scalar 调用，每次仅处理 1 个元素

// === 改造后：GatherMask 解交织（2 条指令完成） ===
uint64_t rsvdCnt = 0;
LocalTensor<float> ropeSrc = kvF32[rms_size_];  // 零拷贝偏移视图

// deinterleave=1: 取偶数索引位
GatherMask(kF32, ropeSrc,
    static_cast<uint8_t>(1), true,
    static_cast<uint32_t>(rope_size_), {1, 1, 8, 0}, rsvdCnt);

// deinterleave=2: 隔 1 取 1，取奇数索引位 → 写入 kF32[half]
GatherMask(kF32[half], ropeSrc,
    static_cast<uint8_t>(2), true,
    static_cast<uint32_t>(rope_size_), {1, 1, 8, 0}, rsvdCnt);
PipeBarrier<PIPE_V>();
```

## GatherMask 参数说明

```cpp
GatherMask(dst, src, deinterleave, isReverse, count, repeatMaskParams, rsvdCnt);
// dst: 输出 tensor（可用 dst[half] 切片拆分两段）
// src: 源 tensor（可用零拷贝偏移视图 src[offset]）
// deinterleave: uint8_t 间隔步长。1=连续取（偶数位）, 2=隔1取1（奇数位）, N=隔N-1取1
// isReverse: bool，默认传 true
// count: uint32_t 要收集的元素总数（非 half）
// repeatMaskParams: 常用默认值 {1, 1, 8, 0}（单 repeat，dst/src block stride=1/8）
// rsvdCnt: uint64_t& 保留参数，必须声明并初始化为 0
```

## 关键修改点

1. **零拷贝偏移**：`LocalTensor<T> ropeSrc = src[offset]` 创建视图，GatherMask 直接在原 tensor 上操作（配合 P97）
2. **PipeBarrier**：GatherMask 后加 `PipeBarrier<PIPE_V>()`，确保 Vector 写入完成后再被后续指令读取
3. **dst 拆分**：利用 `dst[half]` 切片将两段 GatherMask 结果写入同一 tensor 的不同区域
4. **预期收益**：元素数 ≥ 128 时 5-20x 加速；< 64 时收益递减但通常仍优于标量

## 常见陷阱

⚠️ GatherMask 后必须加 `PipeBarrier<PIPE_V>()`，否则 dst 数据可能被后续指令读为旧值
⚠️ `deinterleave` 是 uint8_t 类型，传入 int 需显式 cast：`static_cast<uint8_t>(1)`
⚠️ `count` 传元素总数而非 half；两段 GatherMask 的 count 通常相同（各取一半元素）
⚠️ `rsvdCnt` 必须声明为 `uint64_t` 并初始化为 0
⚠️ GatherMask 从 src 当前位置开始取，需确保 `ropeSrc = src[offset]` 偏移正确

## 代码搜索关键词

```bash
grep -nE "GatherMask|deinterleave|rsvdCnt|GatherMaskParams" op_kernel/*.cpp
```

## 来源

- tmp.md 优化点2：GatherMask 替代 SetValue/GetValue 逐元素解交织（RoPE 场景），2 条向量指令替代 2×half 次标量操作
