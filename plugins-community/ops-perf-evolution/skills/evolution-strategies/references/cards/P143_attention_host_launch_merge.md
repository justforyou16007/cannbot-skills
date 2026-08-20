---
id: P143
bottlenecks: [scalar_loading, no_overlap]
op_families: [attention, flash_attention, matmul]
complexity: L2
conflicts_with: []
synergizes_with: [P35, P54, P131, P142]
requires: []
has_preconditions: true
has_playbook: true
---

# P143: Attention host 端 launch 合并 + per-shape tiling 缓存

## 核心思想
910B2C 上小 shape / fp32 路径的 attention 容易被 host 侧固定开销主导：每次调用重复 `at::empty`、workspace 清零、tiling 结构体 H2D 拷贝、多个 split kernel 独立 launch。把同一次前向中可合并的 host 步骤合并，并把 tiling 结果按 shape 缓存，能把这些开销从热路径移除。

## 代码骨架

```cpp
// === 改造前（多次 launch + 每次重新 tiling）===
// host:
for (h = 0; h < heads; ++h) {
    launch_pack_kernel(Wq_h, Wk_h, Wv_h);   // 多个 launch
}
launch_attention_kernel(...);
for (h = 0; h < heads; ++h) {
    launch_out_proj_kernel(Wo_h);
}
// 每次调用重新构造 tiling 并 H2D 拷贝

// === 改造后（合并 launch + tiling cache）===
// host:
launch_fused_pack_attention_proj(...);        // 单次 launch 或最小 kernel 集合
TilingCacheKey key = {B, S, H, D, dtype};
if (!tiling_cache.count(key)) {
    tiling_cache[key] = ComputeTiling(key);
}
auto& tiling = tiling_cache[key];
// workspace 复用，at::empty 替代 zeros
```

## 关键修改点

1. **减少独立 launch**：若 fp32 hi/lo 路径把 QKV/out 投影拆成多个小 kernel，评估能否合并为单次 kernel 调用或核内串行多 tile。
2. **per-shape tiling 缓存**：以 `(B, S, H, D, dtype)` 为 key 缓存 host tiling 结果和已下发的 workspace 指针；同 shape 再次调用时跳过 H2D 传输与重新 tiling。
3. **workspace 复用**：用 `at::empty` 替代 `at::zeros`（kernel 会全量覆盖时），并把 workspace 生命周期与缓存绑定。
4. **host 打包前置**：把 per-head 的 `narrow().copy_()` 合并为单次 `cat + reshape + strided copy`（与 X38 host 打包配合）。
5. **边界**：合并 launch 不能引入跨 kernel 同步错误；tiling cache 需处理 shape 变化时的失效与内存增长。

## 常见陷阱

- ❌ 为合并 launch 引入隐式同步或 workspace 读写竞争。
- ❌ tiling cache key 遗漏 dtype/stride，导致命中错误 shape。
- ❌ cache 无限增长导致 host 内存泄漏；需设容量上限或 LRU。
- ❌ 小 shape 下缓存查找开销超过 tiling 计算开销；需测阈值后启用。

## 代码搜索关键词

```bash
grep -nE "launch\|at::zeros\|at::empty\|tiling\|H2D\|copy_|narrow\|constant_pad_nd" op_host/*.cpp op_host/*.h
```
