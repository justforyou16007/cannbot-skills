---
id: P122
bottlenecks: [cube_bound, mte1_stall, no_overlap]
op_families: [attention, flash_attention, matmul]
complexity: L1
conflicts_with: []
synergizes_with: [P18, P83, P120, P121]
requires: []
has_preconditions: true
has_playbook: true
---

# P122: KV 堆叠大块化 + L1 容量自适应 Q/KV 划分

## 核心思想

KV tile 是 32/128 小块且尺寸硬编码时，每次 QK^T 的 N 维、PV 的 K 维喂不饱 mmad，且每 32~128 个 KV 就要完成一轮"搬运→计算→握手"，Cube 启动开销和同步开销占比随 tile 变小线性放大。

把 KV 维**堆叠成大块**（基准 `KV_STACK = 512`，paged 场景把多个 128 物理块 stack 成一个逻辑块）：

1. **每次 CV 握手覆盖 512 个 KV**（而非 32/128）：跨核 flag 同步频率降到 1/4~1/16，P121 流水的握手开销被摊薄。
2. **mmad 的 N/K 维拉满**：QK^T 一次算 `[128q × 512kv]`，PV 一次算 `K=512`（L1 tile 256×2 拍），L0 装载次数和启动开销按比例下降。
3. **L1 容量自适应划分**（不硬编码 tile）：K/V 的 L1 块尺寸由 L1 剩余预算动态推出——

```cpp
kDyn = max(RoundUp(embed, 128), 256);                    // K 维：对齐 128，下限 256
budget = L1_MAX(512K) - embedV * KV_STACK(512) * dtype;  // 先扣掉 V 常驻区
qSize  = Q_TILE(128) * kDyn * dtype;                     // 再扣 Q 常驻区
nDyn = (budget - qSize) / kDyn / dtype / DOUBLE_BUFFER;  // KV N 维，含双缓冲
nDyn = min(nDyn, 128); nDyn = RoundDown(nDyn, 32);       // 上限 128，对齐 32
```

Q 常驻 + V 常驻 + K 双缓冲三者瓜分 L1，剩余容量全部折成 KV 的 N 维块——embed 越大块越小，全自动适配。

与 P18/P83 的关系：P18 是 L1 7-buffer 常驻划分的通用模式，P83 是 L1 常驻复用；本卡给出 FA 场景的具体预算公式与堆叠基准值（512/128/32 对齐）。

## 代码骨架

```cpp
// === 改造前：小 KV tile，硬编码 ===
constexpr int BN = 32;                          // 每轮 32 个 KV
for (int t = 0; t < kvLen; t += BN) { load K/V tile; compute; sync; }

// === 改造后：KV 堆叠 512 + L1 预算自适应 ===
constexpr uint32_t MAX_KV_STACK_LEN = 512;      // 逻辑 KV 块
constexpr uint32_t Q_TILE_CEIL = 128;
uint32_t kDynNum = RoundUp(embed, 128); kDynNum = max(kDynNum, 256);
uint32_t maxQKPL1 = L1_MAX_SIZE - embedV * MAX_KV_STACK_LEN * sizeof(T);
uint32_t maxQL1   = Q_TILE_CEIL * kDynNum * sizeof(T);
uint32_t nDynNum  = (maxQKPL1 - maxQL1) / kDynNum / sizeof(T) / DOUBLE_BUFFER;
nDynNum = min(RoundDown(nDynNum, 32), 128);
blockMmadQK.init(resource, nDynNum, kDynNum, MAX_KV_STACK_LEN);
// paged cache：blockStackNum = ceil(512 / blockSize)，跨物理块拼逻辑块
for (kvSIdx ...; kvSIdx < kvEnd; kvSIdx++) {    // 每轮 512 个 KV
    stackSeqTile = min(512, noSkipKvS - kvSIdx * 512);   // 尾块取余
    blockMmadQK(...{rowNum, stackSeqTile, embed}...);
}
```

## 关键修改点

1. **KV 逻辑块从 32/128 提到 512**（或按 UB/L1 预算重算），paged 场景按物理块堆叠 `blockStackNum = ceil(STACK/blockSize)`。
2. **尾块取余**：最后一个 stack tile 用 `noSkipKvS - idx*512`，配合 P123 的 causal 跳过边界。
3. **nDyn/kDyn 全部公式化**，禁硬编码；kDyn 下限 256 保证 K 维效率，nDyn 上限 128 防 L0/L1 溢出。
4. **对齐铁律**：kDyn 对齐 128，nDyn 对齐 32，DOUBLE_BUFFER=2 计入预算。
5. **PV 的 K 维联动**：`kPV = nDyn * kDyn / PV_L1_M`，QK 与 PV 的 L1 划分共享同一份 K/V 常驻区（PV init 传 `L1_QK_SIZE` 偏移复用）。

## 常见陷阱

- ❌ 堆叠后忘记改 workspace 槽尺寸（`Q_TILE × STACK`）→ workspace 溢出踩相邻核数据。
- ❌ nDyn 未对齐 32 / 未除双缓冲 → L1 分配越界，运行时随机踩内存。
- ❌ embed > 256 时仍按 256 切 kDyn → L1 预算公式溢出为负，nDyn 算成巨大值（无符号回绕）。
- ❌ causal 场景尾块仍按 512 满载 → 多读多算且 mask 越界；必须配合 P123 的 `noSkipKvS`。
- ❌ 堆叠块大于实际 kvSeqlen 时不取 min → GM 越界读。

## 代码搜索关键词

```bash
grep -nE "KV_STACK|STACK_LEN|kDyn|nDyn|L1_MAX|DOUBLE_BUFFER|blockStackNum|RoundUp|RoundDown" op_kernel/*.cpp op_kernel/*.h
```
