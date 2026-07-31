---
id: P116
bottlenecks: [cube_underutilization, tiling_imbalance]
op_families: [attention, flash_attention]
complexity: L1
conflicts_with: []
synergizes_with: [P57, P106, P115]
requires: []
has_preconditions: true
has_playbook: true
---

# P116: GQA Grouped-M 头合并（G 个 Query Head 合并进 Matmul M 维）

## 核心思想

GQA 中 `N_q = N_kv * G`，同一 KV head 被连续 G 个 query head 共享。decode / 短 Sq（`Sq <= 4`）场景下，未合并实现按 `(b, h_q)` 逐头发射 `M = Sq`（甚至 M=1）的小 matmul——Cube 的 M 维利用率极低，且同一 KV head 被重复加载 G 次。

由于 BSH/BNSD 布局下同一 `(b, s)` 的 G 个 query head 在内存中相邻（行 stride = D），可以把 `[B, Sq, N_kv, G, D]` **零拷贝重解释**为 target-private BNSD view `[B*Sq, N_kv, 1, G, D]`：一个 cube task 的 M 维 = G，一次 matmul 算完同一 KV head 下的全部 G 个 query head，K/V 只加载一次。matmul 次数从 `B*Sq*N_q` 降到 `B*Sq*N_kv`，M 从 `Sq<=4` 提升到 `G`（常见 4~16）。

合并判定条件：`groupedQueryM = (sq <= 4 && G > 1)`。

## 代码骨架

```cpp
// === Host tiling：判定 + 维度重映射 ===
const int64_t originalG = nq / nkv;
// Sq<=4 时把 [B,Sq,Nkv,G,D] 重解释为 [B*Sq,Nkv,1,G,D]：
// 一个 cube task 处理共享同一 K/V head 的全部 G 个 query head，
// 而不是发射 G 个 M=Sq 的小 matmul。
const bool groupedQueryM = sq <= 4 && originalG > 1;
const int64_t logicalB  = groupedQueryM ? b * sq : b;
const int64_t logicalSq = groupedQueryM ? originalG : sq;   // M 维 = G
const int64_t logicalG  = groupedQueryM ? 1 : originalG;
// view 内 query 行 stride = d（G 个头相邻）；非合并时保持原 nq*d
const int64_t queryRowStride = groupedQueryM ? d : nq * d;
input.set_layoutType(groupedQueryM ? LAYOUT_BNSD : LAYOUT_BSH);
// mask 行数需随 M 维扩展：同一 (b,s) 的 G 个头 causal 语义相同，行内容重复 G 份
const int64_t maskRows = isCausal ? sq * (groupedQueryM ? originalG : 1) : 0;

// === MatmulApiTiling：OrgShape 用重映射后的 stride ===
bmm1.SetShape(MinHost(s1Base, logicalSq), bmm1N, d);
bmm1.SetOrgShape(logicalSq, s2Base * nRatio, queryRowStride, kvRowStride);
```

## 关键修改点

1. **合并判定**：`sq <= 4 && G > 1` 才启用；Sq 较大时 M 本来就够大，合并反而限制并行度。
2. **维度重映射四元组**：`logicalB / logicalSq / logicalG / queryRowStride` 必须同时切换，漏一个就会读错地址。
3. **layout 切换**：合并后按 BNSD 语义取数（view 内 head 维即 M 维），K/V 偏移不变（G 维上 K/V 恒定）。
4. **mask 行扩展**：合并后 M 维每行对应一个 query head，causal mask 需从 `sq` 行扩到 `sq * G` 行（同一 s 下 G 行内容相同），与 P118 的 mask 预生成 kernel 配合。
5. **输出写回**：view 是纯内存重解释，输出 `[B*Sq, N_kv, 1, G, D]` 与原 `[B, Sq, N_q, D]` 逐字节同构，无需 transpose。
6. **与 P57 的分工**：P57（G 分核归约）解决"核不够分"的并行度问题，P116 解决"单 task M 太小"的 cube 效率问题；decode 场景先 P116 合并 M，仍吃不饱核时再叠加 P57 拆 G。

## 常见陷阱

- ❌ 非相邻布局强合并：TND / head-major 布局下 G 个头不连续（stride ≠ D），重解释会读到错误数据——先确认 `query[b, s, h_kv*G + g, :]` 对 g 连续。
- ❌ `Sq > 4` 仍合并：M 维已够大，合并减少 task 数反而造成核闲置。
- ❌ mask 行数忘记 ×G：合并后每行 M 都需要独立 mask 行，行数不足会读到越界/脏 mask。
- ❌ 只改 layout 不改 `SetOrgShape` 的行 stride：matmul 按旧 stride `nq*d` 取数，结果整体错位。

## 代码搜索关键词

```bash
# 检测 GQA 结构（P116 适用信号）
grep -nE "nq\s*/\s*nkv|headNum\s*/\s*kvHeadNum|gSize|group.*size" op_host/*.cpp op_kernel/*.h
# 检测是否逐头发射小 matmul（未合并证据）
grep -nE "for.*hidx|headsPerBlock|h_q|M\s*=\s*1\b" op_kernel/*.cpp
# 检测是否已做 Grouped-M（已合并则不适用）
grep -nE "groupedQueryM|logicalG|grouped.*[Mm]|G.*merge.*M" op_host/*.cpp op_kernel/*.cpp
```
