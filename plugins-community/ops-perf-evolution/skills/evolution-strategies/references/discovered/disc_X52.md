---
id: X52
origin: discovered
discovered_round: 2
discovered_from: "21_GroupedQueryAttention evolution round_2/parallel_2 (x0_oe) + round_3/parallel_2 (x0_oe_x1)"
base_speedup: "torch.npu.Event full-50 geomean 0.929->0.968-0.970（asc-time -13.5% 同窗口 A/B）"
---

# Strategy X52: Host 数据通路消除（reshape-materialization 消除 + 条件 at::empty + 单 pass strided-gather transpose）

## 核心思路
融合 GQA/FA 管线 host 侧是 host-overhead-bound 的（小 shape ~50-70us torch ops + launch，kernel 仅 10-15us）。host 侧每 forward 有大量中间张量物化：
1. `permute().reshape()` 非连续视图强制先物化转置再 copy_——同一转置 GM 往返 2 次。
2. `at::zeros` 零填充在无 padding 区时是纯浪费 launch。
3. 每 forward 重建 shape 无关的常量（mask）缓冲。

三管齐下消除：单 pass strided-gather copy_ 替代 permute+reshape+copy_；条件 at::empty（Dm%64==0 或 hd==Dp 时安全）；shape-keyed mask cache。净效果 fp32 medium/large ~16->~11 launch/forward。

## 适用场景
- 融合管线 host 侧有 permute/reshape/transpose 中间物化、每调用 at::zeros、重复重建 mask
- host-overhead-bound（小 shape / launch 密集）算子
- 与 X39（workspace at::empty）、X35（MatmulImpl host 开销消除）互补

## 实现要点
1. 单 pass strided-gather `view({...}).copy_(permutedView)` 替代 permute+reshape+copy_，删除前置物化。
2. 条件 at::empty 守卫：`Dm%64==0` 或 `hd==Dp` 时安全（无 pad 列）；M-row 尾块死行论证（下游不读、row-isolated）。
3. shape-keyed mask cache：mask 是 (S,kvPad) 纯函数，跨调用复用。
4. dead workspace（如 `(void)ws3`）替换为缓存 dummy。
5. 收益是删除 kernel（不新增），理论上不回归设备时间；精度必须全量 50/50 PASS。

## 来源
21_GroupedQueryAttention 进化 round_2/3（x0_oe + x0_oe_x1，open_exploration 延续）：host launch ~16->~11，torch.npu.Event full-50 0.929->0.968-0.970，asc-time -13.5%。
