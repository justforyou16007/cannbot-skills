# P140 Playbook: MLA（kvHeads=1）的 M-tile 放大到 qHeads

> 本 Playbook 为**强制流程**。
>
> P140 的核心：单个 Cube GEMM 的 M 维上限 = 共享同一份 K/V 的查询行数 = qHeads/kvHeads。GQA 只有 groupSize 行（小），MLA（kvHeads=1）可到 qHeads（128~256）——把写死的小 M-tile 放大，让 QK^T/PV 一次 mmad 吃满 Cube。

## Step 1: 定位共享 K/V 关系与当前 M-tile

```bash
grep -n "kvHeads\|qHeads.*kvHeads\|groupSize\|kCubeMaxM\|mTile\|groupTile" \
    shared/original/op_host/*.cpp shared/original/op_kernel/*.h > /tmp/p140_loc.txt
```

**交付物**（写入 implementation_note 的 "Playbook Step 1"）：
- 各 case 的 `qHeads/kvHeads`（GQA=小 groupSize / MLA=qHeads）。
- 当前 M-tile 常量（kCubeMaxM）与 device 侧 Q/score/prob 缓冲占用。

## Step 2: 判定 MLA 场景并算放大上限

- 仅当 `kvHeads==1`（或共享行数大）时放大有价值；GQA 硬拉 M 无收益。
- 上限 = `min(qHeads, 受 UB/L0 约束的最大 M)`；UB 看 Q compact + score/prob，L0 看 l0A_/l0C_（配合 P138）。

**交付物**：各 case 的共享行数表 + 放大后缓冲占用表。

## Step 3: 放大 + 自适应

```cpp
const int64_t sharedRows = qHeads / kvHeads;
int64_t mTile = sharedRows < kCubeMaxM ? sharedRows : kCubeMaxM;
while (!FitsUB(mTile) || !FitsL0(mTile)) mTile /= 2;   // 运行时 mTileCap
```
- 大 M 时注意 softmax/输出按 M 行向量化、跨行独立 max/sum（每行独立，但整块向量处理）。
- 与 P137 紧凑化、P138 L0 守卫叠加。

**交付物**：改动 diff + mTileCap 推导。

## Step 4: 验证

- MLA case（kvHeads=1、Dk512/576）墙钟对比放大前后；全 case 20/20、无回退。
- msprof 确认 `aic_mac_ratio` / `cube_utilization` 提升。

**退出标准**：MLA case 显著提速、GQA case 不劣化。
