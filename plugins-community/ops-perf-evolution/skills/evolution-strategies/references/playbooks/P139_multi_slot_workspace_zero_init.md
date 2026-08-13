# P139 Playbook: 多槽流水 workspace 全槽零初始化

> 本 Playbook 为**强制流程**。
>
> P139 的核心：多槽（预发射）流水里，"紧凑写 + 整块读"的 workspace 槽必须**所有槽**都零初始化 padded 行，否则残留 NaN/inf → 偶发崩溃/精度错。

## Step 1: 定位 workspace 槽与写/读范围

```bash
grep -n "kPreloadNum\|ZeroWorkspace\|memset\|validCount\|laneBegin\|padded\|Init" \
    shared/original/op_kernel/*.h shared/original/op_kernel/*.cpp > /tmp/p139_loc.txt
```

**交付物**（写入 implementation_note 的 "Playbook Step 1"）：
- 槽数（kPreloadNum / BUFFER_NUM）；每个槽谁写（AIV gather）、谁读（AIC GEMM / softmax）。
- 每槽写入范围 `[0, validCount)` 与读取范围（整 tile 宽）之差——padded 区域清单。

## Step 2: 核对零初始化覆盖

对每个槽、每个被整块读的缓冲（K/V、score、prob…）逐一打勾：
- 该槽的 padded 行是否在初始化中覆盖？
- 只填 slot0？→ 缺陷（后续槽残留未初始化）。

**交付物**：覆盖矩阵（槽 × 缓冲 × 是否初始化）。

## Step 3: 改为全槽零初始化

```cpp
// 修复前（只填 slot0，slot1+ 残留）→ 修复后：遍历全部槽
for (uint32_t slot = 0; slot < kPreloadNum; ++slot) {
    ZeroRows(kvLocal[slot], laneBegin, laneEnd);
}
// 对 AIC 直接读的 prob/score padded 行（[activeG, mPad)）一并清零
```
- 零初始化开销 <1%，安全兜底；不要为了省一次 memset 留隐患。
- 与 P40（双缓冲结构）叠加时，覆盖所有槽的初始化点。

**交付物**：改动 diff + 覆盖矩阵回填。

## Step 4: 验证

- 连续 ≥3 轮全 case 20/20，无 NaN 抖动、无崩溃。
- 偶发崩溃本地不复现时：检查"写范围 < 读范围"的所有槽，而非只查数据越界。

**退出标准**：所有整块读的 padded 区域有确定值；多轮稳定。
