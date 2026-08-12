# P138 Playbook: host tiling L0 容量查询 + 自适应降维

> 本 Playbook 为**强制流程**。
>
> P138 的核心：Cube tiling 的 tile 尺寸（groupTile / baseM 等）由 L0A/L0C 容量决定，而容量随芯片变化（910B2C vs 910c）。写死 → 更小 L0 芯片上 InitBuffer 溢出 → 随机 SIGSEGV。改为运行时查询 + 自适应降维。

## Step 1: 定位 tiling 常量与 L0 占用

```bash
grep -n "groupTile\|kCubeMaxM\|baseM\|baseN\|GetCoreMemSize\|L0\|InitBuffer" \
    shared/original/op_host/*.cpp shared/original/op_kernel/*.h > /tmp/p138_loc.txt
```

**交付物**（写入 implementation_note 的 "Playbook Step 1"）：
- 当前 tile 常量值；device 端 `l0C_ / l0A_` 的推导式（如 `groupTile×256×4B`）。
- 是否已查询 L0 容量（`GetCoreMemSize(L0_A/L0_C)`）。

## Step 2: 计算目标芯片 L0 占用并判定

对 `groupTile`（或 baseM）逐个验证：
- `l0C_ = groupTile × N_tile × 4B` ≤ 查询到的 L0_C？`l0A_` 同理。
- UB 侧（Q/score/prob 缓冲）一并验证。

**交付物**：L0A/L0C 占用表（当前值 / 目标值 / 是否满足）。

## Step 3: 加运行时查询 + 自适应降维

```cpp
uint64_t l0aSize = 0, l0cSize = 0;
platform->GetCoreMemSize(CoreMemType::L0_A, l0aSize);
platform->GetCoreMemSize(CoreMemType::L0_C, l0cSize);
while (l0cSize > 0 && groupTile * kTileN * sizeof(float) > l0cSize) groupTile /= 2;
if (groupTile <= 0) groupTile = 1;
```
- 降维只发生在容量不满足时；满足时保持原值（不牺牲性能）。
- device `InitBuffer` 布局必须与查询结果一致（同一资源源）。

**交付物**：改动 diff + 降维触发条件的单元测试/日志。

## Step 4: 验证

- 本地（开发芯片）全 case 无回归。
- 若能跨芯片复测（如 910c）：确认随机 SIGSEGV 消失、20/20 有性能数据。

**退出标准**：tile 尺寸与 L0 容量恒一致；无 InitBuffer 溢出。
