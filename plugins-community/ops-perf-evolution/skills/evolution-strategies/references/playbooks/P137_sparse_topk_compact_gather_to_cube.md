# P137 Playbook: 稀疏 top-k Attention 的 workspace 紧凑化走 Cube

> 本 Playbook 为**强制流程**。采纳 P137 的子 agent 必须逐步执行，每步验证后才能进入下一步。
>
> P137 的核心：token 级稀疏（top-k）不因"不规则索引"回退 Vector，而是把每个 (head, qSeq) 的 topK K/V 经 AIV 标量 gather 搬成 workspace 里的紧凑 tile（有效行前置、尾部零填充），再喂 AIC mmad。稀疏只加一层搬运契约，不改变"用 Cube"的结论。

## Step 1: 定位当前稀疏→计算路径

```bash
grep -n "sparseIndices\|topK\|GetValue\|DataCopyPad\|Gather\|validCount" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h > /tmp/p137_loc.txt
grep -n "MIX_AIC\|mmad\|Cube\|Matmul\|MmQK\|KERNEL_TYPE" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h >> /tmp/p137_loc.txt
```

**交付物**（写入 implementation_note 的 "Playbook Step 1"）：
- 当前 gather 方式：逐元素标量读 / DataCopyPad 批量 / 是否已有 workspace 紧凑 tile。
- 当前 QK^T / PV 计算落在哪个单元（Vector 模拟还是 AIC mmad）。

## Step 2: 判定是否具备紧凑化落点

逐项确认（任一不满足则 P137 不适用并回退）：
- 有稀疏索引结构（topK / sparseIndices）。
- 具备或可建 workspace：`compact Q[M_tile×Dk]` + `K/V[preload×N_tile×Dk]` + score/prob。
- Cube GEMM（QK^T / PV mmad）可消费紧凑 tile。

**交付物**：落点清单（每个 workspace 槽的尺寸、谁写、谁读、写读范围差）。

## Step 3: 实现紧凑化 + Cube 路径

1. AIV：按 topK 索引 gather K/V → 紧凑行 `[0, validCount)`，尾部零填充（配合 P139 全槽零填充）。
2. AIC：`CubeQK`（M_tile×Dk × Dk×N_tile）+ `CubePV`，padded 行恒 0。
3. AIV↔AIC 同步：gather 完 Set signal → AIC GEMM；AIC 完 Set → AIV 复用槽（参考 P14 双槽流水 + CrossCore flag）。
4. **顺序纪律**：若 gather 仍是逐 lane 标量寻址且主导耗时，先做 gather 向量化（P129/索引排序/批量 MTE2），再做 M 维融合——M-merge 在 gather 未向量化时已验证反升 11-15%。

**交付物**：改动 diff + 每槽写/读范围核对表 + 同步 flag 配对表。

## Step 4: 验证

- 20/20 精度；`cube_utilization` 与 `aic_mac_ratio`（msprof）对比改造前后。
- 全 case 无回退；padded 行零填充保留（防 NaN）。

**退出标准**：`cube_utilization` 显著提升、墙钟下降、无 NaN/崩溃。
