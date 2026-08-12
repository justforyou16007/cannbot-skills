# P135 Playbook: delta-rule 矩阵求逆分治（对角向量前代 + 非对角 cube MM）

> **强制流程**。本 Playbook 指导子 agent 把 delta-rule 算子的 C×C 矩阵求逆从整块串行改为对角前代 + 非对角 cube MM 分治。前置：P132 已落地且确认是 delta rule（有 corr/β 修正）。必须按 Step 1-5 逐步执行，严禁跳步。

## Step 1: 定位关键结构

把 grep 结果写入 `/tmp/p135_locations.txt`：

```bash
# 1.1 求逆位置（消除对象）
grep -nE "solve_triangular|Inverse|inverse|triangular|tril|前代|substitution|attn_1|I-A|I\\+.*kkt" \
    op_kernel/*.{cpp,h} model.py >> /tmp/p135_locations.txt
# 1.2 现有分块/MM 证据
grep -nE "INVERSE_SHAPE|M10|AttnInverseMMCompute|InverseCompute|Mmad|BlockMmad" \
    op_kernel/*.{cpp,h} >> /tmp/p135_locations.txt
# 1.3 确认是 delta rule（非 GLA）
grep -nE "beta|corr|修正|delta|gated" op_kernel/*.{cpp,h} model.py >> /tmp/p135_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，明确求逆来源（delta rule 确认）、当前实现方式（整块 vs 已有分块）、flag 时机。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 求逆方式 | 整块 solve_triangular / 串行回代 | 对角子块向量前代 + 非对角 cube MM | op_kernel/*.cpp:L? |
| 分块粒度 | 无 | INVERSE_SHAPE（对齐 mmad 形状） | op_kernel/*.h:L? |
| 非对角 | 向量/串行 | X10 = X11·M10·X00 单次 mmad | op_kernel/*.cpp:L? |
| 握手 | 无 | 前代完 SetFlag 0x7 → off-diag | op_kernel/*.cpp:L? |

## Step 3: 代码改造

形态识别：从整块求逆（α 形态）改为分治（β 形态）。参考 GDR 的 `InverseCompute`（AIV 对角前代）+ `AttnInverseMMCompute`（AIC off-diag `X10=X11·M10·X00`）分工（见 P135 卡代码骨架）。

## Step 4: 约束复核

| 约束 | 检查 |
|---|---|
| delta rule 确认 | 有 corr/β 修正 → 本卡；GLA（cumprod）无求逆 → 跳过 P135 |
| 分块对齐 | INVERSE_SHAPE 与 mmad M/N/K 对齐 16 |
| flag 时机 | 0x7 在对角前代全部完成后，AIC 才可读 X11 |
| 精度 | 前代保留 fp32 中间计算，输出 bf16（见 P136） |

## Step 5: 编码并自检

```bash
# 5.1 分治结构
grep -nE "InverseCompute|AttnInverseMMCompute|INVERSE_SHAPE" op_kernel/*.{cpp,h}
# 5.2 无整块串行求逆残留
grep -nE "solve_triangular" op_kernel/*.{cpp,h} || echo "NO-TRIANGULAR"
# 5.3 编译 + 精度 + 性能
bash evaluate_ascendc.sh && bash ops/ops-profiling/scripts/msprof_profile_run.sh --quick --output-dir={output_dir} --device={npu}
```

**grep 全过 + 编译/精度 PASS 才完成**；不满足回 Step 2。
