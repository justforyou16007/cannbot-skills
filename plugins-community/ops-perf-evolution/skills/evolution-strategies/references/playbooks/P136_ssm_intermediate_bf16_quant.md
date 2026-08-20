# P136 Playbook: SSM 中间量 bf16 量化 + 状态 UB 常驻（head grouping）

> **强制流程**。本 Playbook 指导子 agent 把线性递归算子的中间量按 bf16 量化、状态 (Dv,Dk) UB 常驻、KV 做 head grouping。前置：P132 三阶段已落地。必须按 Step 1-5 逐步执行，严禁跳步。

## Step 1: 定位关键结构

把 grep 结果写入 `/tmp/p136_locations.txt`：

```bash
# 1.1 中间量 dtype（消除对象）
grep -nE "float |half |bfloat16|float16|v_beta|q_prime|attn_1|k_cumdecay|kg|v_new|attn_inter|curState|state" \
    op_kernel/*.{cpp,h} model.py >> /tmp/p136_locations.txt
# 1.2 状态搬移（每 chunk GM 往返）
grep -nE "CopyIn.*state|CopyOut.*state|state.*Gm|Gm.*state" op_kernel/*.{cpp,h} >> /tmp/p136_locations.txt
# 1.3 head / KV 布局
grep -nE "repeat_interleave|Nv|Nk|num_heads|head.*group|kv_head" op_kernel/*.{cpp,h} model.py >> /tmp/p136_locations.txt
# 1.4 尾块处理
grep -nE "lastChunkSize|seqLen.*%|chunkSize_|pad" op_kernel/*.{cpp,h} >> /tmp/p136_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，明确中间量 dtype 分布、状态搬移频次、KV 布局、尾块。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 中间量 dtype | （如全 fp32） | 块间 bf16；求逆/衰减核心保留 fp32 | op_kernel/*.cpp:L? |
| 状态驻留 | 每 chunk 搬 GM | curState UB 常驻跨 chunk | op_kernel/*.cpp:L? |
| KV 布局 | （如冗余） | Nv//Nk head grouping | op_kernel/*.cpp:L? |
| 尾块 | 无特化 | lastChunkSize 变长 | op_kernel/*.cpp:L? |

## Step 3: 代码改造

形态识别：从中间量全 fp32 / 状态每 chunk 搬 GM（α 形态）改为 bf16 + 常驻（β 形态）。参考 GDR model.py 的 bf16 边界划分与 `curState` 常驻（见 P136 卡代码骨架）。bf16 cast 用 `CAST_RINT`。

## Step 4: 约束复核

| 约束 | 检查 |
|---|---|
| UB 预算 | 各中间量 bf16 双缓冲后 + curState 不超 UB；超则调 chunk 或分阶段 |
| 精度 | 求逆/衰减核心保留 fp32；bf16 只用于块间传递 |
| 状态正确性 | 常驻 curState 在 batch/序列边界正确初始化/切换 |
| head grouping | Nv//Nk 分组索引 host 侧算对，无越界 |

## Step 5: 编码并自检

```bash
# 5.1 bf16 中间量
grep -nE "bfloat16_t|bhalf_t|DownCast|CAST_RINT" op_kernel/*.{cpp,h}
# 5.2 状态常驻（无每 chunk 搬 GM）
grep -nE "CopyIn.*state|CopyOut.*state" op_kernel/*.{cpp,h} || echo "STATE-RESIDENT"
# 5.3 编译 + 精度 + 性能（同一 session 的候选共用评估锁）
flock -w 1800 "{eval_lock_path}" env ASCEND_RT_VISIBLE_DEVICES="{npu_devices}" \
  bash evaluate_ascendc.sh && \
flock -w 1800 "{eval_lock_path}" \
  bash ops/ops-profiling/scripts/msprof_profile_run.sh --quick \
    --output-dir={output_dir} --devices="{npu_devices}"
```

**grep 全过 + 编译/精度 PASS 才完成**；不满足回 Step 2。
