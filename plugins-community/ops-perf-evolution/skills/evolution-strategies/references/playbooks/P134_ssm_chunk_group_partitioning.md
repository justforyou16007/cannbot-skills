# P134 Playbook: chunk-group 任务划分（group 间串行 + group 内多核并行）

> **强制流程**。本 Playbook 指导子 agent 把线性递归算子的核分配从"全序列一次分核"改为 chunk-group 任务划分。前置：P132 三阶段已落地。必须按 Step 1-5 逐步执行，严禁跳步。

## Step 1: 定位关键结构

把 grep 结果写入 `/tmp/p134_locations.txt`：

```bash
# 1.1 现有核分配 / 任务划分
grep -nE "blockDim|GetBlockIdx|GetCoreNum|coreIdx|tid|taskId|former|tail|chunkNum|nvPerCore" \
    op_kernel/*.{cpp,h} op_host/*.cpp >> /tmp/p134_locations.txt
# 1.2 序列/状态循环（判定当前是否全序列一次分核）
grep -nE "for .*seq|for .*len|for .*chunk|for .*pos|state|State|SyncAll" \
    op_kernel/*.{cpp,h} >> /tmp/p134_locations.txt
# 1.3 batch / actual_seq_lengths
grep -nE "actualSeqLengths|seqLength|batch|B\b|initial_state" \
    op_kernel/*.{cpp,h} op_host/*.cpp >> /tmp/p134_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，明确当前任务划分方式、是否跨核状态依赖、batch 处理。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 任务划分 | 全序列一次分核（跨核状态依赖） | chunk-group（group 间串行 + group 内并行） | op_host/*.cpp:L? |
| maxGroupLength | 无 | P_NUM × aiCoreNum × C（GDR: 2×核×64） | op_kernel/*.h:L? |
| Stage1/3 分配 | — | totalChunk=nv×num_chunk 按 coreIdx 均分 | op_kernel/*.cpp:L? |
| Stage2 分配 | — | nvPerCore=ceil(Nv/core) 切核、核内串行 cId | op_kernel/*.cpp:L? |
| batch 循环 | — | 外层按 batch，内层 chunk-group | op_kernel/*.cpp:L? |

## Step 3: 代码改造

形态识别：从全序列一次分核（α 形态）改为 chunk-group（β 形态）。参考 GDR `Process()` 的 batch → chunk-group → RunStage1/2/3 + SyncAll 结构（见 P134 卡代码骨架）。三阶段任务粒度不同：Stage1/3 全并行均分，Stage2 核内串行。

## Step 4: 约束复核

| 约束 | 检查 |
|---|---|
| 串行依赖 | 只有 chunk-group 之间串行，group 内无跨 chunk 状态依赖 |
| 状态初始化 | 每组起点状态 = 上组末状态；batch 首组用 initial_state |
| 尾组 | 最后 group 不满 maxGroupLength 时按实际 length 特化 |
| 核数 | blockDim = aiCoreNum，Stage1/3 任务均分吃满 |

## Step 5: 编码并自检

```bash
# 5.1 chunk-group 结构
grep -nE "maxGroupLength|chunkGroup|chunk_group|RunStage" op_kernel/*.{cpp,h} op_host/*.cpp
# 5.2 Stage2 核内串行（无跨核状态）
grep -nE "nvPerCore|cId.*for|for.*cId" op_kernel/*.cpp
# 5.3 无全序列一次分核
grep -nE "seqLength.*/.*GetCoreNum|totalSeq.*GetBlockIdx" op_kernel/*.{cpp,h} || echo "OK"
# 5.4 编译 + 精度 + 性能（同一 session 的候选共用评估锁）
flock -w 1800 "{eval_lock_path}" env ASCEND_RT_VISIBLE_DEVICES="{npu_devices}" \
  bash evaluate_ascendc.sh && \
flock -w 1800 "{eval_lock_path}" \
  bash ops/ops-profiling/scripts/msprof_profile_run.sh --quick \
    --output-dir={output_dir} --devices="{npu_devices}"
```

**grep 全过 + 编译/精度 PASS 才完成**；不满足回 Step 2。
