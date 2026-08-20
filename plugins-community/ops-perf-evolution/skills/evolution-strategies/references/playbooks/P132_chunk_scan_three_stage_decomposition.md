# P132 Playbook: 线性递归 chunk-scan 三阶段分解（块内预计算 / 块间状态扫描 / 输出投影）

> **强制流程**。本 Playbook 指导子 agent 把无 softmax 的线性递归/SSM 算子从逐 token 串行递推改写为三阶段 chunk-scan。前置：算子已确认归 SSM/Scan 族（无 QK^T、无 softmax）。必须按 Step 1-5 逐步执行，严禁跳步。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p132_locations.txt`：

```bash
# 1.1 串行递推循环（消除对象）
grep -nE "for .*seq|for .*len|while.*seq|state.*state|S_t|h_t|prev" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p132_locations.txt
# 1.2 状态更新点（改写目标）
grep -nE "state|State|S\b|h\b|cumsum|cumprod|exp\(|outer|k.*v.*T|gemm|Mmad" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p132_locations.txt
# 1.3 现有 chunk/并行证据
grep -nE "chunk|CHUNK|block.*seq|n_chunks|SyncAll|blockDim" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p132_locations.txt
# 1.4 数学结构（确认是 delta rule 还是门控衰减）
grep -nE "beta|g\b|gate|decay|corr|kkt|qkt" \
    shared/original/op_kernel/*.{cpp,h} model.py >> /tmp/p132_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，明确：当前递推循环位置、状态更新公式（delta rule `S+=β(k⊗v−corr)` vs 门控衰减 `S=α⊙S+k⊗v`）、是否已有 chunk。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 序列循环 | 逐 token 串行 | 三阶段 chunk-scan（C=64） | op_kernel/*.cpp:L? |
| 块内预计算 | 无（token 级） | Stage1：KKT/QKT/cumsum/求逆|cumprod/kcumdecay/v_inner/q_prime | op_kernel/*.cpp:L? |
| 块间状态 | 逐 token 更新 | Stage2：每 chunk 一次 state 更新（(Dv,Dk) 小矩阵） | op_kernel/*.cpp:L? |
| 输出 | 逐 token 累加 | Stage3：out = attn_inter + masked_qkt@v_new | op_kernel/*.cpp:L? |
| chunk 边界 | 无 | C=64 + 尾块 lastChunkSize 特化 | op_kernel/*.cpp:L? |

## Step 3: 代码改造

形态识别：从逐 token 串行（α 形态）改为三阶段（β 形态）：

```cpp
// 改造前（α 形态）：逐 token 串行
for (int32_t t = 0; t < seqLen; t++) {
    // state = state * exp(g[t]) + outer(v[t], k[t]);  ← 串行，无法并行
    ...
}

// 改造后（β 形态）：三阶段
// Stage1: 块内预计算（Nv×n_chunks 按核并行）——KKT/QKT/衰减/求逆或cumprod/kcumdecay/v_inner/q_prime
// Stage2: 块间扫描（串行，O(seq/C)）—— state = state*decay + v_new^T @ kg
// Stage3: 输出投影 —— out = attn_inter + masked_qkt @ v_new
```

参考结构：`chunk_gated_delta_rule` 参考实现的三阶段（stage1.h/stage2.h/stage3.h），块内预计算含 KKT `k@k^T·β`、`attn_1=(I+KKT·attn)^{-1}`（GDR）或 cumprod 衰减（GLA）、`k_cumdecay=attn_1@gBK`、`v_inner=attn_1@vBeta`、`q_prime=q·scale·exp(g_cum)`。

## Step 4: 约束复核

| 约束 | 检查 |
|---|---|
| UB 预算 | Stage1 各中间量（C×Dk / C×Dv / C×C）双缓冲后不超 UB（见 P136 的 bf16 量化） |
| 对齐 | C×Dk / C×Dv matmul 的 M/N/K 按 16/32B 对齐；尾块 pad 处理 |
| 串行链 | 块间只有 (Dv,Dk) 状态递推，块内无跨 chunk 依赖 |
| 数值 | 衰减 exp / 求逆的 fp32 中间计算保留（见 P135），块间 bf16 |

## Step 5: 编码并自检

```bash
# 5.1 三阶段结构存在
grep -nE "Stage1|Stage2|Stage3|k_cumdecay|v_inner|state.*decay" op_kernel/*.{cpp,h}
# 5.2 无逐 token 串行残留
grep -nE "for .*t.*seqLen|for .*i.*<.*seq" op_kernel/*.{cpp,h} || echo "NO-SERIAL-LOOP"
# 5.3 尾块特化
grep -nE "lastChunkSize|seqLen.*chunkSize.*%" op_kernel/*.{cpp,h}
# 5.4 编译 + 精度 + 性能（同一 session 的候选共用评估锁）
flock -w 1800 "{eval_lock_path}" env ASCEND_RT_VISIBLE_DEVICES="{npu_devices}" \
  bash evaluate_ascendc.sh && \
flock -w 1800 "{eval_lock_path}" \
  bash ops/ops-profiling/scripts/msprof_profile_run.sh --quick \
    --output-dir={output_dir} --devices="{npu_devices}"
```

**5 条 grep 全部通过 + 编译/精度 PASS 才算完成**；不满足回 Step 2 改计划表，严禁跳步。
