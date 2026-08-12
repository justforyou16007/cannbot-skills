# P133 Playbook: 块内预计算 AIC/AIV CV 流水（6-flag 定向握手）

> **强制流程**。本 Playbook 指导子 agent 把线性递归算子 Stage1 块内预计算从每级 SyncAll 改写为 AIC/AIV 定向握手流水。前置：P132 三阶段已落地。必须按 Step 1-5 逐步执行，严禁跳步。

## Step 1: 定位关键结构

把 grep 结果写入 `/tmp/p133_locations.txt`：

```bash
# 1.1 全局同步点（消除对象）
grep -nE "SyncAll|sync_all" op_kernel/*.{cpp,h} >> /tmp/p133_locations.txt
# 1.2 块内预计算各级（matmul 与向量交错处）
grep -nE "Mmad|mmad|AICProcess|gemm|kkt|qkt|cumsum|exp\(|Inverse|solve|cumprod|g_cum|k_cumdecay|v_inner|q_prime" \
    op_kernel/*.{cpp,h} >> /tmp/p133_locations.txt
# 1.3 现有跨核机制
grep -nE "CrossCoreSetFlag|CrossCoreWaitFlag|SetFlag|WaitFlag|PIPE_" \
    op_kernel/*.{cpp,h} >> /tmp/p133_locations.txt
# 1.4 核角色（mix kernel 的 AIC/AIV 分支）
grep -nE "__DAV_C220_CUBE__|__DAV_C220_VEC__|GetSubBlockNum|GetBlockIdx|AIC_AIV" \
    op_kernel/*.{cpp,h} >> /tmp/p133_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，明确块内预计算各级的 C/V 归属、SyncAll 处数、已有 flag 机制。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 同步方式 | SyncAll 全局屏障 | 6 flag 定向握手（0x9/0x8/0xA/0x7/0x6/0x5） | op_kernel/*.cpp:L? |
| cube 任务 | （如并入串行） | KKT/QKT、off-diag 求逆 MM、k_cumdecay/v_inner 上 cube | op_kernel/*.cpp:L? |
| vector 任务 | — | prep/cumsum+exp/对角前代/gBK/VBeta/q_prime 上 vector | op_kernel/*.cpp:L? |
| flag 事件号 | — | stage1 独立号段，避开 stage2/3 | op_kernel/*.h:L? |
| GLA 分支 | — | 去求逆节点（无 0x7 握手） | op_kernel/*.cpp:L? |

## Step 3: 代码改造

形态识别：从每级 SyncAll（α 形态）改为 C/V 分工 + 定向握手（β 形态）。参考 GDR stage1.h 的 `ParaChunkAIC`/`ParaChunkAIV` 分工与 6-flag 拓扑（见 P133 卡代码骨架）。GLA 去掉求逆节点（C2a），链为 `V0a→C1(KKT)→V1a(衰减累积)→V2→C2b`。

## Step 4: 约束复核

| 约束 | 检查 |
|---|---|
| flag 事件号 | 各 stage 独立号段（stage1: 0x9/0x8/0xA/0x7/0x6/0x5；勿与 stage2/3 撞号） |
| SetFlag PIPE | 每个 SetFlag 带正确 PIPE（PIPE_FIX/PIPE_MTE3），保证数据已落 GM |
| 依赖链 | AIC 只在真需要时等 AIV（反之亦然），无多余等待 |
| GLA | 无求逆 → 0x7 握手删除，V1a 改为衰减累积 |

## Step 5: 编码并自检

```bash
# 5.1 无 SyncAll 残留
grep -nE "SyncAll" op_kernel/*.{cpp,h} || echo "NO-SYNCALL"
# 5.2 flag 拓扑存在
grep -nE "CrossCoreSetFlag.*0x[0-9A-F]|CrossCoreWaitFlag.*0x[0-9A-F]" op_kernel/*.{cpp,h}
# 5.3 C/V 分工
grep -nE "__DAV_C220_CUBE__|__DAV_C220_VEC__" op_kernel/*.{cpp,h}
# 5.4 编译 + 精度 + 性能
bash evaluate_ascendc.sh && bash ops/ops-profiling/scripts/msprof_profile_run.sh --quick --output-dir={output_dir} --device={npu}
```

**grep 全过 + 编译/精度 PASS 才完成**；不满足回 Step 2。
