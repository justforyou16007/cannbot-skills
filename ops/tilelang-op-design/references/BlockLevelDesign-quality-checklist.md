# Block-level 设计质量自检清单

| # | 检查项 | 是否必须通过 |
|---|--------|-------------|
| 1 | **任务划分明确**：block 负责输出哪段区域，写出具体输出切片表达式（如 `Output[bz, by, bx*block_M:(bx+1)*block_M, :]`），不是"处理一段输出" | ✅ 必须 |
| 2 | **块内遍历明确**：用代码显式写清楚块内遍历循环（如 `for kv_loops in ...` 的 for 循环代码），写清哪个维在块内循环、遍历哪些输入数据，而不是文字描述 | ✅ 必须 |
| 3 | **布局整理说明**：需要先 pack / 轴合并 / 移位才利于分工的（如 GQA 的 Q/K/V pack、pooling 的 channel 移位），在 block-level 阶段就写明，而不是拖到 tile-level | ✅ 必须 |
| 4 | **C/V 分工明确**：每个子阶段标 `T.Scope("C")`（cube：T.gemm/T.mma/T.copy→L0）或 `T.Scope("V")`（vector：T.tile.*/T.reduce_sum），谁是 cube 谁是 vector 一目了然 | ✅ 必须 |
| 5 | **流水/同步策略明确**：标出是 expert 手动同步；stage 边界全核 barrier（`T.barrier_all` ↔ AscendC `SyncAll`）、阶段内 AIC↔AIV 用 `T.set_cross_flag`/`T.wait_cross_flag`；ring/prelaunch 参数说明 | ✅ 必须 |
| 6 | **workspace 完整**：`workspace_idx` 位置、每个 workspace 张量的 shape/dtype/谁写谁读，列出对照表 | ✅ 必须 |
| 7 | **细粒度计算有 TODO(tile-level)**：不是模糊的"计算 S = Q@K^T"，而是标注 TODO(tile-level)，明确该处细算在 tile-level 落地 | ✅ 必须 |
| 8 | **多 prim_func 模板**：不同 shape 是否需要不同 block 组织（如 rms_norm 按 N 选 merge_n/single_row），用 trace-time 选择并在文件里说明 | ⭕ 推荐 |
| 9 | **技术约束已确认**：非整除 seq、尾块 mask、动态边界（actual_seq_lengths / cu_seqlens）在 block-level 有处理说明 | ⭕ 推荐 |
| 10 | **GEMM 场景分形/容量约束**：block_M/N ≥ 16；`block_M×block_N×sizeof(accum) ≤ L0C(128KB)` | ✅ 必须 |
| 11 | **无全局变量依赖**：维度参数从 tensor shape 或函数参数获取，支持多场景顺序测试 | ⭕ 推荐 |

**通过条件**：必须项（1,2,3,4,5,6,7,10）全部通过。
