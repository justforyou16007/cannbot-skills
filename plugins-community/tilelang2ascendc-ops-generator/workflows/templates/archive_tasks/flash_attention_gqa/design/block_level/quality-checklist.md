# Block-level 设计质量自检清单

> 仿 `ops/tilelang-op-design/references/quality-checklist.md`，针对 **block-level 设计**（对应
> `BlockLevelDesign.md`）的自检清单。生成/修改 `design/block_level/*.py` 后逐项检查。

| # | 检查项 | 是否必须通过 |
|---|--------|-------------|
| 1 | **任务划分明确**：block 负责输出哪段区域，写出具体输出切片表达式（如 `Output[bz, by, bx*block_M:(bx+1)*block_M, :]`），不是"处理一段输出" | ✅ 必须 |
| 2 | **块内遍历明确**：为得到该输出，block 内要遍历哪些输入数据（哪个维在块内循环，如 kv_loops / chunk 序列） | ✅ 必须 |
| 3 | **布局整理说明**：需要先 pack / 轴合并 / 移位才利于分工的（如 GQA 的 Q/K/V pack、pooling 的 channel 移位），在 block-level 阶段就写明，而不是拖到 tile-level | ✅ 必须 |
| 4 | **C/V 分工明确**：每个子阶段标 `T.Scope("C")`（cube：T.gemm/T.mma/T.copy→L0）或 `T.Scope("V")`（vector：T.tile.*/T.reduce_sum），谁是 cube 谁是 vector 一目了然 | ✅ 必须 |
| 5 | **流水/同步策略明确**：标出是 expert 手动同步；stage 边界全核 barrier（`T.barrier_all` ↔ AscendC `SyncAll`）、阶段内 AIC↔AIV 用 `T.set_cross_flag`/`T.wait_cross_flag`；ring/prelaunch 参数说明 | ✅ 必须 |
| 6 | **workspace 完整**：`workspace_idx` 位置、每个 workspace 张量的 shape/dtype/谁写谁读，列出对照表 | ✅ 必须 |
| 7 | **细粒度计算有 TODO(tile-level) 且指向具体 API**：不是模糊的"计算 S = Q@K^T"，而是写出 `T.copy` 到哪个 L1/L0 buffer、`T.mma` 的 init、`T.reduce_*`、写哪个 workspace | ✅ 必须 |
| 8 | **无占位符/模糊描述**：无 `{placeholder}`、无"待补充"；空 `pass` 前必须紧跟具体 TODO(tile-level) | ✅ 必须 |
| 9 | **expert mode 原语合规**：无 `T.parallel` / `T.Parallel` / `T.pipeline` / `T.Pipelined` 等 developer 原语 | ✅ 必须 |
| 10 | **切片写全最后一维**：运行时变长切片显式补 `[..., 0:d]`（如 `x[s0:s0+valid, h, 0:dk]`），避免后端 `maskShapeN=1` 只拷首元素 | ✅ 必须 |
| 11 | **多 prim_func 模板**：不同 shape 是否需要不同 block 组织（如 rms_norm 按 N 选 merge_n/single_row），用 trace-time 选择并在文件里说明 | ⭕ 推荐 |
| 12 | **与 tile-level / golden 对应**：block-level 的每个 C/V 子阶段都能在 tile-level 落地；或明确标注"细算见 tile_level / golden" | ✅ 必须 |
| 13 | **同类参考已列出**：给出具体的 examples / archive_tasks 文件路径（如 `gdn/`、`flash_attention_gqa/design/tile_level/`） | ✅ 必须 |
| 14 | **参考实现差异已说明**：如有外部参考，列出布局 / API / 结构差异（如 gdn 是 BHLD、本算子是 TND） | ⭕ 推荐 |
| 15 | **技术约束已确认**：非整除 seq、尾块 mask、动态边界（actual_seq_lengths / cu_seqlens）在 block-level 有处理说明 | ✅ 必须 |
| 16 | **GEMM 场景分形/容量约束**：block_M/N ≥ 16；`block_M×block_N×sizeof(accum) ≤ L0C(128KB)` | ✅ 必须 |
| 17 | **无全局变量依赖**：维度参数从 tensor shape 或函数参数获取，支持多场景顺序测试 | ⭕ 推荐 |
| 18 | **验证方案含 golden + L0 shape 用例**：golden 参考完整，至少给出 block 整除的 L0 规则用例；分层套件交 `tilelang-op-test-design`，不写"待补充" | ⭕ 推荐 |

**通过条件**：必须项（1,2,3,4,5,6,7,8,9,10,12,13,15,16）全部通过，推荐项至少通过 4/10。
