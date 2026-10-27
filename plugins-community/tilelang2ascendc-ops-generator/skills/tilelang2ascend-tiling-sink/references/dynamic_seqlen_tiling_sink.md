# 动态 seq-len serving：tiling 下沉（策略 X69）部署形态卡

> **激活方式**：本卡随 `tilelang2ascend-tiling-sink` skill 分发，**仅当用户显式要求使用
> tiling 下沉时读取**——默认设计/转译/调优流程（designer/translator/perf）不自动路由到此。

> **适用特征**（确认与本场景相符后再继续）：
> - model.py / tile-level 设计出现逐请求变化的 seq-len：`actual_seq_qlen` / `actual_kv_len` /
>   `cu_seqlens` / varlen TND 拼接布局 / `q_start_locs`，且部署目标是 serving（每步 batch 组成变化）
> - 部署侧症状：每步 host 读值（`aten::item`/`.tolist()`）或 list 落卡（H2D）、host tiling 随值重算、
>   图模式逐层重绑（graph_task_update）、需要 NPUGraph/aclgraph capture-replay
>
> 本卡是**路由卡**：只给成本门、形态选择与各阶段动作；机制与完整证据见文末真源指针，不复制正文。

## 一句话

把 host tiling 的**值依赖**下沉为 device 侧：前置 metadata 算子读 device seq-len 算分核 → 写 GM
元数据（int32/uint32[1024]）→ 主算子 host tiling 变 **shape-only**、kernel 启动后直读 GM 分核。
seq-len 全程 device tensor，D2H/H2D/aten::item 归零；图重放按三原则兼容。

## 先算账（成本门——不达标不要做）

| 项 | 量级 | 出处 |
|---|---|---|
| AICPU producer 固定成本 | 130-317µs/call（本体 <1µs，全是调度开销）+ 双 aclnn 串行 150-190µs | 910_93 板测 |
| AIV producer（轻量计算） | ~4µs（950PR 实测 4.02µs vs AICPU 26.76µs，6.65x） | ops-transformer pr-14127 |
| **净赚判据** | 主 kernel ≳500µs 的 kernel-bound 负载净平/微赚；launch-bound 短负载净亏（实测最差 0.12x） | 板测 C01-C20 |
| 框架方案同样过门 | CANN ENABLE_TILING_SINK 框架实测上限 ~9µs/call（<1% serving）被否决 | pr-14127 探针 |

## 形态选择

- **producer 引擎**：metadata 计算轻（纯向量可表达）→ AIV 普通算子（快、且可入图重放，无拓扑约束）；
  需要 host 型重逻辑 → AICPU 定制算子（须遵守图外重算拓扑，见下）。
- **图重放三原则**（aclgraph capture/replay 兼容）：① AICPU metadata 算子不进图（重放不重执行），
  每步 capture/replay 前 eager 重算进**常驻稳定地址** GM buffer；② 值导出标量传 -1 哨兵从 live
  张量推导（Python 标量会被捕获冻结）；③ 每个 batch 档位独立稳定 buffer（aclnn 静态描述符不跨档复用）。

## 各阶段动作

| 阶段 | 能做 / 该做 | 不能 / 不要 |
|---|---|---|
| designer（Step 0/1e） | varlen 布局（TND 累计和）保持；若分核/调度依赖 seq 值，把"每核任务区间"设计为 **GM 元数据驱动**（kernel 内读 GM 分核，host 只传 shape）；命中触发条件时把成本门结论写进 PERF_DESIGN.md | 不要因"将来可能 tiling 下沉"而改变 kernel 数学主体（策略不动数学，精度 bit 级不变） |
| translator（Step 0-A） | 记录 seq-len 语义（累计 vs 逐请求）、indexing 单位与有效长度，写入转译产物说明，供集成层对接 | 转译产物不含 producer 算子/双 aclnn 两段式——那是集成层工作 |
| perf（Step 3/4） | serving 口径症状（每步 D2H/H2D/重绑）命中时，先读真源 X69 卡再评估；kernel 侧逐任务 GM 标量回读的判断链可**凭证化**（一次性 padFree，写集 bit 级不变，实测 md 税 -22.1%） | 不要在 TileLang 生成物里生成 AICPU 算子；不要对 launch-bound 负载套用此形态 |

## 证据等级（避免把机制当已板测模板）

| 证据 | 等级 | 来源 |
|---|---|---|
| 单算子板测（910_93，C01-C20 门禁，精度 20/20 bit 级 vs 官方） | 板测 | fia-tiling-sink 独立轮 round 1-3 |
| 服务级四节点 A3（DP4×TP16×EP64，GQA DSpark，图重放 16-128 桶，528/528 全对，ITL p50 -6.31%） | 生产验证 | vllm-ascend PR #15336（2026-10） |
| MSA split-KV 应用（950PR；AIV producer 6.65x、padFree 凭证化 md 税 -22.1%、框架探针否决） | 板测+服务链路 | ops-transformer pr-14127 |
| 图重放三原则 | 服务级验证（同 #15336） | — |

## 证据与延伸

本卡自包含完整判据（成本门/形态选择/各阶段动作均可独立执行）。证据出处：

- 单算子板测：npu_fused_infer_attention_score_v2_sink（fia-tiling-sink 独立轮，
  910_93 / CANN 9.1.0，精度 20/20 bit 级 vs 官方）
- 服务级生产验证：vllm-ascend PR #15336（四节点 A3，GQA DSpark 图重放）
- MSA split-KV 应用：ops-transformer pr-14127（AIV producer 6.65x、消费端判断链
  凭证化 md 税 -22.1%、CANN ENABLE_TILING_SINK 框架探针否决）

完整移植 runbook / 设计模式卡 / 失败模式由维护方本地 OKF 知识库持有，不在本仓。
