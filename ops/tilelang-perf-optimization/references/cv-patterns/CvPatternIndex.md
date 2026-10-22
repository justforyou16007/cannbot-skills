# C/V 融合场景索引

用于设计候选、转译合同核对和调优路由。这里的 C/V 是 Cube/Vector。按实际数据依赖选择卡片，**场景编号不是已经测出的 Tv/Tc 分区**。

## 按症状选择

| 看到的计算/数据流 | 先读 | 决策产物 |
|---|---|---|
| 成熟 Matmul 后接 norm/激活，融合是否值得尚不清楚 | [串行基线](serial-baseline.md) | 串行基线、融合候选与实际 dispatch |
| C/V 跨块交错、双 workspace、异步计算后的等待窗口 | [流水和槽位生命周期](pipeline-lifetime.md) | ready/reuse 依赖、启动/排空、每槽容量 |
| gate/softplus/exp/sigmoid 链，没有矩阵乘 | [纯 Vector 链](vector-gating.md) | 公式分支、参数复用、V 核候选 |
| sparse/PA 索引结果喂给 Cube | [稀疏取数与布局](sparse-layout.md) | token/block 单位、页表、有效长度、物化位置 |
| batch reduce、running Top-K、online softmax | [在线消费与归约](online-consumption.md) | 必须保留的状态、输出语义、全量中间体是否仍存在 |
| dot 后 gate/beta/mask，或多份 dot 结果合并 | [局部 epilogue](local-epilogue.md) | 数值域、cast 点、最终写出和局部活跃集 |

卡片的共同格式是：何时读 → 可迁移规则 → 源码案例与限制 → 最小验证 → 固定版本证据。只读取命中卡片；多模式组合时先核对合同，再讨论调度。

## 证据等级与适用边界

- 来自用户提供的 `cann_scenario_research`（2026-09-22）：13 条场景映射、11 个主要 L4 条目；FFN、GqaSparseFwd 各跨两个场景复用。卡片保留 S1-01 等研究编号用于回溯。
- L4 definition 描述测试合同；源码描述一个特定分支的机制；运行时 dispatch 证明该分支被调用；编译、精度和 profiling 才能证明目标平台行为与收益。这四者不可互相替代。
- 本组卡片是静态机制知识，**没有 NPU 编译、板测或性能验收**。Triton、AscendC 与框架调用案例不能视作已验证的 TileLang 模板，也不进入 `archive_tasks`。
- CANN 案例引用 `hicann` 的 GitHub 固定快照；未证明与官方 GitCode 当前 HEAD 一致。vLLM 与 L4 同样固定 commit，迁移时重新核对目标版本。
- 图级双流直接案例、真实 `Tv/Tc` 分区和端到端收益仍缺独立证据；核内 MIX 流水不能证明“核外双流获得九成收益”。

## 写入设计记录

在现有 `design/PERF_DESIGN.md` 或调优记录中补充命中卡片及：

1. 公式/返回值/副作用，shape、dtype、layout、stride、scale、mask、舍入点。
2. 实际 lengths、groupList、有效 sparse 数、非恒等页表；相同 shape/dtype 不等于相同有效工作量。
3. 芯片、软件版本、实际 kernel/tiling key、eager/graph、kernel 数；尚未确认的项列为待验证。
4. 串行与候选的设备耗时、整段耗时、workspace 峰值、误差；字节估算与实测流量分列。

容量按 UB/L1/L2/GM 分层计算。能放进 L2 不保证命中，也不能替代数据就绪和复用同步。测量遵循 [measurement_discipline.md](../measurement_discipline.md)；异常转 [cv_sync_debug.md](../cv_sync_debug.md)。

## 固定版本证据

- [PPT](https://github.com/grandolantow/safe/blob/8a84bfd76c81f0c59e1597a8c42df7e2c81223d9/docs/ppt/%E5%86%85%E5%AE%B9%E5%AF%BC%E8%AF%BB.md)
