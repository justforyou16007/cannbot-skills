---
name: forge-milestone
description: Ascend C 算子架构决策课程表（milestone）的 schema 与门禁校验器，把论文规定的举证义务变成程序判定。当需要起草或修订 kernel 架构 milestone、校验决策五元组与门禁行、判断某个 milestone 能否关闭时使用。
---

# Milestone：架构决策的课程表

## 这个技能解决什么

把参考分析得出的架构决策，组织成一份**可以一行一行实现、一行一行验证**的
课程表，并用程序强制它满足举证义务。

milestone 只存决策和可测目标，**不存实现体**。实现是 coding agent 的事。
这个分离让同一份课程表可以用不同的实现手段去满足，也让「架构错了」和
「实现错了」能被分开定位。

## 两件核心约束

### 决策必须是五元组

每条决策写成 `action / purpose / trigger / cost / dependence`：

| 字段 | 内容 | 缺了会怎样 |
|---|---|---|
| `action` | 做什么 | 无法实现 |
| `purpose` | 为了解决什么 | 无法判断是否达成 |
| `trigger` | **哪个 profiler 症状激活了它** | 变成凭经验的主张 |
| `cost` | 代价与失效条件 | 无法权衡取舍 |
| `depends_on` | 依赖哪些决策 | patch 后不知从哪重执行 |

`trigger` 是其中最关键的一项。说不出「观测到什么才需要它」，这条决策
就不被支持 —— 校验器会直接拒掉。

### 门禁必须有三类行

milestone 关闭时过门禁，门禁由三类行组成：

| 行 | 作用 |
|---|---|
| correctness | 结果还对不对 |
| performance | 快了多少，必须带出处 |
| **mechanism** | **那个机制真的落地了吗** |

第三类最容易被省掉，也最不能省。没有它就分不清「我设计的机制起作用了」
和「碰巧变快了」—— 而把巧合当成机制，会让后续所有基于它的决策建在
错误前提上。

## 工作流程

1. 用 [`references/example-milestone.json`](references/example-milestone.json) 作模板起草
2. 跑校验器：`python3 scripts/milestone_validate.py milestone.json`
3. 按报错逐条补证据，直到通过
4. 交给 coding agent 一次实现一行

## 脚本工具

```bash
# 全量校验
python3 scripts/milestone_validate.py milestone.json

# 只看某几条规则
python3 scripts/milestone_validate.py milestone.json --rules V2,V6

# 机器可读输出
python3 scripts/milestone_validate.py milestone.json --json

# 列出全部规则
python3 scripts/milestone_validate.py --list-rules
```

退出码 `0` 通过，`1` 有违规，`2` 文件无法解析。

## 十条校验规则

| 规则 | 检查什么 | 为什么 |
|---|---|---|
| V1 | 五元组完整 | 缺项的决策无法独立评估 |
| V2 | 决策有 trigger | 没有症状支撑的是主张不是决策 |
| V3 | 派生图无环 | 有环就不知道 patch 后从哪重执行 |
| V4 | 数字有出处 | `[MEASURED, job N]` 或 `[ESTIMATE]`+推导 |
| V5 | 门禁三类行齐全 | 缺 mechanism 行就分不清真机制与巧合 |
| V6 | 未决行已销账 | DEFERRED/INFEASIBLE/N/A 各有举证责任 |
| V7 | 门禁锚点来源受限 | 防止自己定一个好听的目标再去达成 |
| V8 | M0 无绝对性能阈值 | 架构刚落地没调过，绝对阈值会误杀 |
| V9 | 原子组同批落地 | 真子集单独落地会退化 |
| V10 | 架构决策有验证通道 | 区分「真落地」与「声称落地」 |

## 销账的举证责任

未解决的门禁行不能就这么挂着，必须销账。三种销账各有各的门槛：

| 销账 | 必须提供 |
|---|---|
| `DEFERRED` | 证据表明参考达到了该指标 + 当前手段改不动它 + **具名的下游手段**接手 |
| `INFEASIBLE` | 逐指标论证 + 证据表明目标被排除 |
| `N/A` | 结构上不适用的理由 |

`DEFERRED` 不指名下游手段，就不是推迟而是丢弃 —— 这正是校验器要拦的。

## 一个容易踩的坑

`trigger.channel` 写错来源不会报错，只会永远读到空值：

```
架构签名（各段占比、核数）→ msprof:PipeUtilization.csv:aiv_mte2_ratio
性能与精度             → evaluation_results.json:comparison.speedup
```

配对评测**有意不采 msprof**，它输出里的 `pipeline` 和 `bottleneck` 恒为空。
把架构签名的通道指到那里，门禁会变成看起来在跑、实际什么都没验证。
详见 `forge-reference-analysis` 的签名字典第六节。

## 参考资料

- [`references/example-milestone.json`](references/example-milestone.json) — 可直接改的完整样例，本身通过全部十条规则
- [`references/schema.md`](references/schema.md) — 字段逐项说明
