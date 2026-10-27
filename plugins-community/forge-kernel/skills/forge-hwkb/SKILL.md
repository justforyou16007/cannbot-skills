---
name: forge-hwkb
description: Ascend NPU 算子性能经验知识库的准入、检索、no-code 边界 lint 与死路表，按芯片架构分区存放经硬件探针实测认证的可复用经验。当需要查询某个 profiler 症状下已有哪些 kernel 优化经验、登记一条新经验、检查经验卡片是否越过参考边界、或在重试前确认某方向是否已被证伪时使用。
---

# 硬件知识库

## 这个技能解决什么

ForgeKernel 里**唯一跨 campaign 存活的东西**。

单回合的 agent 跑完一轮，留下一个 kernel，但不留下知识 —— 下一轮要为
上一轮已经学到的东西重新付一次代价。这个知识库就是用来打破那个循环的：
一条经验在 attention 反向的 campaign 里被总结出来，一个只共享同款硬件的
GEMM campaign 可以直接拿来用。

能做到跨算子复用，靠的是两道闸门。两道都是程序判定，不是口头约定。

## 两道闸门

### 准入闸：没有实测背书不准进

一条经验必须有**目标架构上的探针实测**给它背书。agent 自己说「这样更快」
不算证据 —— 论文把 agent 自评的探针单列为一类 bug 来源。

认证还要记录实测变化量，因为认证的作用是**把真正起作用的机制和只是伴随
发生的分开**。一次 campaign 达成了目标，不等于它当中提出的每个解释都成立。

### 边界闸：不准留参考实现的痕迹

卡片正文不得出现参考实现的专有名词，出处只能引用公开文档、公开 API 符号、
探针判定。

这不是洁癖。一条带着某个参考内部命名的经验，换个 campaign 就看不懂了，
跨算子复用无从谈起 —— 而跨算子复用正是这个知识库存在的全部理由。

禁用名单由 `seal_reference.py` 在封存参考时自动抽取并写入 `denylist.txt`，
本技能的 lint 直接读它。两段是接上的。

## 卡片结构

按芯片架构分区：`references/dav2201/`、`references/dav3510/`。
硬件机制本就是架构相关的 —— DAV_2201 的 set/wait 强配对经验搬到
DAV_3510 的 BufferID 同步上是错的。

正文必须有四个小节（缺一不可）：

| 小节 | 内容 |
|---|---|
| `## 机制` | 它做了什么，省的是什么，什么时候**不**该用 |
| `## 触发症状` | 哪个观测量到什么程度才考虑它，以及反指征 |
| `## 最小可运行例` | 结构上怎么改，不必是完整代码 |
| `## 已知坑` | 实践中踩过的 |

卡片正文一律用中文（frontmatter 的结构化字段值除外），与仓内约定一致。

## 脚本工具

```bash
# 起草时自检，不入库
python3 scripts/hwkb.py validate --card draft.md

# 校验通过才写入
python3 scripts/hwkb.py add --card draft.md

# 按症状检索（arch agent 的主要入口）
python3 scripts/hwkb.py query --symptom "mte2 占比高" --arch DAV_2201

# 边界检查
python3 scripts/hwkb.py lint --all

# 死路表
python3 scripts/hwkb.py nogo add --symptom "..." --tried "..." --evidence "[MEASURED, job 9]"
python3 scripts/hwkb.py nogo check --symptom "..."
```

## 检索以症状为主键

因为 arch agent 的实际场景是「我观测到了这个，有人遇到过吗」，不是
「我想找关于 tiling 的资料」。

`operator_families` 留空表示**算子无关**，对任何算子族都命中 —— 这正是
跨算子复用的体现。只有确实绑定某个算子族的经验才填它。

## 死路表

记录试过且被证伪的方向，以症状为 key，作为**重试前置条件**。

死路和正向经验一样要带证据：未经测量的放弃不构成死路。否则死路表会把
从没真正试过的方向也堵死，比没有这张表更糟。

`nogo check` 命中时退出码为 `1`，调用方应据此避开，而不是再试一遍。

## 参考资料

- `references/dav2201/`、`references/dav3510/` — 按架构分区的经验卡片
- `references/denylist.txt` — 参考专有名词（由 `seal_reference.py` 生成，可手工追加）
- `references/nogo.json` — 死路表
