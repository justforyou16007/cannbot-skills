---
name: forge-reference-analysis
description: 把封闭参考实现的 profiler 可观测行为翻译成架构决策假设，产出 milestone 课程表的素材。当需要从 msprof/评测数据反推参考 kernel 的 tiling、流水深度、AIC/AIV 分工、buffer 布局等架构决策，或需要判断某项观测能否支撑某个架构主张时使用。
---

# 参考分析：从硬件行为反推架构

## 这个技能解决什么

给你一个**只能测、不能读源码**的参考实现，要回答：它的架构长什么样。

能这么做，是因为架构决策会在硬件行为上留下签名。改变 tiling 会改变搬运量，
改变流水深度会改变各段占用比例，改变核间分工会改变 AIC/AIV 的时间分布。
这些都是 profiler 测得到的。

反过来说也要承认边界：**有些决策 profiler 分辨不出来**。把分辨不出来的
硬说成分辨得出来，是这套方法最容易出的错。所以本技能的输出分两部分 ——
有签名支撑的架构假设，和明确标注的开放问题。

## 两层划分

| 层 | 内容 | 定了之后 |
|---|---|---|
| **架构决策** | tiling 层次、流水级数与 buffer 放置、AIC/AIV 分工、UB/L1 张量布局、重算-vs-存储边界 | 后续调优都在它之下，改它要重做 |
| **战术知识** | 搬运指令选型、同步原语用法、double buffer 深度、尾块处理 | 可在既定架构下逐项调 |

这个划分是整套方法的支点。通用 agent loop 的天花板正是**架构在头几次尝试
就被定死，之后再多时间也只在战术层打转**。所以参考分析只负责架构层 ——
战术层交给 `ascendc-performance-best-practices`。

## 工作流程

1. **取观测量** —— 跑一次配对评测拿 `evaluation_results.json`，必要时用
   `ops-profiling` 采 msprof 的 `kernel_details`
2. **查签名字典** —— 按观测量到 [`references/signature-dictionary.md`](references/signature-dictionary.md)
   找对应的架构决策候选
3. **写决策** —— 每条按五元组 `action / purpose / trigger / cost / dependence`
   成文，交给 `forge-milestone` 校验
4. **标开放问题** —— 字典里标为「分辨不了」的，写进 milestone 的开放问题，
   不要猜

## 硬性要求

**没有 trigger 的决策不成立。** trigger 就是激活这个决策的 profiler 症状。
说不出「观测到什么才需要它」，这条决策就是不被支持的主张，`milestone_validate.py`
的 V2 规则会直接拒掉。

这条要求的作用是切断「凭经验堆优化」。一个架构决策若只因为「一般来说这样更好」
而存在，它就既无法验证也无法在失效时定位。

## 参考资料

| 文档 | 内容 |
|---|---|
| [`references/signature-dictionary.md`](references/signature-dictionary.md) | 观测量 → 架构决策候选 → 分辨不了的部分 |
| [`references/open-questions.md`](references/open-questions.md) | msprof 在昇腾上分辨不了什么，以及替代探针 |
| [`references/arch-2201.md`](references/arch-2201.md) | DAV_2201 的架构事实与签名差异 |
| [`references/arch-3510.md`](references/arch-3510.md) | DAV_3510 的架构事实与签名差异 |

## 上游真源

本技能不自产硬件事实，全部引自：

| 事实类型 | 真源 skill |
|---|---|
| 芯片规格、架构代际、Buffer 容量 | `npu-arch` |
| msprof 字段含义与采集方式 | `ops-profiling` |
| 算子族优化实践（战术层） | `ascendc-performance-best-practices` |
| 评测产出字段 | `ops-evaluation` 的 `evaluate_ops_direct.py` |

查到的数值一律以运行时 `PlatformAscendC` 接口取得的为准，**禁止硬编码**。
