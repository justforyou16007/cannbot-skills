# 机制标签：闭合词汇表与停顿锚点

论文 Appendix B 要求：跨角色交接、供给纪律、消费侧分阶段这类机制，
使用**闭合词汇表**的标签，每个标签都要求**停顿锚定的验证**与记录在案的
顺序证据。

## 为什么要闭合词汇表

不限定词汇时，机制验证行会写成这样：

> 「流水重叠已改善，总时间下降 8%」

这句话读起来像验证，实际什么都没验证。它只说了结果变好，没说**那个机制
真的发生了**。而区分「我设计的机制起作用了」与「碰巧变快了」正是机制行
存在的唯一理由 —— 两者在总时间上长得一模一样，但后者无法复用、无法解释、
下一个算子上就不灵了。

闭合词汇表把这件事变成程序可查的：每个标签规定了**必须出现哪个停顿指标**。
写 `SUPPLY_KEEPUP` 就必须给出 `mte2_wait_ratio` 的变化。给不出，
`milestone_validate.py` 的 V11 规则直接拒。

## 为什么锚点是「停顿」而不是「占比」

占比下降有很多种原因，包括别的段变慢了。停顿指标回答的是另一个问题：
**这个单元有没有在等**。机制生效的直接表现是等待减少，而不是占比变化。

举例：加了 double buffer 之后 `aiv_mte2_ratio` 从 61% 降到 31%。
这可能是搬运真的被藏起来了，也可能只是总时间变长了而搬运绝对时间没变。
但 `aiv_mte2_wait_ratio` 下降，说明的是「不再有人在等搬运」—— 这才是
重叠落地的证据。

---

## 词汇表

所有停顿字段来自 `ResourceConflictRatio.csv`，占比字段来自
`PipeUtilization.csv`，命中率来自 `L2Cache.csv`。字段定义与阈值见
[`ops/ops-profiling/references/csv_fields_reference.md`](../../../../../ops/ops-profiling/references/csv_fields_reference.md)。

### 核间与流水

| 标签 | 机制 | 必需停顿锚点 |
|---|---|---|
| `CV_HANDOFF` | Cube 与 Vector 的交接编排 | `aic_cube_wait_ratio` 或 `aiv_vec_wait_ratio` |
| `SUPPLY_KEEPUP` | 搬运供给跟上计算（对应论文的 gather-supply discipline） | `mte2_wait_ratio` |
| `WRITEBACK_OVERLAP` | 写回与下一轮计算重叠 | `mte3_wait_ratio` |
| `CONSUMER_PHASING` | 消费侧分阶段，错开争用 | `aiv_vec_wait_ratio` |
| `L1_RESIDENCY` | 用 L1 作中间驻留层，减少 GM 直达 | `mte1_wait_ratio` 或 `aic_mte1_ratio` |

### 冲突与布局

| 标签 | 机制 | 必需停顿锚点 |
|---|---|---|
| `BANK_LAYOUT` | UB 地址分配避开 bank 冲突 | `aiv_vec_bank_cflt_ratio` |
| `BANKGROUP_STRIDE` | 调整 block stride 避开 bankgroup 冲突 | `aiv_vec_bankgroup_cflt_ratio` |
| `RESOURCE_SERIALIZATION` | 错开计算单元争用 | `aiv_vec_resc_cflt_ratio` |

### 饱和与局部性

| 标签 | 机制 | 必需停顿锚点 |
|---|---|---|
| `CUBE_SATURATION` | 提高 Cube 利用率 | `aic_cube_wait_ratio` |
| `L2_LOCALITY` | 改善分块顺序的局部性 | `total_hit_rate` |
| `OUTPUT_ALIGNMENT` | 输出地址对齐 | `aic_fixpipe_ratio` |

### 暴露时间

| 标签 | 机制 | 必需停顿锚点 |
|---|---|---|
| `EXPOSURE_REDUCTION` | 缩减未被计算覆盖的时间 | `exposed_pct`（由 `signature_probe.py` 推导） |

> `EXPOSURE_REDUCTION` 的锚点是推导量而非实测字段，所以它比其他标签弱：
> 暴露时间说明有多少时间闲置，不说明闲置在哪一段。用它时应同时用
> `lever_probe.py` 做消融定位，否则这条机制行只是换个说法重复了「变快了」。

---

## 怎么写一条合格的机制行

```json
{
  "id": "M1",
  "label": "SUPPLY_KEEPUP",
  "statement": "搬运供给跟上了计算：不再有单元在等搬运",
  "status": "PASS",
  "anchor": {
    "kind": "reference_measurement",
    "expr": "msprof:ResourceConflictRatio.csv:aiv_mte2_wait_ratio < 4.2",
    "provenance": "[MEASURED, job 7]"
  },
  "ordering_evidence": "job 6 的 mte2_wait_ratio 为 4.2，job 7 降至 0.9；两次采集除本决策外配置相同"
}
```

三件事缺一不可：

- **`label`** 必须来自上面的词汇表（V11 检查）
- **`anchor.expr`** 必须包含该标签要求的停顿字段名（V11 检查）
- **`ordering_evidence`** 记录「改之前是多少、改之后是多少」。论文要求
  recorded ordering evidence，因为单独一个数说明不了变化方向

## 常见的不合格写法

| 写法 | 问题 |
|---|---|
| `"statement": "流水重叠改善，总时间降 8%"` | 没有 label，而且用总时间冒充机制证据 |
| `label: SUPPLY_KEEPUP`，`expr` 里是 `aiv_mte2_ratio` | 用了占比而非停顿。占比下降可能只是别的段变慢了 |
| `label: OVERLAP_IMPROVED` | 不在词汇表内。自造标签等于没有标签 |
| 有 label 有 anchor，没有 `ordering_evidence` | 单个数说明不了变化方向 |

## 新增标签的条件

词汇表是闭合的，但不是冻结的。新增一个标签要满足：

1. 它描述的是一个**机制**，不是一个结果（「更快了」不是机制）
2. 存在一个 msprof 字段能直接反映它是否发生
3. 该字段是**停顿或冲突类**，不是占比类 —— 除非能说明为什么占比在这里
   足以区分真机制与巧合

加标签时同步改三处：本文件、`milestone_validate.py` 的 `MECHANISM_LABELS`、
以及 `forge-milestone` 的 SKILL.md 规则表。
