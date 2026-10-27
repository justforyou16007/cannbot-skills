# milestone.json 字段说明

配套样例见 [`example-milestone.json`](example-milestone.json)，它通过全部十条校验规则，
可直接复制修改。

---

## 顶层

| 字段 | 必填 | 说明 |
|---|---|---|
| `schema_version` | 建议 | 当前 `"1.0"` |
| `campaign` | 建议 | 本轮的算子、架构、参考、目标增益、噪声底 |
| `resource_topology_contract` | V4 检查 | 从参考签名转录的资源拓扑，M0 决策要对照它检查 |
| `milestones` | 是 | 有序的 M0 → Mk |

### `campaign`

| 字段 | 说明 |
|---|---|
| `operator` / `arch` / `soc` | 标识 |
| `generation` | 第几代。自我提升每提升一次加一 |
| `reference` | `kind`（`builtin_aclnn` / `opp_install` / `sealed_kernel`）、`id`、`sealed` |
| `target_gain` | 目标增益 τ，如 `1.05` |
| `noise_floor_pct` | **实测**噪声底，由 `noise_floor.py` 量出，不要照抄论文的 1~5% |

### `resource_topology_contract`

每个条目是 `{value, provenance, channel}`。

它的作用是给 M0 的架构决策提供对照基准 —— 一个声称「用 48 个核」的决策，
要能对上从参考那里转录来的 `block_dim`。

`provenance` 必须是 `[MEASURED, job N]` 或 `[ESTIMATE] <推导过程>`（V4）。
裸 `[ESTIMATE]` 不附推导等同于没有出处，会被拒。

---

## `milestones[]`

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | `M0`、`M1`…… `M0` 有特殊规则（V8） |
| `kind` | 建议 | `foundation`（patch 要从最早依赖决策重启）或 `additive`（原地恢复） |
| `title` | 建议 | 一句话概括 |
| `decisions` | 是 | 决策行 |
| `gate` | 是 | 关闭时的门禁（V5） |

### M0 的特殊地位

M0 装最要紧的决策：算法契约（分解、数据流、中间张量 ABI、精度）
和硬件架构（tile 形状、核间分工、寄存器预算、UB 布局、MMA 粒度）。

它**先关闭但不冻结**：后续发现天花板是架构造成的，arch agent 会重开它。
这正是 ForgeKernel 与单回合 agent 的分水岭。

M0 的门禁**不设绝对性能阈值**（V8），只查正确性、架构谓词和机制行。
理由：架构刚落地还没调过，拿绝对时间卡它会把正确的架构当失败砍掉。
该阶段只能用相对参考的比率。

---

## `decisions[]` — 五元组

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | 全局唯一，如 `D0.1` |
| `layer` | V10 用 | `architectural` 或 `tactical` |
| `action` | 是 | 做什么 |
| `purpose` | 是 | 解决什么问题 |
| `trigger` | 是 | **激活它的 profiler 症状**，见下 |
| `cost` | 是 | 代价与失效条件 |
| `depends_on` | 是 | 依赖的决策 id 列表，构成 DAG（V3），根决策填 `[]` |
| `atomic_group` | 否 | 必须同批落地的决策集标记（V9） |
| `verification_channel` | 架构决策必填 | 验证它真的执行了的机器可读通道（V10） |

### `trigger`

```json
"trigger": {
  "symptom": "参考的 MTE2 占比超过 50%，判为 MTE2 Bound，说明数据复用不足",
  "channel": "msprof:PipeUtilization.csv:aiv_mte2_ratio",
  "observed": {"value": 61.2, "provenance": "[MEASURED, job 1]"}
}
```

`symptom` 说清观测到什么，`channel` 给出能被程序复核的来源。
两者缺一都会被 V2 拒掉。

**channel 的来源要分清**：

```
架构签名（各段占比、核数、频率）→ msprof:<CSV>:<列名>
性能与精度                     → evaluation_results.json:<字段路径>
```

配对评测有意不采 msprof，其输出的 `pipeline` 和 `bottleneck` 恒为空。
把架构签名指到那里不会报错，只会永远读到空值。

### `atomic_group`

标记一组必须一次提交落地的决策 —— 任何真子集单独落地都会退化。
成员必须在同一个 milestone 内（V9），否则不可能同批落地。
只有一个成员的组是无意义的，也会被拒。

---

## `gate` — 三类门禁行

```json
"gate": {
  "correctness_lines": [...],
  "performance_lines": [...],
  "mechanism_lines": [...]
}
```

三类缺一不可（V5）。mechanism 行最容易被省，也最不能省 —— 它是
「机制真落地了」和「碰巧变快了」之间唯一的区分手段。

### 每条行的字段

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | 行内唯一 |
| `statement` | 是 | 这条要证明什么 |
| `status` | 是 | `PASS` / `FAIL` / `OPEN` / `DEFERRED` / `INFEASIBLE` / `N/A` |
| `anchor` | 性能行必填 | 判据来源，见下 |

`OPEN` 表示未销账，milestone 不能带着 `OPEN` 行关闭（V6）。

### `anchor` — 判据来源受限

```json
"anchor": {
  "kind": "derived",
  "expr": "evaluation_results.json:comparison.speedup >= 0.95",
  "derived_from": "resource_topology_contract.baseline_time_us",
  "provenance": "[MEASURED, job 2]"
}
```

`kind` 只允许三种（V7）：

| kind | 含义 |
|---|---|
| `reference_measurement` | 直接来自参考实测 |
| `hardware_spec` | 来自硬件规格（`npu-arch`） |
| `derived` | 从前两者派生，必须填 `derived_from` |

这条限制是防止 agent 自己定一个好听的目标再去达成它。

### 销账的举证责任

| status | 必须提供 |
|---|---|
| `DEFERRED` | `evidence` + `reference_achieves` + `current_levers_cannot` + `downstream_lever` |
| `INFEASIBLE` | `evidence` + `per_metric_argument`（非空列表） |
| `N/A` | `evidence` + `structural_reason` |

`DEFERRED` 不指名 `downstream_lever`，就不是推迟而是丢弃。
这是 V6 重点拦截的情形。

---

## 校验

```bash
python3 ../scripts/milestone_validate.py milestone.json
```

全部十条规则的清单见 [SKILL.md](../SKILL.md)。
