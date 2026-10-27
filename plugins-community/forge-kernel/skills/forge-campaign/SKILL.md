---
name: forge-campaign
description: campaign 驱动：参考封存与适配、噪声底实测、配对计时裁决、嵌套双循环状态机与自我提升。当需要开启一轮 ForgeKernel campaign、判断某个候选是否该被接受、决定何时交回架构层重新分析、或把本代产出封存为下一代参考时使用。
---

# Campaign 驱动

## 这个技能解决什么

把一轮 ForgeKernel 跑起来，并在每个关口上做出**有测量支撑的**判断：
这个候选收不收、现在该继续调还是回头改架构、这条自我提升的链还要不要往下走。

## 嵌套的两层循环

| 层 | 谁在动 | 什么时候动 |
|---|---|---|
| 内层 | coding agent | 一次实现一行决策，一次评测定去留 |
| **外层** | **arch agent** | **内层连续无进展、决策图走穷、或门禁僵持时** |

外层是单回合 agent 没有的东西，也是整套方法的要害。它能**重开 M0 的架构
决策**，而不是在一个已经错了的架构下继续硬调。论文测下来，通用 agent loop
的天花板正是卡在这里：架构在头几次尝试就定死，之后给再多时间也只在战术层
打转。

## 四个步骤

### 1. 封存参考

```bash
python3 scripts/seal_reference.py seal \
    --artifact /path/to/reference --op-name add_layer_norm --work-dir ./forge_work
```

产出两份清单，交给两个独立于提示词的执行点：

- `sealed_paths.txt` → PreToolUse hook，在工具调用层拦截读参考源码
- `denylist.txt` → 知识库 lint，拦截参考专有名词混进可复用经验

参考本身就是二进制（如 CANN 内置实现）时没有源码可封，这一步会如实
提示「未发现源码文件」—— 那种情况天然满足封闭要求。

### 2. 适配参考来源

```bash
python3 scripts/reference_adapter.py prepare --kind builtin_aclnn \
    --op-name add_layer_norm --work-dir ./forge_work
python3 scripts/reference_adapter.py verify --spec ./forge_work/reference_spec.json
```

三种来源归一化成评测脚本认的一个 `--baseline-path`：

| 来源 | 说明 |
|---|---|
| `builtin_aclnn` | CANN 算子库现成实现，天然封闭 |
| `opp_install` | 已构建的自定义算子安装路径，封闭性靠禁读清单维持 |
| `sealed_kernel` | 自己上一代的产出，自我提升用 |

> `builtin_aclnn` 依赖「空 OPP 目录使 aclnn 派发落到内置实现」这一行为，
> **该行为尚未在真机上验证**。`verify` 子命令就是用来在目标机器上把它
> 变成一次有确切结论的检查，不要跳过。

### 3. 量噪声底

```bash
python3 scripts/noise_floor.py --from-results evaluation_results.json
```

论文报的 1%~5% 是**它们环境的实测值，不是常数**。定错了后面全是错的：
定太低会把抖动当成果收进来，还会污染知识库；定太高会把真实的小幅改进
当噪声丢掉，优化提前停在远低于硬件上限的地方。

首选直接读评测结果里的 `cv_pct`（同一 job 内的离散度，正对应论文说的
same-job noise floor）；没有时才回退到多次结果之间的比值离散。

### 4. 跑 campaign

```bash
python3 scripts/campaign.py init --operator add_layer_norm --arch DAV_2201 \
    --reference-kind builtin_aclnn --reference-id aclnnAddLayerNorm \
    --noise-floor 2.3 --target-gain 1.05 --state ./forge_work/campaign_state.json

python3 scripts/campaign.py record --state ./forge_work/campaign_state.json \
    --result evaluation_results.json --decision D0.1

python3 scripts/campaign.py status --state ./forge_work/campaign_state.json
python3 scripts/campaign.py ledger --state ./forge_work/campaign_state.json
```

每次 `record` 会给出裁决和**下一步该做什么**。状态持久化在
`campaign_state.json`，上下文丢了也能从文件恢复进度。

## 裁决规则

```
精度不过            → 回滚（不看增益，算错的 kernel 再快也没意义）
无测量支持          → 回滚（agent 声称有效不算数）
增益 ≤ 噪声底       → 回滚并记为 no-go
增益 > 噪声底       → 接受
明确退化超出噪声     → 回滚并标记为退化
```

边界上的判定取保守：恰好落在噪声底上算**不**超过。收错一个噪声会污染
知识库并误导后续决策，拒错一个真改进只是损失一次机会，两者代价不对称。

## 自我提升

```bash
python3 scripts/campaign.py promote --state ./forge_work/campaign_state.json \
    --artifact ./best_kernel
```

把本代产出封存成下一代的参考，重新分析、重新锚定目标。**源码不跨代边界，
知识库跨** —— 这是两条不重复的反馈边：提升抬高性能锚点，知识库降低够到
它的成本。

链条停在确认的阶段增益跌破噪声底时。到那一步，`promote` 会自动把状态
置为 `stopped` 而不是无谓地再开一代。

> 这个停点是对硬件上限的一次**估计，不是证明** —— 也可能只是这个循环
> 找不到了。论文自己也是这么说的，不要把它当成上限已达的结论。

## 脚本工具

| 脚本 | 用途 |
|---|---|
| `scripts/campaign.py` | 状态机：init / record / status / promote / ledger |
| `scripts/noise_floor.py` | 实测噪声底 |
| `scripts/seal_reference.py` | 封存参考，产出禁读与禁词清单 |
| `scripts/reference_adapter.py` | 三种参考来源归一化 + 环境预检 |

## 依赖的外部能力

| 能力 | 来自 | 用途 |
|---|---|---|
| 配对计时与精度 | `ops-evaluation` 的 `evaluate_ops_direct.py` | 算 g(K,R) |
| 流水线占比 | `ops-profiling` 的 msprof | 架构签名 |
| 芯片规格 | `npu-arch` | 门禁锚点的硬件规格来源 |

> 注意：不要用同目录的 `evaluate_ops.py`，它引用的 `AscendBackend`
> 在仓内不存在。只用 `evaluate_ops_direct.py`。
