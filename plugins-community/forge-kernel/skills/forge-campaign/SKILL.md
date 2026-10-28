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
| **外层** | **arch agent** | **三个口子：内层无进展、决策图走穷、门禁僵持** |

外层是单回合 agent 没有的东西，也是整套方法的要害。它能**重开 M0 的架构
决策**，而不是在一个已经错了的架构下继续硬调。论文测下来，通用 agent loop
的天花板正是卡在这里：架构在头几次尝试就定死，之后给再多时间也只在战术层
打转。只实现一个口子就等于少了两次重开架构的机会。

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

论文报的 1%~5% 是**它们环境的实测值，不是常数**。定太低会把抖动当成果
收进来，还会污染知识库；定太高会把真实的小幅改进当噪声丢掉，优化提前停在
远低于硬件上限的地方。

首选直接读评测结果里的 `cv_pct`（同一 job 内的离散度，正对应论文说的
same-job noise floor）；没有时才回退到多次结果之间的比值离散。

### 4. 跑 campaign

```bash
python3 scripts/campaign.py init --operator add_layer_norm \
    --input-shape "B=8,S=4096,H=1024,fp16" --arch DAV_2201 \
    --reference-kind builtin_aclnn --reference-id aclnnAddLayerNorm \
    --noise-floor 2.3 --target-gain 1.05 --budget 40 \
    --state ./forge_work/campaign_state.json

python3 scripts/campaign.py record --state ./forge_work/campaign_state.json \
    --result evaluation_results.json --decision D0.1 \
    --signature architecture_signature.json --milestone milestone.json \
    --recipe R-2201-001 --symptom "MTE2 占比过高" \
    --nogo-store <forge-hwkb>/references/nogo.json
```

**`--input-shape` 是必填的。** 论文 §3.1：一次 campaign 取一个算子、
一个输入形状、一个封存的参考和一个目标增益。噪声底、架构签名、门禁锚点
全都与 shape 绑定 —— 同一个 tiling 决策在不同 shape 下的 trigger 完全
不同。一份 state 跨 shape 复用会把不同 shape 的结果混进同一个 `best`，
promote 时封存的参考与锚定的目标就对不上了。

## 裁决：机制在回路里

一次 GPU job 的裁决由三部分组成：独立的正确性检查、与参考的配对计时、
**以及 profiling**。

```
精度不过                    → 回滚（算错的 kernel 再快也没意义）
无测量支持                  → 回滚（agent 声称有效不算数）
增益 ≤ 噪声底               → 回滚并记为 no-go
增益 > 噪声底 + 机制已核实  → accept
增益 > 噪声底 + 机制未核实  → accept_unverified
明确退化超出噪声            → 回滚并标记为退化
```

### 为什么要有 `accept_unverified`

增益超过噪声底只说明「变快了」，不说明「设计的机制落地了」。两者在总时间
上长得一模一样，但后者无法复用、无法解释、换个算子就不灵。

`accept_unverified` 的改动**保留**（它确实更快），但：

- **不计入 `best`** —— 否则 promote 会把一个解释不清的结果封成下一代参考
- **不得作为知识库入库依据**（`kb_admissible: false`）

论文 §3.4 专门讲这件事：认证把真正起作用的机制和只是伴随发生的分开；
一次 campaign 达成目标，不等于它当中提出的每个解释都成立。一个巧合加速
若被写进知识库，其他算子会照着它做决策。

机制核验按机制行的**停顿锚点**求值，而且查两件事：

1. 阈值成立
2. 那个字段确实朝声称的方向**移动过**（跟上一轮或参考做差）

只查第一条不够。阈值来自参考实测，「低于 4.2」的含义是「我比参考等得
少」—— 它拦不住「这个 kernel 的停顿本来就只有 0.9，改动做的是别的事」。
那种情况下门禁看起来过了，实际什么都没验证。

四种结果不要混：

| 结果 | 含义 | 连续多次时该做什么 |
|---|---|---|
| 成立 | 阈值成立且字段确实动过 | — |
| 不成立 | 阈值不成立 —— 机制没落地 | **改课程表的机制假设** |
| 平凡通过 | 阈值成立但字段没动 —— 不是本次改动造成的 | 同上 |
| 无法求值 | 字段没采到、缺基线、表达式不认识 | 补 record 的参数或基线 |

不传 `--signature` / `--milestone` 时一律判为未核实 —— 工具不替调用方
假设「没查就是通过」。

### 做差基线

| 情形 | 基线 |
|---|---|
| 有更近的有效台账快照 | 上一轮那条决策 |
| 首轮 | **参考自己的签名**（第 0 轮） |
| 都没有 | 判为无法核验 |

所以 `init` 要带 `--reference-signature`，或事后补：

```bash
python3 scripts/campaign.py set-reference-signature \
    --state ./forge_work/campaign_state.json \
    --signature reference_signature.json
```

台账里存的是签名**快照**（通道→数值），不是文件路径 —— 路径会失效，
而那次测量过去了就补不回来。

`reopen` 作废过的条目不能当基线：它们建立在旧架构假设上，拿它做差会把
两套假设之间的差异算到本次改动头上。`promote` 换参考后参考快照一并清空，
需要对新参考重新采一次。

### 「没核验」与「被证伪」是两回事

两者都判 `accept_unverified`，但连续多次时触发**不同的重入口**，
因为该做的事相反：

| 重入口 | 成因 | 动作 |
|---|---|---|
| `mechanism_never_verified` | record 没带 `--signature` / `--milestone`，或签名里缺该字段 | `fix-invocation` —— 补参数重跑，课程表不用动 |
| `mechanism_refuted` | 传了，而且求值为假 | `re-analyze` —— **改课程表的机制假设** |

后者是个很强的架构信号：**对参考行为的解释错了，而实现碰对了别的东西**。
论文说认证把真正起作用的机制和只是伴随发生的分开 —— 要抓的正是这种情形。

把两者混成一条提示的后果很具体：架构层会去补参数重跑，而真正该改的
机制假设没人碰，循环继续空转。所以 `status` 分别计数、`stall_reasons`
分别列出。

### 停滞归因分三类

| 计数器 | 计什么 | 触发什么 |
|---|---|---|
| `stagnation` | 有有效测量但增益不足 | 架构层重入 |
| `implementation_failures` | 精度不过、评测缺失 | 仅提示，**不**触发架构重入 |
| `mechanism_unverified` | 变快了但机制不算数（再分证伪/无法核验） | 按成因分别导向架构层或补参数 |

前两类分开计是因为编译/精度 bug 是实现问题，不是架构问题。混在一起算，
连续三个编译错误就会白白消耗一次架构重入机会。

## 重开架构决策

```bash
python3 scripts/campaign.py reopen --state ./forge_work/campaign_state.json \
    --milestone milestone.json --decision D0.1 --reason "拓扑决策有缺陷"
```

论文 §3.3：追加型补丁原地恢复，**基础型补丁从最早的依赖决策重启** ——
这正是 M0 的拓扑决策被「重开」而不是「在它底下调」的实现方式。

`reopen` 按 `depends_on` 算反向闭包，把所有（直接或间接）依赖该决策的
台账记录标为作废，并重算 `best`。

靠人记哪些要作废是不牢的。漏掉一条，新 M0 上就还挂着建立在旧假设上的
实现，归因直接错掉 —— 而这恰恰发生在整个流程最关键的那一步上。

## 死路表自动登记

`record` 判为回滚时：

- 给了 `--nogo-store` → 直接写入死路表
- 没给 → 打印一条可直接粘贴执行的 `hwkb.py nogo add` 命令

论文把「回滚」与「记死路」写成一件事。靠 agent 自觉是不牢的：连续失败
的时候最容易跳过的就是这种记账动作，而死路表的全部价值在于跨代存活 ——
漏记的那条，下一代会原封不动再试一遍。

## 知识库的返回路径

```bash
python3 scripts/campaign.py record --recipe R-2201-001 --invented R-2201-009 ...
python3 scripts/campaign.py ledger --recipes
```

论文 §3.4：**处方是返回路径** —— arch agent 在需要的决策行里按名字引用
卡片，campaign 台账同时记录本轮发明的和外来的，这让知识库的贡献可审计
而不是口说无凭。

`ledger --recipes` 给出自产/外来两列，以及「接受的改动里有多少用到了外来
经验」。这个比例是判断知识库值不值得继续投入的依据 —— 没有它，你无法
回答「知识库到底有没有在帮忙」，也发现不了某张卡片其实一直在误导。

## 封存产物库

自我提升建立在一个前提上：**上一代的产出，下一代还拿得到**。

`promote` 若只把产物路径记进 state，那个目录被清掉、被覆盖、被后续迭代
改写，第 n 代就对不上第 n-1 代了 —— 而且没有任何地方能看出参考变过。
跑到第三代时第一代产物多半已经不在原处。

```bash
# 登记外来参考为 gen0
python3 scripts/reference_library.py register --library ~/.forge-kernel/lib \
    --operator add_layer_norm --arch DAV_2201 --input-shape "B=8,S=4096,fp16" \
    --kind builtin_aclnn --id aclnnAddLayerNorm \
    --signature reference_signature.json --time-us 456.5 \
    --bench-task tasks/level2/add_layer_norm --bench-baseline-us 500.0

# campaign 提升时自动入库（库里没有该算子时会先补登参考为 gen0）
python3 scripts/campaign.py promote --state ./forge_work/campaign_state.json \
    --artifact ./best_kernel --library ~/.forge-kernel/lib \
    --signature architecture_signature.json

# 查当前该用哪一代作参考
python3 scripts/reference_library.py resolve --library ~/.forge-kernel/lib \
    --operator add_layer_norm --arch DAV_2201 --input-shape "B=8,S=4096,fp16"

# 校验产物有没有被动过
python3 scripts/reference_library.py verify --library ~/.forge-kernel/lib
```

### 库解决四件事

| | 做法 | 不做会怎样 |
|---|---|---|
| 留存 | 产物**复制**进库，不是记路径 | 工作目录一清，下一代拿不到上一代 |
| 可核 | 每代带校验和 | 参考变了无人知晓，而之前所有加速比都已失效 |
| 身份 | 记 CANN 版本、soc、环境 | 不同环境的数字被放在一起比 |
| 血统 | 记从哪代来、增益多少、机制核没核实 | 「累计 +45%」说不清是相对谁 |

### 准入：机制未核实的不得入库

和知识库同一条规矩。封存一个解释不清的结果，后续各代都会建立在它之上，
而没人知道它为什么快 —— 下一代的「增益」也就无从归因。

`campaign.py record` 判为 `accept`（而非 `accept_unverified`）才算核实过。

### 厂商实现没有产物可封

`builtin_aclnn` 的二进制属于 CANN，拷不走也不该拷。这一类入库时只记
**可观测身份**：架构签名快照、实测时间、环境。

CANN 升级后内置实现变了，签名会漂 —— 这是这类参考唯一的变更检测手段：

```bash
python3 scripts/reference_library.py drift --library ~/.forge-kernel/lib \
    --operator add_layer_norm --arch DAV_2201 --input-shape "B=8,S=4096,fp16" \
    --generation 0 --signature current_signature.json
```

漂了就说明参考不是原来那个，之前的加速比不可比，应重新 register 一代
并重测噪声底。

### 自定义参考入库

把本地某个目录下的算子纳入库作为参考，一条命令完成校验、封存、登记：

```bash
python3 scripts/reference_library.py ingest --library ~/.forge-kernel/lib \
    --source /path/to/my_reference_op \
    --operator add_layer_norm --arch DAV_2201 \
    --input-shape "B=8,S=4096,fp16" --id my_layer_norm \
    --signature reference_signature.json --time-us 420.0 --device-id 3
```

它做三件事，少任何一件这个参考都不算封好：

1. 校验目录确实像一个算子实现
2. **封存** —— 产出禁读清单与专有名词清单
3. 登记为 gen0

第 2 步最容易被漏。手工串 `seal_reference` + `register` 时漏掉它，
一切照常运转，只是 no-code 边界形同虚设：agent 可以随便读这个参考，
而读了之后架构结论就是抄的。

`ingest` 会把两份清单的路径打出来，记得接上：

```bash
export FORGE_SEALED_PATHS=<ingest 输出的 sealed_paths.txt>
fence_check.py --denylist <ingest 输出的 denylist.txt> ...
```

### 按类型检索

要迭代某一类算子时，先问「这一类有没有现成参考可以当起点」：

```bash
# 按算子族
python3 scripts/reference_library.py find --library ~/.forge-kernel/lib \
    --family reduction --arch DAV_2201

# 只要有产物可取的（排除厂商实现）
python3 scripts/reference_library.py find --library ~/.forge-kernel/lib \
    --family matmul --with-artifact
```

族的划分沿用 `ascendc-performance-best-practices` 的分类体系 —— 那边按族
组织优化知识，这边按族找参考，两边对齐才查得通。族可以 `--family` 显式
指定，不给则从算子名推断，推断结果会标明来源（推错比没有更糟：
查的人会以为这一族里没有参考）。

### 最优按累计增益选，不按绝对耗时

多卡机器上卡与卡之间有系统性差异，不同时段的并发水平也不同 ——
**绝对耗时跨环境不可比**。而 `gain_over_parent` 是同一次配对测量里得到的
比值，在哪张卡上测都成立，这正是论文坚持同 job 配对计时的理由。

所以 `best_gen` 按累计增益（各代增益连乘）评选。整条链一个增益都没记时
才退回绝对耗时，并在 `best_basis` 里标明它只在同一环境内可比。

每代的 `device_id` 也记在 manifest 里 —— 多卡实验时，哪些数字可比要看
得出来。

### 挂到 cann-bench

bench 的 `tasks/level*/<算子>/` 提供任务定义（`proto.yaml`、`cases.yaml`、
`golden.py`、`desc.md`）。可以用 `--bench-task` 与 `--bench-baseline-us`
把 campaign 挂上去。

> **bench 本地给不出「最优版本」的 kernel。** `cases.yaml` 的
> `baseline_perf_us` 是**评分基准线**，`golden.py` 是精度标杆（torch/tf，
> 不是高性能实现）。论文里的 public best 是从**托管排行榜**读的，而且
> 明说它从不作为循环的输入。
>
> 所以 `bench_baseline_us` 在库里是**目标标注**，不参与 best 的评选 ——
> best 按实测时间选。真正能当参考的二进制只有两类：CANN 内置实现，
> 和这套系统自己产出的。

## 自我提升

```bash
python3 scripts/campaign.py promote --state ./forge_work/campaign_state.json \
    --artifact ./best_kernel --library ~/.forge-kernel/lib \
    --signature architecture_signature.json
```

> `--library` 是**必填**的。只在 state 里留一个路径等于没留：那个目录被
> 清掉之后下一代就拿不到上一代的产物，而自我提升的前提正是它还拿得到。

把本代产出封存成下一代的参考，重新分析、重新锚定目标。**源码不跨代边界，
知识库跨** —— 这是两条不重复的反馈边：提升抬高性能锚点，知识库降低够到
它的成本。

### 提升后必须重测噪声底

`promote` 会把状态标成「噪声底待重测」，`record` 在重测前会被拒。

新参考是另一个二进制，运行时特征不同，而噪声底论文定义的是**同 job
噪声底**、每次 campaign 实测。沿用上一代的值会让接受判据失准。

```bash
python3 scripts/noise_floor.py --from-results <新一代首次评测结果>
python3 scripts/campaign.py set-noise-floor --state <state> --value <实测值>
```

### 链条什么时候停

确认的阶段增益跌破噪声底时，`promote` 自动把状态置为 `stopped` 而不是
无谓地再开一代。预算耗尽也会停（`--budget`，可用 `set-budget` 提高）。

> 收敛时的停点是对硬件上限的**估计，不是证明** —— 也可能只是这个循环
> 找不到了。论文自己也是这么说的，不要把它当成上限已达的结论。

## 脚本工具

| 脚本 | 用途 |
|---|---|
| `scripts/campaign.py` | 状态机：init / record / reopen / set-reference-signature / set-noise-floor / set-budget / promote / status / ledger |
| `scripts/reference_library.py` | 封存产物库：ingest / register / promote / resolve / find / lineage / list / verify / drift |
| `scripts/noise_floor.py` | 实测噪声底 |
| `scripts/seal_reference.py` | 封存参考，产出禁读与禁词清单 |
| `scripts/reference_adapter.py` | 三种参考来源归一化 + 环境预检 |

## 依赖的外部能力

| 能力 | 来自 | 用途 |
|---|---|---|
| 配对计时与精度 | `ops-evaluation` 的 `evaluate_ops_direct.py` | 算 g(K,R) |
| 架构签名与停顿分解 | `forge-probes` 的 `signature_probe.py` | 机制核验 |
| 手段定价 | `forge-probes` 的 `lever_probe.py` | 停滞时定上界 |
| 转发围栏 | `forge-probes` 的 `fence_check.py` | 计时前拦转发 |
| 芯片规格 | `npu-arch` | 门禁锚点的硬件规格来源 |

> 注意：不要用 `ops-evaluation` 里的 `evaluate_ops.py`，它引用的
> `AscendBackend` 在仓内不存在。只用 `evaluate_ops_direct.py`。
