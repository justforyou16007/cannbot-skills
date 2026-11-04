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
# 中转目录在工作区外：STAGE=/cache/forge-kernel/ref_stage/add_layer_norm
python3 scripts/seal_reference.py seal \
    --artifact "$STAGE" --op-name add_layer_norm \
    --work-dir ./forge_work \
    --public-headers /usr/local/Ascend/cann-8.5.0/aarch64-linux/ascendc/include
```

产出两份清单，交给两个独立于提示词的执行点：

- `sealed_paths.txt` → PreToolUse hook，在工具调用层拦截读参考源码**与配置**
- `denylist.txt` → 知识库 lint，拦截参考专有名词混进可复用经验

**封存范围按单个算子挑，不要封整棵安装树。** 做法是把本算子相关的文件
软链到一个中转目录，再封这个中转目录 —— 封存会对软链做 `resolve()`，
清单里落的是真实绝对路径。封整棵树会同时坏两件事：别的算子里的
`class Kernel` 被当成本参考的专有名词（围栏全是误伤），以及禁读清单长到
几千条后 hook 的比对开销压在每次工具调用上。文件数超过 500 时 seal 会告警。

**中转目录放在工作区之外**（如 `/cache/forge-kernel/ref_stage/<算子名>`，
本机没有 `/cache` 时用 `~/.cache` 下的目录）。边界 hook 对 `Grep`/`Glob`
要判两个方向：目标在封存目录内，以及目标是封存目录的上层 —— 后者递归搜
下去同样能读到参考源码。所以中转目录落在工作目录下时，`Grep path=.`
（搜自己的候选代码）会被一并拦住。位置选错时 seal 会当场警告。

**配置文件（算子的 `.json`）和源码一样禁读。** 它把参考实现了哪些分支、
每个分支用什么 tiling key 全列了出来，照抄它连代码都不用读。

**专有名词会自动减掉公开接口名。** 判据是扫 `ascendc/include`，自动找不到
时用 `--public-headers` 指过去。不做这步减法，`DataCopyPad` 这类接口名会
进禁用名单，于是任何正常候选都被拦 —— 最后的结局是围栏被整条豁免掉。

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
> 这个行为**必须在本机确认一次**。做法：令 baseline 与 evolved 都指向本
> 参考跑一次自评，用 msprof 看 `op_summary` 里的算子名是不是厂商内置实现
> 的名字；确认后回写证据：
>
> ```bash
> python3 scripts/reference_adapter.py verify \
>     --spec ./forge_work/reference_spec.json \
>     --confirm-dispatch --evidence "msprof op_summary 中 Op Name=AddLayerNorm"
> ```
>
> 这不是留记录，是开 campaign 的前提 —— `campaign.py init` 会读
> `dispatch_verified`，为 false 时拒绝。没确认派发，第 0 轮基线测的可能
> 根本不是内置实现，后面每一轮加速比都是对着一个未知对象算出来的。

### 3. 量噪声底

```bash
python3 scripts/noise_floor.py --from-results evaluation_results.json --method cv
```

论文报的 1%~5% 是**它们环境的实测值，不是常数**。定太低会把抖动当成果
收进来，还会污染知识库；定太高会把真实的小幅改进当噪声丢掉，优化提前停在
远低于硬件上限的地方。

首选直接读评测结果里的 `cv_pct`（同一 job 内的离散度，正对应论文说的
same-job noise floor）；没有时才回退到多次结果之间的比值离散。

**只能用跑成功的评测来量。** 崩溃的评测写进文件的是 `time_us=-1` 和
`cv_pct=0.0`，那个 0.0 和「环境极其干净」在形态上一模一样。脚本现在直接
拒绝 `gating=failed` 的结果和 cv 全为 0 的结果：噪声底定成 0 意味着任何
正增益都被判为真实改进。

**自评场景显式传 `--method cv`。** 评测在自评时把 speedup 钉成 1.0，
比值离散法（`--method spread`）必然输出 0.00% —— 那不是环境干净，是这个
测法什么也没测到。脚本会认出这种结果并报错。

**cv 法不含跨进程的位置偏差。** 评测总是先跑 baseline 子进程再跑 evolved
子进程，后者继承频率/缓存劣势，实测 evolved 侧恒慢约 2.7%，方向恒定、
多测几次消不掉。所以 1.05 的名义目标实际需要 5%+ 的真实改进。

### 4. 跑 campaign

```bash
python3 scripts/campaign.py init --operator add_layer_norm \
    --input-shape "B=8,S=4096,H=1024,fp16" --arch DAV_2201 \
    --reference-kind builtin_aclnn --reference-id add_layer_norm \
    --reference-spec ./forge_work/reference_spec.json \
    --noise-floor 2.95 --target-gain 1.05 --budget 40 \
    --device-id 0 \
    --state ./forge_work/campaign_state.json

python3 scripts/campaign.py record --state ./forge_work/campaign_state.json \
    --result evaluation_results.json --decision D0.1 \
    --signature architecture_signature.json --milestone milestone.json \
    --recipe R-2201-001 --symptom "MTE2 占比过高" \
    --nogo-kb-root <forge-hwkb>/references
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

### 机制没落地时，该修的是实现还是课程表

上游的分流规则：

> 声称无冲突布局而冲突仍高 ⇒ **学生没做到（退化回退）⇒ 加知识库卡片，
> 不要改课程表**；只有「参考侧该字段本来就不在声称的位置」才是课程表缺陷

判据是**参考快照**，不是本轮读数。问的是「这条行声称的位置，和参考实际坐
的位置差多远」—— 而不是「参考满不满足这条行的阈值」。后者在最常见的写法上
就错：课程表写 `< 4.2`、参考实测正好 4.2（要求实现追平参考），`4.2 < 4.2`
为假，于是一条完全正当的行会被判成「读错了参考」。

| 情况 | 归因 | 下一步 |
|---|---|---|
| 声称的位置在参考附近（差在 1.25 倍带内） | `implementation` | 补一张经验卡片，**不要改课程表** |
| 声称的位置落在带外 | `curriculum` | 重开那条决策 —— 实现正被要求去追一个参考自己都达不到的数 |
| 参考侧没这个读数，或这条行解析不成可核验的断言 | `implementation` | 见下 |

带宽用的是 `MECHANISM_BAND_FACTOR`（1.25），和验收线的带宽纪律**同一个
常数** —— 两处对「差多远算差太远」的判断不该各有一套。

**判不了时为什么偏向实现失败**：两种误判的代价不对称。

| 误判 | 后果 |
|---|---|
| 判成课程表缺陷，其实是实现 | 架构层被连续叫回来重写课程表，而下一轮实现还是同一个退化回退 —— 改对的东西被改坏、改坏的东西没人碰。**这是一个不会响的失败** |
| 判成实现失败，其实是课程表 | 实现失败计数爬到阈值 → 状态转 `blocked` → **停下来交给人**。这是一个会响的失败 |

这条分流**不依赖任何可选输入**。此前只有 `structure_probe` 能分流，而
`--structure-report` 是可选参数 —— 没传时 `refuted` / `vacuous` 一律落进
`streak_refuted`，也就是默认判「课程表解释错」，正是这条规则要防的方向。
参考快照在 `init` 就必填，所以分流总能做。

### 三道「这个数算不算数」的前置条件

它们和机制核验是不同性质的东西。机制核验问「变快的原因是不是设计的那个」；
这三道问的是更靠前的问题 —— **这个百分比到底是不是这份 kernel 在这张卡上
挣来的**。所以它们不走 `accept_unverified` 那条路（那条路的含义是「数是真的、
解释不清」），而是直接压住 `counts_toward_best`。

| 前置条件 | 怎么查 | 不过时 |
|---|---|---|
| 没有转发给参考 | `--fence-report`（`fence_check.py --json` 的输出） | 阻断档命中 → 回滚 + 归因实现；**没传报告 → 不抬高 best** |
| 设备执行占得住墙上时间 | `signature.launch.task_duration_us` ÷ 结果的 `time_us` ≥ 20% | 不抬高 best |
| baseline 不是缓存的 | 结果的 `measurement.baseline_from_cache` | 直接拒收这份结果 |

围栏裁决有**三**态，不是两态：

| 态 | 含义 | 对 best |
|---|---|---|
| `clean` | 阻断档与通道档都空 | 不压 |
| `clean_with_channels` | 阻断档空，但有未说明的通道（`dlopen` / `ctypes` / 子进程 / 动态 import / 任意 aclnn 接口） | 不压 |
| `violated` | 阻断档命中 | 回滚 |

第三态存在的理由：阻断档那七条规则**全部由算子名派生**，而转发不需要写出
名字 —— `ctypes.CDLL("libopapi.so").aclnnXxx` 里那个名字是运行时拼出来的
字符串。把「阻断档为空」读成「围栏干净」，等于让那条路永久免检。

通道档不压 best（正当实现里也会出现这些构造），但台账里留下记录，要求
逐条说明通向哪里。

**围栏为什么要有牙齿**：脚本本来就有，文档也写明它排在任何计时之前 ——
但做裁决的 `record` 此前完全不看它。于是这道纪律是纯自愿的。而它守的东西
很具体：候选直接调厂商实现时，加速比、精度、所有门禁全过，一行 kernel 都
没写；自我提升场景更隐蔽 —— 第 n 代转发给第 n−1 代，链条可以空转很多轮，
每轮都「没有退化」。围栏是静态扫描，不花机时，所以每轮都该跑。

**设备占比为什么要对账**：两个数早就都在手，但此前没有任何地方把它们放在
一起看过。设备执行只占墙上时间一小部分时，那个百分比量的主要是 host 侧的
下发、排队与同步；而机制核验读的是 msprof 的**设备侧**通道。两者量的东西
几乎不相交时，「增益超噪声底 + 机制已核实」就是两个互不相关的事实被钉在
一起 —— 那个百分比来自 host，那个机制结论来自 device。

**缓存的 baseline 为什么直接拒**：配对计时的全部纪律建立在「两侧在同一个
job 里、位置互为镜像」之上 —— burn-in 把器件加热到稳态，A-B-B-A 抵掉位置
偏置（实测 **2.7%**，方向恒定、多测不收敛）。baseline 一旦来自上一次 job
的缓存，这两个机制同时失效，而且是**静默**失效：字段、cv 全都正常，只是
那个数来自另一个温度、另一个位置。而目标增益只有 5% 这个量级。

## 重开架构决策

```bash
python3 scripts/campaign.py reopen --state ./forge_work/campaign_state.json \
    --milestone milestone.json --decision D0.1 --reason "拓扑决策有缺陷" \
    --price "8.4% via lever:ub_resident" --lever-store ./forge_work/levers.json

# 价格也可以来自一张已认证的知识库卡片 —— 那就省掉一次实测
python3 scripts/campaign.py reopen --state ./forge_work/campaign_state.json \
    --milestone milestone.json --decision D0.1 --reason "拓扑决策有缺陷" \
    --price "8.4% via kb:R-2201-001" \
    --kb-root ../forge-hwkb/references
```

论文 §3.3：追加型补丁原地恢复，**基础型补丁从最早的依赖决策重启** ——
这正是 M0 的拓扑决策被「重开」而不是「在它底下调」的实现方式。

`reopen` 按 `depends_on` 算反向闭包，把所有（直接或间接）依赖该决策的
台账记录标为作废，并重算 `best`。

靠人记哪些要作废是不牢的。漏掉一条，新 M0 上就还挂着建立在旧假设上的
实现，归因直接错掉 —— 而这恰恰发生在整个流程最关键的那一步上。

### 重开之后从哪一份产物接着做

`reopen` 把整条下游的台账作废了，但**工作树还在原地** —— 上面带着那些刚被
作废的改动。编码层如果从当前树接着改，它就是在一个已经被判定建立在错假设上
的实现之上继续叠加，而归因会把后面的结果算到新架构假设头上。

所以 `reopen` 现在直接给出回退目标：**作废范围之外、最后一个机制已核实的
产物**。

| 条件 | 为什么 |
|---|---|
| 作废范围之外 | 范围内的那些正是被判定不可信的 |
| `counts_toward_best` 为真 | 只是变快了的产物自己就解释不清 |
| 有 `artifact_sha` | 回退目标必须是一份能被找回来的东西 |

找不到就说「从参考基线重新开始」。**不编一个看起来合理的回退点** ——
一个错的回退目标比没有目标更糟。

### 修订本身要进台账

重开是一次**课程表修订**，所以记成 `patch` 类条目而不只是一行 `reason`：

```json
"patch": {
  "locus": "D0.1",
  "direction": "分段切分轴判错了",
  "milestone_sha_before": "a1b2c3d4e5f6",
  "price": "8.4% via lever:ub_resident",
  "outcome": "pending"
}
```

不记的话，「这条决策被改过、从哪个课程表指纹改到哪个、结果如何」在台账里
完全无痕。`status` 据此列出**已试过的课程表方向** —— 每一个都花掉了一次
重开的代价（作废整条下游、best 归零），不列出来下一代会把同一个方向原封
不动再试一遍。

`outcome` 在下一次抬高 best 时回填为 `accepted`。必须回填：否则一次**成功的**
修订也会永远停在 `pending`，于是 status 拿它去劝退下一代 —— 一张只记投入
不记产出的账，用起来比没有更糟。

### 没有价格就没有重开

重开要作废整条下游的实现、把 `best` 归零、三个计数器清零 —— 这是整套流程
里代价最大的单个动作，理由不能是一句自由文本。`--price` 的形状是
`<Δ%> via <来源>`，来源四种：

| 来源 | 核对方式 |
|---|---|
| `lever:<手段名>` | 去 `levers.json` 里找这个手段的实测价格 |
| `kb:<卡片 id>` | 去知识库找那张卡片，**并要求它自己过准入校验**，且记录幅度 ≥ 声称的 Δ |
| `probe:<路径>` | 探针读数 |
| `ablation:<名字>` | 消融实验 |

`kb:` 这条路是「读知识库能省一次评测」的落点 —— 但它只在卡片真被核对过
时才成立。上游那句话很直接：**一条转述过来的处方不是价格**。这里曾经无
条件放行，附一句「请确认该卡片确实带探针认证」—— 把判定交给了那个正想
重开的人，而他没有任何动机去查。

四种来源都要先过噪声底：Δ 落在测量抖动里的「价格」不算价格。

## 死路表自动登记

`record` 判为回滚时：

- 给了 `--nogo-kb-root` → 直接写入该知识库的死路表
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

### 知识欠账是自动算的，`promote` 会拦

```bash
python3 scripts/campaign.py harvest --state ./forge_work/campaign_state.json
```

列出本代「该入库而还没入库」的条目。两类：

| 类 | 判据 | 为什么它欠一笔 |
|---|---|---|
| `admissible` | 台账判了 `kb_admissible`（增益超噪声底**且**机制已核实）而没有任何 `recipes_invented` | 这条经验的代价已经付过了，只差写下来 |
| `tripwire` | 一个决策烧掉 ≥ 6 轮才拿到接受判定 | 花得多说明有东西没被预见到，而那个发现只存在于那次会话的上下文里 |

为什么要自动算：知识沉淀排在流程第 11 步，也就是一次长 campaign 的末尾 ——
正是上下文最紧、最容易被跳过的时刻。靠提示词提醒的事，到那个时刻不会发生。
**本仓就是证据**：整套 dav2201 流程建完，库里 1 张卡。

`harvest` 的输出带着写卡片要用的读数（症状、机制结论、机制行、签名文件
路径）。清单要是还得让人回头翻台账找数，到那一步就没人写了。

欠账未结时 `promote` 拒绝执行。这道闸的位置是算过的：

- 排在**确认复测之后** —— 读数本身不可信时，先让人写一张以它为据的卡片
  比不写更糟：那张卡片会被别的 campaign 当成已核实的经验用
- 排在**正式封存之前** —— promote 把 `best` 归零、噪声底置为待重测、产物
  封成下一代的参考。台账还在，但产出这些结论的会话上下文已经没了，
  「增益超噪声底且机制已核实」背后的推理过程过了这一刻只剩一行 `reason`

确实不值得入库时加 `--allow-unharvested`：它会写进台账，而不是悄悄消失。

---

## 课程表与台账对账

这两份文件此前**从不互看**：课程表自己写 `status: PASS`，台账自己记
`mechanism.status`。于是会出现一个很具体的情形 ——

```
课程表：M1 机制行 status=PASS，closed_by 指着 job 7
台账：  那一轮的 mechanism.status=refuted
```

读课程表的人（下一轮的编码层、做 harvest 的人、看 status 的人）得到的是
「这条机制已经核实」，而唯一做过核验的那次说的是相反的话。课程表是三个产物
里**唯一会被下一轮当作前提**的那个，所以这个不一致不是记错一笔账，是让后续
所有决策建立在一个假前提上。

`record` 据此在台账里记一条 `gate_coupling`，并说明**该修哪一侧** ——
那完全由归因决定：

| 归因 | 该修哪一侧 |
|---|---|
| `implementation` | 修实现，课程表原样不动（那条 PASS 是对参考的正确读法） |
| `curriculum` | `reopen` 那条决策，PASS 跟着一起作废 |

**只报告，不改文件。** `record` 不该去写课程表 —— 那是架构层的产物，
`AGENTS.md` 的禁止项里写着编排层与编码层都不许改。

---

## 墙钟预算

```bash
python3 scripts/campaign.py init ... --wall-hours 48
python3 scripts/campaign.py set-budget --state <state> --max-wall-hours 72
```

次数预算挡不住一种很常见的消耗：**每轮都很快、但轮数没涨而时间在走** ——
构建失败重试、环境排障、等卡。那些都不消耗 `used`，于是一个跑了两天的
campaign 看起来预算还很充裕。

没有常驻进程，所以只能按 `created_at` 算差。这是个近似：它把中间停机的时间
也算进去了。取这个近似而不是不做，理由是它的误差方向是**保守**的（报得比
实际机时多，于是提前提醒），而反过来会漏。`status` 与停止原因里都写明
「这是机时的上界，不是机时本身」。

**收敛那条分支只告警，不拦。** 一个该停的 campaign 必须能记下「我收敛了」——
拦住它的后果比丢几条经验更糟：状态机会卡在 `implement`，之后所有停止条件
的判断都从一个错的状态出发。

## 指定设备

一个演进任务用一张卡，卡号在 `init` 时必填：

```bash
python3 scripts/campaign.py init --state ./forge_work/campaign_state.json \
    --operator add_layer_norm --input-shape "B=8,S=4096,fp16" \
    --arch DAV_2201 --device-id 3 ...
```

要并行就**在仓库外起多个任务**，各自指定不同的卡与不同的工作目录。
插件不做调度。

### 两条规矩

**一个任务一张卡。** 两个任务共用一张卡会互相干扰计时，而配对计时的
全部意义就是两个 kernel 处在相同条件下。

**代际之间不换卡。** `promote` 保留 `device`，不重新分配。换卡之后参考
签名、噪声底、绝对耗时全部不可比，而这些不可比是**静默的** —— 数字照常
产出，只是错了。

### 并行跑多个任务时注意

| 要分开的 | 为什么 |
|---|---|
| `--device-id` | 同卡互相干扰计时 |
| `--state` | 两个任务写同一份状态会互相覆盖 |
| 工作目录（`forge_work/`） | 封存清单、评测产物会互踩 |

**可以共用的**：产物库与知识库。它们的写入已加文件锁，并发安全 ——
事实上共用才是对的，跨任务复用经验正是知识库存在的理由。

> 噪声底在什么并发水平下量的，就只在那个水平下成立。并行任务数固定时，
> 应当在**所有任务都起来之后**再量噪声底；否则量出来偏低，之后满载运行
> 时噪声里的「改进」会被收下，而整个过程没有任何报错。

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
| `scripts/campaign.py` | 状态机：init / record / confirm / audit / reopen / set-reference-signature / set-noise-floor / set-budget / harvest / promote / status / ledger |
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
