# ForgeKernel 快速入门

## 概述

ForgeKernel 适用于**手上已有一个想追平或超越的参考实现**的场景：
CANN 算子库里的内置实现、别人写的自定义算子，或者你自己上一轮的产出。

它和"堆优化"的区别在于**参考只能测、不能读**。架构结论必须从 profiler
可观测的硬件行为反推出来 —— 这样得到的方法才能迁移到没有开源实现的算子上。

### 什么时候**不**用它

没有参考实现可对标时用 `ops-perf-evolution`（策略驱动，从自己的 baseline
枚举策略往上爬）。从零生成算子用 `cake`。

一句话分界：**`ops-perf-evolution` 在既定架构下调战术，`forge-kernel` 换架构本身。**

### 工作流

```
步骤 0  参数确认      算子、架构、参考来源、目标增益
步骤 1  封存参考      产出禁读清单 + 专有名词清单
步骤 2  适配与预检    三种参考来源归一化，环境预检
步骤 3  量噪声底      这一步不能跳过，也不能抄论文的经验值
步骤 4  采参考签名    msprof → signature_probe.py 提结构化签名
步骤 5  产出课程表    arch agent 反推架构，写 milestone
步骤 6  逐行实现      coder agent 一次一行
                      围栏 → 评测 → 签名 → record（机制进裁决）
步骤 7  停滞 → 定价 → reopen 重开架构（自动作废下游）
步骤 8  达标 → 封存为下一代参考 → 重测噪声底 → 继续
```

---

## 一、环境搭建

### 前置条件

- CANN Toolkit（配套版本见 [CANN Release Notes](https://www.hiascend.com/cann/document)）
- 已配置 NPU 设备
- Python 3 与 `PyYAML>=6.0`
- 一个可调用的参考实现

### 安装

**方式一：Plugin Marketplace**

```bash
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git
/plugin install forge-kernel@cannbot
```

**方式二：init.sh**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-community/forge-kernel
bash init.sh project claude     # 项目级
bash init.sh global claude      # 全局级
```

### 自检

```bash
cd cannbot-skills
python3 -m pytest plugins-community/forge-kernel/tests/ -q
```

全部通过说明纯逻辑部分（门禁校验、知识库准入、边界拦截、状态机）可用。
这一步不需要 NPU。

---

## 二、跑一轮 campaign

下面以 `add_layer_norm` 为例，参考用 CANN 内置 aclnn 实现。

### 步骤 1：封存参考

```bash
SK=.claude/skills/forge-campaign/scripts

python3 $SK/seal_reference.py seal \
    --artifact /path/to/reference \
    --op-name add_layer_norm \
    --work-dir ./forge_work
```

产出两份清单：

- `forge_work/sealed_paths.txt` → PreToolUse hook 读它，拦截读参考源码
- `forge_work/denylist.txt` → 知识库 lint 读它，拦截参考专有名词入库

参考本身是二进制（如内置实现）时会提示"未发现源码文件"，这是正常的 ——
那种情况天然满足封闭要求。

### 步骤 2：适配与预检

```bash
python3 $SK/reference_adapter.py prepare \
    --kind builtin_aclnn --op-name add_layer_norm --work-dir ./forge_work

python3 $SK/reference_adapter.py verify --spec ./forge_work/reference_spec.json
```

> **预检不过就停下来**，不要带病开跑。verify 会逐项打印 CANN 路径、
> npu-smi、torch_npu、设备状态和参考路径，失败时附环境快照 ——
> 整段输出可以直接回贴排障。

`builtin_aclnn` 依赖「空 OPP 目录使 aclnn 派发落到内置实现」这一行为，
**该行为尚未在真机验证过**。verify 会提示如何确认：跑一次评测后用 msprof
看 `kernel_details` 里出现的算子名是不是厂商内置实现的名字。

### 步骤 3：量噪声底

先跑一次配对评测拿到结果，再量：

```bash
python3 $SK/noise_floor.py --from-results evaluation_results.json
```

输出的"噪声底"就是接受判据的那条线。

> 论文报的 1%~5% 是**它们环境的实测值，不是常数**。定太低会把测量抖动
> 当成果收进来并污染知识库；定太高会把真实的小幅改进当噪声丢掉，
> 优化提前停在远低于硬件上限的地方。所以这一步必须自己量。
>
> 噪声底远超 5% 时先查环境：是否有其他进程占同一张卡、是否 DVFS 降频
> （对比 msprof 的 `Current Freq` 与 `Rated Freq`）。

### 步骤 4：采参考签名

```bash
PB=.claude/skills/forge-probes/scripts

# 先用 ops-profiling 采 msprof，再提成结构化签名
python3 $PB/signature_probe.py --archive-dir ./round_1 \
    --op-name AddLayerNorm --min-gm-bytes 8388608
```

这一步对应论文的 NCU pass。探针把七份 msprof CSV 归并成一份签名：
launch structure、流水占比、**停顿分解**、搬运量与复用倍数、暴露时间，
每个值带着它的来源通道（可直接写进 milestone 的 `trigger.channel`）。

> 它会先查测量可信度。降频、ICache 颠簸、停顿数据缺失任一成立，就
> **扣留全部架构假设** —— 不可信数据推出的架构错得很有说服力。

### 步骤 5：开 campaign

```bash
python3 $SK/campaign.py init \
    --operator add_layer_norm \
    --input-shape "B=8,S=4096,H=1024,fp16" \
    --arch DAV_2201 \
    --reference-kind builtin_aclnn --reference-id aclnnAddLayerNorm \
    --noise-floor 2.3 --target-gain 1.05 --budget 40 \
    --device-id 0 \
    --reference-signature ./forge_work/reference_signature.json \
    --state ./forge_work/campaign_state.json
```

> **`--reference-signature` 传步骤 4 对参考采的那份签名。** 机制行要拿它
> 做差（第 0 轮基线）；不给的话首轮构不成顺序证据，机制一律判为无法核验。
> 忘了可以事后补：`campaign.py set-reference-signature --signature <path>`。

> **`--input-shape` 是必填的。** 噪声底、架构签名、门禁锚点全都与 shape
> 绑定 —— 同一个 tiling 决策在不同 shape 下的 trigger 完全不同。一份
> state 跨 shape 复用会把不同 shape 的结果混进同一个 `best`，promote 时
> 封存的参考与锚定的目标就对不上了。每个 shape 开一份 state。

### 步骤 6：交给 Agent

启动 CLI，输入：

```
用 forge-kernel 优化 add_layer_norm，架构 DAV_2201，
参考用 CANN 内置 aclnn 实现，目标增益 1.05，
工作目录 ./forge_work
```

Orchestrator 会调度 `forge-arch` 反推架构并写 milestone，
再调度 `forge-coder` 逐行实现。每次计时前会先过转发围栏。

---

## 二之二、并行跑多个任务

插件不做调度。要并行就起多个任务，各自指定不同的卡：

```bash
# 任务 A
python3 $SK/campaign.py init --state ./work_a/state.json \
    --operator add_layer_norm --device-id 0 ... &

# 任务 B
python3 $SK/campaign.py init --state ./work_b/state.json \
    --operator matmul_nz --device-id 1 ... &
```

### 必须分开的三样

| | 为什么 |
|---|---|
| `--device-id` | 同卡互相干扰计时 |
| `--state` | 两个任务写同一份状态会互相覆盖 |
| 工作目录 | 封存清单与评测产物会互踩 |

### 可以共用的

产物库与知识库**可以也应该共用** —— 跨任务复用经验正是它们存在的理由。
两者的写入都加了文件锁，并发安全。

### Q：并行时噪声底怎么处理？

噪声底在什么并发水平下量的，就只在那个水平下成立。

所以**等所有任务都起来之后再量**。其余任务还没压上卡时量出来偏低；
之后满载运行，主机 CPU、PCIe、内存带宽都在争用，真实噪声更高。于是噪声
里的「改进」被当成真的收下，还会写进知识库和产物库污染后续各代 ——
**而整个过程没有任何报错**。

### Q：两个任务能不能共用一张卡？

不能。共享会互相干扰计时，配对计时的全部意义就是两个 kernel 处在相同
条件下。

### Q：promote 之后会换卡吗？

不会，`device` 原样保留。换卡之后参考签名、噪声底、绝对耗时全部不可比。

## 三、产出物

```
forge_work/
├── reference_spec.json        ← 参考定义（三种来源归一化后）
├── sealed_paths.txt           ← 禁读清单（hook 用）
├── sealed_manifest.json       ← 封存校验和，可查参考有没有被动过
├── denylist.txt               ← 参考专有名词（知识库 lint 用）
├── milestone.json             ← 架构决策课程表
├── campaign_state.json        ← 状态机 + 裁决台账
├── null_opp/                  ← 空 OPP 目录（builtin_aclnn 来源时生成）
└── gen0/, gen1/ ...           ← 各代产出
```

### 关键查看点

| 想看 | 看哪里 |
|---|---|
| 现在到哪一步了 | `campaign.py status` |
| 每次裁决和理由 | `campaign.py ledger` |
| 知识库有没有在帮忙 | `campaign.py ledger --recipes` |
| 哪些改动机制没核实 | `ledger` 里标注「机制未核实」的条目 |
| 机制是没查还是查了不成立 | `status` 的「机制不算数」一行，分证伪/无法核验两列 |
| 这次机制是跟谁比的 | `record` 输出的「做差基线」，或台账的 `compared_against` |
| 架构决策与门禁 | `milestone.json` |
| 课程表合不合规 | `milestone_validate.py milestone.json` |
| 为什么某个候选被拒 | ledger 里该条的 `reason` |
| 这个症状有没有现成经验 | `hwkb.py query --symptom "..."` |
| 这个方向是不是死路 | `hwkb.py nogo check --symptom "..."` |

---

## 四、常见问题

### Q：hook 拦住了我读参考源码，怎么绕过？

**不要绕过。** 这是整套方法的地基：架构结论必须从硬件行为反推，
读了源码结论就是抄来的，无法迁移到没有开源实现的算子上。

想知道参考怎么做的，正确途径是测它：看流水线占比、看核数、
做消融实验。拦截信息里给了具体做法。

确属误判（这个路径压根不是参考实现）时，改 `sealed_paths.txt`，
不要关 hook。

### Q：为什么 `evaluation_results.json` 里的 `pipeline` 和 `bottleneck` 是空的？

这是**有意为之**。配对评测用 `torch.npu.Event` 计时而不走 msprof，
原因写在它源码里：`acl.prof.init` 每进程只能调一次，多 session 并发采集
会互相抢资源导致计时不稳。

流水线占比要另外走 `ops-profiling` 的 msprof 采。写 milestone 的
`trigger.channel` 时注意区分：

```
架构签名 → msprof:PipeUtilization.csv:aiv_mte2_ratio
性能精度 → evaluation_results.json:comparison.speedup
```

把架构签名指到评测产物不会报错，只会永远读到空值 —— 门禁看起来在跑，
实际什么都没验证。

### Q：milestone 校验报了一堆错，能不能先跳过？

不能。在一份不合规的课程表上干活，产出的证据无法并入门禁，
等于白跑。

报错按规则分组，逐条补即可。最常见的是 V2（决策没有 trigger）——
那说明这条"决策"其实是凭经验的主张，应该挪到战术层，或者补上
支撑它的观测。

### Q：连续几轮都没进展怎么办？

`campaign.py` 会自动转入 `stalled` 状态，这时**交回 arch agent**，
不要自己继续调。

arch agent 此时该做的是消融、给手段定价、必要时**重开 M0 的架构决策**。
论文测出来，通用 agent loop 的天花板正是卡在"架构早早定死之后只在它
之下打转"。在错的架构下继续调，是把时间花在那个天花板底下。

### Q：`promote` 说收敛了，是不是到硬件上限了？

**不是。** 停点是对上限的一次**估计，不是证明** —— 也可能只是这个
循环找不到了。论文自己就是这么说的。

确需继续可以加 `--force`，但更有价值的做法是回头看开放问题清单：
那里列着 msprof 分辨不了、因而一直没被回答的问题，其中可能藏着
下一个数量级。

### Q：裁决为什么是 `accept_unverified`？改动不是明明更快了吗？

更快是真的，但没能确认**是你设计的机制起的作用**。

增益超过噪声底只说明总时间变短了。它可能来自你的机制，也可能来自别的
什么——两者在总时间上长得一模一样，但后者无法复用、无法解释、换个算子
就不灵。

`accept_unverified` 的改动会保留，但不计入 `best`，也不得作为知识库入库
依据。要让它算数，record 时补上 `--signature`（signature_probe 的输出）
与 `--milestone`，让机制行的停顿锚点真正被求值一次。

最常见的原因其实是**没传这两个参数** —— 工具不替你假设「没查就是通过」。

### Q：产物封存库和工作目录是什么关系？

工作目录（`forge_work/`）是**这一次** campaign 的草稿，随时可以清。
封存库（默认 `~/.forge-kernel/lib`）是**跨 campaign** 的产物仓库。

`promote` 带 `--library` 时，产物会被**复制**进库并记校验和与血统。
不带的话只在 state 里留一个路径 —— 工作目录一清，下一代就拿不到
上一代的产物，而自我提升的前提正是「上一代的产出，下一代还拿得到」。

```bash
reference_library.py lineage --library ~/.forge-kernel/lib \
    --operator add_layer_norm --arch DAV_2201 --input-shape "B=8,S=4096,fp16"
```

### Q：能不能直接拿 cann-bench 上最优的算子当参考？

**拿不到。** bench 本地提供的是任务定义：`cases.yaml` 的
`baseline_perf_us` 是**评分基准线**，`golden.py` 是精度标杆（torch/tf，
不是高性能实现）。论文里的 "public best" 是从**托管排行榜**读的，
而且明说它从不作为循环的输入。

所以能挂的是任务身份和目标值：

```bash
reference_library.py register --library ~/.forge-kernel/lib ... \
    --bench-task tasks/level2/add_layer_norm --bench-baseline-us 500.0
```

`bench_baseline_us` 在库里是**目标标注**，不参与 best 的评选 ——
best 按实测时间选。真正能当参考的二进制只有两类：CANN 内置实现，
和这套系统自己产出的。

### Q：机制行明明阈值成立，为什么判「平凡通过」？

因为那个字段跟上一轮（或参考）比**根本没动**。

阈值来自参考实测，所以「低于 4.2」的含义是「我比参考等得少」。但如果
这个 kernel 的 MTE2 停顿本来就只有 0.9，那它天然满足阈值 —— 改动做的
其实是别的事，机制行只是平凡地通过了。门禁看起来过了，实际什么都没验证。

所以 record 会拿本轮签名和基线做差，要求字段确实朝声称的方向移动
（默认至少 5% 相对变化）。方向从比较符推出：`<` 是要降，`>` 是要升。

平凡通过和「阈值不成立」归在同一桶：都说明增益另有来源，都该交回架构层。

### Q：`status` 说 `mechanism_refuted`，和 `mechanism_never_verified` 有什么不同？

差别在于**该做什么**，而且两者相反：

| | 成因 | 动作 |
|---|---|---|
| `mechanism_never_verified` | 没传 `--signature` / `--milestone` | 补参数重跑，课程表不用动 |
| `mechanism_refuted` | 传了，但机制行求值为假 | **改课程表的机制假设** |

后者是个很强的架构信号：改动确实更快，但它声称的机制没有发生 ——
说明**对参考行为的解释错了，而实现碰对了别的东西**。这时去补参数重跑
是白费一轮，参数本来就传对了。

顺带一提，这种情况下实现其实找到了某个有效但还没被解释的东西，
把它弄清楚往往比沿原假设继续推进更有价值。

### Q：为什么连续几个编译错误没有触发架构重入？

因为那是实现问题，不是架构问题。编译失败、精度不达标、评测缺失都计在
`implementation_failures` 里，不消耗架构重入机会。

只有「有有效测量但增益不足」才计进 `stagnation` —— 那才是架构层该关心的
停滞。混在一起算，连续三个编译 bug 就会白白浪费一次重开架构的机会。

### Q：重开 M0 的某个决策后，之前的实现还算数吗？

不算，而且**不要自己记哪些要作废**，用：

```bash
campaign.py reopen --state <state> --milestone milestone.json --decision D0.1
```

它按 `depends_on` 算反向闭包，把所有直接或间接依赖该决策的台账记录标为
作废，并重算 `best`。

漏掉一条的后果很严重：新架构上还挂着建立在旧假设上的实现，测出来的增益
归因就是错的 —— 而这恰恰发生在整个流程最关键的那一步上。

### Q：promote 之后为什么不让我 record 了？

因为噪声底还是上一代的。新参考是另一个二进制，运行时特征不同，而噪声底
论文定义的是**同 job 噪声底**、每次 campaign 实测。

```bash
noise_floor.py --from-results <新一代首次评测结果>
campaign.py set-noise-floor --state <state> --value <实测值>
```

### Q：探针说「测量不可信，已扣留架构假设」，能不能忽略？

不能。扣留的三种情形都会让架构推断系统性出错：

- **降频**：各段被按不同比例拉长，一个计算受限的 kernel 会看起来像搬运受限
- **ICache 缺失率高**：取指开销混进各段占比
- **缺 `ResourceConflictRatio`**：停顿数据是架构判断的一半依据

注意第三条：这份数据是**上板独有，仿真拿不到**。用仿真数据做参考分析，
这一整段必然缺失。

### Q：定价工具说某个方向「测不出来」，是不是该换个测法？

不是，那就是结论。收益落在噪声底以内意味着这个方向**没有可测量的空间**。
再投三轮也出不来 —— 定价的主要价值正是提前砍掉这类投入。

**所有单项都测不出来时**，说明当前架构下这一层没有余量了，
应当回 M0 改架构，而不是继续在这层找。

### Q：报 `from evaluate import AscendBackend` 失败

你用成 `evaluate_ops.py` 了。它引用的 `AscendBackend`、
`compare_performance_advanced`、`evaluate_correctness` 在仓内都不存在。

改用同目录的 **`evaluate_ops_direct.py`**。

---

## 五、可用技能

| Skill | 用途 |
|---|---|
| `forge-reference-analysis` | 症状 → 架构决策的翻译字典，开放问题清单 |
| `forge-milestone` | 决策图 schema 与十条门禁校验 |
| `forge-hwkb` | 硬件知识库：准入、检索、边界 lint、死路表 |
| `forge-campaign` | 封存、预检、噪声底、状态机、自我提升 |
| `forge-probes` | 探针套件：架构签名提取、手段定价、转发围栏 |
| `npu-arch` | 芯片规格与架构差异（真源） |
| `ops-profiling` | msprof 采集与解析 |
| `ops-evaluation` | 配对评测（用 `evaluate_ops_direct.py`） |
| `ascendc-performance-best-practices` | 战术层优化实践，按算子族 |

---

## 总结

1. 参考只能测、不能读 —— hook 在工具调用层强制，不靠提示词自觉
2. 噪声底自己量 —— 它决定什么改动算数，抄不得
3. 决策必须有 trigger —— 说不出观测到什么就要做它，这条决策不成立
4. 架构层和实现层职责互斥 —— arch 不写代码，coder 不改课程表
5. 停滞时回头改架构，不要在错的架构下硬调
