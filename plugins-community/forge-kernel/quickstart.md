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
步骤 4  采参考签名    msprof 取流水线占比与核数
步骤 5  产出课程表    arch agent 反推架构，写 milestone
步骤 6  逐行实现      coder agent 一次一行，一次一测
步骤 7  停滞 → 重开架构
步骤 8  达标 → 封存为下一代参考，继续
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

### 步骤 4：开 campaign

```bash
python3 $SK/campaign.py init \
    --operator add_layer_norm --arch DAV_2201 \
    --reference-kind builtin_aclnn --reference-id aclnnAddLayerNorm \
    --noise-floor 2.3 --target-gain 1.05 \
    --state ./forge_work/campaign_state.json
```

### 步骤 5：交给 Agent

启动 CLI，输入：

```
用 forge-kernel 优化 add_layer_norm，架构 DAV_2201，
参考用 CANN 内置 aclnn 实现，目标增益 1.05，
工作目录 ./forge_work
```

Orchestrator 会调度 `forge-arch` 反推架构并写 milestone，
再调度 `forge-coder` 逐行实现。

---

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
