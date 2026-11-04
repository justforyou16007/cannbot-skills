---
name: forge-probes
description: ForgeKernel 探针套件，把论文用到的 NCU pass、timeline probe、手段定价与转发围栏对应到昇腾的 msprof 工具链。当需要从 msprof CSV 提取 Ascend C 算子的结构化架构签名、判断测量是否可信、给优化手段定价以确定收益上界、分离耦合效应、或在计时前检查候选是否转发到参考实现时使用。
---

# 探针套件

## 这个技能解决什么

论文的仪器是 NCU、CUPTI、CUDA events。跨到昇腾要换成 msprof，但不是
一一对应 —— 有的完整对应，有的只能部分替代，有的根本没有。

本技能提供三个探针，并把对照关系与**缺口**明确写下来。

| 论文的仪器 | 这里的工具 | 对应程度 |
|---|---|---|
| NCU pass | `signature_probe.py` | 完整（且 msprof 多给了搬运量与 L2 命中率）|
| 角色暴露账 | `signature_probe.py` 的 `role_exposure` | 完整，且比上游干净（AIC/AIV 各有独立 PMU，闲置比例能按角色分开算）|
| 逐核负载 / 掉队核 | `signature_probe.py` 的 `core_balance` | 完整（`per_core_cycles.csv`，2× 判据对齐上游 `uneven_spans`）|
| 结构核验（声称的结构有没有建出来）| `ops/ops-simulator` 的 `structural-verification.md` | 部分 —— 静态 IR 与仿真指令表可判搬运路径与同步成对，**最终助记符反汇编不可用**（厂商关闭，见 waiver）|
| timeline probe | `signature_probe.py` 的 `exposure` + `core_balance` + `role_exposure` | **部分** —— 四个子能力里三个可答（总暴露量 / 掉队核 / 残差归谁）；**ramp-plateau-drain 是真盲区**（PM 采样上限 100 Hz = 10ms，而 kernel 是百微秒级，差约 50 倍），已按 waiver 格式写明八条已试通道 |
| 手段定价 / 裁决探针 | `lever_probe.py` | **部分** —— 只有「单项加上」这条腿，没有「全上各减一项」那条。于是上游的 `coupled` 类（单独无用但别人依赖它）读不出来，还会被判成「不要再投」。见 `references/instrument-map.md` |
| 转发围栏 | `fence_check.py` | 完整 |
| CUDA-graph replay、指令级时间线、最终助记符反汇编 | 无 | **没有对应物**。前两项作开放问题处理；第三项是厂商主动关闭（`llvm-objdump` 缺指令表、设备侧 `-S` 被拒、管线无 `.s` 阶段），已写 waiver 并给出替代通道 |

完整对照见 [`references/instrument-map.md`](references/instrument-map.md)。

## 四、结构核验：`structure_probe.py`

回答一个和性能无关的问题：**声称的结构到底有没有被构建出来** —— 而且在
任何性能数出来之前就能答。

```bash
# 源码层（不需要 CANN 工具链）
python3 scripts/structure_probe.py --candidate ./cand \
    --assert "buffer_depth:2" --assert "flag_paired" \
    --output ./forge_work/structure.json

# 连 IR 层一起查（需要 ccec；1=GM、6=UB）
python3 scripts/structure_probe.py --candidate ./cand \
    --assert "operand_space:addrspace(6)" --arch dav-c220-vec

# 看支持哪些断言
python3 scripts/structure_probe.py --list-assertions
```

报告传给 `campaign.py record --structure-report`。

### 为什么需要它

停顿锚点有个结构性弱点：**它能被别的改动顶下去**。减小 tile、改对齐、换
block 映射都会让 `mte2_wait_ratio` 降。于是一个写成单缓冲回退的实现，可能
因为本轮顺手改了 tile 而让停顿恰好落进带内 → 判「机制已核实」→ 这个解释
被封进产物库和知识库，**而双缓冲从未存在**。

反方向同样会错：机制确实建出来了，但当前 shape 下它不是瓶颈，停顿不动 →
判「平凡通过」→ 一个正确的架构决策被当成「增益另有来源」。

结构在不在是二值事实，顶不下去。所以核验顺序是**结构先、停顿后**，而且
结构的结论优先 —— 上游那条规则：**未落地的结构优先于改验收措辞**，
把阈值改宽不会让一个「不在」变成「在」。

### 归因方向

结构不在 ⇒ 判 `refuted` 且归因 **implementation**：该修实现、或给实现补一张
经验卡片，**不要改课程表**。归到课程表那一侧的后果很具体：连续三次把架构层
叫回来重写课程表，而下一轮实现还是同一个退化回退。

### 两层通道与各自的边界

| 层 | 能判什么 | 需要什么 |
|---|---|---|
| 源码层 | 缓冲深度、同步成对、某个构造在不在 | 无 —— 候选自己的源码完全可读（封存 hook 只拦参考） |
| IR 层 | 操作数所在存储（`addrspace(1)`=GM、`addrspace(6)`=UB） | `ccec` + `llvm-dis` |

IR 在指令选择与寄存器分配**之前**，所以证不了「寄存器没溢出」「调度深度
是多少」。那些要走仿真通道，见 `ops/ops-simulator` 的
`structural-verification.md`。

> **拿不到工具链时报错，不放行。** 「查不了」不等于「查过且通过」——
> 一条查不了的结构断言被当成通过，比没有这条断言更糟：它会让人以为
> 核验过了。要显式接受这个缺口才用 `--allow-missing-toolchain`。

## 贯穿三个探针的一条原则

论文 Appendix C 把 **agent 自评的探针单列为一类 bug 来源**。

落到工具上是一个具体要求：**每个探针都要能回答「这个数没采到时会怎样」**。

把「没数据」读成「没问题」是这类工具最常见、也最难发现的失效方式 ——
停顿数据缺失时若按 0 处理，读出来就是「完全没有停顿」，一个本该引发
追查的信号被反转成了健康信号。

三个工具各自的做法：

- `signature_probe.py` —— 缺失标 `present=false` 而非 0；停顿数据缺失时
  **拒绝输出架构假设**。另外区分「没采到」与「解析不动」：前者正常（空值、
  `N/A` 是 msprof 明确的不适用标记），后者**阻塞** —— 有内容而读不动说明
  那一列的量纲或格式和本工具以为的不一样，而同一份 CSV 里别的列照样被读走
  了，于是签名里混着一个读错了的字段，**而它伪装成一个缺失字段**
- `lever_probe.py` —— 缺单项测量时标 `incomplete`，不硬算耦合
- `fence_check.py` —— 扫不到源文件时报错，不返回「通过」

---

## 一、架构签名：`signature_probe.py`

把七份 msprof CSV 归并成一份结构化签名，每个值带上它的来源通道
（milestone 的 `trigger.channel` 可直接引用）。

```bash
# msprof 原始目录（实机会多一层 PROF_GROUP_*，不用自己剥）
python3 scripts/signature_probe.py --prof-dir ./prof_out --op-name AddLayerNorm

# 同一目录下有多次采集时，必须指定用哪一次
python3 scripts/signature_probe.py --prof-dir ./prof_out \
    --op-name AddLayerNorm --group PROF_GROUP_20261030_171959

# 归档目录（msprof_perf_summary.py 产出）
python3 scripts/signature_probe.py --archive-dir ./round_1 --op-name AddLayerNorm

# 给出理论最小搬运量后，额外算出数据复用倍数
python3 scripts/signature_probe.py --archive-dir ./round_1 \
    --op-name AddLayerNorm --min-gm-bytes 8388608 --json
```

**多次采集并存时探针报错，不替你挑。** 它曾经排序取第一个，于是时间最早、
可能只采到两个 metric 的那次被静默选中 —— 缺 metric 的签名里停顿数据是
缺失的，读起来像「完全没有停顿」。报错会列出每组各采到几个 metric。

**冷启动行自动丢弃。** 同一个算子在一次采集里有十来行，头几次核执行时间
超过任务总时长（物理上不可能）。实测冷启动 408.52us vs 稳态 233.16us，
**高 75%**，而它看起来只是个正常数字 —— 这个数会成为整个 campaign 的参考
基线。探针按「丢掉自相矛盾的行，再取中位数附近那一行」挑，过程写进签名的
`row_selection` 字段。需要更严的稳态起点时用 `--steady-from <Task ID>`。

输出分六段：测量可信度、launch structure、pipeline 占比、停顿分解、
搬运量与带宽、暴露时间；之后是**架构假设**与**开放问题**。

### 可信度是硬闸门

下列任一成立，探针**扣留全部架构假设**：

| 情况 | 为什么致命 |
|---|---|
| DVFS 降频 | 各段被按不同比例拉长，一个计算受限的 kernel 会看起来像搬运受限 |
| ICache 缺失率 >15% | 取指开销混进各段占比 |
| `ResourceConflictRatio` 缺失 | 停顿数据是架构判断的一半依据，缺了只能瞎猜 |
| 整次采集没有稳态行 | 所有行都是核执行时间 > 任务时长的冷启动，耗时虚高且不自洽 |

扣留而不是降级，是因为不可信数据推出的架构**错得很有说服力** ——
数据看着正常，错误极难被发现。`campaign.py init` 也会拒绝用
`trustworthy=false` 的签名当第 0 轮基线。

### 数据复用倍数

`--min-gm-bytes` 传算子语义要求的理论最小读取字节数，探针据此算出：

```
复用倍数 = 实测 GM→UB 搬运量 ÷ 理论下界
```

接近 1 说明几乎无冗余；明显大于 1 说明同一份数据被反复搬入。这是对
参考 tiling 策略最有力的量化约束 —— 虽然仍不唯一确定 tile 形状。

---

## 二、手段定价：`lever_probe.py`

把某个环节**整个删掉**（哪怕结果因此是错的）所省下的时间，就是优化它
能拿到的**收益上界**。

```bash
python3 scripts/lever_probe.py add --store ./forge_work/levers.json \
    --baseline-us 456.5 --noise-floor 2.3
python3 scripts/lever_probe.py add --store ./forge_work/levers.json \
    --lever drop_output_store --time-us 429.1
python3 scripts/lever_probe.py add --store ./forge_work/levers.json \
    --lever drop_output_store --lever relax_align --time-us 430.0
python3 scripts/lever_probe.py analyze --store ./forge_work/levers.json
```

### 定价的主要价值是「决定不做什么」

优化循环里最贵的不是方向错了，而是在一个天花板只有 2% 的方向上投三轮
迭代。先定价，这类投入可以提前砍掉，代价只是一次测量。

收益落在噪声底以内的手段会被判为「测不出来」，并列进 `not_worth_pursuing`。

### 耦合分析

两项各省 5%，一起上只省 6% —— 说明它们在抢同一份时间，真实可得是 6%
而不是 10%。不做这个分离，就会按相加去规划，然后发现对不上账。

反过来，一起上省了 14%，说明有协同，应作为 `atomic_group` 同批落地。

### 两条腿：单项不够

单项定价只能回答「它单独值多少」。优化循环实际要问的是「全上之后，再动它
还有没有收益」—— 这两个问题的答案**可以相反**。

| 腿 | 怎么测 | 回答什么 |
|---|---|---|
| 单项 | 只删这一项 | 它单独值多少 |
| 边际 | 全删 − 全删但留下这一项 | 别人都在的情况下它还值多少 |

两条腿合起来给六类：

| 类 | 单项 | 边际 | 含义 |
|---|---|---|---|
| `independent` | 显著 | 显著 | 单独有用，全上时也仍有用 |
| `redundant` | 显著 | 不显著 | 别人已经吃掉了它的收益，不要重复投入 |
| `coupled` | 不显著 | 显著 | **单独看不出收益，但别人依赖它** |
| `inert` | 不显著 | 不显著 | 真的没有可测量的空间 |
| `harmful` | 负 | — | 删掉它反而更慢 |
| `unknown` | 缺测量 | — | 判不了 |

**`coupled` 是只有边际那条腿才能发现的那一类。** 只有单项腿时，它的读数
落在噪声里 → 判「测不出来」→ 进 `not_worth_pursuing` → 工具主动告诉你
不要在一个承重的方向上投入。这不是漏报，是**反向报错**。

所以 `coupled` 的手段会被从 `not_worth_pursuing` 里剔掉，并单列进
`must_keep`。

缺一条腿时判 `unknown` 并说清补哪一次测量 —— 拿单项读数顶替，正是
`coupled` 被判成 `inert` 的那条路。

```bash
# 全删（universe）
python3 scripts/lever_probe.py add --store ./forge_work/levers.json \
    --lever drop_output_store --lever relax_align --time-us 50.0 \
    --variant-hash 7f3a9c1
```

### `--variant-hash`：这个变体真的改过吗

一个「删掉了某个环节」的变体，如果代码和基线一模一样，那它根本没被改过 ——
测出来的差值全部是噪声。而那个差值会被当成这个环节的**价格**，据此决定要不要
在它上面投几轮迭代。

这件事在流程里很容易发生：改动写在一个没被编译进去的分支里、改完忘了重新装
OPP、或者改的是另一个 shape 的路径。三种情形都会得到一个形状完全正常的
结果文件。

指纹与基线相同时拒绝登记。两侧任一没有指纹时不拒 —— 那会把老 store 与
不传指纹的正当用法一起卡住。

### 探针变体不是候选

删掉输出写回的 kernel 算出来的结果是错的。工具把记录标为 `probe_only`，
它们**不得上线，也不得进入接受判定**。

---

## 三、转发围栏：`fence_check.py`

论文 Appendix G 把这道围栏放在**任何计时之前**。

```bash
python3 scripts/fence_check.py --candidate ./my_kernel --op-name add_layer_norm
python3 scripts/fence_check.py --candidate ./my_kernel \
    --spec ./forge_work/reference_spec.json \
    --denylist ./forge_work/denylist.txt
```

### 没有它，度量体系可以被绕过

候选直接调厂商 aclnn 实现：加速比 1.0、精度满分、所有门禁都过，
但一行 kernel 都没写。自演进场景更隐蔽 —— 第 n 代转发给第 n-1 代，
链条可以这样空转很多轮，每轮都「没有退化」。

### 规则

围栏分两档，这一点要紧。

### 阻断档：这是转发的**标志**

| 规则 | 抓什么 |
|---|---|
| `F-ACLNN` | `aclnnXxxGetWorkspaceSize` / `aclnnXxx` 两段式调用（本算子） |
| `F-TORCH-NPU` | `torch_npu.npu_<op>` |
| `F-TORCH-FUNC` | `torch.nn.functional.<op>` |
| `F-ATEN` | `torch.ops.aten.<op>` |
| `F-OPP-SWITCH` | 候选自己改写 `ASCEND_CUSTOM_OPP_PATH`（只抓写，不抓读）|
| `F-REF-PATH` | 引用封存的参考路径 |
| `F-DENY` | 出现参考的专有符号 |

命中即 `passed: false`，`record` 直接回滚并归因到实现。

### 通道档：这是一条**能通向**厂商实现的路

上面七条有一个共同的结构性弱点：**它们全部由算子名派生**。而转发不需要
写出名字 —— `ctypes.CDLL("libopapi.so").aclnnAddLayerNorm` 里那个名字是
运行时拼出来的字符串，`importlib.import_module(cfg.backend)` 里连字符串
都没有。于是七条全绿，而候选一行 kernel 都没写。

| 规则 | 抓什么 |
|---|---|
| `F-DLOPEN` | `dlopen` / `dlsym` |
| `F-FFI` | `ctypes.CDLL` / `cffi` / `ffi.dlopen` |
| `F-OPAPI-SO` | 指名 `libopapi.so` / `libaclnn*.so` / `libascendcl.so` |
| `F-ACLNN-ANY` | 任意 aclnn 接口（不限本算子）—— 转发给另一个厂商算子同样不是自己实现 |
| `F-SUBPROCESS` | 起子进程。子进程里跑什么，静态扫描看不到 |
| `F-DYNIMPORT` | `importlib.import_module` / `__import__` |

**命中不自动判不通过。** 正当实现里也会出现这些构造：装载自己编出来的
`.so`、起子进程跑构建、动态 import 自己的后端模块。把「能通向」和「就是」
混成一档，等于拿整条规则的存活去赌每一次命中都是真的 —— 而一次误伤会让人
`--exempt` 整条关掉，之后真正的转发也无人把守。

但它们**不等于可以不看**：报告里单列 `channels` 字段，`record` 据此把
「围栏干净」与「有未说明的通道」区分开（第三态 `clean_with_channels`），
而不是把两者都读成通过。请逐条说明每个通道通向哪里。

### `third_party` 不再跳过

它曾在 `SKIP_DIRS` 里，那是围栏上最大的一个洞：把一份转发实现放进
`third_party/`、再从候选 import 它，整棵目录连扫都不扫 —— 候选代码里只剩
一行 import，没有任何 aclnn 字样。

「第三方代码不是候选的实现」这句话本身对，但它回答的不是围栏在问的问题。
围栏问的是「这次测到的加速比是不是这份 kernel 挣来的」—— 从哪个目录转发
出去，对这个问题没有区别。构建产物目录仍然跳过。

### 不误伤是硬要求

候选**必须**定义同名算子，那是它的本职。围栏区分「定义它」与「转发给它」：

- 自己的 kernel 定义、`torch.ops.cust.<op>` 调用 → 放行
- 注释里提到 `aclnnXxx` → 放行（注释是说明，不是执行）
- 构建产物目录 → 跳过

误伤的代价比漏判更高：一次误伤会让人去关掉围栏，关掉之后真正的转发
就无人把守了。所以每条都给 `file:line` 证据，并支持 `--exempt` 按规则豁免。

---

## 参考资料

- [`references/instrument-map.md`](references/instrument-map.md) —
  论文仪器 → 昇腾对应物的完整对照，含「没有对应物」的诚实标注

## 上游真源

| 内容 | 真源 |
|---|---|
| msprof 字段定义与阈值 | `ops/ops-profiling` 的 `references/csv_fields_reference.md` |
| 采集方式 | `ops/ops-profiling` 的 `msprof_profile_run.sh` |
| 芯片规格 | `ops/npu-arch` |
| 精度判据 | `ops/ops-precision-standard` |

本技能不自产阈值。所有判定阈值引自上面的真源；唯一的例外是暴露时间的
20% 判定线，那是本工具的工程判断，输出里据实标为 derived 并注明无文档出处。
