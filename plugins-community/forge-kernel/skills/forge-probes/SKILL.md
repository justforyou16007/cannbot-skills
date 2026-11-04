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
  **拒绝输出架构假设**
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

| 规则 | 抓什么 |
|---|---|
| `F-ACLNN` | `aclnnXxxGetWorkspaceSize` / `aclnnXxx` 两段式调用 |
| `F-TORCH-NPU` | `torch_npu.npu_<op>` |
| `F-TORCH-FUNC` | `torch.nn.functional.<op>` |
| `F-ATEN` | `torch.ops.aten.<op>` |
| `F-OPP-SWITCH` | 候选自己改写 `ASCEND_CUSTOM_OPP_PATH`（只抓写，不抓读）|
| `F-REF-PATH` | 引用封存的参考路径 |
| `F-DENY` | 出现参考的专有符号 |

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
