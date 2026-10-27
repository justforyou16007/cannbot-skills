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
| timeline probe | `signature_probe.py` 的暴露时间段 | **部分** —— 只知道闲了多少，不知道闲在哪 |
| 手段定价 / 裁决探针 | `lever_probe.py` | 完整 |
| 转发围栏 | `fence_check.py` | 完整 |
| CUDA-graph replay、指令级时间线 | 无 | **没有对应物**，作开放问题处理 |

完整对照见 [`references/instrument-map.md`](references/instrument-map.md)。

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
# msprof 原始目录
python3 scripts/signature_probe.py --prof-dir ./prof_out --op-name AddLayerNorm

# 归档目录（msprof_perf_summary.py 产出）
python3 scripts/signature_probe.py --archive-dir ./round_1 --op-name AddLayerNorm

# 给出理论最小搬运量后，额外算出数据复用倍数
python3 scripts/signature_probe.py --archive-dir ./round_1 \
    --op-name AddLayerNorm --min-gm-bytes 8388608 --json
```

输出分六段：测量可信度、launch structure、pipeline 占比、停顿分解、
搬运量与带宽、暴露时间；之后是**架构假设**与**开放问题**。

### 可信度是硬闸门

下列任一成立，探针**扣留全部架构假设**：

| 情况 | 为什么致命 |
|---|---|
| DVFS 降频 | 各段被按不同比例拉长，一个计算受限的 kernel 会看起来像搬运受限 |
| ICache 缺失率 >15% | 取指开销混进各段占比 |
| `ResourceConflictRatio` 缺失 | 停顿数据是架构判断的一半依据，缺了只能瞎猜 |

扣留而不是降级，是因为不可信数据推出的架构**错得很有说服力** ——
数据看着正常，错误极难被发现。

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
