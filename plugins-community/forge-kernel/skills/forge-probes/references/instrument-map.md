# 仪器对照：论文用的探针在昇腾上对应什么

论文的实验跑在 NVIDIA Hopper / Blackwell 上，用的是 NCU、CUPTI、CUDA events
这一套。§4.2 说它们在 Ascend 910C 上做跨架构验证时「把 NCU 换成 msprof，
循环原样跑」。

这份文档把那句话展开：逐个仪器说明昇腾上用什么替代、替代得完不完整、
以及**哪些根本没有对应物**。最后一类最重要 —— 假装有对应物比承认没有
危险得多，因为它会让一个本该留作开放问题的猜测混进 milestone。

---

## 一、完整对应

### NCU pass → msprof 七份 metric

论文 §3.2：「一次 NCU pass 给出 launch structure、occupancy、
pipeline utilization 和 stall decomposition。」

| NCU 给的 | 昇腾对应 | 来源 |
|---|---|---|
| launch structure | `Block Dim`、`Mix Block Dim`、`Task Duration(us)` | `OpBasicInfo` |
| pipeline utilization | `aiv_vec_ratio`、`aic_cube_ratio`、`mte1/2/3_ratio`、`fixpipe_ratio` | `PipeUtilization` |
| **stall decomposition** | `*_wait_ratio`、`*_cflt_ratio` | `ResourceConflictRatio` |
| （NCU 没有的）搬运字节数 | `GM_to_UB_datas(KB)` 等 | `Memory` |
| （NCU 没有的）L2 命中率 | `total_hit_rate(%)` | `L2Cache` |

**工具**：`signature_probe.py` 把这七份归并成一份结构化签名。

> 注意 `ResourceConflictRatio` 是**上板独有数据，仿真拿不到**。用仿真
> 做参考分析时停顿这一整段缺失，`signature_probe.py` 会据此拒绝输出
> 架构假设，而不是把缺失当成「没有停顿」。

### 正确性判据 → ops-precision-standard

论文 Appendix D 说每个判据按算子的数值类别选容差，并特别指出
「小输出会轻易通过固定绝对容差」。

仓内 `ops/ops-precision-standard` 已经按数值类别组织：浮点计算、整数计算、
量化计算、随机数生成、非计算类，各有 MARE/MERE 阈值与判定脚本。
**直接用它，不要另建一套。**

### 配对计时 → evaluate_ops_direct.py

论文 Appendix G 要求两个 kernel 在同一 job 内计时，最终 claim 取多次
读数的中位数。

`evaluate_ops_direct.py` 用 `torch.npu.Event` 计时、取中位数、同时给出
变异系数 `cv_pct`。它刻意不走 msprof（`acl.prof.init` 每进程只能调一次，
多 session 并发采集会抢资源导致计时不稳）。

---

## 二、部分对应（有替代，但不等价）

### timeline probe → 暴露时间 + 逐核分布（部分可答）

论文 §3.2：「串行延迟受限的 kernel 还会额外上一次 timeline probe。」

timeline probe 回答四件事，昇腾上**三件有通道、一件是真盲区**：

| timeline probe 的子能力 | 昇腾情况 | 通道 |
|---|---|---|
| 总暴露量（闲了多少） | 可答 | `Task Duration − max(aic_time, aiv_time)` |
| 掉队核 / `uneven_spans` | **可答** | 已归档的 `per_core_cycles.csv`，`(max−min)/max` + `max/min ≥ 2×` 判据 |
| 角色暴露账（谁拥有残差） | **可答，且比上游干净** | AIC/AIV 是物理分离核、各有独立 PMU，所以每个角色的闲置比例能单独算；上游的 `tensor_active` 是一个标量所有角色共用 |
| **ramp / plateau / drain**（闲在哪一段） | **真盲区** | 见下方 waiver |

#### ramp/plateau/drain 的 waiver

按上游的规矩，声明盲区要带「已试过哪些通道 + 为什么都不行」，不能只写
「列为开放问题」。已试八条：

| # | 通道 | 为什么不行 |
|---|---|---|
| 1 | `msprof` op_summary | per-op 聚合，无时间轴 |
| 2 | `SAMPLE_PMU_TIMELINE` / AI Core Utilization 层 | 有时间轴（`timestampNs` + `coreId` + `coreType`，而且我们已经在采），但 `--aic-freq` 官方上限 **100 Hz = 10 ms**，而 kernel 是百微秒级 —— 差约 50 倍，一次 launch 落不到一个采样点。另外 JSON 导出路径「AI Vector Core 指令占比数据无 timeline 信息」，会丢掉 AIV 这一半角色（要做角色账必须走 DB 不走 JSON） |
| 3 | `--aic-metrics=InstrTimeline` / `PipeTimeline` | 仅 950PR/950DT 系列；目标是 910B（A2）。且即便在 950 上也只展示 6 个核，对 20~48 核的跨度分析本身不够 |
| 4 | 上板 `TRACE_START/TRACE_STOP`、`MarkStamp` 打点 | 上板打点仅 950；A2 只在仿真里可用 |
| 5 | msopprof simulator 的逐核 `trace.json` | **结构上唯一能给出全部四项的通道**，但官方定位是「仿真结果仅供参考，算子真实运行情况以实际数据为准」。可用于定位，不能承载数值主张 |
| 6 | `--aic-metrics=KernelScale` + `MetricsProfStart/Stop` | A2 上板可用，但要改 kernel 源码（撞 no-code 边界），而且它测「声明的」分段而非「发现的」相位 —— 不是同一台仪器 |
| 7 | mstx / `--msproftx=on` | host+device 的 launch 粒度区间，给不出 launch 以下的分段，也给不出逐核 |
| 8 | `--instr-profiling` | 5 Hz，比现有通道更粗；且与 `--ai-core`/`--aic-mode`/`--aic-freq`/`--aic-metrics`/`--task-time` 互斥，会顶掉现有采集 |

**结论**：在 A2 真实硬件上取得时间分辨的相位切分做不到。上游那条
`interval-too-coarse` 守卫**必须保留，而且会一直触发** —— 这不是缺陷，
这正是它存在的理由。唯一能原样保留相位语义的做法是把计量区间拉到秒级
（上千次 launch 的循环），但那时含义变成「整轮的相位」而不是「单 kernel
内的相位」，换了口径就必须在 milestone 里写清。

> 以上证据来自官方文档与公开实验记录，**本仓未在实机复核**。

#### 总暴露量的推导

能做的替代是从已有字段推导**暴露时间**：

```
暴露时间 = Task Duration − max(aic_time, aiv_time)
```

取 max 而非相加，因为两者在 MIX 算子里并行；相加会高估覆盖、低估暴露，
从而把一个真有问题的 kernel 显示成健康的。

**差在哪里**：timeline probe 告诉你时间花在哪一段，暴露时间只告诉你有
多少时间闲置。暴露量里混着三件事 —— 启动开销、核间等待、尾部收敛。
其中**尾部收敛可以单独测**（逐核分布，见上表），剩下两项才是真的分不开。
所以判为串行延迟受限之后：先看 `core_balance` 是否报了掉队核（那说明该改
划分方式而不是流水重叠），再用 `lever_probe.py` 对剩余部分做消融定位。

**工具**：`signature_probe.py` 的 `exposure` / `core_balance` /
`role_exposure` 三段。暴露阈值 20% 是本工具的工程判断，**无文档出处**，
输出里据实标为 derived；掉队核的 2× 判据对齐上游 timeline probe 的
`uneven_spans`。

> `core_balance` 的比例**直接在 cycle 上算，不做频率换算**。这一点要紧：
> `aicore_time(us)` 的官方定义预设了「所有 Block 同时调度且执行时长相等」，
> 拿它反推主频再去度量逐核**不等**时长，口径自相矛盾，而且官方写明 DVFS 下
> 该字段不准确、不建议参考。而 `(max−min)/max` 是比值，不需要知道频率。
> （需要频率真值时正解是 `msprof_<时间戳>.db` 的 `AICORE_FREQ` 表。）

### occupancy → 无直接对应，可用核数占比代理

NCU 的 occupancy 是「实际驻留 warp 数 / 理论最大」。昇腾没有对应概念。

最接近的代理是 `Block Dim` 与该芯片可用核数之比。但这个比值的含义不同：
它反映的是任务切分是否铺满核，不反映单核内的并发度。

**不要把它当 occupancy 用。** 写进 milestone 时应当如实写成
「核数利用率」，并说明它与 occupancy 不是一回事。

---

### orchestration probe → 角色暴露账（部分可答）

论文的 `orchestration` 探针是五个里最大的一个，回答两类问题：

| 子能力 | 昇腾情况 |
|---|---|
| **角色暴露账**（谁让计算管道空转） | **可答，且比上游干净**。`exposed = duration × (1 − active_pct)` 三个因子都现成，而 AIC/AIV 是物理分离核、各有独立 PMU，所以每个角色的闲置比例能单独算；上游的 `tensor_active` 是一个标量，所有角色共用它 |
| 逐核负载分布 | **可答**，见上面 timeline 段的 `core_balance` |
| **同步拓扑普查**（18 个结构轴：barrier 环的 stride 与 slot 数、哪些角色 arrive 哪些 wait、in-flight 深度） | **无直接对应物** —— 编程模型不同：CUDA 是 mbarrier / named barrier，昇腾是 `SetFlag/WaitFlag` 与 BufferID。但「同步是否成对发射」可以走结构核验那条静态 IR 通道判 |

**角色映射的边界**（要写进 milestone 的 caveat）：CUDA 的 producer/consumer
warp group 是同一个 SM 内的**软件**角色，数量与分工由 kernel 自己定；
A2 的 AIC/AIV 是两类**物理**核，配比由 SoC 固定、经 `Mix Block Dim` 暴露。
所以这里的「角色」只有两个，不是 N 个。论文那条读法规则 ——「暴露时间
最多的角色拥有残差，诊断的 locus 必须落在它治下」—— 依然成立（在昇腾上
甚至更硬，因为 AIC/AIV 的决策面在代码里就是分开的分支），但粒度更粗；
纯 Vector / 纯 Cube 算子退化成单角色，此时这一段不产生信息，如实标
`not_applicable`。

**工具**：`signature_probe.py` 的 `role_exposure` 段。

---

## 三、论文有、仓内原本没有，本插件新建

### verdict probe / 手段定价 → lever_probe.py

论文 §3.3：arch agent 重新进入时「给手段定价、跑裁决探针以分离单项
效应与耦合效应」。Appendix I 的案例：单独去掉输出写回拿到 6.0%。

**仓内原本没有任何对应工具。**

`lever_probe.py` 做三件事：

1. **定价** —— 把某环节整个删掉省下的时间，就是优化它的**收益上界**
2. **分离耦合** —— 两项各省 5% 但一起只省 6%，说明在抢同一份时间
3. **给出「别做什么」** —— 收益落在噪声内的方向不要投迭代

定价的主要价值是**决定不做什么**。优化循环里最贵的不是方向错了，
而是在天花板只有 2% 的方向上投三轮迭代。

> 探针变体会产出**错误结果**（删掉输出写回的 kernel 算得不对）。
> 它们不是候选，工具会标为 `probe_only`，不得上线也不得进接受判定。

### fence → fence_check.py

论文 Appendix G：「一道围栏，拒绝任何调用或转发到该 kernel 的候选 ——
在任何计时之前运行。」

**仓内原本没有。** 没有它，整个度量体系可以被绕过：候选直接调厂商
aclnn 实现，加速比 1.0、精度满分、所有门禁都过，但一行 kernel 都没写。
自演进场景下更隐蔽 —— 第 n 代转发给第 n-1 代，链条可以空转很多轮。

昇腾上的判别依据是 aclnn 的两段式调用（`aclnnXxxGetWorkspaceSize` +
`aclnnXxx`），候选自己的 kernel 不会长这样。

**难点是不能误伤**：候选必须定义同名算子，那是它的本职。工具区分
「定义它」与「转发给它」，并给出 `file:line` 证据让人能当场判断。

---

## 四、论文有、昇腾上确实没有

这几项在本插件里不做假替代，直接作为开放问题处理。

| 论文的 | 为什么昇腾上没有 | 怎么办 |
|---|---|---|
| CUDA-graph replay 计时 | 无对应机制 | 用 `torch.npu.Event` 批量计时取中位 |
| CUPTI kernel span | msprof 的 kernel-only device time 可近似，但采集方式与配对评测冲突 | 配对计时与 profiling 分两次跑 |
| warp 级寄存器预算 | 编程模型不同（SIMD 为主） | 对应到 UB/L0 容量约束，但不是一回事 |
| 指令级时间线（profiler 侧） | msprof 算子级是聚合计数器，无逐指令粒度 | 列为开放问题，用消融逼近 |
| **最终指令助记符反汇编** | 厂商工具链三处同时关闭，见下方 waiver | 用结构核验替代意图，不追求助记符文本 |

> 「指令级时间线」这一行尤其要守住。把消融结果说成「时间线显示……」是把
> 推断伪装成观测，而这类错误因为看起来有数据支撑，排查起来最费时间。

### 最终指令反汇编的 waiver

上游有五个 `sass` 工具，其中三个**只需要静态反汇编文本**（单参数吃纯文本、
零外部依赖）。那种实现形态在昇腾上无法 1:1 移植。已试四条：

| 尝试 | 结果 |
|---|---|
| `llvm-objdump -d` 真实 `.o` | 认得 `elf64-hiipu`、能列符号与偏移，**每条指令都是 `<not available>`** |
| `llvm-objdump --version` | `hiipu64` **在** Registered Targets 里 —— 后端在，MC 反汇编表没随包发布 |
| `bisheng -cc1 ... -S` | `error: unsupported option '-S' on device side` |
| `-save-temps` | 只落 `.ccei`（预处理后的源）与 `.o`；管线是 `-emit-obj -x cce-cpp-output`，**没有 `.s` 阶段** |

官方口径一致：毕昇编译器工具清单里没有反汇编器，`llvm-objdump` 文档自述
「不会打印汇编信息，仅打印符号表和偏移量」。不存在 `cce-objdump`。

**但 sass 探针的意图可以落地。** 它要回答的是「声称的结构到底有没有被
构建出来」—— 而这件事有两条通道，都不需要助记符：

| 通道 | 能判什么 | 代价 |
|---|---|---|
| `ccec -emit-llvm -c` + `llvm-dis` | 操作数所在存储（IR 的 address space：`addrspace(1)`=GM、`addrspace(6)`=UB）、同步 intrinsic 是否成对 | 只要 CANN toolkit，**无硬件无仿真器**，秒级 |
| msopprof simulator 的 `instr_exe.csv`（`pipe` / `detail` 两列）+ `ub.{rd,wr}_log.dump` | 搬运的源与目的、走了哪条 pipe、UB 逐条读写；`instr_log.dump` 还带 cycle 与 PC 可算发射间隔 | 跑一次仿真 |

第一条正好堵上游记录的那个翻车模式 ——「预留了寄存器预算、但操作数仍走
共享内存往返」在 IR 的地址空间上是二值可判的。它的边界也清楚：IR 在指令
选择与寄存器分配**之前**，所以证不了「寄存器没溢出」。

判据口径见 `ops/ops-simulator/references/structural-verification.md`。

**降级手段**（仅用于 A/B）：`llvm-objcopy --dump-section .aicore_binary=`
取出 device image，再 `--dump-section .text=` 配 `xxd` 做字节级比对。能回答
「改了一行、二进制变没变」（识别空操作改动），不能做结构普查。

> 以上证据来自官方文档与公开实验日志（CANN 9.1.0），**本仓未在实机复核**。

---

## 五、一条贯穿始终的原则

论文 Appendix C 把 **agent 自评的探针单列为一类 bug 来源**。

落到工具上就是：每个探针都要能回答「这个数没采到时会怎样」。

- `signature_probe.py`：缺失标 `present=false`，不折成 0；停顿数据缺失
  直接拒绝输出架构假设
- `lever_probe.py`：缺单项测量时标 `incomplete`，不硬算耦合
- `fence_check.py`：扫不到源文件时报错，不返回「通过」

把「没数据」读成「没问题」是这类工具最常见也最难发现的失效方式。
