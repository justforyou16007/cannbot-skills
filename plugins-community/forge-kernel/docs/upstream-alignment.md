# 与论文开源实现对齐：逐步差异与决策表

对照对象：[OpenBMB/ForgeKernel](https://github.com/OpenBMB/ForgeKernel)（论文的开源实现）。
本文给出**每一步的具体差异**，供决定哪些直接替换、哪些保留。

所有上游引用都带文件与行号，可自行复核。标注的性质：

| 记号 | 含义 | 决策倾向 |
|---|---|---|
| **A** | 纯逻辑差异。上游那段代码不依赖 CUDA/NCU，判据可直接搬过来 | 建议替换 |
| **B** | 形态冲突。上游是常驻 driver 进程，我们是 Claude Code 插件，同一个不变量要换执行点 | 适配，不照搬 |
| **C** | 平台不可移植。依赖 NCU/SASS 等昇腾没有的通道 | 保留现状 + 文档承认盲区 |
| **D** | 我们更强。上游没有对应物 | 保留，不要替换 |

> 本轮（2026-11-03）已修的部分在每格里标「已修」。未标的就是当前仍存在的差异。

---

## 决策速查

想先看结论的话，这是按「替换代价 ÷ 收益」排的：

| 优先 | 项 | 性质 | 代价 | 不做的后果 |
|---|---|---|---|---|
| 1 | 精度判据换成对高精度真值的误差比 | A | 中 | 降精度换速度在 1e-3 容差下恒过；promote 后锚点逐代漂移，N 代后与真值距离无界 |
| 2 | 重开必须定价 + 徒劳重开上限 | A | 低 | 架构层可凭一句话无限重开，每次重置全部计数器，状态机看起来一直健康 |
| 3 | 轨迹记 `sha` / `base_sha` / `reading_of` | A | 低 | 「上一轮就是当前代码的样子」这个前提无法核实；做差基线可能是错的而数字看着正常 |
| 4 | 课程表双模校验 + 与上一版做差 | A | 中 | 悄悄放宽阈值或删掉机制行完全无痕 |
| 5 | 知识库词表机械抽取 + 禁用词/叙事 lint | A | 低 | 卡片可写成 campaign 日记；词表靠人记得封存 |
| 6 | 自动计欠账 + 提炼卡片 | B | 高 | 知识一张都攒不出来（本仓现状：1 张卡） |
| 7 | 两窗口计时 + host-bound 审计 | A | 中 | 读的是含 host 税的 e2e；两侧 host 路径厚度不同，加速比里混着非 kernel 的差 |
| 8 | 窗口内容契约（window.toml） | B | 中 | 我们两侧共用 call_spec，结构上不会发生 —— 优先级可降 |
| 9 | 三工作负载几何平均 | A | 中 | 只对 seed-42 这一组数有效的快速路径读出来就是真加速 |
| 10 | 墙钟预算 / NEUTRAL 构建期 | B | 低 | 前者无常驻进程可挂；后者 M0 搭建阶段只能全记成 revert |

---

## ① 任务与参考定义

| | 上游 | 我们 |
|---|---|---|
| 入口 | `tasks/<op>/task.toml` + `description.md` + `shapes.json`，三件缺一不可 | `reference_adapter.py prepare` → `reference_spec.json` |
| 参考模式 | **四种**，决定 teacher 能看到什么：`source`（源码+硬件行为）/ `blackbox`（只有硬件行为）/ 不写（先由 `/goal` 生成一份参考）/ 榜单（冻结榜单的题目与评测）<br>`README_zh.md:145-168` | 三种来源：`builtin_aclnn` / `opp_install` / `sealed_kernel`。**没有模式概念** |
| 角色不对称 | **边界按角色区分**：`source` 模式下 teacher 可读参考源码，student 永不可读。`agent_policy.py:184-189` 决定哪个角色拿到哪套 deny 规则 | hook 对所有 agent 一视同仁 |
| 派发确认 | 不适用（参考是文件或 library，不靠派发） | 已修：`verify --confirm-dispatch --evidence` 回写，`init` 读它，为 false 时拒绝 |

**差异性质：B（角色不对称）+ A（参考模式）**

角色不对称这条很要紧：上游自进化**第一代之后强制走 source 模式**——「teacher 读上一代的最好版本，student 读不到」（`README_zh.md:177-182`）。我们 gen≥1 时两个 agent 都读不到，所以我们的架构 agent 比上游的瞎。

要替换的话，Claude Code 这边能做的是：hook 读 `CLAUDE_AGENT_NAME` 之类的上下文区分调用者（需确认 Claude Code 是否向 hook 传入 agent 身份；若不传，这条做不了）。

**待决策**：
- [ ] 引入 `source` / `blackbox` 两种模式，gen≥1 自动转 source（配合角色不对称）
- [ ] 保留现状：所有代次、所有角色一律不可读

---

## ② 参考封闭（no-code 边界）

| | 上游 | 我们（本轮修复后） |
|---|---|---|
| 封存对象 | 四种形态：源码 bundle → **keyed payload**（全文件 + 逐文件 sha256，只在内存成包，密钥只进 gitignore 的 `submit.local.toml`）；编译产物 `.so`（源码留在 agent 读不到的另一棵树）；**library 型**（不是文件：`deny_modules` 导入名 + `deny_urls` 仓库）；上一代自己的 `kernel.py` | 路径清单 + sha256 manifest，内容全程明文留在磁盘上。已修：源码**与配置 json** 一并禁读 |
| 真边界 | **Bash sandbox 的 `filesystem.denyRead`**（内核/seatbelt 级，怎么拼路径都挡），带 `failIfUnavailable: true`。`agent_policy.py:268-279` | PreToolUse hook。已修：路径归一化 + 命令 token 抽取 + 上层目录双向判（五种绕法实测全堵） |
| hook 的自我定位 | `agent_policy_hook.py:14-23` 原话：*「This is a tripwire, not a boundary.」* | 我们的 hook **就是**边界 |
| 执行点数量 | **五处互相独立**：sandbox / project deny 规则 / PreToolUse hook / judge 内部的 `kernel_fence` / job 内的 `blackbox_guard` | 一处（hook）+ 一个没有调用方的 `fence_check` |
| 策略载体保护 | deny `Edit(<ws>/.claude/**)` 与 `Edit(HOOK_SCRIPT)`，且**每一回合重写 settings.json**（`cli.py:389-393`，注释：上一回合对该目录做的任何事都不会留到下一回合） | 已修：hook matcher 加 `Edit/Write`，拦住对封存清单、hook 自身、hooks.json 的改写。**但没有「每回合重写」这个机制** |
| 网络侧 | deny `WebFetch`/`WebSearch` + `Bash(curl:*)` 等前缀 + `allowedDomains: []` / `strictAllowlist`。docstring 写明：审计早期 run 的会话时发现 **student 去抓自己正在写的那个算子的公开实现并改写进 kernel** | 已修：`hooks.json` 加了同类 deny 规则 |
| 失效时的行为 | **hook 脚本缺失 → 拒绝开跑**；策略读不出 → `exit 2`；sandbox 起不来 → 停住 agent | 已修：声明了清单却读不出 → `exit 2`；`init` 做一次**实测**自检（构造探针调用过一遍 hook，期望返回 2） |
| 改动检测 | 三处：job 内比 payload sha256 / 每轮迭代 `harness_drift` 逐 blob 比对并**回滚** / `.so` 连 PROVENANCE.txt 一起提交 | `seal_reference.py manifest` 是人工命令，没挂任何门禁；新增文件这个方向查不出来 |
| 反向检查 | `closed_ref.py:653-695` `check_tree`：候选树里有没有参考的明文（文件 sha256 命中，或 ≥24 字符的 distinctive line 重叠 ≥40%），不干净就拒绝交付 | **没有**。F-DENY / L-DENY 只认符号名，改个类名就过 |

**差异性质：A（反向检查、改动检测挂门禁）+ B（sandbox、每回合重写）+ D（配置 json 一并封存是我们按昇腾实际加的，上游无对应物）**

**待决策**：
- [x] `check-tree` 两级判据：整文件 sha256（只用 manifest，不读参考）+ 特征行（去注释、压空白、丢掉过短与高重复的行）。
      报告里写明**抓不到什么**：改过名字、重排过结构的搬运两级都抓不到 ——`findings` 为空不等于「没有搬运」。
- [x] `record` 调 `verify_manifest`，三个方向（删除 / 变更 / **新增**）。
      新增那个方向不是「记录对不上」这么轻：新文件没进清单就没进禁读清单，hook **不会拦**对它的读取 ——
      参考树上多出一个 agent 能随便读的源文件，而比对已记录那些的校验全绿。`opp_install` 那类活的安装目录尤其容易碰上。
      清单缺失时不硬拒（内置实现作参考、老 state 本来就没有），由调用方告警。
- [x] 每个被封文件的真实父目录**一路加到产物根**。
      hook 对 Grep / Glob 的判据是「请求路径是不是某个被封条目的祖先」，而产物根是软链、
      或源码散在产物之外时，真实父目录既不等于产物根、也不在逐文件清单里。漏掉中间任何一层，
      对那一层的递归请求就会放行，而它底下照样有被封的文件。到产物根为止 ——再往上是公共目录。
- [ ] 保留 hook 作为唯一执行点（Claude Code 无 sandbox 等价物时只能如此）

---

## ③ 噪声底与计时纪律

| | 上游 | 我们（本轮修复后） |
|---|---|---|
| 噪声底定义 | `self_evolve.py:50-56`：`spread = (max−min)/min` 取本代**确认比值**的离散，`max(0.015, spread)` —— 1.5% 是硬下限，注释称「循环的平台期地板」 | 已修：口径对齐（全极差比最小值 + 1.5% 下限）。此前是半极差比中位数且无钳位 —— 同一批数据算出的噪声底只有论文的**一半** |
| 单次读数 | warmup 20 → 50 次 e2e 取中位 / 5 窗口×50 连发取中位 → A-B-B-A 取两读均值 → 3 个 workload 几何平均 → 3 个 job 取中位 | 已修：warmup 20、burn-in 2s、A-B-B-A 四轮每侧取两读均值 |
| 位置偏置 | **机制消除**。`timing.py:380-397` 注释写明同一个 kernel 分别当两侧测差约 **2.7%**，偏向先被测那侧；burn-in + 镜像两个机制缺一不可 | 已修：分叉脚本做 burn-in + A-B-B-A，实测把 2.6% 消到 1.0000 |
| 计时窗口 | **两个窗口都报**：e2e（含 host 税）与 device 摊销窗口，另有 `time_host_enqueue`（循环内不 sync，量 python+launch）与 `host_bound_ratio = host/device`，阈值 0.5 | **只有一种**：每次 rep 前后各 synchronize，纯 per-call e2e，host 派发全在窗口内 |
| host-bound 审计 | 参考侧 host-bound → **硬拒开跑**（`baseline.py:246-248`，live 代价 11.5 小时 + 1.86× 虚高目标）；student 侧 host-bound → 写进下一轮诊断 | **没有任何等价物**。msprof 的 `task_duration_us`（真正的 device 侧时长）已采到并存进签名，但**从不与 `time_us` 对账** |
| 窗口内容契约 | `window.toml` 声明两侧秒表各装了什么（inputs/outputs 四个列表必填），单侧出现的项必须给 `[[asymmetry]]` 并归类 symmetrize/price/accept，`loop start` 校验不过拒绝开跑（exit 13） | **没有**。但我们两侧共用同一份 `call_spec`、同一批 `.pt` 输入、同一个派发路径，所以上游要防的「签名不对称」结构上不会发生 |
| 输入轮换 / 只读核对 | 2-4 份输入缓冲轮换（让按指针 key 的 host 缓存暴露成计时爆炸而不是藏起来）；prep 张量快照比对（计时调用改写了输入就判本轮失败） | 输入加载一次，53 次调用全用同一批张量、同一组地址。无轮换、无快照 |
| 工作负载数 | `PERF_WORKLOADS = 3`（同一形状、不同内容/分布），gate 取两侧几何平均之比，`host_bound_ratio` 取**最差**那个 | 一个 shape 一个固定 draw（`manual_seed(42)`） |
| 跨 job 比值 | SINGLE-REF CONTRACT：参考不能骑同一个 job 时，**一个 `ref_*` 键都不发**，宁可没有比值也不要跨 job 的参考读数 | `--baseline-cache` 命中时 baseline 的时间来自另一次 job，结果里没有任何标记。已修：结果里写出 `baseline_from_cache`，但 `record` 还不据此拒绝 |

**差异性质：A（两窗口 + host-bound 审计 + 多工作负载 + 输入轮换）+ C（window.toml 的 asymmetry 对我们基本不适用）**

**待决策**：
- [ ] 评测加「不 sync 连发」入队循环与「一对 Event 套 N 次连发」摊销窗口，输出 `host_enqueue_us` / `device_median_us` / `host_bound_ratio`
- [x] `record` 对账 `signature.launch.task_duration_us` ÷ 结果 `time_us` ≥ 20%；不到则**不抬高 best**（而不是只告警 —— 告警会被读过就忘）
- [~] 几何平均有了（`_geomean` 聚合**跨 shape** 的 speedup），但**每个 shape 只有一套输入**（单个 `manual_seed(42)`）；`host_bound_ratio` 这个量根本不存在，所以「取最差那个」无从谈起
- [ ] 输入做 2-4 份轮换；计时前后对每个输入张量取 hash 比对
- [x] `record` 见到 `measurement.baseline_from_cache` 为真直接拒收，拒绝理由里带上 2.7% 这个可核对的量
- [ ] 不引入 window.toml（两侧共用 call_spec，结构上不会不对称）

---

## ④ 探针与架构签名

| | 上游 | 我们（本轮修复后） |
|---|---|---|
| 主通道 | NCU：可加计数器求和、强度量取耗时最长那次 launch，launch 数随行带走 | msprof 七组 metric —— **七次独立的程序运行** |
| 五个探针 | `stall`（时间从哪些指令漏掉，账必须加总到 90%）/ `timeline`（时间**何时**漏：ramp/plateau/drain）/ `lever-pricing`（2N+2 设计：base、各单项、全上、全上各减一项）/ `orchestration`（声明的同步结构有没有被构建出来 + 角色暴露账）/ `sass`（结构/操作数源/调度深度/launch 形状） | `signature_probe`（七份 CSV 归一成签名）+ `lever_probe`（只有单项与两两组合） |
| 采集前置 | 两个小工具，**每次 loop 启动各问一次**：作业里裸 `python3` 能不能 import torch；**这台机器能不能 profile**（一个小 kernel 跑一次 ncu，权限不足就降级为纯计时） | `reference_adapter verify` 查 CANN 路径 / npu-smi / torch_npu / 参考路径 / 派发。**不查「msprof 能不能采」**，也不随循环复检 |
| 数据质量 | 三态：字段不存在 / 存在且为 0 / **reader 解析失败**（第三类响亮失败，原话「NOT missing data; fix the reader, do not waive」） | 两态。`safe_float` 把解析失败一律降级为「没采到」 |
| 跨读物一致性 | 同一次采集、同一地址空间内 join，越界明文拒绝（`JOIN-FAIL reason=address-space-mismatch`） | 已修：新增 `cross_metric` 检查（各份挑中的算子名、Block Dim 不一致即阻塞） |
| 可信度闸门 | 降频（采集期间轮询取忙时中位）/ 样本下限 500 / 账加总 90% / join 率 ≥50% / 短 kernel mis-aim | 已修五闸：频率（**此前恒为 unknown 且不阻塞**，而本仓 msprof 指南明写 `op_summary.csv` 没有这两列）/ 取指 / 停顿列 / 跨组 / 稳态行 |
| 定价设计 | add 腿 + **remove 腿**（全上各减一项）。SKILL 明写这是**唯一能读出** `coupled`（单独无用但别人依赖它）的方式；六类分类 | 只有单项与两两组合，**无 remove 腿**。三项以上的变体被静默丢弃。于是上游的 `coupled` 在我们这里一律判「测不出来 / 不要再投」—— **建议方向与上游相反** |
| 空操作检测 | 变体与 base 字节相同即 `raise`（原话：no-op variant 是这个仪器唯一会撒谎的方式）；cubin sha 比对 | **没有**。探针脚本写错（sed 没匹配、改的是注释）读出的收益落在噪声内 → 判死路并记录在案 |
| 角色暴露账 | `exposed_ns = duration × share × (1 − tensor_active)`，按角色排序与参考并排，**Δexposed 最大的角色拥有残差，诊断的 locus 必须落在它治下** | `exposure`（一个标量）+ `most_blocked`（只有名字）。两者**从不相乘** —— 而所有输入都已在手 |
| 逐核分布 | `timeline_probe` 的 span_ratio / uneven_spans（同角色跨度差 >2× 告警） | `instrument-map.md` 说「没有等价物」。但 `PROF_Sample/.../aicore.db` 的逐核 cycle **已被 ops-profiling 解析并归档**在我们读的那个目录里 |

**差异性质：A（remove 腿、空操作检测、角色暴露账、第三态、采集前置）+ C（sass 静态普查、ramp/plateau/drain）**

**待决策**：
- [x] 两条腿 + 六类分类（independent / redundant / **coupled** / inert / harmful / unknown）。`coupled` 从 `not_worth_pursuing` 剔除并单列进 `must_keep` —— 此前它会被列进「不要再投」，那不是漏报而是反向报错
- [x] `--variant-hash`，与基线相同时拒绝登记。**没做成必填** —— 两侧任一没有指纹时不拒，否则会把老 store 与不传指纹的正当用法一起卡住
- [x] `signature_probe` 新增 `role_exposure` 段：`exposed_us × wait_i / Σwait`，按最大项产出一条带通道的假设
- [x] 读 `per_core_cycles.csv`，给出 `(max−min)/max` 与慢核 ID（**落在顶层 `core_balance` 键**，不在 `exposure` 段内）
- [x] 新增 `parse_cell` 三态；解析不动的列**阻塞**架构假设。顺带把 `check_trust` 从「extract_stalls 之后」挪到**全部取数之后** —— 此前 memory / cache 两段里的解析失败它看不见，而那正是搬运量与命中率所在的两段
- [ ] 新增 `probe_env` 子命令：用最小算子跑一次 msprof 并断言 `aiv_vec_bank_cflt_ratio` 这一列真的出现
- [x] `instrument-map.md` 补 orchestration 一节，并把「没有时间线等价物」改写为「没有时间线，但有逐核 cycle」

---

## ⑤ 课程表与门禁校验

| | 上游 | 我们（本轮修复后） |
|---|---|---|
| 三个产物的边界 | milestone（决策、参数与契约，**从不含实现体**）/ 硬件知识库（**唯一含代码的产物**）/ meta-milestone（蒸馏策略）。`README_zh.md:134-137` 明说这三者之间的边界就是「保证参照的代码不会进入 kernel」的构件 | **没有「milestone 不得含实现体」这条规则**（已 grep 确认）。在上游那边这是承重墙（teacher 能读源码，milestone 是唯一泄漏通道）；在我们这边因为 hook 全拦所以是第二层 —— 但逃生舱一开、或封存漏挑文件，它就是唯一那道墙 |
| 规则数 | 约 30 条，**authoring / patch 双模**：新蒸馏的 v1 全量严格，补丁只约束它新增的门禁行，并把「只改数值的重调」与「换了 VERIFY 类别的新主张」区分开（原话：一条保留数字但换了类别的行，是旧主张的词穿着新衣服） | V0–V11 十二条，**单一模式，且不与上一版做差**。于是悄悄放宽阈值或删掉机制行完全无痕 |
| 强制裁定 | 四条正交轴（同步 / 取数 / 相位 / 操作数源），每条**必须表态**（哪怕表态为 `none`），非 none 即生出一条条件门禁义务，某些取值还生出**转录义务** | 12 个标签是「写了才查」。架构层只要**不写**某条机制行，那个维度完全不受约束 |
| 门禁行状态 | 四态：PASS（附 `[MEASURED, job]` 证据）/ DEFERRED / INFEASIBLE / N/A。**没有 FAIL** | 已修：删掉 FAIL（它看起来像终态却零举证——实测把全部行改成 FAIL 校验照样通过）；PASS 要写 `closed_by` |
| DEFERRED 纪律 | 链式下传每一跳重发三条件；接手门禁是一次**反伪实验**；最后一个结构 milestone 必须终裁（原话：不允许任何工作带着未裁定的行结束）；正确性行与「本 milestone 自己握着手段的指标」不得推迟 | 已修：下游手段必须存在且在更靠后的 milestone、最后一个 milestone 不得留 DEFERRED、正确性行不得推迟。**链式下传的「每跳重发」与「接手门禁是反伪实验」没有** |
| PRICED | 结构类门禁行必须带 `[PRICED: <没有它慢多少>, <job N \| KB \| ref-price>]`。原话：SASS 形状说的是参考长什么样；**只有探针测出这个形状值多少钱，它才成为门禁行** | `lever_probe` 有定价能力，但**没有任何规则要求门禁行先定价**，`campaign.py` 也从不读 `levers.json` |
| 不许钉分配结果 | `_ALLOC_PIN` 机械拒绝把 `regs_per_thread == N` 这类**在参考上观测到的分配结果**写成门禁行。门禁只写上界 + 溢出为零，具体数字留给实现选 | 已修：`schema.md` 原先**主动教**这个反模式（「一个声称用 48 个核的决策要能对上从参考转录来的 block_dim」），现已改写并给出两种用法的对照表。**但校验器没有机械拒绝** |
| M0 约束 | 七项：无绝对性能行、拓扑与签名冻结、**强制完工对账审计**、整机 smoke 证书作为门禁行入场券、高风险机制逐个上机、两次失败必须降级、占位实现必须登记 | 已修：M0 不该有性能行（此前是**强制要求**，与上游相反）；V8 扩到扫 M0 全部门禁行。**smoke 证书、占位实现登记、两次失败降级都没有** |
| 阈值带宽 | 停顿类机制行默认取参考实测 ×1.25，更宽要在台账写理由。实证：一条锚在参考 1.6 的行被写成 ≤6.0，学生以两倍于参考的等待量舒服坐在带内、轮轮过门禁 | 已修：`MECHANISM_BAND_FACTOR = 1.25`，超带且无 `band_rationale` 判 refuted |

**差异性质：A（双模校验、强制裁定、PRICED、不许钉分配结果、milestone 不含实现体）**

这一步是 A 类最集中的地方 —— `milestone.py` 是纯逻辑，不依赖 CUDA。

**待决策**：
- [x] 新增一条规则：milestone 不得含实现体（代码块、函数体、`__aicore__` 等）
- [x] 做差校验：`validate --against <上一版>`，对阈值放宽、机制行删除、比较符翻转产出违规（**没有 `--mode` 开关**，形态是「全量规则 + 给了上一版才跑的 PATCH_RULES」）
- [x] milestone 顶层加必填 `adjudications`（闭合词汇表 + 显式 `none`），非 none 即要求对应标签的机制行
- [~] V14 强制 `priced: {delta_pct, source}` 且校验 delta 为正、source 非空，但**不与 `levers.json` 对账**（价格仍是自述的数）；触发条件是「这行带 `structural_anchor`」而不是「这行属于结构类」，没带 anchor 的结构行逃得掉
- [x] 校验器拒绝「指标 == 参考实测值」形状的门禁行（只允许上界/下界 + 配对的溢出行）
- [ ] 保留 V0（上游无对应物：空文档/键名拼错/门禁行写成一句话）

---

## ⑥ 实现与转发围栏

| | 上游 | 我们（本轮修复后） |
|---|---|---|
| 执行点 | `kernel_fence(problem)` 在 **judge 内部**、correctness 与 ncu 两条路径各跑一次，命中就 emit `{"status": "FENCE"}`，driver 当作 correctness FAIL | `fence_check.py` 是独立 CLI，**没有调用方**。只在 `forge-coder.md` 里用提示词要求先跑；`record` 不收围栏证据 |
| 规则维度 | 按**通道**禁，不按算子名：`ctypes/cffi/subprocess/importlib/dlopen/dlsym/RTLD_/load_library/cpp_extension/.so/sys.modules/inspect`，另加 `deny_modules` 整词，连 `##`（token pasting）都算 | 五条规则里四条由算子名推导。已修：多形态算子名（驼峰输入此前让**全部规则失配**）、词边界匹配。**通道类规则一条都没有** |
| 跳过目录 | 不跳过 | `SKIP_DIRS` 含 `third_party` → 从被跳过的目录 import 转发实现，围栏看不见 |
| 代码身份 | 每轮记 `sha`（落地提交）、`base_sha`（**构建在哪个 HEAD 上**，原话：是血统不是顺序）、`cluster`（哪台机器）、`reading_of`（这一行的读数复用了哪一行） | 台账没有这些字段。于是 `previous_snapshot` 的整个前提「被保留下来的最后一条就是当前代码的样子」**无法核实** |
| 构建期 | 第三个裁决 `NEUTRAL`（已提交、未评测）；构建期子步不买整机评测，sha 留作恢复检查点；门禁行永远有整机读数且必须单独成行 | 取不到增益就 `revert` 并计入实现失败。M0 搭建阶段只有两个出口：不 record（恢复点与统计失真）或一堆 revert |
| 构建链 | driver 自己管 | 已修：`forge-coder.md` 补上 `build_ops → generate_pybind → 评测`，并写清 cmake/setup.py 产出的 torch 扩展走不通 |

**差异性质：A（通道规则、代码身份字段）+ B（fence 进裁决、NEUTRAL）**

**待决策**：
- [x] 六条通道规则（`F-DLOPEN` / `F-FFI` / `F-OPAPI-SO` / `F-ACLNN-ANY` / `F-SUBPROCESS` / `F-DYNIMPORT`），**advisory 档**：报告但不自动判不通过。
      分档的理由是误伤代价 —— 正当实现里也会出现这些构造（装载自己编的 .so、起子进程跑构建），
      而一次误伤会让人 `--exempt` 整条关掉。`record` 据此给第三态 `clean_with_channels`：不压 best，但台账留痕
- [x] `third_party` 已从 SKIP_DIRS 拿掉。它是围栏上最大的一个洞：把转发实现放进去再 import，整棵目录连扫都不扫
- [x] `record` 新增 `--fence-report`：命中则回滚并归因实现；**没传则不抬高 best**。
      没做成硬必填 —— 硬拒会把一批正当用法一起卡住，而人会反射性地去加豁免参数，
      那和静默放行是同一件事。压住 best 既有牙齿、又不堵路：best 是 promote 封进
      下一代参考的那个东西，压住它就等于「这一轮不能成为下一代的起点」
- [~] `artifact_sha` / `base_sha` / `milestone_sha256` 有了，`reading_of` 只在 confirm 条目上，`device_id` 只在 state 顶层（但有结果与 state 卡号的强制对账）；**`diff_signature` 完全没有**
- [x] 新增 `campaign.py neutral` 子命令（独立命令而非 `record --neutral` ——它不读结果文件，挂在 record 下会让那些必填参数失去意义）。
      它解决的是：重命名、拆函数、提常量而取值不变这类改动仍然改变**代码身份**，而机制做差的前提是
      「最后一条台账记录就是当前代码的样子」。没有这条路时只有两条选择 ——为必然无变化的改动花一次机时，
      或者不记（而不记是免费的，所以它是会实际发生的那条）。
      指纹与上一条相同时拒绝：那常常意味着改动没落到被测的产物上。

---

## ⑦ 配对计时与精度

| | 上游 | 我们（本轮修复后） |
|---|---|---|
| 精度判据 | **对高精度真值的误差比**：参考在 float64 重跑得真值，判 `e_cand > K·e_ref + max(FLOOR_ULPS·u(dtype), 2^-17)`，K=3。模块第一段就是针对容差法写的：容差检查分不出「在参考舍入的地方舍入」和「用 bf16 累加 / tf32 相乘 / fp8 量化」—— 1e-2 下两者都过。B200 上 745 个降精度读数抓住 643 个，其中 194 个是官方容差放过的 | `torch.allclose(rtol=1e-3, atol=1e-3)` 对 **baseline 自己的输出** |
| 锚点漂移 | 真值是 float64 重算，**不随代际移动** | `promote` 之后 baseline 变成上一代产物，容差重新以它为锚 —— **每代允许再漂 1e-3，N 代后与数学真值的距离无界，而每一代都显示「精度通过」** |
| 压力输入 | 三种 stress profile（offset / spikes / amp），覆盖自然抽样到不了的范围 | 无 |
| 缺失处理 | strict 下**缺 block 等于 correctness 失败** | 已修：字段缺失判 revert（此前返回 `None` 直接落到增益判据） |
| 测量元数据 | 结果里带窗口、workload 数、host 比 | 已修：`measurement` 段（镜像轮数 / burn-in / warmup / 卡号 / 是否走缓存） |

**差异性质：A —— 这是整份清单里代价收益比最好的一条**

精度判据不依赖 NCU：只要能在 CPU 上用 float64 重算参考语义就行，而 `call_spec` 已经给了调用形态。

**待决策**：
- [x] 精度锚到不随代际移动的真值（CPU/fp64 重算，或仓内 `ops-precision-standard` 的 MARE/MERE 判据）
- [x] 判据换成误差比 `e_cand > K·e_ref + floor`，K 与 floor 按昇腾 dtype 标定
- [ ] 加三种 stress profile
- [ ] 保留现状：继续用 1e-3 容差对 baseline（承认逐代漂移，并在 promote 时记录累计漂移次数）

---

## ⑧ 裁决

| | 上游 | 我们（本轮修复后） |
|---|---|---|
| 裁决枚举 | 内层 `KEEP / REVERT / NEUTRAL`；外层 12 个（含 `HOLD_PATCH` / `AUDIT_CLEAN` / `UNPRICED_REOPEN` / `REOPEN_CAPPED` / `INFEASIBLE_STOP` / `INFEASIBLE_REFUSED`） | `accept / accept_unverified / revert / regression`，已修新增 `confirm` |
| 「变快了但解释不清」 | 不是一个裁决状态，而是**门禁的裁决耦合**：端到端 PASS 但机制行没中 → **这个 milestone 整体不得关闭**，先重新归因增益从哪来 | `accept_unverified` 在「不抬 best」这一半等价，**缺另一半**（课程表照样可以关） |
| 输入可信 | harness 文件被改就 REVERT；卡片自己那份 harness 字节 overlay 到每一个 job；解析不出就客观记一条 REVERT | 已修：`gating` / 卡号 / 本轮抖动三道入口门禁 |
| 跳变确认 | `watermark_margin = 0.03`：一个 KEEP 要抬 best 且跳幅超 margin，**先买一次确认 job**，两读一致则取中位数而不是那次快读 | 已修：`confirm` 子命令（两次都达标才收工，best 取较小） |
| 停机后确认 | 对 best sha 复测 **3 job**，median 作最终 claim，还检查三个样本是不是都落在同一台机器上（那是一次抽样不是三次） | 两次读数。`promote` 要求 best 经过复测 |
| 看似达标 | `_confirm_apparent_target`：必须先买一次同期测量才允许结束整个 run。实测事故：一个自参考 run 在 iteration 0 就「达标」，字节相同的代码，锚点和 student job 落在不同节点，差 5.1%，目标是 5% | 已修（同 confirm） |

**差异性质：B（裁决耦合、harness overlay）+ A（3 job 确认）**

**待决策**：
- [x] `check_gate_coupling`：机制判 refuted/vacuous 而课程表那条行标着 PASS 时，台账记一条 `gate_coupling` 并**说明该修哪一侧**（由归因决定）。
      **只报告不改文件** —— record 不该去写课程表，那是架构层的产物
- [x] `CONFIRM_READINGS = 3`，best 取**中位数**。
      从「每一次都达标」改成「中位达标」：前者对一次**偏低**抽样没有抵抗力，而偏低和偏高一样常见 ——
      于是一个真达标的结果会被一次运气差的复测否掉，然后再测一次，机时反而更费。
      **代价：每次确认多花一次机时**（2 → 3）。这是 A 类同步的直接后果，若机时更紧可把 CONFIRM_READINGS 改回 2。
      上游在这个位置还查「三个样本是不是都落在同一台机器上」—— 那条**无法照搬**（本仓按设计单卡，
      跨机采样会破坏可比性，样本必然同机）。换成查三读之间的离散度：差得比噪声底 ×3 还多，
      说明它们不是同一个量的三次抽样，中位数对它们没有意义。
- [ ] 保留两次（机时考虑）

---

## ⑨ 机制核验

| | 上游 | 我们 |
|---|---|---|
| 判据 | **没有**「阈值 + 字段移动」这个机械序对。三层是：机制行必须写明期望方向**和幅度**；阈值受参考值约束（×1.25）；「字段是否真动了」由 meta agent 读 `ncu-trace` 的跨轮演化表判断 | **序对判据是我们自创的，而且更强**（`campaign.py:387+`）。已修：阈值绑参考 ×1.25 |
| 平凡通过 | 有对应物但不是裁决状态：DEFERRED 条件 2「指标逐位相同或朝反方向变化，两者都算不响应」，由 agent 读教条判断 | `vacuous` 机械化了它，并用比较符反推方向（避免维护一张会漂的增减表）。**我们更强** |
| 归因分流 | 同一观测**明确分流**：声称无冲突布局而冲突仍高 ⇒ **学生没做到（退化回退）⇒ 加知识库卡片，不要改课程表**；只有「参考侧该字段本来就不在声称的位置」才是课程表缺陷 | `refuted` 的注释断言得很硬：「这**不是**流程问题，而是对参考行为的解释错了」→ 连续三次把架构层叫回来重写课程表。**结论与上游相反** —— 改对的东西被改坏、改坏的东西没人碰 |
| 静态反驳 | `orch_tools` 在**任何性能数之前**就能反驳：蓝图说每半块两个 barrier，编译出的 SASS 回答它们存不存在。路由规则：未落地的结构**优先于**改验收措辞 | 只能被**一个停顿比例**反驳。候选源码我们完全可读（hook 只拦参考），而「`TQue` 深度是不是 2」「`SetFlag/WaitFlag` 是否成对」这类结构事实从源码直接可判 —— 这条通道一条也没用 |
| M0 对账 | `m0_audit_prompt.md:5-9` 原话：门禁只拿课程表自己声明的值做比较，**整个门禁体系里没有任何地方回答「课程表本身画对了吗」**。于是首次整机 PASS 时**强制停机**，对每个与参考差 ≥2× 的指标三择一裁定 | 已修：新增 `campaign audit` 子命令（逐通道并排、≥2× 且无机制行锚着即报）。**但不是强制停机** —— 没有任何地方要求它必须跑过 |

**差异性质：D（序对判据、vacuous 保留）+ A（归因分流、静态反驳）+ B（对账强制化）**

**待决策**：
- [x] 分流不再依赖任何可选输入。判据是**参考快照**（init 必填），
      问的是「这条行声称的位置与参考实际位置差多远」，带宽复用
      `MECHANISM_BAND_FACTOR`（1.25）—— 和验收线的带宽纪律同一个常数。
      - 带内 → `implementation`（补经验卡片，不改课程表）
      - 带外 → `curriculum`（重开决策 —— 实现正被要求追一个参考自己都达不到的数）
      - 判不了 → `implementation`，因为那是一个**会响**的失败（计数爬到阈值 →
        状态转 `blocked` → 交给人），而反向误判不会响
      **不**用「参考是否满足这条行的阈值」来判：那个判法在最常见的写法上就错 ——
      课程表写 `< 4.2`、参考实测正好 4.2（要求实现追平参考），`4.2 < 4.2` 为假，
      于是一条完全正当的行被判成「读错了参考」
- [x] `mechanism-labels.md` 每个标签补可选 `structural_anchor`（候选源码里必须出现的构造），`record` 先验结构再验停顿
- [x] `audit` 把结果盖进 state（`m0_audit`），`next_action` 在本代没对过账或对账未通过时返回 `audit` 动作。
      **没做成硬停机** —— 硬停会在本代还没有任何读数时就把流程卡住（那时没有签名可对账），
      而一个在正当起点上就报错的门禁会被反射性绕过。所以条件是「本代已有机制已核实的读数」。
- [ ] 保留现状：`audit` 作为可选工具

---

## ⑩ 外层重入

| | 上游 | 我们 |
|---|---|---|
| 平台期判定 | `detect_plateau`：窗口内有没有**新下界**（window=3，noise_floor=1.5%）。重复拿同样大小的增益也算停滞 | 任何 `accept` / `accept_unverified` 都把停滞计数清零，而 `best` 只在 `gain > best` 时才动。于是每轮都交出 +3%（相对参考）时，状态**永远是 implement** |
| 重开定价 | **「没有价格就没有重开」**：`reopen_price: <Δ> via probe:<path>#<n> \| lever:<name> \| KB:<entry>`，`_verify_reopen_price` 拿它比对本步探针读数/定价表/知识库条目，对不上就拒绝、记 `UNPRICED_REOPEN`、**课程表原样不动** | `reopen` 只要 `--decision` 和一句自由文本 `--reason` |
| 徒劳上限 | `reopen_max_fruitless = 2`：连续两次重开以 `REVERT_PATCH` 收场后，再提重开直接拒（`REOPEN_CAPPED`）。计数**从磁盘读**，进程重启仍有效 | 没有任何地方统计重开了几次、几次白干。每次 `reopen` 把三个计数器全部归零 |
| 作废传播 | `invalidation.py` 把 `companions`（原子组）与 `depends_on` 并列进不动点；作废是**物理**的：git 树回到检查点，milestone 一并还原 | `dependents_closure` 只走 `depends_on`（忽略 `atomic_group`）；**磁盘上什么都不变**，也没有可回退的产物快照 |
| 补丁评价 | `_judge_patch`（`_PATCH_IMPROVE_EPS = 0.015`，刻意与 plateau 噪声底对齐）：`KEEP_PATCH` / `HOLD_PATCH`（补丁后那段没产出任何读数，事务保持打开）/ `REVERT_PATCH`（还原 milestone + 回退树） | 没有对应物。课程表只有一份、原地覆盖、不留旧版 —— **连回滚都做不到** |
| 课程表记忆 | `reflect_summary` 把 `REVERT_PATCH` 的方向标成 `[dead end] ... do not repeat this direction` 注入下一次诊断 | 死路表记的是**实现方向**（决策级）。课程表的修订方向没有任何记录 |
| 实现失败 | `max_consecutive_reverts = 3` → `return "stuck"` → 一次 teacher meta step | `implementation_failures` 只递增、只在 status 里打印，`_advance_state` 从不读它。连续 30 次精度不过，`next_action` 每次都说 `implement` |
| 预算 | 三维：墙钟 `hours = 4`、`max_iters = 60`、`max_meta = 3`，另有软预留（距墙钟 1 小时内不再开定价作业） | 已修：评测次数默认 60。**没有墙钟，没有外层重入次数上限** |

**差异性质：A（定价、徒劳上限、停滞判据、补丁评价、实现失败动作）+ B（物理回退、墙钟）**

这一步 A 类最多，而且全是纯状态机逻辑。

**待决策**：
- [x] 停滞递增判据改为「本轮没有抬高 best」，而不是「增益没超噪声底」（此前已完成，勾选状态是旧的）
- [x] `reopen` 必填 `--price "<Δ%> via lever:<name>|probe:<path>"`，拿 `levers.json` 校验引用存在且 Δ 超噪声底（此前已完成，勾选状态是旧的）
- [x] state 加 `reopen_fruitless` 计数，超 2 次拒绝重开（落盘，不随进程归零）（此前已完成，勾选状态是旧的）
- [x] `dependents_closure` 的种子集加入同 `atomic_group` 成员（此前已完成，勾选状态是旧的）
- [x] `reopen` 记 `patch` 条目；`outcome` 在下一次抬高 best 时回填为 `accepted`。`status` 列出已试过的课程表方向。
      回填是必须的：否则一次**成功的**修订也永远停在 pending，于是 status 拿它去劝退下一代 —— 只记投入不记产出的账比没有更糟
- [x] `implementation_failures` 超阈值进独立的 `STATE_BLOCKED`（不走架构重入）（此前已完成，勾选状态是旧的）
- [x] `rollback_target`：作废范围之外、最后一个机制已核实且有 `artifact_sha` 的产物。找不到就说「从参考基线重新开始」—— 不编一个看起来合理的回退点
- [x] `--wall-hours` / `set-budget --max-wall-hours`，按 `created_at` 算差。明确标注「含停机时间，是机时的上界」—— 误差方向保守（提前提醒）而不是漏

---

## ⑪ 收工与自我提升

| | 上游 | 我们（本轮修复后） |
|---|---|---|
| 停止条件 | 两条：增益 ≤ `1.0 + floor`；**这一代结束时没有「确认过的最好 kernel」** | 第一条有。第二条靠产物库准入门禁间接挡（`gen>0 且 mechanism_verified is not True` 拒绝入库），不是显式停止条件 |
| 封存下一代 | `self_evolve.py` 把最好 kernel 拷进新任务的 `reference/`，**每一代的任务自成一体**（可提交进 git、在别处重跑） | `promote --library`：拷贝 + 校验和 + 血统 + 机制门禁。已修：**真的调 seal**（此前 `sealed: True` 只是个字段），参考改指库里的只读拷贝 |
| 角色 | 第一代之后强制 source 模式：teacher 读上一代最好版本，student 读不到 | 两个 agent 都读不到（见 ① 的角色不对称） |
| 目标步进 | `--step 1.05` | `--phase-target-gain` 默认 1.05（一致） |
| 重置 | 新任务是全新 run，不继承任何东西 | 已修 + 原有：best 归零、噪声底待重测、参考签名清空、三个计数器归零、台账按代过滤 |
| 产物库 | **没有**共享产物库这个东西 | 整库文件锁覆盖「读索引→算代号→复制→写回」+ 真多进程压测。**我们更强** |

**差异性质：D（产物库）+ B（角色不对称）+ A（第二条停止条件显式化）**

**待决策**：
- [ ] 把「这一代没有确认过的 best」做成显式停止条件（现在是入库时才报错，时机太晚）
- [ ] 保留产物库（上游无对应物，不要替换）

---

## ⑫ 知识沉淀

| | 上游 | 我们（本轮修复后） |
|---|---|---|
| 词表来源 | `derive_reference_tokens`：每次运行从**解析后的参考目录机械抽取**（文件名 + def/class 名），注释原话：*「so a source-mode operator is covered without anyone remembering to edit this file」*；缺路径**显式报警**而非静默降级 | 已修：`seal` 同步写入知识库根（此前 `seal` 写 `forge_work/`、`hwkb` 读 `kb_root/`，无人搬运 → **L-DENY 从不触发**，而一条名叫「两段不能脱节」的测试自己手工把两段接上） |
| 启动门禁 | 每次 loop 启动对整棵树跑 `lint_kb_dir`，有命中就**拒绝启动**；KB 有未收获的改动时也拒绝启动（exit 3） | 已修：`lint` / `validate` / `add` 都报告 L-DENY 是否生效，结论里标明覆盖范围。**没有「拒绝启动」这个时刻** |
| 认证 | 是一个**动作**：探针推到 GPU、解析 `PROBE PASS`，之后 teacher 才被允许发 `RECIPE_CERTIFIED`；**引用时再查一次**（`_KB_CERT_MARK`），不带印记的卡片拒绝作为价格来源（原话：a transcribed recipe is not a price） | 已修：`certified_by` 可带 `state`+`decision` 指针，入库时回台账核对 `kb_admissible`（增益超噪声底**且**机制已核实）。无指针时告警不拒收 |
| 自动收获 | 两个自动欠账：任何探针仗烧掉 ≥ `kb_tripwire_rounds`（默认 6）轮 GPU 后才赢 → 入欠账；任何 KEEP 的契约轴无计数签名发生变化 → 入欠账。欠账在产生那一刻打日志，harvest 时把前后轴读数**原样注入** prompt，写完再 lint 一轮带一次返修 | **没有任何对应物**。`recipes_invented` 是 agent 手传的标志位；唯一的提示在 `AGENTS.md` 步骤 11（长 campaign 的末尾 —— 正是 context 吃紧、最容易跳过记账的时刻）。本仓就是证据：整套 dav2201 流程建完，库里 1 张卡 |
| 人类专用归档 | 按**读者**劈两层：卡片的 Sources 给「学生 + 人」，只能是官方文档/公开 API 符号/本仓探针判定；参考实现的文件名、函数名、行号、转述内容全部去 `docs/kb-provenance/<entry>.md`，该目录**不在任何 agent 的读集里** | 三类来源全部在卡片内部。**没有人类专用归档** —— 真从参考行为反推出证据的 agent 没有合法位置安放它 |
| 禁用词 / 叙事 | 直接禁 `oracle` / `teacher` / `reference implementation`，边界**只认 ASCII**（注释写明：Python 的 `\b` 把 CJK 当词字符，`oracle` 后紧跟一个中文字就会溜过去，「而 agent 确实会往条目里写 CJK 散文」）；另禁运行日期、`iter N`、commit sha | 一条都没有。而我们的卡片**强制中文**，恰好落在上游专门踩过的那个坑上 |
| 复用 | 三层：架构树每轮无条件进 student 读集（index 是一张「症状 → 机制 + 结论」密集表）；G3 认证后 driver 在**下一个** prompt 里点名新卡片并要求先读；**带印记的卡片可以替代一次新测量**为重开定价 —— 读知识库能省一次 GPU job | `hwkb query` 按症状检索 + agent 自己决定用不用。引用卡片是**纯成本**，没有经济激励。`cmd_add` 不更新 INDEX.md，而 `iter_cards` 又跳过 INDEX |
| 隔离 | lint 不过先返修两轮，仍不过就把文件**移出学生可读的树**到 `kb_quarantine/` | 已修：`lint --all` 报 `L-PARSE`（此前跳过解析不动的卡片 —— 而那正是粘了源码的卡片）。**没有隔离区** |
| 死路表 | 劈两半：**按算子**在 milestone 的 §4，每行带 `retry_precondition`（学生被告知那是重试前置条件表而不是禁令表）；bundle 被 REVERT 时**禁的是组合不是成员** | 一张全局扁平 `nogo.json`，按症状子串命中。已修：损坏不再被读成空表、精度失败不再登记为死路。**没有 retry_precondition、没有形状范围、`nogo check` 不看已记录的 operator 与 input_shape** |

**差异性质：A（词表抽取、禁用词、叙事 lint、retry_precondition）+ B（启动门禁、自动收获、隔离区）**

**已做（本轮）**：
- [x] 加禁用词表（`oracle` / `teacher` / `参考实现` …），边界**只认 ASCII** → `L-BANNED`
- [x] 叙事规则（日期 / 第 N 轮 / `iter N` / commit sha / 算子名）→ `L-NARRATIVE`；
      `certified_by` 与 `provenance` 刻意不扫 —— 那两处的职责是指向一次具体测量
- [x] `references/provenance/<id>.md` 人类专用归档 + 必填 `provenance_archive` 指针；
      `hooks.json` 的 `permissions.deny` 挡住 `Read/Grep/Glob` —— 目录布局只保证
      「不主动递给它」，deny 才保证「它拿不到」
- [x] 算子名清单 `operator_names.txt` 由 `seal` 与 denylist **同一次写入**
      （多一个独立写入点就是多一条可能断掉的线，L-DENY 当初就是这么失效的）
- [x] 两道外部文件驱动的闸在失效时点名报告，结论行写明「这句话不覆盖它们」
- [x] `campaign.py harvest`：`admissible`（判了 `kb_admissible` 而没写卡片）+
      `tripwire`（烧掉 ≥ 6 轮才过）两类欠账；输出带着写卡片要用的读数
- [x] `promote` 在欠账非空时拒绝，位置排在确认复测之后、正式封存之前；
      收敛那条分支只告警不拦（否则状态机卡在 `implement`）
- [x] 死路表每条必填 `retry_precondition`，加 `input_shape`，`--arch` 改必填；
      `check` 把命中分「有约束力」与「仅参考」两档
- [x] `add` 自动重写 `INDEX.md` 的标记区间（手写导读原样保留）；
      `lint --all` 比对自动段与目录内容（`L-INDEX`）
- [x] `lint --all` 同时跑 `check_admission` + `check_certification_against_ledger`；
      `--quarantine` 把不过检的卡片移到 `references/quarantine/`
- [x] `kb:<卡片 id>` 可以当重开价格 —— 但要卡片自己过准入校验，且记录幅度
      ≥ 声称的 Δ（「一条转述过来的处方不是价格」）

**还没做**：
- [x] `forge-arch` 的步骤表里，读 `INDEX.md` 排在 `query` **之前**并写明「无条件读完」。
      `query` 是拉取式的 —— agent 得先怀疑「这个现象别人遇到过」才会去查，而它通常不会怀疑。
- [x] `init --kb-root` 开跑前把整棵树过一遍边界 lint，有命中就拒绝启动（`--allow-dirty-kb` 显式放过）。
      放在 init 的理由：知识库被读的时机是架构层写课程表的时候，此时拦下来**还没花任何机时**；
      跑到中途才发现，那张卡片已经参与过几轮决策，而哪几轮受了影响无从分辨。
- [x] `QUARANTINE_AFTER_FAILURES = 2`，计数记在卡片 frontmatter 里（跨进程存活），过检后清零。
      frontmatter 读不动的卡片直接隔离 —— 计数没处记，而那类卡片最常见的成因恰恰是正文里粘了源码。

---

## 形态冲突清单（B 类，不能照搬）

这几条上游靠「常驻 driver 进程」实现，我们是插件，需要换执行点：

| 上游的时刻 | 我们可挂的位置 | 代价 |
|---|---|---|
| `loop start` 校验（window 契约、脏 KB、未收获、job-env） | `campaign.py init` + `record` 前置 | init 变重；record 每轮多几次检查 |
| 每回合重写 settings.json | 无等价物。只能靠 hooks.json 静态配置 + Edit/Write 保护 | 策略可被改写的窗口关不严 |
| 墙钟预算与软预留 | `record` 时按 init 时间戳算差 | 只在调用 record 时才生效，中间挂住不会被发现 |
| 物理回退（git reset + milestone 还原） | 台账记 artifact 哈希 + `reopen` 输出回退命令，由人或 agent 执行 | 回退不是强制的 |
| 构建期 NEUTRAL | `record --neutral`（要 `--sha`，不要 `--result`） | 需要先有代码身份字段 |
| 自动 harvest 注入下一个 prompt | `campaign.py harvest` 产出清单 + `promote` 门禁 | 不是自动的，靠门禁逼 |

---

## 平台不可移植清单（C 类，建议保留并在文档里承认）

| 上游能力 | 昇腾侧情况 |
|---|---|
| SASS 静态普查（结构/操作数源/调度深度/launch 形状） | **未查证**是否有 AI Core 二进制反汇编通道。`instrument-map.md` 现在写的是「列为开放问题」，而上游对这种状态有明确规矩：合法的三态是 delivered / **waived（必须带理由和已试通道清单）** / delivered-mismatch |
| timeline 的 ramp / plateau / drain | msprof 算子级无时间线。**但逐核 cycle 已归档**，能解释 exposure 的「尾部收敛」那一个分量 |
| orchestration 同步拓扑普查（mbarrier / named barrier 的 18 个结构轴） | 编程模型不同（`SetFlag/WaitFlag`、BufferID），无直接对应物。但**角色暴露账**有通道（所有输入已在手） |
| window.toml 的 asymmetry 定价 | 我们两侧共用 call_spec、同一批输入、同一个派发路径，结构上不会不对称 |

---

## 我们更强的地方（D 类，建议明确保留）

| 项 | 上游情况 |
|---|---|
| 「阈值成立 + 字段因本次改动而移动」序对判据 | 没有。靠 agent 读跨轮演化表判断 |
| `vacuous`（平凡通过）单列一类，用比较符反推方向 | 没有同名概念 |
| 机制核验进裁决：`accept_unverified` 不抬 best + 产物库 `mechanism_verified` 准入门禁 | `self_evolve.py` 只看增益超噪声底，不要任何机制证据 |
| 通道歧义检查（同名指标在 Cube/Vector 各一份时拒绝猜） | NCU 指标名天然唯一，无此需要 |
| 产物库跨任务整库锁 + 真多进程压测 | 单 driver 进程，**没有**共享产物库 |
| 公开接口白名单自动从 `ascendc/include` 反推 | 同类问题只靠文档提醒人别填错 |
| 结构化的「架构假设 + 开放问题」对象（带 claim/channel/strength/caveat） | 前缀键文本 + prompt 规则，假设形成完全在模型手里 |
| V0 课程表结构完整（空文档 / 键名拼错 / 门禁行写成一句话） | 不机器校验卡片与课程表结构 |
| 配置 json 一并封存 | 它的 `problem.json` 是 card 自己的，无此需要 |
| 噪声底拒绝崩溃评测（`gating=failed` / cv 全 0 / 自评 speedup 被钉成 1.0） | 没有对应物 |

---

## 怎么用这份表

每一步的「待决策」里勾选。建议的推进顺序：

1. **先做 A 类里的纯状态机部分**（⑩ 的停滞判据、定价、徒劳上限；⑥ 的代码身份字段）—— 不碰测量、不碰平台，改完就能看出循环行为的差别
2. **再做精度判据**（⑦）—— 单条收益最大，但要先确定昇腾侧 float64 真值怎么算
3. **再做课程表双模与强制裁定**（⑤）—— 纯逻辑，但会让现有课程表全部需要重写
4. **B 类最后做**，因为它们要重新设计执行点，而且会改变 `init` / `record` 的调用契约
5. **C 类只改文档**：把夸大与漏报的地方按 waiver 格式写清（已试通道 + 理由），不要写成「开放问题」
