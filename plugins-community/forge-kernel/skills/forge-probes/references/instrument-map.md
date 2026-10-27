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

### timeline probe → 暴露时间推导

论文 §3.2：「串行延迟受限的 kernel 还会额外上一次 timeline probe。」

**昇腾上 msprof 的算子级数据没有时间线视图。** 没有等价物。

能做的替代是从已有字段推导**暴露时间**：

```
暴露时间 = Task Duration − max(aic_time, aiv_time)
```

取 max 而非相加，因为两者在 MIX 算子里并行；相加会高估覆盖、低估暴露，
从而把一个真有问题的 kernel 显示成健康的。

**差在哪里**：timeline probe 告诉你时间花在哪一段，暴露时间只告诉你有
多少时间闲置。所以它的作用是**判断要不要继续深挖**，不是给出结论。
判为串行延迟受限之后，下一步应当用 `lever_probe.py` 做消融定位。

**工具**：`signature_probe.py` 的 exposure 段。判定阈值 20% 是本工具的
工程判断，**无文档出处**，输出里据实标为 derived。

### occupancy → 无直接对应，可用核数占比代理

NCU 的 occupancy 是「实际驻留 warp 数 / 理论最大」。昇腾没有对应概念。

最接近的代理是 `Block Dim` 与该芯片可用核数之比。但这个比值的含义不同：
它反映的是任务切分是否铺满核，不反映单核内的并发度。

**不要把它当 occupancy 用。** 写进 milestone 时应当如实写成
「核数利用率」，并说明它与 occupancy 不是一回事。

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
| 指令级时间线 | msprof 算子级无此粒度 | 列为开放问题，用消融逼近 |

> 最后一行尤其要守住。把消融结果说成「时间线显示……」是把推断伪装成
> 观测，而这类错误因为看起来有数据支撑，排查起来最费时间。

---

## 五、一条贯穿始终的原则

论文 Appendix C 把 **agent 自评的探针单列为一类 bug 来源**。

落到工具上就是：每个探针都要能回答「这个数没采到时会怎样」。

- `signature_probe.py`：缺失标 `present=false`，不折成 0；停顿数据缺失
  直接拒绝输出架构假设
- `lever_probe.py`：缺单项测量时标 `incomplete`，不硬算耦合
- `fence_check.py`：扫不到源文件时报错，不返回「通过」

把「没数据」读成「没问题」是这类工具最常见也最难发现的失效方式。
