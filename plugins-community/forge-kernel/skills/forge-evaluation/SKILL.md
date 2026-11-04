# forge-evaluation：配对计时与精度判据

本技能是 `ops-perf-evolution/skills/ops-evaluation` 的**分叉**，只供
forge-kernel 使用，另加一个它没有的精度判据。

## 为什么分叉

下面这些改动会改变**测量口径**与机时成本。共享脚本还被 `ops-perf-evolution`
等插件使用，它们的基线与历史数据建立在旧口径上 —— 分叉让 forge-kernel 的
判据纪律不外溢到别的插件。

| 脚本 | 来源 |
|---|---|
| `scripts/evaluate_ops_direct.py` | 分叉自共享脚本，见下方差别表 |
| `scripts/precision_gate.py` | **新增**，共享脚本没有对应物 |
| `scripts/common/` | 从共享脚本整段拷贝，未改 |

---

## 一、配对计时：`evaluate_ops_direct.py`

```bash
python3 scripts/evaluate_ops_direct.py {op_name} \
    --call-spec call_spec.json \
    --baseline-path {参考路径} \
    --evolved-path {已安装的候选 OPP 路径} \
    --device-id 0 --shapes-mode target \
    --output evaluation_results.json
```

### 与共享脚本的差别

全部围绕「同 job 配对计时要可比」这一条。

| 改动 | 为什么 |
|---|---|
| **位置镜像**（A-B-B-A 四轮，每侧取两轮均值） | 共享脚本永远先跑 baseline 再跑 evolved。器件的第一个窗口最快（冷、满 boost），之后随温度漂移 —— 实测同一份代码分别当两侧测差约 **2.7%**，方向恒定、多测不收敛，而目标增益只有 5% |
| **burn-in**（计量前用被测这一侧的 kernel 加热器件） | 只镜像不 burn-in，冷启动偏置只是换到另一侧；只 burn-in 不镜像，慢速热漂移还在。两个机制缺一不可 |
| **warmup 3 → 20** | 3 次不足以越过前几次冷启动 |
| **两侧抖动都记** | baseline 侧的 `cv_pct` 不再硬写 0.0。比值的相对方差是两侧之和，只记一侧会让噪声底系统性偏低。另记 `round_spread_pct` —— 那是**跨进程**的那部分噪声，同进程内的 cv 量不到它 |
| **测量元数据** | 结果里写出 `device_id` 与 `measurement` 段（镜像轮数 / burn-in / warmup / 是否走了 baseline 缓存）。`campaign.py record` 据此核对这份结果是哪张卡上、什么测法得到的 |
| **自包含** | 删掉了指向 `ascendc-evaluation/.../AscendPerformanceTest.py` 的死引用 —— 那个文件在本仓不存在，而计时早已改用 NPU Event |
| **第二个计量窗口** | 一对 Event 套 N 次连发（同步开销摊掉）+ 只下发不等的墙上时间。两个数一比回答一件此前答不了的事：`time_us` 量的主要是 kernel，还是 host 的下发与排队 |
| **输入轮换** | 每个 shape 生成 2 份输入轮换使用 |
| **输入摘要比对** | 计时前后各取一次，抓「kernel 原地改了输入」 |
| **压力分布** | `--stress-profile normal\|wide\|extreme` |

### 为什么要第二个计量窗口

第一个窗口每次都 `sync`，所以它量到的是「一次下发 + 一次执行 + 一次同步」。

机制核验读的是 msprof 的**设备侧**通道。如果 `time_us` 主要由 host 决定，
那「增益超噪声底 + 机制已核实」就是两个互不相关的事实被钉在一起 —— 百分比
来自 host，机制结论来自 device。而这个错很难自己暴露：两个数都正常，只是
它们量的不是同一件事。

`host_bound_ratio = host_enqueue_us / device_amortized_us`，超过 0.5 时
`record` 不抬高 best。

> **这段代码只能在真设备上验证**（开发机无 torch_npu），所以它包在 `try`
> 里：失败时把原因记进 `amortized_error` 而不是让整次评测挂掉。一个新加的
> 诊断量不该有能力毁掉一次已经排到的机时窗口。判定逻辑在 `campaign.py`
> 一侧，已单测。

### 为什么输入要轮换

一份输入反复用在所有 trial 上，有两个具体后果：

| 后果 | 为什么要紧 |
|---|---|
| 缓存命中被烘进读数 | 两侧都热，所以比值不一定偏，但**绝对耗时**偏低 —— 而绝对耗时是噪声底、设备占比、host 受限这几个判据的分母 |
| 按输入身份记忆的实现 | 一个按 `data_ptr + shape` 做 memoize 的实现会从第二个 trial 起直接返回上次结果。它的 cv 会极低、加速比极高，而所有门禁都过 |

轮换两份同时破掉这两件事。取 2 而不是更多是机时考虑 —— 份数越多生成与落盘
越贵，而破掉上面两件事只需要「不止一份」。

第 0 份同时平铺在 shape 目录下：`precision_gate` 与既有的结果目录布局都读
那个位置。不平铺的话这次改动会悄悄断掉精度判据的输入，而它只会报「找不到
输入」，看起来像是没生成。

### 输入摘要比对抓什么

「kernel 原地改了输入」。两层后果，第二层更隐蔽：

- 后面每个 trial 测的不是同一份工作（第 2 次起输入已被第 1 次改过），而 `cv`
  看起来完全正常 —— 它量的是耗时的离散度，不是输入的一致性
- 精度比对拿的是**改过之后**的输入重算真值，于是一个会改输入的实现在精度上
  也显得正确

摘要只取每个张量的形状、dtype、若干采样点之和 —— 全量 hash 在大张量上太贵，
而要查的是「变过没有」，不是「变成了什么」。

`record` 见到 `inputs_mutated` 为真直接拒收这份结果。

### 压力分布：让误差比判据真的触发

误差比判据是**检测**降精度的机制，压力分布是让它**触发**的条件。一个用 bf16
累加的 kernel 在 `randn` 这种良性输入上完全可能过关 —— 那些值都在 1 附近，
累加误差没有机会堆积起来。

| 分布 | 针对哪类降精度 |
|---|---|
| `normal` | 默认。常规输入下对不对 |
| `wide` | 量级跨 6 个数量级。大值小值混在一起时低精度累加器会把小值整个吃掉（大数吃小数），这正是 bf16/fp16 累加的特征 |
| `extreme` | 贴近 dtype 表示范围的边界 + 0 + 极小值。暴露中间量被砍到更低精度时的溢出/下溢 —— fp32 中间量砍成 fp16 时，一个 70000 的中间结果直接变成 inf |

只动浮点：整型与 bool 没有「量级跨度」这回事，硬改只会把它们改成非法索引或
越界值 —— 那测的是错误处理，不是精度。

> **三种分布的实际效果未在硬件上标定。** 它们的构造方式有依据（每一条都对应
> 一个具体的降精度形态），但「它们确实能把降精度实现抓出来」这件事需要一次
> 硬件标定 —— 做法同 `precision_gate` 的 `calibration_note()`：拿一批已知
> 降精度的实现测一遍，看这三种分布下误差比是否真的越线。

压力分布与轮换份数都写进结果的 `measurement` 段。不写出来的话，一份 `normal`
分布下的精度结论和一份 `extreme` 分布下的看起来一模一样，而两者的强度差
很远。

```bash
# 急着看方向时可以退回单轮（省一半机时，但 2.7% 的偏置回来）
python3 scripts/evaluate_ops_direct.py {op_name} \
    --call-spec call_spec.json \
    --baseline-path {参考路径} --evolved-path {候选路径} \
    --device-id 0 --mirror-rounds 1
```

---

## 二、精度判据：`precision_gate.py`

```bash
python3 scripts/precision_gate.py --inputs ./inputs/shape_s0 \
    --baseline-output ./baseline_profiling/shape_s0/baseline_output_s0.pt \
    --candidate-output ./evolved_profiling/shape_s0/evolved_output_s0.pt \
    --torch-reference "torch.nn.functional.layer_norm(a + b, (1024,))" \
    --output ./forge_work/precision.json
```

报告传给 `campaign.py record --precision-report`。

### 为什么不能用容差比对

默认的做法是 `allclose(候选, baseline, rtol=1e-3, atol=1e-3)`。它**分不出
两件完全不同的事**：

| | |
|---|---|
| 诚实实现 | 在参考舍入的地方舍入，只是归约顺序不同 |
| 降精度换速度 | 用 bf16 累加、tf32 相乘、把 fp32 中间量砍成 fp16 |

1e-3 下两者都过。收紧容差会把前一类误杀 —— 归约顺序不同本来就产生这个
量级的差。区分它们要换一个量：**对高精度真值的误差比**。

```
truth    参考语义在 float64 下重算的结果
e_ref    参考自己的输出（原生 dtype）相对 truth 的相对 L2 误差
e_cand   候选输出在同一组输入上的同一个量
判失败   e_cand > K · e_ref + max(FLOOR_ULPS · u(输出 dtype), FLOOR_REL)
```

### 还修掉了一个代际漂移

容差比对的锚是 **baseline 的输出**。而 `promote` 之后 baseline 变成上一代
产物，锚点跟着移动 —— **每代允许再漂 1e-3，N 代之后与数学真值的距离无界，
而每一代都显示「精度通过」**。

误差比的锚是 float64 真值，由参考**语义**算出来，与代际无关。

### 需要 `torch_reference`

误差比需要一个能在 float64 下跑的参考语义。装好的 OPP 二进制跑不了
float64，所以 `call_spec` 要给 `torch_reference`，两种写法：

| 写法 | 例子 |
|---|---|
| 表达式 | `torch.nn.functional.layer_norm(a + b, (1024,))`，其中 `a`/`b`/… 绑定到 `input_0`/`input_1`/…，kwargs 按名绑定 |
| 模块:函数 | `mypkg.ref:add_layer_norm`，收 `(args, kwargs)` |

**拿不到它时报告 `unavailable`，不是报告通过。** 精度判据缺失时静默放行，
等于把「没查」说成「查过且通过」—— 而算错的 kernel 再快也无意义。

### 常数需要在昇腾上重新标定

`K_RATIO = 3` 与 `FLOOR_REL = 2^-17` 取自论文实现的同名判据，而它那组常数是
在 B200 上跨 14 个问题 1956 个读数标定的。`FLOOR_ULPS` 我们没读到上游取值，
默认 1.0。

标定之前这条判据的**方向**是对的（它能分出降精度），**阈值**只是起点。
标定方法：拿一批**诚实**实现（参考在 fp32 下重跑、换归约顺序、已知正确的
历史版本）测出 `e_cand/limit` 的分布，再拿一批**故意降精度**的（bf16 累加、
fp16 中间量）测同一个量，K 取能把两组分开的值。

判据本身可以离线自检，不需要 torch、不需要 NPU：

```bash
# 诚实实现（归约顺序不同）应当通过
python3 scripts/precision_gate.py --check-criterion \
    --e-ref 1e-7 --e-cand 2e-7 --dtype float32

# bf16 累加应当被拦
python3 scripts/precision_gate.py --check-criterion \
    --e-ref 1e-7 --e-cand 1e-3 --dtype float32
```

---

## 与其他技能的关系

| 技能 | 分工 |
|---|---|
| `forge-probes` 的 `signature_probe.py` | 架构签名。配对评测**有意不采 msprof**，所以它输出的 `pipeline` 与 `bottleneck` 恒为空 |
| `forge-probes` 的 `structure_probe.py` | 结构核验。在任何性能数之前判「声称的结构在不在」 |
| `forge-campaign` 的 `noise_floor.py` | 从本脚本的结果量噪声底。自评场景要显式传 `--method cv` |
| `ops-perf-evolution` 的 `ops-evaluation` | 共享脚本的来源。**不要混用** —— 两边的测量口径不同 |
