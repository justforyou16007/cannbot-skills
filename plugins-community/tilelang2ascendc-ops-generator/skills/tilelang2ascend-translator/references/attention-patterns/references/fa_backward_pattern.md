# FA 反向模式（FAG / FlashAttentionScoreGrad）：DSL→AscendC 转译参考

> 本文档是 attention-patterns 体系中的 **FA 反向（backward）模式**，由
> `AttentionPatternIndex.md` 路由（命中信号：`dy`/`softmax_max`/`softmax_sum`/`attention_in`
> 输入 + `dq/dk/dv` 输出）。原文件名为 `dsl2Ascendc_FA_Bwd.md`。

本文档以生产版 FAG 算子（arch22 / Atlas A2\A3 训练系列）为参照实现，总结 FA 反向算子的
计算结构、模板体系、实测性能画像与已验证优化点，用于指导 DSL（TileLang）FA 反向设计
到 AscendC kernel 的转译与调优。

参照实现归档：`output/flash_attention_score_grad_archive/`（arch22 S1s2Bn2gs1s2 通用路径
独立 re-host，8/8 用例精度 PASS，含 verify_archive.py 验证脚本与 docs/perf/ 实测归档）。
设计文档原文：归档 `docs/FAG算子设计介绍.md`。

---

## 1. 计算流程（6 步，P 重计算式反向）

输入：q/k/v/dy + 前向统计量（softmax_max、softmax_sum）+ 前向输出 attention_in，
布局 BNSD，keep_prob=1.0、无 pse 时为最简形态：

```
1. P  = SimpledSoftmax(Mask(Q @ K^T) * scale)        # 重计算前向 P，驻留 UB
2. dP = dy @ V^T
3. dS = P ∘ (dP - FlashSoftmaxGrad(dy, attention_in)) * scale
4. dq = (dS @ K) * scale
5. dk = (dS^T @ Q) * scale
6. dv = P^T @ dy
```

- 共 5 个 matmul（步骤 1/2/4/5/6），Cube 侧负责；softmax 重建、FlashSoftmaxGrad、
  逐元素减/乘为 Vector 侧。
- **转译要点**：dS 的两个 matmul 消费者（dq 用 dS、dk 用 dS^T）意味着 dS 需同时保留
  正向与转置视图，或接受一次转置搬运；P 同理（dv 用 P^T）。转置通过 matmul 的
  transpose 标志消化，不要显式物化转置张量。
- GQA 语义：dk/dv 按 kv head 数（N2）输出，kernel 对 G 组内归约（多核累加到
  workspace 或核内循环归约，取决于模板）。

## 2. 参照实现的 kernel 架构（三段式单入口）

生产 arch22 实现为 Pre → Main → Post 三个阶段；归档 re-host 将三者融合进**单个
kernel 入口顺序执行**（阶段间 `SyncALLCores`）：

| 阶段 | 职责 | 关键事实 |
|---|---|---|
| Pre | fp32 workspace 清零（dq/dk/dv 累积区）；fp32 时直接清零 dvGm | INIT_OUTPUT 模板参数控制；覆盖性是 host 侧免 zeros 的依据 |
| Main（S1s2Bn2gs1s2 模板） | 6 步主计算，CV 协同 | AIC:AIV=1:2，CV 数据配比 1:16（S1 方向 nRatio=8）；dq/dk/dv 以 fp32 累加进 workspace |
| Post | workspace fp32 → 输出 dtype 的 cast + scale（Muls）写回 GM | ND 路径按 qSize/kvSize/vSize **全量覆写**输出 |

**host 侧输出可以 `at::empty` 而不必 `at::zeros`**（实测消除 3 个 ZerosLike kernel，
占 device 时间 11.3%）——成立条件：Pre 清零 + Main 累加 + Post 按 numel 全量写回，
三者保证输出每个元素都被覆写。fp32 特例：Post 不写 dv（CAST_DV 条件不成立），
dv 由 Pre 清零 dvGm + Main 直接累加覆盖。

### 独立构建 / 转译落地的工程坑（实测）

1. `RegisterAscendBinary` 有大小上限：单入口合并二进制 ~8.9MB 可注册，17MB 报
   107000 param invalid → 多 dtype/mask 入口必须拆成多个独立 ascendc_library target。
2. tiling 结构体不能作为按值 kernel 参数（stub 生成器只认 `struct Name {`，生产多为
   `class`）→ 保持 `GM_ADDR tiling` + 设备侧 8B 字逐字拷贝。
3. 独立 host stub 无系统 workspace 偏移：入口内自行 `SetSysWorkspaceForce` +
   `GetUserWorkspace`，host 额外分配 16MB（dav-c220 RESERVED_WORKSPACE）。
4. 生产编译选项对齐：`--cce-auto-sync=off --cce-long-call=true`（kernel 内显式同步）。
5. （S1s2Bn2 小 shape 路径 re-host 实测）mm12 非 fp32 输出为 NZ、BNSD 走 POST2KERNEL
   融合 Post、**scale 施加挂载在 pseType==PSE_OUTER_ADD_MUL_TYPE(1) 分支上**
   （无 pse 也必须置 1，否则 P 未乘 scale 直接 exp 溢出）、cube_op_bn2.h 的裸布局常量
   需在独立 TU 显式补文件作用域定义。调试手段：host 侧 dump user workspace
   分区域比对（S/dP 为 NZ、mm34Input 为 fp16）+ kernel 内 DumpTensor 定位到
   softmax 输入未乘 scale。

## 3. 生产多模板体系（arch22，按优先级路由）

设计文档 §5：模板按 shape 特征路由，序号越小越泛化。**统一走泛化模板是有实测代价的**
（见 §4）：

| 模板 | 核间/核内切分 | 路由条件 | 适用场景 |
|---|---|---|---|
| S1s2Bn2gs1s2（泛化，归档路径） | 核间 BN2GS1，核内 S1/S2 | 支持所有 shape（兜底） | 大 S；小 S 时 CV 交互频繁 |
| S1s2Bn2 | 核间 B/N2，核内 G/S1/S2 | S1<1024 && S2<1024 && (B·N2·2>CoreNum \|\| G==1) | 小 S 大 BN2 / MHA（**归档已 re-host**，实测加限 D≤64：D>64 时移植路径慢于泛化） |
| Ngs1s2Bn | 核间 B/N2.o，核内 N2.i/G/S1/S2 | G·S1·S2≤32KB && G==1 && S2<1536 | 小 G·S1·S2 |
| Bngs1s2B | 核间 B，核内 B.i/N2/G/S1/S2 | N2·G·S1·S2≤64·128 | 极小内层 |
| SameAB（_sab 后缀） | UB 切 S1S2，多核切 S2 | S1≥1024 or S2≥1024；确定性；TND 大 S | Cube 主核视角，V0/V1 复用左矩阵、L0C 累加 |

- `_sab` 后缀 = Cube 主核模板（V0 统一发起 matmul，V0/V1 各处理一半）；
  减少带宽诉求，大部分场景更优（设计文档 §6.1）。
- 确定性计算走独立 det 模板（避免多核同地址累加）。

## 4. 实测性能画像（Ascend910B2，msprof 7 组 aic-metrics）

对归档泛化路径（S1s2Bn2gs1s2）8 个用例（MHA/GQA、fp16/bf16/fp32、causal、D≠Dv、
D=177/192 非常规 head dim）：

### 4.1 瓶颈定性（小 shape，S≤512）

| 指标 | case0 (MHA fp16) | case3 (GQA bf16) | 判定 |
|---|---|---|---|
| aic scalar 占比 | 75~78% | 93% | **标量/同步控制主导**，非计算瓶颈 |
| aic mac 占比 | 2.1% | 12.5% | Cube 严重欠载 |
| HBM 带宽 | 1.63 GB/s | 0.72 GB/s | 完全非带宽瓶颈 |
| 核间负载不均 | 7~9%（达标） | **31.8%（严重）** | 小 shape 尾核效应 |
| 头开销 | 5~7% | — | kernel launch 固定成本 |

结论：小 shape 下泛化模板的 **CV 同步与标量控制结构成本** 是主瓶颈（scalar 75%+），
其次是核间切分的尾核不均。生产算子对这类用例路由到 SAB/S1s2Bn2 模板正是为了
规避该结构成本。归档 2026-08-14 已补充 S1s2Bn2 路由（D≤64 小 shape 命中，
命中用例提速 10~40%），vs 生产算子纯 kernel geomean 0.91x → 0.92~0.98x。

### 4.2 对照数据

- vs torch 组合参考实现（fp32，24 个 kernel/迭代：Cast 20% + 逐元素 ~50% +
  matmul 仅 20%）：归档单融合 kernel **全面领先，geomean 2.20x**（quick 口径，
  1.23x~3.94x）。
- vs 生产算子（npu_fusion_attention_grad，纯 kernel）：geomean **0.91x**；
  非常规 head dim（D=177、D=192/128）反超 1.2x~2.0x。
- fp32 组合实现的隐形成本：约一半耗时在 Cast + 逐元素中间结果搬运，融合 kernel
  天然免疫。

## 5. 优化点清单（按已验证/潜在分类）

### 5.1 已落地验证（本归档，含实测收益）

1. **host 输出 at::zeros → at::empty**：Pre+Main+Post 已全量覆写输出，host zeros 纯冗余。
   收益：消除 3 个 ZerosLike kernel（case0 device 时间 ↓11.3%），geomean 0.83x→0.91x。
   验证方法：覆盖性核查（tiling 中 qPostBlockTotal==qSize==numel）+ 全量精度 PASS。
2. **D2H 返回路径优化**：包装层返回前按输入设备分流——输入已在 NPU 直接返回
   NPU 张量；CPU 输入用 pinned buffer + non_blocking 批量拷回（torch_npu 实测
   pinned 比 pageable .cpu() 快 ~3.3x）。case0 端到端 484us→189us（↓61%）。
   通用原则：model_new 里无条件的 `.cpu()` 在 profiling harness（NPU 输入）下是
   纯浪费，且与"输出设备跟随输入"的参考语义不一致。

### 5.2 潜在优化点（转译新实现时考虑）

1. **按 shape 路由模板**：S1/S2 < 1024 且 B·N2 大（或 G==1）时，泛化模板的 CV 交互
   开销显著（scalar 75%+），应路由 S1s2Bn2 类模板；S1 或 S2 ≥ 1024 路由 SameAB
   （Cube 主核，左矩阵复用 + L0C 累加）。DSL 侧可用 autotune 或规则路由复现。
2. **核间负载均衡**：小 shape 多核切 S2 时尾核不均可达 31.8%（>30% 阈值），
   考虑 block 数向核数取齐 / block swat / 负载偏斜感知切分。
3. **省 Pre/Post 的条件路径**（950 regbase 模板思路）：当 dk/dv 无需多核累加
   （如 B·N2·S2 > cubeCoreNum && G==1），可省去前置 GM 清零与后置 cast/Muls，
   直接输出。arch22 泛化路径无此捷径，转译时可按路由条件生成简化变体。
4. **CV 配比调优**：A2 上 C:V=1:2、数据配比 1:16（nRatio=8）是生产默认值；
   DSL 侧 S1_c/S1_v 配比应作为 tiling 参数可调。
5. **double buffer / CV preload**：V 侧 ping-pong double buffer 与 Cube 双发
   （C 侧提前发射两块）是流水下限保障（设计文档 §4）；目标使 MTE2 或 Vector
   达 bound。

## 6. 转译检查清单（DSL FA 反向 → AscendC）

- [ ] P 重计算 + dS 双消费（dS/dS^T）通过 matmul transpose 标志消化，不物化转置
- [ ] fp32 workspace 累加 + Post cast 写回；确认输出全量覆写后 host 用 at::empty
- [ ] 模板路由规则落入 tiling（S/S2 阈值 1024、G==1、B·N2·2>CoreNum、G·S1·S2≤32KB）
- [ ] CV 配比（A2: 1:16，nRatio=8）与 double buffer 进入 tiling 参数
- [ ] GQA：dk/dv 按 N2 输出 + 组内归约语义对齐 golden
- [ ] causal 以上三角 mask 表达（True=屏蔽），与生产 sparse_mode=0 语义对齐
- [ ] 多入口拆分独立 target（RegisterAscendBinary ~8.9MB/入口）
- [ ] tiling 走 GM_ADDR + 设备侧逐字拷贝；编译选项 --cce-auto-sync=off
- [ ] 验证：fp16 容差 2e-2 / bf16 5e-2 / fp32 1e-4（vs fp32 golden）；
      性能 quick 口径 vs torch 组合 + verify 口径 vs 生产算子双重判定
- [ ] **dq 的相对误差指标陷阱（实测）**：dq = dS @ K 且 dS 每行和≈0（softmax 梯度性质），
      dq 是消减残差，golden 值普遍 ~1e-3 量级（case0：99.8% 元素 <1e-2）。
      mean(|diff|/(|golden|+eps)) 这类逐元素相对指标会被近零 golden 放大 30~300 倍：
      fp16/bf16 下 dq MERE 必然超 2^-10/2^-7 阈值（P 或 dS 任一中间量按输入 dtype
      舍入即超线，torch 模拟复现），**生产算子逐位一致地同样失败**（inner_precise 无效）。
      判定 dq 应改用 mean(|diff|)/mean(|golden|)、绝对误差 + matched_ratio，
      或以生产算子输出为 golden；不要为通过该指标把 dq 链路改 fp32（性能代价大且偏离生产设计）。
