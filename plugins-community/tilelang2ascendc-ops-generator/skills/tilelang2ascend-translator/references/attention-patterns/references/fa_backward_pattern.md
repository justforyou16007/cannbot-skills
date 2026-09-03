# FA 反向模式（FAG / FlashAttentionScoreGrad）：DSL→AscendC 转译参考

> 本文档是 attention-patterns 体系中的 **FA 反向（backward）模式**，由
> `AttentionPatternIndex.md` 路由（命中信号：`dy`/`softmax_max`/`softmax_sum`/`attention_in`
> 输入 + `dq/dk/dv` 输出）。原文件名为 `dsl2Ascendc_FA_Bwd.md`。

本文档以生产版 FAG 算子（arch22 / Atlas A2\A3 训练系列）为参照实现，总结 FA 反向算子的
计算结构、模板体系、实测性能画像与已验证优化点，用于指导 DSL（TileLang）FA 反向设计
到 AscendC kernel 的转译与调优。

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

生产 arch22 实现为 Pre → Main → Post 三个阶段；工程落地时将三者融合进**单个
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
5. （S1s2Bn2 小 shape 路径实测）mm12 非 fp32 输出为 NZ、BNSD 走 POST2KERNEL
   融合 Post、**scale 施加挂载在 pseType==PSE_OUTER_ADD_MUL_TYPE(1) 分支上**
   （无 pse 也必须置 1，否则 P 未乘 scale 直接 exp 溢出）、参照实现公共头文件中的裸布局
   常量需在独立 TU 显式补文件作用域定义。调试手段：host 侧 dump user workspace
   分区域比对（S/dP 为 NZ、mm34Input 为 fp16）+ kernel 内 DumpTensor 定位到
   softmax 输入未乘 scale。

## 3. 生产多模板体系（arch22，按优先级路由）

模板按 shape 特征路由，序号越小越泛化。**统一走泛化模板是有实测代价的**
（见 §4）：

| 模板 | 核间/核内切分 | 路由条件 | 适用场景 |
|---|---|---|---|
| S1s2Bn2gs1s2（泛化兜底） | 核间 BN2GS1，核内 S1/S2 | 支持所有 shape | 大 S；小 S 时 CV 交互频繁 |
| S1s2Bn2 | 核间 B/N2，核内 G/S1/S2 | S1<1024 && S2<1024 && (B·N2·2>CoreNum \|\| G==1) | 小 S 大 BN2 / MHA（实测加限 D≤64：D>64 时该路径慢于泛化模板） |
| Ngs1s2Bn | 核间 B/N2.o，核内 N2.i/G/S1/S2 | G·S1·S2≤32KB && G==1 && S2<1536 | 小 G·S1·S2 |
| Bngs1s2B | 核间 B，核内 B.i/N2/G/S1/S2 | N2·G·S1·S2≤64·128 | 极小内层 |
| SameAB（_sab 后缀） | UB 切 S1S2，多核切 S2 | S1≥1024 or S2≥1024；确定性；TND 大 S | Cube 主核视角，V0/V1 复用左矩阵、L0C 累加 |

- `_sab` 后缀 = Cube 主核模板（V0 统一发起 matmul，V0/V1 各处理一半）；
  减少带宽诉求，大部分场景更优。
- 确定性计算走独立 det 模板（避免多核同地址累加）。

## 4. 实测性能画像（Ascend910B2，msprof 7 组 aic-metrics）

对泛化路径（S1s2Bn2gs1s2）的 8 个用例（MHA/GQA、fp16/bf16/fp32、causal、D≠Dv、
D=177/192 非常规 head dim）实测：

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
规避该结构成本。补充 S1s2Bn2 路由后（D≤64 小 shape 命中），命中用例提速 10~40%，
vs 生产算子纯 kernel geomean 0.91x → 0.92~0.98x。

### 4.2 对照数据

- vs torch 组合参考实现（fp32，24 个 kernel/迭代：Cast 20% + 逐元素 ~50% +
  matmul 仅 20%）：单融合 kernel **全面领先，geomean 2.20x**（quick 口径，
  1.23x~3.94x）。
- vs 生产算子（npu_fusion_attention_grad，纯 kernel）：geomean **0.91x**；
  非常规 head dim（D=177、D=192/128）反超 1.2x~2.0x。
- fp32 组合实现的隐形成本：约一半耗时在 Cast + 逐元素中间结果搬运，融合 kernel
  天然免疫。

## 5. 优化点清单（按已验证/潜在分类）

### 5.1 已落地验证（含实测收益）

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
   （C 侧提前发射两块）是流水下限保障；目标使 MTE2 或 Vector
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

---

## 7. 扩展语义（sliding window + softcap）转译与性能迭代实证（50 case 全量）

在 §1-§2 的最简形态之上扩展 **sliding window（window_left/right）+ softcap（tanh 钳制）**
语义（生产算子泛化路径不支持，DSL 侧原生实现），BSHD 输入。50 case 覆盖 fp32/fp16/bf16、
causal/window/softcap 组合、GQA g∈{2,3,4,7}、D∈{16..128}、Sq 3~400。

### 7.1 性能演进账本（稳态 event 口径）

| 阶段 | 结构 | vs torch |
|---|---|---|
| 初版（阶段式拆分 7 launch，精度 50/50） | C1→V1→C2dq→post→C2kv→post 全串行 | quick 0.96x（失真口径）；大 case 稳态 0.53x |
| +结构优化 6 项（host 精简/GQA 累加/ws fp16/条带切分/mask tile 跳过/ZeroL1 跳过） | 5 launch | 大 case 0.90x（quick 5-case） |
| +CV 融合（fag_main MIX kernel） | Pre→Main→Post 单入口（对齐 §2 三段式融合结构） | 稳态 fp32 3.8x / fp16 1.78x |
| +守卫链/向量链收尾（opt4） | 同上 | **50-case 稳态 geomean 3.85x，50/50 全胜**（小 4.9-9.9x / 中 2.1-5.2x / 大 1.78-3.88x） |

关键结论：**阶段式拆分是"正确性优先"的合法首版，但 CV 融合是逼近生产实现的必经路径**
（CV 融合后大 shape 收益 ~1.9x）；torch 总账上 mask tile 跳过 + window/softcap
原生支持带来额外收益，最终 50-case 稳态 geomean 3.85x。

### 7.2 结构优化清单（全部实测，按收益排序）

1. **Host 精简（fp32 e2e ~1.8x，CPU-bound 消除）**：症状为 host 纯发射耗时（6934us）>
   device（4986us）。措施：wrapper 8 次物化 transpose 改 BNSD 直传 + mma trans 标志；
   softmax_max/sum 的 select+contiguous+cast+copy 链改 kernel 内 stride 读；
   rowIdx/colIdx 的 at::arange 改 kernel 内 createvecindex；wsP/wsDS 的 at::zeros
   去除（ws 布局无 pad 行时 c2kv 从不读 pad，原 zeros 纯防御性；**注意：核内清零
   无 pad 行的 ws 会 GM 越界**）。host op ~25 → ~11。
2. **mask tile 跳过（causal case c1 767→185us、v1 593→215us）**：全 mask tile 不进
   grid。**注意：生产算子 sparse_mode=0（全 tile 计算）路径并无此技术**，它来自
   生产 sparse 分支；DSL 设计应原生包含（hi/lo 边界求 tile 相交，空相交跳过整个
   tile 的 C+V 计算）。
3. **ZeroL1 满 tile 跳过（c2dq 653→256us、c2kv 1319→560us）**：全零 operand tile
   跳过拷贝本身；**MTE1_MTE2/MTE2_MTE1 守卫必须无条件保留**（L1 复用 WAR 竞态，
   曾致 dq 大面积小误差）。
4. **wsS/wsDP fp32→fp16 存储（ws 流量减半）**：fixpipe F322F16/F322BF16（官方
   CAST_RINT 语义）随路转换，与 golden 舍入一致，50/50 精度不回退。
5. **核间连续条带切分**：StripRange（former/base 均衡）替代 `task += cores` 轮询，
   改善 L2 局部性（§4.1 实测核间负载不均最高 31.8% 佐证该维度收益真实）。
6. **GQA dk/dv 累加**：fp32 走 CAST_DV（L0C 组内累加直写输出、跳过 post_kv）；
   fp16/bf16 受 §7.3-2 镜像约束保留 g-slab + post 归约。
7. **kernel 合并 7→5 发射**（fag_c2 / fag_post 合并），launch 边界即全设备同步，
   每次合并都是纯收益。

### 7.3 精度契约（违反即系统性偏差或 NaN，全部实测取证）

1. **torch 标量乘口径 per-dtype**：`fp16 tensor × python float` 标量先舍入 fp16；
   `bf16 tensor × python float` 标量保持 fp32(opmath) 计算后再舍入；fp32 全精度。
   host 侧 scale/softcap 必须 per-dtype 舍入（bf16 用错口径引入 ~2^-9 系统偏差，
   全量 bf16 case MERE 超标）。
2. **fp16/bf16 golden 逐头 RINT 实体化镜像**：golden 对 matmul 输出逐头 RINT 实体化
   后再做 GQA group-sum，因此 **GQA 组内不可 L0C 直接累加**（破坏镜像）；仅 fp32
   可 CAST_DV 直写。
3. **尾 k 块分形宽度一致性**：GEMM A 侧 L0A 固定 64 宽加载时，B 侧 L1 stride/L0B
   装载/Mmad k 也必须固定 64（`krA=AlignUp(kr,16)<64` 时两侧分形条带宽度不匹配，
   m 第二个 16 行分形起读到错位 pad 零——大 shape dq 全错的主因）；pad 行须来自
   ZeroL1 清零区保证数学等价。
4. **索引装载尾部残留 NaN**：列索引按 `ceil32(vn*4)` 字节装载时尾部残留前序行索引
   （值 < Skv）→ pad 列误判 valid → `at::empty` 垃圾（Inf/NaN 位型）经 `NaN*0=NaN`
   漏入 dS pad 列。装载宽度必须覆盖全部有效+pad 区（或显式清零）；连跑内存复用时
   才暴露，单 case 调试不可复现。

### 7.4 MIX kernel / CV 融合工程坑（本算子两轮实证）

- **flagId 安全段 {2..10, 14}**：MatmulImpl 每个 matmul 对象保留 flagId [0, 2N-1]，
  SyncAll 占 11-13；冲突表现为间歇性 C/V 竞态（DumpTensor 插桩会掩盖）。竞态判据：
  **stress ≥50 次零失败 + 全量 PASS**，单次 PASS 不算数。
- **Matmul<> 别名 + 异步三件套**（SetSysWorkspaceForce + AIC clearWorkspace +
  REGIST_MATMUL_OBJ）；硬编码 MatmulImpl 造成"IterateAll 恒同步"假象。
  **裸 SyncAll()（AIV-only）**；SyncAll<false>（MIX 变体）在 client-server 下死锁。
- **跨 tile 累加** `IterateAll<false>(ws, enAtomic=true) + End()` 原子加 fp32 ws；
  GM 可见性 = 写流与发 mma 消息流同一；双 AIV 半区写须先会合再单流发射。
- **507033 三种实证形态**：① 宽签名（32 参数）×完整类触发 MPU crash（保持 ~13 参数
  窄签名 + 单 user workspace + Init 内 512B 对齐切分）；② 双 CO1 队列；③ **910B2C
  不支持 FIX pipe 读 L0C 与 M pipe 写 L0C 并发**——任何 L0C Mad/Fix 重叠（含单 CO1
  队列深度 2 的规避方案）不可行，生产前瞻深度 2 方案在本平台实测证伪。507033 后设备 retain
  ~20s，复验前须先探活（否则误判修复失败）。
- **CO1 TQue 双未决 EnQue 时 DeQue 呈 LIFO**（S/dP 交换）——fixpipe 批处理路径
  不可行，保持串行 Mad→Fix。
- **守卫链精细化方法**：SetWaitFlag 前向保护语义逐站点推导最小集（本算子分为
  Stats{MTE2_V,V_MTE2} / Load{+MTE3_V,MTE3_MTE2} / Store{+PipeBarrier 归约} 三级）；
  收益 ~20us 级（AIV scalar 主体是 ping-pong 等待停顿而非守卫开销，预期管理）。
- **UB buffer 写宽复核**：列索引 Cast 写 256B 到 128B buffer 溢出覆盖相邻 sfmg
  （原操作顺序下无害，sfmg 提升后暴露）——改 buffer 布局/生命周期时复核所有相邻
  buffer 写宽。
- **Nd2NzParams stride 字段全部 uint16_t**：BSHD 大行距（H*D=9216）下 16*ld 溢出
  → Zz 路径逐组单发规避。
- **fp32 attention_in 是无操作视图**（transpose+to(fp32)，BNSD 内存）：eval clone
  输入会掩盖，kernel 需双布局步长兼容。

### 7.5 测量口径（性能结论的前置条件）

- **quick 批量模式含输入构造**（rand/cast/H2D 与 forward 同 stream 计量），小 shape
  被放大 20-200x（~17ms 地板），比值压缩到 ~1.0x 失真，偶发 ref 侧 31x 异常产生
  虚假高加速比；其 per-case 绝对值不可横向分析，geomean 仅作"不劣于"门禁。
  （forward-only 修复曾实施验证，后按需求回退；当前仍为旧口径。）
- **稳态 event（输入外置、warmup 3+、10 次平均）才是真实水位**；DVFS run 间波动
  ±25%，所有对比必须同次背靠背。
- 三方对账法（torch / 本实现 / 生产算子 两两比值应互洽，误差 <1%）是发现口径错误
  与基准错误的有效手段。
- **对比基准的有效性边界**：参照基准在其已验证 case 包络外的 shape 上可能静默
  出错（实测：g=4 GQA + 非 64 对齐 seq 尾块组合下基准实质性错误而本实现逐位一致）
  ——任何对比结论必须先做 golden 抽查，仅限基准已验证包络内有效。
