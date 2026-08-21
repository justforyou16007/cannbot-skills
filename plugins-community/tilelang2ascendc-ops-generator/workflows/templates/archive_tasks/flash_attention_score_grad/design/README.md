# FlashAttentionScoreGrad TileLang 设计

参照 `workflows/templates/archive_tasks/conv2d_base/design` 的格式，
为归档 kernel（arch22 S1s2Bn2gs1s2 通用模板路径的生产 re-host）补写的
TileLang 设计表达。归档本身不含 TileLang 实现（生产 AscendC 源码直接
re-host），本目录是对该 kernel 结构的设计级复刻，供 DSL→AscendC 转译
对照；更完整的转译要点见
`skills/tilelang2ascend-translator/references/attention-patterns/references/fa_backward_pattern.md`。

## 结论

FAG（FA 反向）按 **Pre → Main → Post 三段式单 kernel 入口** 组织，
对应生产 `fag_s1s2_entry_common.h` 的入口体：

```text
Pre : AIV 清零 fp32 workspace（dq/dk/dv 累积区；fp32 时直接清 dvGm）
Main: S1s2Bn2gs1s2 CV 融合主计算（6 步，5 个 matmul + softmax/sfmg 向量）
Post: AIV 读 fp32 workspace，Muls(scale) + Cast，按 numel 全量写回 dq/dk/dv
```

主计算（设计文档 `docs/FAG算子设计介绍.md` §1，keep_prob=1.0/无 pse 化简）：

```text
S  = Q @ K^T * scale           (mm1, Cube)
P  = exp(S - softmax_max) / softmax_sum，mask 位置置零（Vector，复用前向统计量）
dP = dy @ V^T                  (Cube，与 mm1 同窗口双发 = CV preload)
sfmg = rowsum(dy ∘ attention_in)（FlashSoftmaxGrad，Vector，D 方向 sfmgd 切分）
dS = P ∘ (dP - sfmg) * scale   (Vector)
dq += dS @ K ; dk += dS^T @ Q ; dv += P^T @ dy  (Cube，fp32 workspace 累加)
```

对应文件：

- `fag_common.py`：共享配置（FagConfig）、合法性 gate、CV 切分参数、
  核间 block 切分、生产多模板路由（参考）；
- `block_level/fag_s1s2.py`：block 域契约、Pre/Main/Post 阶段边界、
  CV 双子图（SubGrapA/SubGrapB）ping-pong 与跨 Scope 同步骨架；
- `tile_level/fag_s1s2.py`：各级 buffer（L1/L0A/L0B/L0C/UB）、
  搬运与计算的完整设计。

## AscendC 函数与 TileLang 设计映射

| 生产 AscendC 类/函数 | 设计 factory / helper | 等价语义 |
| --- | --- | --- |
| `FlashAttentionScoreGradPre::Process` | `_pre_zero_workspace` | 按 PreBlockFactor/Total/Tail 分核清零 fp32 workspace（INIT_OUTPUT=true） |
| `FlashAttentionScoreGradS1s2Bn2gs1s2::Process` | `fag_s1s2_tile` 的 main 循环 | 核间 BN2GS1 合并轴 × 核内 (s1o, s2o) 双层遍历 |
| `SubGrapA` / `SubGrapB` | `_sub_graph_a` / `_sub_graph_b`（block） | 相邻 s2o tile 的 CV ping-pong（eventIdList 双通道） |
| mm1（`Matmul<...NORM_DISABLE_INIT>`） | `_c1_compute_s_dp` 中首个 `T.mma(init=True)` | S = Q @ K^T，L0C fp32 |
| mm3/mm4 + CV 双发 | `_c1_compute_s_dp` 中第二发 `T.mma` | dP = dy @ V^T，与 S 同窗口发射覆盖 V1 计算 |
| `CalcSoftMax`（复用前向统计量） | `_v1_softmax_sfmg_ds` 的 sub/exp/div | P = exp(S - max)/sum（非在线 softmax，直接重建） |
| `CalcAttenMaskBool` | `_v1_softmax_sfmg_ds` 的 mask 分支 | atten_mask True=屏蔽位置置零；causal=上三角 mask |
| `CalcSoftMaxGrad` / `...PingPong`（SFMG） | `_v1_softmax_sfmg_ds` 的 sfmg 段 | rowsum(dy ∘ attention_in)，D 方向 sfmgd(outer/inner/tail) 切分 |
| dS 合成 | `_v1_softmax_sfmg_ds` 末尾 sub/mul/mul | dS = P ∘ (dP - sfmg) * scale |
| mm3/mm4 累加（init=False） | `_c2_accumulate_dqkv`（`init=(s2o==0)`） | dq+=dS@K、dk+=dS^T@Q、dv+=P^T@dy，fp32 workspace |
| `FlashAttentionScoreGradPost::Process`（ND） | `_post_writeback` | Muls(scale) + Cast(CAST_ROUND)，按 qSize/kvSize/vSize 全量写回 |
| `FuzzyForBestSplit` / 块切分 | `cv_tile_params` / `block_split` | 核内 (s1_inner, s2_inner) 与 CV 配比、核间 blockStarts/Ends |
| host `fag_tiling_standalone.cpp` | `fag_s1s2_block_contract` | tiling 参数（block/cv/sfmgd/postBase 等）的设计级计算 |

## 逐函数语义核对

- 布局：BNSD（生产 mock_layout 重映射 BNSD(2)→0）；GQA 语义为
  dk/dv 按 n2 头输出、对 g=n1/n2 组内归约（golden 侧 repeat_interleave +
  view-sum 对齐）。
- softmax：反向不跑在线 softmax，直接用前向统计量
  `softmax_max/sum[..., :1]` 重建 P；mask 位置先填 -40000 等效为置零。
- 转置消化：dS^T、P^T、K^T、V^T 全部由 mma 的 transpose 标志表达，
  不物化转置张量。
- 累加与精度：Cube 输入 fp16/bf16、L0C/workspace fp32 累加；
  fp32 dtype 时 dv 不经 workspace（Pre 清 dvGm、Main 直写、Post 跳过 dv，
  即生产 CAST_DV 条件）。Post 的 dq/dk 乘 scaleValue。
- 输出覆盖性：Post 的 qPostBlockTotal=qSize=b·n1·s1·d 等恰好铺满输出
  numel，是 host 侧 `at::empty`（替代 `at::zeros`）的正确性依据
  （实测消除 3 个 ZerosLike kernel，device 时间 ↓11.3%）。
- 尾块：S1/S2 尾部由 s1Tail/s2Tail、Post 的 TailNum 表达；
  NZ 输出（mm12IsNZOut/mm345IsNZOut）在归档用例不命中，设计保持 ND。

## CV 配比与流水

- A2 训练系列 AIC:AIV = 1:2；CV 数据量配比 1:16（Cube 基本块 128×s2_inner
  fp16 vs Vector s1_cv_inner×s2_cv_inner fp32，S1 nRatio=1、S2 ratio=8 组成）。
- CV preload（Cube 双发，设计文档 §4.2）：C1 同一窗口连发 mm1(S) 与 mm(dP)，
  使 Cube 搬运覆盖 V1 的 Vector 计算；SubGrapA/B 双通道 ping-pong 隐藏
  CV 交互。
- V 侧 double buffer：S/dP/P/dS 的 UB 缓冲按 slot=s2o%2 双份；
  sfmg 的 D 方向切分（sfmgd）在大 value_d 时多轮累加。

## Buffering 与驻留

| 数据 | 通路 | 驻留 |
| --- | --- | --- |
| q/k/v/dy tile | GM→L1(zn)→L0A/L0B | 每 (s1o,s2o) tile 一轮 |
| S/dP | L0C(fp32)→GM workspace（CV 交互） | ring slot = s2o%2 |
| P/dS | UB(fp32)→GM workspace | ring slot 双份 |
| dq/dk/dv 累积 | L0C→fp32 workspace | 全程（init=(s2o==0) 起累加） |
| 输出写回 | workspace→UB→GM | Post 按 post_base_num 分块 |

## 生产多模板路由（设计参考，归档统一走泛化路径）

`fag_common.production_template_route` 复刻了生产路由条件
（S1s2Bn2 / Ngs1s2Bn / Bngs1s2B / SameAB）。归档统一走泛化模板的实测代价
（docs/perf/，Ascend910B2）：小 shape 下 aic scalar 占比 75~93%、
mac 2~12%、核间负载不均最高 31.8%、非带宽瓶颈——vs 生产算子纯 kernel
geomean 0.91x 的差距主因即模板结构成本；vs torch 组合实现 geomean 2.20x。
非常规 head dim（D=177、D=192/128）归档侧反超 1.2~2.0x。

## 小 shape 模板路由（2026-08-14 补充）

归档在泛化路径之外补充了 **S1s2Bn2 小 shape 路径**（re-host 自生产同名模板）：
host 按生产 `IsCapable` 化简条件路由——S1/S2<1024、D==Dv、**D≤64**（D>64 实测
本移植路径慢于泛化，回退泛化）、非 GQA 小 BN2 等；命中走 bn2 入口
（`fag_bn2_entry_*`，POST 融合主 kernel、无 Pre 阶段），否则走泛化。
实测归档 8 用例中 case0/case4/case6 命中，单用例提速 10%~40%
（纯 kernel vs 生产算子 geomean 0.91x → 0.92~0.98x）。

re-host 关键坑（调试实测）：
1. bn2 非 fp32 的 mm12 输出为 **NZ 格式**（GetTilingKey 的 mmOutFormat 规则），
   内核模板实参必须匹配，否则 vector 读到未初始化数据（全 NaN）；
2. BNSD 走 **POST2KERNEL** 变体（POST=true，post 融合主 kernel），
   TND 才用独立 Post 类；
3. **scale 施加挂载在 pseType 分支上**：kernel 仅在
   `pseType == PSE_OUTER_ADD_MUL_TYPE(1)` 时执行 DoMulsScale——无 pse 也必须
   置 1 占位，否则 P = exp(S_raw - max)/sum 未乘 scale 直接溢出（dump 实测复现）；
4. `cube_op_bn2.h` 以裸标识符引用 BNGSD/SBNGD/BSNGD/TND，独立入口 TU 需在
   include 前提供文件作用域定义（生产 opp TU 依赖包含顺序的宽松查找）。

## 与生产实现的差异

1. 生产经 tilingkey 模板分发 + opp 框架 GET_TILING_DATA；归档/设计为
   单入口直发，`GM_ADDR tiling` + 设备侧逐字拷贝（stub 生成器只认
   `struct Name {`），多 dtype/mask 拆 6 个独立 target
   （RegisterAscendBinary 单文件 ~8.9MB 上限）。
2. 生产的 TND/rope/pse/dropout/sparse 分支不在归档范围内
   （固定 BNSD / keep_prob=1.0 / 无 pse / sparse_mode=0 / 无 rope）。
3. 设计把 Pre/Post 表达为同一 kernel 内的顺序阶段（与归档入口一致）；
   生产 opp 体系下三者是同一 kernel 内的类实例顺序调用，语义等价。
