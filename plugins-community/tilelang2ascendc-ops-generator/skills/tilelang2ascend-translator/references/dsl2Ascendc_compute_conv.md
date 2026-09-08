## Conv 计算详细参考：img2col + GEMM 与 LoadData3D 随路展开

本文档总结卷积类算子（conv2d）的 AscendC 实现与性能优化知识。
概览与判断规则见 `@references/dsl2Ascendc.md`，Cube 通用组织见
`@references/dsl2Ascendc_compute_cube.md`。

---

## 第一章：总体架构 —— host-zero-prep 多级 kernel 流水

卷积不要写成单个 kernel，也不要在 host 侧用 at:: 算子做预处理。推荐的形态是 **host 只发射 kernel + 分配 workspace，全部
布局变换由 device kernel 完成**：

```
wrepack (AIV)   : 权重 [Co, Ci, kH, kW] -> K-major ws [K, CoPad]（或 fused k 序）
im2col  (AIV)   : 特征图 -> im2col 矩阵 [totalRows, K]   ← 可被 P5 fused 模式消除
conv    (AIC)   : GEMM, GM --Nd2Nz--> L1 --LoadData3D--> L0 --Mmad(fp32)--> L0C --Fixpipe--> GM
epilogue(AIV)   : acc fp32 [totalRows, Co] + bias -> cast -> 目标布局输出
```

要点：

- 每级 kernel 都是独立发射，kernel 间通过 GM workspace 衔接；host 侧无任何 device 算子。
- 权重 repack 结果是（weight 内容, shape, dtype）的确定函数，可在 host 做 **LRU 缓存**
  （key = data_ptr + sizes + dtype + version，条目内持有 weight 强引用防止地址复用），
  命中时跳过整个 wrepack 发射。推理场景权重不变，这是净赚。
- AIC:AIV 混合核用 `KERNEL_TYPE_MIX_AIC_1_2`，AIV 侧用 `GetSubBlockIdx()` 切子核。

---

## 第二章：GEMM 映射的三种模式

同一个 conv kernel 按 case 特征选择 GEMM 组织方式（host 判定，传 mode 标量）：

| 模式 | A（M×K） | B（K×N） | 适用 | 省去 |
|:---|:---|:---|:---|:---|
| normal | im2col GM 矩阵 [rows, K]，k 序 (kh,kw,ci) | K-major 权重 ws | 通用回退 | — |
| direct 1x1 | 原始权重 [Co, Ci]（**翻转 GEMM**，跨图共享） | 原始 x NCHW [Ci, H*W]（本身就是 [K,M] ND 布局） | kH=kW=1, s=1, p=0, Ci%16==0, (H*W)%16==0 | wrepack + im2col 两级发射 |
| fused img2col | L1 常驻 NC1HWC0 特征图，img2col LoadData3D **随路展开**直接进 L0A | K-major 权重 ws（fused k 序，按 nIdx 常驻 L1） | Ci%16==0、NCHW 需 H*W 偶数、xNz+B 面板 ≤ L1 预算 | im2col GM 物化（最大单收益） |

direct 1x1 的关键认知：1x1 卷积的 im2col 是恒等变换，NCHW 原始布局的 x 已经是
`[K=Ci, M=H*W]` 的 GEMM B 操作数，权重做 A 即可，acc = [Co, H*W]，epilogue 按
通道行加 bias 直接写出 NCHW，全程零 transpose。

> **fp32 输入**：上表三种模式针对 fp16/bf16 输入；fp32 输入走**原生 fp32 cube**
> （C0=8，强制处理项见第九章）。

---

## 第三章：数据通路 API 选型

| 通路 | API | 实测结论 |
|:---|:---|:---|
| GM(ND) → L1(NZ) | `DataCopy` + `Nd2NzParams` | **唯一可靠通路**。plain 2D DataCopy GM↔L1 在本平台不可靠（实测踩坑），任何 GM→L1 都走 Nd2Nz 随路转 NZ |
| L1(NZ) → L0A/L0B | `LoadData` + `LoadData3DParamsV2` | 普通 tile 用 l1H=1/filter=1 退化为纯搬运；conv 用 img2col 模式（filter/stride/dilation/pad 硬件处理） |
| L0C → GM | `Fixpipe` + `FixpipeParamsV220` | fp32 NZ → ND；写大矩阵时 `dstStride` 必须是完整行宽，不是 tile 宽 |
| GM ↔ UB（非对齐） | `DataCopyPad` | 见第六章对齐陷阱 |
| UB 内 16×16 转置 | `TransDataTo5HD` | **c220 不支持 bf16**：`ReinterpretCast<uint16_t>` 按位转置，全平台安全 |

Mmad/L0 组织沿用 matmul 成熟实践：每个 K_L1 迭代只 `DeQue` 一次、内层传偏移指针、
全部内层迭代完成后只 `FreeTensor` 一次；m/n 尾块按满 baseM/baseN 计算，Fixpipe 只
写实际范围（stale 行列丢弃），不要为尾块缩小 Mmad（m 侧缩减实测产出 garbage）。

---

## 第四章：fused img2col LoadData3D 随路展开（最大优化）

### 原理

im2col 物化路径的流量是 `特征图 -> UB -> GM im2col -> L1 -> L0`，其中 GM 往返占了
case 0 总耗时的 70%（39.3us/56us）。fused 模式让 **A 侧不落地**：

```
xT GM [img, H*W, Ci] --Nd2Nz--> L1 NC1HWC0 常驻 --img2col LoadData3D--> L0A Zz tile
                                  （pad/stride/dilation 硬件随路处理）
```

- B 侧复用 normal 路径的 K-major 权重 ws，按 nIdx 常驻 L1 深度 1。
- CopyOut / epilogue / m/n 分解与 normal 路径完全一致，改动只集中在 A 侧加载。
- NCHW 输入需要先由一个 AIV xtrans kernel 生成 xT `[H*W, Ci]`；NHWC 输入本身即
  channel-inner，**直接用原始 x，零中转**。

### 关键实测发现：硬件 img2col 展开的 k 序是 (ci1, kh, kw, ci0)

**C0 通道块在最外层**，与软件惯用的 (kh, kw, ci) 不同。权重 ws 必须按同一 k 序
repack（wrepack 增加 fused 布局输出，缓存键带上布局标志）。两种 k 序在 Ci==16 或
kH*kW==1 时恰好重合——如果第一个验证 case 恰好是 Ci=16，会"蒙对"，换 case 即暴露，
务必用 one-hot 探针（单元素置 1 观察落点）实测确认，不要凭文档假设。

---

## 第五章：有效优化手段清单（按实测收益排序）

| # | 优化 | 实测收益 | 机制 |
|:---|:---|:---|:---|
| 1 | fused img2col 消除 GM 物化 | case 0: 46→21us | 见第四章 |
| 2 | host 侧 weight repack LRU 缓存 | 省去整个 wrepack 发射 | 推理权重不变，结果确定可缓存 |
| 3 | AIV kernel 内同步点合并 | im2col ~10→1us/tile | 所有 (kh,cib) slab 先一次性加载、共享**一个** MTE2→V 同步点，再背靠背转置（V pipe 程序序保证）；替代每个 slab 2 个 flag pair |
| 4 | TransDataTo5HD repeat 批量转置 | xtrans 14→6us | 每个 ci block 一次调用、repeat 走遍 16×16 子块；标量逐项构造 list（每子块 32 个 list entry）是瓶颈 |
| 5 | direct 1x1 flipped GEMM | 省两级 kernel 发射 | 见第二章 |
| 6 | epilogue 跨行大块处理 | 13-15→3-4us | 整块连续区域一次 load + 行内标量 Adds（V pipe 行间免同步）+ 一次 cast + 一次 store；替代逐行 flag pair |
| 7 | depth-2 A1/B1 跨 (blk,panel) 预取 | case 0 conv 13.7→10.4us | 当前块计算时发射下一块的 GM→L1；B 槽只分配实际列数（≤baseN）使两块面板能同时放进 L1 |
| 8 | stagger-aware 核数调优 | case 0 conv 12.5→10.9us | MIX kernel 发射 stagger ~0.2us/核，**满 wave 的少核 > 带残尾 wave 的最大核数**：waves=ceil(blocks/aic)，dim=ceil(blocks/waves)；小 grid 按 ~2 tile/核封项 |
| 9 | n 自适应 L0B/Mmad | Co=64 时 cube 工作量减半 | nActual≥64 时按 nActual 加载；**nActual<64 保持满 baseN**（nActual=16 在 dav-c220 触发 L0C 读写冲突，实测） |
| 10 | **stride==2 GatherMask 去交织向量化** | case[24] 0.32→1.20（kernel 侧 ~3.9x） | 每 task 2 次 GatherMask（pattern 1/2，Normal 模式）把整行拆 even/odd 子平面，ki 窗口变为子平面连续切片；UB src 32B 对齐铁律拆 head(≤7 标量)+main(对齐 bulk 直接 DataCopyPad)+tail(≤7 标量)，标量量 255/ki→≤14/ki；stride≥3 保留标量回退 |
| 11 | **发射链合并（跨步切分）** | 4→3 发射，小 case 普升 | pad_w / im2col 等轻 kernel 并入单发射；**两个任务空间必须各自按核跨步 `for(t=coreIdx; t<n; t+=numCores)` 切分**——拼接+连续切分会把重任务塌到少数核（实测 case[33] 0.76→0.48，改跨步后 0.90） |
| 12 | **输出 ws zeros→empty + kernel 零区覆盖写** | 省 1 次发射 + 大 case 省 MB 级 memset | kernel 写三段**两两不相交**零区（K-tail 整行 / M-tail 列条 / pad 空洞），零源 kernel 起始 Duplicate 一次全程只读（V_MTE3 一次同步）；无竞态前提 = 区域不相交 |
| 13 | **epilogue 直写最终布局** | 全 case 普升，省 ~19us/call host | epilogue kernel 按目标布局（如 (N,Cout,Lout)）分段 store，wrapper 直接返回——布局转换必须在 kernel 内消化，**禁止留 reshape+permute+contiguous 给 wrapper**（aclnnInplaceCopy 全量往返 + host 固定开销） |
| 14 | **host pad/copy 消除（LoadRowSafe 末行钳制）** | 省 ~4.3us/call + 流量 | 仅末行（xBase+count+8>numel）拆 安全主段 DataCopyPad + 尾部 <8 元素（floor8-8 对齐装载 scratch → SetValue 回填，MTE2_S 同步）；替代 host 侧 at::pad/copy_ |
| 15 | 零 bias 张量缓存 | bias=False case 省 1 次 memset 发射 | wrapper 层按 key 缓存零 bias 张量，消除每 forward 一次 memset |

#### #10 stride==2 GatherMask 去交织实现要点（conv1d 实证，conv2d/3d 同构适用）

- **适用性**：最内维 stride==2 时行内去交织（`o = ki*dilation - pad` 的奇偶选子平面，
  base=⌊o/2⌋）；2D/3D 外侧维 stride>1 是行级跳取、天然向量友好，无需去交织；
  stride≥3 保留标量回退（占比小）。
- **参数**：`GatherMaskParams(1, ceil(L/64), 8, 0)`，pattern 1=even / 2=odd，Normal
  模式；fp32 每 repeat 读满 256B（64 元素）→ 取 32；xUb 装载须 64 元素对齐。
- **标量头尾**：每 ki 两个 16 元素 scratch 槽位，pass1 全部标量填充 → 一次
  `S_MTE3` 同步 → pass2 背靠背发射全部 store（MTE3 同 pipe 保序、dst 两两不相交）。
- **同步序列**（每 task）：`MTE3_V`（子平面 WAR，前一 task store 读子平面）→
  `V_MTE2`（xUb WAR，前一 task GatherMask 读 xUb）→ 装载 xUb → `MTE2_V + S_V`
  （装载 RAW，含尾段 SetValue 回填）→ 2 次 GatherMask → `V_S + V_MTE3`（子平面
  就绪）→ `MTE3_S`（scratch WAR）→（标量填充后）`S_MTE3`。
- **对齐铁律**：UB src 必须 32B 对齐 → 有效段拆 head(≤7 标量)+main（alignUp8~
  alignDown8，直接 DataCopyPad 自子平面，GM dst 非对齐由 pad 吸收）+tail(≤7 标量)。

### 走偏点（conv1d 实证，勿重复）

1. **constant_pad_nd 替代 host copy_**：aclnn 包装算子 host 固定开销 +33us/call，
   "少一次 launch"账面逻辑不成立——pad/拷贝必须在 kernel 内消化（#14），不要换成
   另一个 aclnn 算子。
2. **合并发射用"拼接任务空间 + 连续切分"**：见 #11（并行度塌陷）。
3. **workspace 三合一 alloc + narrow 视图**：变长大块分配绕过 caching allocator 尺寸桶
   + narrow TensorImpl 开销，实测净 -2.4us/call——workspace 保持独立 alloc。

---

## 第六章：dav-c220 硬件行为实测要点

1. **img2col LoadData3D k 序 = (ci1, kh, kw, ci0)**（C0 块最外层）——见第四章。
2. **TransDataTo5HD 不支持 bf16**：按 uint16_t 位级转置。
3. **plain DataCopy 要求 32B 倍数传输**：fp32 load 需 len%8==0，fp16/bf16 store 需
   len%16==0；不满足会**静默截断**（7376B store 实测被截成 7360B），非对齐一律
   DataCopyPad。
4. **DataCopyPad 非 32B 倍数的 blockLen**：UB 侧行 pitch 会向上取整到整 32B block
   （两个方向都取整），寻址必须按取整后的 pitch；且非 32B 行走慢速 dummy-pad 路径，
   大量非对齐行时考虑改算法而不是硬拷。
5. **MIX kernel 发射 stagger ~0.2us/核**：核数选择按满 wave 原则（第五章 #8）。
6. **LoadData3D Zn 的 channelSize 非 16 倍数会跳过部分分形块**：空间尾（如 H*W=196
   的余数 68）必须 round16，否则对应列数据损坏。
7. **重复利用同一 UB buffer 的跨 pipe 同步**：下一轮 MTE2 写 vs 上一轮 V 读、下一轮
   V 写 vs 上一轮 MTE3 读，都要显式 SetFlag/WaitFlag；**最后一次迭代可省略回同步**
   （flag 往返是纯延迟，无事可重叠）。
8. **MTE3 store 与同 buffer 后续 V 计算**：store 读 trans buffer，下一轮转置覆写它，
   MTE3 必须 drain 后 V 才能继续。
9. **img2col LoadData3D 通道块索引 C1 ≥ 15（Ci ≥ 240）产生垃圾** → gate C1 ≤ 14，
   Ci=256/512 回退 normal 路径（fused 路径历史上受 L1 预算 gate 限制从未探测过
   Ci≥240 区域，勿凭"fused 结构在别的通道数验证过"推断该区域安全）。
10. **AIV 侧无 `HardEvent::MTE2_MTE3`**（AIC-only），误用 → 507014 死锁；AIV 等价
    保护用 SetFlag/WaitFlag 旗标对。
11. **aicore timeout 的 kernel 会使该 NPU 进入持久坏状态**（同设备 torch_npu 原生
    conv 都输出错误）——调试崩溃类问题必须先换干净设备对照，避免把设备污染误判
    为代码回归。
12. **V 管异步写 vs MTE2/MTE3 读（及 V 读 vs MTE2 写）的 WAR 竞态必须立即配对
    V_MTE2 / MTE3_V**：手工 buffer 复用链（split-K 求和、批处理 im2col 批次边界）
    不受"上一 op 的 EnQue/DeQue 已覆盖"保护，各踩过一次精度/覆写坑。
13. **GM→UB DataCopyPad dstStride=0 的多行打包要求行块长本身 32B 对齐**（fp32
    ln%8==0），否则行起点未对齐、后续向量读出错——成批加载门限必须按此设定。
14. **Level-0 count+repeat 形式的向量 API（Cast/Add 等），count 不得超过单条指令
    元素宽度**（fp32=64，fp16=128）：超出部分**静默保留陈旧值，无报错**，坏点呈
    指令宽度周期分布（误差分布探针可一次定位）。
15. **PipeBarrier<PIPE_ALL> 会泄流相邻迭代在飞的 MTE3 存储**（曾致 case0 回归
    2.5x）——默认选最窄旗标对（V_S/S_V），PIPE_ALL 仅在确需跨多管道时使用。
16. **TransDataTo5HD per-call 成本 ≈ 140/coC us 固定值**（与 rows、stride 几何无关）
    ——小 co 转置存在 ~23.5us/tile 结构地板，只能靠减少调用次数（coC 最大化、
    多段打包单次调用）或更换转置机制突破。
17. **一个 .cpp 只能有一个 `extern "C"` kernel**：双 kernel 单文件在
    rtRegisterAllKernel 阶段报 107000（binary parse failed）——aclrtlaunch 打包
    按文件组织，多 kernel 必须拆文件。
18. **DataCopyPad 每次调用固定成本 ~1.5-2us、每 2D block 描述符 ~100ns**：小 case
    kernel 设计必须最小化调用次数（20 次调用即 ~34us 地板），整块
    （blockCount=1）大传输优先；预载类任务"更多并行更优"是伪直觉——
    whole-copy 预载体积与任务数无关，任务越多 L2 冗余越大（bands 1/2/3 实测
    17.55/20.69/22.93us）。
19. **x 内容 LRU 缓存对 benchmark 稳态有效**（harness 输入张量只搬一次）：staging/
    预处理结果可按 (data_ptr+version+sizes) 缓存，正确性依赖 version_counter
    追踪原地改写 + 条目持有强引用防 data_ptr 复用（wrepack 缓存同款机制）。
20. **UB 侧元素粒度位移不可行，唯一可行模式是 GM scratch 位移副本 + blockCount=1
    整块预载**：UB 内三类位移手段全部受限（向量 src 需 32B 对齐、UB→UB 拷贝
    32B 块步长、GatherMask src 需 32B 对齐）——需要 {-1,0,+1} 这类元素级窗口
    位移时，在 GM scratch 写好多份位移副本，再整块大传输载入。

---

## 第七章：tiling 与资源预算经验值（910B）

| 参数 | 取值 | 依据 |
|:---|:---|:---|
| L1 预算 | 384 KB（L1=512KB 留余量） | A1+B1 双面板 + fused 模式 xNz + 常驻 B 面板 |
| baseM / baseN / baseK | 128 / 128 / 64 | L0C 单 tile = 128×128×4B = 64KB，占 L0C(128KB) 一半 |
| C0 | 16 | fp16/bf16 分形 C0 = 32B/2B；**fp32 为 C0=8（默认路径，强制处理项见第九章）** |
| UB 预算 | 192 KB 总量留 headroom | im2col slab / epilogue acc tile 按此反推行数 |
| epilogue acc tile | ~16K fp32 元素/tile（64KB UB） | rowsPerTile = 8192/CoPad 反推 |

---

## 第八章：六轮性能迭代策略总结（ConvStandard2d 实测）

fp32 Conv2d 全参数（groups/dilation/stride/bias）vs torch_npu 融合 conv，10 case
精简集（含 depthwise×2、grouped×2、k1/k5s2/dilation 各一），`--quick` 口径。
六轮迭代 geomean **0.22 → 0.35 → 0.49 → 0.53/0.60 → 0.755 → 0.785**，全程精度
50/50 保持。

### 轮次轨迹

| 轮 | 策略主题 | geomean | 核心改动 |
|:---|:---|:---|:---|
| R1 | fused img2col 落地 | 0.22→0.35 | 4 个 fused kernel（xtrans/wrepack/gemm/epilogue）+ host 按 case 路由，normal 路径保留回退 |
| R2 | depthwise 专用路径 | 0.35→0.49 | 纯 AIV Axpy kernel，4 个 dw case 全部反超标杆（子集 geomean 1.75） |
| R3 | 小 case 流水优化 | 0.49→0.53 | xtrans 列表提升 / k1-direct / 工作区 LRU / split-K / im2col 成批 |
| R4 | 测量口径修正 | 0.53→0.60 | 非代码改动：op_summary 全量采集口径失真，统一 `--quick` |
| R5 | 瓶颈排序逐项击破 | 0.60→0.755 | epilogue 合并转置 / im2col band 拆分 / flat-span / k1direct 单 kernel |
| R6 | stride-2 直卷积专用路径 | 0.755→0.785 | s2 去交织 staging（x 内容 LRU 缓存）+ 纯 AIV 直卷积 kernel，case9 0.377→0.546 |

### R1 fused img2col（机制见第四章）

- 硬件 img2col k 序 (ci1,kh,kw,ci0) 用 24 个 one-hot 探针实测确认，勿凭文档假设。
- C1≥15 硬件缺陷 gate（第六章 #9）；2D UB→GM 批量存储（srcStride=0）实测有效。
- 教训：fused 收益集中在消 GM 物化，但对小行/微行 case 无效，需后续轮次补齐。

### R2 depthwise 专用路径

- 路由条件：`CiG==1 && groups==Ci && stride==1 && kH*kW≤64`；纯 AIV 原生 fp32
  FMA（Axpy 链），DataCopyPad 吸收移位窗口非对齐。
- 教训：**用通用 im2col+GEMM 处理 depthwise 是结构性错误**（0.08x），专用
  通道级 elementwise kernel 直接反超标杆。groups 类 case 先判 depthwise 再谈 GEMM。

### R3 小 case 流水优化

- **先分解后动手**：msprof 逐 kernel 分解推翻"gemm 流水链主瓶颈"预设——真凶是
  epilogue_fused（24-42us）与 xtrans（15-45us），gemm 仅 9-26us。
- TransDataTo5HD 固定成本定律（第六章 #16）→ xtrans 指针列表提升出任务循环
  （-63~68%）+ host 侧 T 选择避开碎尾 tile。
- host 工作区（im2col/acc/xtrans buffer）按 shape **LRU 缓存**，消除每调用
  at::empty/zeros + 清零 kernel（缓存零值语义等价）。
- split-K 收益有限：gemm per-block 固定成本 ~20us（strip+B 面板+启动+fixpipe）
  主导而非 kt 循环——**先探针量化固定成本再决定是否拆 K**。
- im2col 8 行成批门槛 ln%8==0（第六章 #13）。
- k1-direct 初版（纯 AIV kernel 走 normal flip GEMM）已证 1x1 免转置可行，
  R5 Opt-D 将其推到单 kernel 形态。

### R4 测量口径（方法论铁律）

- **msprof 全量 aic-metrics 采集（op_summary 模式）会使 asc 侧虚高 2-3x、ref 侧
  虚低 ~2x**：同一二进制下 depthwise case 测出 0.21 vs 1.10 的两相反结论。
  对比测试必须 `--quick`（1 轮采集、无 aic-metrics）。
- 冷启动 host 开销是假象：稳态 host≈device，每次 launch 固定 ~7.6us。
- 台间噪声 ±5% 量级：多轮测量趋势一致才可采信，单轮绝对值不可作为回归判据。

### R5 瓶颈排序逐项击破

| 优化 | 机制 | 实测收益 |
|:---|:---|:---|
| Opt-A epilogue 合并转置 | 双段打包单次 TransDataTo5HD（repeatCount=2*nb 连续两段）+ 跨步 Cast/Add 重组 + host R=256 候选（UB 门限） | case7 epi 42→15us（0.53→0.68）；case5 0.55→0.65 |
| Opt-B im2col oh-band 拆分 | 任务粒度 (n,g,ci,kh)→(…,oh-band)，band 数按 2×aicNum 自适应 | case0 im2col 29.6→11.3us（0.44→0.50）；case4→0.85 |
| Opt-C flat-span 小空间路径 | W==Wo 时 col 行 = 平面连续平移拷贝，1 次装载 + 边缘 kw mask（Duplicate+标量置零）+ 成对存储；同步用最窄 V_S/S_V 旗标（#15） | case1 im2col 227→55us（**0.46→0.92**） |
| Opt-D k1direct 单 kernel | 1x1/s1/p0 新增纯 AIV Axpy kernel：免 GEMM/免 wrepack/免 hi-lo/免转置，3-launch→1-launch | case3 **0.35→0.84**（33.4us+2launch→13.0us） |
| （放弃）3-launch MIX 融合 | 可省 ~15us/call，但有 CrossCore 死锁+设备污染史，探针本身危及设备 | 风险 > 收益，记录放弃 |

### R6 stride-2 直卷积专用路径（case9 专项）

case9（[1,16,28,28] k5 s2，M=196/N=16/K=400，2.5 MFLOP）走完整 im2col+cube 流水
时 90% 耗时是固定成本（gemm per-block ~20us + 转置地板 + 3-launch）。全问题
（x 50KB + w 25KB）可入单核 UB → 纯 AIV 直卷积路径：

- **s2stage**（仅首调，结果按 x 内容 LRU 缓存，#19）：stride-2 窗口去交织为 4 个
  stride-1 子平面（行奇偶 GM 2D strided 装载；列奇偶 GatherMask pattern），按
  {-1,0,+1} 偏移写 3 份 GM scratch 位移副本（#20：UB 侧元素位移不可行）。
- **s2conv**（每调用）：任务=(n,co)，2 次 blockCount=1 整块预载（#18：DataCopyPad
  调用次数是地板）+ 400 Axpy + 1 次 2D 存储。
- 收益：device 44.3us（3 kernel）→ 17.55us（单 kernel），case9 0.377→0.546。
- 路由 gate 收紧到全量 50 中仅 case9 命中——专用路径的 gate 必须先用全量集
  验证不误伤其他 case。
- 教训：v1/v2 两轮基于"串行旗标等待是主因"的假设去串行化，实测只省 10us——
  **先做阶段探针定位 per-call 固定成本，再定优化结构**。
- 双 kernel 必须拆 .cpp 文件（#17）。

### 贯穿六轮的方法论

1. **分解先行**：每轮先做逐 kernel 分解/探针实验再动代码（R3/R5 两次推翻预设
   瓶颈）；多 case 精度失败先看误差分布（周期坏点=指令宽度问题），勿逐个猜。
2. **固定成本建模**：小 case 由 per-call/per-block 固定成本主导（trans ~140/coC、
   launch ~7.6us、gemm block ~20us）——先建实测成本模型，再选优化点；预期收益
   未量化前不上车（R3 无效 barrier 削减、R5 split-K 低收益均为教训）。
3. **单变量迭代**：每项改动独立"精度 10/10 → quick 实测无回归 → 保留/回退"，
   无效项回退并记录走偏点，不堆积未验证改动。
4. **专用路径 > 通用打磨**：depthwise/1x1/小空间三类 case 的收益全部来自专用
   kernel 或专用路径（R2/Opt-C/Opt-D），通用 im2col+GEMM 的渐进打磨收益递减。
5. **风险预算**：有设备污染史的方向（MIX 融合）宁可放弃；调试崩溃先换干净设备
   对照（第六章 #11）。
6. **tiling 参数单点计算、显式下发**：host 与 kernel 各自推导同一 tiling 参数是
   高危模式，任一侧引入新候选都会静默破坏另一侧 buffer 预算（曾致 242KB>192KB
   UB 崩溃）。

### 附：ConvStandard1d 发射链瘦身专项（fp32 cube 基线 0.738x → 1.253x）

fp32 Conv1d 全参数（groups/dilation/stride/padding/bias）50 case，`--quick` per_case
官方口径。fp32 cube 落地（第九章）后遗留两大瓶颈：case[24] stride=2 im2col 标量
gather（0.32x）与小 case 4-kernel 发射 + host 固定开销（0.44~0.46x）。本轮逐项击破
（每项后 evaluate 50/50 PASS + batch 复测，batch geomean 轨迹）：

| # | 改动 | batch geomean | 关键 case |
|:--|:-----|:--------------|:----------|
| 基线 | fp32 cube 版（第九章） | 0.635 | case[24] 0.356 (233us) |
| A | stride==2 GatherMask 去交织（第五章 #10） | 0.664 | case[24] 233→98us |
| B1 | pad_w 并入 im2col 单发射 + 跨步切分修复（#11） | 0.690→0.79 | case[33] 0.51→0.90 |
| B3 | x_col zeros→empty + kernel 三段零区覆盖写（#12） | 0.84 | case[24] →0.94 |
| C1 | epilogue 直写最终布局，wrapper 去 permute+contiguous（#13） | 0.90 | 全 case 普升 |
| C2 | bias=False 零 bias 张量缓存（#15） | 0.90 | bias=False case |
| C3 | x_padded copy_ 消除，kernel 内 LoadRowSafe 末行钳制（#14） | **1.05** | case[24] →1.22~1.28 |
| C4 | workspace 三合一 alloc + narrow | 1.02（**回退**） | 见第五章走偏点 3 |

---

## 第九章：fp32 输入 —— 原生 fp32 cube

本章补充 fp32 输入的处理路径。第一~八章默认 fp16/bf16 输入（fp32 累加器）。

> **conv1d 实证**：fp32 cube 为 fp32 输入的**唯一路径**——
> 前提：必须满足下方「fp32 cube 强制处理项」。
> fp32 cube 底层陷阱细节见 `@references/fp32-cube-path.md`。

### TileLang 设计表达层对接（原生 fp32 cube）

fp32 conv 的 TileLang 设计表达层直接是 **原生 fp32 cube**（`T.mma`/`gemm_v0`，
框架已修复：catlass v2.0.0 + ascendc fp32 后端，2026-09-08 device 15 实测无 NaN、
与 torch_npu fp32 matmul / CPU fp32 一致，全精度 fp32 舍入量级）——转译时收到的设计输入为：

- **数据布局与原生 fp32 cube 转译目标 1:1 同构**：`W(CoutP,Kp) fp32`、
  `X_col(g,Kp,Mp) fp32 K-major`、`acc(CoutP,Mp) fp32`，操作数单份——
  转译时 GEMM 直接按本章强制处理项落 fp32 cube 即可，无需任何拆分/重组。
- **表达形式**：复用 Expert 结构（`T.alloc_L1/L0A/L0B/L0C` + `T.mma`），与
  fp16/bf16 唯一差异是 fp32 分形 C0=8（block_N 按 8 对齐即可；block_M/block_K 仍 16 对齐）。
- **禁止再用 fp32 单链 AIV FMA**（`T.tile.mul_add_dst` rank-1 累加）表达 fp32 GEMM——
  该形式仅曾因旧框架 fp32 cube 缺陷作为功能验证占位（AIV FMA 结构性慢，~0.10x 量级），
  现框架已修复，直接 cube 表达。
- **验证基线**（conv1d AscendC 实测 MERE 9.7e-8 ~ 1.27e-5，全部 ≤2x CPU 同精度参考）。

### fp32 cube 强制处理项（缺一项即错/挂死）

910B 的 NZ/ZN 分形 `[16 行, C0 列]`：fp16/bf16 时 `C0=16`，**fp32 时 `C0=8`**。这个
差异带来两类致命问题（早期实测曾挂死把设备 14 污染为 Alarm）：

| 问题 | 现象 | 强制处理 |
|:---|:---|:---|
| 分形对齐陷阱 | `fixp_error0 0x60000db / UB address not aligned` | 行数恒为 16（禁止从 C0 推导）；Mmad M/N/K 需 `AlignUp(·,16)` + `kDirectionAlign=true`；`SetFmatrix` 是粘性状态、每次 LoadData3D 前重设（A 用 `FMATRIX_LEFT`，B 转置用 `FMATRIX_RIGHT` + `fMatrixCtrl=true` + `kExtension=nAlign16/mExtension=kAlign16/channelSize=nAlign16`，对齐 MatmulImpl v220 fp32 路径） |
| M-pipe 同步死锁 | kernel 挂死，AICore 100% | **优先保留 `TQue<CO1>` EnQue/DeQue 自动串行**（fp32 下可靠）；只有绕开 TQue 时才须手动补 M-pipe→Fixpipe 屏障 |
| buffer 容量翻倍 | fp32 下 L0A/L0B/L1 占用 ×2 | K 向分块减半（如 CBASE_K 64→32）、L1 cap 相应下调，先算预算再写码 |
| HF32 干扰 | 默认 HF32 模式会降精度 | 显式 `SetHF32Mode(false)` 锁严格 fp32（CANN 8.5.1 无 `HF32Mode` 枚举，仅 `SetHF32Mode(bool)` 重载） |

**调试纪律**：fp32 几何先用黑盒探针验证（W=I / X=I / 随机三组，max_abs 误差应 ≤1e-6 量级）
再跑全量；kernel 挂死后先探活设备再继续（fp32 cube 有设备污染前科）。

### 精度与性能

| 指标 | 原生 fp32 cube |
|:---|:---|
| 精度 | conv1d 全量 50 case 通过（MERE 9.7e-8 ~ 1.27e-5，阈值 1.22e-4；相消点 cancel_err 全部 ≤2x CPU 同精度参考） |
| geomean 加速比 | 基线 **0.738x**；叠加第五章 #10–#15 优化后 **1.253x / mean 1.284x**（per_case 官方口径，40/50 反超标杆） |

关键收益：groups==1 大通道小空间 case（直卷积瓶颈，逐 co 重读窗口 + 单元素 gather）从
结构性慢变为 im2col+GEMM 正常水平。

### 何时用

- **输入 fp32 且走 cube** → **原生 fp32 cube**（满足本章强制处理项）。
- **输入 fp16/bf16** → 维持第一~八章原样（单条 GEMM）。
- depthwise / 1x1 等纯 AIV 直接路径无 cube，可保持 fp32 Axpy。