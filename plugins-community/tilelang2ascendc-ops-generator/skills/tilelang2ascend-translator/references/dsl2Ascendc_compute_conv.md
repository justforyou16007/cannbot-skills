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

---

## 第七章：tiling 与资源预算经验值（910B）

| 参数 | 取值 | 依据 |
|:---|:---|:---|
| L1 预算 | 384 KB（L1=512KB 留余量） | A1+B1 双面板 + fused 模式 xNz + 常驻 B 面板 |
| baseM / baseN / baseK | 128 / 128 / 64 | L0C 单 tile = 128×128×4B = 64KB，占 L0C(128KB) 一半 |
| C0 | 16 | fp16/bf16 分形 C0 = 32B/2B |
| UB 预算 | 192 KB 总量留 headroom | im2col slab / epilogue acc tile 按此反推行数 |
| epilogue acc tile | ~16K fp32 元素/tile（64KB UB） | rowsPerTile = 8192/CoPad 反推 |
