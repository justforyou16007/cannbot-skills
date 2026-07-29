# Conv2dBase TileLang 设计

## 结论

`conv2d_base` 新增以下 TileLang 原语：

```python
T.tile.im2col(dst_l0a, src_l1, ...)
```

权重由 TileLang AIV repack kernel 整理为二维 K-major 布局，随后使用
layout-aware `T.copy`：

```text
GM [K,N] ND --T.copy--> L1 NZ --T.copy--> L0B ZN
```

L0C 使用 `T.copy` 写入按完整 M/N tile 补齐的 FP32 accumulator
workspace。TileLang AIV epilogue 完成 bias、裁剪、类型转换和
NCHW/NHWC 恢复。

对应文件：

- `block_level/conv2d_base.py`：共享配置、合法性 gate、mode router 和统一导出；
- `block_level/direct_1x1.py`：direct grid、persistent task 与 K-panel 循环；
- `block_level/fused_im2col.py`：fused grid、persistent task 与 K-panel 循环；
- `block_level/materialized_im2col.py`：AIV producer、AIC consumer、
  persistent task 和 GM workspace 依赖；
- `tile_level/conv2d_base.py`：统一导出入口，以及 xtrans、weight repack、
  epilogue 三个公共辅助 kernel；
- `tile_level/direct_1x1.py`：完整的 1x1 flipped GEMM；
- `tile_level/fused_im2col.py`：完整的 L1→L0A 窗口展开与 GEMM；
- `tile_level/materialized_im2col.py`：完整的 AIV 软件 im2col producer
  与 AIC GEMM consumer。

## 完整 kernel pipeline

统一 `ModelNew` 只做参数校验、静态 shape/view 和 kernel launch 编排：

```text
direct:
  raw OIHW + raw NCHW flattened view
    -> direct GEMM -> common epilogue

fused:
  NCHW xtrans（NHWC 直接复用 raw channel-inner）+ weight_repack/fused
    -> L1-resident im2col GEMM -> common epilogue

materialized:
  weight_repack/materialized
    + raw NCHW/NHWC -> software im2col -> GM [N,M,K]
    -> materialized GEMM -> common epilogue
```

## AscendC 函数与 TileLang 设计映射

TileLang 的公开 factory 使用 `snake_case`，核心语义词与 AscendC
类方法保持一致；原有 factory 名作为兼容 alias 保留给 wrapper。

| AscendC 类/函数 | TileLang factory 或阶段 | 等价语义 |
| --- | --- | --- |
| `Conv2dBaseKernel::Process`（direct 分支） | `conv2d_base_process_direct` | `(N,Co-tile,HW-tile)` task 遍历与 K-panel GEMM |
| `Conv2dBaseKernel::Process`（materialized） | `conv2d_base_process` | `(N,M-tile,Co-tile)` task 遍历，从 im2col GM 取 A |
| `Conv2dBaseKernel::ProcessFused` | `conv2d_base_process_fused` | x/weight L1 驻留并遍历输出 M tile |
| `ComputeBlock` / `ComputeBlockFused` | 各 `process*` kernel 的 block body | 完成一块输出的 K-panel 累加 |
| `Prefetch`、`LoadA`、`LoadB` | K pipeline 中前两次 `T.copy` | GM ND 到 L1 matrix tile |
| `LoadA3D` | `T.tile.im2col` | NC1HWC0 L1 滑窗展开到 L0A |
| `SplitA`、`SplitB` | L1 到 L0A/L0B 的 `T.copy` | 将 L1 panel 转成 Cube operand layout |
| `Compute` | `T.mma(..., init=...)` | 首个 K panel 清零 L0C，后续 panel 累加 |
| `CopyOut` | L0C 到 accumulator 的 `T.copy` | 写 FP32 中间结果，bias 留给 epilogue |
| `Conv2dIm2colKernel::Process` | `conv2d_base_im2col_process` | 在 fast/legacy producer 间静态分派 |
| `ProcessFast` | 内层 `process_fast` prim func | 分组加载、转置/展开并写 im2col GM |
| `ProcessLegacy` | 内层 `process_legacy` prim func | 通用标量索引解码并写 im2col GM |
| `Conv2dXtransKernel::Process` | `conv2d_base_xtrans_process` | NCHW 转 `[N,HW,Ci]` channel-inner GM |
| `LoadBlock` / `TransBlock` | xtrans 的 load/16×16 transpose 阶段 | `[Ci,HW]` slab 到 `[HW,Ci]` stage |
| `Conv2dWeightRepackKernel::Process` | `conv2d_base_weight_repack_process` | OIHW 到对应模式的 K-major 权重 |
| `Conv2dEpilogueKernel::Process` | `conv2d_base_epilogue_process` | 路由 direct/NHWC/NCHW 输出布局 |
| `ProcessDirect` / `ProcessNHWC` / `ProcessNCHW` | epilogue 中按 `mode/data_format` 的索引分支 | 读 FP32、加 bias、舍入转换并恢复输出布局 |

Block-level 使用同名加 `_block` 的 factory，例如
`conv2d_base_process_fused_block`，表示相同函数边界下的 grid、驻留与
pipeline 设计。`conv2d_base_direct_1x1`、`conv2d_base_fused`、
`conv2d_base_materialized_gemm` 等名称是兼容入口。

## 逐函数语义核对

- 布局：normal/materialized 的 A 为 `[M,K]`，B 为 `[K,Co]`，输出
  accumulator 为 `[N,M,Co]`；direct 保持 AscendC 的 flipped GEMM，
  即 `[Co,Ci] @ [Ci,HW] -> [Co,HW]`。
- K 顺序：fused 使用 `(ci1,kh,kw,ci0)`，与 `LoadData3D` 一致；
  materialized 使用 `(kh,kw,ci1,ci0)`，与独立 im2col producer 及
  `fused=0` weight repack 一致。
- 驻留：fused 的 x 在 image 范围驻留，B 在 Co tile 范围驻留；设计按
  `(image,n_block)` 分组连续处理 M blocks，从而保持 AscendC
  `curImg/curNIdx` cache 的生命周期含义。
- task 循环：AIC 使用 block/core 轮转；AIV helper 使用 tile/core 轮转。
  materialized producer 完成后，其 GM workspace 才交给 AIC consumer。
- 存储通路：AIC 为 GM→L1→L0A/L0B→MMA→L0C→GM；AIV helper 为
  GM→UB→GM。UB 与 L1 之间没有直接 copy。
- MMA 初始化：每个输出 block 的首个 K panel 使用 `init=True`，后续
  panel 使用 `init=False`，对应 `cmatrixInitVal`。
- epilogue：accumulator 保持 FP32；bias 先转换为 FP32，输出转换使用
  `CAST_RINT`，随后按 direct/NHWC/NCHW 的公共张量布局写回。

`xtrans` 为 fused NCHW 输入执行 16 通道分组转置；fused NHWC 已是
channel-inner，直接使用原输入。`weight_repack` 为 fused/materialized
生成各自 K 顺序的 K-major 权重。公共 epilogue 从 direct/matrix
accumulator 读取有效区，加入可选 bias，FP32 转输入 dtype，并按
NCHW/NHWC 写输出。

## 设计文件的完整契约

Block-level 入口负责参数合法性、实现路由、stage 顺序和辅助 AIV
schedule；三个实现文件分别给出 task domain、core 映射、resident
对象、各级 buffer 和主循环。Tile-level 入口提供端到端
`TilePipelineContract`，三个实现文件提供静态 plan 和对应 Expert kernel。

每个主计算 plan 均包含：

- 逻辑问题维度以及 M/N/K 的存储维度；
- GM、L1、L0A/L0B/L0C 和 accumulator 的布局；
- block/core 数量、persistent task 顺序和每核任务数；
- GM→L1→L0、MMA、L0C→GM 的 buffer 生命周期；
- K panel 数、流水深度以及第一次 MMA 的初始化语义；
- 与 xtrans、weight repack、software im2col、epilogue 的接口契约。

三条路径的 workspace 关系为：

| 路径 | 主计算输入 | 中间 workspace | accumulator |
| --- | --- | --- | --- |
| direct | raw OIHW、raw NCHW flattened view | FP32 accumulator | `[N,Co,HW]` FP32 |
| fused | channel-inner x、fused K-major weight | xtrans、weight repack 输出 | `[N,M,Co]` FP32 |
| materialized | GM im2col、materialized K-major weight | `[N,M,K]` im2col | `[N,M,Co]` FP32 |

所有 tile-level kernel 使用 Expert mode：显式分配 UB/L1/L0，
显式编排 `T.copy`、`T.serial/T.unroll`、`T.Pipelined` 和 `T.mma`。

## 三条实现路径

### 1x1 direct

当前实现对满足以下条件的 NCHW 卷积直接使用 flipped GEMM：

```text
[Co,Ci] @ [Ci,H*W] -> [Co,H*W]
```

条件为 `KH=KW=1`、stride/dilation 为 1、padding 为 0、`Ci % 16 == 0`
。与 AscendC host 路由一致，该路径跳过 weight repack、xtrans 和 im2col；
OIHW 在 `KH=KW=1` 时直接视作 `[Co,Ci]`，连续 NCHW 空间维直接视作
`[Ci,H*W]`。主 kernel 按 K panel 执行：

```text
weight GM -> L1 -> L0A
x GM      -> L1 -> L0B
                    -> T.mma -> fp32 accumulator GM
```

Ci、Co 和 H*W 的存储维度补齐到 Cube tile，epilogue 裁剪到真实输出。

### Fused im2col

主路径的数据流为：

```text
x channel-inner GM
  -> T.copy / ND-to-NZ
L1 physical NC1HWC0
  -> T.tile.im2col
L0A Zz
  -> T.mma(weight L0B)
L0C fp32
  -> T.copy
padded accumulator GM
  -> epilogue
NCHW/NHWC output
```

它与现有 `conv2d_base.h` 的 `ProcessFused` / `LoadA3D` 路径对应，并消除
`[N, Ho*Wo, K]` 的 GM im2col workspace。

当前 host 的 fused 合法性 gate 是：

- 非 direct 1x1；
- `Ci % 16 == 0`；
- NHWC 可直接使用 channel-inner 输入；NCHW 的 xtrans 路径还要求
  `H*W` 为偶数，以满足 32B 行搬运约束；
- `Ci_storage*H*W*dtype_bytes
  + K_storage*BASE_N*dtype_bytes <= 500 KiB`。

`T.tile.im2col` lowering 必须在编译期执行同类 target/profile 合法性检查。
若输入已经由更上层显式物化为合法 NC1HWC0 GM 布局，NCHW xtrans 的
`H*W` 限制可由该 producer 自己消解，而不是成为 im2col 原语的固有限制。

### Materialized im2col

fused gate 未满足时，使用两个顺序启动的 kernel：

```text
x -> AIV software im2col -> GM [N,M,K]
                             |
                             v
                AIC tiled GEMM -> accumulator -> epilogue
```

软件 im2col 直接接受 public 接口的 raw NCHW `[N,Ci,H,W]` 或 raw NHWC
`[N,H,W,Ci]`，根据静态 `data_format` 生成对应 tensor shape 和读取索引。
它对每个 `(m,k)` 显式计算：

```text
oh = m // Wo
ow = m % Wo
k  = (kh, kw, ci1, ci0)
ih = oh*stride_h - pad_h + kh*dilation_h
iw = ow*stride_w - pad_w + kw*dilation_w
```

空间越界、通道补齐和 M/K 尾部均写零，结果写入补齐后的
`[N,M_storage,K_storage]` GM workspace。后续 AIC kernel 使用完整的
`GM im2col -> L1 -> L0A`、`GM K-major weight -> L1 -> L0B`、`T.mma`
和 `L0C -> accumulator` 流程。

materialized producer 提供两种静态调度：

- fast：固定 `[BASE_M,BASE_K]` tile，并展开 K 维索引解码与加载序列；
- legacy：保留串行 K 循环，逐项完成 stride、dilation、padding 和
  `(kh,kw,ci1,ci0)` 映射，覆盖通用配置。

## Buffering 与驻留

三种 AIC GEMM 的 K-panel 循环使用 `T.Pipelined(..., num_stages=2)`，
表达深度为 2 的 L1 ping-pong：当前 panel 执行 L1→L0 和 MMA 时预取下一
panel。fused 路径把当前 image 的 channel-inner feature map 与当前 N tile
的完整 K-major 权重驻留 L1，K 循环只生成 L0A 窗口块并切取 L0B panel。
M/N/K 的存储维均补到完整 tile，`valid_m/valid_n/valid_k` 约束真实有效区，
最终由公共 epilogue 裁剪。

## TileLang intrinsic 接口

设计新增 `T.tile.im2col`，用于 L1 NC1HWC0 到 L0A 的滑窗展开。它需要在
TileLang IR 和 Ascend lowering 中注册。

xtrans、weight repack、materialized im2col 和 epilogue 使用
`tilelang-api-best-practices` 中已有的 Expert mode 接口：
`T.alloc_ub/L1/L0A/L0B/L0C`、`T.copy`、`T.serial/T.unroll`、
`T.tile.fill/add/cast`、16×16 `T.tile.transpose`、`T.mma` 与显式索引
赋值。
其中 `T.copy` 负责同形状 tile 搬运，shape/layout 变化由循环中的源、目标
索引关系表示。

## `T.tile.im2col` 的语义

建议接口：

```python
T.tile.im2col(
    dst_l0a,
    src_l1,
    image_shape=(H, W, Ci),
    kernel=(KH, KW),
    stride=(SH, SW),
    dilation=(DH, DW),
    padding=(left, right, top, bottom),
    pos_m=m_start,
    pos_k=k_start,
    valid_m=m_actual,
    valid_k=k_actual,
    pad_value=0.0,
)
```

源是物理 NC1HWC0 的 L1 tile，目标是 L0A 矩阵 tile。该操作同时包含滑窗
采样、重叠元素复用、stride、dilation、padding、M/K 子块选择和 L0A 布局
重排。它作为独立的窗口展开原语，使 `T.copy` 保持矩阵/布局搬运语义。

Ascend lowering 对应：

```text
image_shape       -> l1H, l1W, channelSize
kernel            -> filterH, filterW
stride            -> strideH, strideW
dilation          -> dilationFilterH, dilationFilterW
padding           -> padList
pos_m / valid_m   -> mStartPt / mExtension
pos_k / valid_k   -> kStartPt / kExtension
pad_value         -> padValue
```

最终生成 `AscendC::LoadData3DParamsV2<T>` 和
`AscendC::LoadData(dst_l0a, src_l1, params)`。

## K 顺序契约

现有 fused-im2col 实现中，`LoadData3D` 的实测 K 顺序是：

```text
(ci1, kh, kw, ci0)
k = (((ci1 * KH + kh) * KW + kw) * 16 + ci0)
```

因此 fused 权重必须 repack 为：

```text
weight_kmajor[k, co]
```

并使用完全相同的 K 编码。materialized-im2col 路径当前使用
`(kh, kw, ci1, ci0)`；两者仅在 `Ci == 16` 或 `KH*KW == 1` 时自然一致。
wrapper/host 必须把 K 顺序作为所选实现模式的一部分，不能让同一个未标记的
K-major workspace 在两条路径间复用。

## 与现有 fused-im2col 实现的差异

现有 fused-im2col 实现的尾通道通过 `nActual/nEff` 控制 `LoadData2D` 和
`Mmad`，而设计稿把 K-major 权重和 accumulator 的 N 维显式补到完整
`BASE_N` tile。这样
TileLang 中两级 `T.copy` 始终是同逻辑形状，epilogue 只读取真实 `Co`。
这是 DSL 层为保持 copy 语义而做的存储约定，不改变有效输出。

设计把 xtrans、weight repack 和 epilogue 作为 wrapper 或独立 kernel，
三个 tile-level 实现文件分别覆盖 direct、fused-im2col 和
materialized-im2col，统一入口负责导出四个 kernel factory。
