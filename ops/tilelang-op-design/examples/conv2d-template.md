# Conv 类算子设计模板（img2col + GEMM）

> 本文是 conv2d / conv3d / 类卷积算子的 TileLang-Ascend **专用设计模板**。通用骨架见 [`design-template.md`](design-template.md)，本节只填 conv 特有结论。

---

## 0. 结构选型（Conv 强制分支，先于一切）

Conv 类算子（`nn.Conv*` / `F.conv*`）必须显式三选一，并记录结论：

> ⚠️ **fp32 输入前置判断（先于三选一）**：输入 dtype == fp32 且主计算走 Cube 时，**设计目标 = 原生 fp32 cube**（单 K 路径、fp32 操作数直进 cube；C0=8 分形 + 转译层强制处理项见 §12.2）。`T.mma`/`gemm_v0` fp32 **框架已修复**（catlass v2.0.0 + ascendc fp32 后端；实测：gemm_v0 fp32 非转置/transpose_B 与 mma fp32 均无 NaN、结果确定、与 torch_npu fp32 matmul / CPU fp32 一致，K=1024 时 max_abs ≤1.4e-4，全精度 fp32 舍入量级），**fp32 GEMM 直接用原生 fp32 cube 表达，禁止再走 AIV fp32 FMA 单链**。详见下方 §12。
>
> 📌 **AscendC 转译层注意**：原生 fp32 cube 在转译层已实证可行且更优（conv1d：50/50 过 cann-bench 新标准含相消点；叠加布局/发射链优化后 geomean 1.253x / mean 1.284x，40/50 case 反超标杆）——fp32 算子转译时按原生 fp32 cube 实现（强制处理 C0=8 分形/SetFmatrix/kDirectionAlign/SetHF32Mode(false)）。

| 模式 | 适用条件 | 省去 |
|------|---------|------|
| direct 1x1 flipped GEMM | `NCHW && kH=kW=1 && s=1 && p=0 && d=1 && Ci%16==0` | wrepack + im2col 两级发射 |
| fused img2col（`T.tile.im2col` 随路展开） | `Ci%16==0` 且（`NHWC` 或 `NCHW 的 H*W 为偶数`）且 `xNz+B 面板 ≤ L1 预算` | im2col GM 物化（最大收益） |
| normal（materialized im2col + K-major wrepack） | 通用回退 | — |

路由逻辑（完整代码见附录 §1 `select_mode`）：先判 direct → 再判 fused（L1 预算用**物理分配**而非逻辑 Co）→ 否则 normal。

约束：
- 满足 fused/direct 条件却退回 normal 时，必须在「风险点」写理由。
- **禁止默认标量逐元素 gather 的 im2col**：k'=(ci,kh,kw) 解码按段连续块/向量化表达（附录 §4 的 `unrolled_k` C0 分组转置分支是正确范式）；标量豁免仅限占比 <5% 的偶发小尾块。
- 权重 repack 是（weight 内容, shape, dtype）的确定函数，设计中标注可 host 侧 LRU 缓存。

---

## 1. 概述

### 1.1 算子名称 / 1.2 功能描述

{算子名}；把卷积映射为 GEMM（im2col），按 §0 三模式路由。

### 1.3 数学公式

$$
\text{out}[n,co,oh,ow] = \text{bias}[co] + \sum_{ci,kh,kw} x[n,ci, oh\cdot s_h-p_h+kh\cdot d_h,\ ow\cdot s_w-p_w+kw\cdot d_w] \cdot w[co,ci,kh,kw]
$$

GEMM 视角：`A[M,K] = im2col(x)`，`B[K,N] = w`（K-major），`C[M,N] = acc`（fp32），其中 `M=Ho*Wo`，`K=Ci*kH*kW`，`N=Co`。

### 1.4 算法描述（三级流水）

```
wrepack (AIV) : 权重 [Co,Ci,kH,kW] -> K-major ws [Kpad,CoPad]
im2col  (AIV) : 特征图 -> im2col [totalRows,K]（fused/direct 可省）
conv    (AIC) : GEMM  GM->L1->L0A/L0B->MMA(fp32)->L0C->acc
epilogue(AIV) : acc + bias -> cast -> 目标布局输出
```

### 1.5 数据流图

```text
direct:  raw OIHW + raw NCHW view -> flipped GEMM [Co,Ci]@[Ci,HW] -> common epilogue
fused:   NCHW xtrans（NHWC 免）+ wrepack(fused k 序) -> L1 常驻 im2col GEMM -> epilogue
normal:  wrepack + software im2col -> GM [N,M,K] -> materialized GEMM -> epilogue
```

---

## 2. 编程模式选型

### 2.1 模式结论

**Expert**（所有 AIC GEMM 主计算）；辅助 AIV kernel（xtrans/wrepack/im2col/epilogue）同为 Expert（显式 `alloc_ub` + `T.Scope("V")`）。

### 2.2 选型理由

主计算含 `T.mma`（Cube）+ 显式 `alloc_L1/alloc_L0A/L0B/L0C` + `T.Scope("C"/"V")` 核分离；fused 路径需要 L1 常驻控制与 `T.tile.im2col` 的显式参数，均需 Expert 的手动内存/同步控制。

### 2.3 模式影响

| 维度 | 选择 |
|------|------|
| 内存分配 | `T.alloc_L1 / T.alloc_L0A / T.alloc_L0B / T.alloc_L0C / T.alloc_ub` 显式 |
| 计算方式 | `T.mma`（Cube）+ `T.tile.*` / 索引赋值（Vector） |
| 作用域 | `T.Scope("C")` / `T.Scope("V")` 显式核分离 |
| 同步 | `pass_configs` 的 `AUTO_SYNC` + `MEMORY_PLANNING` 自动；跨核交接见 §8 |

---

## 3. API 映射设计

### 3.1 / 3.2 公式拆解与 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 模式 |
|------|----------|-------------|------|
| 权重重排 | `w[Co,Ci,kH,kW] -> ws[Kpad,CoPad]` | `T.copy`（repack kernel 内索引赋值） | Expert(AIV) |
| x 归一化 | `x NCHW -> xT [HW,Ci]`（fused） | `T.tile.transpose` + `T.copy`（xtrans） | Expert(AIV) |
| im2col 展开 | `x -> im2col [M,K]` | `T.tile.im2col`（fused，L1→L0A）或软件 producer（normal） | Expert |
| GM→L1 | 搬运 A/B 面板 | `T.copy` + `T.annotate_layout(make_zn_layout)` | Expert |
| L1→L0 | SplitA/SplitB | `T.copy` | Expert |
| GEMM | `C += A@B` | `T.mma(a_l0, b_l0, c_l0, init=(panel==0))` | Expert(Cube) |
| L0C→acc | 写 fp32 累加 | `T.copy(c_l0, acc[...])` | Expert |
| epilogue | `y = cast(acc + bias)` | `T.tile.add` + `T.tile.cast(CAST_RINT)` | Expert(AIV) |

### 3.3 计算伪代码（direct 1x1，完整骨架见附录 §2）

```python
with T.Kernel(cores, is_npu=True) as (cid, _):
    a_l1 = T.alloc_L1((BASE_M, BASE_K), dtype); b_l1 = T.alloc_L1((BASE_K, BASE_N), dtype)
    a_l0 = T.alloc_L0A((BASE_M, BASE_K), dtype); b_l0 = T.alloc_L0B((BASE_K, BASE_N), dtype)
    c_l0 = T.alloc_L0C((BASE_M, BASE_N), "float32")
    for task in T.serial(tasks_per_core):
        block = cid + task * cores
        if block < total_blocks:
            with T.Scope("C"):
                for panel in T.Pipelined(k_panels, num_stages=2):
                    T.copy(weight[co_start, ci_start, 0, 0], a_l1)   # LoadA
                    T.copy(x[image, ci_start, hw_start//w, hw_start%w], b_l1)  # LoadB
                    T.copy(a_l1, a_l0); T.copy(b_l1, b_l0)           # Split
                    T.mma(a_l0, b_l0, c_l0, init=(panel == 0))       # Compute
                T.copy(c_l0, acc[image, co_start, hw_start])         # CopyOut
```

### 3.4 API 可行性确认

- `T.tile.im2col`：本项目新增原语（L1 NC1HWC0 → L0A 滑窗展开，对应 AscendC `LoadData3DParamsV2`），接口与语义见文末「附录 · T.tile.im2col 语义」。
- `T.mma`、`T.copy`、`T.alloc_L1/L0A/L0B/L0C`、`T.Scope("C"/"V")`、`T.Pipelined`、`make_zn_layout`：均已验证（见文末附录完整代码）。

### 3.5 技术约束确认

| 约束 | 是否涉及 | 处理方案 |
|------|---------|---------|
| 三维 Kernel 不支持 | 是（N/C/H/W 多维） | 一维 `T.Kernel(cores)` + 手动 block 索引分解（`image/co_start/hw_start`） |
| threads 限制 | 否 | 单 AIC + 单 AIV 核分离（`T.Scope`） |
| 动态循环边界 | 是（尾块） | 静态 `T.Pipelined(k_panels)` + `if panel * BASE_K < k_size` 条件 |
| 流水线动态边界 | 否 | K 面板数静态化（`k_storage // BASE_K`） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| x | NCHW `[N,Ci,H,W]` / NHWC `[N,H,W,Ci]` | fp16/bf16 | 原始输入 |
| weight | `[Co,Ci,kH,kW]` | fp16/bf16 | 原始权重 |

### 4.2 输出张量

| 参数 | Shape | dtype |
|------|-------|-------|
| y | NCHW `[N,Co,Ho,Wo]` / NHWC `[N,Ho,Wo,Co]` | fp16/bf16 |

### 4.3 中间缓冲区

| Buffer | Shape | 层级 | 用途 |
|--------|-------|------|------|
| a_l1 / b_l1 | `[BASE_M,BASE_K]` / `[BASE_K,BASE_N]` | L1 | A/B 面板（NZ 布局） |
| a_l0 / b_l0 | `[BASE_M,BASE_K]` / `[BASE_K,BASE_N]` | L0A/L0B | Cube 操作数 |
| c_l0 | `[BASE_M,BASE_N]` | L0C | fp32 累加 |
| acc | `[N,M_storage,Co_storage]` | GM | fp32 accumulator workspace |
| im2col ws | `[N,M_storage,K_storage]` | GM | normal 路径专用 |

### 4.4 内存搬运路径

```text
direct:  GM[w/x] --T.copy--> L1 --T.copy--> L0A/L0B --T.mma--> L0C --T.copy--> acc(GM)
fused:   GM[xT] --T.copy--> L1(NC1HWC0 常驻) --T.tile.im2col--> L0A
         GM[ws] --T.copy--> L1(K-major 常驻) --T.copy--> L0B --> T.mma --> L0C --T.copy--> acc
normal:  GM[x] --software im2col--> GM[N,M,K] --T.copy--> L1 --T.copy--> L0A --> T.mma --> acc
```

### 4.5 UB 内存预算（辅助 AIV kernel）

- xtrans：slab `[Ci][T]` + stage `[T][Ci]` ≤ ~48KB，T=128/64/32 按 Ci 折半。
- epilogue：`[64,64]` fp32 acc + `[64,64]` 输出 tile ≈ 32KB。

### 4.6 动态轴

无（conv 输入静态 shape）；尾块用 `T.min(BASE_M, m_size - m_start)` 处理。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], pass_configs={
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
})
```

---

## 5. Tiling 策略

### 5.1 计算类型

**混合**（Cube `T.mma` + Vector epilogue / im2col / repack）。

### 5.2 Block 划分

| 常量 | 值 | 依据 |
|------|----|------|
| BASE_M | 128 | L0C 单 tile `128×128×4B=64KB` 占 L0C 一半 |
| BASE_N | 128 | 同上 |
| BASE_K | 64 | L0 load 分块 |
| C0 | 16 | fp16/bf16 分形 C0 = 32B/2B；fp32 为 C0=8（fp32 输入见 §12 专项） |

`block_num = batch × m_blocks × n_blocks`（`m_blocks=M_storage/BASE_M`，`n_blocks=Co_storage/BASE_N`）。

### 5.3 约束分析

- L0A: `BASE_M≥16, BASE_K≥32` ✓；L0B: `BASE_K≥32, BASE_N≥16` ✓；L0C: `BASE_M≥16, BASE_N≥16` ✓。
- L1 预算：fused 需 `2*(H*W*Ci_storage + K_storage*BASE_N)` ≤ 500KB。
- UB：epilogue 32KB < 192KB ✓。

### 5.4 注意事项

- m/n 尾块按满 baseM/baseN 计算，CopyOut 只写实际范围（stale 行列丢弃），**不要为尾块缩小 Mmad**。
- n 自适应：`nActual≥64` 按实际列加载；`nActual<64` 保持满 baseN（nActual=16 在 dav-c220 触发 L0C 读写冲突）。

---

## 6. 循环与调度结构

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| (image, m, n) block | block 级并行 | `T.Kernel(cores)` | 一维核 + 手动索引分解 |
| K 面板 | 流水 | `T.Pipelined(k_panels, num_stages=2)` | 深度 2 预取重叠 |
| 尾块条件 | 串行 + if | `T.serial` + `if ...` | 边界裁剪 |

### 6.3 流水线优化

K 面板用 `T.Pipelined(num_stages=2)`；fused 路径 x/weight 在 L1 **常驻**（按 (image,n_block) 分组），K 循环只生成 L0A 窗口块 + 切 L0B 面板。

### 6.4 尾块处理

`m_storage/n_storage/k_storage` 全部 round_up 到 BASE，CopyOut/epilogue 按 `valid_m/valid_n/valid_k` 裁剪真实区。

---

## 7. 同步策略

**模式**：自动同步（`AUTO_SYNC` + `MEMORY_PLANNING` pass），跨核 producer→consumer 依赖用 launch 顺序 + GM workspace 衔接（无跨核 flag）。

---

## 8. 融合算子设计（GEMM + epilogue，是）

conv 主计算是 Cube，epilogue 是 Vector，属 CV 融合。本模板按 **Expert 模式**（显式 `T.Scope("C"/"V")` 核分离 + acc(GM) 中转），与 Developer 模式「消 workspace/vid」的默认写法不同：

- Cube 核（`T.Scope("C")`）：`T.mma` → `T.copy(c_l0, acc)` 写 GM fp32 累加。
- Vector 核（`T.Scope("V")`，独立 epilogue kernel）：`T.copy(acc)` → `T.tile.add(bias)` → `T.tile.cast(CAST_RINT)` → 写 y。
- 跨核经 GM accumulator workspace 衔接，无 `alloc_shared` 直连。

> 若需 Developer 模式（消 workspace），参考 [design-template.md](design-template.md) §8.2；conv 因需 L1 常驻 + `T.tile.im2col` 显式控制，仍推荐 Expert。

---

## 9. 验证方案

### 9.1 Golden 函数

```python
import torch.nn.functional as F
def golden(x, weight, bias, stride, padding, dilation, groups):
    return F.conv2d(x.float(), weight.float(), bias.float() if bias is not None else None,
                    stride=stride, padding=padding, dilation=dilation, groups=groups)
```

### 9.2 L0 门槛测试计划（规则 shape，block 整除）

| 用例名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| l0_basic_1x1 | `[1,64,32,32]` w`[64,64,1,1]` | fp16 | direct 1x1 路径 |
| l0_fused_3x3 | `[1,64,56,56]` w`[64,64,3,3]` p1 | fp16 | fused img2col 路径 |
| l0_normal_5x5 | `[1,32,32,32]` w`[64,32,5,5]` p2 d2 NHWC | bf16 | normal 回退 |

### 9.3 精度标准

fp16: atol 2⁻¹⁴ / rtol 2⁻⁹；bf16: atol 2⁻¹⁰ / rtol 2⁻⁶；输出 fp32 中间 acc 后 cast（完整分层套件由测试设计环节生成，不在本文枚举）。

---

## 10. 风险点与注意事项

- **K 顺序契约**：fused img2col 的硬件展开 k 序是 `(ci1,kh,kw,ci0)`（C0 块最外层），software im2col 是 `(kh,kw,ci1,ci0)`；两者仅在 `Ci==16` 或 `kH*kW==1` 时重合。weight repack 必须按所选模式用同一 k 序，**不得让未标记的 K-major ws 跨路径复用**。
- **C1≥15（Ci≥240）硬件缺陷产生垃圾** → gate `C1≤14`，Ci=256/512 回退 normal。
- **fused L1 预算用物理分配**：B 面板按 BASE_N 完整分配（非逻辑 Co），否则预算误判。
- **满足 fused/direct 却回退 normal 必须写理由**（[decision-tree.md](../references/decision-tree.md) §5 强制）。
- **禁止默认标量 gather im2col**：主数据路径必须向量化，标量豁免仅 <5% 尾块。
- **fp32 输入走原生 fp32 cube**（设计与转译目标均为 fp32 cube，`T.mma`/`gemm_v0` fp32 已支持，见 §0 前置判断与 §12；无需再走 AIV FMA 单链）。

---

## 11. 交付清单

### 11.1 目录结构（完整代码见文末「附录」）

```text
design/
├── README.md                 # 本设计的结论 + 契约（可精简为 design.md）
├── block_level/
│   ├── conv2d_base.py        # 共享常量、select_mode 路由、resolve_design、auxiliary schedules
│   ├── direct_1x1.py / fused_im2col.py / materialized_im2col.py
└── tile_level/
    ├── conv2d_base.py        # 统一导出 + xtrans/wrepack/epilogue 辅助 kernel
    └── direct_1x1.py / fused_im2col.py / materialized_im2col.py
```

### 11.4 实现顺序

1. `select_mode` 路由 + `resolve_design`（先定模式）→ 2. block 级 grid/contract → 3. tile 级端到端 contract → 4. 主计算 kernel → 5. 辅助 AIV kernel → 6. Golden + L0 验证。

---

## 12. fp32 输入专项：原生 fp32 cube（TileLang 直接表达）

> 本节是 fp32 输入 conv 的专用设计结论（§0 已路由：**设计/转译目标 = 原生 fp32 cube**）。
> `T.mma`/`gemm_v0` fp32 **框架已修复**（catlass v2.0.0 + ascendc fp32 后端；实测：gemm_v0 fp32 非转置/transpose_B 与 mma fp32 均无 NaN、结果确定、与
> torch_npu fp32 matmul / CPU fp32 一致，全精度 fp32 舍入量级）——fp32 GEMM 直接用原生
> fp32 cube 表达，**禁止再走 AIV fp32 FMA 单链**。fp32 cube 的转译层陷阱与强制处理项见 §12.2。

### 12.1 设计决策（强制）

| 输入 dtype | 主计算类型 | 设计决策 |
|-----------|-----------|---------|
| fp16 / bf16 | GEMM/Conv | 单条 `T.mma`，标准 fp16 cube（C0=16） |
| **fp32** | GEMM/Conv | **原生 fp32 cube**（`T.mma`/`gemm_v0` 直接表达，C0=8 分形；转译层强制处理项见 §12.2） |
| 任意 | 纯 Vector（无 `T.mma`） | 可直接 fp32 Axpy/向量化，无 cube 无此约束 |

### 12.2 fp32 cube 的陷阱与边界（AscendC 转译层强制处理项）

Ascend 910B 的 NZ/ZN 分形 `[16 行, C0 列]`：fp16/bf16 时 `C0=16`，**fp32 时 `C0=8`**。这个分形差异引入两类致命问题：

1. **分形对齐陷阱**：凡是从 `C0` 推出行数/`srcStride`/`mSize` 的代码，fp16 下恰好等于 `/16` 全对，fp32 下全错，报 `fixp_error0 0x60000db / UB address not aligned`。
2. **M-pipe 同步死锁**：fp32 路径绕开 TQue 自动同步后需手动补 M-pipe→Fixpipe 屏障，漏补则 Cube 与 Fixpipe 竞态死锁，AICore 挂死并污染设备（实测曾把设备 14 污染为 Alarm）。

结论：**设计与转译目标均为原生 fp32 cube**（TileLang 表达层直接 `T.mma`/`gemm_v0` fp32，
转译层逐项强制处理，conv1d 实证定稿）——转译层强制处理项：
① 行数恒 16 不从 C0 推导 + Mmad `AlignUp(·,16)` + `kDirectionAlign=true`；
② `SetFmatrix` 粘性状态每次重设（对齐 MatmulImpl v220 fp32 路径）；
③ `SetHF32Mode(false)` 锁严格 fp32；④ 保留 `TQue<CO1>` 自动 M→FIX 串行，不绕开手写屏障；
⑤ buffer 容量翻倍适配（K 分块减半）。缺一项即错/挂死。

### 12.3 TileLang 设计表达（原生 fp32 cube）

fp32 GEMM 直接复用 §2/§3 的 Expert 结构（`T.alloc_L1/L0A/L0B/L0C` + `T.mma`），与
fp16/bf16 唯一差异是 fp32 分形 C0=8。要点：

| 项 | fp32 cube 表达 |
|----|---------------|
| 操作数 dtype | `dtype="float32"`、累加器 `"float32"`；无 hi/lo 拆分、无 Cast 截断，单 MMA 路径 |
| 分形 | C0=8（fp16/bf16 为 16）；block_M/block_K(kL0Size) 须 16 对齐，block_N 按 C0=8 对齐即可（实测 block_N=8/16/24 均通过） |
| 内存 | L0A/L0B 容量按 fp32 翻倍预算（§12.2 强制处理项 ⑤），K 分块减半 |
| 精度 | fp32 cube 结果与 torch_npu fp32 matmul / CPU fp32 一致（实测，全精度 fp32 舍入量级，K=1024 时 max_abs ≤1.4e-4） |

通用 TileLang 坑位（同样适用于 fp32 cube）：
- `@tilelang.jit(out_idx=[-1])` 下输出由框架自动分配，调用时**不要显式传入输出**（否则读回未写入的零缓冲）。
- 纯向量辅助核（xtrans/wrepack/epilogue）必须挂 `TL_ASCEND_AUTO_SYNC`，否则 GM→UB→Vector→UB→GM 无同步 round-trip 即错。
- Expert 模式 L0C **直接 `T.copy(C_L0, out[...])` 写回 GM**，避免 L0C→alloc_ub→GM 中转的 codegen 缺陷；bias 拆成独立向量核。

### 12.4 验证结论（实测910B2C）

| 验证项 | 结果 |
|--------|------|
| `T.gemm_v0` fp32（非转置，256³/512³/1024³） | 无 NaN、结果确定、与 torch_npu fp32 matmul / CPU fp32 一致（K=1024 时 max_abs 1.14e-4 vs fp64 参考，全精度 fp32 舍入量级） |
| `T.mma` fp32（Expert L0A/L0B/L0C） | 同上 |
| `T.gemm_v0` fp32 transpose_B（QK，128³） | 无 NaN、正确 |

> 精度口径：TileLang fp32 cube 为**全精度 fp32**（与 CPU fp32/fp64 参考 max_abs ≤1.4e-4
> @ K=1024，非 HF32/tf32 降精度），可直接作为 fp32 GEMM 的功能验证与精度口径。

### 12.5 性能

| 实现 | geomean 加速比 |
|------|---------------|
| **原生 fp32 cube + 布局/发射层优化**（conv1d） | **1.253x geomean / 1.284x mean**（per_case 官方口径；0.738x fp32 cube 基线 → GatherMask 去交织 + 发射链 4→3 + epilogue 直写布局等 6 项优化） |

关键收益：groups==1 大通道小空间 case（直卷积瓶颈，逐 co 重读窗口 + 单元素 gather）从结构性慢变为 im2col+GEMM 正常水平。

---

## 附录：骨干代码 + 注解（block_level + tile_level）

### 1. 路由与共享配置（block_level 骨干）

```python
BASE_M = 128; BASE_N = 128; BASE_K = 64   # L0C 单 tile 128×128×4B=64KB 占一半
C0 = 16                                    # fp16/bf16 分形 C0 = 32B/2B（fp32 为 C0=8，见 §12）
FUSED_L1_BUDGET = 500 * 1024               # fused 的 L1 预算

@dataclass(frozen=True)
class Conv2dConfig:
    """算子配置：batch/ci/h/w/co/kernel_h/kernel_w + stride/pad/dilation/data_format/dtype。"""

def select_mode(cfg) -> str:
    """三模式路由（先于一切）。direct 1x1 → fused img2col → materialized。"""
    direct = (cfg.data_format == "NCHW" and cfg.kernel_h == cfg.kernel_w == 1
              and cfg.stride_h == cfg.stride_w == 1 and cfg.pad_h == cfg.pad_w == 0
              and cfg.dilation_h == cfg.dilation_w == 1 and cfg.ci % C0 == 0)
    if direct:
        return "direct_1x1"
    ci_pad = round_up(cfg.ci, C0)
    k_storage = round_up(ci_pad * cfg.kernel_h * cfg.kernel_w, BASE_K)
    # fused 的 L1 预算用【物理分配】——B 面板按 BASE_N 完整分配，而非逻辑 Co
    l1_bytes = 2 * (ci_pad * cfg.height * cfg.width + k_storage * BASE_N)
    fused = (cfg.ci % C0 == 0
             and (cfg.data_format == "NHWC" or (cfg.height * cfg.width) % 2 == 0)
             and l1_bytes <= FUSED_L1_BUDGET)
    return "fused_im2col" if fused else "materialized_im2col"

def resolve_design(cfg) -> DesignRoute:
    """每模式的 stage 链（launch 顺序）。"""
    # direct_1x1:      (process_direct, epilogue)
    # fused_im2col:    (xtrans?, weight_repack, process_fused, epilogue)  # NCHW 才需要 xtrans
    # materialized:    (weight_repack, software_im2col, materialized_gemm, epilogue)
```

### 2. direct 1x1 flipped GEMM（block_level 骨干）

> 注解：1x1 的 im2col 是恒等变换，x NCHW 本身就是 `[K=Ci, M=H*W]` 的 B 操作数，权重做 A 即可，零 transpose、零 wrepack/im2col。

```python
def direct_1x1_block_contract(cfg):
    """flipped GEMM [Co,Ci] @ [Ci,H*W] 的 grid + 布局契约。"""
    ci_storage = round_up(cfg.ci, BASE_K); co_storage = round_up(cfg.co, BASE_M)
    hw_storage = round_up(cfg.height * cfg.width, BASE_N)
    total_blocks = cfg.batch * (co_storage//BASE_M) * (hw_storage//BASE_N)
    return {"logical_gemm": "[Co,Ci] @ [Ci,H*W]", "k_panels": ci_storage // BASE_K,
            "pipeline_stages": 2, "dataflow": "GM->L1->L0->MMA->L0C->GM"}

@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def conv2d_base_process_direct(cfg):
    with T.Kernel(cores, is_npu=True) as (cid, _):
        a_l1 = T.alloc_L1((BASE_M, BASE_K), cfg.dtype)   # A 面板（NZ，make_zn_layout）
        b_l1 = T.alloc_L1((BASE_K, BASE_N), cfg.dtype)   # B 面板（NZ）
        a_l0 = T.alloc_L0A((BASE_M, BASE_K), cfg.dtype)  # Cube 操作数
        b_l0 = T.alloc_L0B((BASE_K, BASE_N), cfg.dtype)
        c_l0 = T.alloc_L0C((BASE_M, BASE_N), "float32")  # fp32 累加
        for task in T.serial(tasks_per_core):
            block = cid + task * cores
            if block < total_blocks:
                with T.Scope("C"):
                    for panel in T.Pipelined(k_panels, num_stages=2):
                        T.copy(weight[co_start, ci_start, 0, 0], a_l1)  # LoadA: GM ND→L1 NZ
                        T.copy(x[image, ci_start, ...], b_l1)           # LoadB
                        T.copy(a_l1, a_l0); T.copy(b_l1, b_l0)          # Split: L1→L0A/L0B
                        T.mma(a_l0, b_l0, c_l0, init=(panel == 0))      # Compute: 首 panel 清零
                    T.copy(c_l0, acc[image, co_start, hw_start])        # CopyOut: L0C→acc(GM)
```

### 3. fused img2col（block_level 骨干）

> 注解：A 侧不落地——x 在 L1 常驻 NC1HWC0，`T.tile.im2col` 随路展开到 L0A；B 侧 K-major 权重按 `(ci1,kh,kw,ci0)` 序常驻。按 `(image, n_block)` 分组，使 x 与 B 同时常驻。

```python
def fused_im2col_block_contract(cfg):
    """fused grid、常驻范围、L1 足迹。"""
    m_storage = round_up(ho * wo, BASE_M); k_storage = round_up(Ci*kH*kW, BASE_K)
    n_storage = round_up(round_up(co, C0), BASE_N)
    resident_groups = batch * n_blocks   # x 常驻 (image)，B 常驻 (n_block)
    return {"weight_k_order": ("ci1", "kh", "kw", "ci0"),      # LoadData3D 实测 k 序
            "residency_scope": "(image, n_block) across all m_blocks"}

@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def conv2d_base_process_fused(cfg):
    with T.Kernel(cores, is_npu=True) as (cid, _):
        x_l1 = T.alloc_L1((hw, ci_storage), cfg.dtype)      # 常驻 image（NC1HWC0）
        b_l1 = T.alloc_L1((k_storage, BASE_N), cfg.dtype)   # 常驻 B 面板
        for group in T.serial(groups_per_core):
            with T.Scope("C"):
                T.copy(x[image, 0, 0], x_l1); T.copy(weight[0, n_start], b_l1)  # 常驻
                for m_block in T.serial(m_blocks):
                    for k_panel in T.Pipelined(k_panels, num_stages=2):
                        T.tile.im2col(a_l0, x_l1, image_shape=(H,W,Ci),
                                      kernel=(kH,kW), stride=(sH,sW), dilation=(dH,dW),
                                      padding=(pW,pW,pH,pH),
                                      pos_m=m_start, pos_k=k_start,
                                      valid_m=..., valid_k=..., pad_value=0.0)  # LoadA3D
                        T.copy(b_l1[k_start, 0], b_l0)   # SplitB
                        T.mma(a_l0, b_l0, c_l0, init=(k_panel == 0))
                    T.copy(c_l0, acc[image, m_start, n_start])
```

### 4. materialized im2col（block_level 骨干）

> 注解：normal 回退 = AIV software im2col producer 物化 `[N,M,K]` GM，AIC GEMM consumer 消费。producer 有两种静态调度：`unrolled_k`（BASE_K 编译期固定 + C0 分组转置，正确范式）与 `serial_k`（逐项标量索引解码，通用回退）。

```python
def materialized_launch_pipeline(cfg):
    """producer→consumer 依赖 = im2col workspace [N,M_storage,K_storage]。"""
    return {"launch_order": ("software_im2col", "materialized_gemm", "epilogue"),
            "producer_core": "AIV", "consumer_core": "AIC",
            "k_order": ("kh", "kw", "ci1", "ci0"),
            "producer_strategy": "c0_grouped_transpose" if (ci%C0==0 and kW<=7) else "scalar_index_decode"}

# producer（AIV, T.Scope("V")）: T.alloc_ub([BASE_M,BASE_K]) + T.tile.fill 零填充
#   unrolled_k: NCHW 按 C0 分组载入 + 16x16 transpose；NHWC 按 C0 向量拷
#   serial_k:   (m,k) 双重 T.serial + 越界 T.And 条件 + 索引赋值
#   -> T.copy(tile_ub, workspace[image, m_start, k_start])
# consumer（AIC, T.Scope("C")）: 与 direct 的 GEMM 结构完全一致（LoadA/LoadB→Split→T.mma→CopyOut）
```

### 5. tile 级统一导出 + 辅助 kernel（tile_level 骨干）

> 注解：tile 级负责「端到端 launch + 中间布局契约」，并把 xtrans / weight_repack / epilogue 三个 AIV 辅助 kernel 统一导出。

```python
@dataclass(frozen=True)
class TilePipelineContract:
    mode, stages, x_layout, weight_layout, accumulator_layout, output_layout, workspace_layout

def resolve_tile_pipeline(cfg, mode):
    """每模式的 launch 顺序 + 布局契约。"""
    # direct:       stages=(process_direct, epilogue);            acc=[N,Co_storage,HW_storage]
    # fused:        stages=(xtrans?, weight_repack, process_fused, epilogue); acc=[N,M_storage,Co_storage]
    # materialized: stages=(weight_repack, software_im2col, materialized_gemm, epilogue); ws=[N,M_storage,K_storage]

# 三个 AIV 辅助 kernel（均为 T.Kernel + T.alloc_ub + T.Scope("V")）:
#   xtrans        : NCHW x[img,Ci,H,W] --T.tile.transpose(16x16)--> xT[img,H*W,Ci]
#   weight_repack : OIHW --索引赋值(k 序解码)--> ws[K_storage,Co_storage]（fused=(ci1,kh,kw,ci0) / normal=(kh,kw,ci1,ci0)）
#   epilogue      : acc(fp32) --T.tile.add(bias)--> T.tile.cast(CAST_RINT)--> y（NCHW/NHWC）
```

### 6. 三模式 tile 级 plan（tile_level 骨干）

> 注解：每个 mode 一个 `*Plan` dataclass（静态 storage/grid/layout 契约）+ `make_*_plan(cfg)`（校验 + 构造）+ `@tilelang.jit` 主 kernel（镜像 block_level 的 Process，用 plan 驱动）。计划先行、kernel 后行是「设计即文档」的关键。

```python
@dataclass(frozen=True)
class Direct1x1Plan:      # 同 FusedIm2colPlan / MaterializedIm2colPlan
    hw, ci_storage, co_storage, hw_storage, m_blocks, n_blocks, k_panels, ...
    a_gm_layout = "raw OIHW viewed as [Co,Ci]"
    b_gm_layout = "raw NCHW viewed as [N,Ci,HW]"
    c_gm_layout = "[N,Co_storage,HW_storage] FP32"

def make_direct_1x1_plan(cfg): ...   # 校验 routing 条件 + 算 storage/grid
def make_fused_im2col_plan(cfg): ... # 校验 C0 完整组 + L1 预算
def make_materialized_im2col_plan(cfg): ...  # producer/consumer 契约 + strategy
```

### T.tile.im2col 语义

> 源是物理 NC1HWC0 的 L1 tile，目标是 L0A 矩阵 tile。该操作同时包含滑窗采样、重叠元素复用、stride、dilation、padding、M/K 子块选择和 L0A 布局重排。

```python
T.tile.im2col(
    dst_l0a, src_l1,
    image_shape=(H, W, Ci),
    kernel=(KH, KW),
    stride=(SH, SW),
    dilation=(DH, DW),
    padding=(left, right, top, bottom),
    pos_m=m_start, pos_k=k_start,
    valid_m=m_actual, valid_k=k_actual,
    pad_value=0.0,
)
```

> Ascend lowering 对应 `AscendC::LoadData3DParamsV2<T>` + `AscendC::LoadData(dst, src, params)`。
> K 顺序契约：fused 用 `(ci1,kh,kw,ci0)`（LoadData3D 实测 k 序，C0 块最外层），materialized 用 `(kh,kw,ci1,ci0)`；两者仅 `Ci==16` 或 `kH*kW==1` 时重合。
