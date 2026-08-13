# Cube Kernel 实现模式（生成端参考）

## 1. 核心约束（5 条，去重 kernel-rules）

1. **AIC 只跑 Cube 原语**：`DataCopy(GM→L1)` / `LoadData(L1→L0A|L0B)` / `Mmad` /
   `Fixpipe(L0C→GM|UB|L1)` / `PipeBarrier<PIPE_M>` / `CrossCoreSetFlag/WaitFlag` /
   核内 `SetFlag/WaitFlag<HardEvent::MTE1_MTE2|MTE2_MTE1>`。
2. **严禁 vector 原语**（Duplicate/Cast/Muls/Add/Mul/Exp/Softmax/Reduce/Brcb...）→ coredump。
   详同目录 `kernel-rules.md` RK-A04。
3. **分形**：L1 内部 NZ 分形；GM/UB 是 ND。`DataCopy(Nd2NzParams)` 负责 GM(ND)→L1(NZ)；
   `Fixpipe` 负责 L0C(NZ)→GM(ND)。
4. **L0C 永远 FP32**（硬件固定），即使 A/B 是 FP16/BF16。
5. **跨核走 GM workspace + 裸 CrossCore API**，不共享 UB。

---

## 2. 数据通路与"少搬多算"原则 ★核心★

### 2.1 三层 buffer 流水

```
GM(ND) ──DataCopy(Nd2NzParams)──▶ L1(NZ) ──LoadData──▶ L0A|L0B(NZ) ──Mmad──▶ L0C(NZ) ──Fixpipe──▶ GM(ND)
        ↑ 大块搬运（行级）           ↑ 小块切（典型 128）                  ↑ ND 落地
```

### 2.2 少搬多算（最常被忽视的性能原则）

**反模式（低效）**：每个 N 维小块都走完整 GM→L1→L0 链路。等于把 GM 带宽当 L1 带宽用，性能折半。

**正模式**：GM→L1 一次搬**大块**（受 L1 容量上限约束取最大），L1→L0 按内层小块**多次切**，
同一块 L1 数据被多个 L0 块复用。

```cpp
// 正模式骨架（去 FA 私有命名）
constexpr uint32_t L1_BIG_ROWS = PickByL1Capacity();  // 在 L1 容量内取最大

for (nL1Start = 0; nL1Start < totalRowsN; nL1Start += L1_BIG_ROWS) {  // 外层：GM→L1 大块
    uint32_t rowsN = MinU32(L1_BIG_ROWS, totalRowsN - nL1Start);
    auto bL1 = rightQueL1_.AllocTensor<DTYPE>();
    LoadRightGmToL1(bL1, ...);                            // ★ 一次搬一大块
    rightQueL1_.EnQue(bL1); bL1 = rightQueL1_.DeQue<DTYPE>();

    for (mL1Start = 0; mL1Start < totalRowsM; mL1Start += L1_M_ROWS) {  // 中层：左矩阵行块
        for (nL0Start = 0; nL0Start < rowsN; nL0Start += BLOCK_N) {     // 内层：L1→L0 切小块
            LoadData + Mmad;                              // 同一 L1 数据被多次复用
        }
    }
    rightQueL1_.FreeTensor(bL1);                          // L1 数据被所有 (mL1, nL0) 消费完才释放
}
```

**L1 大块尺寸选择依据**：
- 受 L1 容量上限约束：`L1_BIG_ROWS × K × sizeof(DTYPE)` ≤ 该 L1 buffer 预算。
- 在容量允许下取最大，让 GM→L1 次数最少。
- 典型量级：dim ≤ 128 时取 256 行；dim > 128 时降到 128 行。**具体值由 L1 预算反推，不要写死**。

### 2.3 左矩阵复用是"少搬多算"的另一种形态

当左矩阵（A）跨外层 N 块复用时：A 一次 GM→L1 加载，所有 N 块的 Mmad 都读同一份 L1 A 数据，
把 GM→L1 摊到多次 Mmad 上。

**实现要点**：A 用 `TBuf<TPosition::A1>`（单 buffer，无队列语义），跨外层循环复用；
每次重新写入 A 前必须 `SetWaitFlag<HardEvent::MTE1_MTE2>()` 等 L1 中前一次 MTE1 读完毕，
才能让 MTE2 重写。

---

## 3. buffer 类型与 bufferNum 决策

**★关键概念区分★**：`TQue<pos, depth>` 模板里的 `depth` 与 `InitBuffer(que, num, len)` 的 `num`（bufferNum）是**两件事**：
- `depth`：队列能容纳的连续 EnQue 次数（中间没有 DeQue）。**推荐永远写 1**（编译器对 depth=1 优化），与 double buffer **无关**。
- `num`（bufferNum）：实际分配的物理 buffer 数，**控制是否启用 double buffer**。

| 场景 | 类型 | InitBuffer num | 为什么 |
|:---|:---|:---|:---|
| 左矩阵跨外层块复用 | `TBuf<A1>` | 1（TBuf 单 buffer） | 单次写入多次读出，TQue 队列语义反而碍事 |
| 右矩阵每次重载 | `TQue<A1, 1>` | 2 | 双 buffer 隐藏 GM→L1 的 MTE2 延迟 |
| 跨阶段复用同队列（如两阶段共用 L1） | `TQue<A1, 1>` | 3 | 跨阶段流水深度大 |
| L0A / L0B / L0C | `TQue<A2/B2/CO1, 1>` | 2 | 让 LoadData/Mmad/Fixpipe 三阶段并行 |

**TBuf vs TQue 的本质差异**：TBuf 无队列语义，跨 pipe 同步必须手动 `SetFlag/WaitFlag`；
TQue 自带 MTE2↔MTE1 同步，AllocTensor/EnQue/DeQue/FreeTensor 自动管。详同目录 `kernel-rules.md` RK-A03。

**InitBuffer 签名差异**（均在 TPipe 上调用）：
- `pipe.InitBuffer(TBuf<pos>& buf, uint32_t byteSize)` —— TBuf 专用，传字节数。
- `pipe.InitBuffer(TQue<pos, depth>& que, uint32_t num, uint32_t byteSize)` —— TQue 专用，传 (bufferNum, 单 buffer 字节数)。

---

## 4. M/N/K 三层循环的两种模式

### 4.1 模式 A：左矩阵复用，K 维内层累加

**结构**：外层 N 行块（GM→L1 大块）→ 中层 M 行块 → 内层 N 列块 → 最内层 K 维分段累加。
**累加层次**：`cmatrixInitVal = (ki == 0)`，每个 (M, N) 块内 ki 循环累加 K 维。
**Mmad 形式**：`Mmad(cL0, aL0, bL0, mp)` —— cL0 由 Mmad 内部持有，三参数版本。
**典型场景**：Q × K^T → S 类（左矩阵 Q 复用，K 维 = dim，内层按 BLOCK_N 切）。

### 4.2 模式 B：左矩阵重载，K 维跨外层块累加

**结构**：外层 M 行块 → 中层 K 块（每块独立加载 A 和 B）→ 内层 K 列块。
**累加层次**：`cmatrixInitVal = isFirstK`，跨 K 外层块累加到**同一个 cL0**（cL0 在 M 外层
循环顶部分配，跨整个 K 外层循环持有）。
**Mmad 形式**：`Mmad(cL0, aL0, bL0, cL0, mp)` —— 四参数版本，显式传 cL0 作累加源。
**典型场景**：P × V → O 类（左矩阵 P 每块重载，K 维 = s2 跨块累加，cL0 跨整个 K 持有）。

### 4.3 两种模式差异速查

| 维度 | 模式 A | 模式 B |
|:---|:---|:---|
| 左矩阵加载 | 外层一次，跨 K 块复用 | 每 K 块重新加载 |
| K 维累加层次 | 内层 ki 循环（K 按 BLOCK_N 切） | 外层 K 跨块 |
| cL0 生命周期 | 一个 (M, N) 块内 | 跨整个 K 外层循环 |
| Mmad 形式 | 三参数 | 四参数 |

### 4.4 cmatrixInitVal × cmatrixSource × Mmad 重载组合矩阵 ★关键★

最容易搞混的三个参数。**cmatrixInitVal** 控制 cL0 是否清零；**cmatrixSource** 控制 cL0 是
Mmad 内部持有还是显式第四参数；**Mmad 三参数 vs 四参数** 是配套的语法形式。

| 场景 | cmatrixInitVal | cmatrixSource | Mmad 形式 | 语义 |
|:---|:---|:---|:---|:---|
| K 内层首块（清零并写入） | `true` | `false` | `Mmad(cL0, a, b, mp)` | cL0 = A×B |
| K 内层续块（累加） | `false` | `false` | `Mmad(cL0, a, b, mp)` | cL0 += A×B（内部 cL0） |
| K 跨外层首块（清零） | `true` | `false` | `Mmad(cL0, a, b, cL0, mp)` | cL0 = A×B（显式 cL0） |
| K 跨外层续块（累加） | `false` | `false` | `Mmad(cL0, a, b, cL0, mp)` | cL0 += A×B（显式 cL0） |

**为什么模式 B 要用四参数**：cL0 跨外层循环持有，必须显式传入让 Mmad 知道累加源；
模式 A 的 cL0 只在 (M, N) 内层持有，Mmad 内部默认即可。

**为什么 cmatrixSource 通常写 false**：当前 AscendC 版本下，四参数 Mmad 已通过显式 cL0 参数
表达累加源，cmatrixSource 字段保持 false 即可；写 true 会改变语义路径，无必要。

**常见错误**：
- 模式 B 误用三参数 Mmad → 每个 K 块的 cL0 互相覆盖，K 维累加失效。
- cmatrixInitVal 永远写 false → cL0 从不清零，首块累加到 L0C 残留脏数据。
- cmatrixInitVal 永远写 true → 每次清零，K 维累加变覆盖。

---

## 5. 关键 API 字段语义（物理含义，非字段列表）

### 5.1 Nd2NzParams（GM ND → L1 NZ）

```cpp
AscendC::Nd2NzParams params;
params.ndNum            = 1;
params.nValue           = m;            // ND 行数
params.dValue           = n;            // ND 列数（要搬运的有效列）
params.srcNdMatrixStride = 0;
params.srcDValue        = ld;           // ★ ND 相邻行起始间隔（元素数）= src 真实行宽
params.dstNzC0Stride    = m;            // ★ 同一 ND 行内多行 NZ 间隔 = m
params.dstNzNStride     = 1;            // ★ NZ 中相邻 Z 行间隔 = 1
params.dstNzMatrixStride = 0;
```

**字段物理含义**（NZ 分形 = 每列 C0 个元素打包）：
- `srcDValue` 是 src ND 视角下行间距。**不同 layout 下同一概念行间距不同**：
  - TND-like（同 token 多 head 连续）：行间距 = `headNum × dim`
  - BNSD-like（同 head 多 token 连续）：行间距 = `dim`
  - 关键是理解"src 的真实行宽是什么"，不要套公式
- `dstNzC0Stride` = m 让 NZ 分形中同一 Z 列的 m 个元素连续存储，是 L1→L0 LoadData 的预期布局。
- `dstNzNStride` = 1 让相邻 Z 列紧挨，整个 NZ 矩阵连续。

**最容易错的字段**：`srcDValue` 传对齐值而非真实行宽 → NZ 布局错位，下游 LoadData 全错。

**dst 行宽 ≠ src 行宽 时**：用重载版本显式传 dst stride。

### 5.2 LoadData3DParamsV2（L1 NZ → L0A/L0B 通用路径）

```cpp
AscendC::LoadData3DParamsV2<T> params;
params.l1H              = 1;
params.l1W              = colC0Stride;  // ★ L1 中按 C0 对齐后的列宽（决定 L1 行间距）
params.channelSize      = k;            // K 维深度
params.kExtension       = k;
params.mExtension       = m;            // M 维
params.strideH = params.strideW = 1;
params.filterH = params.filterW = 1;
params.dilationFilterH = params.dilationFilterW = 1;
AscendC::LoadData(dst, src, params);
```

适用：L1→L0A（ZZ 格式）或 L1→L0B（Zn 格式）的通用 3D 路径。

### 5.3 LoadData2DParams（L1 NZ → L0B 带 2D 转置的快速路径）

```cpp
AscendC::LoadData2DParams p;
p.startIndex    = 0;
p.srcStride     = 1;                    // 单位：512B 分形
p.dstGap        = 0;                    // 单位：512B 分形
p.ifTranspose   = false;
p.repeatTimes   = (k / c0) * (n / c0);  // ★ 总 512B 分形数
AscendC::LoadData(dst, src, p);
```

适用：L1 NZ → L0B Zn 且 colC0Stride == n 的快速路径。当 `colC0Stride != n` 时按 `k / c0`
外层循环，每次 `repeatTimes = n / c0`。

**何时用 2D 而非 3D**：colC0Stride 与 n 相等时 2D 更快（少一层 mExtension/kExtension 解析）。

### 5.4 MmadParams

见 §4.4 组合矩阵。**m/n/k 三轴必须 C0 对齐**（FP16/BF16 C0=16，FP32 C0=8），不对齐必须先在
L1 层把数据凑齐到 C0 对齐，再用 Fixpipe 的 mSize/nSize 截回有效值。

### 5.5 FixpipeParamsV220（L0C NZ → GM ND）

```cpp
AscendC::FixpipeParamsV220 params;
params.nSize        = n;                          // ★ 真实有效列数
params.mSize        = m;                          // ★ 真实有效行数
params.srcStride    = m;                          // ★ 单位 C0_Size（NZ 分形内行间距）
params.dstStride    = (dstStride == 0) ? n : dstStride;  // ★ ND 路径单位是元素数
params.ndNum        = 1;
params.srcNdStride  = 0;
params.dstNdStride  = 0;
AscendC::Fixpipe(dst, src, params);
```

**字段物理含义**：
- `mSize/nSize` 是 ND 输出的有效行列数，**不是对齐值**。传对齐值会多写 padding 区，GM 越界。
- `srcStride` 是 L0C NZ 分形内相邻 ND 行的间距。NZ 分形每列 C0 个元素，所以 srcStride 单位是
  C0_Size（"跳过多少个 C0 元素"）。传 M 维 C0 对齐后的行数。
- `dstStride` 是 GM ND 输出每行的元素跨度。

**最容易错的字段**：`dstStride` 传字节数而非元素数 → GM 写入位置全错。

---

## 6. 同步点

### 6.1 核内（pipe 内）

| 时机 | API | 为什么需要 |
|:---|:---|:---|
| 左矩阵 GM→L1 后、LoadData 读前 | `SetWaitFlag<HardEvent::MTE2_MTE1>()` | TBuf 无队列语义，MTE2 写与 MTE1 读需显式同步 |
| 左矩阵 buffer 复用、前一次 MTE1 读未完 | `SetWaitFlag<HardEvent::MTE1_MTE2>()` | 防 MTE2 重写覆盖未读完的 L1 数据 |
| 每次 Mmad 前 | `PipeBarrier<PIPE_M>()` | K 维连续累加时让前一次 Mmad 写完 cL0 再启动下一次读 cL0 |
| LoadData / Mmad / Fixpipe 之间 | TQue 自动 | AllocTensor/EnQue/DeQue/FreeTensor 自带同步 |

**★ TBuf 正反向同步铁律（缺任意一向导致精度错误或 coredump）**：Cube 侧任何 `TBuf`（L1/L0A/L0B 等）都必须**同时**配置正向和反向两个方向的 `SetWaitFlag`：
- **正向**：producer 写完通知 consumer 可读（`MTE2_MTE1`，写后立刻 Set，读前 Wait）
- **反向**：consumer 读完归还 producer 才可覆写（`MTE1_MTE2`，读完 Set，下次写前 Wait）

两个方向缺任意一个，轻则精度错误，重则 coredump。`TQue` 自动处理两个方向，只要改用 `TBuf` 就必须手动补齐。

**PipeBarrier<PIPE_M> 的精确时机**：只在同一 cL0 被多次 Mmad 累加（ki 循环 / K 跨块）时需要，
让前一次 Mmad 的 cL0 写回完毕再启动下一次 Mmad 读 cL0。单次 Mmad（不累加）不需要。

### 6.2 跨核（裸 API）

只用 `AscendC::CrossCoreSetFlag<0x2, PIPE_XXX>(notifyId)` + `AscendC::CrossCoreWaitFlag<0x2, PIPE_XXX>(notifyId)`。

| 数据流 | 写方 pipe | 读方 pipe |
|:---|:---|:---|
| Cube Fixpipe → Vector DataCopy | `PIPE_FIX` | `PIPE_MTE2` |
| Vector DataCopy → Cube DataCopy | `PIPE_MTE3` | `PIPE_MTE2` |

**PIPE 选择**：写方按"数据真正写完时所在 pipe"（Fixpipe→FIX；DataCopy UB→GM→MTE3）；
读方按"准备读时所在 pipe"（DataCopy GM→UB→MTE2）。notifyId 全算子唯一。详同目录 `kernel-rules.md` RK-A01。

---

## 7. 对齐与容量（极简）

- `C0 = 32 / sizeof(DTYPE)`：FP16/BF16 → 16，FP32 → 8。
- L1 行宽、L0A/L0B 行宽、MmadParams.m/n/k：全部 C0 对齐。
- Fixpipe mSize/nSize：传**有效值**（非对齐）。
- 内层块典型 128（910B/910_93 Cube 最优块，**不是固定值，按 L1/L0 容量与 K 维反推**）。
- 每个 `InitBuffer` 必须有 `// 预算: <公式> = <数值> Byte, 上限 <硬件容量> Byte` 注释。
  详同目录 `kernel-rules.md` RK-B01。

---

## 8. 高频错误（只列非平凡项）

- ❌ **每个内层小块都 GM→L1→L0**：违反"少搬多算"，性能折半。正确做法见 §2.2。
- ❌ 模式 B 误用三参数 `Mmad(cL0, a, b, mp)`：K 维跨块累加失效。必须四参数。
- ❌ `cmatrixInitVal` 恒 true 或恒 false：K 维累加结果错误。
- ❌ `Fixpipe.dstStride` 传字节数：GM 写入位置全错。单位是元素数。
- ❌ `Fixpipe.mSize/nSize` 传对齐值：GM 越界写入 padding 区。
- ❌ `Nd2NzParams.srcDValue` 传对齐值：NZ 布局错位，下游 LoadData 全错。
- ❌ **Cube 侧 TBuf 缺失正向或反向 SetWaitFlag**：必须同时配正向（`MTE2_MTE1`）和反向（`MTE1_MTE2`），缺任意一方向轻则精度错误，重则 coredump。TQue 自动处理，TBuf 不行。详见 §6.1 铁律。
- ❌ TBuf（如左矩阵复用 buffer）写后不 `SetWaitFlag<MTE2_MTE1>`：下游读到不完整数据。
- ❌ L1 buffer depth 选错（左矩阵复用场景用 TQue depth=2）：浪费 L1 或语义错乱。

### 8.1 写前必查补充（高频坑）

- ❌ **`MmadParams.m` 传 1**（如 MQA G=1、singleM=1）：硬件 Mmad.m 最小为 2，m=1 触发 VEC illegal config / 输出错。正确：`mp.m = (mRows==1)?2:mRows`，Fixpipe.mSize 仍取真实 mRows。
- ❌ **`Nd2NzParams.srcDValue`（行距 ld）传 G\*D**：BSND `[B,S,N1=G,D]` 每行 D 元素，行距应 = **D**（不是 N1\*D）。传 G\*D 会读到下一 s1 的同 head（G=1 时 G\*D==D 巧合正确，G>1 全错；G\*D 过大致 Nd2Nz 寻址越界 crash）。`LoadQToL1/LoadDyToL1/ComputeMm5 dO` 等 GM→L1 加载都要核 ld=D。
- ❌ **K 维多 tile 累加 cmatrixInitVal 时序错**：跨 D 分 tile 累加时，首 tile（isFirstK）cmatrixInitVal 置初值、续 tile 累加；跨 inner-step（如 dq 跨 selected-chunk）累加用 AtomicAdd 时，每步 cmatrixInitVal 须为增量（非清零），且目标 workspace 预清零。
- ❌ **ClearL1（MTE1 写 L1）后 ND2NZ/LoadData（MTE2 读 L1）缺 `SetWaitFlag<MTE1_MTE2>` 守卫**：
  清零未完成的 L1 被 ND2NZ 搬走 → pad 区读到脏数据。典型潜伏竞态：fp16/bf16 小 K 下时序
  巧合长期通过，fp32 dqk=192（行字节数翻倍拉长 MTE1 清零窗口）才首次暴露。**正确做法**：
  ClearL1 与下游首个 ND2NZ 之间插 `SetWaitFlag<HardEvent::MTE1_MTE2>`（MTE2 等 MTE1）。
  注意与 §2.3/§8 尾部两条的区别：那些是"MTE2 写 L1 → MTE1 读"方向的守卫，本条是反向。
  来源：SDPA fp32 Cube 路径（2026-08-12），同轮修复 fp16/bf16 同型潜伏 bug
  （正确性代价 bf16/fp16 大 case -4~-7%，应接受）。
