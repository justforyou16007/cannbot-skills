# Vector Kernel 实现 API 指导（生成端参考）

## 1. 核心约束（6 条）

1. **AIV 只跑 vector 原语**：本文件列出的所有 API + `DataCopy(GM↔UB)` + `CrossCoreSetFlag/WaitFlag`
   + 核内 `SetFlag/WaitFlag<HardEvent::MTE2_V|V_MTE3|MTE3_V|MTE2_MTE3>`。
2. **严禁 cube 原语**（Mmad/LoadData/Fixpipe/PipeBarrier<PIPE_M>）→ 编译错误或 coredump。
3. **数据通路**：所有计算在 UB 内。GM→UB(VECIN)→计算→UB(VECOUT)→GM。
4. **跨核走 GM workspace + 裸 CrossCore API**。
5. **每个 subBlock 独立处理自己的行段**（一核 2 个 subBlock，按 `& 1U` 切）。
6. **CV 混合核模式下，所有 vector 代码路径必须用 `if ASCEND_IS_AIV { ... }` 保护**：
   AIC 没有 VECCALC/VECOUT/MTE3 管道，AIC 误入 vector 代码（如调用 `Duplicate`）会 coredump 或卡死。
   包括初始化路径（如清零 output、softmax 默认值预置）也必须先判 AIV。

---

## 2. UB buffer 规划与 depth 决策

### 2.1 通用布局模式

**★关键概念区分★**：`TQue<pos, depth>` 模板里的 `depth` 与 `InitBuffer(que, num, len)` 的 `num`（bufferNum）是**两件事**：
- `depth`：队列能容纳的连续 EnQue 次数。**推荐永远写 1**（编译器对 depth=1 优化），与 double buffer **无关**。
- `num`（bufferNum）：实际分配的物理 buffer 数，**控制是否启用 double buffer**。

```cpp
TQue<TPosition::VECIN,  1> inputQue;        // GM/workspace → UB 输入（depth=1，InitBuffer num=2 启用 double buffer）
TQue<TPosition::VECOUT, 1> outputQue;       // UB → GM/workspace 输出（depth=1，InitBuffer num=1）
TBuf<TPosition::VECCALC> workBuf;           // 临时工作空间（softmax work / Select work / 中间结果）
TBuf<TPosition::VECCALC> initStateBuf;      // 跨块累积状态（如初始 -INF / 0）
TBuf<TPosition::VECCALC> stateRingBuf;      // 跨块传递的状态 ring（多 slot 复用）
```

### 2.2 bufferNum 决策

| buffer | InitBuffer num | 为什么 |
|:---|:---|:---|
| inputQue | 2 | 双缓冲隐藏 GM→UB 的 MTE2 延迟，生产下一块时消费当前块 |
| outputQue | 1 | 输出后立即释放，不需要双缓冲 |
| workBuf / stateBuf | TBuf 单 buffer | 跨阶段复用，TQue 队列语义反而碍事 |

### 2.3 大小决策原则

- **input/output queue 单 buffer 字节数**（即 `InitBuffer(que, num, byteSize)` 的 byteSize）：受 UB 容量约束，典型量级 32KB（**不是固定值，按 tile 元素数 × sizeof(dtype) 反推**）。
- **work buffer**：按 API 要求的最小空间给。SoftmaxFlashV2 / Select 等都有 workLocal 要求。
- **state ring buffer**：单 slot 容纳一个处理块的完整状态，slot 数 = 跨阶段流水深度（在 InitBuffer byteSize 里乘上）。
- 每个 `InitBuffer` 必须有 `// 预算: <公式> = <数值> Byte, 上限 <硬件容量> Byte` 注释。
  详同目录 `kernel-rules.md` RK-B01。

---

## 3. 子核行分担（公式来历）

```cpp
subBlockNum  = GetSubBlockNum();              // 910B 上 = 2
subBlockIdx  = GetSubBlockIdx();              // 0 或 1
subBlockRows = totalRowsM / subBlockNum;      // 按行对半切

// 把 dealRows 切给前 subBlock（向上对齐到 16）
vecDealM = ((dealRows + 15) / 16 + 1) / 2 * 16;
//              ↑ 向上 16 对齐        ↑ 加一份再对半 ↑ 再对齐到 16
```

**三步含义**：
1. `(dealRows + 15) / 16`：把 dealRows 向上对齐到 16（vector 指令 16 元素对齐）。
2. `+ 1) / 2`：对半切，给前 subBlock 一半（向上取整）。
3. `* 16`：再对齐到 16（vector 指令对齐）。

**结果**：偶数 subBlock 拿到 `vecDealM` 行（前半，对齐到 16），奇数 subBlock 拿剩余
`dealRows - vecDealM` 行。当 dealRows ≤ 16 时直接全给偶数 subBlock，奇数不工作（vecDealM=0）。

**适用场景**：一行元素数较少（如 128 行 × small_cols）时，让 2 个 subBlock 并行处理不同行段。
若一行已经占满 UB，subBlock 切分会增加调度开销，不适用。

---

## 4. 实用 API 深度解析

> 通用前置：count 形式 API（`Add(dst, src0, src1, count)`）的 count 应 32B 对齐，非对齐拆 repeat。
> FP32 一个 dataBlock=8 元素，一个 repeat=64 元素；FP16/BF16 一个 dataBlock=16，一个 repeat=128。

### 4.1 Duplicate / Fill — UB / GM 标量填充

**两类填充场景必须分清**：

| 场景 | API | 数据通路 | 适用核 |
|:---|:---|:---|:---|
| UB 内填充（init 状态 / softmax 默认值 / 临时清零） | `AscendC::Duplicate` | 标量广播到 UB | **仅 AIV**（vector 原语） |
| GM 直接填充（清零 workspace / output） | `AscendC::Fill` | DMA 直写 GM | AIC / AIV 均可 |

**Duplicate — UB 填充**：

```cpp
AscendC::Duplicate(dstUb, SOFTMAX_NEG_INF, N);              // 初始化 -INF（softmax 状态）
AscendC::Duplicate(dstUb, 0.0f, N);                          // 初始化 0
AscendC::Duplicate(dstUb, static_cast<DTYPE>(0), total);     // 整段清零
```

**为什么用 Duplicate 而非 Muls(dst, src, 0.0f)**：Duplicate 走标量广播通路，性能比 Muls 高一个量级。
Muls 是"逐元素乘 scalar"，Duplicate 是"直接填充"，后者不需要读 src。

**约束**：scalar 类型必须与 dst 元素类型一致；`uint64_t` 场景下只能填 0。

**Fill — GM 填充（清零 workspace/output 必用）**：

```cpp
// 函数原型：Fill(GlobalTensor<T>& gm, uint64_t size, T value)
AscendC::Fill(outGm_[outOffset], totalElems, static_cast<DTYPE>(0));     // 直接清零 GM 一段
AscendC::Fill(workspaceGm_, workspaceElems, 0.0f);                       // 清零 workspace
```

**★反模式（禁止）★**：清零 GM 用 "`Duplicate` UB 中转 + `DataCopy` 写回 GM" 循环。
- 浪费带宽（GM→UB→GM 多绕一圈）。
- Duplicate 是 vector 原语，CV 模式下 AIC 跑会 coredump。
- 多次小循环 `DataCopy` 利用率低，抢 AIC/AIV 流水。
- **必须用 `Fill` 一次性直接清零 GM**。

**Fill 的硬约束**：
- **必须在所有 `InitBuffer` 之前调用**（该接口与 buffer 初始化存在时序冲突）。意味着清零 GM 应放在 **Kernel Init 阶段**（先 Fill 再 InitBuffer），而不是 Process 阶段。
- 后续若涉及 UB 使用（如 `DataCopy` 读这段 GM 到 UB），需设置 `MTE3_MTE2` 同步：MTE2 等 MTE3 完成。
- 多核同时清零不同 GM 区段时，所有核未必同时完成，跨核依赖该数据的场景需补 `SyncAll()`。
- 模板参数 T 必须与 GM 元素类型一致；支持的类型：uint16_t/int16_t/half/uint32_t/int32_t/float。

**★多核分段清零 + SyncAll 标准模式★**：清零大段 GM（workspace / output 兜底）时，单核清零浪费带宽，
应按 vector 核数分段并行清零，然后 SyncAll 等所有核完成：

```cpp
// 时机：Kernel Init 阶段、所有 InitBuffer 之前，仅AIV参与
if ASCEND_IS_AIV {
    uint32_t aivCoreIdx = GetBlockIdx() / GetSubBlockNum();   // CV 模式下 AIV coreIdx
    uint32_t perCore = CeilDiv(totalElems, usedAivCoreNum);   // 每核负责的元素数
    uint64_t start  = static_cast<uint64_t>(aivCoreIdx) * perCore;
    uint64_t size   = MinU64(perCore, totalElems - start);    // 尾核钳位
    AscendC::Fill(outGm_[start], size, static_cast<DTYPE>(0));
}
// SyncAll 保证所有核清零完成
AscendC::SyncAll();

// 之后所有 InitBuffer 调用 ...
pipe.InitBuffer(...);

```

**反模式（禁止）**：
- 单核清零整段 GM，其他核闲置 —— 带宽利用率低。
- 在 Process 阶段调 `Fill` —— 违反 "InitBuffer 之前调用" 约束，行为未定义。若必须在 Process
  阶段清零，只能用 `Duplicate + DataCopy`（但需要 UB 中转、AIC 上不能跑 Duplicate，不推荐）。

### 4.2 Brcb — 行级标量广播的关键原语 ★核心★

**Brcb 是实现"每行乘/除以不同标量"链路的核心**。完整链路：

```
行级标量状态: state[M, 8]    ← 每行 8 个相同值（按向量对齐存储）
        │ Brcb（每 8 元素广播成 1 个 dataBlock）
        ▼
tmp:        tmp[M, 128]      ← 每行 128 个相同值（FP32 一个 blk=8 元素，128/8=16 blk）
        │ Mul repeat（src1=tmp，每行广播）
        ▼
dst:        dst[M, 128]      ← 每行乘以不同标量
```

```cpp
// 把 [dealRows, 8] 的行级状态展开成 [dealRows, 128]
LocalTensor<float> tmpExpBrcbUb = workBuf.Get<float>();
LocalTensor<float> expStateUb   = stateRingUb[slotOffset];   // [dealRows, 8]
AscendC::Brcb(tmpExpBrcbUb, expStateUb,
              (dealRows + BRCB_NUM - 1) / BRCB_NUM,           // repeatTimes = ceil(dealRows/8)
              AscendC::BrcbRepeatParams(1, BRCB_NUM));        // {srcStride=1, dstStride=8}
// 之后 Mul repeat 把 tmpExpBrcbUb 广播乘到目标
```

**约束**：
- src 每 8（FP32）/ 16（FP16/BF16）元素**必须完全相同**，否则结果是错乱的 8 个不同值被复制 16 次。
- src 起始偏移按 8/16 对齐。
- BRCB_NUM = `32 / sizeof(float)` = 8（FP32）。

**为什么必须 Brcb 展开**：Mul/Div 的 repeat 形式要求 src1 按 dataBlock 对齐（每 blk 内同值才能广播）。
softmax 类算子输出的 `[M, 8]` 格式正好是 Brcb 的输入格式（每行 8 个相同值），是 softmax + 行级缩放
链路的天然契合点。

### 4.3 Select — 按 byte mask 三元选择

```cpp
AscendC::SelectWithBytesMaskShapeInfo selectInfo;
selectInfo.firstAxis    = rows;          // 行数
selectInfo.srcLastAxis  = cols;          // 源列数
selectInfo.maskLastAxis = cols;          // mask 列数

AscendC::Select(dstUb, srcUb, NEG_INF, maskUb, workBuf, selectInfo);
```

**语义**：`dst = mask ? src : scalar`。mask 是 uint8（每元素 1 byte），非 0 视为 true。

**约束**：
- `maskLastAxis` 必须 32B 对齐（mask 按 32 个 byte 对齐）。
- `workLocal` ≥ `srcLastAxis * firstAxis`，作中间空间。

### 4.4 Exp / Cast

```cpp
AscendC::Exp(dstUb, srcUb, count);                                  // 自然指数
AscendC::Cast(dstHalf, srcFloat, RoundMode::CAST_ROUND, count);     // FP32 → FP16/BF16
AscendC::Cast(dstHalf[rowOff], srcFloat[rowOffAligned],
              RoundMode::CAST_ROUND, actualCols);                   // 逐行 cast（非对齐 dim）
```

**逐行 Cast 的场景**：当输出 dim 非 C0 对齐（如 dim=96，dimAlign=128），UB 内矩阵按 dimAlign=128
存储（每行尾有 32 元素 padding），但输出 GM 只接受 actualCols=96 有效元素。这时不能整段 cast
（会把 padding 也 cast 进去），必须**逐行 cast 只取 actualCols 个元素**。

**约束**：Cast 的 RoundMode 不可省略，`CAST_ROUND` 最常用（其他有 FLOOR/CEIL/CAST_TRUNC，按精度需求选）。

### 4.5 Muls / Adds（标量运算）

```cpp
AscendC::Muls(dstUb, srcUb, scale, count);                   // 整体乘标量
AscendC::Adds(dstUb, srcUb, static_cast<float>(1e-30f), n);  // 防 0 除
```

**Muls 外置 scale vs SoftmaxFlashV2 IS_SCALE=true 的二选一**：
- SoftmaxFlashV2 模板参数 `IS_SCALE=true` 会在 softmax 内部自动乘 scale。
- 选择**外置 Muls**时：scale 与 softmax 解耦更清晰；scale 和 softmax 的 oriSrcK 是两个独立维度。
- **不能既外置 Muls 又开 IS_SCALE=true**：会乘两次 scale。

**Adds 防 0 除的标准模式**：`Adds(sum, sum, 1e-30f)` 让 sum 永不为 0，后续 Div 安全。

### 4.6 Add / Sub / Mul / Div / Max（双目）

```cpp
AscendC::Add(dst, src0, src1, count);                  // 逐元素加
AscendC::Sub(dst, src0, src1, count);                  // 逐元素减
AscendC::Mul(dst, src0, src1, count);                  // 逐元素乘
AscendC::Div(dst, src0, src1, count);                  // 逐元素除
AscendC::Max(dst, src0, src1, count);                  // 逐元素取最大（规约用）
```

规约类算子（如 flash decode）的核心算子组合：`Sub → Exp → Mul → Add → Div → Max` 实现
`beta_i = exp(max_i - max_global)`、`sum_global = Σ sum_i × beta_i`、`weight_i = sum_i × beta_i / sum_global`。

### 4.7 Mul / Div + BinaryRepeatParams（行级标量广播） ★核心★

"每行乘/除以不同标量"的底层实现是 Mul/Div 的 repeat 形式 + BinaryRepeatParams，与 §4.2 Brcb 配合。

```cpp
AscendC::BinaryRepeatParams repeatParams;
repeatParams.src0BlkStride = 1;                                // src0 blk 内连续
repeatParams.src1BlkStride = 0;                                // ★ src1 blk 内不移动 → 行内广播
repeatParams.dstBlkStride = 1;
repeatParams.src0RepStride = columnCount / blockElementNum;    // ★ src0 跨行步进
repeatParams.src1RepStride = 1;                                // ★ src1 跨 rep（每行不同标量）
repeatParams.dstRepStride = columnCount / blockElementNum;

AscendC::Mul(dstUb[offset], src0Ub[offset], src1Ub,
             repeatElementNum /*mask*/, dealRowCount /*repeatTimes*/, repeatParams);
```

**字段语义对照**：
- `src0BlkStride=1, src1BlkStride=0`：src0 blk 内连续读 N 个，src1 blk 内只读 1 个广播到 N 个。
- `src0RepStride=columnCount/blockElementNum`：src0 跨行（一整行的 blk 数）。
- `src1RepStride=1`：src1 每 rep（即每行）取一个新标量。

**两种模式选择**：
- **不展开模式**：src1 直接是 `[M, 8]` 状态，src1BlkStride=0、src1RepStride=1，让 Mul 在 blk/rep 维度自动广播。适用于 columnCount < `REPEATE_STRIDE_UP_BOUND × blockElementNum`（FP32 下 ≈ 2048）。
- **Brcb 展开模式**：先 Brcb 把 `[M,8]` 展开到 `[M,128]`，再 Mul repeat。src1RepStride=columnCount/blockElementNum
  （每行同值，已展开），src1BlkStride=1。适用于 columnCount 超限（src1RepStride 上限 255）的场景。

### 4.8 DataCopy（GM↔UB）

```cpp
AscendC::DataCopy(dstUb, srcGm[offset], count);                // workspace GM → UB
AscendC::DataCopy(dstGm[offset], srcUb, count);                // UB → workspace GM
```

带 stride 的多行拷贝（按行带 padding 的矩阵）：

```cpp
AscendC::DataCopyParams copyParams;
copyParams.blockCount = rows;
copyParams.blockLen   = cols / 32;                          // ★ 单位 block(32B)
copyParams.srcStride  = (srcRowWidth - cols) / 32;          // ★ 前块尾到后块头的间隔
copyParams.dstStride  = (dstRowWidth - cols) / 32;
AscendC::DataCopy(dstUb, srcGm[offset], copyParams);
```

**关键约束**：`srcStride/dstStride` 语义是"前一块尾到后一块头的间隔"，**不是行首跨度**。
GM 侧单位字节，UB 侧单位 dataBlock(32B)。非对齐搬运走 `DataCopyPad`（详 `ascendc-datacopy-guide`）。

---

## 5. 在线 softmax 三态跨块传递（attention 类算子专用） ★核心★

**本节仅适用于在线 softmax（attention 类）算子**。其他算子（如纯规约/逐元素）忽略本节。

在线 softmax 的核心是 max/sum/exp **三态跨处理块传递**：当前块的 softmax 需要基于"前所有块的
全局 max/sum"做偏移修正，避免溢出和精度损失。

### 5.1 为什么需要三态

朴素 softmax 在一整行上做 `exp(x - max) / sum(exp(x - max))`，需要先 reduce 整行 max 和 sum。
在线 softmax 把一行切成多块处理，每块算出局部 max/sum 后，需要与前块的全局 max/sum 合并：

- 新全局 max = `max(prevGlobalMax, curLocalMax)`
- 修正系数 = `exp(prevGlobalMax - newGlobalMax)`（前块的 exp 和 sum 都要乘这个）
- 新全局 sum = `prevGlobalSum × 修正系数 + curLocalSum × exp(curLocalMax - newGlobalMax)`

这就是 SoftmaxFlashV2 内部做的事。

### 5.2 多 slot ring buffer 设计

跨块传递需要保存"上一块的状态"供当前块读。设计要点：

```cpp
// 状态按 [slot, stateStride] 排布，stateStride 容纳一行处理块的完整状态
constexpr uint32_t STATE_SLOTS = PickByPipelineDepth();   // 跨阶段流水深度（典型 2~3）
constexpr uint32_t STATE_PER_SLOT = stateRows * FP32_BLOCK_ELEMENT_NUM;  // 每行 8 个 float
pipe.InitBuffer(maxStateBuf, STATE_SLOTS * STATE_PER_SLOT * sizeof(float));
pipe.InitBuffer(sumStateBuf, STATE_SLOTS * STATE_PER_SLOT * sizeof(float));
pipe.InitBuffer(expStateBuf, STATE_SLOTS * STATE_PER_SLOT * sizeof(float));
```

**slot 数选择**：等于同时活跃的跨阶段数。典型 3（Vec1 阶段读 chunk N-1 状态、Vec2 阶段读 chunk N-1
的 sum、chunk N 自己写入），depth=2 会让 chunk N 写入覆盖 chunk N-1 还没读完的状态。

**首轮初始化**：第一个处理块没有"前块"，必须用 Duplicate 初始化 inMax=-INF、inSum=0。
否则 softmax 偏移错乱。

### 5.3 SoftmaxFlashV2 模板参数选择

```cpp
AscendC::SoftmaxFlashV2<
    float,                          // dstDtype（FP32，softmax 必须）
    IS_OUTPUT_LARGE,                // softmax 输出 UB 是否容纳整个 src
    IS_SCALE,                       // 是否内部乘 scale（与外置 Muls 二选一）
    SAVE_EXP,                       // 是否单独输出 exp（与 Vec2 阶段 O *= exp(state) 配合）
    FORMAT,
    SoftmaxConfig                  // 输出格式（是否预 Brcb 展开）
>(dst, sum, max, src, exp, inSum, inMax, workBuf, smTiling, srcShape);
```

**IS_OUTPUT_LARGE**：true 走完整输出路径（UB 够大）；false 走分块输出（UB 装不下整个 src）。
判断依据：`dealRows × outerCols × sizeof(float)` 是否 ≤ workBuf 预算。

**SAVE_EXP**：即使设 false，SoftmaxFlashV2 内部也会算 exp；SAVE_EXP 只控制是否单独输出。
若后续阶段（如 `O *= exp(state)`）需要 exp，设 true 并接收 exp tensor。

**SoftmaxConfig**：`SOFTMAX_OUTPUT_WITHOUT_BRC` 表示输出不带 BRC（每行 max/sum 各 8 元素，
不预展开成 128），由调用方按需 Brcb（详 §4.2）。

---

## 6. Cast 路径决策

### 6.1 softmax 类算子标准 Cast 路径

```
GM(FP16/BF16) ─DataCopy─▶ UB(FP16/BF16) ─Cast▶ UB(FP32) ─softmax/Mul/Add─▶ UB(FP32)
                                                                          │
                                                                          ▼
                                              GM(FP16/BF16) ◀─Cast─ UB(FP32)
```

**为什么 softmax 走 FP32 而非 FP16**：softmax 涉及 Exp（指数增长）和 ReduceSum（大量累加），
FP16 范围（±65504）和精度（3 位有效数字）不够，会溢出或舍入误差累积。FP32 是 softmax 的标准精度域。

### 6.2 输出 Cast 的两种形态

```cpp
// 形态 1：dimAlign == actualDim，整段 Cast
AscendC::Cast(outHalf, oUb, RoundMode::CAST_ROUND, dealRows * dim);

// 形态 2：dimAlign != actualDim，逐行 Cast（去掉 padding）
for (uint32_t i = 0; i < dealRows; i++) {
    AscendC::Cast(outHalf[i * actualDim], oUb[i * dimAlign], RoundMode::CAST_ROUND, actualDim);
}
```

**为什么 dimAlign ≠ actualDim 要逐行**：UB 中矩阵按 dimAlign=C0 对齐存储（每行尾有 padding），
但 GM 输出只接受 actualDim 有效元素。整段 Cast 会把 padding 也写入 GM，导致行错位。

---

## 7. 跨核同步 + 高频错误

### 7.1 跨核

只用 `AscendC::CrossCoreWaitFlag<0x2, PIPE_XXX>(notifyId)`（读前等）+
`AscendC::CrossCoreSetFlag<0x2, PIPE_XXX>(notifyId)`（写完通知）。

```cpp
// Vector 读 Cube 写的 workspace 前：PIPE_MTE2 等
AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(notifyId);
auto sSlot = wsSGm_[slotOffset];

// Vector 写回 workspace 后：PIPE_MTE3 通知
AscendC::DataCopy(wsPGm_[slotOffset], pHalf, tileSize);
AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(notifyId);
```

**PIPE 选择**：读方按"准备读时所在 pipe"（GM→UB→MTE2）；写方按"数据真正写完时所在 pipe"
（UB→GM→MTE3）。notifyId 全算子唯一。详同目录 `kernel-rules.md` RK-A01。

### 7.2 高频错误（只列非平凡项）

- ❌ **清零 GM 用 Duplicate+DataCopy 循环**而非 `Fill`：浪费带宽、CV 模式下 AIC 跑 Duplicate 会 coredump。详 §4.1。
- ❌ **CV 模式下 vector 代码路径未用 `if ASCEND_IS_AIV` 保护**：AIC 误入会卡死。详 §1 约束 6。
- ❌ **Fill 在 InitBuffer 之后调用**：违反时序约束，行为未定义。详 §4.1。
- ❌ **Brcb src 每 8 元素不相同**却期望标量广播：结果是 8 个错乱值被复制。
- ❌ **SoftmaxFlashV2 首轮没初始化 inMax=-INF, inSum=0**：softmax 偏移错乱。
- ❌ **状态 ring buffer slot 数不足**导致当前块写入覆盖前块还没读完的状态：读到错乱 sum。
- ❌ **Muls 外置 + SoftmaxFlashV2 IS_SCALE=true 同时开**：scale 乘两次。
- ❌ **Compare 的 dst 当 float 读**：实际是 uint8 mask（每元素 1 byte）。
- ❌ **Cast 省略 RoundMode**：编译错误。
- ❌ **dimAlign≠actualDim 时整段 Cast**：GM 行错位（padding 被写入）。
- ❌ **BinaryRepeatParams src1RepStride 写 0**：所有行用同一标量，与"每行不同"相反。
- ❌ **DataCopy srcStride/dstStride 当行首跨度**：实际是"前块尾到后块头间隔"，写错导致行错位。
- ❌ **跨核 CrossCoreSetFlag 漏调或者多调**：对端 CrossCoreWaitFlag 死锁。

### 8.1 写前必查补充（高频坑）

- ❌ **单 TBuf `GetWithOffset` tensor 未 `SetShapeInfo`+`SetSize`**：默认 1D flat shape。`SoftmaxGradFront/SoftmaxGrad` 在 softmaxShapeInfo 为空时读 tensor 自身 shape（flat→srcNDinfo 错）→ retile 用 helpTensor 大小按错 shape 重算 → V 指令按错 shape 越界 UB（errcode `VEC ub address out of bounds`）。参照 arch22 InitUB 对 softmax 相关 tensor 显式 SetShapeInfo(2D ND)。
- ❌ **`SoftmaxGrad` 传 sgShape 与 host tiling srcK 不一致**：host `SoftMaxGradTilingFunc` shape 用 `{singleM, 8}`（srcK=8），kernel 若传 `sgShape{dealRows, D2}`（srcK=D2）→ VEC illegal config。改用 `SoftmaxGradFront`（不传 shape）或 sgShape 与 tiling 同口径。
- ❌ **单 TBuf 分时复用 buffer 的"后写未覆盖前读残留"**：dy/O 共用同一 UB 区时，下一 step O 的 DataCopyPad 未覆盖上一 step dy 残留 → Cast 读到 dy → rowsum/softmax 错（task0 对、task1+ 放大）。分时复用 buffer 切换时须确保新写完整覆盖（或独立 buffer）。
- ❌ **AtomicAdd 目标 workspace 未预清零 / 清零对写方核不可见**：`SetAtomicAdd` 是读-改-写，依赖 GM 初始=0。`SyncAll<false>` 是同类核到达 barrier，**不 drain pipeline、不保证跨核 GM 可见**。须 FillWorkspaceZero + CrossCore flag/PipeBarrier 保证写方核读到清零后的 GM。
- ❌ **CV 1:2 下两 AIV subBlock 都处理全范围**：Scatter/累加被两 subBlock 各做一遍 → 结果 2x。须按 subBlockIdx 切分行/块范围。
- ❌ **softmaxMax/sum 读取 layout 错**：GM 紧凑 `(B,N,S,G)`，若当 `[dealRows,8]` Brcb 块读 → 行 i 用 G[i\*8] 而非 G[i]（行0 巧合对、行1+ 错）。须 DataCopyPad 紧凑读 + Brcb 行级广播。
- ❌ **DataCopy(GM↔UB) batch/size 非 8/32B 对齐**：VEC illegal config / CCU address check crash。须 size AlignUp-8（或 32B）多搬，GM 输出段用 DataCopyPad 精确尾段（不可越界写用户输出 tensor）。
- ❌ **AIV sub-block 写 workspace 目标地址缺 rowStart_ 偏移**：fp32 hi/lo 分片路径下，两 sub-block
  均向 `workspace[0]` 写入 SplitTileToWs/DataCopy，sub-block 1 覆盖 sub-block 0 的数据 → AIC
  读到错乱行 → fp32 多 tile 精度灾难（MRE≈1.0, cos≈0.2）。**正确做法**：目标地址必须包含
  `(uint64_t)rowStart_ * dim` 偏移（如 `kvSlot[(uint64_t)rowStart_ * dimQK]`），确保 sub-block 0
  写 rows 0-31、sub-block 1 写 rows 32-63。详见 `kernel-rules.md` RK-D01。
- ❌ **fp16/bf16 的 P（softmax 输出）单次 Cast 后直接 MMA**：P cast 到 fp16/bf16 有固有精度边界
  （尾数截断），单 tile 碰巧通过，多 tile 在线累加误差增长 → 大 Skv case matched_ratio≈0.88~0.90、
  MERE 超阈值，误差分布呈全面 1 ULP（非局部错行）。**正确做法（P hi/lo 拆分双 MMA）**：
  P_fp32 先 Cast 到 P_hi（fp16/bf16），再 `P_lo = Cast(P_fp32 - Cast_back(P_hi))`，
  PV 用两次 Mmad 累加（`O += P_hi·V; O += P_lo·V`），等价于 P 以 ~2 倍尾数位宽参与 MMA，
  matched_ratio 0.899→1.0、MERE 1.1e-2→1.6e-5。三个配套约束缺一不可：
  1. **启用条件统一为 `sizeof(QType)==2`**（fp16 和 bf16 都需要，不要只给 bf16 —— 否则 fp16
     大 Skv case 全量验证回归）；buffer 分配（如 pBufL1b_）与 Get 的条件必须同口径，
     防止"只给 bf16 分配但 Get 无条件执行"的 fp16 报错。
  2. **Cast 写 P_lo 后、DataCopy 前必须 `SetWaitFlag<V_MTE3>` 同步**（命中流水线同步陷阱表
     第一项）：缺同步则 P_lo 读到未完成 Cast 的结果 → 双 MMA 后全错（err 1~2），且单看
     双 MMA 通路隔离实验（第二次 Mmad 改用 P_hi，ratio=2.0）是正确的，极易误判方向。
  3. fp32 路径无此问题（无 Cast 截断），保持单 MMA。
  来源：ScaledDotProductAttention case[7] bf16 [2,8,577,128→136] 及全量 case[17]/[44] fp16。
- 💡 **fp32 matmul 的正确前提与已验证路径**：注意区分两层限制——**TileLang 框架**的
  mma_macro_generator 仅支持 16/8bit 输入（probe 实证），故 TileLang 阶段 fp32 只能表达为
  Vector；但 **910B2C（Atlas A2）Cube 硬件支持 float×float→float Mmad**（Mmad.md 表7），
  吞吐为 fp16 的 1/4（cube_k=4，见 cube_instruction_theoretical_perf_summary.md 表1），
  开 HF32 模式为 1/2（尾数截断、精度有损）。⚠️ 不要写成"平台不支持 fp32 MMA"（SDPA 任务
  kernel 注释曾误写）。
  **路径 ① fp32 直接 Cube Mmad 已实证**（SDPA 2026-08-12，`SdpaKernel<float>` 复用 fp16/bf16
  CV 融合结构，fp32 全程无 Cast 无 hi/lo）：case [2,15,239,192]×Skv=275 从 2307us→185us
  （12.5x，0.042x→0.522x），MERE 2.6e-7，fp32 谱 geomean 0.32→0.78。实现要点：
  BASE_K_T=64（双缓冲 L0A/L0B 适配 64KB）、C0=8 K 对齐（恒满足 kDirectionAlign=false 的
  ceil(K/8)×8）；**小 shape 按 qkElems≥2048 分流保留 Vector**——Cube 的 host 固定开销
  （workspace at::zeros、tiling H2D、MIX launch）在小 shape 使端到端反转（device kernel time
  持平 ≠ 端到端持平，阈值须 A/B 实测）。备选：② Q/K/V hi/lo bf16 3-MMA（~1/3 fp16，
  P hi/lo 同思路已验证）；③ HF32（1/2 fp16，需实测能否过 fp32 atol=1e-3）。
- ✅ **逐行 GetValue+标量 op 循环的广播化（V1 已验证模式）**：online softmax 的
  `P = S - runMax` 逐行 `GetValue+Adds(-m)` → `Brcb(runMax) + BinaryRepeat Sub`
  （x-m ≡ x+(-m) 位级一致）；tail mask 逐行 Add → 单条广播 Add（BinaryRepeatParams
  src1RepStride=0，mask 向量自动每行复用）。实测消除每 tile 最大一处标量循环（32 次
  GetValue + 32 条 Adds），fp16 大 case 单 case -73us。Brcb 官方语义注意：源操作数恒连续读
  （repeat i 读 [8i,8i+8)）、参数为 {dstBlkStride, dstRepStride}、不支持 src/dst 同地址、
  位于 PIPE_V（详 Brcb.md 与 test_operator_vbrcb.cpp）。
- ❌ **GM→UB meta 拷贝（MTE2）的守卫类型必须匹配消费者 pipe 而非数据本身**：online softmax
  的 alpha/sum meta 由 DataCopy（MTE2）载入后，若消费者是标量管 `GetValue`，守卫用
  `SetWaitFlag<MTE2_S>`；一旦把下游改为 V 管的 Brcb/Mul/Div（广播化改造），守卫必须同步改
  `MTE2_V`——否则 V 管读到未完成搬运的 meta → 错误 alpha rescale。阴险点：fp32 路径时序
  余量大可巧合通过，fp16/bf16 kvLoops≥2 才暴露（V 侧紧凑）。⚠️ 附带教训：SDPA 第二轮中
  V2 的 RowMuls/RowDivs 广播化即使修正了该守卫仍引入未解析的精度破坏（疑似事件池/调度
  二阶效应），两轮未收敛按"精度回归当场回退"纪律回退标量实现——**V 侧广播化改造应按
  V1/V2 分开落地、分别验证，便于二分定位**（bisect 证实 V1 两处广播化正确有效）。
- ❌→✅ **V2 overlap 竞态族（Heisenbug family）根因已闭合：finalize 的 Cast→DataCopy UB
  交接仅靠 `PipeBarrier<PIPE_V>` 不够，必须显式 `SetWaitFlag<V_MTE3>`**（SDPA R6 专项，
  2026-08-18）。**根因机制**：fp16/bf16 `FinalizeOutputChunk` 的
  `Cast(outHalf) → PipeBarrier<PIPE_V> → DataCopy(Pad)(outGm)` 交接仅依赖 PipeBarrier
  保证 V→MTE3 序；当 V 管被事件等待 stall（**任何 V 管时序变动都可触发**——R3 广播化、
  R5 软流水踩中的正是同一漏网点），标量发射管前跑把紧随的 MTE3 拷贝一并入队，
  PipeBarrier 拦不住 → MTE3 把 outHalf 中**上一 chunk 的 Cast 残留**拷出。**修复 E1**：
  fp16/bf16 finalize 两分支各补一条显式 `SetWaitFlag<V_MTE3>`（fp32 路径本有此守卫，
  这正解释了 dtype 偏斜：fp32 全过、2B 必中）。**失败四特征**（同族判定）：①多次运行
  逐位不同；②失败行 = 前一 chunk 原值逐位（陈旧源 = 本地 Cast buffer）；③
  `PipeBarrier<PIPE_ALL>` 全排空不敏感；④集中 2B dtype + 多 kvLoop。**判别链（可复用）**：
  列级取证（每列非 ref 即 out[r-8] 逐位，陈旧窗口 = 128B Cast 迭代粒度）→ dtype 按守卫
  类型分组 → late-issue 探针（消重叠保逻辑）逐位一致证重叠本身是竞态源 → 补守卫实验
  20/20 全清证实。**事件 ID 方向正式证伪**（Set/Wait 账本数学闭合 + E1 零事件层改动即
  修复）——R3/R5 的"TQue 双缓冲交接 × 动态事件 ID"疑似方向及三类事件层修复（专用
  parity TBuf + 每 parity 单持久 MTE2_V 事件、瞬态 V_MTE2 WAR 守卫、撤出 inputQue1_）
  全部打在错误层面，勿再投入。E1 后 C1 软流水正确性已证（50/50×2 + case35×100 零失败，
  idx=49 -12%），交付禁令解除；但 overlap 交付见下条教训。
- 💡 **overlap 类优化的收益/开销 case 分化可能无静态谓词可分**：C1（V2 chunk 预取软流水）
  E1 后正确性闭合，主目标 idx=49 -12%、43/46/40 各 -12%、fp16 锚点提升，但小 fp32 case
  一致开销 +6~40%（Sq 小、任务数欠载、kvLoops≤2 短流水时，每 tile 前奏 + 每 chunk 事件
  开销无 overlap 可摊），且 win/loss 与 shape 无干净对应（同构 case 16 胜 / 19 负）→
  无法按 shape 谓词条件化，整体按纪律回退储备（.round_backup/r6_c1e1/）。兑现路径：
  运行时条件化（如按 kvLoops×行数工作量阈值启用预取）或削减 per-chunk 事件开销（更深
  流水摊薄）。**纪律：小 case 密集的任务集上 overlap 类改动必须逐 case 全量 A/B，并预设
  "无干净谓词则回退"**。
- ✅ **b 小 c 大的单核 matvec 瓶颈 → 拆 kernel 多核化 + host 谓词分流（SPSA K23 模式，
  2026-08-20，1 轮达标）**：fused kernel `grid=b` 在 b≤2 且 c=2048 时仅 1-2 个 AIV pair
  串行执行 c·c2 级 matvec×N（案例：~8M MACs + ~32MB 权重流量，段耗时 ~1400us）。
  **做法**：将 fused kernel 按数据依赖拆为 N 个小 kernel（launch 边界即全局 barrier），
  matvec 按输出 64-row/col tile 切分，`grid = b×split`（`split=min(aivCoreNum/b,
  tiles/2)`，vid 作独立 worker，BalancedRange 均衡划分）。**关键纪律**：
  1. **host 谓词分流**（案例：`useSplit=(c≥1024 && b≤4)`，命中 8/50 case）——小 case
     维持原 fused 少-launch 路径，零回归风险；拆分路径只在大 c 小 b 时启用；
  2. **精度保障三件套**：每 tile/每行 op 顺序与 fused 逐位一致 + 跨 kernel 中间数据走
     GM fp32 无损 + 原有 RoundTrip（模拟参考舍入）位置保留 → 拆分后 fp32 matched_ratio
     仍 1.000000、50/50 精度保持；
  3. 收益判定：段耗时 ~1400us→~100us，远超 6 次额外 launch + workspace 开销；
     case7 0.332→1.441、case8 0.425→2.036、geomean 1.1985→1.6756。
  来源：61_SequentialPolarizedSelfAttention（output2/perf_tuning/optimization_log.md）。
- 💡 **DataCopyPad 尾块正确模式**（TileLang tl_templates/ascend/common.h copy_gm_to_ub
  实证）：尾块先 `Duplicate` 全 tile 填 0，cp 参数 `{validRows,
  validCols*sizeof(T), (srcRowLen-validCols)*sizeof(T),
  (dstN-validCols)*sizeof(T)/32, 0}`（该 dstStride 公式经代数验证恒正确），
  padParams(isPad=(validCols*sizeof%32!=0), 0, isPad?1:0, 0)。
- 💡 **ReduceSum 方向与 dtype 限制**：列归约用 `Pattern::Reduce::RA`、行归约用 `AR`；
  A2 上仅支持 fp32。标量入 GM：`LocalTensor.GetValue` + `Duplicate` 成行再 DataCopyPad
  （直写慢但可用；与 TileLang 侧"标量 GM 写不生效"不同）。
