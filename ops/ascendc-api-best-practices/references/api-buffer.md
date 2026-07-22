# UB 缓冲区管理指南

TBuf/TQue 选择、Double Buffer 流水线并行、批量搬运模式。

---

## 目录

1. [TBuf vs TQue 选择](#tbuf-vs-tque-选择)
2. [TQue 详解](#tque-详解)
3. [TBuf 详解](#tbuf-详解)
4. [Double Buffer 流水线并行](#double-buffer-流水线并行)
5. [双缓冲决策矩阵（何时开 / 何时不开）](#双缓冲决策矩阵何时开--何时不开)
6. [MIX (CV 融合) 模式下双缓冲与 SyncAll 的交互](#mix-cv-融合-模式下双缓冲与-syncall-的交互)
7. [批量搬运 + 逐行计算模式](#批量搬运--逐行计算模式)

---

## TBuf vs TQue 选择

| 场景 | 推荐类型 | 说明 |
|-----|---------|------|
| MTE2/MTE3 搬运缓冲区 | `TQue<VECIN/VECOUT>` | 需要与 Vector 并行，需要 EnQue/DeQue |
| 纯 Vector 计算缓冲区 | `TBuf<VECCALC>` | 不涉及 MTE 搬运，用 `Get<T>()` 获取 |
| Double Buffer | `TQue` + `InitBuffer(que, 2, size)` | 在 InitBuffer 中设置 num=2 开启 |

---

## TQue 详解

### 模板参数

```cpp
template <TPosition pos, int32_t depth, auto mask = 0> class TQue;
```

| 参数 | 说明 |
|------|------|
| `pos` | 队列逻辑位置：`VECIN`, `VECOUT`, `A1`, `A2`, `B1`, `B2`, `CO1`, `CO2` |
| `depth` | 队列深度，表示可连续 EnQue/DeQue 的次数 |
| `mask` | 数据格式转换（ND↔NZ）或编译期优化参数 |

### depth 参数关键说明

| depth 值 | 适用场景 | 说明 |
|---------|---------|------|
| `depth=1` | **默认推荐**，非 Tensor 原地操作 | 编译器有特殊优化，性能更好 |
| `depth=0` | **Tensor 原地操作** | 需要设置 |
| `depth=2` | 连续 2 次 EnQue 场景 | 与 InitBuffer 的 num 参数独立 |

**注意**：`depth` 与 Double Buffer 无关。Double Buffer 由 `InitBuffer` 的 `num` 参数控制。

```cpp
// ✅ 非连续入队（普通场景）：depth=1 即可
AscendC::TQue<AscendC::TPosition::VECIN, 1> que;
pipe->InitBuffer(que, 1, size);
auto tensor = que.AllocTensor<T>();
que.EnQue(tensor);
tensor = que.DeQue<T>();
que.FreeTensor(tensor);
```

### Double Buffer 配置

**Double Buffer 是在 `InitBuffer` 的 `num` 参数中设置，与模板参数 `depth` 无关。**

| InitBuffer 参数 | 作用 | 说明 |
|----------------|------|------|
| `InitBuffer(que, num, size)` | `num` 控制 Double Buffer | `num=1`=单 Buffer，`num=2`=开启 Double Buffer |
| 模板参数 `depth` | 队列深度 | 表示可连续 EnQue 的次数 |

```cpp
// ✅ 开启 Double Buffer：在 InitBuffer 中设置 num=2
AscendC::TQue<AscendC::TPosition::VECIN, 1> que;  // 模板 depth=1 即可
pipe->InitBuffer(que, 2, size);  // num=2 开启 Double Buffer

// ✅ 关闭 Double Buffer
AscendC::TQue<AscendC::TPosition::VECIN, 1> que;
pipe->InitBuffer(que, 1, size);  // num=1 单 Buffer
```

### TQue Buffer 数量限制

| 产品系列 | eventID 数量 | 最大 TQue 数量 |
|---------|-------------|---------------|
| Atlas 训练系列 | 4 | 4 |
| Atlas 推理系列 AI Core | 8 | 8 |
| Atlas 推理系列 Vector Core | 8 | 8 |
| Atlas A2/A3 系列 | 8 | 8 |

**注意**：
- 不开启 Double Buffer（num=1）：最多可申请 8 个 TQue
- 开启 Double Buffer（num=2）：每个 TQue 占用 2 个 buffer，最多只能申请 4 个 TQue

```cpp
// 开启 Double Buffer 时，最多只能申请 4 个 TQue
pipe->InitBuffer(que0, 2, size);  // ✅
pipe->InitBuffer(que1, 2, size);  // ✅
pipe->InitBuffer(que2, 2, size);  // ✅
pipe->InitBuffer(que3, 2, size);  // ✅
pipe->InitBuffer(que4, 2, size);  // ❌ 超过限制
```

### TQue 正确用法

```cpp
// TQue：需要队列管理（MTE 搬运相关）
// 模板 depth=1 即可，Double Buffer 在 InitBuffer 的 num 参数中设置
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
pipe->InitBuffer(inQueueX, 2, bufferSize);  // num=2 开启 Double Buffer

AscendC::LocalTensor<half> x = inQueueX.AllocTensor<half>();
AscendC::DataCopyPad(x, xGm, {1, (uint32_t)(size * sizeof(half)), 0, 0, 0}, {false, 0, 0, static_cast<half>(0)});
inQueueX.EnQue(x);
// ...
AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
inQueueX.FreeTensor(xLocal);
```

---

## TBuf 详解

### 特性

| 特性 | 说明 |
|------|------|
| 内存用途 | 只能参与计算，无法执行 EnQue/DeQue |
| 内存分配 | 每次 InitBuffer 只分配一块内存 |
| Tensor 释放 | 无需手动释放 |

```cpp
// TBuf：纯计算缓冲区
AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
pipe->InitBuffer(workBuf, bufferSize);

// ✅ 使用 Get<T>() 获取 Tensor，无需释放
AscendC::LocalTensor<float> work = workBuf.Get<float>();
// ... 计算逻辑 ...
// 无需 FreeTensor
```

### TBuf 需要双缓冲时怎么办

TBuf 没有内建双缓冲机制（无 EnQue/DeQue 事件管理）。两种手动方案：

**方案一：两个独立 TBuf + 手动轮转**（推荐，语义清晰）

```cpp
AscendC::TBuf<AscendC::TPosition::VECCALC> bufA, bufB;
pipe->InitBuffer(bufA, tileSize);
pipe->InitBuffer(bufB, tileSize);

for (int i = 0; i < N; i++) {
    auto cur  = (i % 2 == 0) ? bufA.Get<float>() : bufB.Get<float>();
    auto next = (i % 2 == 0) ? bufB.Get<float>() : bufA.Get<float>();
    Prefetch(next, i + 1);   // 往另一块预取
    Compute(cur, i);         // 用当前块计算
}
```

**方案二：单 TBuf 按 offset 切半 + 奇偶索引**（省一次 InitBuffer，UB 紧张时用）

```cpp
AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
pipe->InitBuffer(buf, tileSize * 2);   // 一次申请两倍

auto half0 = buf.Get<float>();                  // 前半
auto half1 = buf.Get<float>()[tileElems];       // 后半（offset 切分）
// 奇偶轮改用 half0/half1
```

> ⚠️ 手动轮转时同步责任在使用者：往 next 写入的 DMA 与从 cur 读取的计算之间需要 `PipeBarrier` 或 SetFlag/WaitFlag 保证，TBuf 不会像 TQue 那样自动插入事件依赖。

---

## Double Buffer 流水线并行

### 核心认知

**Double Buffer 不是"用2块内存计算"，而是"用2块内存做搬入/搬出，使 MTE2/MTE3 与 Vector 计算并行"。**

本质：**内存搬运与计算并行，掩盖搬运延迟**。

### 硬件原理

- **MTE2**：搬运工，GM → UB
- **Vector**：加工员，计算
- **MTE3**：搬运工，UB → GM

### 时间线对比

**无 Double Buffer（串行）**：
```
Row 0: [MTE2][Vector][MTE3]
Row 1:                      [MTE2][Vector][MTE3]
```

**有 Double Buffer（并行）**：
```
Row 0: [MTE2-B0][Vector-B0][MTE3-B0]
Row 1:          [MTE2-B1][Vector-B1][MTE3-B1]
                  ↑ MTE2与Vector并行！
```

### 实现原则

| Buffer 类型 | InitBuffer num | 说明 |
|------------|----------------|------|
| `TQue<VECIN>` (MTE2 搬运) | **2** | num=2 开启 Double Buffer，与 Vector 并行 |
| `TQue<VECOUT>` (MTE3 搬运) | **2** | num=2 开启 Double Buffer，与 Vector 并行 |
| `TBuf<VECCALC>` (纯计算) | - | TBuf 不涉及 MTE 搬运 |

### 正确用法

```cpp
// 1. Init: num=2 开启 Double Buffer
pipe->InitBuffer(inQueueX,  2, tileSize * sizeof(T));
pipe->InitBuffer(outQueueY, 2, tileSize * sizeof(T));
pipe->InitBuffer(workBuf, workSize * sizeof(T));

// 2. Process: 单循环结构，TQue 自动轮转
for (int i = 0; i < totalTiles; i++) {
    CopyIn(i);   // MTE2 异步搬运
    Compute(i);  // Vector 计算
    CopyOut(i);  // MTE3 异步搬出
}

// 3. CopyIn
void CopyIn(int i) {
    LocalTensor<T> x = inQueueX.AllocTensor<T>();
    DataCopyPad(x, xGm[i * tileSize], {1, (uint32_t)(tileSize * sizeof(T)), 0, 0, 0}, {false, 0, 0, static_cast<T>(0)});
    inQueueX.EnQue(x);
}

// 4. Compute
void Compute(int i) {
    LocalTensor<T> x = inQueueX.DeQue<T>();
    LocalTensor<T> y = outQueueY.AllocTensor<T>();
    Add(y, x, constTensor, tileSize);
    outQueueY.EnQue(y);
    inQueueX.FreeTensor(x);
}

// 5. CopyOut
void CopyOut(int i) {
    LocalTensor<T> y = outQueueY.DeQue<T>();
    DataCopyPad(yGm[i * tileSize], y, {1, (uint32_t)(tileSize * sizeof(T)), 0, 0, 0});
    outQueueY.FreeTensor(y);
}
```

### 为什么能并行？

| 操作 | 特性 |
|------|------|
| `DataCopy` | 异步 DMA，立即返回 |
| `EnQue` | 非阻塞，标记就绪 |
| `DeQue` | 阻塞，等待就绪 |

### 常见误区

| 误区 | 正确理解 |
|------|---------|
| 需要手动拆成 Ping/Pong 两套代码 | 单循环 + `InitBuffer(que, 2, size)` 自动管理 |
| depth 模板参数控制 Double Buffer | Double Buffer 由 `InitBuffer` 的 `num` 参数控制 |
| depth 越大越好 | 模板 depth 通常设为 1，性价比最高 |
| 所有 buffer 都要 num=2 | 只有涉及 MTE 搬运的才需要 Double Buffer |
| InitBuffer 可以放循环里按需调用 | **InitBuffer 必须在循环外一次性调用**；循环内反复 InitBuffer 会重置队列状态，导致数据错乱/行为异常 |

---

## 双缓冲决策矩阵（何时开 / 何时不开）

双缓冲不是默认最优解。判断标准只有一条：**同类存储资源上是否存在连续迭代流，使"第 i+1 次搬运"能与"第 i 次计算"重叠**。不存在迭代流的 buffer 开 num=2 是纯浪费（UB/L1/L0 容量减半可用量）。

### 决策表

| Buffer 场景 | num | 理由 | 实证 |
|-------------|-----|------|------|
| 主循环 CopyIn 队列（VECIN/A1，逐 tile 迭代） | **2** | 经典场景：MTE2 与 Vector/Mmad 重叠 | flash_attention `queL0A_/queL0B_` num=2（matmul 内层 K-loop） |
| 主循环 CopyOut 队列（VECOUT/CO1，逐 tile 迭代） | **2** | MTE3 与下一次计算重叠 | 标准 elementwise 流水线 |
| **L0C 累加器** | **1（禁止 2）** | 链式 Mmad（`cmatrixInitVal=0`）依赖**同一块 L0C 驻留累加**；双缓冲破坏累加语义。且 L0C→GM 的 Fixpipe 在 stage 末尾才发生，无迭代流可隐藏 | flash_attention `queL0C_` num=1 |
| 被全局 barrier 串行化的 stage 间队列 | **1** | `SyncAll` 冲刷整条流水，stage 之间不存在可重叠的迭代流 | gdr_cv（MIX CV 融合）AIC 侧 a1/b1/a2/b2/c1 **全部 num=1** |
| 单次读写的 buffer（prologue 加载一次 / epilogue 写出一次） | **1** | 无第 i+1 次搬运，双缓冲无对象 | flash_attention `outputQue1_` num=1 |
| tile 极小（搬运延迟 < EnQue/DeQue 事件开销） | **1** | 双缓冲的事件管理开销超过隐藏的搬运延迟，负收益 | 参考 P98 自适应 dispatch 思想 |

### 反模式：全队列一律 num=2

```cpp
// ❌ 无脑全开：L0C 累加器也开双缓冲 → 链式 Mmad 结果错误 + L0 容量浪费
pipe.InitBuffer(a1_, 2, SZ_A1);
pipe.InitBuffer(b1_, 2, SZ_B1);
pipe.InitBuffer(c1_, 2, SZ_C1);   // L0C 必须 num=1！

// ✅ 按迭代流逐个判定（gdr_cv 实例：stage 被 SyncAll 串行 → 全部 num=1）
pipe.InitBuffer(a1_, 1, SZ_A1);
pipe.InitBuffer(b1_, 1, SZ_B1);
pipe.InitBuffer(a2_, 1, SZ_A2);
pipe.InitBuffer(b2_, 1, SZ_B2);
pipe.InitBuffer(c1_, 1, SZ_C1);
```

---

## MIX (CV 融合) 模式下双缓冲与 SyncAll 的交互

MIX kernel 的重叠设计分**两个正交层级**，先考虑层 1 再考虑层 2：

| 层级 | 机制 | 重叠对象 |
|------|------|---------|
| 层 1：核间重叠 | AIC/AIV 分工 + GM workspace 交接 | Cube 计算（AIC）与 Vector 计算（AIV）并行 |
| 层 2：核内重叠 | 队列 num=2 双缓冲 | 同一核内相邻迭代的搬运与计算并行 |

关键交互规则：

1. **`SyncAll<false>()` 冲刷整条流水**：barrier 两侧的 stage 是严格串行的，跨 stage 的队列双缓冲**无任何重叠机会**——stage-serialized 的 MIX kernel（如 gdr_cv）所有队列 num=1 即可。
2. **双缓冲只在 barrier 内部的连续迭代流里有意义**：典型是 AIC 侧 matmul 内层 K-loop（flash_attention 的 L0A/L0B num=2），该循环内没有 SyncAll。
3. **不要在 SyncAll 两侧用同一队列的双缓冲"假装流水"**：AIC 在 stage N 写入的 buffer，AIV 在 stage N+1 读取，靠的是 barrier 保证可见性，与队列 num 无关；GM workspace 才是跨核交接的正确载体（见 [api-mix-mode.md](api-mix-mode.md) §7）。
4. **UB/L1 预算联动**：MIX 下 AIC（L1/L0）与 AIV（UB）容量独立核算，双缓冲只放大对应侧的占用；AIV 侧 UB 预算紧张时优先砍双缓冲（num 2→1）而不是砍 tile。

---

## 批量搬运 + 逐行计算模式

### 适用场景

处理多行数据时，批量搬运减少 MTE2/MTE3 调用次数，充分利用带宽。

### 模式结构

```
CopyInBatch(N行) → 逐行计算(N次) → CopyOutBatch(N行)
```

### 代码模板

```cpp
__aicore__ inline void ProcessBatch()
{
    uint32_t totalRowsToProcess = endRow - startRow;
    if (totalRowsToProcess == 0) return;
    
    for (uint32_t tile = 0; tile < tilesPerCore; tile++) {
        uint32_t startLocalRow = tile * tileRows;
        
        // 边界检查：防止 uint32_t 下溢
        if (startLocalRow >= totalRowsToProcess) break;
        
        uint32_t remaining = totalRowsToProcess - startLocalRow;
        uint32_t rowsThisTile = (remaining < tileRows) ? remaining : tileRows;
        
        CopyInBatch(startLocalRow, rowsThisTile);
        ComputeBatch(rowsThisTile);
        CopyOutBatch(startLocalRow, rowsThisTile);
    }
}
```

### Host 侧 Tiling 计算

```cpp
// A2/A3 UB = 192KB
constexpr uint64_t UB_SIZE = 192 * 1024;
constexpr uint32_t MAX_BLOCK_COUNT = 4095;  // DataCopyPad blockCount 限制

// bytesPerTileRow: double buffer (in*2 + out*2)
uint32_t bytesPerTileRow = paddedColsT * typeSizeBytes * 4;

// tileRows
uint32_t tileRows = (UB_SIZE - overheadBytes) / bytesPerTileRow;
tileRows = std::max(1u, std::min(tileRows, MAX_BLOCK_COUNT));
```

### 注意事项

1. **tileRows 限制**：DataCopyPad 的 `blockCount` 最大 4095
2. **尾核处理**：`startLocalRow >= totalRowsToProcess` 时提前退出
3. **stride 计算**：UB 侧 stride 单位是 32 字节块，GM 侧是字节

---

## 多 stage 共享 L1 / L0 Buffer 的常量一致性

### 适用场景

mix kernel（`__mix__(N, M)`）或多 stage 算子中，同一对 L1 / L0 Buffer 经常被多个 Compute stage 函数共享访问做轮转（例如 GEMM 算子中两个连续 Mmad 计算共享同一对 L1 输入 buffer）。

### 必须一致的常量

各 stage 函数内使用以下常量，**必须与 InitBuffer 的实际分配字节数一致**：

- 单 slot 元素数（`slotElems`）
- 单 slot 字节数（`slotBytes`）
- per-slot stride / 槽偏移基数

### 典型踩坑

```cpp
// InitBuffer 时分配：
buf.InitBuffer(matAL1_, 64 * 1024 * PRELOAD_NUM);   // 每 slot 64KB

// ComputeStage1 中（正确）：
const uint32_t slotElems = 64 * 1024 / sizeof(DATA_T);   // 与 InitBuffer 一致
auto a = matAL1_.Get<DATA_T>()[loopSlot * slotElems];

// ComputeStage2 中（错误！）：
const uint32_t slotElems = 128 * 1024 / sizeof(DATA_T);  // ❌ 误写 128KB
auto b = matAL1_.Get<DATA_T>()[loopSlot * slotElems];    // task=0 偏移=0 蒙混，task=1+ 读越界脏数据
```

### 症状

- 任务 task=0 输出正常（偏移=0，即使常量错也不越界，读到的还是合法分配区）
- 任务 task=1+ 输出 NaN / inf（偏移到 buffer 末尾外的脏数据）
- "偶数 task PASS / 奇数 task FAIL" 或 "首个 task PASS / 后续 task 全爆炸" 型周期性错误

### 工程约束

把所有 per-slot 常量提到单一头文件或单一 constexpr 定义，所有 stage 引用同一定义：

```cpp
// constants.h
constexpr uint32_t L1_BUF_A_SLOT_BYTES = 64 * 1024;
constexpr uint32_t L1_BUF_B_SLOT_BYTES = 64 * 1024;
constexpr uint32_t L1_BUF_A_SLOT_ELEMS = L1_BUF_A_SLOT_BYTES / sizeof(DATA_T);
```

避免在每个 stage 函数内各自声明 `const uint32_t slotElems = ...;`。