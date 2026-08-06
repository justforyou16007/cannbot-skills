# Kernel 编码纪律（生成端 + 审查端共用）

本文件是全量算子生成中 **代码层硬约束 + 配套审查方法** 的合并清单。覆盖已复现的生成代码问题：

| 问题 | 对应规则 |
|------|----------|
| 1. 跨核 crosscore 同步：死锁 / 缺数据依赖 | RK-A01 |
| 2. 核内同步依赖缺失（TBuf / TQue） | RK-A02、RK-A03 |
| 3. UB / L1 / L0 容量越界（UBOOM） | RK-B01 |
| 4. AIC 中调用 vector 指令 coredump | RK-A04 |
| 5. UB / L1 / L0 buffer 规划缺少可追溯依据（重点 L1） | RK-B02 |
| 6. tiling 边界 off-by-one（end 参数语义） | RK-C01 |
| 7. impl 阶段私改 common.h 契约无留痕 / cube 与 vec 并发改冲突 | RK-A05 |
| 8. AIV sub-block 写 workspace 目标地址缺 rowStart_ 偏移 → fp32 精度灾难 | RK-D01 |

- **转译实现时**：按“禁止 / 要求”执行。
- **代码审查时**：按“检查方法”检查；只报告有明确代码证据的违反项。

## 条目格式

```
### RK-<类别><编号> <标题>
- **解决问题**：上表中的问题编号
- **锚点**：在代码哪里看（grep 关键字 / 文件位置）
- **禁止**：反模式
- **要求**：正模式
- **来源**：踩坑原因
- **检查方法**：如何扫描（grep 命令 + 抽样规则）
- **等级**：P0（必须返工）/ P1（应该返工）/ 风险（记录但放行）
```

P0：必然导致 hang / coredump / 越界 / 精度错误
P1：可能导致边界问题或实现意图不可追溯
风险：可读性问题，不影响正确性

---

## 类别 A：同步与依赖

### RK-A01 跨核同步必须配对、调用时序正确、notifyId 唯一

- **解决问题**：1
- **锚点**：所有 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 调用
- **禁止**：
  - 只有 `CrossCoreSetFlag` 没有 `CrossCoreWaitFlag`（或反之）—— 读方读不到数据，或写方覆盖未读数据
  - 写方写 workspace 后**未** `CrossCoreSetFlag`，读方先 `CrossCoreWaitFlag` —— **死锁**（读方永远等不到 flag）
  - 同一对同步复用同一 notifyId 多次 —— flag 残留导致跳过等待
  - 用 `SetFlag<HardEvent::...>` / `WaitFlag<HardEvent::...>` 做跨核同步（这是核内 API，跨核不生效）
  - 用 workspace 当隐式同步通道（无 flag，靠"先后执行"假设）
- **要求**：
  - 跨核同步对（写方核 → 读方核）：写方 `AscendC::CrossCoreSetFlag<0x2, PIPE_XXX>(notifyId)`，
    读方 `AscendC::CrossCoreWaitFlag<0x2, PIPE_XXX>(notifyId)`，notifyId 相同
  - **PIPE 选择**：写方按写入 workspace 的实际 pipe 选（如 `PIPE_MTE3`、`PIPE_FIX`），读方按读出
    workspace 的实际 pipe 选（如 `PIPE_MTE2`），不要照抄
  - **调用时序**：写方必须在 workspace 数据真正写完后（如 MTE3 搬运完成）才 `CrossCoreSetFlag`；
    读方必须在读 workspace 之前 `CrossCoreWaitFlag`
  - **notifyId 唯一**：每对跨核同步分配独立 notifyId，不得跨场景复用
- **来源**：跨核同步错配是 hang 和随机精度错误的首要原因
- **检查方法**：
  ```
  # 1. 找出所有跨核同步调用
  grep -nE 'CrossCoreSetFlag|CrossCoreWaitFlag' op_kernel/
  # 2. 找误用核内 API 做跨核同步
  grep -nE 'SetFlag<.*Hard|WaitFlag<.*Hard' op_kernel/
  ```
  对每个 `CrossCoreSetFlag`：
  - 确认存在对端 `CrossCoreWaitFlag` 且 notifyId 相同 → 不配对记 P0
  - 追踪调用时序：SetFlag 必须在 workspace 写完之后、WaitFlag 必须在读之前 → 时序颠倒记 P0
  - 确认 notifyId 在本算子内只用于这一对同步 → 复用记 P0
- **等级**：配对 / 时序 / notifyId 违反记 P0；矩阵表缺失记 P1

### RK-A02 TQue 必须配对 EnQue / DeQue

- **解决问题**：2（TQue 部分）
- **锚点**：所有 `AllocTensor` 调用点
- **禁止**：
  - `AllocTensor` 后直接使用 tensor，未入队 —— 下游 pipe 读到脏数据
  - 只 `DeQue` 不 `EnQue`（或反之）—— 死锁或数据竞争
  - `EnQue` 和 `DeQue` 的 pipe 配对与实际数据流不符
- **要求**：完整路径 `Alloc → EnQue<...> →（同核跨 pipe 或跨核消费）→ DeQue<...> → 用 → Free`
- **来源**：TQue 队列语义缺失会导致数据竞争和随机精度错误
- **检查方法**：
  ```
  grep -nE 'AllocTensor' op_kernel/
  ```
  对每个命中点，追踪同一函数或紧邻调用链内是否有对应 `EnQue<...>` 和 `DeQue<...>` → 缺失记 P0
- **等级**：P0

### RK-A03 TBuf 核内同步必须按方向选对 API

- **解决问题**：2（TBuf 部分，仅核内）
- **锚点**：`LoadData` / `DataCopy` / `LoadNdGmToNzL1` / `LoadNzL1ToZzL0A` /
  `LoadNzL1ToZnL0BWithTranspose` 等 TBuf 写入/读出位置
- **禁止**：
  - 写入 TBuf 后无任何同步 —— 下游读到不完整数据
  - 用 TBuf 当 TQue 用（不带队列语义却依赖自动同步）
  - 不区分同步方向、PIPE 配对是否正确，机械套用 `SetWaitFlag<HardEvent::...>()`
  - 反向同步使用 `SetWaitFlag` 封装（封装 API 仅适用于正向同步）
- **要求**：
  - **识别同步方向**：
    - **正向同步**（生产者 pipe 写完 → 消费者 pipe 读，方向与数据流一致，如 `MTE2_V` 是 V 等 MTE2、
      `MTE1_MTE2` 是 MTE2 等 MTE1）：使用 `op_kernel/<算子>_common.h` 的
      `SetWaitFlag<HardEvent::XXX>()` 封装（内部 `SetFlag<evt>(eventId); WaitFlag<evt>(eventId);`）
    - **反向同步**（消费者 pipe 完成 → 生产者 pipe 才能复用 buffer，方向与数据流相反，如
      `MTE2_MTE1` 是 MTE1 等 MTE2 把 L1 读完、`V_MTE2` 是 MTE2 等 V 把 UB 读完）：
      按实际硬件约束选择可用 API，不得直接套用 `SetWaitFlag`
  - **PIPE 配对正确**：HardEvent 的两个 pipe 必须与实际数据流方向一致
    （如 MTE2→V 用 `HardEvent::MTE2_V`；不要写成 `V_MTE2`）
  - 不确定方向时查 `ascendc-sync-guide`，不要按名字猜
- **来源**：TBuf 不带队列语义，所有时序依赖必须显式同步；HardEvent 方向写反或 PIPE 选错
  会编译过但运行时同步失效
- **检查方法**：
  ```
  grep -nE 'LoadData|DataCopy|LoadNd|LoadNz' op_kernel/
  grep -nE 'SetWaitFlag|SetFlag<.*Hard|WaitFlag<.*Hard|HardEvent::' op_kernel/
  ```
  对每个写 TBuf 的位置：
  - 确认写方紧邻有同步 —— 完全没有记 P0
  - 解析 HardEvent：确认 PIPE 顺序与实际数据流方向一致（如 `MTE2_V` 用于"MTE2 写完 V 读"，不能反）→
    方向错配记 P0
  - 若是反向同步却用了 `SetWaitFlag` 封装 → 记 P0
- **等级**：P0

### RK-A04 cube 类内禁止出现 vector 指令

- **解决问题**：4
- **锚点**：CubeKernel 类（`op_kernel/*cube*.h` / `*cube*.cpp`，或文件内以 Cube 命名的类）
- **禁止**：cube 类作用域内出现 `Duplicate` / `Adds` / `Muls` / `Cast` / `Exp` / `Softmax` /
  `ReduceMax` / `ReduceSum` / `Brcb` / `Div` / `Mul` 等任何 vector 原语
- **要求**：cube 类只允许 cube 路径原语，除了DataCopy(Cube路径 GM->L1)、LoadData、Mmad、Fixpipe等API，其他不了解的API必须通过`ascendc-cube-guide`查询是 cube 原语再用，
  - 所有 vector 计算必须放到 VectorKernel 类，通过 workspace 传数据
- **来源**：AIC 硬件不支持 vector 指令，调用会 coredump；编造的 cube API 名会导致编译失败
- **检查方法**：
  ```
  # 1. 在 cube 文件中搜 vector 原语
  grep -nE 'Duplicate|Adds|Muls|Cast|Exp|Softmax|ReduceMax|ReduceSum|Brcb|Div|Mul|Add|Sub|Max' op_kernel/*cube*
  ```
  - 命中 vector 原语 → P0
- **等级**：P0

### RK-A05 共享契约变更必须同步实现侧

- **解决问题**：7
- **锚点**：host 与 kernel 共享的参数结构、notifyId 宏、buffer 数和算子常量。
- **禁止**：只改一侧的字段、硬编码冲突的 notifyId，或使 Cube / Vector 对同一字段作出不同解释。
- **要求**：共享字段的名称、类型、单位和生产/消费方必须一致；新增同步 ID 时记录其写方 pipe、读方 pipe 与配对关系。
- **检查方法**：对照 host 与 kernel 的共享定义，确认每个字段、notifyId 和 buffer 数在双方一致；对每个 SetFlag/WaitFlag 核对唯一 ID 与配对。
- **等级**：P0

---

## 类别 B：容量与 buffer

### RK-B01 InitBuffer 即时静态占用空间，所有 buffer 总量不超硬件上限

- **解决问题**：3
- **锚点**：所有 `InitBuffer` / `TQueBuffer` / `TBufBuffer` / `pipe.InitBuffer(...)` 调用点
- **核心认知**：`pipe.InitBuffer()` 在**调用时**即在 UB / L1 / L0 上**静态分配**空间，不存在"分时复用"——
  无论 buffer 当前是否活跃，只要已被 InitBuffer，就持续占用硬件空间直到核函数结束。
  **所有 InitBuffer 的字节数之和**（跨所有 TQue / TBuf / pipe）必须 ≤ 硬件上限。
- **禁止**：
  - `InitBuffer(size)` 不写计算依据；只写 magic number
  - **以"分时使用、不同时活跃"为由多分配 buffer**：这是 UBOOM 的首要根因，分时不等于分空间
  - 任意维度的所有 buffer 字节总和超硬件上限（参考值：910B UB 192KB/核、L1 512KB/核）
- **要求**：调用**上方一行**注释 `// 预算: <公式> = <数值> Byte, 累计已用 <已用> Byte, 上限 <硬件容量> Byte`，
  且**所有 InitBuffer 总和** ≤ 上限
- **来源**：UB 溢出（UBOOM）；agent 误认为"不同时用就不占空间"，实际 InitBuffer 即静态占用
- **检查方法**：
  ```
  grep -nE 'InitBuffer' op_kernel/
  ```
  - 逐行累加所有 InitBuffer 的字节数，验证总和 ≤ 上限 → 超出记 P0
  - 若代码注释或 review 中出现"分时复用 UB"字样 → 直接记 P0，视为错误认知
  - 上方注释缺公式 + 数值 + 硬件上限 → 记 P1
- **等级**：总量超上限 / 分时复用误用 → P0；缺注释或公式不一致 → P1

### RK-B02 buffer 分区必须具备可复查的容量与用途依据（重点 L1）

- **解决问题**：5
- **锚点**：所有 L1 / L0 / UB buffer 的 `InitBuffer` 调用，**重点关注 L1**（最易自由发挥）
- **禁止**：
  - buffer 划分（数量、用途、tile shape、大小公式）没有实现依据
  - 在不说明容量、数据流或 API 限制的前提下，任意拆分、合并或新增 buffer
  - 用“性能优化”为名调整 buffer 但未在代码附近说明理由
- **要求**：
  - 每个 L1/L0/UB buffer 注释或相邻代码必须能说明用途、tile 大小、数量和容量公式
  - 因 AscendC API 限制或容量压力调整 buffer 时，必须在改动点说明具体理由
- **来源**：coder 在写 L1 buffer 时容易“自由发挥”（改 tile shape、加临时 buffer），导致容量或数据流无法复查
- **检查方法**：
  - 随机抽 2-3 个 L1 buffer，确认用途 / tile shape / 数量 / 容量公式可从代码和 tiling 推导
  - 无法说明 buffer 分区理由或容量公式 → 记 P0
- **等级**：P0

---

## 类别 C：接口语义

### RK-C01 tiling 中 end 类参数为左闭右开

- **解决问题**：6
- **锚点**：TilingData 字段、tiling 函数、kernel 主循环中所有以 `End` / `end` 命名的字段
  （如 `gs1Start` / `gs1End` / `gs2Start` / `gs2End` / `kvStart` / `kvEnd`）
- **禁止**：
  - 循环条件用 `<=`（`for (i = start; i <= end; ++i)`）—— 多算一个 block，越界 GM
  - 尾块大小用 `end - start + 1` —— 多算一个元素，越界
  - 字段语义在 tiling 侧和 kernel 侧理解不一致（写方按 `[start, end)`，读方按 `[start, end]`）
- **要求**：所有 end 类参数语义为 `[start, end)`，循环用 `<`，尾块大小用 `end - start`；
  TilingData 字段表（planner 产出）的"语义"列必须标注 `[start, end)`
- **来源**：常见 off-by-one 错误源头，多次复现
- **检查方法**：
  ```
  # 循环条件用 <= 配 end 字段
  grep -nE '<=.*[Ee]nd' op_kernel/ op_host/
  # 尾块大小 +1
  grep -nE '[Ee]nd.*-.*[Ss]tart.*\+.*1' op_kernel/ op_host/
  ```
  命中即视为潜在 off-by-one，人工确认后 → 记 P0
- **等级**：P0

---

### RK-C02 Cast 精度转换舍入模式

- **解决问题**：精度转换时舍入模式错误导致误差超阈值
- **锚点**：所有 Cast 调用（FP32↔FP16/BF16）
- **禁止**：
  - 低精度→高精度（FP16/BF16 → FP32）时使用 `ROUND` 或 `RINT`
  - 高精度→低精度（FP32 → FP16/BF16）时省略舍入模式或使用 `NONE`
- **要求**：
  - 高精度→低精度：显式指定 `ROUND` 或 `RINT`
  - 低精度→高精度：显式指定 `NONE`
- **来源**：Cast 默认舍入模式在不同精度方向的语义不一致，导致隐式精度损失
- **检查方法**：
  ```
  # 查找所有 Cast 调用
  grep -nE 'Cast.*FP(16|32)|Cast.*BF16' op_kernel/
  # 人工确认：高→低必须有 ROUND/RINT，低→高必须有 NONE
  ```
- **等级**：P0

---

## 类别 D：MIX kernel sub-block 数据分区

### RK-D01 AIV sub-block 写 workspace 必须按 rowStart_ 偏移目标地址

- **解决问题**：MIX kernel fp32 hi/lo 分片路径下，多 AIV sub-block 并行向同一块 workspace
  写入 Q/K/V 分片数据时，sub-block 1 覆盖 sub-block 0 的数据，导致 AIC 读到错误的行段，
  fp32 精度灾难性退化（MRE≈1.0, cos_sim≈0.2）。
- **锚点**：`op_kernel/*_vec.h` 中所有 `SplitTileToWs` / `ProducerAcquire` 调用点，
  以及任何 AIV 侧 `DataCopy(workspace[offset], ub, ...)` 写入跨核共享 workspace 的位置。
- **禁止**：
  - 多个 sub-block 向同一 workspace slot 的**相同偏移**写入数据 —— sub-block 1 会覆盖
    sub-block 0 已写入的数据
  - 依赖"sub-block 串行执行"假设 —— sub-block 可能并行，写入顺序不确定
  - 仅源地址包含 `rowStart_` 偏移而目标地址遗漏
- **要求**：
  - 每个 sub-block 向 workspace 写入时，目标地址必须包含本 sub-block 的行偏移：
    `workspace[(uint64_t)rowStart_ * dim]`，确保 sub-block 0 写入 rows 0..SUB_ROWS-1，
    sub-block 1 写入 rows SUB_ROWS..2*SUB_ROWS-1
  - 此要求适用于所有 AIV→AIC workspace 写入路径：SplitQ、SplitKV、以及任何 DataCopy
    写 workspace 后由 AIC ConsumerAcquire 读取的场景
  - 已经正确包含偏移的现有代码（如 `pSlot[(uint64_t)rowStart_ * BLOCK_N]`）不受影响
- **来源**：ScaledDotProductAttention fp32 路径 `SplitQ`/`SplitKV` 中两 sub-block 均写
  `workspace[0]`，sub-block 1 覆盖 sub-block 0 的 Q/K/V hi/lo 数据 → AIC 加载到错乱的行 →
  attention 权重全错 → 多 tile 精度 MRE=0.79~1.46。单 tile 碰巧通过（有效行数少且 padding
  行覆盖不影响有效输出）。
- **检查方法**：
  ```
  # 1. 找出所有 sub-block 写 workspace 的位置
  grep -nE 'ProducerAcquire|SplitTileToWs|DataCopy.*Slot|DataCopy.*slot' op_kernel/*_vec*
  # 2. 对每个写入点，确认目标地址包含 rowStart_ 偏移
  grep -nE 'rowStart_' op_kernel/*_vec*
  ```
  - 若 ProducerAcquire 返回的 slot 直接传给 SplitTileToWs 或 DataCopy 而无 rowStart_ 偏移 → 记 P0
  - 人工确认：源地址和目标地址是否都包含 sub-block 行偏移 → 缺任意一侧记 P0
- **等级**：P0（导致 fp32 精度灾难性退化）

---

## 扩展占位

后续发现新的代码层常见问题，按上述格式追加条目：

- 在已有类别下追加 `RK-XX<下一个编号>`（A 类下一个 RK-A06，B 类下一个 RK-B03，C 类下一个 RK-C03，D 类下一个 RK-D02）
- 若是新类别，新建"类别 X：xxx"小节，编号续接
- 每条规则同时包含生成端约束（禁止/要求）和审查端方法（检查方法/等级），无需同步两份文件
