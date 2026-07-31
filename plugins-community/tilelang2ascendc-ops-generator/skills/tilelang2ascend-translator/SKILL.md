---
name: tilelang2ascend-translator
description: >
  AscendC kernel 转译与实现专家 Skill。将 TileLang 设计转译为 AscendC kernel，
  并生成 model_new_ascendc.py 调用 AscendC kernel。
  当 TileLang 设计完成需要转译为 AscendC kernel 时，使用此 skill。
argument-hint: >
  输入：output_dir 目录路径（包含 tile_level/ 和 model_new_tilelang.py）。
  输出：kernel/ 下的 AscendC 实现、model_new_ascendc.py。
---

# AscendC Kernel 转译 Skill

你是一名 AscendC kernel 转译与实现专家。你的目标是将 TileLang 设计转译为 AscendC kernel，并生成 `{output_dir}/model_new_ascendc.py` 调用 AscendC kernel，最终通过 AscendC 验证。TileLang 在这里是设计输入，不是 correctness gate。

## 前置条件
本阶段开始前，以下产物必须已经存在：
- `{output_dir}/design/tile_level/` — TileLang tile-level 设计，作为转译输入
- `{output_dir}/model_new_tilelang.py` — TileLang 绑定层/设计表达，可参考但不作为正确性依据

## 关键限制
- 必须将核心计算融合成单个算子实现，不要拆分成多个独立算子。
- `model_new_ascendc.py` 中禁止使用 torch 算子；只允许进行张量创建，张量变换以及调用你实现的自定义算子。
- 在 AscendC 实现中应尽可能避免标量逐元素写法，优先使用块级或向量化操作；只有在确实无法避免时才使用标量逻辑。🛑 编写 Compute 前必须对照 `.claude/skills/tilelang2ascend-translator/references/scalar-to-vector.md` 中的 7 种标量反模式，确认不落入标量写法；生成后必须执行步骤 1.5 的标量残留自检。
- 只允许修改或新增 `{output_dir}/` 目录中的文件，不要改动其他目录中的文件。
- 只允许读取当前工作区目录结构内的文件与子目录；禁止读取当前工作区之外的任何路径，包括父目录、兄弟目录、用户目录、绝对路径以及系统其他目录。
- 禁止读取 `.claude/skills/tilelang2ascend-translator/references/TileLangAscendProgrammingGuide.md`；该文档是 TileLang 编程指南，仅供 TileLang 阶段使用，与本阶段无关。
- 严格按照算子描述生成kernel，ascend c kernel的功能应该和标杆完全一致，不能出现部分功能使用ascend c，部分使用torch算子的情况
- 即使测试用例中不包含某个功能或者分支对应的case，也要生成对应的ascend c kernel代码
- **🛑 同步机制强制门禁**: 涉及 MIX_AIC / CrossCore / WorkspaceQueue / 死锁 / 全零输出 时，必须先完成 **步骤 0-C** 的同步 checklist。详见下方步骤 0-C 章节。

## 目标任务目录结构
```text
.
├── {output_dir}/         # 当前活跃任务目录
│   ├── model.py          # 参考 PyTorch 模型，禁止修改
│   ├── <op_name>.json    # 原始测试用例文件（备份保留）
│   ├── <op_name>.json.bak# 原始 .json 备份
│   ├── design/           # TileLang DSL 用于表达 kernel 设计
│   │   ├── design.md     # 设计文档（简单算子路径）或 不存在（复杂算子路径）
│   │   ├── block_level/  # TileLang block-level 设计（已由上一阶段完成）
│   │   └── tile_level/   # TileLang tile-level 设计（已由上一阶段完成，作为转译输入）
│   ├── kernel/           # AscendC kernel（op_host/ + op_kernel/ 分层）
│   │   ├── CMakeLists.txt
│   │   ├── setup.py      # whl 打包配置
│   │   ├── ops.h         # 算子声明
│   │   ├── register.cpp  # torch.ops.npu.* 注册（仅注册）
│   │   ├── op_host/
│   │   │   └── <op_name>.cpp  # Host 端: tiling + EXEC_KERNEL_CMD
│   │   ├── op_kernel/
│   │   │   └── <op_name>.cpp
│   │   └── utils/        # 固定工具（从模板复制，不生成）
│   │       └── torch_kernel_helper.h
│   ├── test/             # 测试目录
│   │   ├── <op_name>-test-cases.md
│   │   └── test_<op_name>.py
│   ├── model_new_tilelang.py # 上一阶段产物，可参考但不要修改
│   └── model_new_ascendc.py  # AscendC wrapper → 内部调用 torch.ops.npu.<op>()
└── <other_tasks>/        # 其他历史任务，可作为参考实现
```

## Skill 参考资料
本 skill 提供以下参考资料：
- `.claude/skills/tilelang2ascend-translator/references/dsl2Ascendc.md` — TileLang 转 AscendC 指南
- `.claude/skills/tilelang2ascend-translator/references/TileLang-AscendC-API-Mapping.md` — TileLang 与 AscendC API 映射表
- `.claude/skills/tilelang2ascend-translator/references/scalar-to-vector.md` — 标量计算向量化指南
- `.claude/skills/tilelang2ascend-translator/references/AscendCVerification.md` — AscendC 验证指南
- `.claude/skills/tilelang2ascend-translator/references/attention-patterns/AttentionPatternIndex.md` — Attention / FlashAttention 类算子的模式路由索引（TND、paged KV cache、mask/causal、GQA/MQA、MLA、topk sparse KV、sink attention）
- `.claude/skills/tilelang2ascend-translator/scripts/evaluate_ascendc.sh` — AscendC 评测脚本
- `workflows/templates/archive_tasks/` — 历史成功任务，host/kernel 完整参考实现（**编译/运行时错误时优先查阅**）

### 🛑 官方文档目录（asc-devkit，强制查阅）

以下所有路径相对于 `asc-devkit/` 根目录。**编写/修改任何 kernel 代码前，必须先查阅对应的官方文档。禁止凭记忆或猜测 API 签名、参数、dtype 支持矩阵。**

| 查阅入口 | 内容 | 何时查阅 |
|----------|------|---------|
| `asc-devkit/docs/api/Ascend-C-API列表.md` | API 分类总览与快速索引 | 每次代码生成前 |
| `asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/类型转换/Cast.md` | Cast API 签名与 dtype 支持 | 使用 Cast 时 |
| `asc-devkit/docs/api/SIMD-API/基础数据结构/GlobalTensor/GlobalTensor简介.md` | GlobalTensor 完整 API | 使用 GlobalTensor 时 |
| `asc-devkit/docs/api/SIMD-API/基础数据结构/LocalTensor/LocalTensor简介.md` | LocalTensor 完整 API | 使用 LocalTensor 时 |
| `asc-devkit/docs/guide/算子实践参考/SIMD算子实现/矢量编程/基础矢量算子.md` | CopyIn→Compute→CopyOut 标准范式 | 每次代码生成前 |
| `asc-devkit/docs/guide/算子实践参考/SIMD算子实现/矢量编程/TBuf的使用.md` | UB 临时缓冲区管理 | 分配 TBuf 时 |
| `asc-devkit/docs/guide/算子实践参考/SIMD算子实现/融合算子编程/` | 多步计算融合模式 | 融合算子时 |
| `asc-devkit/examples/01_simd_cpp_api/` | 官方 SIMD C++ 示例 | API 用法不确定时 |
| `workflows/templates/archive_tasks/rms_norm/` | EXEC_KERNEL_CMD 正确传参模式 | 编写 op_host 时 |

除非用户明确指定其他目录，否则默认使用传入的 `output_dir` 作为当前任务目录。
其他任务目录可以作为参考实现。

---

### 🛑 步骤 0-A: Attention 算子模式路由（Attention / FlashAttention 类算子强制执行）

🛑 **无论算子类型如何，第一步必须读取 model.py**：
   Read `{output_dir}/model.py` 的 forward() 方法，逐行检查计算逻辑。
   转译阶段的输入固然是 tile_level 设计文件，但判断算子是否属于 Attention 类
   必须回到 model.py 的原始计算逻辑。**禁止**凭 tile_level 文件名或记忆跳过此步。

**触发条件**：读取 model.py 后，检查 forward() 是否包含以下任一特征：
- `softmax(Q @ K^T / sqrt(d)) @ V` 或等价 attention 计算模式（如 `F.softmax(matmul(Q, K^T) / sqrt(dk)) @ V`）
- `scaled_dot_product_attention` / `F.scaled_dot_product_attention`
- 类名包含 `Attention` / `SDPA` / `Flash`
- tile-level 设计中包含 Q/K/V 三输入 attention 结构

如果触发条件满足，必须逐个完成以下 checklist：

```
0-A.1 🛑 读取 AttentionPatternIndex.md（必须，不可跳过）:
    Read .claude/skills/tilelang2ascend-translator/references/attention-patterns/AttentionPatternIndex.md

0-A.2 🛑 逐条回答"生成前问题"中的 7 个诊断问题，记录命中的模式:
    1. 输入是标准 [B,H,S,D] 还是 (T,H,D) 拼接布局？
    2. K/V 是连续 tensor 还是 paged cache？
    3. Hq 和 Hkv 是否相等？
    4. Dqk 和 Dv 是否相等？
    5. 是否有 sink_k/sink_v？
    6. 是否有 indices/topk？
    7. 是否有 causal、padding、显式 mask？
    
    如果 7 项全否定 → 命中"标准 Attention" → 下一步 0-A.3 读 archive 模板
    如果任一命中 → 下一步 0-A.3 读对应的 pattern 文档（可组合）

0-A.3 🛑 只读取命中的文档（渐进式披露，只读需要的）:
    - 命中模式 → Read 对应文档顶部的"先读这个"

0-A.4 🛑 在思考中确认:
    - 已读的 pattern 文档列表及其关键语义边界
    - 组合顺序（多模式命中时按 TND → Head Sharing → MLA → Sink → Sparse → Paged → Mask 顺序理解）
    - 本算子的 AscendC 转译策略应与命中的 pattern 对齐
```

**门禁规则**：
- 如果触发条件满足但 0-A.1-0-A.4 未完成 → **禁止**进入步骤 1，**禁止**编写任何 kernel/ 代码
- 如果触发条件不满足 → 跳过步骤 0-A，直接进入步骤 0-B
- 禁止凭记忆或经验跳过模式文档直接转译

---

## 🛑 步骤 0-B: 查阅官方文档（每次代码生成/修改前强制执行）

**在 Edit/Write 任何 `kernel/` 下的代码文件之前，必须完成以下查阅步骤。此步骤不可跳过。**

```
0.1 阅读标准范式:
    asc-devkit/docs/guide/算子实践参考/SIMD算子实现/矢量编程/基础矢量算子.md
    → 确认 CopyIn→Compute→CopyOut 的完整流水线模式

0.2 阅读 EXEC_KERNEL_CMD 正确模式:
    workflows/templates/archive_tasks/rms_norm/kernel/op_host/rms_norm.cpp
    → 确认: 所有 tiling 参数必须是独立标量左值，禁止传 struct 指针
    → 确认: blockDim = usedCoreNum（多核统一分发），禁止 host 侧逐核循环
    → 确认: 核数来源于平台 API 动态获取（GetCoreNumAic/Aiv），非硬编码常量

0.3 逐个查阅要使用的 API 文档:
    根据算子计算逻辑，列出所有将使用的 AscendC API，然后**逐个**查阅以下精确路径的文档。
    ⚠️ 每个 API 必须确认: ① 模板参数（类型/非类型）② 函数参数（个数/类型）③ dtype 支持矩阵 ④ work buffer 需求。
    **禁止凭记忆或猜测 API 签名**。

    ── 数据搬运 ──
    - DataCopyPad → asc-devkit/docs/api/SIMD-API/基础API/Memory数据搬运/DataCopyPad(ISASI).md
      ⚠️ 签名两态: GM→UB 4参(dst,src,cp,pp), UB→GM 3参(dst,src,cp)
    - DataCopy → asc-devkit/docs/api/SIMD-API/基础API/Memory数据搬运/DataCopy/DataCopy.md

    ── 类型转换 ──
    - Cast → asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/类型转换/Cast.md
      ⚠️ 确认 bfloat16→float32 和 float32→bfloat16 的 RoundMode 参数

    ── 矢量计算 (Memory) ──
    - Mul → asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Mul.md
    - Add → asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Add.md
    - Sub → asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Sub.md
    - Rsqrt → asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Rsqrt.md

    ── 标量计算 (Reg) ──
    - Muls → asc-devkit/docs/api/SIMD-API/基础API/Reg矢量计算/基础算术/Muls-27.md
    - Adds → asc-devkit/docs/api/SIMD-API/基础API/Reg矢量计算/基础算术/Adds-28.md
    - Rsqrt (scalar) → 与矢量 Rsqrt 同族，查阅基础算术目录

    ── 高阶 API ──
    - ReduceSum → asc-devkit/docs/api/SIMD-API/高阶API/归约操作/ReduceSum接口/ReduceSum-90.md
      ⚠️ 模板: <T, pattern, isReuseSource>, 参数: (dst,src,workBuf,srcShape[],srcInnerPad)
      ⚠️ GetReduceSumMaxMinTmpSize → 同目录 GetReduceSumMaxMinTmpSize.md
    - Broadcast → asc-devkit/docs/api/SIMD-API/高阶API/张量变换/Broadcast.md
      ⚠️ 模板: <T, dim, axis, isReuseSource>, dim∈{1,2}, axis∈{0,1}
    - Sort → asc-devkit/docs/api/SIMD-API/高阶API/排序操作/Sort.md
      ⚠️ 降序需 Negate→Sort→Negate；元素数 pad 到 64 倍数；需索引 buffer + 临时 buffer
    - MrgSort → asc-devkit/docs/api/SIMD-API/高阶API/排序操作/MrgSort-93.md
      ⚠️ 需 MrgSortParams 参数结构体
    - Sort32 → asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/排序组合（ISASI）/Sort32.md
    - GatherMask → asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/比较与选择/GatherMask.md
      ⚠️ 确认 deinterleave/repeatMaskParams 参数语义；调用后需 PipeBarrier<PIPE_V>
    - Cos → asc-devkit/docs/api/SIMD-API/高阶API/数学计算/Cos接口/Cos.md
      ⚠️ GetCosMaxMinTmpSize → 同目录 GetCosMaxMinTmpSize.md
    - Sin → asc-devkit/docs/api/SIMD-API/高阶API/数学计算/Sin接口/Sin.md
      ⚠️ GetSinMaxMinTmpSize → 同目录 GetSinMaxMinTmpSize.md

    ── 同步控制 ──
    - PipeBarrier → asc-devkit/docs/api/SIMD-API/基础API/同步控制/核内同步/PipeBarrier(ISASI).md
      ⚠️ 确认 PIPE_MTE2/PIPE_MTE3/PIPE_V/PIPE_ALL 各 barrier 的放置位置规则
    - CrossCoreSetFlag/WaitFlag → .claude/skills/tilelang2ascend-translator/references/ascendc-sync-guide.md
      ⚠️ 确认 mode2 下 Set/Wait 两侧 PIPE 参数完整且配对
      ⚠️ AIC 侧: Set<0x2, PIPE_FIX> + Wait<0x2, PIPE_FIX>
      ⚠️ AIV 侧: Set<0x2, PIPE_MTE2> + Wait<0x2, PIPE_MTE2>（MTE3 写用 PIPE_MTE3）
      ⚠️ 封装泛型工具类（如 WorkspaceQueue）时 ProducerAcquire/ConsumerAcquire 必须将 PIPE 模板化传入

    📋 **查阅完成后，在思考中列出 API 签名清单**:
    对每个 API 记录:
    - 完整模板参数 (如 ReduceSum<float, AscendC::Pattern::Reduce::AR, false>)
    - 完整函数参数名和类型
    - work buffer 需求 (需要/不需要, 如需要则列出 GetXxxMaxMinTmpSize 的查阅结果)
    - dtype 约束

0.4 查阅 TBuf 用法:
    asc-devkit/docs/guide/算子实践参考/SIMD算子实现/矢量编程/TBuf的使用.md
    → 确认 UB 临时缓冲区的正确分配模式

0.5 🛑 验证所有 work buffer 尺寸（运行时正确性铁律）:
    对于每个使用 TBuf<uint8_t> 作为 work buffer 传入的 API，**必须在 host 端通过对应的
    GetXxxMaxMinTmpSize 计算正确尺寸，禁止在 kernel 中硬编码 work buffer 大小**。

    │ Work Buffer 使用者 │ 尺寸获取 API (host 端调用) │ API 文档 │
    │-------------------│---------------------------│---------│
    │ ReduceSum         │ GetReduceSumMaxMinTmpSize │ asc-devkit/docs/api/SIMD-API/高阶API/归约操作/ReduceSum接口/GetReduceSumMaxMinTmpSize.md │
    │ ReduceMax         │ GetReduceMaxMaxMinTmpSize │ asc-devkit/docs/api/SIMD-API/高阶API/归约操作/ReduceMax接口/GetReduceMaxMaxMinTmpSize.md │
    │ ReduceMin         │ GetReduceMinMaxMinTmpSize │ asc-devkit/docs/api/SIMD-API/高阶API/归约操作/ReduceMin接口/GetReduceMinMaxMinTmpSize.md │
    │ Cos               │ GetCosMaxMinTmpSize       │ asc-devkit/docs/api/SIMD-API/高阶API/数学计算/Cos接口/GetCosMaxMinTmpSize.md │
    │ Sin               │ GetSinMaxMinTmpSize       │ asc-devkit/docs/api/SIMD-API/高阶API/数学计算/Sin接口/GetSinMaxMinTmpSize.md │
    │ SinCos            │ GetSinCosMaxMinTmpSize    │ asc-devkit/docs/api/SIMD-API/高阶API/数学计算/SinCos接口/GetSinCosMaxMinTmpSize.md │
    │ Broadcast         │ GetBroadCastMaxMinTmpSize │ asc-devkit/docs/api/SIMD-API/高阶API/张量变换/GetBroadCastMaxMinTmpSize.md │
    │ Sort              │ GetSortTmpSize / GetSortMaxMinTmpSize │ asc-devkit/docs/api/SIMD-API/高阶API/排序操作/GetSortTmpSize.md │

    **验证步骤 (每次编写 kernel 前强制执行)**:
    a. 列出本算子所有使用 work buffer 的 API
    b. 逐一查阅上表中对应的 GetXxxMaxMinTmpSize 文档
    c. 确认每个 API 的 work buffer 最小/最大尺寸计算方法
    d. 在 host 端 tiling 函数中调用 GetXxxMaxMinTmpSize，将结果作为 tiling 参数传入 kernel
    e. kernel 中 InitBuffer 的 work buffer 尺寸必须来自 tiling 参数，**禁止**硬编码为固定值
    f. 在思考中记录: 每个 work buffer 的计算结果和对应的 API 名称

    ⚠️ 本步骤为**运行时正确性硬性要求**。硬编码 work buffer 尺寸 < 实际所需最小值
       会导致 vector core timeout (507034) / UB 内存违例等运行时错误。

⚠️ 未完成以上 0.1-0.5 全部步骤前，禁止进入步骤 1 编写代码。Attention 类算子还必须完成步骤 0-A。
   查阅完成后，在思考中明确列出已查阅的文档路径及其关键约束。
```

---

### 🛑 步骤 0-C: 同步机制门禁（涉及跨核同步 / MIX_AIC / 输出异常时强制执行）

**触发条件**（任一满足即触发）：
- kernel 使用了 `KERNEL_TYPE_MIX_AIC` 混合核模式
- 代码中出现 `CrossCoreSetFlag` / `CrossCoreWaitFlag` / `WorkspaceQueue`
- 运行时出现**全零输出**、死锁、hang、vector core timeout (507034)
- 编译后功能验证 FAIL 但无编译错误

如果触发条件满足，必须逐个完成以下 checklist：

```
0-C.1 🛑 读取 ascendc-sync-guide.md（必须，不可跳过）:
    Read .claude/skills/tilelang2ascend-translator/references/ascendc-sync-guide.md 全文

0-C.2 🛑 逐条在思考中确认以下 checkpoint:
    ① PIPE 配对:
       - AIC 侧所有 CrossCore Set/Wait → PIPE_FIX
       - AIV 侧所有 CrossCore Set/Wait → PIPE_MTE2（MTE3 写操作用 PIPE_MTE3）
       - WaitFlag 是否漏写 PIPE 模板参数
    ② CV1:2 模式信号计数:
       - AIC Set 1 次 → 两个 AIV 各 Wait 1 次
       - 两个 AIV 各 Set 1 次 → AIC Wait 2 次（等两个 AIV 都完成）
    ③ 封装泛型工具类:
       - WorkspaceQueue ProducerAcquire/ConsumerAcquire 是否通过模板参数将 PIPE 传入两侧
    ④ InitFreeSlots:
       - 是否仅 Consumer 侧调用一次（禁止 Producer/Consumer 两侧重复调用）
    ⑤ 条件分支:
       - 是否可能跳过 Set/Wait 导致对方死等（如提前 return）
    ⑥ TQue BUFFER_NUM:
       - 是否 ≥ 循环中同时持有的 queue tensor 数量 + 1

0-C.3 🛑 如果存在 AIC↔AIV 交叉依赖（如 AIC 等 AIV 的 SIG_P_READY，AIV 同时等 AIC 的 SIG_O_READY）:
    - 画出信号时序图，确认不存在循环等待（A 等 B 设 X，B 同时等 A 设 Y）
    - 确认 PRELAUNCH 延迟是否足够打破循环依赖
```

**门禁规则**：
- 触发条件满足但 0-C.1-0-C.3 未完成 → **禁止** Edit/Write 任何涉及同步的代码
- 禁止凭经验修改 CrossCore 参数而不查阅 sync-guide
- 此门禁在**每次修复迭代**中都需重新检查（不仅限于首次）

---

### 🛑 步骤 0-T: AscendC Tiling 路由（ascendc-tiling-design 已覆盖类别强制执行）

**触发条件**（属于以下任一类别即触发）：
- `matmul`：读取 `{output_dir}/model.py` 的 forward() 后发现 `torch.matmul`、`F.linear`、`@`、`torch.mm`、`torch.bmm` 等；或文件名/类名含 `Matmul`/`MatMul`/`BatchMatMul`/`GroupedMatmul`
- `flashattention`：文件名/类名含 `Attention`/`FlashAttention`/`SDPA`；forward 中出现 `scaled_dot_product_attention`、`softmax(Q @ K^T / sqrt(d)) @ V`
- `reduction`：文件名/类名含 `Reduce`/`Softmax`/`LayerNorm`/`RMSNorm`/`ArgMax`/`ArgMin`/`ReduceSum`/`ReduceMean`；forward 中出现 `torch.sum`、`torch.mean`、`torch.softmax`、`F.softmax`、`F.layer_norm`、`F.rms_norm`
- `sort`：文件名/类名含 `Sort`/`TopK`/`ArgSort`；forward 中出现 `torch.sort`、`torch.topk`
- `elementwise`：单输入/多输入逐元素运算且无广播；文件名/类名含 `ReLU`/`Sigmoid`/`SiLU`/`GELU`/`SwiGLU`/`Sin`/`Cos`/`Abs`/`Exp`
- `broadcast`：多输入逐元素运算且输入 shape 不同需广播对齐
- `conversion`：文件名/类名含 `Transpose`；forward 中出现 `tensor.transpose`、`tensor.permute`、`torch.transpose`
- `conv`：文件名/类名含 `Conv`/`Conv2d`/`Convolution`；forward 中出现 `nn.Conv1d`/`nn.Conv2d`/`nn.Conv3d`、`F.conv1d`/`F.conv2d`/`F.conv3d`

如果触发条件满足，必须逐个完成以下 checklist：

```
0-T.1 🛑 读取 ascendc-tiling-design 分类表与对应文档（必须，不可跳过）:
    Read .claude/skills/ascendc-tiling-design/SKILL.md 的算子分类表
    按类别 Read 对应的 patterns.md：
    - matmul → references/matmul/patterns.md
    - flashattention → references/flashattention/patterns.md
    - reduction → references/reduction/patterns.md
    - sort → references/sort/patterns.md
    - elementwise → references/elewise/patterns.md
    - broadcast → references/broadcast/patterns.md
    - conversion → references/conversion/patterns.md
    - conv → 本 skill 内路由（conv 不在 ascendc-tiling-design 分类表，改查本 skill references）：
        * Read .claude/skills/tilelang2ascend-translator/references/dsl2Ascendc_compute_conv.md（必须全文）
        * Read .claude/skills/tilelang2ascend-translator/references/dsl2Ascendc_compute_cube.md

0-T.2 🛑 按类别读取深度设计文档（必须，不可跳过）:
    - matmul:
        * Read .claude/skills/ascendc-tiling-design/references/matmul/ascendc-api-matmul-tiling.md
        * GroupedMatmul 额外 Read ascendc-api-gmm-tiling.md
        * Read ascendc-api-best-practices skill 中的 references/api-matmul.md
        * 在思考中确认目标平台 NpuArch（调用 npu-arch skill）：
          - DAV_2201 (A2/A3) → 使用 MatmulImpl + MatmulApiTiling 路径
          - DAV_3510 (A5) → 禁止使用 MatmulImpl，必须走 Blaze/tensor_api 路径
    - reduction → Read .claude/skills/ascendc-tiling-design/references/reduction/algorithms.md
    - sort → Read .claude/skills/ascendc-tiling-design/references/sort/alg-two-level-mrgsort.md
    - conv → Read workflows/templates/archive_tasks/conv2d_base/kernel/op_kernel/ 的
      im2col / wrepack / conv / epilogue 四类 kernel（至少 fp16 版本全文），
      确认其零标量（无 SetValue/GetValue 逐元素访问）、xtrans + fused 布局结构
    - 其他类别 → 视 patterns.md 中指引继续阅读

0-T.3 🛑 在思考中确认 Tiling 实现约束:
    通用约束（所有类别）：
    - 多核切分、UB 切分、Buffer 规划、分支覆盖必须与 ascendc-tiling-design 对应类别指南一致
    - 核数通过 platform_ascendc 或 npu-arch 动态获取，禁止硬编码
    - blockDim 与多核切分结果一致，kernel 内部通过 GetBlockIdx() 计算工作范围

    matmul 额外约束：
    - Host 侧调用 MatmulApiTiling::GetTiling(TCubeTiling&) 后，强制覆写 M/N/Ka/Kb
    - 计算 mTotalCnt / nTotalCnt / totalBlock
    - 若 totalBlock < aicCoreNum，执行多核切分扩展（保持 ALIGNED_H=16 对齐）
    - usedCoreNum = min(totalBlock, aicCoreNum)；blockDim 取该值
    - Kernel 入口必须有 if (GetBlockIdx() >= totalBlock) return; 越界守卫
    - Kernel 端按 blockIdx → (mIdx, nIdx) 计算偏移和尾块尺寸，调用 SetSingleShape
    - TilingHeader 通过 blocking copy 下发到 device（tilingHost.to(at::kPrivateUse1)）
    - BatchMatMul 必须填充 MultiBatchInfo
    - A 的转置优先使用 SetAType(..., isTrans=true) 或 SetTensorA(..., true)，禁止 host 侧额外 transpose+contiguous

    reduction 额外约束：
    - 明确归约轴、算法选择（tree/sequential/online）、精度保持策略
    - 大 shape 分块归约时考虑 workspace 累加

    sort 额外约束：
    - 多级归并 tiling、边界填充到 64 倍数、索引 buffer 规划

    elementwise / broadcast 额外约束：
    - 向量化 tile 大小、广播对齐、内存合并访问

    conversion 额外约束：
    - 转置轴映射、对齐填充、32B/512B 对齐搬运

    conv 额外约束（🛑 性能结构铁律，依据 dsl2Ascendc_compute_conv.md 第五章实测清单）：
    - GEMM 模式必须按 compute_conv.md 第二章三模式（normal / direct 1x1 /
      fused img2col LoadData3D 随路展开）显式选型；满足 fused 条件却退回 normal
      必须在思考中说明理由
    - 禁止默认标量逐元素 gather 实现 im2col（archive conv2d_base 全部 kernel 零
      SetValue/GetValue）；标量豁免仅限偶发小尾块（如 <32B k' 尾块）——
      主数据路径（im2col k' gather 主体、epilogue 主写出、wrepack 主存储）
      不属于豁免范围，占比 <5% 的偶发路径才可豁免且须注释说明理由
    - 复用/移植历史任务实现时，0-T.3 逐条核对不得省略；历史实现的性能结构缺陷
      （标量主路径、无缓存 wrepack、host 侧布局变换）不得以"精度已验证"为由继承
      ——性能结构与功能正确性独立判定
    - weight repack 结果是（weight 内容, shape, dtype）的确定函数，host 侧必须设计
      LRU 缓存（key = data_ptr + sizes + dtype + version，条目持有 weight 强引用），
      命中时跳过整个 wrepack 发射
    - dtype 适配（fp32 hi/lo fp16 拆分、bf16 等）时必须声明「保留的优化结构清单」：
      fused/normal 模式、零标量、TransDataTo5HD 批量转置、epilogue 跨行块处理
      逐项确认保留或说明丢弃理由
    - 声称"与 archive conv2d_base 同结构"时必须附证据（SetValue/GetValue grep
      计数、xtrans/fused 标志存在性对照），无证据视为未验证

0-T.4 🛑 查阅历史成功任务（可选但强烈建议）:
    - 若存在 workflows/templates/archive_tasks/ 下与本算子形态最接近的参考实现，优先阅读其 host/kernel 代码
```

**门禁规则**：
- 触发条件满足但 0-T.1-0-T.3 未完成 → **禁止**进入步骤 1 编写任何 kernel/ 代码
- 禁止凭记忆或经验跳过 `ascendc-tiling-design` 文档直接转译
- 此门禁在**每次修复迭代**中若涉及 Tiling 相关修改，都需重新检查

---

## 流程
执行以下各步骤前，必须先完成 **步骤 0-A（如触发）、步骤 0-B、步骤 0-C（如触发）和步骤 0-T（如触发）的全部查阅**，再开始实现、验证与迭代。

### 步骤 1: TileLang 转译成 AscendC

将 `{output_dir}/design/tile_level/` 下的 TileLang 设计转译为对应的 AscendC 实现。生成以下文件：
- `{output_dir}/kernel/op_host/<op_name>.cpp` — Host 端 (tiling 计算 + kernel launch)
- `{output_dir}/kernel/op_kernel/<op_name>.cpp` — Device 端 (CopyIn → Compute → CopyOut)
- `{output_dir}/kernel/ops.h` — 算子函数声明
- `{output_dir}/kernel/register.cpp` — torch.ops.npu.* 注册
- `{output_dir}/kernel/setup.py` — whl 打包配置
- `{output_dir}/kernel/CMakeLists.txt` — CMake 编译配置
- `{output_dir}/kernel/utils/kernel_common.h` — CopyTiling 等公共工具
参考文档：`.claude/skills/tilelang2ascend-translator/references/dsl2Ascendc.md`
**🛑 实施转译前必须先完成步骤 0-A（如触发）、步骤 0-B、步骤 0-C（如触发）和步骤 0-T（如触发）的全部查阅，然后阅读 `.claude/skills/tilelang2ascend-translator/references/TileLang-AscendC-API-Mapping.md` 逐一确认 API 映射。禁止跳过 Mapping 直接编写 AscendC 代码。**

**op_host/<op_name>.cpp** 模式：
- include `torch_kernel_helper.h` + `tiling/platform/platform_ascendc.h`
- 🛑 **核数获取（禁止硬编码）**：根据算子计算特征选择正确的 API，  **禁止 `constexpr int32_t = 20` 或 `min(20, ...)` 等硬编码**：
  ```cpp
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();

  // 纯 Vector 计算（Norm/激活函数/逐元素等）→ GetCoreNumAiv()
  int32_t totalCoreNum = platform->GetCoreNumAiv();

  // 纯 Cube 计算（MatMul/矩阵乘）→ GetCoreNumAic()
  int32_t totalCoreNum = platform->GetCoreNumAic();

  // Cube+Vector 融合（Attention/CV融合等）→ CalcTschBlockDim()
  int32_t totalCoreNum = platform->CalcTschBlockDim(
      sliceNum, platform->GetCoreNumAic(), platform->GetCoreNumAiv());
  ```
- 使用平台 API 获取 `GetCoreMemSize(UB)`
- Block 级 tiling: Cache Line 512B 对齐，formerNum/formerLength/tailNum/tailLength
- UB 级 tiling: bufferCoefficient 推导，32B 对齐 tileLength
- **🛑 EXEC_KERNEL_CMD 传参铁律**: 所有 tiling 参数必须是**独立标量左值**，**禁止传 struct 指针**。参照 `workflows/templates/archive_tasks/rms_norm/kernel/op_host/rms_norm.cpp` 的正确模式
- blockDim = usedCoreNum（多核统一分发），kernel 内部通过 `GetBlockIdx()` 计算工作范围
- `EXEC_KERNEL_CMD` 所有参数必须为**左值**（具名变量），禁止传入临时变量/右值/字面量。`double` 先转 `float` 局部变量，`bool` 用 `int64_t` 替代，表达式先赋给局部变量再传入

**MatMul 类算子额外要求**（当触发步骤 0-T 且类别为 matmul 时）：
- 必须包含 `matmul_trans_a_tiling.h` 风格的 TilingHeader，内含 `TCubeTiling cubeTiling` 和派生计数字段
- 使用 `AscendC::tiling::TCubeTiling` 类型，禁止错误 `reinterpret_cast` 为 `optiling::TCubeTiling&`
- 工作区大小使用 `MatmulGetTmpBufSizeV2(tilingPtr->cubeTiling, bufSize)` 查询
- CMakeLists.txt 需链接 `torch torch_python c10`，确保 .so 加载符号解析成功

**op_kernel/<op_name>.cpp** 模式：
- template class `Kernel<OpName>` 含 Init/Process/CopyIn/Compute/CopyOut
- BUFFER_NUM = 2 (double buffer)；如算子需要在循环中同时持有多个 queue tensor，需相应增大 BUFFER_NUM
- DataCopyPad 用于 GM↔UB 搬运
- FP16/BF16 升精度到 FP32 计算
- 整核/尾核偏移和尾块对齐处理

**MatMul 类算子额外要求**（当触发步骤 0-T 且类别为 matmul 时）：
- 使用 `AscendC::MatmulType<TPosition::GM, CubeFormat::ND, T, isTrans>` 声明 A/B 类型
- `Init()` 中调用 `mm_.Init(&header_.cubeTiling, pipe)` 和 `mm_.SetHF32(false, 0)`
- `Process()` 中按 `blockIdx → (mIdx, nIdx)` 分发，计算 `curSingleM / curSingleN`
- 调用 `mm_.SetSingleShape(curSingleM, curSingleN, header_.K)`
- 调用 `mm_.SetTensorA(aGm_[offsetA], transA)` 和 `mm_.SetTensorB(bGm_[offsetB], transB)`
- 调用 `mm_.IterateAll(cGm_[offsetC], 0)`，之后视需要 `PipeBarrier<PIPE_ALL>()`

   **ops.h** 模式：
   ```cpp
   namespace ascend_kernel {
   at::Tensor <op_name>(<参数列表>);
   }
   ```

   **register.cpp** 模式：
   ```cpp
   #include "ops.h"
   #include <torch/library.h>

   TORCH_LIBRARY_FRAGMENT(npu, m) {
       m.def("<op_name>(<schema>) -> Tensor");
   }
   TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
       m.impl("<op_name>", TORCH_FN(ascend_kernel::<op_name>));
   }
   ```

### 🛑 步骤 1.5: 标量残留自检（每次生成/修改 op_kernel 后、编译前强制执行）

`op_kernel/<op_name>.cpp` 生成或修改后，**必须**对照 `.claude/skills/tilelang2ascend-translator/references/scalar-to-vector.md` 末尾的「综合检查清单」逐项执行 grep 检查（共 7 项），确认没有标量反模式残留：

```
1.5.1 执行检查清单 7 条 grep（标量归约 / raw pointer / 嵌套比较交换 /
      逐元素 gather-scatter / 逐行 GetValue+Muls / 逐元素 SetValue-GetValue / 其余 for 循环）

1.5.2 命中任一检查项 → 按 scalar-to-vector.md 中对应编号的模式修复：
      模式 1 标量归约 → ReduceSum/WholeReduceSum
      模式 2 标量逐元素 → Adds/Muls 向量链
      模式 3 标量排序 → Sort32/MrgSort
      模式 4 逐元素标量算术 → 零拷贝子张量切片 + 批量向量运算
      模式 5 混合调度（唯一豁免：单 tile 元素数 < 128，须在注释中说明理由）
      模式 6 逐行 Muls → BroadCast + 批量向量
      模式 7 逐元素 SetValue/GetValue → GatherMask

1.5.3 修复涉及新 API（Sort/GatherMask/BroadCast 等）时，必须回到步骤 0-B.3
      查阅对应 API 文档确认签名与 work buffer 需求（GetSortTmpSize 等），
      禁止凭记忆使用

1.5.4 全部检查项无命中（或命中项已修复/已注释豁免理由）→ 进入步骤 2
```

**门禁规则**：
- 未完成 1.5.1-1.5.4 → **禁止**进入步骤 2 编译验证
- 每次修复迭代中修改了 `op_kernel/*.cpp` 后，必须重新执行本自检
- 尾块处理、BLOCK 级循环等合法 for 循环不属于残留，但需在思考中确认其不可向量化

### 步骤 2: 编写 model_new_ascendc.py + 编译验证

编写 `{output_dir}/model_new_ascendc.py`，采用**双路径加载**模式：
- 优先 `import <op_name>_ext`（whl 安装后自动触发 TORCH_LIBRARY 注册）
- 失败回退 `torch.ops.load_library()` 直加载 `kernel/build/<op_name>_ext*.so`
- forward() 中调用 `torch.ops.npu.<op_name>(...)`

示例：
```python
import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_LIB_PATTERN = str(_KERNEL_BUILD / "<op_name>_ext*")

try:
    import <op_name>_ext  # noqa: F401 — whl path
except ImportError:
    # Fallback: direct .so loading
    if _LIB_PATTERN not in "".join(sys.path):
        import glob as _glob
        _libs = _glob.glob(_LIB_PATTERN)
        if _libs:
            torch.ops.load_library(_libs[0])

class ModelNew(nn.Module):
    def forward(self, x, ...):
        ...
        return torch.ops.npu.<op_name>(x, ...)
```

**禁止**在 model_new_ascendc.py 中使用 `torch.*` / `F.*` 计算算子。
然后调用 `.claude/skills/tilelang2ascend-translator/scripts/evaluate_ascendc.sh {output_dir}` 编译并验证（内部 cmake + make + whl 安装）。

---

### 步骤 3: 错误修复迭代

迭代上限为 **5 次**（与 ascend-kernel-developer Phase 4.5A 对齐）。每次修复前必须执行以下步骤：

#### 🛑 3.0 修复前查阅（每次修复强制执行，不可跳过）

**根据错误类型，查阅对应的 asc-devkit 文档或历史案例：**

🛑 **跨核同步专项门禁（涉及 CrossCore / WorkspaceQueue / MIX_AIC / 全零输出时强制执行）**:
- **必须先完成步骤 0-C 的全部 checklist**（含读取 ascendc-sync-guide.md 全文 + 逐条确认 6 项 checkpoint）
- 未完成步骤 0-C → **禁止** Edit/Write 任何 CrossCore/WorkspaceQueue 相关代码

| 错误类型 | 必须查阅 |
|---------|---------|
| **编译错误: API 签名不匹配** | `asc-devkit/docs/api/Ascend-C-API列表.md` → 定位 API → 查阅该 API 的独立 .md 文档确认正确签名 |
| **编译错误: 类型不匹配** | `asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/类型转换/Cast.md` 确认 dtype 支持矩阵 |
| **编译错误: GlobalTensor/LocalTensor** | `asc-devkit/docs/api/SIMD-API/基础数据结构/` 下对应简介.md |
| **运行时 vector core exception / UB 违例 / all-zero output** | ① 🛑 **优先执行步骤 0-C** 完成 sync checklist<br>② `asc-devkit/docs/guide/算子实践参考/.../TBuf的使用.md` 检查 buffer 大小<br>③ `workflows/templates/archive_tasks/rms_norm/` 对比 EXEC_KERNEL_CMD 传参模式<br>④ 检查是否有 struct 指针被传给 `EXEC_KERNEL_CMD`（常见根因） |
| **运行时 hang/死锁 / 跨核数据不流通** | 🛑 **必须先执行步骤 0-C**（含读取 ascendc-sync-guide.md 全文 + 6 项 checkpoint），再逐项排查 |
| **运行时 vector core timeout (507034)** | 🛑 这是硬件级别的 core 挂起错误。按顺序排查:<br>① **work buffer 尺寸**: 检查所有 API 的 work buffer (ReduceSum/Cos/Sin/Broadcast) 是否通过 GetXxxMaxMinTmpSize 正确计算 — 硬编码不足是最常见根因<br>② **Buffer 总溢出**: 计算所有 InitBuffer 分配的总 UB 字节数，确认不超过 GetCoreMemSize(UB)<br>③ **PipeBarrier 配对**: 每个 GM→UB (MTE2) 后必须有 PIPE_MTE2 barrier; 每个 V 计算块结束后必须有 PIPE_V barrier; 每个 UB→GM (MTE3) 前必须有 PIPE_V barrier<br>④ **循环边界**: 检查所有循环的边界类型一致性 (int32_t vs int64_t)，确认不会因类型不匹配导致死循环<br>⑤ **隔离法**: 将 kernel 逐步简化为 identity copy，每次恢复一个操作，定位触发 timeout 的具体 API<br>⑥ **参考历史**: 查阅 workflows/templates/archive_tasks/ 中相似规模的融合算子，对比 work buffer 计算方式 |
| **精度不匹配 (MERE/MARE 超标)** | 调用 `ascendc-precision-debug` skill（见步骤 4） |

**⚠️ 在查阅完成并在思考中列出根因分析之前，禁止 Edit/Write 任何 kernel 代码。**

#### 3.1 分析错误输出，结合查阅结论确定根因
#### 3.2 修改 kernel/ 下的代码
#### 3.3 运行 evaluate_ascendc.sh
#### 3.4 如果 PASS → 完成，退出
#### 3.5 如果 FAIL 且迭代次数 < 5 → 回到 3.0
#### 3.6 如果 FAIL 且达到 5 次 → 进入步骤 4 (精度 skill 深度诊断)

---

### 步骤 4: 精度 Skill 深度诊断（步骤 3 耗尽后）

当步骤 3 的 5 次迭代无法解决时，按以下顺序调用精度 skill：

```
4.1 🛑 调用 Skill "ascendc-precision-debug"，传入 output_dir + 错误输出
    等待返回诊断结论和修复建议。此步骤不可跳过。

4.2 根据建议修改 kernel/ 代码，运行 evaluate_ascendc.sh

4.3 如果仍 FAIL 且连续失败 < 7 次 → 回到 4.1

4.4 如果 7 次后仍 FAIL → 
    🛑 调用 Skill "ascendc-precision-tuning"，传入 output_dir + 错误输出
    等待返回取证→审计→修复分析。此步骤不可跳过。

4.5 根据建议修改 kernel/ 代码，运行 evaluate_ascendc.sh

4.6 如果仍 FAIL 且连续失败 < 5 次 → 回到 4.4

4.7 如果所有步骤耗尽仍 FAIL → 报告当前状态，记录 trace
```

## 精度验证标准

**五类决策矩阵**（由 `verification_ascendc.py` 自动判定）：

| 类别 | 触发条件 | 判定标准 |
|---|---|---|
| 非计算类 | `ASCENDC_NON_COMPUTE=1` | view-as-int 二进制完全一致（含 NaN bit pattern） |
| bool 输出 | 输出 dtype 为 bool | `torch.equal` 严格相等 |
| 整数计算类 | 输入最高精度为 int 且输出为 int | `|actual − golden| == 0` |
| 量化计算类 | 输入最高精度为 float 且输出为 int | `|actual − golden| <= 1` |
| 浮点计算类 | 输出为 float 类型 | 三项 AND 判定（max_error_cap + matched_ratio ≥ 0.9 + MERE < rel_threshold） |

- **输入类型自动推断**：从实际输入 tensor 中取最高精度 dtype 分类为 float / int / no_tensor。
- **浮点三项判定**：① 100% 元素满足 `|diff| <= atol + rtol * |golden|`；② 分桶 matched_ratio ≥ 0.9（小值域绝对误差 / 正常域相对误差）；③ MERE（均值相对误差）< 阈值。三项全部通过才算通过。
