---
name: tilelang2ascend-kernel-generalizer
description: 将本轮通过 evaluate_tilelang.sh 的 TileLang 编译输出转译为运行期参数化的标准 AscendC 算子工程。仅用于复杂算子的 TileLang 验证 PASS 路径。
---

# TileLang 编译输出泛化转译

将已经通过 TileLang 验证的生成源码作为单 shape 语义基线，依次完成运行期参数化和 ABI 工程化，交付可构建、可验证的标准 AscendC 算子工程。只泛化已定位的 shape 相关内容，不重新设计计算逻辑。

## 路由与边界

仅在以下条件均满足时执行：

1. 当前任务是复杂算子，且**本轮** `evaluate_tilelang.sh {output_dir}` 返回 PASS。
2. `{output_dir}/model_new_tilelang.py` 与 `{output_dir}/design/tile_level/` 完整存在。
3. 能从当前任务的 TileLang 运行结果或任务产物取得该 PASS 所对应的 AscendC 生成源码。

以下情况返回 `GENERALIZATION_NOT_AVAILABLE`，由 Agent 使用既有 `tilelang2ascend-translator`：TileLang 验证失败、仅有代理验证、框架不可用、生成源码不可获得，或源码与 PASS 用例不对应。

禁止：

- 读取任务目录外的编译缓存作为生成源码来源。
- 用未验证的新算法、重排计算顺序或删除同步来“顺便优化”。
- 在 ABI 包装阶段以修复工程问题为名修改设备侧计算逻辑；设备侧逻辑问题必须回到本 Skill 的泛化验证阶段处理。

## 输入基线

开始改写前，完成并记录以下证据：

1. TileLang PASS 的命令、退出状态、用例文件与时间。
2. 生成源码与通过用例的对应关系；将源码副本保留在 `{output_dir}/` 下，避免后续依赖外部缓存。
3. `model.py`、`design/tile_level/` 与生成源码的输入/输出、dtype、shape 维度和 kernel 入口对照。

不能建立以上对应关系时停止泛化并返回 `GENERALIZATION_NOT_AVAILABLE`，不得猜测源码语义。

## 执行闭环

### 1. 冻结单 shape 基线

先以原始 PASS shape 构建并运行取得的生成源码。该基线不通过时，不得开始参数化；检查源码采集与封装是否失真。

建立最小可追溯记录：源码文件校验值、PASS case、入口签名、输入/输出 dtype 与布局。后续每次改写均以此记录为基线做 diff。

### 2. 提取运行期公式并最小泛化

逐项标记生成源码中的常量，并记录原值、来源、运行期公式和验证 shape：

| 类型 | 处理原则 |
| --- | --- |
| 输入 shape、stride、GM 偏移、任务总数 | 改为由 host 输入和运行期 shape 推导 |
| buffer 容量、对齐与 workspace 大小 | 仅在 AscendC API 与官方文档允许时改为运行期公式；保留对齐与上限检查 |
| grid、每核任务数、尾块长度 | 由运行期 shape 和硬件查询推导，并保留越界守卫 |
| dtype 特有布局或入口 | 保持独立分支；不得合并为未经验证的统一布局 |
| 算法常量、同步 flag、barrier、运算顺序 | 保持原样，不属于泛化对象 |

新增、替换或不确定的 AscendC API 用法，先查 `asc-devkit/docs/api/` 对应文档或已验证官方示例；纯粹把已有常量替换为运行期参数时，不额外引入 API。

按以下顺序改写，每次只处理一个常量类别：

1. 先由 host 输入推导 shape、stride、GM 偏移、grid、每核任务和 workspace；公式同时写入报告。
2. 再替换 device 侧索引、buffer 容量和尾块保护；保留原有对齐、上限、dtype 分支、同步关系、核间协议和计算顺序。
3. 先以 tilelang 风格 call ABI 产出 `{output_dir}/ascendc/<op>.cpp`，验证原 PASS shape 与精简用例；不得直接跳到多文件工程。
4. buffer 重叠、offset 衔接或数值全错时，优先对照同 shape 的静态生成产物：逐个替换动态包装，随后按 stage 二分定位。不得从零另写 mini 工程复现。

### 3. ABI 工程化交付

泛化单文件通过精简用例后，将其转换为 `{output_dir}/kernel/` 标准工程。此阶段遵循 `tilelang2ascend-abi-packaging` 的交付约定，且只做 ABI 适配与结构搬运。

1. **设备侧**：将运行期 shape/标量收拢为对齐的 tiling 结构体；删除 tilelang JIT dispatcher 残留；按 dtype 提供薄入口和 launch 包装。shape 不是新的多入口轴，仍由运行期 tiling 处理。
2. **工程拆分**：按 `kernel_common`、cube/vector stage、kernel 编排、入口/launch 头职责拆分。拆分只能移动连续代码与提升局部状态到 Context，不能改变数据流或同步。
3. **host 与注册**：host 负责 shape 校验、tiling、每核 workspace、dtype 分发和左值形式的 launch 参数；`ops.h` 与 `register.cpp` 仅声明和注册 `torch.ops.npu.<op>`。输出 dtype 按算子契约显式指定，不能盲从输入 options。
4. **构建与封装**：使用 direct-launch 的既有工程模板；所有设备入口保持在一个设备编译单元。依赖随工程 vendor，交付构建不得依赖 TileLang 工作区路径。
5. **Python wrapper**：`model_new_ascendc.py` 加载扩展并调用注册算子，不以 torch 算子回退实现计算。

在工程化前恢复全量 JSON 用例；Phase 2 的精简集只用于泛化迭代，不可替代最终验收。

### 4. ABI 交付检查表

**设备侧检查**

- tiling 结构体字段、对齐、host 写入顺序与 device 读取顺序完全一致；运行期参数不得重新隐式编码为 shape 专用常量。
- 每个入口明确匹配其实现类型的 `KERNEL_TASK_TYPE_DEFAULT`；纯 Vector、纯 Cube 与 CV 融合不可混用。
- launch 包装传递的 block 数、stream、GM 地址和 tiling 指针均来自具名左值；保留生成源码已有的同步和 workspace 偏移关系。
- 设备代码多入口只允许以 dtype 为轴。不得因 shape、尾块或布局分支复制新的静态 kernel。

**host 与注册检查**

- host 的 block 数、对齐长度、尾块与 workspace 步长，和 device 内的任务划分、offset 公式逐项对照；workspace 按实际使用的核数分配。
- 输出张量 dtype、shape、layout 以算子契约为准，不能通过输入 `options()` 间接继承错误 dtype。
- `register.cpp` 只含 schema 与 `TORCH_LIBRARY_FRAGMENT` / `TORCH_LIBRARY_IMPL` 注册；host 实现在 `op_host/`，设备入口在 `op_kernel/`。
- wrapper 仅加载扩展和调用 `torch.ops.npu.<op>`；调用签名、属性类型和 dtype 分发必须与注册 schema 一致。

**构建与依赖检查**

- CMake 显式设置非空的构建类型；设备代码采用 direct-launch 编译，且全部 `__global__` 入口在同一设备编译单元。
- 构建 flags、AscendC 头文件和链接库以项目既有 direct-launch 模板为准；新增或不确定的参数先查官方文档或已验证模板。
- 交付前将所需 `tl_templates` / catlass 等依赖 vendor 到工程内，最终构建只引用工程自身与 CANN 环境。

### 5. 失败定位与阶段回退

| 现象 | 优先检查 | 处理位置 |
| --- | --- | --- |
| 编译或链接失败 | tiling 字段、入口/launch 声明、direct-launch 构建参数 | ABI 工程化 |
| 输出全零、部分未写 | 入口是否实际启动、block/stream、GM 参数与 launch 符号 | ABI 工程化 |
| 同一 shape 数值全错 | buffer 重叠、workspace 步长、GM offset 和动态包装替换点 | 最小泛化 |
| 仅尾块或边界失败 | 对齐、余数分支、grid/每核任务和越界守卫 | 最小泛化 |
| ABI 工程全量回归失败 | 先将泛化单文件原样置入工程对照，区分包装层与设备逻辑 | 先 ABI，再按结果回退 |

每个问题按“现象、定位、根因、改动、复验”记入 `GENERALIZE_DEBUG_LOG.md`。ABI 层已定位为无责后，才允许回到运行期公式或设备索引继续迭代。

## 验证门禁

按以下顺序执行，任一步失败均不得跳过：

1. 原始 PASS shape：验证泛化版与单 shape 基线一致。
2. 精简用例：先运行 `validate_ascendc_impl.py` 与 `evaluate_ascendc.sh`，逐 case 隔离失败。
3. 边界用例：覆盖每个运行期公式的最小值、最大值、非对齐尾块和全部 dtype 分支；仅使用任务已定义的合法 shape 域。
4. ABI 工程：设备代码与泛化单文件 diff 仅允许 tiling ABI、JIT 残留清理和结构拆分；然后 clean build。
5. 全量用例：恢复 `.json.bak` 后，以 `ASCENDC_CLEAN_BUILD=1` 运行既有 `evaluate_ascendc.sh`。

将每一轮输入范围、修改的常量类别和验证结果写入 `{output_dir}/ascendc/GENERALIZE_DEBUG_LOG.md` 与 `trace.md`。构建、host、注册问题在 ABI 阶段修复；任何 device 数值问题回到步骤 2 重走验证。A/D 类错误沿用 Phase 4 的现有重试规则。

## 交付物

- `{output_dir}/ascendc/<op>.cpp`：已验证的泛化单文件基线。
- `{output_dir}/ascendc/GENERALIZE_REPORT.md`：PASS 证据、常量清单、运行期公式、容量核算和验证矩阵。
- `{output_dir}/ascendc/GENERALIZE_DEBUG_LOG.md`：问题、定位、改动和复验记录。
- `{output_dir}/kernel/`：标准 AscendC 算子工程。
- `{output_dir}/model_new_ascendc.py`：仅调用自定义 AscendC 算子。
- `{output_dir}/trace.md`：路由、PASS 基线、工程化与最终验收结果。
