---
name: tilelang2ascend-kernel-generalizer
description: 将本轮通过 evaluate_tilelang.sh 的 TileLang 编译输出转译为运行期参数化的标准 AscendC 算子工程。仅用于复杂算子的 TileLang 验证 PASS 路径；不修改或要求用户修改 TileLang 编译器。
---

# TileLang 编译输出泛化转译

将已经通过 TileLang 验证的生成源码作为单 shape 语义基线，转译为可构建、可验证的 AscendC 标准算子工程。只泛化已定位的 shape 相关内容；不重新设计计算逻辑，也不处理 TileLang 编译器问题。

## 路由与边界

仅在以下条件均满足时执行：

1. 当前任务是复杂算子，且**本轮** `evaluate_tilelang.sh {output_dir}` 返回 PASS。
2. `{output_dir}/model_new_tilelang.py` 与 `{output_dir}/design/tile_level/` 完整存在。
3. 能从当前任务的 TileLang 运行结果或任务产物取得该 PASS 所对应的 AscendC 生成源码。

以下情况返回 `GENERALIZATION_NOT_AVAILABLE`，由 Agent 使用既有 `tilelang2ascend-translator`：TileLang 验证失败、仅有代理验证、框架不可用、生成源码不可获得，或源码与 PASS 用例不对应。

禁止：

- 修改、重编、打补丁或要求用户修改 TileLang 编译器、安装目录、缓存和评测脚本。
- 读取任务目录外的编译缓存作为生成源码来源。
- 用未验证的新算法、重排计算顺序或删除同步来“顺便优化”。

## 输入基线

开始改写前，完成并记录以下证据：

1. TileLang PASS 的命令、退出状态、用例文件与时间。
2. 生成源码与通过用例的对应关系；将源码副本保留在 `{output_dir}/` 下，避免后续依赖外部缓存。
3. `model.py`、`design/tile_level/` 与生成源码的输入/输出、dtype、shape 维度和 kernel 入口对照。

不能建立以上对应关系时停止泛化并返回 `GENERALIZATION_NOT_AVAILABLE`，不得猜测源码语义。

## 泛化步骤

### 1. 冻结单 shape 基线

先以原始 PASS shape 构建和验证一次生成源码对应的 AscendC 工程。该基线不通过时，不得开始参数化；检查源码采集与工程封装是否失真。

### 2. 建立常量清单

逐项标记生成源码中的常量，并记录原值、来源、运行期公式和验证 shape：

| 类型 | 处理原则 |
| --- | --- |
| 输入 shape、stride、GM 偏移、任务总数 | 改为由 host 输入和运行期 shape 推导 |
| buffer 容量、对齐与 workspace 大小 | 仅在 AscendC API 与官方文档允许时改为运行期公式；保留对齐与上限检查 |
| grid、每核任务数、尾块长度 | 由运行期 shape 和硬件查询推导，并保留越界守卫 |
| dtype 特有布局或入口 | 保持独立分支；不得合并为未经验证的统一布局 |
| 算法常量、同步 flag、barrier、运算顺序 | 保持原样，不属于泛化对象 |

调用任何 AscendC API 前，查阅 `asc-devkit/docs/api/` 中对应官方文档；不得猜测运行期 buffer、tiling 或 launch API 的行为。

### 3. 最小改写

1. 每次只泛化一个常量类别，并在改动后与单 shape 基线做 diff。
2. 先完成 host 侧 shape/stride/workspace 公式与范围检查，再修改 device 侧索引、buffer 和尾块逻辑。
3. 保留生成源码中的计算顺序、dtype 分支、同步关系、核间协议和错误保护；这些不是清理目标。
4. 使用 `tilelang2ascend-operator-project-init` 的既有结构产出 `kernel/op_host/`、`kernel/op_kernel/`、注册文件和 `model_new_ascendc.py`。
5. 只修改 `{output_dir}/`。若发现必须改 TileLang 编译器才能继续，立即返回 `GENERALIZATION_NOT_AVAILABLE`。

## 验证门禁

按以下顺序执行，任一步失败均不得跳过：

1. 原始 PASS shape：验证泛化版与单 shape 基线一致。
2. 精简用例：运行 `validate_ascendc_impl.py` 与 `evaluate_ascendc.sh`。
3. 边界用例：覆盖每个运行期公式的最小值、最大值、非对齐尾块和全部 dtype 分支；仅使用任务已定义的合法 shape 域。
4. 全量用例：恢复 `.json.bak` 后执行既有 AscendC 验证。

将每一轮输入范围、修改的常量类别和验证结果写入 `{output_dir}/trace.md`。A/D 类错误沿用 Phase 4 的现有重试规则，修复范围限于 `{output_dir}/kernel/` 与 wrapper。

## 交付物

- `{output_dir}/kernel/`：标准 AscendC 算子工程。
- `{output_dir}/model_new_ascendc.py`：仅调用自定义 AscendC 算子。
- `{output_dir}/trace.md`：PASS 证据、源码对应关系、常量清单、运行期公式与验证矩阵。
