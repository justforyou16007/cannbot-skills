---
name: tilelang2ascend-kernel-generalizer
description: 将本轮已通过 evaluate_tilelang.sh 验证的 TileLang 输出泛化为标准 AscendC 算子工程。仅在复杂算子的 TileLang 验证明确 PASS 后使用；不得修改或要求用户修改 TileLang 编译器。
---

# TileLang 验证后泛化 Skill

将已验证的 TileLang 输出作为实现基线，改写为运行期参数化的 AscendC 算子工程。目标是复用已验证语义，而不是修复、替换或绕过 TileLang 编译器。

## 前置门禁

仅在以下条件全部满足时执行：

1. 当前任务为复杂算子，且本轮 `evaluate_tilelang.sh {output_dir}` 的结果明确为 PASS。
2. `{output_dir}/model_new_tilelang.py` 和 `{output_dir}/design/tile_level/` 存在。
3. 能够通过当前任务中的 TileLang 运行结果或其已生成的任务产物取得对应的 AscendC 源码；不得读取外部缓存、修改编译器、打补丁或要求用户修改其环境。

任一条件不满足时，写明 `GENERALIZATION_NOT_AVAILABLE` 与原因并返回，由上层改走 `tilelang2ascend-translator`。不得把框架不支持、验证失败或仅代理验证视为 PASS。

## 工作流

1. 保留验证证据。记录 TileLang 验证命令、退出状态和所用输入；后续只使用与该 PASS 对应的生成源码。
2. 从生成源码建立对照表。列出 shape 常量、GM 偏移、buffer 容量、grid、dtype 分支及同步点，并与 `design/tile_level/` 对照确认语义。
3. 进行最小泛化改写。
   - 仅将 shape 烘焙值、地址偏移和 buffer 容量替换为运行期参数或受限 dispatch。
   - 保留运算顺序、dtype 分支、同步关系、核间协议和尾块保护；不以“简化”为由重排它们。
   - 使用现有 `tilelang2ascend-operator-project-init` 模板产出标准 `kernel/op_host/`、`kernel/op_kernel/`、注册文件和 `model_new_ascendc.py`。
   - 只修改 `{output_dir}/`；禁止修改 TileLang 编译器、其安装目录或评测脚本。
4. 验证交付物。执行既有 `validate_ascendc_impl.py` 和 `evaluate_ascendc.sh`；在 `{output_dir}/trace.md` 记录泛化来源、运行期公式和验证结果。
5. 按既有 Phase 4 A/D 重试规则修复泛化产物。修复范围仅限 `{output_dir}/kernel/` 与 wrapper，不得回写 TileLang 编译器。

## 输出

- `{output_dir}/kernel/`：可构建的标准 AscendC 算子工程。
- `{output_dir}/model_new_ascendc.py`：仅调用自定义 AscendC 算子。
- `{output_dir}/trace.md`：包含 PASS 证据、泛化改动和 AscendC 验证结果。
