# 失败模式（OKF）

本目录存放进化过程中发现的失败模式，供后续任务避坑和反查。

frontmatter 关键字段：
- `schema_version: okf.v1`
- `kind: failure_pattern`
- `phenomenon`: `compile_error` | `precision_failure` | `perf_regression` | `build_failure`
- `severity`: `blocking` | `major` | `minor`
- `status`: `stub` | `active` | `deprecated` | `fixed`
- `failed_combination`: 导致失败的策略组合（如 `[P22, A1]`）
- `operator_type`: 出现过的算子族
- `occurrences`: 命中次数（由脚本维护）

条目正文结构：
- **Symptom**: 失败现象
- **Root cause class**: 根因归类
- **Failed combination**: 失败的策略组合
- **Resolution**: 如何解决或绕过
- **Evidence**: 来源算子/轮次/节点
