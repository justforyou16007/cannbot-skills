# 跨算子已验证方案（OKF）

本目录存放经多算子验证的完整优化方案，每个方案一文件。

frontmatter 关键字段：
- `schema_version: okf.v1`
- `kind: proven_solution`
- `operator_type`: elementwise | reduction | attention | matmul | ...
- `confidence`: `single_op` | `cross_op_verified`
- `status`: `active` | `deprecated` | `retracted`
- `strategy_ids`: 关联的 P/D/A/R/X 系列策略 ID

条目正文遵循 `a3/proven_solutions/INDEX.md` 头部模板：Source / Speedup / Technique description / Key code change / Applicability conditions。
