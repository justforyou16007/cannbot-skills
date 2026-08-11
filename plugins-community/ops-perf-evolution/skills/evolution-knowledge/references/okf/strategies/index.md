# X 系列进化发现策略（OKF）

本目录存放进化过程中自动发现的策略卡片，与 `skills/evolution-strategies/references/discovered/` 双轨保留。

每个文件对应一个 `disc_X{n}` 策略，frontmatter 包含：
- `schema_version: okf.v1`
- `kind: discovered_strategy`
- `confidence`: `single_op` | `cross_op_verified`
- `status`: `stub` | `active` | `deprecated` | `retracted`
- `op_families`: 来自 INDEX.json vocabulary 的算子族列表
- `bottlenecks`: 来自 INDEX.json vocabulary 的瓶颈标签列表
- `tags`: 自由标签

检索入口：`evolution-knowledge` skill 的查询脚本（`skills/evolution-knowledge/scripts/query.py`）。
写入入口：`evolution-knowledge` skill 的维护协议（`skills/evolution-knowledge/scripts/migrate_discovered_to_okf.py`）。
