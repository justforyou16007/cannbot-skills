# ops-perf-evolution OKF 知识库

> Open Knowledge Format (OKF) v1 知识库，沉淀 AscendC 算子性能进化过程中发现的策略、方案与失败模式。
> 与 `skills/evolution-strategies/references/discovered/` 双轨保留：原目录为只读兼容层，本目录为增强 OKF 版本（带 frontmatter 结构化字段）。

## 目录结构

```
okf/
├── index.md                         # 本文件
├── strategies/
│   ├── index.md
│   └── discovered/                  # X 系列进化发现策略（disc_X*.md）
└── runbooks/
    ├── index.md
    ├── proven-solutions/            # 跨算子可复用的完整优化方案
    └── failure-patterns/            # 编译/精度/性能失败模式
```

## 读取优先级

运行期读取顺序：
1. `okf/strategies/discovered/`（X 系列，进化发现策略）
2. `okf/runbooks/proven-solutions/`（已验证跨算子方案）
3. `okf/runbooks/failure-patterns/`（失败模式，用于 anti 提示）
4. 原 `a3/` 参考文档（硬件/优化模式/算法洞察）

## 写入协议

所有写入由 `evolution-knowledge-maintain` skill 代理完成，遵循：
- 先查重（按 kind + key 字段），命中则 update 而非 new
- 增量更新：在既有卡上补充，不整体覆盖
- 矛盾不静默覆盖：并存 + `> ⚠️ 矛盾标记` + 日期/来源
- 候选模式默认 `status: stub`，跨算子验证后提升为 `active`

## 索引

OKF 卡片由 `skills/evolution-strategies/scripts/generate_index_json.py` 扫描并合并进 `skills/evolution-strategies/references/INDEX.json`，支持按 `op_families`、`bottlenecks`、`phenomenon`、`tags` 结构化检索。
