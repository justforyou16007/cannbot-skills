---
source_key: evolution-strategies#card/P108_matmul_platform_routing
id: P108
title: Matmul Platform Routing
doc_path: cards/P108_matmul_platform_routing.md
bottlenecks: [portability]
op_families: [matmul]
complexity: L1
triggers:
  - Kernel needs to run on multiple NPU generations
  - NpuArch is DAV_3510 (Ascend 950) or a future architecture not covered by this guide
  - Existing tiling/host code assumes DAV_2201 (Atlas A2/A3) only
conflicts_with: []
synergizes_with: [P46]
requires: []
has_preconditions: false
has_playbook: false
references:
  - ops/ascendc-tiling-design/references/matmul/ascendc-api-matmul-tiling.md
---

# P108: Matmul 平台路由（Platform Routing）

## 核心思想

Ascend C Matmul Tiling 设计强烈依赖硬件特征（`NpuArch`、`GetCoreNumAic`、L1/L0 容量、Cube 数据通路）。`DAV_2201`（Atlas A2 / A3）适用 `MatmulImpl` + `MatmulApiTiling` 路径；而 `DAV_3510`（Ascend 950）应走 Blaze / tensor_api 路径。P108 要求 Host 在编译期或运行期按 `NpuArch` 路由到对应的 Tiling / Kernel 实现，避免把 A2/A3 的 `TCubeTiling` 假设套用到不兼容平台。

## 代码骨架

```cpp
// === Host 端平台路由示例 ===
auto platform = PlatformAscendC(context);
auto socVersion = platform.GetSocVersion(); // 或从 attr 读取

if (socVersion == "Ascend910B" || socVersion == "Ascend910_93") {
    // DAV_2201 路径
    MatmulApiTiling::GetTiling(cubeTiling);
    // 应用 P104/P105/P106/P107
} else if (socVersion == "Ascend950") {
    // DAV_3510 路径：走 Blaze / tensor_api
    // 不应用本 guide 中的 TCubeTiling 假设
} else {
    // 未知平台：兜底报错或走通用路径
}
```

## 关键修改点

1. **版本来源**：优先从 `PlatformAscendC` 获取；若 CANN 版本差异大，可用 `ascendc-docs-search` 确认字段名。
2. **条件编译 vs 运行时分支**：对性能敏感路径可用 TilingKey / 模板特化；调试期可用运行时 `if`。
3. **隔离平台差异**：不同平台使用独立的 Tiling 结构体，避免共用 `TCubeTiling` 导致字段错位。
4. **扩展约定**：新增架构出现时，按 `NpuArch` 新增 reference 文件或更新平台适配说明，禁止在原有 A2/A3 路径中硬编码新平台逻辑。

## 常见陷阱

⚠️ 在 `DAV_3510` 上仍使用 `MatmulApiTiling`，编译或运行时报 unsupported。
⚠️ 用字符串比较 `socVersion` 时未覆盖别名（如 `Ascend910B2C`、`Ascend910B3` 等）。
⚠️ 平台分支后未重新计算 `blockDim`、`usedCoreNum` 等派生参数。
⚠️ 把 `DAV_2201` 的 `baseM=256` 直接套用到不支持该 Cube 粒度的架构。

## 代码搜索关键词

```bash
grep -nE "PlatformAscendC|GetSocVersion|NpuArch|DAV_2201|DAV_3510|Ascend950|Ascend910B" \
    op_host/*_tiling.cpp op_kernel/*.cpp
```

## 来源

- `ops/ascendc-tiling-design/references/matmul/ascendc-api-matmul-tiling.md` §适用平台 / §平台路由
