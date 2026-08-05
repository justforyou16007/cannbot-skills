---
name: evolution-strategies
description: AscendC kernel optimization strategy library with 146 strategies across D/P/A/R/X series, supporting tiered retrieval, compatibility checking, and autonomous strategy discovery. 当需要为 AscendC kernel 选择优化策略、按瓶颈/算子族筛选策略、检查策略兼容性或登记进化发现的新策略时使用。
---

# Evolution Strategies

AscendC 内核优化策略库，包含 146 策略，覆盖数据类型 (D)、性能 (P)、精度 (A)、A5 架构 (R)、进化发现 (X) 五大系列。

## 文件布局

```
references/
  scenario-routing/           # [选策略入口] 族判定 + 各族优化工作流（先于 index 查表）
    README.md                 # 族判定表（结构判据，不靠算子名）
    _common-prerequisites.md  # 通用前置判定链（C1-C5，所有族共享）
    flashattention.md / matmul.md / convolution.md / reduction.md / ...  # 族工作流
  strategy-index.md           # 策略卡注册表 + 补充反查（快速查表 / 瓶颈查表）
  strategy-compatibility.md   # 互斥表、依赖关系、推荐组合
  INDEX.json                  # 程序化索引（generate_index_json.py 生成）
  cards/                      # 策略卡片（146 张：P121 / D5 / A8 / R8）
  preconditions/              # 适用性硬门控 YAML（每策略一份）
  playbooks/                  # 分步落地 SOP（每策略一份）
  families/                   # 算子族指南（attention / flashattention / matmul / reduction 等 8 族）

  discovered/                 # [可写] 进化运行时发现的策略
    disc_X1.md ...            # X 系列，由 agent 动态写入
```

## 策略分层检索协议

### L0 (Universal) — 所有算子适用

18 个基础策略：D1-D5, P1-P13, A1-A8

**必读**: 先读 `references/scenario-routing/README.md` 按结构判据归族，打开对应族 workflow 按阶段选卡；`references/strategy-index.md` 的查表仅作补充反查。

### L1 (Advanced) — 按算子类别选择

进阶策略：P14-P131（按族见 scenario-routing 各 workflow）

根据算子类别（CV fusion, attention, matmul, quantization, reduction, sort等）选择适用的 L1 策略。参考 `strategy-index.md` 中的 "When to Apply" 列。

特别关注标量计算向量化系列 (P94-P98)：对任何新生成的 kernel 优先检查是否含有标量 for 循环，这是单次优化中收益最大的模式。

### L2 (Discovered) — 进化发现策略

X 系列（disc_X1.md, disc_X2.md, ...），由进化过程动态发现并注册。

## 策略兼容性检查

读取 `references/strategy_compatibility.md` 检查：

1. **互斥关系**: 部分策略不能同时使用（如 P1 vs P22/P19/P20）
2. **依赖关系**: 部分策略需要前置策略（如 P1 requires P5+P7）
3. **UB 预算**: Light (<50%) / Medium (50-75%) / Heavy (>75%)
4. **推荐组合**: 7 个经过验证的黄金组合

## 策略发现协议（写入 discovered/）

当 open_exploration 节点取得显著提升（speedup > best_score_before * 1.10）时：

1. 读取 `references/strategy-index.md` 检查新发现的优化方法是否已有覆盖
2. 如果是全新策略：
   - `Glob references/discovered/disc_X*.md` 获取当前最大编号
   - `Write references/discovered/disc_X{n+1}.md` 记录新策略
   - `Edit references/strategy-index.md` 追加新行到"探索发现策略"分类

> **可变性说明**: `strategies/` 目录为只读精选库（P/D/A/R 系列）。`discovered/` 目录为运行时可写（X 系列）。`strategy-index.md` 仅允许追加 X 系列条目。

## A5 架构扩展

当目标芯片为 Ascend 950 (351x/Regbase) 时，额外读取 `references/strategy_index_a5.md`：

- R1: VF fusion（合并多个 VF 函数，减少 Load/Store）
- R2: Register reuse（减少 RegTensor 溢出）
- R3: Dual-issue（指令调度优化）
- R4: Hardware loop（硬件循环构造优化）
- R5: Unaligned memory optimization
- R6: Low-latency tree reduction
- R7: SIMD/SIMT hybrid programming
- R8: Mutex synchronization
