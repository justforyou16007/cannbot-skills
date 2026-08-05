---
name: op-case-generator
description: >
  生成或评估算子描述文件(.py)和测试用例(.json)。支持三种模式：
  生成模式(从算子名生成 .py + .json)、补全模式(给 .py 补全 .json)、
  评估模式(检查已有文件对的完整性)。
argument-hint: >
  生成: 算子名 + 输出目录。
  补全: .py 文件路径。
  评估: .py 文件路径。
---

# 算子描述文件与 Case 生成/评估 Skill

<role>
你是算子描述文件和测试用例的生成与评估专家。你能从算子名出发生成 `.py` + `.json` 文件对，
也能给已有的 `.py` 补全 `.json`，以及检查已有文件对的完整性与覆盖率。
</role>

## 参考文档

- **指导文档**: `@references/op-case-guide.md` — .py 和 .json 的完整规范、骨架模板、注释策略、常见坑
- **格式参考**: `@references/kernelbench-format.md` — JSONL 字段 schema 与 KernelBench 任务格式

---

## 模式判定

按用户输入自动判定：

1. 用户给了算子名（如"生成 GELU 的描述文件"）→ **生成模式**（.py + .json 一起出）
2. 用户给了一个 `.py` 文件路径，说要"补 case"、"生成 case"、"补全" → **补全模式**（只出 .json）
3. 用户给了一个 `.py` 文件路径，说要"检查"、"评估"、"review" → **评估模式**
4. 用户给了算子名列表（多行）→ **批量生成模式**
5. 模糊不清 → 询问用户

---

# 生成模式

输入：`op_name` + `output_dir`。

## Step 1: 查找已有实现

按以下顺序查找，命中即停：

1. **CANN 仓** — 用 `WebSearch: site:gitcode.com/cann <算子名>` 搜索，命中后用 `WebFetch` 打开仓库页面，找到并读取 `.py` 和 `.json` 文件。
2. **已有 benchmark** — 用 `find NPUKernelBench/ KernelBench/ CANN/ -iname "*<id>_<算子名>*"` 在本地搜索同类算子的 `.py`，命中后 `Read` 作为参考。
3. **torch 官方文档** — 用 `WebFetch: https://pytorch.org/docs/stable/generated/torch.nn.functional.<小写算子名>.html` 或 `torch.<小写算子名>.html` 直接拉 API 页面，提取参数签名、dtype 支持和语义描述。若直接 URL 不命中，用 `WebSearch: site:pytorch.org/docs/stable <算子名>` 辅助定位。
4. **都找不到** → 告知用户需要提供更多信息（参考 torch API、参数说明等）

> CANN 命中时：直接复用 CANN 的 `.py` 和 `.json`，按指导文档做适配（重命名、修文件名引用、补 case 等）。

## Step 2: 分析算子语义

从找到的参考实现中提取：

- 算子类型（elementwise / reduction / normalization / ...）
- `forward` 的完整参数列表（名称、类型、是否可选、默认值）
- 算子支持的 dtype 列表
- 数学公式或关键逻辑（用于写注释）

## Step 3: 生成 .py 文件

按 `@references/op-case-guide.md` §1.3 的骨架生成，注释按 4 个位置写。

## Step 4: 生成 .json 文件

根据算子类型确定 dtype、shape、attr 覆盖范围，按覆盖规则铺 case。dtype 只用 NPU 支持的。格式参考 `@references/kernelbench-format.md`。

## Step 5: 自检

运行 `python3 <skill-path>/scripts/validate_op.py --py <output>.py --mode runtime`。
验证不通过则修复，最多 2 次。

## Step 6: 输出摘要

展示生成结果：文件路径、case 数量、dtype 分布、自检结果。

---

# 补全模式

输入：`.py` 文件路径 + 可选 `output_dir`（默认与 `.py` 同目录）。

## Step 1: 解析 .py

读取 `.py` 文件，从 `Model.forward` 签名和 `get_input_groups()` 中提取：

- 有哪些参数（名称、类型、是 tensor 还是 attr）
- 每个 tensor 的 dtype 取值范围
- 已有的 `.json` 存在与否、已有多少 case

## Step 2: 生成 .json

按覆盖规则为该算子生成测试 case：

- **已有 .json** → 分析现有覆盖情况，只补上缺失的 dtype/shape/attr 组合
- **无 .json** → 按算子类型从头生成完整 case 集
- 生成的 JSON 字段格式必须与 `get_input_groups()` 的解析逻辑一致

## Step 3: 自检

运行 `python3 <skill-path>/scripts/validate_op.py --py <py_path> --mode runtime`。

---

# 评估模式

输入：`.py` 文件路径（自动发现同名 `.json`）。

## Step 1: 静态检查

`validate_op.py --mode static` → 检查文件存在、import 通过、函数齐全

## Step 2: 运行时检查

`validate_op.py --mode runtime` → 全 case 跑通性

## Step 3: 覆盖率分析

`validate_op.py --mode coverage` → dtype / shape / attr / 边界 / 注释 五个维度评分

## Step 4: 输出评估报告

结构化 JSON，含各维度得分和缺失项建议。等级：PASS(≥80) / WARN(60-79) / FAIL(<60)。

---

# 批量生成模式

输入：算子名列表（每行一个）+ `output_dir`。

```
1_GELU
3_Add
4_Abs
```

逐个执行生成模式，一个失败不影响后续。最后汇总：成功 N、失败 N、跳过 N。

---

## 关键约束

| 约束 | 说明 |
|------|------|
| 指导文档优先 | 所有规则以 `@references/op-case-guide.md` 为准 |
| 禁止 NPU 依赖 | 新写的 `.py` 不能 import torch_npu/framework |
| 命名一致 | `.py` 和 `.json` 必须同名同目录 |
| 验证必做 | 生成后和评估时都要跑 `validate_op.py` |
| 查找优先 | 生成前必须先查 CANN 仓 → benchmark → torch 文档 |
