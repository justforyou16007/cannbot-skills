# 算子描述文件和 Case 生成指导

> **⚠️ 人工审核要求**：skill 自动生成的 `.py` 描述文件和 `.json` case 文件**必须经过人工介入审核**后才能用于下游生成，审核要点：① 参考实现语义是否正确——**参考实现不一定只是单纯调用 torch API，需结合算子实际情况调整**（torch 与 NPU 的语义差异、组合算子拆解、dtype 处理等，见 1.3）；② case 的 dtype/shape 覆盖是否合理；③ 注释里的公式与实现是否一致。确认无误后再提交或交付。

生成算子的输入是一对同名文件：

```
{id}_{OpName}.py    ← PyTorch 参考实现 + 解释注释
{id}_{OpName}.json  ← 每行一组测试输入（JSONL）
```

下游（单算子生成或批量评测）加载这两个文件，跑参考实现得到标准输出，生成器再产出算子 kernel 与之对齐。硬约束：

- `.py` 和 `.json` 必须**同名**，在同一目录
- `.py` 必须包含 `class Model(nn.Module)`、`get_input_groups()`、`get_init_inputs()`
- `get_input_groups()` 返回的 case 数与 JSON 行数一致
- `Model(*get_init_inputs()[i])(*get_input_groups()[i])` 全 case 跑通

任一不满足，下游 agent 可能误判为自身代码 bug，增加不必要的排查和修复成本。

---

# 第一部分 · 算子描述文件（.py）

## 1.1 写法决策

按优先级依次判断，命中即停：

1. **查 CANN 仓** — 用 `WebSearch: site:gitcode.com/cann <算子名>` 搜索，命中后 `WebFetch` 打开仓库读取 `.py`/`.json` → 有则直接用（跳到 1.2）
2. **查已有 benchmark** — 用 `find NPUKernelBench/ KernelBench/ CANN/ -iname "*<算子名>*"` 在本地搜 → 有则参考
3. **查 torch API** — 用 `WebFetch: https://pytorch.org/docs/stable/generated/torch.nn.functional.<小写算子名>.html` 拉页面，不命中则 `WebSearch: site:pytorch.org/docs/stable <算子名>` → 有则基于 API 新写（跳到 1.3）
4. **拼小算子实现** → 以上都没有，用基础 torch 算子组合实现（跳到 1.4）

---

## 1.2 情况 ①：CANN 仓已有

在 https://gitcode.com/cann 搜索算子名，有则直接复用。

---

## 1.3 情况 ②：基于 torch API 新写

### 骨架要点

```python
# imports: os, json, torch, torch.nn, torch.nn.functional（禁止 import torch_npu / framework）

class Model(nn.Module):
    """<算子功能（中文）>。官网链接、数学公式、NPU 限制。"""
    def __init__(self):
        super(Model, self).__init__()          # 标准：空 __init__，所有参数放 forward

    def forward(self, ...) -> torch.Tensor:    # 参数 + 行内注释（含义、约束）
        # 按算子实际计算逻辑写，需要几步写几步，不限于一行 API 调用
        return <torch API 调用 / 实际计算逻辑>

def get_input_groups():
    # 逐行读同名 JSON，按 inputs 数组生成 [args, ...] 列表
    # dtype_map 含全部 dtype 及别名（fp32/float32, fp16/float16, bf16/bfloat16）
    # make_tensor 按 dtype 分支：int→randint, bool→randint, uint→int64+cast, 其他→randn
    # tensor_list 同时支持 "shapes"（同 dtype）和 "tensors"（异构）两种 JSON 格式
    ...

def get_init_inputs():
    return []   # 标准模式
```

完整模板参见 `validate_op.py` 验证通过的任意 NPUKernelBench 文件。

### 注释要求

**建议在 4 个位置加注释**，让生成 kernel 的模型更容易理解接口语义：

| 位置 | 内容 |
|------|------|
| `Model` docstring | 算子功能（中文）、官网链接、数学公式、NPU 实现要点 |
| `forward` 参数 | 每个参数的含义、shape 约束、dtype 限制、是否可选 |
| `__init__` | 说明为何无参数（所有配置值通过 forward 传入） |
| 复杂逻辑前 | 用 `# 1. / # 2. / ...` 把公式拆成可读步骤 |

### 注意

- **多输出**：`forward` 返回 tuple 时，注释标注每个输出的含义
- **Backward 算子**：含 `grad_output` + `self_or_result`，forward 中 fp16→fp32→fp16，注释标注梯度逻辑
- **tensor_list**：JSON 有两种格式（`"shapes"` 同 dtype / `"tensors"` 异构），get_input_groups 需兼容
- **多 dtype**：不同 tensor 可声明不同 dtype，各自独立生成
- **uint dtype**：CPU torch 不支持 uint16/32/64，forward 中显式 cast 为 float32

---

## 1.4 情况 ③：组合算子

torch 无直接 API，用基础 torch 算子拼装。`forward` 直接写拼装代码，注释逐步标注语义和对应公式项。

---

## 1.5 常见坑

| 症状 | 原因 | 修复 |
|------|------|------|
| `len(groups) != len(inits)` 且 inits 非空 | 改了 JSON 没同步改 `get_init_inputs` | 改为返回 `[]`（标准）或等长 list |
| `Model()` 报缺少参数 | `__init__` 有必填入参 | 参数移到 forward，`__init__` 留空 |
| `forward()` 参数不对 | JSON inputs 和 forward 签名不一致 | 对齐顺序和数量 |
| 整数 dtype 报 `randn` 不支持 | 没走 make_tensor int 分支 | 用骨架的 `make_tensor` |
| uint dtype 报错 | CPU torch 不支持 | forward 中 cast float32 |
| tensor_list 读取失败 | JSON 用 `shapes` 但 .py 读 `tensors` | .py 同时处理两种 |
| 生成的 kernel 语义偏了 | 注释不够详细 | 补 torch 文档 + 小算子拆解注释 |

---

# 第二部分 · Case 文件（.json）

JSONL（每行一个独立 JSON），格式不强制，只要 `.py` 的 `get_input_groups()` 能解析即可。以下是一个参考格式：

```json
{"inputs": [{"name": "x", "type": "tensor", "dtype": "float16", "shape": [128]}, {"name": "alpha", "type": "attr", "dtype": "float", "value": 1.0}]}
```

参考字段：

| 字段 | 说明 |
|------|------|
| `name` | 参数名，对应 `forward` 形参 |
| `type` | `tensor` / `tensor_list` / `attr` |
| `required` | `true` / `false`（false 可省略） |
| `dtype` | NPU 支持的 dtype：`float16`/`float32`/`bfloat16`/`int32`/`int64`/`bool`/... |
| `shape` | 列表如 `[128, 256]`，标量用 `[]` |
| `value` | attr 的实际值 |

核心约束就一条：**dtype 用 NPU 支持的，字段跟 `.py` 的解析逻辑对得上**。

## 常见坑

| 症状 | 原因 | 修复 |
|------|------|------|
| `KeyError: 'inputs'` | CANN 扁平格式与 inputs 骨架混用 | 新写统一用 inputs 数组 |
| 整行都 fail | dtype 不被算子支持 | 删掉该 dtype |
| 个别 case fail | shape 落在不支持区间 | 删该 case 或拆分 |
| 参数数量不对 | JSON inputs 与 forward 签名不一致 | 对齐 |

---

# 第三部分 · Skill 使用

## 安装

将整个 `ops/op-case-generator` 目录放到项目的 `.claude/skills/` 下：

```
.claude/skills/op-case-generator/
├── SKILL.md
├── references/
│   └── kernelbench-format.md
└── scripts/
    └── validate_op.py
```

Claude Code 启动时自动加载 `.claude/skills/` 下的所有 skill，无需额外配置。

## 使用

### 生成（给算子名，出 .py + .json）

```
生成算子描述文件：op_name=25_GELU, 输出到 /path/to/output/
```

只需给算子名和输出目录，skill 自动查 CANN 仓 → benchmark → torch 文档，推断 API 和 dtype。

### 补全（给 .py，出 .json）

```
给 /path/to/25_GELU.py 补全测试用例
```

skill 解析 `.py` 的 forward 签名和 get_input_groups 逻辑，按覆盖规则生成匹配的 `.json`。

### 批量生成（多个算子）

```
批量生成：1_GELU, 3_Add, 4_Abs, 5_SwiGLU, 输出到 /path/to/output/
```

### 评估（检查已有文件对）

```
检查 /path/to/25_GELU.py 的完整性
```

自动发现同名 `.json`，输出结构/运行时/覆盖率各维度得分和缺失项。

## 命令行自检（不使用 skill）

```bash
python3 scripts/validate_op.py --py <file.py> --mode static    # AST 静态检查
python3 scripts/validate_op.py --py <file.py> --mode runtime   # 全 case 跑通性
python3 scripts/validate_op.py --py <file.py> --mode coverage  # 覆盖率评分
python3 scripts/validate_op.py --py <file.py> --mode all       # 以上全部
```
