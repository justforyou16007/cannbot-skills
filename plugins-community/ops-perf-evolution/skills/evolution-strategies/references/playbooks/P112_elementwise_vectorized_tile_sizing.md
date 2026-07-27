# P112 Playbook: Elementwise 向量化 Tile 尺寸设计

> **强制流程**。本 Playbook 指导子 agent 对 Elementwise 算子按 Ascend C Tiling 方法论优化 tile 尺寸与向量化。必须按 Step 1-5 逐步执行。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p112_locations.txt`：

```bash
# 1.1 maxElemNum / ubFormer / blockFormer
grep -nE "maxElemNum|ubFormer|blockFormer|blockTail|ELEM_ALIGN" \
    shared/original/op_host/*_tiling.{cpp,h} > /tmp/p112_locations.txt
# 1.2 主循环 / CopyIn / CopyOut / Compute
grep -nE "for\s*\(.*blockIdx|CopyIn|CopyOut|Compute" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p112_locations.txt
# 1.3 dtype / cast / FP32 中间
grep -nE "Cast<|FP32|float32|half|bf16" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p112_locations.txt
# 1.4 shape 比较 / broadcast 判断
grep -nE "shape|broadcast|dims|stride" \
    shared/original/op_host/*_tiling.{cpp,h} >> /tmp/p112_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，列出 tile 计算位置与循环结构。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| shape 检查 | ? | 输入输出完全一致 | op_host/*_tiling.cpp:L? |
| maxElemNum | ? | 按 bufferNum*dtypeBits 计算 | op_host/*_tiling.cpp:L? |
| ubSplitAxis | ? | 最内轴向外第一个放不下 | op_host/*_tiling.cpp:L? |
| blockFormer | ? | fusedProduct / coreNum 向上取整 | op_host/*_tiling.cpp:L? |
| 多核对齐 | ? | 512 倍数 | op_host/*_tiling.cpp:L? |
| dtype 分支 | ? | FP16/BF16 Add/Sub 升精度 | op_kernel/*.cpp:L? |

**排除规则**：输入输出 shape 不同 → 走 P113 Broadcast。

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 简单 1D**：单输入单输出，循环为 `for blockIdx`。
- **形态 β — 多输入链式**：如 `Mul → Add → Cast`，需要多个 VECIN/VECOUT。
- **形态 γ — 带 tail 的 1D**：首 block 与 tail block 长度不同。
- **形态 δ — 升精度分支**：FP16/BF16 加减链路需要 Cast→FP32→Cast。

声明 `form: alpha | beta | gamma | delta`。

### 3B. Canonical Template（形态 α）

```cpp
// Host tiling
uint32_t maxElemNum = (ubSize - extraSize) * 8 / (bufferNum * maxDtypeBits);
maxElemNum = FloorAlign(maxElemNum, 256 * 8 / minDtypeBits);

// 找到 ubSplitAxis
uint32_t curProduct = 1, ubSplitAxis = 0;
bool allFit = true;
for (int i = shapeLen - 1; i >= 0; --i) {
    curProduct *= dims[i];
    if (curProduct > maxElemNum) {
        ubSplitAxis = i; curProduct /= dims[i]; allFit = false; break;
    }
}
if (allFit) curProduct /= dims[0];
uint32_t ubFormer = (shapeLen == 1) ? maxElemNum : maxElemNum / curProduct;
uint32_t ubOuter = CeilDiv(dims[ubSplitAxis], ubFormer);
uint32_t ubTail = dims[ubSplitAxis] - (ubOuter - 1) * ubFormer;

// 多核切分
uint32_t fusedProduct = ubOuter * Product(dims, 0, ubSplitAxis);
uint32_t blockFormer = CeilDiv(fusedProduct, coreNum);
uint32_t blockNum = CeilDiv(fusedProduct, blockFormer);
```

### 3C. Variant Notes

- **形态 β**：多输入链式需增加 `bufferNum`，并在 UB 内一次性完成所有 vector 操作，避免中间结果回 GM。
- **形态 γ**：tail block 用单独的 `curLen` 变量，CopyIn/Compute/CopyOut 均使用 `curLen`。
- **形态 δ**：在 Compute 前 Cast→FP32，计算后 Cast 回原始 dtype；注意 FP32 中间 buffer 预算。
- **核利用率不足**：循环 `maxElemNum -= CACHE_LINE` 重算，直到 `blockNum` 接近 `coreNum`。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P112 约束核查：
  1. 输入输出 shape 完全一致？ yes/no
  2. maxElemNum = ___，bufferNum = ___，maxDtypeBits = ___
  3. ubSplitAxis = ___，ubFormer = ___，ubOuter = ___，ubTail = ___
  4. blockFormer = ___，blockNum = ___，blockTail = ___
  5. 多核对齐（512）满足？ yes/no
  6. UB 对齐（256B）满足？ yes/no
  7. FP16/BF16 Add/Sub 是否启用升精度？ yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: maxElemNum 计算出现
(grep -cE "maxElemNum\s*=" modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 1

# 检查 2: ubSplitAxis 计算出现
(grep -cE "ubSplitAxis" modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 1

# 检查 3: blockFormer / blockNum 出现
(grep -cE "blockFormer|blockNum" modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 2

# 检查 4: tail block 处理
(grep -cE "blockTail|curLen|tail" modified_files/op_kernel/*.{cpp,h} modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 1

# 检查 5: FP16/BF16 升精度分支（若触发形态 δ）
(grep -cE "Cast\s*<\s*float|Cast\s*<\s*half|Cast\s*<\s*bfloat16" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 2（形态 δ）或 0（其他形态，需说明）
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P112 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: form={alpha|beta|gamma|delta}, tile sizing template applied
Step 4: constraints checked (maxElemNum=___, blockNum=___)
Step 5: all 5 grep checks passed
```

## 常见陷阱

- 把 broadcast 当 elementwise → shape 错误。
- `maxElemNum` 未按 dtype 位宽折算 → UB 溢出。
- 多输入链式未增加 bufferNum → 预算不足。
- tail block 使用固定 tile 长度 → 越界。
- 升精度分支缺少 Cast 回原始 dtype → 输出类型错误。
