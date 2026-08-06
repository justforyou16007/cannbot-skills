# 尾块处理

FA 类算子里常见的两种：

**1. 行方向（q/序列尾）**：普通 `T.copy` 即可，越界读写都自动处理——
`LegalizeSafeMemoryAccess` pass 把越界读零填充（越界的行按 0 读进来），越界写按 buffer
类型处理：GM 写自动包 `IfThenElse` 跳过、UB/L1 写自动填 padding（默认 0）。因此行方向
的越界行是确定的 0，行方向计算（如 softmax 逐行独立）完全无害，不用管。

**2. 列方向（K/V 尾块）**：这里的越界**列**不能靠编译器——越界列从 K 读到的 0 是*有效分数*
（`exp(0·scale)=1`），会进 softmax 分母，必须自己把越界列填 `-1e30`（仅最后一块）：

```python
if tail_valid != 0:  # 最后一个 KV 块
    T.tile.createvecindex(col_idx, 0)        # 列索引 0..block_N
    T.copy(col_idx, col_idx_f)                # int32 -> fp32
    T.tile.compare(cmp_mask, col_idx_f, tail_valid - 1, "LE")  # 有效列 -> bit=1
    T.tile.select(acc_s_ub[r, :], cmp_mask, acc_s_ub[r, :],
                  -1e30, "VSEL_TENSOR_SCALAR_MODE")
```

- 越界列 `exp(-1e30)≈0`，softmax 概率 0
- 填 `-1e30` 而非 `0`：`exp(0)=1` 会污染分母
- `cmp_mask` 是 `uint8` 按位打包（宽 `block_N//8`），不是 bool 数组
