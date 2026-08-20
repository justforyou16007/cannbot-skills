# P120 Playbook: FA 向量点积模拟 Matmul → Cube mmad 重写

> **强制流程**。本 Playbook 指导子 agent 把 attention 算子的 QK^T / PV 从 Vector 指令模拟（Mul+ReduceSum / 标量乘加）重写为 Cube mmad。这是 L2 架构级改写，必须按 Step 1-5 逐步执行，严禁跳步。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p120_locations.txt`：

```bash
# 1.1 向量模拟矩阵运算证据
grep -nE "Mul\(|Muls\(|ReduceSum|WholeReduceSum|Add\(" \
    shared/original/op_kernel/*.{cpp,h} > /tmp/p120_locations.txt
# 1.2 输入 fp32 Cast 提升位置
grep -nE "Cast\(" shared/original/op_kernel/*.{cpp,h} >> /tmp/p120_locations.txt
# 1.3 Q/K/V 搬运路径（DataCopy 参数、stride）
grep -nE "DataCopyPad|DataCopyExtParams|Nd2Nz" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p120_locations.txt
# 1.4 tile 常量（BM/BN/Df 等）
grep -nE "constexpr.*BM|constexpr.*BN|static.*BM|static.*BN" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p120_locations.txt
# 1.5 标量读 vector 结果（V→S 同步）
grep -nE "GetValue|SetFlag.*V_S|WaitFlag.*V_S" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p120_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，明确：QK^T 当前由什么指令实现、PV 当前由什么指令实现、fp32 Cast 发生在哪几处、当前 tile 尺寸、逐元素 GetValue 有几处。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| QK^T 实现 | Mul+ReduceSum 逐对点积 | mmad（Q fp16/bf16 × K^T → S fp32 L0C） | op_kernel/*.cpp:L? |
| PV 实现 | 标量 Muls+Add 逐元素 | mmad（P 输入dtype × V → O fp32 L0C） | op_kernel/*.cpp:L? |
| 输入 Cast | Q/K/V 全量 Cast fp32 | 删除，输入 dtype 直达 L1 | op_kernel/*.cpp:L? |
| K 的布局 | 行主序 + 隐式逐元素转置 | ColumnMajor 视图 + nd2nz→nZ 免转置 | op_kernel/*.cpp:L? |
| Q 的加载 | 每 tile 重复搬 UB | preload 一次常驻 L1 | op_kernel/*.cpp:L? |
| L0A/L0B/L0C | 未使用 | 全部双缓冲（STAGES=2） | op_kernel/*.cpp:L? |
| S/P 中间通路 | UB 驻留 | L0C→GM workspace（fp32 S / 输入dtype P）；**若 `BLOCK_M*KV_CHUNK*4B` 可纳入 UB 预算，优先 C 直写 VECCALC 零 GM scratch（3_FusionAttention 0.03x→0.84x 的关键，配 P117 C position 一致性）** | op_kernel/*.cpp:L? |
| 标量 GetValue 读 | 逐元素 V→S | 删除（score 由 cube 产出，行归约用 BlockReduceMax/Sum） | op_kernel/*.cpp:L? |

## Step 3: 代码改造

### 3A. 形态识别

- **Canonical**：纯向量 attention——QK^T 用 `Mul+ReduceSum`，PV 用 `Muls+Add`，softmax 逐元素 `GetValue`。
- **变体 A**：已有 Matmul API（MatmulImpl）但输入先 Cast fp32 → 只做精度链路修复（删除 Cast、P downcast 回输入 dtype）。
- **变体 B**：单核向量实现但 tile ≥64 → 同样适用，L1 tile 直接按 Canonical 取值。
- **变体 C**：矩阵维 < 64 的小 shape 专用算子 → **不适用**，退回 P42/P39（判定见 P13）。

### 3B. Canonical Template

```cpp
// 改造前（QK^T 逐对点积）
for (qi) for (kj) { Mul(rw, qf[qi*Df], kf[kj*Df], Df); ReduceSum(sp[..], rw, rt, Df); }

// 改造后（mmad + nd2nz 分形搬运 + 事件流水）
copyGmToL1B(l1K, gK[colMajorView], Nd2NzParams{...});   // K: ColumnMajor→nZ，免转置
SetFlag<HardEvent::MTE2_MTE1>(e); WaitFlag<HardEvent::MTE2_MTE1>(e);
copyL1ToL0A(l0A, l1Q);  copyL1ToL0B(l0B, l1K);          // ifTranspose=false
SetFlag<HardEvent::MTE1_M>(e);  WaitFlag<HardEvent::MTE1_M>(e);
Mmad(l0C, l0A, l0B, {m, n, k, /*initL0C=*/(kIdx==0)});  // S fp32 累加在 L0C
SetFlag<HardEvent::M_FIX>(e);   WaitFlag<HardEvent::M_FIX>(e);
copyL0CToGm(gS[wsSlot], l0C);                           // fixpipe 到 workspace
```

### 3C. Variant Notes

- 若 GM 行 stride > 65535 → nd2nz 降级为按行多次单条 DataCopy（STRIDE_LIMIT 回退）。
- 若 embed 非 16 对齐 → nZ/zN 的 `MakeLayout` 自动 RoundUp 到 C0(16)，GM 侧用 DataCopyPad 语义补齐。
- 若 PV 的 P 来自 GM workspace → P 按 RowMajor→zN 搬运，V 按 RowMajor→zN，OTmp fp32 写 L0C。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P120 约束核查：
  1. L1 预算：Q常驻(128*kDyn*2B) + KV双缓冲 ≤ L1_MAX(512K)？ 计算式 = ___
  2. L0A/L0B tile ≤ 64K、L0C tile ≤ 128K？ 计算式 = ___
  3. Q/K/V 进 cube 前无 fp32 Cast？ yes/no
  4. P 已 downcast 回输入 dtype（bf16=CAST_RINT / fp16=CAST_NONE）？ yes/no
  5. 小 tile (m/16)*(n/16)<10 处已加 PipeBarrier<PIPE_M>？ yes/no
  6. 逐元素 Mul+ReduceSum / GetValue 已全部删除？ yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: mmad 已引入
(grep -cE "Mmad|mmad|BlockMmad|Matmul" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 2: 向量点积已删除
(grep -cE "ReduceSum|WholeReduceSum" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: == 0

# 检查 3: 分形搬运已引入
(grep -cE "Nd2Nz|nd2nz|nZ|zN|ColumnMajor" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 1

# 检查 4: 逐元素标量读已删除
(grep -cE "GetValue" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: == 0

# 检查 5: 双缓冲/事件流水
(grep -cE "SetFlag|WaitFlag|STAGES|DOUBLE_BUFFER" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 2
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P120 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: cube mmad rewrite applied (qk_mmad=yes, pv_mmad=yes, fp32_cast_removed=yes)
Step 4: constraints checked (l1_budget=___, p_downcast=___)
Step 5: all 5 grep checks passed
```

## Step 6: Known Pitfalls

- 改写后精度全错 → 检查 K 的 layout 视图是否真的是 ColumnMajor（行主序存储按列读），nZ `MakeLayout` 行列是否写反。
- 结果前 16 列正确后面全错 → L0B 的 K 维未按 C0(16) 对齐，检查 `RoundUp(embed,16)`。
- 性能反而更慢 → tile 太小（<64）cube 启动开销主导，或 L0C 未双缓冲导致 FIX/M 互等。
- 偶发 NaN → L0C→GM 的 fixpipe 与下一轮 Mmad 之间缺 `FIX_M` 事件，L0C 被覆盖。
- 编译报 nd2nz 参数非法 → GM stride 超 65535，走 STRIDE_LIMIT 降级路径。
