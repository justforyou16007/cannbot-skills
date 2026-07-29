# P115 Playbook: FlashAttention Tiling 路由

> **强制流程**。本 Playbook 指导子 agent 对 FlashAttention 类算子按 Ascend C Tiling 四层架构完成选型与节点化设计。必须按 Step 1-5 逐步执行。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p115_locations.txt`：

```bash
# 1.1 FA 相关参数：head / hidden / seq / causal / GQA / MLA
grep -nE "headNum|kvHeadNum|hiddenSize|seqLen|causal|GQA|MHA|MLA" \
    shared/original/op_host/*_tiling.{cpp,h} > /tmp/p115_locations.txt
# 1.2 dtype / quant / sparse / feature flags
grep -nE "dtype|MXFP8|INT8|AntiQuant|sparse|PSE|RoPE|Sink" \
    shared/original/op_host/*_tiling.{cpp,h} >> /tmp/p115_locations.txt
# 1.3 matmul / softmax / sync 结构
grep -nE "Matmul|softmax|SyncAll|SetFlag|WaitFlag|FixPipe" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p115_locations.txt
# 1.4 tiling / block / task 切分
grep -nE "block|tile|task|splitKV|mEff|Bk" \
    shared/original/op_host/*_tiling.{cpp,h} >> /tmp/p115_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，列出 FA 形态与 trait 位置。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 基础形态 | ? | GQA / MHA / MLA | op_host/*_tiling.cpp:L? |
| Hq / Hkv / G | ? | G = Hq / Hkv | op_host/*_tiling.cpp:L? |
| 量化 trait | 无 / 有 | MXFP8 / INT8 / AntiQuant / 无 | op_host/*_tiling.cpp:L? |
| 稀疏 trait | 无 / 有 | BlockSparse / SlidingWindow / 无 | op_host/*_tiling.cpp:L? |
| feature flags | 无 / 有 | PSE / RoPE / Sink / ... | op_host/*_tiling.cpp:L? |
| 设计节点 | 缺失 | G→N12 逐步落地 | design/ 文档 |
| 组合校核 | 未做 | 在 composition.md 确认 | implementation_note.txt |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 标准 GQA/MHA**：按 `base_design.md` §1-§10 落地。
- **形态 β — MLA**：latent absorption，走 `base_design.md` §11。
- **形态 γ — 量化 FA**：叠加 `quantization_design.md`，需处理 scale 轴与 P-scale 回传。
- **形态 δ — 稀疏 FA**：叠加 `sparse_design.md`（当前占位），需先确认可行性。

声明 `form: alpha | beta | gamma | delta`。

### 3B. Canonical Template（形态 α）

```cpp
// Host 侧关键决策
uint32_t G = headNum / kvHeadNum;          // GQA group size
bool isMHA = (G == 1);
bool isMLA = /* 公式含 latent absorption */;

// 节点化设计落地映射
N1: kernelType, UB mode, Cube block, pipe depth
N2: task-level vs split-KV reduce
N3: roofline(mEff, AI) -> feasible Bk/M/N range
N4: Softmax V1(max-trick) or V2(online)
N6: task dimension + load balance
N7: UB/L1/L0/GM partition
N8: compile-time specialization
N10: pipeline scheduling + SyncAll
N12: DESIGN.md output
```

### 3C. Variant Notes

- **形态 β（MLA）**：KV 经 latent compression，`kvHeadNum=1`，注意 split-KV 未验证。
- **形态 γ（量化）**：MXFP8 需修正 Roofline AI；scale 字段加入 Host Tiling。
- **形态 δ（稀疏）**：当前 `sparse_design.md` 为占位骨架，落地前需先补全契约。
- **feature flags**：PSE/RoPE/Sink 等按 `feature_flags.md` 叠加，不可与基础形态冲突。

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P115 约束核查：
  1. 基础形态 = GQA / MHA / MLA，依据？ ___
  2. Hq = ___, Hkv = ___, G = ___
  3. 正交 trait：量化=___ 稀疏=___ feature flags=___
  4. 组合是否在 composition.md 已验证？ yes/no / 未验证需说明
  5. 是否按 G→N12 顺序推进？ yes/no
  6. 跨核槽 / L1 rotation 槽 / V2 自读自写槽是否使用不同常量？ yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 基础形态参数存在
(grep -cE "headNum|kvHeadNum|G\s*=|gMaxPerTask" modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 1

# 检查 2: Softmax 模式选择
(grep -cE "V1|V2|online_softmax|max_trick" modified_files/op_kernel/*.{cpp,h} modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 1

# 检查 3: 同步原语使用
(grep -cE "SyncAll|SetFlag|WaitFlag|HardEvent" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 2

# 检查 4: Roofline / mEff / Bk 相关
(grep -cE "mEff|Bk|AI|roofline" modified_files/op_host/*_tiling.{cpp,h}; true)
# 期望: >= 1

# 检查 5: 常量语义分离（跨核、L1、V2 使用不同命名）
(grep -cE "CROSS_CORE_SLOT|L1_ROTATION_SLOT|V2_SELF_SLOT" modified_files/op_kernel/*.{cpp,h}; true)
# 期望: >= 2（若命名不同但语义已分离，需在 note 中说明）
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P115 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: form={alpha|beta|gamma|delta}, FA routing applied
Step 4: constraints checked (base=___, traits=___, nodes=G→N12)
Step 5: all 5 grep checks passed
```

## 常见陷阱

- 把 MHA/MLA 当独立子族 → 它们只是基础形态变化。
- 跳过 Roofline 直接定 tile → 性能不达标。
- 未校核量化 × 稀疏组合 → 可能未验证。
- 同一算子混用 GQA 与 MLA → 违反架构约束。
- 常量槽位复用 → 跨核同步或 L1 rotation 出错。
