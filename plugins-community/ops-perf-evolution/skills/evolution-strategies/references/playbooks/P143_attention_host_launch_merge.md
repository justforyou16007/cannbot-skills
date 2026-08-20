# P143 Playbook: Attention host 端 launch 合并 + per-shape tiling 缓存

> 本 Playbook 为**强制流程**。采纳 P143 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P143 的核心是**压减 attention 算子 host 侧固定开销：合并 split kernel launch、缓存 per-shape tiling、复用 workspace**。

## Step 1: 定位关键结构

```bash
# host 侧 launch、tiling、workspace 分配
grep -nE "launch\|at::zeros\|at::empty\|tiling\|TilingKey\|constant_pad_nd\|narrow\|copy_" \
    shared/original/op_host/*.cpp shared/original/op_host/*.h > /tmp/p143_locations.txt
grep -nE "split\|fused\|pack\|cat\|reshape" \
    shared/original/op_host/*.cpp >> /tmp/p143_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- host launch 位置与次数
- tiling 计算与 H2D 下发位置
- workspace 分配位置（zeros/empty）
- per-head / per-split 权重打包位置

## Step 2: 改造计划表（强制填写）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| launch 次数 | `?` | 最少化 | `op_host/*.cpp:L?` |
| tiling 计算 | `?` (每次调用) | per-shape cache | `op_host/*.cpp:L?` |
| workspace | `?` (zeros/每次申请) | empty + 复用 | `op_host/*.cpp:L?` |
| 权重打包 | `?` (per-head copy) | cat+reshape+单次 copy | `op_host/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 多个独立 kernel launch**：QKV pack、attention、out proj 各自 launch，可合并。
- **形态 β — 同 kernel 内 split 成多次调用**：如 fp32 hi/lo 多 GEMM 各自 launch，可在核内串行或 host 端合并参数后一次 launch。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α — 合并 launch + tiling cache）

```cpp
// === 改造前 ===
for (int h = 0; h < heads; ++h) {
    launch_pack_qkv(h);   // 多 launch
}
launch_attention(...);
for (int h = 0; h < heads; ++h) {
    launch_out_proj(h);
}

// === 改造后 ===
// 1. per-shape tiling cache
TilingKey key{b, s, h, d, dtype};
auto it = tiling_cache.find(key);
if (it == tiling_cache.end()) {
    it = tiling_cache.emplace(key, ComputeTiling(key)).first;
}
auto& tiling = it->second;

// 2. workspace 复用
auto workspace = workspace_pool.get_or_create(key, size);

// 3. 合并为最少 launch
launch_fused_attention_pack_proj(args, tiling, workspace);
```

### 3C. Variant Notes

- **与 X38 host 打包配合**：per-head `narrow().copy_()` 先合并为 `cat + reshape + strided copy`，再叠加 P143 的 launch 合并。
- **评估纪律**：合并后的评估仍必须遵循全量 geomean 与同窗口基线。
- **cache 失效**：shape 变化或 dtype 变化时 key 必须变化；建议 LRU 上限 128 条。

## Step 4: 约束复核

```
约束 1: 合并 launch 不能引入跨 kernel 数据竞争或错误同步
约束 2: tiling cache key 必须唯一标识 (B, S, H, D, dtype, stride)
约束 3: workspace 复用前必须确认 kernel 会全量覆盖，否则仍须清零
约束 4: cache 需有容量上限，防止 host 内存无限增长
约束 5: 小 shape 下缓存查找开销须低于重新 tiling 开销，否则禁用
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- 合并前 launch 次数 = ?
- 合并后 launch 次数 = ?
- cache 命中/未命中次数 = ?
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有 tiling cache 或复用逻辑
grep -cE "tiling_cache|cache.*tiling|unordered_map.*Tiling" modified_files/op_host/*.cpp
# 期望: >= 1

# 检查 2: 使用 at::empty 替代 at::zeros（在确认全量覆盖处）
grep -cE "at::empty" modified_files/op_host/*.cpp
# 期望: >= 1

# 检查 3: launch 次数减少（与原代码对比）
grep -cE "launch\|aclnn" modified_files/op_host/*.cpp
# 期望: 合并后明显减少（需在报告中给出前后数值）

# 检查 4: 有 workspace 复用或池化
grep -cE "workspace_pool|workspace_cache|reuse" modified_files/op_host/*.cpp
# 期望: >= 1

# 检查 5: 无 per-head narrow copy 热点
grep -cE "narrow\(\).*copy_|copy_\(.*narrow" modified_files/op_host/*.cpp
# 期望: == 0 或已合并为单次 cat+reshape
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 合并 launch 后同步错误 | 检查 event / stream 依赖，必要时显式同步 |
| cache key 冲突导致错误 tiling | key 加入 dtype、stride、pad 信息 |
| cache 内存无限增长 | 加 LRU / 容量上限 |
| workspace 未覆盖导致脏值 | 确认 kernel 全覆盖，否则保留清零 |
| 小 shape 缓存收益为负 | 设置启用阈值，如总 workload < 100μs 不缓存 |

---

**完成清单**：
```
[P143 Playbook Completion]
Step 1: done (/tmp/p143_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: launch_before=? launch_after=? cache_hits=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
