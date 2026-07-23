# P100 Playbook: GatherMask 替代逐元素 SetValue/GetValue

> **强制流程**。P100 把从连续 tensor 中按固定间隔取元素重组数据（如 RoPE 奇偶分拆、解交织）的逐元素 `SetValue`/`GetValue` 循环，替换为 AscendC `GatherMask` API。O(1) 条向量指令替代 O(n) 次标量操作，且零拷贝。**必须**按本 Playbook 逐步执行。

## Step 1: 定位逐元素 SetValue/GetValue 解交织反模式

执行下面的 grep，把结果写入 `/tmp/p100_locations.txt`：

```bash
# 1.1 逐元素 SetValue/GetValue 循环
grep -nE "for\s*\([^)]*\+\+[^)]*\).*SetValue.*GetValue" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null > /tmp/p100_locations.txt

# 1.2 解交织/奇偶分拆模式
grep -nE "even.*odd|odd.*even|interleave|deinterleave|2\s*\*\s*i\s*\+" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p100_locations.txt

# 1.3 已有 GatherMask 使用
grep -nE "GatherMask|rsvdCnt|GatherMaskParams" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p100_locations.txt

# 1.4 固定间隔取元素（stride=2,4,...）
grep -nE "GetValue\(.*\+\s*2\s*\*\s*i\)|GetValue\(.*\+\s*4\s*\*\s*i\)" \
    shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h 2>/dev/null >> /tmp/p100_locations.txt
```

**交付物**（`implementation_note.txt` "Playbook Step 1"）：
- **SetValue/GetValue 循环位置**：逐元素解交织的文件 + 行号
- **固定间隔取元素位置**：`GetValue(offset + k*i)` 模式
- **已有 GatherMask**：是否已使用
- **数据规模**：half / rope_size 等变量范围

## Step 2: GatherMask 改造计划表（强制）

| 调用点 (file:line) | 当前反模式 | 源 tensor | 间隔/偏移 | 输出两段 | count | 预计收益 |
|---|---|---|---|---|---|---|
| kernel.cpp:145 | 2×half SetValue/GetValue | kvF32[rms_size_:] | deinterleave=1/2 | even/odd | rope_size | 5-20x |
| ... | ... | ... | ... | ... | ... | ... |

**排除规则**（下列情况**不改**）：
- **取元素间隔不固定**：GatherMask 要求固定 deinterleave 步长
- **count < 64**：小数据量收益递减
- **已有 GatherMask**：已经是向量化解交织
- **dst 与 src 重叠且无法通过切片拆分**：GatherMask 要求输出区域可明确划分

## Step 3: 改造实施

### 3A. 形态识别

- **形态 α — 奇偶分拆**：从 tensor 分离偶数位和奇数位（RoPE 典型）
- **形态 β — 固定 stride 采样**：如每 4 个取 1 个
- **形态 γ — 解交织后接向量运算**：GatherMask 结果继续进入 Max/Min/Mul 链

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 α — 奇偶分拆）

```cpp
// === 改造前：逐元素 SetValue/GetValue 解交织 ===
int half = rope_size_ / 2;
LocalTensor<float> kF32 = ropeBuf_.Get<float>();
for (int i = 0; i < half; ++i) {
    kF32.SetValue(i, kvF32.GetValue(rms_size_ + 2 * i));        // 偶数位
}
for (int i = 0; i < half; ++i) {
    kF32.SetValue(half + i, kvF32.GetValue(rms_size_ + 2 * i + 1)); // 奇数位
}

// === 改造后：GatherMask 解交织（2 条指令） ===
uint64_t rsvdCnt = 0;
LocalTensor<float> ropeSrc = kvF32[rms_size_];  // 零拷贝偏移视图

// 偶数位：deinterleave=1, offset=1
GatherMask(kF32, ropeSrc,
    static_cast<uint8_t>(1), true,
    static_cast<uint32_t>(rope_size_),
    {1, 1, 8, 0}, rsvdCnt);

// 奇数位：deinterleave=2, offset=2 → 写入 kF32[half]
GatherMask(kF32[half], ropeSrc,
    static_cast<uint8_t>(2), true,
    static_cast<uint32_t>(rope_size_),
    {1, 1, 8, 0}, rsvdCnt);
PipeBarrier<PIPE_V>();
```

**关键替换要点**：
1. `deinterleave=1` = 连续取（偶数位用 deinterleave=1 + offset=1）
2. `deinterleave=2` = 隔 1 取 1（奇数位用 deinterleave=2 + offset=2）
3. `rsvdCnt` 必须声明为 `uint64_t` 并初始化为 0
4. GatherMask 后必须加 `PipeBarrier<PIPE_V>()`
5. `count` 传元素总数（非 half）

### 3C. Variant Notes

**形态 β — 固定 stride=4 采样**：
```cpp
GatherMask(dst, src, static_cast<uint8_t>(4), true, count, {1, 1, 8, 0}, rsvdCnt);
```

**形态 γ — 解交织后接向量链**：
```cpp
GatherMask(evenBuf, src, 1, true, count, {1,1,8,0}, rsvdCnt);
GatherMask(oddBuf,  src, 2, true, count, {1,1,8,0}, rsvdCnt);
PipeBarrier<PIPE_V>();
Mul(out, evenBuf, oddBuf, count);  // 继续向量链
```

**与其他策略协同**：
- + **P97 零拷贝切片**：`ropeSrc = src[offset]` 视图替代显式拷贝
- + **P95 向量链**：GatherMask 结果进入 Adds/Muls/Max/Min 链
- + **P67 Counter 模式**：GatherMask 后的向量处理可用 Counter 模式

## Step 4: 约束复核

```
P100 改造约束核查：
  1. deinterleave 是 uint8_t 并已 cast: yes/no
  2. rsvdCnt 是 uint64_t 并初始化为 0: yes/no
  3. count 传元素总数（非 half）: yes/no
  4. GatherMask 后有 PipeBarrier<PIPE_V>: yes/no
  5. dst 两段不重叠（如 kF32 和 kF32[half]）: yes/no
  6. 源 tensor 视图偏移正确: yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: 逐元素 SetValue/GetValue 循环减少
BEFORE=$(grep -cE "for\s*\([^)]*\+\+[^)]*\).*SetValue.*GetValue" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
AFTER=$(grep -cE "for\s*\([^)]*\+\+[^)]*\).*SetValue.*GetValue" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "SetValue/GetValue loops: before=$BEFORE after=$AFTER"
# 期望：AFTER < BEFORE

# 检查 2: GatherMask 使用增加
GM_B=$(grep -cE "GatherMask\s*\(" shared/original/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
GM_A=$(grep -cE "GatherMask\s*\(" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "GatherMask calls: before=$GM_B after=$GM_A"
# 期望：GM_A > GM_B

# 检查 3: rsvdCnt 声明
RSVD=$(grep -cE "uint64_t\s+rsvdCnt|rsvdCnt\s*=\s*0" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "rsvdCnt declarations: $RSVD"
# 期望：>= 1

# 检查 4: PipeBarrier<PIPE_V> 在 GatherMask 后
PB=$(grep -cE "PipeBarrier\s*<\s*PIPE_V\s*>" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "PipeBarrier<PIPE_V>: $PB"
# 期望：>= 1

# 检查 5: deinterleave 已 cast 为 uint8_t
CAST=$(grep -cE "static_cast\s*<\s*uint8_t\s*>\s*\(\s*[12]\s*\).*GatherMask|GatherMask.*static_cast\s*<\s*uint8_t\s*>" modified_files/op_kernel/*.{cpp,h} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
echo "uint8_t cast for deinterleave: $CAST"
# 期望：>= 1
```

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：GatherMask 模板参数 | 检查 `deinterleave` 是否为 `uint8_t`，`rsvdCnt` 是否为 `uint64_t&` |
| 运行结果全 0 | 漏了 `PipeBarrier<PIPE_V>()`；后续指令读到旧值 |
| 偶数位/奇数位顺序反了 | `deinterleave` 和 offset 的搭配错误；deinterleave=1 取偶数，deinterleave=2 取奇数 |
| count 传错导致越界 | count 应传元素总数，不是 half |
| 小 count（<64）收益低 | 保留原 SetValue/GetValue；不改 |
| dst 两段重叠写入 | 确保第二段从 `dst[half]` 开始，且 half 不超出 dst 容量 |

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P100 Playbook Completion]
Step 1: done, N SetValue/GetValue deinterleave loops identified
Step 2: refactor plan filled (src=___, deinterleave=___, count=___)
Step 3: GatherMask applied, even/odd split via dst[half]
Step 4: rsvdCnt=uint64_t, count=total, PipeBarrier=___, dst non-overlap ✓
Step 5: all 5 grep checks passed (Set/Get loops: X→Y, GatherMask: X→Y, rsvdCnt: X)
Step 6: no pitfalls / {列出触发的}
```
