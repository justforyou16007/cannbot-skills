---
id: P119
bottlenecks: [mte2_stall, no_overlap]
op_families: [attention, flash_attention, softmax]
complexity: L2
conflicts_with: []
synergizes_with: [P38, P115, A5]
requires: []
has_preconditions: true
has_playbook: true
---

# P119: Two-Pass Softmax → Online Softmax 单遍 KV 流式改写

## 核心思想

两遍 softmax——pass1 扫全部 KV 求行 max，pass2 再扫一遍算 `exp(s - max)` 并累加分子分母——把 **K 从 GM 读两遍、V 读一遍**，MTE2 流量接近 2×，且两遍之间无任何流水重叠。online softmax（flash 范式）把 max/sum/acc 三个状态改为 running 更新，**K/V 各只读一遍**：

```
每块 S2 tile：
  s = QK^T * scale                      (cube)
  m_new = max(m_old, rowmax(s))
  p = exp(s - m_new)
  alpha = exp(m_old - m_new)            // rescale 因子
  l = l * alpha + rowsum(p)             // 分母 running sum
  acc = acc * alpha + p @ V             // 分子 running 累加（cube）
  m_old = m_new
收尾：out = acc / l
```

GM 流量减半只是账面收益；真正的质变是 **QK^T(cube) → softmax update(vector) → PV(cube) 在同一 S2 循环内首尾相接**，配合状态 buffer 常驻（P38）后 cube/vector 流水可以持续重叠。

精度说明：fp32 累加 + max-trick 下 online 更新本身数值稳定（A5）。改写后**不要**保留逐 token Kahan 补偿——它每 token 引入 5+ 次额外 vector 往返，会打碎流水。

## 代码骨架

```cpp
// === 改造前：两遍，K 读两次 ===
// pass1: for j in S_kv { load K_j; dot; rowmax }        // K 第 1 遍
// pass2: for j in S_kv { load K_j; dot; exp; load V_j; acc += p*V_j }  // K 第 2 遍 + V

// === 改造后：单遍 online 更新 ===
Duplicate(rowMax, -INF, s1); Duplicate(rowSum, 0, s1); Duplicate(acc, 0, s1 * d);
for (int64_t s2 = 0; s2 < skv; s2 += s2Base) {           // K/V 各读一次
    // 1) cube: sTile = Q @ K[s2:s2+s2Base]^T * scale
    // 2) vector: tileMax = RowMax(sTile); newMax = Max(rowMax, tileMax)
    // 3) vector: p = Exp(sTile - newMax); alpha = Exp(rowMax - newMax)
    // 4) vector: rowSum = rowSum * alpha + RowSum(p)
    // 5) vector: acc  *= alpha   （broadcast 到 d 维）
    // 6) cube:   acc += p_cast @ V[s2:s2+s2Base]
    // 7) rowMax = newMax
}
// 收尾：out = acc / rowSum（rowSum==0 的全 mask 行输出 0）
```

## 关键修改点

1. **删掉 pass1**：行 max 不再预扫描，改为 tile 级 running max；K 的 GM 读取次数 2→1。
2. **三状态常驻 UB**：rowMax/rowSum/acc 跨 S2 循环常驻（P38 的 buffer 组织方式），acc 用 fp32。
3. **rescale 顺序**：先算 `alpha = exp(m_old - m_new)`，**先 rescale rowSum/acc 再累加当前 tile**，顺序反了结果错。
4. **p 的 dtype**：PV matmul 前把 p cast 回输入 dtype（fp16/bf16），分子累加仍在 fp32 acc 中。
5. **全 mask 行兜底**：causal 下存在全屏蔽行（m 保持 -INF、l=0），归一化时 `l == 0 → out = 0`，不要除零。
6. **causal 逐 tile 裁剪**：visible 范围按 tile 边界判断，配合 P118 的预生成 dense mask 直接乘加，不要在 token 级 `continue`。
7. **与 cube 链接轨**：单遍结构是 P116/P117 的前置——逐 token 两遍结构无法有效利用 matmul tile。

## 常见陷阱

- ❌ rescale 与累加顺序颠倒：必须 `acc = acc * alpha + p @ V`，写成 `acc = (acc + p @ V) * alpha` 全错。
- ❌ `exp(m_old - m_new)` 中 m_old 初值：必须 -INF（`exp(-INF - m_new) = 0`），用 0 初始化第一块会把 acc 缩放错。
- ❌ 保留逐 token Kahan：online + fp32 已够，Kahan 的每 token 5 次 vector 往返会把流水打碎。
- ❌ 归一化不防零：全 mask 行 `l == 0` 产生 NaN。
- ❌ tile 内先减 tileMax 再合并：p 必须减全局 running max（m_new），只减 tile max 会溢出。

## 代码搜索关键词

```bash
# 检测两遍结构（P119 适用信号）
grep -nE "pass|PASS|for.*S_kv|for.*skv" op_kernel/*.cpp
grep -nE "Exp\\s*\\(|ReduceSum|WholeReduceSum|BlockReduceMax" op_kernel/*.cpp
# 检测是否已是 online（已改写则不适用）
grep -nE "SoftmaxFlash|online|rescale|rowMax|running" op_kernel/*.cpp
```
