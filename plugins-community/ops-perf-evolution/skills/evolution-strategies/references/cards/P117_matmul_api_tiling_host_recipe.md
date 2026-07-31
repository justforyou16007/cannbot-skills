---
id: P117
bottlenecks: [cube_underutilization]
op_families: [attention, flash_attention, matmul, cv_fusion]
complexity: L1
conflicts_with: []
synergizes_with: [P46, P86, P92]
requires: []
has_preconditions: true
has_playbook: true
---

# P117: 双 BMM 链 MatmulApiTiling Host 配方（SetBufferSpace + SetFixSplit + depthA1 流水微调）

## 核心思想

Attention 类算子的 QK^T / PV 是两个 chained matmul。手写 L1/L0 buffer 管理与 K 维切分（Nd2Nz/LoadData/Mmad/Fixpipe 全套）既易错又拿不到框架的自动流水优化；应在 **host 端用 `matmul_tiling::MatmulApiTiling` 分别为 bmm1/bmm2 生成 TCubeTiling**，kernel 侧交给 Matmul 高阶 API 消费（P46）。

三个容易漏掉的配方细节：

1. **`SetBufferSpace(L1, L0C)`**：显式告知真实片上容量，tiling 才能打满 L1 多缓冲；
2. **`d ∈ (64, 128]` 时 `SetFixSplit(baseM=128, baseN=128, baseK=alignUp(d,16))`**：固定 base block 避免默认切分在典型 head_dim 上产生碎块；
3. **depthA1 2→3 微调**：当 `baseK >= singleCoreK`（一个 A tile 已覆盖 K）且 `depthA1 == 2` 时改为 3，让第三级 A1 流水与当前 cube 计算重叠。默认 `GetTiling` 不做此调整，必须手动改。

## 代码骨架

```cpp
// === Host：为 bmm1 (QK^T) 生成 tiling ===
matmul_tiling::MatmulApiTiling bmm1;
bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputDtype, false);
bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputDtype, true);  // K 转置
bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
              matmul_tiling::DataType::DT_FLOAT);                    // fp32 累加输出
bmm1.SetShape(MinHost(s1Base, sq), bmm1N, d);                      // single: (M, N, K)
bmm1.SetOrgShape(sq, s2Base * nRatio, queryRowStride, kvRowStride); // org + 真实行 stride
bmm1.SetBias(false);
if (bmm1.SetBufferSpace(l1Size, l0cSize) != 0) return -1;          // 显式片上容量
if (d > 64 && d <= 128) {                                          // 典型 head_dim 固定切分
    bmm1.SetFixSplit(MinHost(128, AlignUp(sq, 16)),
                     MinHost(128, AlignUp(skv, 16)), AlignUp(d, 16));
}
bmm1.GetTiling(tiling.bmm1TilingData);
// depthA1 2→3：一个 A tile 覆盖 K 时，第三级 A1 流水与 cube 计算重叠
if (tiling.bmm1TilingData.baseK >= tiling.bmm1TilingData.singleCoreK &&
    tiling.bmm1TilingData.depthA1 == 2) {
    tiling.bmm1TilingData.depthA1 = 3;
}
tiling.bmm1TilingData.shareMode = 0;
tiling.bmm1TilingData.shareL1Size = l1Size;
tiling.bmm1TilingData.shareL0CSize = l0cSize;

// bmm2 (PV)：A=P(fp32→cast 回输入 dtype)、B=V 不转置，SetShape(min(s1Base,sq), d, bmm1N)
```

## 关键修改点

1. **bmm1/bmm2 分别建 tiling**：形状、转置标记、OrgShape stride 都不同，不可复用同一个 MatmulApiTiling 对象的结果。
2. **SetOrgShape 的行 stride 必须按真实 layout**（BSH 是 `nq*d`，grouped-M view 是 `d`，见 P116），stride 错 = 整体错位。
3. **CType 用 `DT_FLOAT`**：matmul 输出 fp32 累加，softmax/归一化后再 cast 回输入 dtype（配合 A1）。
4. **SetFixSplit 只在 `d ∈ (64, 128]`** 收益明确；d≤64 默认切分已足够，d>128 走 dOuter 循环。
5. **depthA1 微调条件必须同时满足** `baseK >= singleCoreK && depthA1 == 2`，乱改会破坏流水假设。
6. **TilingData GM→local 拷贝用 8 字节对齐字传输**（`uint64_t` 数组 + 尾部字节补齐），不要逐字节发 GM 事务——direct-launch 场景每次 kernel 启动都要读 tiling，逐字节拷贝是启动开销大头。

## 常见陷阱

- ❌ bmm1 的 BType 忘记转置标记：QK^T 是 `Q @ K^T`，K 必须 `transpose=true`，否则 shape 对不上或结果错误。
- ❌ `SetBufferSpace` 失败不检查返回值：L1/L0C 超配时 GetTiling 产出不可用 tiling，必须显式 fallback（缩小 s1Base/s2Base 重试）。
- ❌ depthA1 无条件改 3：`baseK < singleCoreK`（K 需多轮循环）时改 3 会读越界 A1 buffer。
- ❌ 两个 bmm 共享一份 TCubeTiling：singleCoreN/baseK 不同，kernel 侧必挂。

## 代码搜索关键词

```bash
# 检测双 BMM 链（P117 适用信号）
grep -nE "bmm1|bmm2|qk|pv|matmul" op_host/*_tiling.cpp op_kernel/*.cpp
# 检测是否已用 MatmulApiTiling（已用则转向 P86 调优）
grep -nE "MatmulApiTiling|SetBufferSpace|SetFixSplit|GetTiling" op_host/*.cpp
# 检测手写 cube 搬运（未用高阶 API 的证据）
grep -nE "Nd2Nz|LoadData|Mmad|Fixpipe" op_kernel/*.cpp
```
