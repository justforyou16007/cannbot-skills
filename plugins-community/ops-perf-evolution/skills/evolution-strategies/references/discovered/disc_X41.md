---
id: X41
origin: discovered
discovered_round: 3
discovered_from: round_3/parallel_2 (QuantMatmul_evo_20260812_145219)
base_speedup: 1.49x canonical / 0.68x raw (vs PyTorch)
---

# Strategy X41: singleCoreN 对齐 baseN 倍数（量化 Matmul L0B 填充利用率修复）

## 核心思路
量化 matmul 大 shape 的 AIC 瓶颈是 **L0B-fill:mad 比 / L2→L1 喂给率**（aic_mte2_ratio≈0.92 近饱和，mac_ratio≈0.45），而非 MAC。当 `MatmulApiTiling` 返回的 `singleCoreN` 不是 `baseN` 的整数倍（如 342 = 256+86）时，每个 block 的最后一个 base-N tile 是 masked partial（86 宽只用 256 宽的 ~1/3），浪费 ~1/3 的 L0B 填充。把 `singleCoreN` **round-up 到最近的 baseN 倍数**（342→512=2×baseN），使每个 block 由整 base-N tile 组成，L0B 利用率 100%，AIC 时间大幅下降。

## 适用场景
- int8/int4 量化 Matmul（Matmul<> 高层 API + MatmulApiTiling），Cube 累加 int32
- 大 M×N shape 且 `singleCoreN % baseN != 0`（N 落在 baseN 间隔中间）
- 评估口径：Cube mm 主导的 shape（epilogue <5%），kernel Duration 衡量
- X11 的补充：X11 聚焦 baseM 扩升，X41 聚焦 singleCoreN 对齐（两个独立杠杆）

## 实现要点
1. 在 `GetTiling + 多核 shrink 循环`之后新增对齐逻辑：`singleCoreN = ceil(singleCoreN / baseN) * baseN`；若对齐后 `totalBlock < aicCoreNum`（核数不足），回退到 `baseN` 本身（如 case8: 342→256 保核心网格）。
2. **不改 singleCoreM**：避免触发 X11 硬约束 `singleM < baseM` 的 AIC 崩溃（case0/9 的 singleCoreM=43 < baseM=48 是 API 原样，不强改）。
3. **对齐过头会变差**：singleCoreN 扩到 2x 对齐（512→1024）时 block 过大破坏 AIC 负载均衡（block 数减半），实测 case2 296→357us 变差——对齐到最小 ≥ singleCoreN 的 baseN 倍数即可，勿过度。
4. SetBufferSpace 对 deep-K shape 无效（depthA1 天然封顶），不做。

## 效果数据（QuantMatmul，Ascend910B2C）
- case2 (16×256×2048, cube 主导): 722→296us（-59%）
- case3 (12×128×1024): 269→122us（-55%）
- case8 (1×1024×8192, deep-K): 145→90us（-38%）
- asc kernel geomean 64.38→50.39us（-21.7%）

## 来源
自动发现于 QuantMatmul 进化 Round-3 第 2 parallel（x0_fix1_x1 open_exploration），
Ascend910B2C/CANN8.5.1，10/10 bit-exact 精度。
