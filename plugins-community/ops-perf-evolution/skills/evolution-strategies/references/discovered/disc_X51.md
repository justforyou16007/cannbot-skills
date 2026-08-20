---
id: X51
origin: discovered
discovered_round: 12
discovered_from: "34_NystromAttention evolution round 12 (n32/n33, parent n29)"
base_speedup: "50-case 全量集 geomean 0.78x → 1.0055x vs PyTorch（n32）+29%；case36-47 大 S 小 landmark 从 0.23-0.34x → 0.62-1.14x（n32）；n33 open exploration 亦达 1.0034x（41/50 valid）"
---

# Strategy X51: 大 S 小 landmark 统一 cube-core 路由阈值扩展

## 核心思路
低秩/近似注意力（Nyström landmark approximation）在 **S 大但 landmark L 小** 的区域
（如 S=463-897、L=2-11），score GEMM（Ql@Kl^T / Q@Kl^T / Ql@K^T）退化为**小 M（landmark）×
大 K（S）** 的窄 GEMM。若路由阈值保守（`useCubeCore` 要求 LP>=16），这些 case 会被留在
legacy vector 路径，性能仅 0.23-0.34x。**把阈值放宽到 LP>=8（或 S>=128 时直接走 cube-core
链）**，让这些 case 进入统一 fp32 cube-core 链（MatmulImpl + SetHF32(false) 全精度），
可提到 0.62-1.14x。

## 适用场景
- 低秩注意力 / landmark-based attention（Nyström、Performer 类）中 S 大而 landmark 小的 case
- 任何"小 M × 大 K"窄 GEMM 被路由到 vector 而 cube 空转的场景
- 判定：score GEMM 的 M 维（landmark/head-dim 组合）低于原阈值但 cube 可用（DHP 对齐）

## 实现要点
1. **阈值放宽**：`useCubeCore` 判定从 `LP>=16` 改为 `LP>=8`（或 `S>=128` 直接进 cube-core），
   覆盖大 S 小 landmark 区域。
2. **精度保障**：fp32 走 MatmulImpl + `SetHF32(false)` 全精度单 GEMM（不要用 fp32 手动 mmad，
   见失败模式：CubeGemm\<float\> 在 PINV 链产生错误结果）。
3. **对齐护栏**：`DHP%16==0` 守卫（DHP=60/78 等非 16 对齐在 cube-core 触发 vector-core 异常
   507035）；非对齐走快/慢双路径。
4. **M 尾块处理**：sub-16 M-tail（S%64==1）CubeGemm 输出 garbage → 合并尾 tile + blockM 4→5。
5. **core_round 非 32B 对齐**：DHP%8!=0 时 vector core exception（case15/17/23 崩溃），
   需快/慢双路径处理。

## 来源
自动发现于第 12 轮进化，算子 34_NystromAttention，50-case geomean 0.78x → 1.0055x
（n32，P108 useCubeCore 扩展）；n33 open exploration 独立验证（LP>=16→LP>=8）。
