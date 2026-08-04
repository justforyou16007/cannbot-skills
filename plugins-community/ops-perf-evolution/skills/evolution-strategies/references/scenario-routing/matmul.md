# Matmul 系优化工作流（含 CV 融合 / MoE）

> 适用：MatMul / BatchMatMul / MatMulBias / GroupedMatmul / MoE FFN，以及 matmul+向量后处理的 CV 融合算子。

## Step 0 — 族判定确认

命中 [README.md](README.md) 判定表第 3 行（矩阵乘主干 / MatmulImpl·Mmad·MatmulApiTiling 证据）即归本族。注意：**向量/标量模拟的 matmul 也归本族**（那是收益最大的情况）；若 softmax 夹在两个 matmul 中间则归 FA 族。

## Step 1 — 通用前置

打开 [_common-prerequisites.md](_common-prerequisites.md) 过 C1-C5。Matmul 族常见命中：**C1**（向量模拟 matmul → P102/P103，数量级收益，务必最先做）。

## Step 2 — 族内阶段（按序推进）

| 阶段 | 判定条件（满足才做） | 命中用卡 | 跳过条件 |
|---|---|---|---|
| M1 计算路径判定 | kernel 未用 MatmulImpl 高层 API（手写搬运/mmad 时序） | P46（MatmulImpl 高阶 API）；手动 mmad 时序优化 → P87；数据通路效率 → P86 | 已是 MatmulImpl 路径 |
| M2 多核扩展 | `GetTiling` 返回 totalBlock < aicCoreNum（核没吃满） | P104（多核切分扩展，重算 singleCoreM/N） | totalBlock ≥ 核数 |
| M3 L2 切分 | 大算子（M×N×K 超 L2 容量） | P105（L2 tile sizing）→ 配 P107（serpentine 遍历）；K≫M·N 时 P72（Split-K，注意与 P9 确定性互斥） | 算子小，L2 无压力 |
| M4 batch 合并 | batchB==1 且 !transA | P106（batch 合并进 M 维） | 无 batch 冗余 |
| M5 平台路由 | 需跨平台（DAV_2201 / DAV_3510） | P108（平台路由：MatmulImpl+MatmulApiTiling vs Blaze/tensor_api） | 单平台目标 |
| M6 host 配方 | QK^T/PV 式双 BMM 链或 host tiling 可调 | P117（MatmulApiTiling host 配方：SetBufferSpace/SetFixSplit/depthA1） | kernel 侧已最优 |
| M7 CV 流水与核配比 | AIC:AIV 配比失衡 / 双向数据复用差 | P16（代价模型分核）、P17（CV 块设计）、P51（K 维阈值动态核配比）、P73（AIC:AIV 配比）、P75（双 AIV 拆分） | 纯 Cube 算子 |
| M8 缓存与复用 | L1/L0 复用率低、GM 地址冲突 | P71（IBShare L1 共享）、P83（L1 常驻分区，扩展 P53）、P78（片上缓存）、P52（L2 hint，与 P74 互斥）、P93（K 轴错峰访存）、P92（base block 算存比） | 复用已达标 |
| M9 MoE 支路 | GroupedMatmul / 专家并行 | P50（SwiGLU 融合）、P48/P49（量化分发/硬件反量化）、X10（设备侧分组调度 + NZ 权重缓存） | 非 MoE |
| M10 量化 trait | int8/fp8 权重或输出 | 转 [quantization.md](quantization.md) 支路（D4/P48/P49/P70）；小 M GEMV 分流见 X12 | 无量化 |
| M11 反向/梯度 | matmul 反向（dA/dB） | P79（Grad Cube2+Cube3 融合） | 前向算子 |

## 组合约束

- P104 改变单核形状后必须重算 P105 的 L2 tile（[../strategy-compatibility.md](../strategy-compatibility.md) §1）
- P52 与 P74 互斥；P72 与 P9 互斥
- 推荐组合见 compatibility §3：Matmul 深度调优（P46+P71+P83+P86）、多核访存优化（P64+P65+P66）
