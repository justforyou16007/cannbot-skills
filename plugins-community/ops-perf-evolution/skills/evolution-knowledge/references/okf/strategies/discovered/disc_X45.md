---
id: X45
origin: discovered
discovered_round: 3
discovered_from: "21_GroupedQueryAttention evolution rounds 2-3 (n1 split-into-GEMM-load / n2_cont weight-stationary / x0_x1 ws3-elimination / x0_x1_x1 attention software-pipelining)"
base_speedup: "全量 best 0.657->0.926（r3p2 16-case hard-shape spot, 小/中 case 1.0-1.6x 会拉高全量）; 大 fp32 case A 侧 split-into-load +1.3~10.3%、B 侧 weight-stationary c35 +12%、attention pipelining asc -4.2% A/B"
confidence: single_op
status: active
op_families: [attention, flash_attention, matmul, gqa]
bottlenecks: [mte2_stall, no_overlap, undersize_transfer]
tags: [gqa, prefill, gm-roundtrip, split-into-load, weight-stationary, dual-residency, attention-pipelining, ws3-elimination, l1-residency, workspace-ring]
strategy_ids: [X38, X32, X33, P1, P122]
---

# X38: 融合管线 GM 往返消除（split-into-GEMM-load + B 侧 weight-stationary + attention 软件流水）

## 核心发现（21_GroupedQueryAttention 进化 round 2-3）
大 fp32 投影 GEMM 是 **MTE2(GM->L1)-bound**（aic_mte2_ratio~0.68，cube/fixpipe idle）。
融合管线里每一趟中间张量落 GM 再读回都是纯浪费。三个独立有效的 GM 往返消除杠杆：

1. **A 侧 split-into-GEMM-load**（X32 续，round_2/parallel_0）：
   fp32 激活的 fp16 hi/lo 拆分不再先写 xHi/xLo 到 GM 再读回（2写2读），而是 MIX kernel
   （AIV 在 UB 就地拆，经 workspace ring 交给 AIC；AIC 把整 [64,kPad] A tile 驻留 L1 跨
   column-task 复用）。A 侧 MTE2 从 nNum/row-block 降到 1/row-block。大 fp32 case +1.3~10.3%
   （c44 +10.3%, c47 +9.4%）。**关键工程教训**：初版每 task 的 producer/consumer 让 AIV 对
   同一 A tile 每 by 重拆 nNum 次，回归；改 bx（行块）粒度 + L1 A-stationary 才生效。

2. **B 侧 weight-stationary 混合网格**（round_2/parallel_1, MGROUP=6）：
   任务网格 (bx,by) -> (by, M-group)。B tile（hi/lo 两平面）每 k-step 从 GM->L1 只载一次，
   跨 MGROUP=6 个 M 块复用（B GM->L1 /6，B:A 流量 ~12:1 时 B 是 MTE2 主体）。c35 +12%、
   c37 +4.5%。**混合门槛**：仅当 mNum>MGROUP 且 wsTasks>=核数 才用 WS 网格，小 shape 回退
   扁平网格（WS 额外 MTE1_MTE2 屏障对非 MTE2-bound 是纯开销，c11/c14/c17 回归 8-13%）。
   L0C 128KB 硬约束 MGROUP*64*64*4B <= L0C，MGROUP=6（96KB）为安全值，MGROUP=7（112KB）
   因 M-group 尾块不均回归。

3. **attention 软件流水 + ws3 O-ring 消除**（round_2/parallel_2 + round_3/parallel_2）：
   - **ws3 O-ring 消除**：AIC PV fixpipe 直接写 O 到 attnOut（quantPre F322F16/F322BF16），
     删除 ws3 GM 写+读往返与 AIV ComputeOutput stage，attention 3-ring -> 2-ring 流水。
   - **AIC 软件流水**（round_3/parallel_2）：AIC 每迭代先发 QK^T(task+1) 再算 PV(task)，
     使 QK^T 的 cube 在 PV 等 P 槽期间已进流水；批量 mmad（FireMmad3/2，一次 MTE1_M sync
     发 3 个 hi/lo mmad）消除 per-mmad 标量同步（aic_scalar_ratio ~0.49 是 attention
     latency-bound 的主因）。workspace RING_DEPTH 2->3 让 AIV softmax 更早跑。asc -4.2%
     geomean A/B（c46 -9.8%）。

## 关键负结果（避免重复）
- **A+B 双驻留融合失败**（round_3/parallel_1）：把 A 侧 split-ring 与 B 侧 weight-stationary
  融合成单 MIX kernel 的 gridMode=1 路径回归 6-20%——WS 网格把 N-group 分到各核，每 column-task
  从 ring 重读 A row-block（A 侧 MTE2 ~nGroups x），抵消 A 驻留收益；L1(512KB) 无法同时驻留
  R>1 个 [64,kPad] A tile 与 B tile。**结论：A 驻留与 B 驻留需分别用扁平/WS 网格，不能在同
  一网格内同时做**。
- **MGROUP>6**（7）：L0C 112KB 近满 + M-group 尾块不均（7+7+2），c35/c44 回归。

## 适用场景
- GQA/FA prefill 融合管线，中间张量（split hi/lo、qkv、attnOut、workspace）落 GM 再读回。
- 投影 GEMM MTE2-bound：A 侧 over-read 严重 -> split-into-load + L1 A-stationary；
  B 侧重拉 -> weight-stationary 混合网格（MGROUP 由 L0C 决定）。
- attention latency-bound（aic_cube <15% 且 aic_scalar 高）-> 软件流水 + 批量 mmad + 深 ring。
- workspace 往返 -> 直接消费 cube 输出（fixpipe quantPre 直写终态）。

## 实现要点
- split-into-load 必须按行块（bx）粒度做，A tile L1 驻留跨 column-task 复用；MIX kernel 的
  AIV split 用 mode2 硬件配对（两 AIV 各 set 一次 -> AIC wait 一次）。
- WS 网格的混合门槛是形状自适应（mNum>MGROUP 且 wsTasks>=核数），小 shape 必须回退扁平。
- L0C 预算：MGROUP*64*64*4B <= 128KB，MGROUP=6（96KB）安全。
- attention 软件流水需 RING_DEPTH>=3 给 AIV 超前空间；批量 mmad 背靠背依赖硬件 RAW scoreboard。
- 全部改动保持算子签名不变；精度用 fp16 hi/lo（~2^-21）在 50-case 全量验证。

## 来源
21_GroupedQueryAttention 进化 round 2-3（EVO_DIR=.../GroupedQueryAttention_evo_20260812_102642）：
- n1（split-into-GEMM-load + L1 A-stationary，估全量 0.6636）：round_2/parallel_0
- n2_cont（weight-stationary MGROUP=6，估 0.713）：round_2/parallel_1
- x0_x1（kvPad 自适应 + ws3 O-ring 消除）：round_2/parallel_2
- x0_x1_x1（attention 软件流水 + 批量 mmad + RING_DEPTH=3，16-case hard-spot 0.926）：round_3/parallel_2
- n1_cont（A+B 双驻留融合失败，回退扁平无回归）：round_3/parallel_1
