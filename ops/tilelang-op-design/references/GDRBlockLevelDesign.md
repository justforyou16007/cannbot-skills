# Chunked Gated Delta Rule（GDR）Block-Level Design

> 算子：Chunked Gated Delta Rule（chunk 版门控 Delta 规则，线性注意力）
> 源实现：`workflows/templates/archive_tasks/chunk_gated_delta_rule/design/block_level/`
> 本文是 [BlockLevelDesign.md](./BlockLevelDesign.md) 的一个算子实例：stateful（状态沿 chunk
> 递推）线性注意力的块级组织。只讲三件事：**三阶段划分、任务划分、求逆的性能优化**。

## 0. 一句话数学

输入 TND 布局：`query/key (T, Nk, Dk)`、`value (T, Nv, Dv)`、`beta (T, Nv)`、`g (T, Nv)`、
`initial_state (B, Nv, Dv, Dk)`、`actual_seq_lengths (B,)`。每条序列按 `chunkSize = 64` 切成 chunk，
状态 `S (Dv, Dk)` 沿 chunk 递推。以一个 chunk（长度 C）为单位：

```text
g_cum     = cumsum(g)                        # (C,)
decay     = exp(g_cum[i] - g_cum[j])         # 下三角（j <= i，因果含对角）
kb = k·β ; vb = v·β
A         = -(kb @ k^T) · decay              # 严格下三角
X         = (I - A)^{-1}                     # 前代求逆
v_inner   = X @ vb
k_cumdecay = X @ (kb · exp(g_cum))
attn_c    = (q @ k^T) · decay                # 因果含对角
v_prime   = k_cumdecay @ S ;  v_new = v_inner - v_prime
attn_inter = (q · exp(g_cum)) @ S
out       = attn_inter + attn_c @ v_new
S        := exp(g_cum[-1])·S + (k·exp(g_cum[-1]-g_cum))^T @ v_new
```

两个关键结构属性：同一 chunk 内 `X = (I-A)^{-1}` 是强序列依赖；不同 `(value-head, chunk)` 互相独立；
`S` 沿 chunk 递推导致 chunk 之间必须串行。

## 1. 三阶段划分

按数据依赖切成三个 stage（对应 AscendC `stage1/2/3.h`），每 stage 结束后全核 `SyncAll`：

- **Stage1（within-chunk 预计算）**：kkt/qkt/g_cum/求逆 X/`v_inner`/`k_cumdecay`/`q_prime`/`kg`。全部
  `(nv, chunk)` 任务按核均分，互不依赖，纯并行。
- **Stage2（跨 chunk 递推）**：读 `curState`，`attn_inter = q_prime @ S^T` 直写 `out`，
  `v_inner += k_cumdecay @ S^T`（原子加），`final_state = g_last·S + v_inner^T @ kg`（原子加）。状态链，
  按 value-head 切核、核内串行 chunk。
- **Stage3（chunk 内贡献累加）**：`attn_c = mask(qkt·decay)·scale`，`out += attn_c @ v_new`（原子加）。
  不依赖状态，无串行，按 `(nv, chunk)` 均分。

Stage1 → Stage2 → Stage3 每段之间 `T.sync_all()`（对应 AscendC `SyncAll<false>`）；段内 AIC↔AIV 用
`T.set_cross_flag`/`T.wait_cross_flag`（信号编号 0..7 对应 AscendC stage 内 event id）。

## 2. 任务划分

状态递推决定块级组织只能是 **chunk-group**：group 内 `(nv, chunk)` 并行、group 之间串行。

```text
blockDim       = aiCoreNum ≈ NUM_PHYSICAL_CORES(20)
maxGroupLength = P_NUM(2) × aiCoreNum(20) × chunkSize(64) = 2560   # 一组覆盖行数
Process()（每核同 body）:
    for bid in 0..b-1:
        seqStart/seqEnd 由 actual_seq_lengths 累计
        for pos in seqStart..seqEnd step maxGroupLength:    # chunk-group 循环
            cg = {startPos:pos, length:min(maxGroupLength, seqEnd-pos)}
            curState = (pos==seqStart) ? initState[b] : finalState[b]
            RunStage1(cg); SyncAll
            RunStage2(cg, curState, finalState[b]); SyncAll
            RunStage3(cg); SyncAll
```

每个 stage 把 `(nv, chunk)` 任务按不同维度分给 20 个核：

- **Stage1/3**：`totalChunk = nv × num_chunk` 按 `coreIdx` **均分**（former/tail），`taskId` 反解
  `(nvId = taskId % nv, cgId = taskId // nv)`。
- **Stage2**：按 `nv` **切核**（`nvPerCore = ceil(Nv/coreNum)`），核内 `for cId` **串行整条 chunk 链**
  ——保证同一 value-head 的状态递推落在同一核上，running 状态自然串联，无跨核状态同步。

## 3. 求逆的性能优化

`X = (I-A)^{-1}`（A 为严格下三角 `(C, C)`，C=64）是 GDR 的瓶颈：前代是串行标量循环，直接全矩阵做
会退化。优化策略是**分块 + C/V 分工**，把一个串行问题切成「向量侧做小对角块、cube 侧做大块矩阵乘」：

```text
A = [[M00, 0 ],      X = [[O00, 0 ],      # O00 = (I+M00)^-1,  O11 = (I+M11)^-1
     [M10, M11]]           [O10, O11]]          # 对角块，向量前代
O10 = O11 @ M10 @ O00                            # off-diag，cube gemm
```

实现（tile_level/stage1.py）：

1. **分块**：`F = C // 2`，A 切成两个对角块 `M00/M11`（各 `(F,F)`）+ 一个 off-diag `M10`（`(F,F)`）。
2. **对角块 O00/O11 = 向量前代**（`T.Scope("V")`，对应 AscendC `InverseAIV`）：对小三角块做串行前代，
   向量单元逐元素处理，避免标量循环退化。
3. **off-diag O10 = O11 @ M10 @ O00 用 cube gemm**（`T.Scope("C")`，对应 AscendC `AttnInverseMMCompute`）：
   `gemm(O11, M10) → round bf16 → gemm(·, O00)`，两个小 gemm 串起。这与对角块的向量前代**并行**，
   AIC/AIV 流水同时跑。
4. **A 保持纯 fp32 进求逆**（对应 AscendC `KKBetaCompute`+`InverseCompute`：`attnUbFloat` 乘完
   β/exp(差) 后不进 bf16 直接 `InverseAIV`）。这是精度关键——若 A 先量化 bf16，量化误差经
   `(I-A)^{-1}` 反演被放大成 1~2.5 ULP，直接导致输出 mismatch（tilelang 曾因此全 case 挂掉，去掉
   cast 后全 PASS）。
