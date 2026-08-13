# SDPA / 标准 Attention TileLang 构建环境决策（Ascend910B2C 实测）

一句话：标准 `[B,H,S,D]` SDPA（无 mask、Hq==Hkv）在当前 TileLang build 上直接用 **Developer 模式 + num_stages=1**，不要尝试 Expert cross_flag 流水（死锁），也不要开 num_stages=2（alpha 竞态）。

## 先读这个

命中信号：算子是 `matmul(Q,K^T)*scale → softmax → matmul(P,V)` 的标准 SDPA/FlashAttention，
输入 `[B, H, S, D]`，无 block_table/indices/sink/causal（有则先读对应模式文档，再回本文件看构建决策）。
本文件只负责：**TileLang 编程模式选择、pipeline stage 决策、性能上限预期**。block/tile 层级设计模板仍读
`workflows/templates/archive_tasks/flash_attention/`。

## 1. 编程模式路由（2026-08 实测结论）

| 模式 | 结论 | 证据 |
| --- | --- | --- |
| Expert cross_flag 流水 | ❌ **死锁，禁用** | aicore timeout；archive flash_attention 和 GQA 在当前 TileLang build 均 hang（对照实验确认是 build 问题，非算子写法问题） |
| Developer 模式 | ✅ **唯一可用** | `threads=2 + gemm_v0 + T.Pipelined + 4 个 pass_configs`，10/10 case 精度 PASS |

**决策**：跳过 Expert 尝试，直接 Developer 模式起写。不要再为 cross_flag 死锁耗费迭代轮次。

## 2. num_stages 决策

- `num_stages=1`（串行 C1→V1→C2→V2）：✅ 精度正确，当前唯一可用配置。
- `num_stages=2`：❌ Developer 模式下引入 V1(t)/V2(t-1) 的 **alpha（online softmax rescale 系数）竞态**，
  kv tile ≥3 时误差随 tile 数增长。恢复 2 级流水需要 race-free alpha 通道（GM meta 传递），
  但 threads=2 下 GM meta 寻址受限，当前不可行（记录在 PERF_DESIGN V6）。

**性能含义**：大 case Cube 与 Vector 完全串行是主瓶颈。Amdahl 估算：C 侧（2×gemm）与 V 侧
（softmax+merge）时间约 6:4，ns=2 完美 overlap 最多省 40%，大 case speedup 上限 ~0.7~0.9，
geomean 上限 ~0.95。再往上需要 Expert 模式修复或 AscendC 手写流水。

## 3. 性能基线特征（910B2C，50→10 case 精简口径）

- **小 case（Sq≤59）**：1.7~3.2x。reference 在小 shape 上 launch/调度开销占比高，单 kernel 融合优势明显。
- **大 case Cube 路径（fp16/bf16）**：0.46~0.54x，ns=1 串行是主因。
- **大 case fp32 路径**：⚠️ 前提澄清——**TileLang 框架**的 MMA macro 不支持 fp32 输入
  （probe 实证：mma_macro_generator 仅 16/8bit），故 TileLang 阶段 fp32 只能 Vector（0.048~0.21x）；
  但 **910B2C Cube 硬件支持 float×float→float Mmad**（Mmad.md 表7，cube_k=4，吞吐 1/4 fp16）。
  **AscendC 转译阶段 fp32 应直接走 Cube Mmad（已实证，2026-08-12）**：`SdpaKernel<float>`
  复用 fp16/bf16 CV 融合结构、无 Cast 无 hi/lo，case [2,15,239,192]×Skv=275 达 0.522x
  （185us vs 标杆 97us），fp32 谱 geomean 0.32→0.78，全任务 geomean 0.583→0.750。
  实现要点：BASE_K_T=64（双缓冲 L0A/L0B）、C0=8 K 对齐；**小 shape 按 qkElems≥2048 分流
  保留 Vector**（Cube host 固定开销——workspace at::zeros + tiling H2D + MIX launch——在小
  shape 使端到端反转；device kernel time 持平 ≠ 端到端持平，阈值须 A/B 实测）。
  备选：hi/lo bf16 3-MMA（~1/3 fp16）、HF32（1/2 fp16，精度有损需实测）。
  注意 ClearL1→ND2NZ 必须 `SetWaitFlag<MTE1_MTE2>` 守卫（详 cube-impl-patterns.md §8.1）。
- **geomean 基线**：TileLang 层 ~0.80；AscendC 转译后 ~0.58（统一 P hi/lo 后 fp16 略降）。

## 4. 转译阶段精度预案（与 vector-impl-patterns.md 联动）

设计阶段即可预知、转译阶段必现的精度问题：

1. **fp16/bf16 P cast 边界** → P hi/lo 拆分双 MMA（含 V_MTE3 同步、`sizeof(QType)==2` 统一条件），
   详 `skills/tilelang2ascend-translator/references/attention-impl-patterns/vector-impl-patterns.md` §8.1。
   症状：多 tile 大 Skv case matched_ratio≈0.88~0.90，误差全面 1 ULP 分布，单 tile 碰巧通过。
2. **fp32 多 tile workspace 分区** → AIV sub-block 写 workspace 必须带 rowStart_ 偏移（RK-D01）。

## 5. 调试方法论（本次验证有效的定位链）

精度问题按此顺序排查，可避免走弯路：

1. **误差分布分析**：全面 1 ULP 均匀分布 = cast 精度边界（换实现）；局部错行/错块 = 寻址/同步 bug。
2. **dtype 对照**：fp16 通过 + bf16 挂 → bf16 尾数更短，先怀疑 cast 边界；两者都挂 → 查逻辑。
3. **DumpTensor 分段定位**：先确认 S（QK 结果）是否正确，缩小区间。
4. **替换排除法**：手动 fp32 online softmax 替换 SoftmaxFlashV2，结果一致 → 排除 API 嫌疑。
5. **隔离实验**：怀疑双 MMA 通路时，第二次 Mmad 改用 P_hi（ratio 应=2.0）验证通路本身。
