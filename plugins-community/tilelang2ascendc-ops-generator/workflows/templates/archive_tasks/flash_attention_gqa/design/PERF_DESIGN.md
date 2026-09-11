# GQA 性能设计检查（PERF_DESIGN）

> 步骤 1e（block 级初检）+ 2e（tile 级终检）结论。算子：GQA（BSND，G 折叠进 M，CV 融合）。

## 1. 算子类型判定（optimization-guide §一）

- **CV 融合型**：含 `IS_ASCEND_AIC`（QK/PV mma）+ `IS_ASCEND_AIV`（softmax/rescale/mask）。
- 优化范围与优先级：先核内（Cube：K/V L1 双缓冲、Q 常驻 L1；Vector：broadcast 向量化、指令融合）→ 再核间（ring 流水 prelaunch=2/ring_slots=3 + 手写 cross flag）→ Fixed Core（grid=20 物理核 + task 守卫）。

## 2. Block 级初检（1e，performance-antipatterns 逐项）

| 检查项 | 结论 | 理由 |
|--------|------|------|
| launch core 数（A：任务多仍按任务数 launch） | 未命中 | Fixed Core：`T.Kernel(20)` + `task_idx < block_num` 守卫（FA 母本结构） |
| launch core 数（B：任务少仍 24 核 launch） | 部分命中 | grid 固定 20；最小 case（B2 Nkv8 S128 G4 → block_num=16）有 4 空核。空核仅进守卫即退，FA 已验证该形态可接受；留作待验证项 V1 |
| CV 未 overlap | 未命中 | ring 流水 prelaunch=2/ring_slots=3 + 三面 cross flag（手写，T.Pipelined/AUTO_CV_COMBINE 按 2026-09 裁决禁用），C1/V1 与 C2/V2 跨迭代重叠 |
| tile size 过小 | 未命中（prefill）/ 结构性存在（decode） | prefill block_m=128 凑满 L0/UB；decode（S=1, G≤32）block_m=ceil16(G) 仅 16~32 行——decode 是 KV-scan DMA bound，M 小不损主路径；留待验证项 V2 |
| AIV 未流水 | 不适用 | 非纯 AIV 算子；V 侧经 ring 与 C 侧重叠 |

## 3. Cube 高级策略路由（cube_advanced_strategies.md）

| 策略 | 结论 | 理由 |
|------|------|------|
| pingpong | 选用（基线） | K/V L1 双缓冲由 ring_slots=3 + AUTO_SYNC/MEMORY_PLANNING 承载；L0 双缓冲依赖编译器规划 |
| swat | 暂不启用 | 任务轮询均分；decode 大 batch（B=128 → 1024 tasks）尾碎片 <2%；留待验证项 V3 |
| streamk | 不适用 | 最小 block_num=16 < 20 的 case 存在（小 prefill），但 K 维（Skv≤128 的 case）不满足 K≥2048；decode 的欠并行不成立（block_num≥128） |
| fullload | 部分采用 | Q 常驻 L1（task 级单 buffer）；K/V 全量驻留不可行（Skv 最长 4096×256×2B=2MB > L1） |
| scale_coalescing | 待验证 | decode 的 per-g Q splice 是 bs=1 行 × bg 次小 DMA（256B~4KB）；V4 |
| mte2_preload | 待验证 | ring prelaunch=2 已有跨 task 重叠；KV 首轮预取收益 V5 |

## 4. CV 锁步风险结论（cv_sync_debug.md §4）

- 流水结构：ring_slots=3 的 GM workspace 三级轮转 + 三面 CrossCoreFlag（S_READY/P_READY/O_READY），C2(t-2) 与 C1(t) 同迭代并发——非"C 写→V 读→下一个 C"串行乒乓。
- 锁步回避要点：cross flag 电平语义（生产不超前两槽以上——ring_slots=3 > prelaunch=2 保证）；vid 分工写进循环边界（`h_start = vid*(block_m//2)`），set_cross_flag 无守卫两 AIV 必达。
- 已知风险：zero-work AIV 守卫只包计算、不许跳过 set（CV 同步三硬规 #2）——tile-level 实现时遵守。

## 5. 性能迭代待验证清单

| # | 待验证项 | 假设 | 验证方法 | 命中时的修改点 |
|---|---------|------|---------|---------------|
| V1 | 小 case（block_num<20）空核开销 | 空核守卫即退，开销可忽略 | msprof 小 case kernel time vs 任务数 launch 变体 | used_core_num=min(20, block_num) |
| V2 | decode block_m=ceil16(G) 的 cube 利用率 | decode 为 KV-scan DMA bound，M 小无损 | msprof decode case 的 MTE2/M 占比 | 若 M bound，考虑 g_blk 合并多 s_blk（S=2/4 时 bs 合并） |
| V3 | 大 batch 尾波负载不均 | 51.2 tasks/core 的 0.2 尾波 <2% | msprof 逐核时间分布 | 任务 swizzle/降序排序 |
| V4 | decode Q per-g splice 小 DMA | bs=1 时 bg 次 256B DMA 可忽略（相对 Skv×D 的 KV 流量） | msprof decode MTE2 busy | s-major 行排布（行=s*bg+g）合并为 1 次 [bg, D] DMA |
| V5 | KV 首轮 MTE2 预取缺口 | prelaunch=2 已覆盖 | 流水图 MTE2 gap | 调整 prelaunch/ring_slots |

## 6. Conv 三模式选型

不适用（非 Conv 算子）。

---

# A3 回炉第一波（P1+P2+P4，2026-09-04）

> 依据 docs/perf_diff_analysis.md。基线 v1 线上 63.43（性能 13.43/50，avg 0.46 vs A3-FIAS），
> 本地 A2 geomean 2.24x。本波只做 P1+P2+P4；P3（pto M2 崩溃未解）/P5（ISSUE-6 承重修复）挂起。
> 专家参照：catlass/examples/23_flash_attention_infer/（block_mmad_fai_qk_normal.hpp /
> block_mmad_fai_pv_normal.hpp / fai_kernel.cpp）。

## R1. 三项改动的机制与我方可行形态（容量核算驱动）

### P4（causal mask 对角块分离 + 整 tile 向量化）—— 独立低风险

- **现状**：对角块内 bm2 次串行 `CompareScalar+Select` + 每次 2 个 `PipeBarrier<PIPE_V>`
  （ascendc L579-590；bm2=64 时 64 次串行 + 128 次 barrier）。
- **专家形态**：非对角块零 mask（diffS 循环裁剪已有）；对角块整 tile 一次
  `CreateVecIndex 列索引 + 行 bound broadcast + 单次 Compare/Select`。
- **我方可行形态**：对 **sub_per_band**（bs>=bm2，prefill 主收益面 case0/2/7/9/18/19），
  半核 bm2 行在单 band 内（bs=2*bm2），`s_local = h_start + h_i` **线性无折返**，
  行 bound = `s0 + h_start + diff_s + h_i` 是线性序列 → `createvecindex` 直接生成
  [bm2] bound 向量，broadcast 成 [bm2, block_n] 与列索引整 tile `compare(LE)` +
  `select(MASK_NEG)` 各一次。对 **bands_per_half**（bs<bm2，decode/MTP case15/16），
  s_local=r%bs 周期折返，用 createvecindex 生成行号后逐元素取模构造 bound（bs 小，
  仅 2 个 causal case，非主收益面；若取模向量化表达困难可保留逐行，记待验证）。
- **容量**：新增 [bm2] bound 向量 + [bm2, block_n] 比较掩码（uint8）+ broadcast 临时，
  UB 增量 < bm2*block_n 字节级，最劣 d128（bm2=64）约 +32KB，UB 183KB→215KB 超 192KB！
  **容量约束**：比较掩码复用 tmp_ub（bm2*bn*2 uint8 已存在），bound 向量复用 col_idx
  槽位（mask_col/col_idx 在非对角块闲置）。需重核 UB（见 R2）。

### P2（K/V L1 单缓冲→双缓冲 STAGES=2）—— 中风险，收益明确

- **现状**：C1 `copy_gm_to_l1(k_l1)` 单缓冲，GM→L1 的 MTE2 在 mma 关键路径上
  （每次等 MTE2 完成才 L1→L0→mma）。C2 的 V 加载同理。
- **专家形态**：`l1BTensor[STAGES=2]`，`l1KvPingPongFlag = nkblockIdx%2`；处理当前块时
  预取下一块到另一槽（`WaitFlag<MTE1_MTE2>(1-flag)` → `copyGmToL1B(1-flag, next)` →
  `SetFlag<MTE2_MTE1>(1-flag)`），当前块 mma 前 `WaitFlag<MTE2_MTE1>(flag)` →
  `copyL1ToL0B` → `SetFlag<MTE1_MTE2>(flag)`。MTE2(GM→L1) 与 MTE1(L1→L0)+mma 完全重叠。
- **我方可行形态**：k_l1/v_l1 改 [2, block_n, dim_align] 双槽，ring 内预取下一 KV 块。
  引入 MTE1_MTE2/MTE2_MTE1 旗标对（专家 :247-258 模式）。**这违背"同步逐字节保留"，
  需重新精度验证**（perf_diff_analysis L19 已预警）。
- **L1 容量**：双缓冲 K/V + Q + S = 2*(block_n*da) + block_m*da + block_m*block_n (T 字节)。
  最劣 d128 prefill（bm=128, da=128, fp16）：2*128*128*2 + 128*128*2 + 128*128*2
  = 64K+32K+32K = 128KB... 实际 K 双槽+V 双槽+Q+S = 2*32K(K)+2*32K(V)+32K(Q)+32K(S)=192KB
  < 512KB ✓。L1 容量充裕。

### P1（KV 基块 128→512 堆叠）—— 高风险，prefill 受 L0C 容量限制

- **专家形态**：ring 外层步进 blockStackNum=4（每步 512 列）；C1 内层 blockStackIdx 0..3
  子循环，4 块 S 在 L0C 内列堆叠（stackTile 偏移，LayoutC(rowNum,stackTile,512)），
  L0C 2 半区乒乓（locPingPongFlag），每 512 列一对 cross flag + 合并 fixpipe。
- **我方 L0C 容量硬约束**（fp32 累加器，L0C=128KB）：
  - L0C 512 列堆叠 [block_m,512]：d128 prefill bm=128 → 256KB ✗；d256 bm=64 → 128KB 满
    且还需 acc_o_l0c ✗；decode bm=16 → 32KB ✓。
  - L0C 256 列半堆叠（2 半区，fixpipe÷2）：d128 bm=128 → 128KB 满 + acc_o ✗。
  - **结论：L0C 堆叠仅 decode（block_m<=32）可行，prefill（bm=128/64）不可行。**
- **我方可行形态（prefill 适配）**：不做 L0C 内堆叠，只做 **GM workspace_s 512 列连续槽
  + cross flag ÷4**。ring 外层步进 4，C1 内层 4 块各独立 mma(init=True)（保留 d_chunks
  K 维 chunk 规避 pto M2）+ 各 fixpipe 到 ws_s 的 512 列槽的不同 128 列偏移，4 块齐后
  才 set 一对 SIG_S_READY。V1 等 512 列齐，分 4 次读 128 列子块做 online softmax
  （UB 仍按 128 列子块控容量，perf_diff_analysis L29）。
  - 收益：cross flag(C1→V1) ÷4；4 次 fixpipe 连续发起（MTE 流水）部分摊薄启动开销；
    ws_s GM 局部性改善（V1 连续读 512 列）。
  - 拿不到：L0C 堆叠的 fixpipe 次数 ÷2/÷4（prefill 容量不可行）。
- **decode 增强**：decode（block_m<=32）可进一步做 L0C 堆叠。但 decode 的 KV 流量主导，
  cross flag ÷4 收益已覆盖主要部分；L0C 堆叠的边际收益留作待验证项（不硬塞）。
- **风险**：ring 结构重写（步进 4 + 内层子循环 + 尾块不足 4 块的钳位 + causal kv_end
  与 stack 粒度交互 + cross flag 电平语义在 512 列粒度下的握手配对）。同步是承重墙
  （ISSUE-5/6 前科），必须逐字节对照已验证产物。

## R2. 容量核算表（P1+P2+P4 叠加后，最劣配置）

待 P1/P2/P4 设计落定后重核 UB/L1/L0C，写入本节与 GENERALIZE_REPORT.md。
（UB 关键：P4 的整 tile 掩码需复用既有 buffer 控增量；P2 的 L1 双缓冲容量已核 ✓）

## R3. 实施顺序与分项验证策略

任务要求 P1/P2 同动 ring 内层一起重构。执行上分项落地、逐项验证归因
（允许分项收敛，禁止硬塞）：
1. **P4 先行**（独立低风险）：只改 V1 mask 段，不动 ring/C1/C2。验证因果 mask 正确性。
2. **P2 叠加**（中风险）：改 C1/C2 的 K/V GM→L1 为双槽预取 + MTE1_MTE2/MTE2_MTE1 旗标对。
3. **P1 收尾**（高风险）：ring 步进 4 + ws_s 512 列槽 + cross flag ÷4。
每项以 evaluate_tilelang 功能验证（allclose_ok 口径，matched_ratio<0.9 为 ISSUE-7
已知固有极限不作门禁）+ 泛化后官方 compare_tensors 分级验证 + 本地 perf 逐 case 对比
归因收益。无收益/回退项记录证据并回滚。
