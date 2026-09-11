# GroupedMatmul 性能设计检查（PERF_DESIGN）

> 步骤 1e（block 级初检）+ 步骤 2e（tile 级终检）强制产物。
> 平台：910B3（A2 / DAV_2201），cube 核数运行期 GetCoreNumAic（=20）。

## 1. 算子类型判定

**纯 Cube 型**（AIC 独占，AIV 返回）。

依据：本任务为 s-s-s + SPLIT_M + 非量化 fp16/bf16/fp32（ascendc-api-gmm-tiling.md
AIV/AIC 调度比例表：非量化 fp16/bf16/fp32 → Cube Only）。bias 通过 BT bias-init
mma 在 cube 侧随路完成（L0C 从 BT 预置），无独立 vector epilogue，无 CV 融合。

optimization-guide §一 对应优化范围：Cube 型 —— Split-K（本任务 K 方向各组内
串行累加，不 split）、Double Buffer（L1/L0 双缓冲）、Fixed Core（固定物理核数
launch，核内 serial 分多任务）、use_swizzle 任务映射（L2 局部性）。

## 2. block 级反模式检查（1e）

| 检查项 | 结论 | 理由 |
|--------|------|------|
| launch core 数（block_num >> 物理核） | **修正** | 总 block 数 = total_m_blocks×n_num 可远超 20 核。采用 Fixed Core：launch = min(total_blocks, CORE_NUM)，核内 `T.serial` 遍历多任务（matmul 蓝本 idiom） |
| launch core 数（block_num < 物理核仍按物理核 launch） | **修正** | 小 case（如 case8 E=1 M=256 N=256 → 2×1=2 block）按 min(block_num, CORE_NUM) launch，避免空转 |
| CV 未 overlap | 不适用 | 纯 cube，无 CV |
| tile size 过小 | 未命中 | fp16 block_M=128/block_N=256/block_K=64/K_L1=256，L1 用 384KB/512KB、L0A/L0B/L0C 用满，已达容量上限 |
| AIV 未流水 | 不适用 | 纯 cube |

## 3. Cube 高级策略路由（cube_advanced_strategies）

- **选用**：基础 preload 双缓冲流水（L1 S1=2 + L0 S2=2），对齐 catlass
  `MmadAtlasA2PreloadAsync<1,2,{2,4},{4,2},1>` 的 preload=1/L1=2 结构。
- **预留扩展**：分组调度的 block_metadata 表与组内 K 串行结构，不阻碍后续
  pingpong/swat 在多 block 间的扩展；当前规模（单 block 128×256）下基础双缓冲
  已能填满 mma 流水，高级策略收益有限，记入待验证清单 V2。

## 4. CV 锁步风险（cv_sync_debug §4）

不适用（纯 cube，无 C/V 乒乓）。

## 5. 关键设计决策（硬件公理 → 设计）

硬件公理（910B3/A2）：L0A/L0B 64KB、L0C 128KB、L1 512KB、UB 192KB、BT 1KB；
Mmad fp16 16×16×16（C0=16）、fp32 C0=8；搬运 GM→L1→L0→L0C→GM 单向。

| 决策 | 取值 | 公理依据 |
|------|------|----------|
| fp16/bf16 tile | block_M=128, block_N=256, block_K=64, K_L1=256, S1=2, S2=2 | L1: 2×128×256×2 + 2×256×256×2 = 384KB≤512KB；L0A 2×128×64×2=32KB、L0B 2×64×256×2=64KB、L0C 128×256×4=128KB 恰好 |
| fp32 tile | block_M=128, block_N=128, block_K=64, K_L1=128, S1=2, S2=2 | L0C 128×128×4=64KB（fp32 C0=8，N 砖减半）；L1 2×128×128×4×2=256KB |
| 分组调度 | block_metadata 预计算表（host 展开 group_list） | 三维 Kernel 不支持 → 一维 grid + metadata 查表（grouped_gemm 模式）；空组不产 block 天然跳过 |
| bias | BT bias-init mma（T.mma(...,bias=bias_bt)），bias 全程 fp32 | 官方 conv bias 路径；BT 1KB 容 [1,256] fp32=1KB 恰好 |
| transpose_weight | kernel 内 T.copy(B_L1,B_L0,transpose=True) | L1→L0B 转置装载，host 零搬运（反作弊） |
| 尾块 | K 尾零填充（pad_value=0，对累加无贡献）+ M/N 尾 fixpipe 钳位 | matmul 蓝本已实证 idiom；pad 清零防 0×inf=NaN 污染 |

## 性能迭代待验证清单

| # | 待验证项 | 假设 | 验证方法 | 命中时的修改点 | 状态 |
|---|---------|------|---------|---------------|------|
| V1 | 编译器钳位+零填充的尾块 idiom 在 grouped 多 block 共享输出下无竞态 | 各 block 输出区间 [m_start, m_start+valid_m) 不重叠，fixpipe 钳位不写越界 | 全量 case（含质数 M）精度 | 若竞态，改用 metadata valid_m 限制写出范围 | **已闭环（命中）**：ISSUE-2 证实竞态存在，已按修改点改用 valid_m 钳位，全量 PASS |
| V2 | 基础双缓冲已填满 mma 流水，无需 pingpong/swat | 单 block 128×256 的 K 循环足够深 | msprof op PipeUtilization | 若 mma 利用率低，引入 L0A/L0B 4-stage | **已闭环（不命中）**：基线 geomean 3.031x 达标，无需高级策略 |
| V3 | BT bias-init mma 与手写 flag 流水兼容 | bias 在首个 mma（init）预置 L0C，与 k=0,kk=0 的 init 同步 | 带 bias case 精度 | 若冲突，bias 改 epilogue 后加（UB） | **已闭环（命中）**：ISSUE-1 证实手写 flag 缺 MTE1→M 同步致 bias 失效，改 AUTO_SYNC 修复 |

> 步骤 4 性能迭代结果：基线 geomean=3.031x ≥ 0.6x 达标线，基线即达标，未进入
> 迭代循环。见 perf_tuning/{baseline.json,optimization_log.md,final_report.md}。
> 新发现项 [N1] host N-padding 开销（8246us/28%）记入 final_report，留 Phase 4 评估。

---

## 6. tile 级反模式终检（2e）

| 检查项 | 结论 | 理由 |
|--------|------|------|
| 逐元素/逐行 for loop | 未命中 | 全部 T.copy / T.mma 块级原语，无标量/逐行循环 |
| 冗余全局同步 | 未命中 | set_flag/wait_flag 精确依赖（mte1/mte2、m/mte1、fix/m），无 barrier_all；intrinsic 手写流水 |
| 指令未融合 | 未命中 | bias 走 BT bias-init mma（4 操作数融合），无独立 mul+add epilogue |
| 多行 tile 循环内归约 | 不适用 | 无归约算子 |
| AIV 未流水/双 buffer | 不适用 | 纯 cube |

optimization-guide 三类确认项：
- Cube 型：Split-K（本任务各组内 K 串行累加，不 split——组间 M 已并行）✅；
  Double Buffer（L1 S1=2 + L0 S2=2）✅；Fixed Core（min(total_blocks, CORE_NUM)
  launch + 核内 serial）✅。

2e 新增待验证项：无（V1-V3 维持）。

## 7. 设计定稿

tile_level/grouped_matmul.py 为本阶段定稿产物，作为 Phase 4 泛化转译输入。
关键结构（供 Phase 4）：
- 一维 grid + block_metadata[total_m_blocks,3] 查表（g, m_start, valid_m）
- L1 S1=2 / L0 S2=2 双缓冲手写 flag 流水，T.mma 直调
- bias：bias_L1(RowMajor) → bias_bt → 首个 mma bias 操作数
- transpose_weight：B_L1 [block_N, K_L1] 读 + L1→L0B transpose=True
- 尾块：K 尾零填充 + M/N 尾 fixpipe 钳位
