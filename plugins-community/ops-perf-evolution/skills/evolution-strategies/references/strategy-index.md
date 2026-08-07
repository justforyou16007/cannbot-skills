# AscendC Kernel Optimization Strategy Index

Source: 43 production-grade operators + 6 CV fused operator expert implementations (IdeaPool). Select all applicable strategies for your operator, then **Read each referenced detail file** before writing AscendC code.

## 编号说明
策略卡共 146 张：P 系列（性能，P1-P131）/ D 系列（数据类型，D1-D5）/ A 系列（精度，A1-A8）/ R 系列（A5 架构，R1-R8）。P 系列编号存在间隙（P15, P23, P27, P31, P36, P39），为预留位。
所有有效策略 ID 均在下方表格中列出。Agent 在 strategy_combination 中只应使用下方表格中的 ID。

## Layer 定义
- **L0 (Universal)**: 适用于几乎所有算子的通用策略（D1-D5, P1-P12, A1-A8）
- **L1 (Advanced)**: 适用于特定类别算子的高级策略（P14-P126）
- **L2 (Discovered)**: 进化过程中发现的新策略（X1+）

## Tags 体系
- `datapath`: 通用数据搬运通路优化
- `cv_fusion`: Cube+Vector 融合算子
- `flash_attention`: Flash Attention 系列
- `quantization`: 量化相关
- `moe`: MoE 模型特定
- `matmul`: 矩阵乘相关
- `manual`: 需要手动调优

---

## 多数据类型支持策略 (D1-D5) [Layer: L0]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| D1 | Mixed precision architecture | L0 | datapath | Any op with FP16/BF16 input | playbooks/D1_mixed_precision.md |
| D2 | Template kernel type dispatch | L0 | datapath | Compile-time multi-type kernel | playbooks/D2_template_kernel.md |
| D3 | TilingKey-driven type dispatch | L0 | datapath | Host-side multi-type selection | playbooks/D3_tilingkey_dispatch.md |
| D4 | FP8/INT4 quantization conversion | L0 | quantization | Quantization output operators | playbooks/D4_fp8_int4_conversion.md |
| D5 | BF16-specific handling | L0 | datapath | BF16 output with rounding control | playbooks/D5_bf16_special_handling.md |

## 性能优化策略 — 通用 (P1-P12) [Layer: L0]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P1 | Double buffering (BUFFER_NUM=2) | L0 | datapath | Any memory-bound kernel | playbooks/P1_double_buffering.md |
| P2 | Adaptive tiling (Split-N vs Split-D) | L0 | datapath | Variable row/column dimensions | playbooks/P2_adaptive_tiling.md |
| P3 | Small-D multi-row merging | L0 | datapath | Hidden size ≤ 640 or small D | playbooks/P3_small_d_optimization.md |
| P4 | Multi-core load balancing | L0 | datapath | Uneven data distributions | playbooks/P4_load_balance.md |
| P5 | Pipeline sync (PipeBarrier/events) | L0 | datapath | All double-buffered kernels | playbooks/P5_pipeline_sync.md |
| P7 | 32B alignment + DataCopyPad | L0 | datapath | Non-aligned input shapes | playbooks/P7_data_alignment.md |
| P8 | UB memory partitioning | L0 | datapath | Kernels with multiple UB tensors | playbooks/P8_ub_memory_partitioning.md |
| P9 | Deterministic output (workspace) | L0 | datapath | Training ops needing reproducibility | playbooks/P9_deterministic_output.md |
| P10 | Vectorized data copy | L0 | datapath | CopyIn/CopyOut optimization | playbooks/P10_vectorized_copy.md |
| P11 | Tail block handling (GatherMask) | L0 | datapath | Pooling/gather with uneven splits | playbooks/P11_tail_block_handling.md |
| P12 | Broadcast & mask operations | L0 | datapath | Operators with mask inputs, broadcasting | playbooks/P12_broadcast_mask.md |

## 性能优化策略 — CV融合/高级调优 (P14-P18) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P14 | CV pipeline preload (3-slot task cache) | L1 | cv_fusion, flash_attention | Flash Attention Cube-Vector 流水预发射 | playbooks/P14_cv_pipeline_preload.md |
| P16 | Cost-driven core partition | L1 | cv_fusion, manual | Alignment-aware 代价模型分核 | playbooks/P16_cost_driven_core_partition.md |
| P17 | CV parallel block sizing | L1 | cv_fusion | CV 并行基本块设计 | playbooks/P17_cv_parallel_block_sizing.md |
| P18 | L1 7-buffer resident partition | L1 | cv_fusion, flash_attention | L1 七缓冲常驻分区 | playbooks/P18_l1_7buf_resident_partition.md |

## 性能优化策略 — 高级数据搬运 (P19-P45) [Layer: L1]

### 队列管理与双缓冲 (P19-P22)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P19 | Custom ping-pong double buffer | L1 | datapath | 自定义 Ping-Pong 双缓冲管理 | playbooks/P19_custom_pingpong_double_buffer.md |
| P20 | Triple buffer rotation | L1 | datapath | 三缓冲轮转（搬入/计算/搬出完全重叠） | playbooks/P20_triple_buffer_rotation.md |
| P21 | Matrix 2×2 buffer policy | L1 | datapath | 2×2 矩阵缓冲策略 | playbooks/P21_matrix2x2_buffer_policy.md |
| P22 | TQueBind bidirectional reuse | L1 | datapath | TQueBind 双向复用 | playbooks/P22_tquebind_bidirectional_reuse.md |

### DataCopy 参数与对齐 (P24-P26)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P24 | ND2NZ format copy | L1 | datapath | ND→NZ 格式转换搬运 | playbooks/P24_nd2nz_format_copy.md |
| P25 | DataCopy pad alignment | L1 | datapath | DataCopy 填充对齐优化 | playbooks/P25_datacopy_pad_alignment.md |
| P26 | Stride fused transpose | L1 | datapath | 步长融合转置 | playbooks/P26_stride_fused_transpose.md |

### 同步与流水控制 (P28-P30)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P28 | HardEvent producer-consumer | L1 | datapath | HardEvent 生产者-消费者同步 | playbooks/P28_hardevent_producer_consumer.md |
| P29 | Cross-core AIC-AIV sync | L1 | cv_fusion | 跨核 AIC-AIV 同步 | playbooks/P29_crosscore_aic_aiv_sync.md |
| P30 | Dense PipeBarrier vector | L1 | datapath | 密集 PipeBarrier 向量同步 | playbooks/P30_dense_pipebarrier_vector.md |

### 特殊搬运模式 (P32-P33)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P32 | Scatter block stride | L1 | datapath | 散射块步长搬运 | playbooks/P32_scatter_block_stride.md |
| P33 | Gather offset table | L1 | datapath | 聚集偏移表搬运 | playbooks/P33_gather_offset_table.md |

### 缓冲驻留与复用 (P34-P45)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P34 | Weight load-once reuse | L1 | datapath | 权重一次加载多次复用 | playbooks/P34_weight_load_once_reuse.md |
| P35 | TBuf resident accumulator | L1 | datapath | TBuf 常驻累加器 | playbooks/P35_tbuf_resident_accumulator.md |
| P37 | Grad accumulator cross-batch | L1 | datapath | 梯度跨 batch 累加器 | playbooks/P37_grad_accumulator_cross_batch.md |
| P38 | Softmax state resident | L1 | flash_attention | Softmax 状态常驻 buffer | playbooks/P38_softmax_state_resident.md |
| P40 | Workspace double buffer | L1 | datapath | Workspace 双缓冲 | playbooks/P40_workspace_double_buffer.md |
| P41 | Gamma/scale resident | L1 | datapath | Gamma/scale 常驻 buffer | playbooks/P41_gamma_scale_resident.md |
| P43 | Grad ping-pong deferred scatter | L1 | datapath | 梯度 Ping-Pong 延迟散射 | playbooks/P43_grad_pingpong_deferred_scatter.md |
| P44 | Transpose buffer resident | L1 | datapath | 转置 buffer 常驻 | playbooks/P44_transpose_buffer_resident.md |
| P45 | TopK index cache iterator | L1 | datapath | TopK 索引缓存迭代器 | playbooks/P45_topk_index_cache_iterator.md |
## 性能优化策略 — CV融合算子架构模式 (P46-P52) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P46 | MatmulImpl high-level API | L1 | cv_fusion, matmul | 替代手写 matmul 流水线 | playbooks/P46_matmulimpl_highlevel_api.md |
| P47 | Diagonal block scheduling | L1 | cv_fusion, matmul | 多核 M×N 分块 L2 cache 优化 | playbooks/P47_diagonal_block_scheduling.md |
| P48 | Multi-quantization mode dispatch | L1 | quantization, matmul | 多量化模式编译期分发 | playbooks/P48_multi_quant_dispatch.md |
| P49 | Hardware-accelerated dequantization | L1 | quantization | AscendDequant 硬件反量化 | playbooks/P49_hw_dequantization.md |
| P50 | SwiGLU fusion pipeline | L1 | moe, cv_fusion | MoE FFN SwiGLU 融合 (narrow scope) | playbooks/P50_swiglu_fusion.md |
| P51 | Dynamic AIC/AIV core ratio | L1 | cv_fusion, matmul | K 维阈值动态核配比 | playbooks/P51_dynamic_aic_aiv_ratio.md |
| P52 | L2 Cache Hint optimization | L1 | cv_fusion, matmul | 多核竞争场景 L2 禁用 | playbooks/P52_l2_cache_hint.md |

## 性能优化策略 — 片上缓存与常驻复用 (P53-P56) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P53 | L1 resident reuse | L1 | cv_fusion, flash_attention | Q 矩阵 L1 常驻跨迭代复用，消除重复搬运 | playbooks/P53_l1_resident_reuse.md |
| P54 | Launch overhead optimization | L1 | datapath | 减少 kernel launch 开销，合并小 kernel | playbooks/P54_launch_overhead_optimization.md |
| P55 | Block reduce-max optimization | L1 | datapath | 块级 ReduceMax 优化，二分累加 | playbooks/P55_block_reduce_max_optimization.md |
| P56 | DataCopy batch output | L1 | datapath | 批量 DataCopy 输出，减少搬运次数 | playbooks/P56_datacopy_batch_output.md |

## 性能优化策略 — FlashAttention/Decode 专用 (P57-P58) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P57 | FlashDecode G-partition reduction | L1 | flash_attention | FlashDecode G 维分区归约 | playbooks/P57_flashdecode_g_partition.md |
| P58 | TND load-balanced partition | L1 | flash_attention | T/N/D 维负载均衡分区 | playbooks/P58_tnd_load_balance_partition.md |

## 性能优化策略 — 数据搬运与格式优化 (P59-P66) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P59 | Output transpose fusion | L1 | datapath | 输出转置融合，减少额外搬运 | playbooks/P59_output_transpose_fusion.md |
| P60 | KV NZ format optimization | L1 | matmul, flash_attention | KV cache NZ 格式优化 | playbooks/P60_kv_nz_format_optimization.md |
| P61 | L2 cache optimization | L1 | flash_attention | 关闭双页表开启 L2 Cache，KV 数据缓存 | playbooks/P61_l2_cache_optimization.md |
| P63 | Iterate async interface | L1 | matmul | 异步迭代接口，隐藏搬运延迟 | playbooks/P63_iterate_async_interface.md |
| P64 | GM address conflict avoidance | L1 | datapath | GM 地址冲突规避，优化多核访存 | playbooks/P64_gm_address_conflict_avoidance.md |
| P65 | UB bank conflict avoidance | L1 | datapath | UB bank 冲突规避，优化地址分配 | playbooks/P65_ub_bank_conflict_avoidance.md |
| P66 | GM 512B alignment bandwidth | L1 | datapath | GM 512B 对齐带宽优化 | playbooks/P66_gm_512b_alignment.md |

## 性能优化策略 — Vector 计算效率与标量向量化 (P67-P69, P94-P103) [Layer: L1]

P67-P69 为 Vector 计算效率基础策略（Counter 模式、低延迟归约、UB 融合链）。P94-P101 为标量→向量专项替代策略，覆盖归约（P94）、逐元素运算（P95）、排序（P96）、零拷贝切片（P97）、自适应调度（P98）、Broadcast 批量（P99）、GatherMask 标量替代（P100）、GatherMask 列分量解交织（P101）八种反模式。P102-P103 为架构级策略：当标量/向量在**模拟矩阵运算**（matmul/dot/三角求解）时，向量化收益封顶，需 MIX CV 融合重写（P102）及 Cube Neumann 三角求逆（P103）。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P67 | Vector counter mode | L1 | datapath | Vector counter 模式，Scalar 控制流优化 | playbooks/P67_vector_counter_mode.md |
| P68 | Low-latency reduction | L1 | datapath | 低延迟归约指令组合 (BlockReduceSum+WholeReduceSum) | playbooks/P68_low_latency_reduction.md |
| P69 | UB fused vector chain | L1 | datapath | UB 融合连续 Vector 计算链，中间结果保留 UB | playbooks/P69_ub_fused_vector_chain.md |
| P94 | Scalar reduction → Vector Reduce API (ReduceSum/ReduceMax/WholeReduceSum) | L1 | datapath | 逐元素标量归约循环（for(i) sum+=x[i]），Reduce/Softmax/Norm 算子 | cards/P94_scalar_reduction_to_vector.md |
| P95 | Scalar element-wise → Adds/Muls vector chain | L1 | datapath | 逐元素标量算术循环（for(i) y[i]=f(x[i])），Elementwise/Norm 算子 | cards/P95_scalar_elementwise_to_vector.md |
| P96 | Hardware sort engine (Sort32/MrgSort) replacement | L2 | datapath | 标量冒泡/选择排序或 TopK 选择，Sort/TopK/NMS 算子 | cards/P96_hardware_sort_replacement.md |
| P97 | Zero-copy sub-tensor slicing + batch vector ops | L1 | datapath | 从 UB tensor 按偏移取子集做批量运算，需避免逐元素 gather | cards/P97_zero_copy_subtensor.md |
| P98 | Hybrid scalar/vector adaptive dispatch (per-tile workload) | L1 | datapath | 小 tile 向量化收益为负（EnQue/DeQue 开销>标量循环），需自适应切换 | cards/P98_hybrid_scalar_vector_dispatch.md |
| P99 | BroadCast + batch vector op replaces row-wise scalar loop | L1 | datapath | 小向量广播到大矩阵做逐元素运算，避免逐行 Muls/Adds dispatch 开销 | cards/P99_broadcast_batch_vector_ops.md |
| P100 | GatherMask replaces element-wise SetValue/GetValue | L1 | datapath | 按固定间隔从连续 tensor 取元素重组（解交织、RoPE 奇偶分拆等） | cards/P100_gathermask_replacement.md |
| P101 | GatherMask replaces host-side column split (`.select().contiguous()`) | L1 | datapath | interleaved (N,4)→4 路分量分离，消除 Host 侧列拆分和多路 DMA | cards/P101_gathermask_host_column_split.md |
| P102 | Scalar/vector matmul emulation → MIX CV fusion rewrite | L1 | cv_fusion, matmul, attention, special | 标量/向量循环模拟矩阵运算（嵌套 dot、伪 matmul、三角求解），维度 ≥64 → 架构级重写（数量级收益） | cards/P102_scalar_matmul_to_cv_fusion_rewrite.md |
| P103 | Cube Neumann product triangular solve | L1 | cv_fusion, attention, special | MIX 架构下标量 forward substitution 串行链 → (I-X)⁻¹=Π(I+X^{2^i}) 11 Mmad 替代 | cards/P103_cube_neumann_triangular_solve.md |

## 性能优化策略 — Matmul/量化专用 (P70-P72) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P70 | FixPipe buffer inline quantization | L1 | matmul, quantization | FixPipe buffer 内联量化 | playbooks/P70_fp_buffer_inline_quant.md |
| P71 | Matmul IBShare L1 sharing | L1 | matmul | Matmul IBShare L1 共享，减少重复搬运 | playbooks/P71_matmul_ibshare_l1.md |
| P72 | Matmul Split-K | L1 | matmul | K 轴切分多核并行，AtomicAdd 累加 | playbooks/P72_matmul_split_k.md |

## 性能优化策略 — CV 并行与核配比 (P73-P75) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P73 | CV parallel AIC:AIV ratio | L1 | cv_fusion | AIC:AIV 核配比与多 Workspace 流水 | playbooks/P73_cv_parallel_aic_aiv_ratio.md |
| P74 | Selective L2 cache disable | L1 | datapath | 选择性 L2 cache 禁用，减少多核竞争 | playbooks/P74_selective_l2_cache_disable.md |
| P75 | Dual AIV workload split | L1 | cv_fusion | 双 AIV M/S1 轴负载拆分 | playbooks/P75_dual_aiv_workload_split.md |

## 性能优化策略 — Sparse Attention 专用 (P76-P78) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P76 | V_TEMPLATE KV pre-merge workspace | L1 | flash_attention | KV 预合并 Workspace 管理 | playbooks/P76_v_template_kv_premerge.md |
| P77 | L1 chunk iterator sparse aggregation | L1 | flash_attention | L1 分块迭代器稀疏自适应聚合 | playbooks/P77_l1_chunk_iterator_sparse_aggregation.md |
| P78 | On-chip cache matmul acceleration | L1 | matmul | 片上缓存加速 Matmul | playbooks/P78_onchip_cache_matmul_acceleration.md |

## 性能优化策略 — 高级流水与融合 (P79-P84) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P79 | Grad Cube2+Cube3 fusion | L1 | matmul | 梯度 Cube2+Cube3 融合 | playbooks/P79_grad_cube23_fusion.md |
| P80 | Sink task pipeline injection | L1 | flash_attention | Sink task 流水注入与 S2 block skip | playbooks/P80_sink_task_pipeline_injection.md |
| P81 | Resident UB/TBuf reuse | L1 | datapath | 小参数/状态常驻 UB/TBuf，跨迭代复用 | playbooks/P81_resident_ub_tbuf_reuse.md |
| P83 | L1 resident reuse partition | L1 | matmul | L1 常驻复用多缓冲分区 | playbooks/P83_l1_resident_reuse_partition.md |
| P84 | Vector compute efficiency patterns | L1 | datapath | Counter 模式+低延迟指令+UB 融合链 | playbooks/P84_vector_compute_efficiency_patterns.md |

## 性能优化策略 — 缓冲区复用与 Matmul 调优 (P85-P88) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P85 | On-chip buffer zone reuse | L1 | datapath | 片上 buffer 分区复用 | playbooks/P85_on_chip_buffer_zone_reuse.md |
| P86 | Matmul API transport efficiency tuning | L1 | matmul | Matmul 内部数据通路效率调优 | playbooks/P86_matmul_api_transport_efficiency_tun.md |
| P87 | Manual mmad pipeline timing control | L1 | matmul, manual | 手动 Mmad 流水时序控制 | playbooks/P87_manual_mmad_pipeline_timing_control.md |
| P88 | Multiphase compute phase decomposition | L1 | cv_fusion | 多阶段计算拆分，独立 TPipe 资源重分配 | playbooks/P88_multiphase_compute_phase_decomposit.md |

## 性能优化策略 — UB 共享与 Matmul 微调 (P89-P93) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P89 | UB high-level API temp buffer sharing | L1 | datapath | SoftMax 等高阶 API 与计算阶段 TBuf 分时复用 | playbooks/P89_ub_api_temp_buffer_sharing.md |
| P90 | Conditional PipeBarrier by matrix size | L1 | matmul, flash_attention | 手动 Matmul L0C 累加模式按 fractal 数动态同步 | playbooks/P90_conditional_pipebarrier.md |
| P91 | Softmax gradient pre-computation phase | L2 | flash_attention | FA 反向 dY*O ReduceSum 预计算为独立 VecSfmg 阶段 | playbooks/P91_softmax_grad_precompute.md |
| P92 | Matmul base block compute-to-memory ratio | L1 | matmul, cv_fusion | 优化 baseM/baseN/baseK 最大化 Cube 计算访存比 | playbooks/P92_matmul_block_compute_ratio.md |
| P93 | Matmul K-axis staggered memory access | L1 | matmul | 多核 Matmul K 轴错峰访问缓解 GM 地址冲突 | playbooks/P93_matmul_k_stagger_access.md |

## 性能优化策略 — Ascend C Tiling 系统化设计 (P104-P115) [Layer: L1]

这些策略来自 `ascendc-tiling-design` skill 的系统化 Tiling 方法论，聚焦现有 P 系列未覆盖的 Tiling 设计维度（多核扩展、L2 切分、Batch 合并、Serpentine、平台路由、归约/排序/逐元素/广播/转置/FlashAttention 的 Tiling 流程）。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P104 | Matmul multi-core tiling expansion | L1 | matmul, tiling | `MatmulApiTiling::GetTiling` 返回 `totalBlock < aicCoreNum`，需手动缩小 singleCoreM/N 以充分利用多核 | playbooks/P104_matmul_multi_core_tiling_expansion.md |
| P105 | Matmul L2 cache tile sizing | L1 | matmul, tiling | 大算子启用 L2 级切分，设置 `mTileBlock×nTileBlock` 并配合 Serpentine 反向遍历 | playbooks/P105_matmul_l2_cache_tile_sizing.md |
| P106 | Matmul batch-merge-to-M tiling | L1 | matmul, tiling | `batchB==1 && !transA` 时将 batch 维度合并到 M，提升数据局部性 | playbooks/P106_matmul_batch_merge_to_m.md |
| P107 | Matmul serpentine traversal scheduling | L1 | matmul, tiling | L2 Tile 内 base 块 serpentine 反向遍历，提升 A/B 数据复用 | playbooks/P107_matmul_serpentine_traversal.md |
| P108 | Matmul platform routing | L1 | matmul, tiling | 根据 NpuArch 选择 `MatmulImpl`+`MatmulApiTiling`（DAV_2201）或 Blaze/tensor_api（DAV_3510） | cards/P108_matmul_platform_routing.md |
| P109 | Reduction multi-axis tiling | L1 | reduction, tiling | 多轴归约场景下的轴合序、AR/ARA 分阶段 Tiling 设计 | playbooks/P109_reduction_multi_axis_tiling.md |
| P110 | Reduction UB accumulator reuse | L1 | reduction, tiling | 归约累加器在 UB 中常驻复用，减少中间结果搬运 | playbooks/P110_reduction_ub_accumulator_reuse.md |
| P111 | Sort two-level merge tiling | L1 | sort, tiling | Sort/TopK 多级归并 Tiling，边界填充与索引 buffer 规划 | playbooks/P111_sort_two_level_merge_tiling.md |
| P112 | Elementwise vectorized tile sizing | L1 | elementwise, tiling | 逐元素算子的向量化 tile 大小选择、内存合并与多核分发 | playbooks/P112_elementwise_vectorized_tile_sizing.md |
| P113 | Broadcast aligned tiling | L1 | broadcast, tiling | 广播算子的对齐 Tiling、向量化加载与内存合并 | playbooks/P113_broadcast_aligned_tiling.md |
| P114 | Transpose ND tiling with alignment | L1 | conversion, tiling | ND 转置的轴映射、对齐填充与 32B/512B 对齐搬运 | playbooks/P114_transpose_nd_tiling_with_alignment.md |
| P115 | FlashAttention tiling routing | L1 | flash_attention, tiling | FlashAttention 类算子的 Tiling 模式路由（标准/GQA/MLA/paged 等） | playbooks/P115_flashattention_tiling_routing.md |

## 性能优化策略 — GQA/FlashAttention 实战优化 (P116-P119) [Layer: L1-L2]

这些策略覆盖 attention 类算子（尤其 GQA decode）的四个结构性优化点：G→M 维合并、host 端 matmul tiling 配方、mask 预生成剥离、online softmax 改写。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P116 | GQA grouped-M head merge | L1 | flash_attention, gqa, tiling | GQA decode（Sq≤4, G>1）时把 G 个 query head 零拷贝重解释进 matmul M 维，消除 G 个 M=1 小 matmul | playbooks/P116_gqa_grouped_m_head_merge.md |
| P117 | MatmulApiTiling host recipe | L1 | attention, matmul, tiling | QK^T/PV 双 BMM 链 host 端接入 MatmulApiTiling：SetBufferSpace + d∈(64,128] SetFixSplit + depthA1 2→3 | playbooks/P117_matmul_api_tiling_host_recipe.md |
| P118 | Mask precompute parallel AIV | L1 | flash_attention, mask | causal mask 下沉独立 AIV kernel 全核并行预生成；按需 dense right-down 替代通用 2048×2048 压缩表 | playbooks/P118_mask_precompute_parallel_aiv.md |
| P119 | Two-pass to online softmax | L2 | flash_attention, softmax | 两遍/逐 token softmax 改写为 online 单遍 KV 流式（running max/sum/acc rescale），K/V GM 读取减半 | playbooks/P119_two_pass_to_online_softmax.md |

## 性能优化策略 — MHA Cube 化实战优化 (P120-P123) [Layer: L1-L2]

这些策略覆盖纯向量 attention 到 cube 流水的完整改造链：mmad 重写 → CV 流水 → tile 大块化 → causal 跳过。四张卡存在依赖顺序：先 P120 建立 cube 路径，再 P121/P122 铺流水与大块化，P123 随时可叠加。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P120 | FA vec-dot to cube mmad | L2 | flash_attention, cv_fusion | QK^T/PV 由 Mul+ReduceSum/标量乘加模拟时重写为 Cube mmad：K ColumnMajor 免转置 + Q 常驻 L1 + P 回写输入 dtype | playbooks/P120_fa_vec_dot_to_cube_mmad.md |
| P121 | FA SplitFuse CV pipeline | L2 | flash_attention, cv_fusion | FA 四阶段串行/SyncAll 时改写为 AIC QK+PV / AIV softmax+rescaleO 流水：GM workspace 多槽 + 三 flag 握手 + PRE_LAUNCH 预发射 | playbooks/P121_fa_splitfuse_cv_pipeline.md |
| P122 | KV stack large tile + L1 adaptive | L1 | flash_attention, matmul | KV tile ≤128 或 L1 划分硬编码时：KV 堆叠 512 摊薄握手 + nDyn/kDyn 按 L1 预算公式自适应 | playbooks/P122_kv_stack_large_tile_l1_adaptive.md |
| P123 | Causal KV skip + boundary mask | L1 | flash_attention, mask | causal/sparse 全量循环 + 逐元素 mask 兜底时：noSkipKvS 循环界裁剪 + 边界 tile mask 特化 + dtype×mask 编译期 4 入口分发 | playbooks/P123_causal_kv_skip_boundary_mask.md |

## 性能优化策略 — MLA Decode 实战优化 (P124-P126) [Layer: L1-L2]

这三张卡覆盖 MLA decode 的三个增量优化点：K 维拼接、任务边界流水、变长调度。依赖顺序：P124 要求 P120 已建立 cube 路径；P125 要求 P121 已铺 CV 流水；P126 独立可先行。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P124 | MLA nope+rope concat GEMM K | L1 | flash_attention | QK^T 的 nope/rope 两段独立计算再相加时：拼成单 GEMM K 维（embed-split 128 分段流水 + Q 拼接常驻 L1 + 单 L0C 累加一次写回） | playbooks/P124_mla_nope_rope_concat_gemm_k.md |
| P125 | Cross-task cube prefetch | L2 | flash_attention, cv_fusion | CV 流水多任务 kernel 任务边界断流时：KV 循环尾部一拍预读下一任务 tiling 并预发射其首块 QK^T（含 Q→L1），pingpongIdx 跨任务连续 | playbooks/P125_cross_task_cube_prefetch.md |
| P126 | GM tiling table task schedule | L1 | flash_attention | decode 变长 kvSeqlen 仍按 totalQ 均匀切分 + 除法反解坐标时：host 拍平任务表（kvSeqlen 降序）+ kernel process 轮询领取，一次 launch 跑全 batch | playbooks/P126_gm_tiling_table_task_schedule.md |

## 性能优化策略 — Conversion 模式特化与 Conv 实战优化 (P127-P131) [Layer: L1-L2]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P127 | Conversion pattern specialization routing | L1 | conversion, transpose | permute/transpose/split 等纯布局变换算子：R1 view 短路→R2 维合并→R3 退化路由→R4 模式专用 kernel→R5 fallback；禁止单一 generic gather 主路径 | playbooks/P127_conversion_pattern_specialization_routing.md |
| P128 | FP32-via-BF16 hi/lo 3-GEMM | L2 | conv, matmul, fp32 | Cube 不吃 fp32 时：hi+lo 双 bf16 分量 3-GEMM 保精度（~1e-4）换 Cube 吞吐（ex-X17） | playbooks/P128_fp32_via_bf16_hilo_3gemm.md |
| P129 | Fused img2col LoadData3D | L2 | conv, im2col | conv kHkW>1：A 侧免 GM 物化随路展开；特征图超 L1 用 band 窗口化（ex-X18） | playbooks/P129_fused_img2col_loaddata3d.md |
| P130 | 1x1 Conv flipped GEMM | L1 | conv, pointwise | 1×1 卷积：原始布局即 GEMM 操作数，免 wrepack/im2col 两级发射（ex-X19） | playbooks/P130_1x1conv_flipped_gemm.md |
| P131 | Weight repack LRU cache | L1 | conv, matmul, host-cache | 推理式重复调用：host 侧缓存权重 repack 结果，命中跳过 repack kernel 发射（ex-X20） | playbooks/P131_weight_repack_lru_cache.md |

## 精度优化策略 (A1-A8) [Layer: L0]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| A1 | FP32 intermediate computation | L0 | datapath | BF16/FP16 with precision requirement | playbooks/A1_fp32_intermediate.md |
| A2 | Welford numerically stable mean/var | L0 | datapath | LayerNorm, BatchNorm, RMSNorm | playbooks/A2_welford_algorithm.md |
| A3 | Rounding mode control (CAST_*) | L0 | datapath | Any Cast to lower precision | playbooks/A3_rounding_mode.md |
| A4 | SetFlag/WaitFlag event sync | L0 | datapath | Data dependencies across pipes | playbooks/A4_pipeline_barrier.md |
| A5 | Softmax numerical stability | L0 | datapath | Softmax, attention score ops, NaN/Inf risk | playbooks/A5_softmax_stability.md |
| A6 | High-precision rsqrt (Newton-Raphson) | L0 | datapath | Normalization ops needing rsqrt | playbooks/A6_high_precision_rsqrt.md |
| A7 | Index & boundary safety | L0 | datapath | Gather/scatter, user-controlled indices | playbooks/A7_index_boundary.md |
| A8 | Quantization-specific precision | L0 | quantization | Quantization output, dequantization with bias | playbooks/A8_quant_precision.md |

## 探索发现策略 (Discovered Strategies) [Layer: L2]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| X1 | Inline Target Capture | L2 | — | Reduce + single-element lookup ops | discovered/disc_X1.md |
| X2 | Big-Tensor-On-Device, Index-Only-Over-Bus | L2 | gather, routing, bus-bound | Host-coordinated gather/scatter where data tensor ≫ index tensor | discovered/disc_X2.md |
| X3 | End-to-End Dtype Passthrough | L2 | elementwise, foreach, multi-dtype, bandwidth-bound | Multi-dtype elementwise where wrapper upcasts fp16/bf16→fp32 and inflates HBM traffic ~5x; sink dtype boundary into kernel | discovered/disc_X3.md |
| X4 | Concat-Fused Strided DMA + Two-Level Skv Reduction | L2 | decode-attention, latency-bound, GQA N_kv=1 | Collapse nope|rope concat into strided K-tile DMA + two-level vmul→vcadd→vcadd + one-pass FA-2; kills per-key DMA/sync latency (decode 0.11-0.27x→0.25-0.79x) | discovered/disc_X4.md |
| X5 | GQA-Decode GEMV→GEMM (M=N_q) | L2 | decode-attention, compute-domain, GQA N_kv=1, cube | For GQA decode/MTP (S∈{1,2}, N_kv=1) take decode M=N_q(128/64) so QK^T=Q[N_q,D_qk]@K^T and PV=P[N_q,S_kv]@V become dense cube GEMM instead of N_q vec GEMVs; stream K once (NC=256), V once (sub-col nd2nz), N-tile PV to L0C; decode 11-33x (geomean 0.57x→3.53x) | discovered/disc_X5.md |
| X6 | Merge Partial + Reduce Kernel for Split-KV FlashDecode | L2 | decode-attention, split-kv, kernel-fusion | Split-KV 场景 partial 与 reduce 两个 kernel 融合，消除 reduce kernel 的独立 launch/stream 开销（链路 149.58→99.53us，-33.5%） | discovered/disc_X6.md |
| X7 | Token Block Size (TB) Tuning for Pure-Vector Attention | L2 | decode-attention, pure-vector, tiling | 纯 Vector attention 的 token block 尺寸调优（同窗口 mean_asc -4.2%，case 级最高 -42.6%） | discovered/disc_X7.md |
| X8 | Host-Side Precomputation of Constant Offset Tables | L2 | host-precompute, offset-table, latency-bound | 常量偏移表（如 Pi 转置）移到 Host 预计算，kernel 内零重复计算（mean_asc 149.58→125.64us，-16.0%） | discovered/disc_X8.md |
| X9 | Deferred Scaling (反量化仿射代数重排/延迟缩放) | L2 | dequant, affine, vector-efficiency | 反量化仿射结构代数重排，延迟缩放减少逐元素乘加链（单核 aiv_time -33%，task_duration 18.23→12.91us） | discovered/disc_X9.md |
| X10 | Device-side Grouped Scheduling + NZ Plane Weight Cache | L2 | moe, matmul, cube, grouped | Grouped matmul（group_list cumsum）Cube 化：AIC 设备侧自建分组调度（empty group 跳过），weight NZ plane 静态缓存 + m-fastest tile 排序多核共享 strip，fixpipe NZ 步长=orgM×16 原始值（level4_grouped_matmul_swiglu_quant 0.0026x→4.82x vs PyTorch） | discovered/disc_X10.md |
| X11 | ND-drain + L0B-feed-driven baseM Scaling | L2 | matmul, quantization, cube, c220 | 量化 matmul Cube 利用率修复：多 fragment NZ drain 为芯片缺陷（与 orgM 对齐无关）→ CType 路由 ND；瓶颈是 L0B-fill:mad 比而非 L2 refetch → baseM 扩至 L0C 上限（int32 128@256）；重叠尾 strip 解 singleM<baseM/singleN≤16 崩溃；AIV 反量化 strip 化（strip 所有权+per-strip 向量+TQue 双缓冲+单条 2D DataCopyPad）（level3_quant_matmul 0.1999x→0.8274x vs 官方 QuantBatchMatmulV3） | discovered/disc_X11.md |
| X12 | Pure-VEC Streaming Weight-Quant GEMV (小 M 分流) | L2 | weight-quant matmul, GEMV, small-M, c220 | 量化 GEMV 纯 VEC 流式路径：scale 提出 K 循环 + mixed-precision Axpy + 确定性两阶段 K-split（c220 SetAtomicAdd 丢更新禁用原子加）+ occupancy makespan tiling；小 M(1-8) 3.3-8.8x，大 M 须分流 cube（level3_weight_quant_batch_matmul 0.98x geomean，小 M case 最高 8.77x） | discovered/disc_X12.md |
| X13 | Shape 自适应双引擎分流 (M-Threshold Hybrid Routing) | L2 | weight-quant matmul, skinny-GEMM, GEMV+GEMM mixed shapes | Host 侧按 M 路由两条完整 kernel 路径（小 M 纯 VEC streaming 吃带宽极限 / 中大 M cube 吃 MAC），中间区间按 K*N 流量摊销固定开销做实测校准二级阈值；路由只换引擎不改数值（共享精确代数分解）（level3_weight_quant_batch_matmul VEC 0.98x / cube 1.51x → 混合 2.03x geomean） | discovered/disc_X13.md |
| X14 | Mix-Kernel 跨核生产者/消费者融合 (Pairwise AIV-Producer/AIC-Consumer) | L2 | weight-quant matmul, multi-kernel pipeline, dequant-pass elimination, c220 | 硬件 int8-cube 路径证伪后，将独立 VEC dequant pass 与 cube kernel 融合为单 mix kernel：AIC ci 与 AIV(2ci,2ci+1) 配对按 strip 序生产/消费 write-once workspace，GM-flag 协议同步（128B 步长 + MTE3 DataCopyPad 发布 + __gm__ volatile 有界轮询 + DCCI）；launch 3→2、cast 与 cube 重叠（level3_weight_quant_batch_matmul 2.03x→2.44x，dequant kernel launch=0） | discovered/disc_X14.md |
| X15 | 递归拆分-并行投影外提 (Recurrent Split: Hoist Parallel Projection) | L2 | rnn, lstm, gru, scan, recurrent, dataflow | 含串行循环算子将非递归投影（全部时间步 x@W_ih）合并为单次 host 侧大 GEMM 外提走 Cube，自定义内核只保留严格串行的递归残差（h@W_hh+门激活+状态更新），递归权重 UB 常驻+超规格回退；串行链计算量 O(seq×(I+H)×4H)→O(seq×H×4H)（LSTM 0.202x→0.852x mean） | discovered/disc_X15.md |
| X16 | 循环算子 H-slice 多核切分 (Recurrent H-Slice Multi-Core Split) | L2 | rnn, lstm, gru, scan, recurrent, multi-core, small-batch | 时间维不可并行时按 hidden 维切多核：每核独占 [hStart,hEnd) 及 4 门控列、W_hh 切片 UB 常驻跨时间步复用、c/h 切片独立更新零核间通信；host tiling 按 shape 在 batch-parallel/H-slice 间调度（batchNum≤4 或 H≥256&&batchNum≤32 走 H-slice）；**禁忌：门控列为 {j,H+j,2H+j,3H+j} 跨步取，禁连续 [4hStart,4hEnd) 切**；与 X15 正交可组合（纯 kernel 0.335x→0.799x，移植进 X15 架构 0.852x→2.12x mean） | discovered/disc_X16.md |
| X17 | FP32-via-BF16 Hi/Lo 拆分 3-GEMM | L2 | fp32, cube, split-precision, conv, matmul | 910B Cube 不吃 fp32 输入时：x≈hi+lo 双 bf16 分量 3-GEMM（hh+hl+lh，fp32 累加）+ 单趟 split-xtrans；±inf 语义天然保持；2-GEMM 已证精度不足否决；L0 队列深度按每 chunk tile 数定（双 tile 用 depth-4）；fp32 geomean 0→0.50 | discovered/disc_X17.md |
| X18 | Fused img2col LoadData3D 随路展开（含 band 窗口化） | L2 | conv, im2col, cube, bandwidth-bound | 卷积 A 侧免 GM 物化：NC1HWC0 常驻 L1 + img2col LoadData3D 随路展开进 L0A；xNz 超 L1 时 band 行带窗口化（padTop'/padBottom' 重算保 Ho 光栅恒等）；硬件 k 序 (ci1,kh,kw,ci0) 须 one-hot 探针确认；W<24 窄图/kExtension>64 门控回退；消 70% im2col 占比，大 case 0.32→0.59 | discovered/disc_X18.md |
| X19 | 1×1 Conv 翻转 GEMM（原始布局即操作数） | L2 | conv-1x1, pointwise, launch-bound | kH=kW=1/s=1/p=0 且 Ci%16==0、(H*W)%16==0 时：A=原始权重 [Co,Ci]、B=原始 x [Ci,H*W]，免 wrepack+im2col 两级发射，epilogue 通道行直写 NCHW；UB 中转解除对齐门控已证净负；实测 1.0~1.98× | discovered/disc_X19.md |
| X20 | Host 侧权重预打包结果 LRU 缓存 | L2 | host-cache, weight-repack, inference, conv, matmul | conv/matmul 等带权重 repack 步骤的算子，repack 输出是确定函数，推理式重复调用按 (data_ptr+sizes+dtype+version+布局) 做 LRU 缓存，命中跳过整个 repack 发射；条目持 weight 强引用防地址复用；每次调用省 5-15us | discovered/disc_X20.md |
| X21 | 910B2C Cube 化平台纪律 + 数据驱动 Kahan 精度路由 | L2 | conv3dbf, fp32, kahan, precision-routing | conv 类 fp32 算子 Cube 化时：平台纪律（GetCoreNum 保底、no_half_ufp16 指令表、fp16 3-GEMM）＋数据驱动精度路由（小数值走 fp16 直算、大数值走 Kahan 补偿），绕过 fp32-Cube 缺失并保精度 | discovered/disc_X21.md |
| X22 | Depthwise 直接卷积（绕过 im2col+GEMM） | L2 | depthwise, conv2d, vector-conv, im2col-bypass | depthwise(groups==Cin,cig==1) 用单 AIV fp32 向量卷积直接算逐通道 kh*kw dot：无 im2col/GEMM/fp16 分裂/GM workspace/Cube；Tile=(image,co-block8,oh-block) 保 32B 对齐，对齐 interior 向量化+标量 head/tail 独立 corr 缓冲消 S/V hazard；depthwise gen-time 提升 18-84x | discovered/disc_X22.md |
| X23 | Batched Spatial 全宽向量化 + 编译器驱动标量预取 | L2 | conv, cv_fusion, elementwise, s-pipe-latency-hiding | 滑窗卷积 kw==1/W 维逐点：把 P=floor(64/W) 个连续 oh 打包进 64-lane fp32 向量，每 (co,tap) 1 次权重标量读驱动 P*W 宽 Axpy（向量指令+标量读降 P 倍）；Boundary 零填充 + strided DataCopyPad；`#pragma unroll 4` 驱动编译器软件流水隐藏 ~43cy S-pipe 标量读延迟（⚠️手工多标量预取触发编译器误编译，禁用）；W%8==0 门控；geomean 0.36x→3.5x | discovered/disc_X23.md |
| X24 | AIC_ONLY 任务类型压设备 launch 地板（固定开销主导算子） | L2 | matmul, elementwise, reduction, aic-only, device-launch-floor | 微小 shape 固定开销主导算子（cube_util 极低、标量占比高）：任务类型从 MIX_AIC_1_2 改 `KERNEL_TYPE_AIC_ONLY` 去掉每 AIC 的 2 个空转 AIV，设备固定地板 ~7-8μs→~4.3-4.5μs；配 compact tiling 头（65→4 int32）；与手动 mmad 叠加 12.09→6.88μs（1.757x vs 基线）；上限：910B kernel launch 地板 ~4.3-5.6μs | discovered/disc_X24.md |

---

## How to Use

1. **先归族**：打开 [`scenario-routing/README.md`](scenario-routing/README.md)，用结构判据确定算子族，打开对应族 workflow
2. 按 workflow 阶段顺序选卡（通用前置 → 族内阶段），本文档表格仅作补充反查
3. Read each referenced `playbooks/XXX.md` detail file (P102/P103 无 playbook，读对应 cards/ 卡片)
4. Apply the selected patterns in `tiling_pass`, `init_pass`, `process_pass`, `process_nonaligned_pass`
5. **Check compatibility**: Read `strategy-compatibility.md` to verify no mutual exclusions and all dependencies are met

### 族判定快速路由（选策略前先归族）

**已下沉为独立场景路由目录**：[`scenario-routing/README.md`](scenario-routing/README.md)。先按结构判据归族（FA / conv / matmul / quantization / reduction / sort / conversion / 小族 / elementwise），再打开对应族的 workflow 文件按阶段顺序选卡——不要在下表平铺选卡。下表仅作补充反查。

### 按算子类型快速查表 (Quick Reference by Operator Type)

| Operator Type | L0 Strategies | L1 Strategies |
|---------------|---------------|---------------|
| LayerNorm / RMSNorm / GroupNorm | D1, D2, P1, P2, P3, A1, A2, A6 | P19, P55, P68, P69, P81, P84, P94, P95, P98, P99 |
| Quantization ops | D1, D4, D3, P2, P4, A3, A8 | P48, P49, P70 |
| Element-wise (foreach) | D1, D2, P1, P2, A1, A4 | P67, P69, P84, P89, P95, P97, P100, P101, P112 |
| Softmax / Attention | D1, P1, P5, A3, A4, A5 | P14, P18, P38, P55, P68, P81, P94, P100, P119, P123 |
| Pooling / Gather | D1, P1, P2, P11, A7 | P33, P56, P59 |
| Optimizer ops | D1, P1, P2, P9, A1 | P81, P84, P94, P95 |
| Broadcast / Mask ops | D1, P1, P12, A5 | P67, P99, P113, P118 |
| Index / Scatter ops | D1, P1, P7, A7 | P32, P64, P100 |
| Special / Complex ops | D2, D3 | P85, P88, P96, P102 |
| CV Matmul (Cube+Vector) | D1, D3, P1, P4, P7, A3 | P14, P17, P46, P47, P49, P51, P52, P63, P70, P71, P72, P73, P78, P79, P83, P86, P87, P90, P92, P93, P102, P103, P104, P105, P106, P107, P108, P117, P120 |
| CV FFN (MoE) | D1, D3, P1, P4, A1 | P46, P48, P49, P50, P73, P75, P88 |
| Flash Attention | D1, P1, P5, A4, A5 | P14, P16, P17, P18, P29, P38, P53, P57, P58, P60, P61, P76, P77, P80, P90, P91, P115-P126（**按 scenario-routing/flashattention.md 的 SOP 阶段顺序选用，勿平铺**） |
| Reduction (mean/sum/max/min) | D1, P1, P2, A1, A2 | P68, P69, P109, P110 |
| Sort / TopK | D1, P1, P2, A7 | P96, P111 |
| Transpose / Layout Conversion | D1, P1, P7, P10 | P127（模式特化路由）, P26, P114 |
| Convolution (conv1d/2d/3d, depthwise, ConvTranspose) | D1, P1, P7, A1 | P130（1×1）, P128（fp32）, P129（fused im2col）, P131（repack 缓存）——按 scenario-routing/convolution.md 模式分发 |
| Data Path (通用数据通路) | D1, D5, P1, P8, P10, P20, P25, P40, P54, P65, P66, P74 | — |

### FA 算子族优化 SOP

**已迁入场景路由**：[`scenario-routing/flashattention.md`](scenario-routing/flashattention.md)（S0 形态判定 → S1 结构改写 → S2 CV 流水 → S3 内存搬运 → S4 mask/稀疏特化 → S5 调度均衡 → S6 通用收尾）。

### 按瓶颈类型查表 (Bottleneck-Driven Strategy Selection)

供世界模型 profiling_evidence 参考：

| Bottleneck Type | Primary | Secondary | Anti |
|-----------------|---------|-----------|------|
| mte2_stall | P1, P10 | P2, P7, P19, P18, P56, P59, P66, P92, P93, P101, P112, P113, P114, P118, P119, P124, P127, P129, P130 | P3 |
| mte3_stall | P1 | P8, P19, P40, P56, P59 | — |
| tiling_imbalance | P4, P2 | P11, P47, P51, P58, P72, P73, P75, P104, P106, P116, P126 | P3 |
| scalar_loading | P5, P67 | P2, P54, P90, P97, P99, P100, P101 | — |
| scalar_compute | P5, P67, P94, P95, P98, P99, P100, P102 | P84, P103, P111, P120, P126, P127 | — |
| compute_bound | D1, P46, P94, P95, P96, P102 | A1, P47, P68, P69, P79, P84, P91, P92, P103, P120, P123, P127, P128, P130, P131 | — |
| near_optimal | P98 | — | — |
| no_overlap | P1 | P10, P19, P63, P118, P119, P121, P122, P125 | — |
| partial_overlap | P1, P2 | P8, P18, P26, P53, P83, P90, P125 | — |
| undersize_transfer | P2, P10 | P7, P25, P45, P66 | — |
| icache_miss | — | — | — |
| bus_contention | P8, P65 | P1, P28, P64, P93 | — |
| l2_cache_thrash | P52, P74 | P61, P78, P105, P107 | — |
| ub_memory_pressure | P8, P85 | P81, P88, P89, P110, P111 | P20 |
| tiling_mismatch | P104, P108 | P105, P106, P107, P115 | — |
| cube_underutilization | P104, P106 | P102, P115, P116, P117, P122, P123, P124 | — |
| sync_overhead | P4, A4 | P111, P115 | — |
| vec_underutilization | P112 | P67, P69 | — |
| tail_block | P11 | P112, P113, P114 | — |
