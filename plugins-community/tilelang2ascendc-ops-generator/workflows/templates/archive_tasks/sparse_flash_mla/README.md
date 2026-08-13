# sparse_flash_mla 归档

SparseFlashMla（压缩稀疏 MLA 注意力）的完整归档样例，与
`flash_attention_gqa` 同构：kernel 源码（照搬 ops-transformer 打包算子）+
CSV 驱动用例 + golden + 直调/设计双验证路径。生成 MLA 类算子时以此为本。

## 算子领域知识

- **metadata-first 两段式**：`sparse_flash_mla_metadata`（AICPU 算子，输出
  `[1024]` int32 的 FA 36 核 × 9 字段 + FD 72 核 × 8 字段任务表）→
  `sparse_flash_mla`（TTK 模板主算子，22 个 GM 指针入口）。decode 场景
  metadata 常只有 FA slot0 一核（`nextGs1Idx==0 && s2EndIdx==0` 编码末核
  语义），主算子内部展开全部 gS1 任务。
- **TTK 模板算子**：`ASCENDC_TPL_ARGS_DECL(SparseFlashMla, ...)` 7 个模板
  参数组合空间，`arch22`（910B）/ `arch35`（950）双实现分支，只能由 CANN
  算子框架构建（ascendc cmake 无 TTK 支持）。
- **布局与约束**：BSND（batch, seq, head, dim），KV_N=1、D=512 为算子约束，
  K 与 V 共享存储（ori_kv / cmp_kv 二合一）。
- **三场景**：
  | scenario | ori_kv | cmp_kv | cmp_sparse_indices |
  |----------|--------|--------|--------------------|
  | SWA（sliding window） | 全量 KV，band/causal/no-mask 窗口 | 不用 | 不用 |
  | HCA（heavy compression） | KV 窗口 | 压缩前缀 `floor(causal/ratio)` 行 | 不用 |
  | CSA（compressed sparse） | KV 窗口 | 压缩前缀 | 每查询行 topk 索引（前 min(K,threshold) 项、-1 停止、>=threshold 跳过） |
- **910B 平台约束**（aclnn host 校验，非 Ascend950）：ori_mask_mode 仅 4、
  win(127,0)、cmp_mask_mode 仅 3；无 cmp → ratio=1；HCA → ratio=128；
  CSA → ratio=4 且 topk∈{512,1024}。详见 `kernel/README.md` 平台约束小节
  与 `model.py::_ascendc_supported`。

## 归档结构

```
sparse_flash_mla/
├── basic_case.csv / general_case.csv   # 用例集（7 例：2 SWA-decode 类 + prefill/nomask/HCA/CSA）
├── model.py            # golden（逐行 fp32 softmax + sinks 虚拟 key）+ CSV 驱动输入构造
│                       #   get_input_groups() 全量 / _ascendc() 910B 子集 / _tilelang() 除 CSA
├── model_new_ascendc.py  # 直调候选：metadata → 主算子全链路（import 时 setdefault ASCEND_CUSTOM_OPP_PATH）
├── model_new_tilelang.py # TileLang 候选（CSA raise NotImplementedError）
├── design/
│   ├── sparse_flash_mla_common.py      # 常量/掩码模式/配置
│   ├── tile_level/sparse_flash_mla.py  # 简化直通实现（910B 全量验证）
│   └── block_level/sparse_flash_mla.py # 与打包算子调度的对齐描述（2D 核网格 + 双段 KV ring 流水）
├── kernel/             # 打包源码 + 直调适配层（构建/部署/契约见 kernel/README.md）
└── README.md           # 本文档
```

## 验证矩阵（2026-08-12 全绿）

| 管线 | 脚本 | 用例集 | 结果 |
|------|------|--------|------|
| 静态验证 | validate_tilelang_impl.py | - | PASS |
| TileLang | evaluate_tilelang.sh（verification_tilelang.py） | `get_input_groups_tilelang`（5 例，除 CSA） | 5/5 matched |
| 直调 | evaluate_ascendc.sh（verification_ascendc.py） | `get_input_groups()`（2 例 910B 子集） | 2/2 pass（matched_ratio=1.0, MERE≈0.0002） |
| golden 全量 | model.py 自验（CSV 直接构造） | 7 例（含 CSA） | 全部 finite |

- CSV 保持全语义用例集（no-mask/causal/小 topk CSA 等，供 golden/tilelang）；
  `get_input_groups()` 只返回 910B 打包算子支持的子集（`_ascendc_supported`
  过滤：ori_mask_mode=4、win(127,0)、cmp_mask_mode=3、HCA ratio=128、
  CSA ratio=4 + topk 512）——直调验证不依赖改动验证管线脚本。
- golden 输入用**正分布** `rand*2`：softmax 输出不过零，MERE（NPU Benchmark
  判定 mean rel_err < 2^-7）可过；含负输入（`rand*4-2`）时 MERE≈0.03 超阈值。
- 注意 HCA/CSA 用例需满足 S2 ≥ cmp_ratio（合法压缩）：S2 < ratio 时
  `cmp_residual_kv = S2 % ratio` 使算子恢复的 ori_len（cmp_len×ratio+residual）
  超出真实长度，与 golden 系统性错位（910B 约束要求 HCA ratio=128，故
  HCA 用例 S2 需 ≥128）。

## 知识沉淀

- **参考文档（主）**：translator 技能
  `.claude/skills/tilelang2ascend-translator/references/attention-patterns/references/sparse_flash_mla_package_op.md`
  ——算子领域知识 + 910B 平台约束 + 直调部署 + 调用契约 + 验证管线要点
- **演进知识库（lessons/，OKF 经验教训）**：
  - [TTK 打包算子直调部署（ASCEND_CUSTOM_OPP_PATH / binary_info_config.json）](../../../../skills/tilelang2ascend-trace-recorder/lessons/process/ttk-packaged-op-direct-call-910b.md)
  - [归档 CSV 与 SoC 平台约束——selector 钩子模式](../../../../skills/tilelang2ascend-trace-recorder/lessons/process/archive-csv-soc-constraint-selector.md)
  - [MERE 超标先查输入分布（softmax 过零）](../../../../skills/tilelang2ascend-trace-recorder/lessons/precision/mere-input-distribution-softmax-zero-crossing.md)
