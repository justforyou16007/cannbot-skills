# SparseFlashMla 打包算子参考（压缩稀疏 MLA）

一句话：SparseFlashMla 是 **metadata-first 的 TTK 模板打包算子**，注意力流 = ori 段
（KV 窗口）+ cmp 段（压缩前缀）双段拼接；本机直调验证（910B2C + CANN 9.0.0）已全链路
打通，本文档沉淀算子领域知识、部署路径、平台约束与验证管线要点。

> 与 `mla_pattern.md`（DeepSeek MLA：q_nope/q_rope + latent kv_cache，Dqk≠Dv）区分：
> SparseFlashMla 是**压缩稀疏**变体（KV_N=1、D=512，K/V 共享存储，无 rope 拆分），
> 命中信号是算子名 `SparseFlashMla` / 输入含 `cmp_kv`、`cmp_sparse_indices`、
> `metadata` 张量。

## 1. 算子领域知识

- **metadata-first 两段式**：`sparse_flash_mla_metadata`（AICPU 算子，输出 `[1024]`
  int32 任务表 = FA 36 核 × 9 字段 + FD 72 核 × 8 字段）→ `sparse_flash_mla`
  （TTK 模板主算子，22 个 GM 指针入口，arch22=910B / arch35=950 双实现分支）。
- **TTK 模板算子**：`ASCENDC_TPL_ARGS_DECL(SparseFlashMla, ...)` 7 模板参数组合，
  **无法用 ascendc cmake（legacy_modules）直启**——只能由 CANN 算子框架
  （ops-transformer 构建系统 + asc_opc）实例化。
- **布局约束**：BSND（batch, seq, head, dim）；KV_N=1、D=512 为算子硬约束；K/V
  共享存储（`ori_kv` / `cmp_kv` 各为一整块）。
- **三场景**：
  | scenario | ori_kv | cmp_kv | cmp_sparse_indices |
  |----------|--------|--------|--------------------|
  | SWA（sliding-window） | 全量 KV，band/causal/no-mask 窗口 | 不用 | 不用 |
  | HCA（heavy compression） | KV 窗口 | 压缩前缀 `floor(causal/ratio)` 行 | 不用 |
  | CSA（compressed sparse） | KV 窗口 | 压缩前缀 | 每查询行 topk 索引（前 min(K,threshold) 项、-1 停止、>=threshold 跳过） |
- **metadata 末核语义**：decode 场景常只有 FA slot0 一核（`nextGs1Idx==0 &&
  s2EndIdx==0` 编码"末核"——kernel 内 `bN2EndIdx` 不加 1），主算子内部展开全部
  gS1 任务（见 `sparse_flash_mla_swa_kernel.h::ProcessMainLoop`）。

## 2. 910B 平台约束（aclnn host 校验，非 Ascend950 分支）

校验源：`sparse_flash_mla_metadata_check.h`（打包源码）。**host tiling 的 Check 不
限制**（GRAPH_SUCCESS），拒绝发生在 aclnn 层（EZ0024）/ AICPU ParamsCheck。

| 条件 | 910B 要求 |
|---|---|
| ori_mask_mode | 仅 4（BAND） |
| ori_win_left / right | 127 / 0 |
| cmp_mask_mode | 仅 3 |
| 无 cmp_kv | cmp_ratio = 1 |
| HCA（cmp_topk==0） | cmp_ratio = 128 |
| CSA（cmp_topk>0） | cmp_ratio = 4 且 topk ∈ {512, 1024} |

Ascend950 放宽：ori_mask_mode ∈ {0,3,4}、win ≥ -1（-1 = 无限）、cmp_mask_mode ∈ {0,3}、
cmp_ratio ∈ [1,128]。**归档 CSV 的 golden/tilelang 语义用例（no-mask、causal、
小 topk CSA）在 910B 直调必报 EZ0024**——用 selector 过滤（见 §5）。

## 3. 直调部署路径（本机验证通过）

```bash
cd /home/y00889327/ops-transformer/build
cmake .. -DASCEND_OP_NAME=sparse_flash_mla,sparse_flash_mla_metadata
cmake --build . --target sparse_flash_mla_ascend910b -j 1      # TTK 实例化，单实例约 25-30 分钟
cmake --build . --target op_host_aclnn cust_opapi optiling cust_opmaster \
                       cust_proto cust_aicpu_kernels ops_config_ascend910b -j $(nproc)
cmake --build . --target install -j $(nproc)
cd build_out && bash install.sh
# 运行期两个必需环境变量：
export ASCEND_CUSTOM_OPP_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
```

- **`ASCEND_CUSTOM_OPP_PATH` 是 AICPU 算子的必需 env**：AICPU 引擎
  （libaicpu_ascend_engine.so）只从该变量扫描 `op_impl/cpu/config/cust_aicpu_kernel.json`，
  不自动遍历 vendor。缺失报 `BinaryGetFunction 0x7080001`，设错可能 core dump。
  进程内 `os.environ.setdefault` 有效（GE 首次 NPU op 才读）。
- **`binary_info_config.json` 必须重生成**：主算子 kernel 注册表
  （`op_impl/ai_core/tbe/kernel/config/ascend910b/`）由 `ops_config_ascend910b`
  目标按当前 `-DASCEND_OP_NAME` 生成——旧配置 json 只含当时配置的算子，
  换配置不重跑则主算子报 `EZ1001 Get regInfo failed`。
- **锁残留**：TTK 编译中断后必须 `rm -rf gen/kernel_meta_* gen/compile_stop.flag
  gen/compile_pids.txt` + 杀残留 asc_opc，否则报 "Another process is using this dir"。
- **直调适配层**：aclnn C API 包装 + `TORCH_LIBRARY_FRAGMENT(npu)` catchall
  （`m.def(schema, TORCH_FN(fn))`）——零 tensor 参数算子（metadata）无需 dispatch
  key 即可 `torch.ops.npu.*` 调用；at::Tensor→aclTensor 转换照搬 ops-transformer
  `aclnn_common.h`（CANN Open License，随归档携带）。直调主算子 q 仅支持
  fp16/bf16（fp32 报 EZ1001 dtype 不支持）。

## 4. 调用契约（直调 wrapper 必须遵守）

- **`sinks` 必填**：tiling 的 GetSinks 拒绝 nullptr；传 `[H]` 零向量等价 golden
  无 sinks 语义。
- **`cmp_residual_kv` 条件必填**：cmp_kv 存在且 cmp_mask_mode==3、cmp_ratio!=1 时；
  恢复压缩前长度 `ori_len = cmp_len * ratio + residual`。全量 BSND 时按
  `seqused_ori_kv=[ori_seq]`、`seqused_cmp_kv=[cmp_seq]`、`cmp_residual_kv=[ori_seq % ratio]`。
- **CSA topk 语义**（对照 arch35 `ComputeS2LoopInfo`/`GetRealCmpS2Idx`）：每查询行
  仅考虑 indices 前 `min(topk, threshold)` 项；-1 终止；>=threshold 跳过但继续扫。
- **metadata 格式**：`[1024]` int32，字段顺序见
  `kernel/op_kernel/sparse_flash_mla_metadata.h`（FA: core_enable/bn2/m/s2 start/end/
  workspace_idx/s2_max_num;FD: core_enable/bn2_idx/m_idx/workspace_idx/num/m_start/m_num）。

## 5. 验证管线要点

- **selector 钩子模式**（与 tilelang 管线同构）：归档 model.py 提供
  `get_input_groups()`（全量）/ `get_input_groups_ascendc()`（910B 支持子集）/
  `get_input_groups_tilelang()`（除 CSA）；`verification_ascendc.py::_get_input_groups`
  优先取 ascendc selector。CSV 保持全量，golden/tilelang 语义不丢。
- **MERE 与输入分布**：evaluate_ascendc.sh 用 NPU Benchmark 三项判定（MERE =
  mean(|diff|/(|golden|+1e-7)) < 2^-7 for bf16）。softmax 类算子输入**必须正分布**
  （`rand*2`，gqa 式）：含负输入（`rand*4-2`）使输出过零，近零元素 rel_err 放大，
  MERE≈0.03 超阈值；正分布 MERE≈0.0002 轻松过。
- **验证矩阵（2026-08-12 全绿）**：golden 7 例 finite；tilelang 5/5 matched；
  ascendc 直调 2/2 pass（swa_decode + swa_prefill，matched_ratio=1.0）。
  如需 910B 直调 HCA/CSA，按 §2 约束加参数对齐用例（ratio 128/4、topk 512）。

## 6. 归档与相关文档

- 归档：`workflows/templates/archive_tasks/sparse_flash_mla/`（总 README + CSV +
  golden + model_new_ascendc/tilelang + design/{block,tile}_level + kernel/）
- 构建/部署/契约细节：归档 `kernel/README.md`
- 演进知识库（lessons/）：`process/ttk-packaged-op-direct-call-910b.md`、
  `process/archive-csv-soc-constraint-selector.md`、
  `precision/mere-input-distribution-softmax-zero-crossing.md`
