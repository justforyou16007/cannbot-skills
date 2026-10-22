# 稀疏取数：先统一索引单位，再喂给 Cube

## 何时读

SparseFlashAttention、paged KV、topk 选择后接矩阵乘。不要因为都叫 topk 就复用相同地址公式。

## 可迁移规则

| 输入索引语义 | 地址变换 |
|---|---|
| 逻辑 token t，page size P | logical_page=t//P，offset=t%P，physical_page=block_table[logical_page] |
| 逻辑 block b，block 与 cache page 一一对应 | physical_page=block_table[b]，块内 token offset 另生成 |

先确认第二行的一一对应条件；block size 与 page size 不同要另建映射。无效索引的安全 load 和无效 score 的 mask 是两个动作，置零 KV 不自动排除 softmax 概率。

## 源码案例与限制

- **S4-01 / SFA arch35**：`GetRealCmpS2Idx → GetkeyOffset → CopyInKvSparse → CopyOutKvUb2Gm → IterateBmm1SFA`，逻辑稀疏 token 经 blockTable 到物理 KV，再整理为规则 UB/小 GM 块，Cube 从 GM 搬到 L1。**A5/arch35 案例也仍有 GM 中转**。
- **S4-02 / MiniMax M3 GQA A5**：`_gqa_sparse_fwd_kernel` 的选择索引为逻辑 BLOCK，先查 `block_table[batch,blk]`，再 `tl.load K/V → tl.dot`，kernel 内不建立全量 gather tensor。wrapper `_as_triton_main_kv_cache` 的 `torch.stack` 可能产生拷贝，必须计入整段。

## L4 适配边界

- GqaSparseFwd：L4 cache `[pages,kv_heads,128,2D]`，目标 `[pages,2,128,kv_heads,D]`。需要核对 stride/重排，不是 reshape。
- L4 用 query 绝对位置确定 topk 有效前缀且不约定 sentinel；目标按 `topk_idx>=0` 计数。尾部随机值不能当有效索引。计数不等于压紧：确认有效项位于 kernel 实际遍历的前缀，必要时适配并计入成本。
- `block_size_q=1` 是已研究映射的限制；不能将其泛化到任意 query block 或 FP8。
- SFA L4 的 PA reference 接收 block_table，但所读 PA_BSND 路径缺少逻辑页→物理页映射；不能作为乱序页表的独立 golden。`_to_cumsum` 的 90% 启发式还会把 lengths `[1,11]`、总长12误当累计值，正确前缀和应为 `[1,12]`。这些是静态发现，非板测失败报告。

## 最小验证

用非恒等页表、跨页索引、尾块、重复/无效索引、不同有效长度做区分性测试；明确长度输入是逐段长度还是累计边界。golden 独立实现页表映射，分别验 cache 适配和 Attention 数学。记录小 workspace 与全量物化的峰值/搬运，勿由源码 Tensor 数推断 HBM 流量。

## 固定版本证据

- [SFA-VEC](https://github.com/hicann/ops-transformer/blob/bbb4dafe296119266a0a64133f17bc42b6597ed3/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_vector_mla_arch35.h)
- [SFA-CUBE](https://github.com/hicann/ops-transformer/blob/bbb4dafe296119266a0a64133f17bc42b6597ed3/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_cube_mla_arch35.h#L1-L235)
- [SFA-META](https://github.com/hicann/ops-transformer/blob/bbb4dafe296119266a0a64133f17bc42b6597ed3/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_kvcache.h#L1-L260)
- [GQA](https://github.com/vllm-project/vllm-ascend/blob/5f8a1286a2d04d35b94ebc8a961057c48a7b82d2/vllm_ascend/models/minimax_m3/ops/msa_m3_triton_a5.py)
- [L4-SparseFlashAttention](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/SparseFlashAttention/definition.json#L1-L105)
- [L4-GqaSparseFwd](https://github.com/grandolantow/ascendc-kernelgen-data/blob/3cf7bccc9a3c9a2a730ca3ccf6a4b0ac22d05c72/npu-kernelbench/data/kernel_generator/level4/GqaSparseFwd/definition.json)
