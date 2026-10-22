# Attention Pattern Index

生成 attention 算子前先读本文件，用它决定需要看的模式文档。模式可以组合，但不要临时发明新模式。

渐进式披露原则：不要一上来读完所有模式文档。先用本索引定位命中的模式，再只读对应文档顶部的 `先读这个`；只有实现细节不够时，再继续读该文档后面的 `核心规则 / 地址公式 / 边界`。

## 稀疏索引与融合补充

若 `indices/topk` 表示选中的 **block**，先读 `ops/tilelang-perf-optimization/references/cv-patterns/sparse-layout.md`（安装后 `.claude/skills/tilelang-perf-optimization/references/cv-patterns/sparse-layout.md`）；下表 `topk_kv_sparse_pattern.md` 默认逻辑 token，不能直接套其除 page-size 地址公式。先核对 block/page 对应、cache 布局与有效前缀，再进入 Tile 设计。Online 状态与全 mask 边界见同目录 `online-consumption.md`。

## 核心模式

| 如果看到 | 读这个文档 |
| --- | --- |
| `actual_q_len` / `actual_kv_len` / `(T, H, D)` 拼接布局 | `@references/attention-patterns/references/tnd_pattern.md` |
| `block_table` / `page_table` / `k_cache` / `v_cache` / paged KV cache | `@references/attention-patterns/references/page_attention_pattern.md` |
| `mask` / causal / padding / visibleCols / `-inf` score 注入 | `@references/attention-patterns/references/mask_causal_pattern.md` |
| `Hq != Hkv` / GQA / MQA / `n_heads` vs `n_kv_heads` | `@references/attention-patterns/references/head_sharing_pattern.md` |
| `q_nope` / `q_rope` / `k_nope` / `k_rope` / `headdim_qk != headdim_v` / latent KV | `@references/attention-patterns/references/mla_pattern.md` |
| `indices` / `topk` / selected token / sparse gather | `@references/attention-patterns/references/topk_kv_sparse_pattern.md` |
| `sink_k` / `sink_v` / sink token / fixed prefix KV source | `@references/attention-patterns/references/sink_pattern.md` |
| `dy` / `softmax_max` / `softmax_sum` / `attention_in` 输入、`dq/dk/dv` 输出 → **FA 反向（FAG）** | `@references/attention-patterns/references/fa_backward_pattern.md` |
| **以上全部不命中** → 标准 Attention `[B, H, S, D]` | 标准 FlashAttention 结构（Q 分块循环 + online softmax rescale + PV 累加 + C/V split 流水） |

## 组合顺序

多个模式同时出现时，按这个顺序理解：

```text
1. TND：如果输入是 (T, H, D)，先恢复 batch 边界。
2. Head Sharing：确定 q_head -> kv_head。
3. MLA：如果 Dqk 和 Dv 分离，确定 QK 路和 PV 路各自消费哪些维度。
4. Sink：如果有 sink source，确定 sink/local 的 source dispatch。
5. Sparse TopK：如果有 indices，先确定 logical token 或 block 单位，再确定有效范围。
6. Paged KV：如果是 paged cache，按已确定的索引单位映射到 physical page/row。
7. Mask/Causal：在 softmax 前处理可见性、padding、causal。
8. Backward：若为反向（dy/softmax 统计量输入），上述模式只决定数据路径，计算图按 fa_backward_pattern 的三段式组织。
```

组合命中时仍然按需阅读：

- 先读每个命中文档的 `先读这个`，确认该模式在当前算子中承担的唯一职责。
- 优先读主导布局的模式：TND 或 Paged KV。
- `Head Sharing` 只决定 `kv_head`；不要在它里面找 page/sparse 的寻址规则。
- `MLA` 只决定 `Dqk/Dv` 和 latent 行的列消费规则；不要在它里面找 page 本身的映射规则。
- `Mask/Causal` 最后读；它只改 score visibility，不改 source 路径。

## 生成前问题

1. 输入是标准 `[B, H, S, D]`，还是 `(T, H, D)` 拼接布局？
2. K/V 是连续 tensor，还是 paged cache？
3. `Hq` 和 `Hkv` 是否相等？
4. `Dqk` 和 `Dv` 是否相等？
5. 是否有 `sink_k/sink_v`？
6. 是否有 `indices/topk`？
7. 是否有 causal、padding、显式 mask？
8. 是前向还是反向（是否有 dy/softmax_max/softmax_sum/attention_in 输入）？

根据答案选择上面的文档。若 7 个问题全部否定 → 标准 Attention，按标准 MHA 结构组织：block 级 Q 分块循环、O 分块循环、softmax rescale、C/V split 流水线。
