# sparse_flash_mla kernel

Unlike `flash_attention`, this case does **not** ship a locally built AscendC
kernel: the canonical implementation of `sparse_flash_mla` is a packaged op of
`cann_ops_transformer` (`torch.ops.cann_ops_transformer.sparse_flash_mla`),
which `model_new_ascendc.py` calls directly. There is therefore no
`build.sh`/`CMakeLists.txt` here; the directory documents the op contract and
keeps a snapshot of the kernel source generated from the TileLang design.

## Op call contract (used by model_new_ascendc.py)

The op is metadata-first: a `sparse_flash_mla_metadata` call resolves the
output shapes / task partition, then the main `sparse_flash_mla` call computes
the result. Both take **BSND** layout tensors (batch, seq, num_heads,
head_dim). The exact keyword contract, verified against the ops-transformer
`sparse_flash_mla_tiling.cpp` / `sparse_flash_mla_metadata.h`:

```python
metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
    num_heads_q, num_heads_kv, head_dim,
    seqused_q, seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv,
    batch_size, max_seqlen_q, max_seqlen_ori_kv, max_seqlen_cmp_kv,
    ori_topk, cmp_topk, cmp_ratio,
    ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
    layout_q="BSND", layout_kv="BSND", has_ori_kv=True, has_cmp_kv=...,
)
out = torch.ops.cann_ops_transformer.sparse_flash_mla(
    q, ori_kv=ori_kv, cmp_kv=cmp_kv, cmp_sparse_indices=cmp_sparse_indices,
    ori_block_table=None, cmp_block_table=None,
    cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None,
    seqused_q=None, seqused_ori_kv=..., seqused_cmp_kv=..., cmp_residual_kv=...,
    sinks=sinks, metadata=metadata, softmax_scale=...,
    cmp_ratio=..., ori_mask_mode=..., cmp_mask_mode=...,
    ori_win_left=..., ori_win_right=...,
    layout_q="BSND", layout_kv="BSND",
    topk_value_mode=1,   # topk applies to cmp_sparse_indices rows of cmp_kv
    return_softmax_lse=False,
)
```

- `q` `[B, S1, H, D]`, `ori_kv` `[B, S0, KV_N, D]`, `cmp_kv`
  `[B, S2, KV_N, D]`, `cmp_sparse_indices` `[B, S1, KV_N, topk]` (CSA only),
  `sinks` `[H]`, `attrs` `[6]` fp32. KV_N=1 and D=512 are operator
  constraints; K and V share storage (`cmp_kv`/`ori_kv`).
- `attrs`: `[softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode,
  ori_win_left, ori_win_right]`; `-1` means "unused". `softmax_scale`
  is always set explicitly; `cmp_ratio` is `-1` when there is no cmp segment.
- Mask modes: `0` no-mask, `3` causal, `4` band (window).

### Contract details verified against the packaged implementation

- **`sinks` is required.** `GetSinks` in the tiling rejects a nullptr even
  though the product README lists it as optional. Pass a zeros vector
  `[H]` fp32 for "no sinks" (matches the golden's zero-sinks semantics).
- **`cmp_residual_kv` is required whenever `cmp_kv` is present with
  `cmp_mask_mode == 3` and `cmp_ratio != 1`** (`CheckSingleParaCmpResidualKv`).
  It restores the pre-compression length used for the cmp-side causal
  threshold: `ori_len = cmp_len * cmp_ratio + residual`. For full-length
  BSND tensors pass `seqused_ori_kv = [ori_seq]`,
  `seqused_cmp_kv = [cmp_seq]`, `cmp_residual_kv = [ori_seq % cmp_ratio]`.
- **CSA topk semantics** (verified against the arch35 CSA kernel
  `ComputeS2LoopInfo`/`GetRealCmpS2Idx`): per query row, only the first
  `min(topk, threshold)` entries of `cmp_sparse_indices` are considered at
  all; a `-1` entry stops the scan; entries `>= threshold` are masked out
  (causal rule), the scan continues past them. `topk_value_mode=1`.
- **Metadata output** is `[1024]` int32 = FA slots (36 AIC cores × 9 fields)
  followed by FD slots (72 AIV cores × 8 fields), i.e. the per-core
  start/len/dst offsets the main op consumes.
- **Platform constraints.** A5/950PR: `Q_N` ∈ [2, 128] (1 unsupported),
  `ori_mask_mode` ∈ {0, 3, 4}, `cmp_mask_mode` ∈ {0, 3}, `ori_win_left` -1 or
  ≥ 0, `cmp_ratio` ∈ [1, 128]. A2/A3: `ori_mask_mode` only 4,
  `cmp_mask_mode` only 3, `win_left/right` 127/0, HCA `cmp_ratio` = 128,
  CSA `cmp_ratio` = 4, `topk` ∈ {512, 1024}.

Three scenarios are covered by the same op:

| scenario | ori_kv | cmp_kv | cmp_sparse_indices |
|----------|--------|--------|--------------------|
| SWA (sliding-window attention) | full KV, band/causal/no-mask window | unused | unused |
| HCA (head-compression attention) | KV window | compressed prefix, leading `floor(causal / cmp_ratio)` rows | unused |
| CSA (chunk-stream attention) | KV window | compressed prefix | per-query top-k row indices, gathered as extra attention keys (first `min(K, threshold)` entries, -1 stops) |

## TileLang kernel snapshot

`design/tile_level/sparse_flash_mla.py` is the TileLang DSL source of a
SWA/HCA kernel (one kernel instance per batch row, ori window + cmp prefix
walked with flash-softmax and attention sinks as virtual keys). It compiles
for every row of `general_case.csv` on the 910B toolchain. The emitted AscendC
for the headline `swa_decode` config is snapshotted in:

- `sparse_flash_mla_kernel_swa_decode.cpp`

Regenerate it (or any other config) with the tilelang-enabled interpreter
(`/usr/local/python3.11.10/bin/python3`, `design/` on the path):

```python
import sys
sys.path.insert(0, "design")
from tile_level.sparse_flash_mla import sparse_flash_mla_fwd, SparseFlashMlaConfig
cfg = SparseFlashMlaConfig(batch=1, q_seq=1, heads=128, dim=512, ori_seq=2048,
                           cmp_seq=0, cmp_ratio=1, ori_mask_mode=4,
                           cmp_mask_mode=3, ori_win_left=127, ori_win_right=0,
                           dtype="bfloat16")
print(sparse_flash_mla_fwd(cfg).get_kernel_source())
```

CSA (topk gather of `cmp_kv` rows) is not yet expressible as a branch-free
TileLang kernel; `model_new_tilelang.py` raises `NotImplementedError` for it
and the golden path (`model.py`) remains the reference — see
`design/block_level/sparse_flash_mla.py` for the extension point discussion.
