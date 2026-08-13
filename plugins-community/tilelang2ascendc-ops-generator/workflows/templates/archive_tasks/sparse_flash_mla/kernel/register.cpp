#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("sparse_flash_mla_metadata(int num_heads_q, int num_heads_kv, int head_dim, *, "
          "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, "
          "Tensor? seqused_q=None, Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, "
          "Tensor? cmp_residual_kv=None, Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, "
          "int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_ori_kv=None, int? max_seqlen_cmp_kv=None, "
          "int? ori_topk=None, int? cmp_topk=None, int? cmp_ratio=None, int? ori_mask_mode=None, "
          "int? cmp_mask_mode=None, int? ori_win_left=None, int? ori_win_right=None, str? layout_q=None, "
          "str? layout_kv=None, bool? has_ori_kv=None, bool? has_cmp_kv=None) -> Tensor",
        TORCH_FN(op_api::SparseFlashMlaMetadata));

    m.def("sparse_flash_mla(Tensor q, *, "
          "Tensor? ori_kv=None, Tensor? cmp_kv=None, "
          "Tensor? ori_sparse_indices=None, Tensor? cmp_sparse_indices=None, "
          "Tensor? ori_block_table=None, Tensor? cmp_block_table=None, "
          "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, "
          "Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, "
          "Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, "
          "Tensor? cmp_residual_kv=None, "
          "Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, "
          "Tensor? sinks=None, Tensor? metadata=None, "
          "float softmax_scale=1.0, int cmp_ratio=1, "
          "int ori_mask_mode=4, int cmp_mask_mode=3, "
          "int ori_win_left=127, int ori_win_right=0, "
          "str layout_q=\"BSND\", str layout_kv=\"BSND\", "
          "int topk_value_mode=1, bool return_softmax_lse=False) -> (Tensor, Tensor)",
        TORCH_FN(op_api::SparseFlashMla));
}
