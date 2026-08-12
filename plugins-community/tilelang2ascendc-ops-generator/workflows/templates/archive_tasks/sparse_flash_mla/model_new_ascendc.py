#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SparseFlashMla AscendC wrapper (local direct call).

The operator is a CANN-framework packaged op: the kernel sources live in
``kernel/op_host`` + ``kernel/op_kernel`` and are built by the ops-transformer
framework build (``-DASCEND_OP_NAME=sparse_flash_mla,sparse_flash_mla_metadata``,
see ``kernel/README.md``), which installs a ``custom_transformer`` vendor into
the CANN OPP and provides the aclnn interface (``aclnnSparseFlashMla`` +
``aclnnSparseFlashMlaMetadata`` in ``libcust_opapi.so``).

This wrapper loads the torch extension ``kernel/build/sparse_flash_mla_ext*``
(see ``kernel/op_host/sparse_flash_mla.cpp`` and ``kernel/register.cpp``),
which registers ``torch.ops.npu.sparse_flash_mla_metadata`` and
``torch.ops.npu.sparse_flash_mla`` backed by those aclnn calls, then mirrors
the two-step flow: metadata first (task partition), then the main op.

Two packaged-op contract details verified against the ops-transformer
implementation (``sparse_flash_mla_tiling.cpp``) are handled here:

* ``sinks`` is **required** by the tiling (``GetSinks`` rejects a nullptr),
  although the product README lists it as optional. ``None`` is therefore
  translated to a zeros vector, matching the golden's "no sinks" semantics.
* ``cmp_residual_kv`` is **required** whenever ``cmp_kv`` is present with
  ``cmp_mask_mode == 3`` and ``cmp_ratio != 1`` (``CheckSingleParaCmpResidualKv``).
  It restores the pre-compression KV length used for the cmp-side causal
  threshold: ``ori_len = cmp_len * cmp_ratio + residual``. For full-length
  BSND tensors the wrapper derives ``seqused_ori_kv = [ori_seq]``,
  ``seqused_cmp_kv = [cmp_seq]`` and ``cmp_residual_kv = [ori_seq % cmp_ratio]``
  exactly as the reference pytest does.
"""

import glob
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# --- 运行期 vendor 环境 ---
# ASCEND_CUSTOM_OPP_PATH：AICPU 引擎（metadata 算子）唯一扫描
# op_impl/cpu/config/cust_aicpu_kernel.json 的入口，不自动遍历 vendor，
# 缺失时 metadata 算子报 BinaryGetFunction 0x7080001。GE 在首次 NPU op
# 时才读取，故进程内 setdefault 有效；LD_LIBRARY_PATH（vendored
# libcust_opapi.so）需在 shell 里导出（见 kernel/README.md 步骤 5）。
if not os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
    _opp = os.environ.get("ASCEND_OPP_PATH",
                          "/usr/local/Ascend/cann-9.0.0/opp")
    _vendor = os.path.join(_opp, "vendors", "custom_transformer")
    if os.path.isdir(_vendor):
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = _vendor

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_LIB_PATTERN = str(_KERNEL_BUILD / "sparse_flash_mla_ext*")

# Prefer whl import (triggers TORCH_LIBRARY registration in register.cpp)
try:
    import sparse_flash_mla_ext  # noqa: F401
except ImportError:
    # Fallback: direct .so loading from kernel/build/
    if _LIB_PATTERN not in "".join(sys.path):
        _libs = glob.glob(_LIB_PATTERN)
        if _libs:
            torch.ops.load_library(_libs[0])


def _decode_attrs(attrs):
    """Decode [softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode,
    ori_win_left, ori_win_right] from the attribute tensor."""
    if attrs is None:
        return {
            "softmax_scale": None,
            "cmp_ratio": None,
            "ori_mask_mode": 4,
            "cmp_mask_mode": 3,
            "ori_win_left": 127,
            "ori_win_right": 0,
        }
    values = attrs.detach().cpu().float().reshape(-1).tolist()
    return {
        "softmax_scale": float(values[0]),
        "cmp_ratio": int(values[1]) if values[1] >= 0 else None,
        "ori_mask_mode": int(values[2]),
        "cmp_mask_mode": int(values[3]),
        "ori_win_left": int(values[4]),
        "ori_win_right": int(values[5]),
    }


def sparse_flash_mla_ascendc(
    q,
    ori_kv,
    cmp_kv=None,
    cmp_sparse_indices=None,
    sinks=None,
    attrs=None,
):
    """Build metadata on host, then run the SparseFlashMla main op."""
    attr = _decode_attrs(attrs)

    if q.ndim != 4 or ori_kv.ndim != 4:
        raise ValueError("q and ori_kv must be 4D [B, S, N, D]")
    if q.shape[-1] != 512:
        raise ValueError(f"SparseFlashMla requires D=512, got {q.shape[-1]}")
    if ori_kv.shape[-2] != 1:
        raise ValueError(f"SparseFlashMla requires KV_N=1, got {ori_kv.shape[-2]}")
    if not all(t.is_contiguous() for t in (q, ori_kv)):
        raise ValueError("q and ori_kv must be contiguous")
    if cmp_kv is not None and not cmp_kv.is_contiguous():
        raise ValueError("cmp_kv must be contiguous")
    if cmp_sparse_indices is not None and not cmp_sparse_indices.is_contiguous():
        raise ValueError("cmp_sparse_indices must be contiguous")

    softmax_scale = attr["softmax_scale"]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    cmp_ratio = 1 if attr["cmp_ratio"] is None else attr["cmp_ratio"]

    if cmp_kv is not None:
        # full-length BSND tensors: expose the effective lengths explicitly so
        # the op recovers the pre-compression length for the cmp-side causal
        # threshold (cmp_len * cmp_ratio + residual == ori_seq). Required by
        # the tiling when cmp_mask_mode == 3 and cmp_ratio != 1. The aclnn
        # check requires per-batch lengths (shape size == batch_size).
        batch = int(q.shape[0])
        cmp_seq = int(cmp_kv.shape[1])
        seqused_ori_kv = torch.tensor(
            [int(ori_kv.shape[1])] * batch, dtype=torch.int32, device=q.device)
        seqused_cmp_kv = torch.tensor(
            [cmp_seq] * batch, dtype=torch.int32, device=q.device)
        cmp_residual_kv = torch.tensor(
            [int(ori_kv.shape[1]) % cmp_ratio] * batch, dtype=torch.int32,
            device=q.device)
    else:
        seqused_ori_kv = None
        seqused_cmp_kv = None
        cmp_residual_kv = None

    if sinks is None:
        # the packaged op requires a non-empty sinks tensor (tiling rejects
        # nullptr); zeros == "no sinks" in the golden semantics
        sinks = torch.zeros((q.shape[2],), dtype=torch.float32, device=q.device)

    # metadata-first: the task partition must exist before the main op runs
    metadata = torch.ops.npu.sparse_flash_mla_metadata(
        int(q.shape[2]),          # num_heads_q
        int(ori_kv.shape[2]),     # num_heads_kv
        int(q.shape[3]),          # head_dim
        seqused_q=None,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        batch_size=int(q.shape[0]),
        max_seqlen_q=int(q.shape[1]),
        max_seqlen_ori_kv=int(ori_kv.shape[1]),
        max_seqlen_cmp_kv=int(cmp_kv.shape[1]) if cmp_kv is not None else 0,
        ori_topk=0,
        cmp_topk=int(cmp_sparse_indices.shape[-1]) if cmp_sparse_indices is not None else 0,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=attr["ori_mask_mode"],
        cmp_mask_mode=attr["cmp_mask_mode"],
        ori_win_left=attr["ori_win_left"],
        ori_win_right=attr["ori_win_right"],
        layout_q="BSND",
        layout_kv="BSND",
        has_ori_kv=True,
        has_cmp_kv=cmp_kv is not None,
    )

    out, _lse = torch.ops.npu.sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=None,
        cmp_block_table=None,
        cu_seqlens_q=None,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=float(softmax_scale),
        cmp_ratio=cmp_ratio,
        ori_mask_mode=attr["ori_mask_mode"],
        cmp_mask_mode=attr["cmp_mask_mode"],
        ori_win_left=attr["ori_win_left"],
        ori_win_right=attr["ori_win_right"],
        layout_q="BSND",
        layout_kv="BSND",
        topk_value_mode=1,
        return_softmax_lse=False,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        ori_kv: torch.Tensor,
        cmp_kv=None,
        cmp_sparse_indices=None,
        sinks=None,
        attrs=None,
    ):
        return sparse_flash_mla_ascendc(
            q, ori_kv, cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_sparse_indices, sinks=sinks, attrs=attrs,
        )


def get_init_inputs():
    return []
