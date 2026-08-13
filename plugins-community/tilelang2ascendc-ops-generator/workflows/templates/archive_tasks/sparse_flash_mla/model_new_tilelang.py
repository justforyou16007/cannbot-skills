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

"""SparseFlashMla TileLang wrapper (SWA / HCA).

The TileLang design covers the SWA and HCA attention modes for any query
sequence length (decode is the headline scenario). The kernel expects the S
dimensions of ``ori_kv``/``cmp_kv`` padded to BLOCK_N alignment and a dummy
``cmp_kv`` for pure-SWA inputs; this wrapper performs that preparation and
decodes the same ``attrs`` tensor contract used by ``model.py``.

CSA (per-query topk gather of ``cmp_kv`` rows) is not covered by the design
yet; use ``model_new_ascendc.py`` (packaged CANN operator) for CSA inputs.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
for _p in (str(_TASK_DIR), str(_TASK_DIR / "design")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from design.tile_level.sparse_flash_mla import (  # noqa: E402
    sparse_flash_mla_fwd as tl_sparse_flash_mla_fwd,
    SparseFlashMlaConfig,
    BLOCK_N,
    BLOCK_H,
)


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
        "softmax_scale": float(values[0]) if values[0] >= 0 else None,
        "cmp_ratio": int(values[1]) if values[1] >= 0 else None,
        "ori_mask_mode": int(values[2]),
        "cmp_mask_mode": int(values[3]),
        "ori_win_left": int(values[4]),
        "ori_win_right": int(values[5]),
    }


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
        if q.ndim != 4 or ori_kv.ndim != 4:
            raise ValueError("q and ori_kv must be 4D [B, S, N, D]")
        if q.shape[-1] != 512:
            raise ValueError(f"SparseFlashMla requires D=512, got {q.shape[-1]}")
        if ori_kv.shape[-2] != 1:
            raise ValueError(f"SparseFlashMla requires KV_N=1, got {ori_kv.shape[-2]}")
        if cmp_sparse_indices is not None:
            raise NotImplementedError(
                "CSA (per-query topk gather) is not covered by the TileLang "
                "design; use model_new_ascendc.py (packaged CANN operator) "
                "for CSA inputs"
            )

        dtype_map = {torch.float16: "float16", torch.bfloat16: "bfloat16"}
        if q.dtype not in dtype_map:
            raise ValueError(
                f"tilelang design supports float16/bfloat16 only, got {q.dtype}"
            )
        dtype_str = dtype_map[q.dtype]

        batch, q_seq, heads, dim = q.shape
        ori_seq = int(ori_kv.shape[1])
        cmp_seq = int(cmp_kv.shape[1]) if cmp_kv is not None else 0
        if heads % BLOCK_H != 0:
            raise ValueError(f"tilelang design requires N1 % {BLOCK_H} == 0, "
                             f"got N1={heads}")

        attr = _decode_attrs(attrs)
        cmp_ratio = 1 if attr["cmp_ratio"] is None else attr["cmp_ratio"]
        softmax_scale = attr["softmax_scale"]
        if softmax_scale is None:
            softmax_scale = -1.0  # kernel default: 1 / sqrt(dim)

        if cmp_kv is None:
            # pure SWA: pass a dummy cmp_kv (never read, cmp_blocks == 0)
            cmp_kv = torch.zeros((batch, 1, 1, dim), dtype=q.dtype,
                                 device=q.device)
            cmp_seq = 0
            cmp_ratio = 1
        if sinks is None:
            # golden semantics: no sinks == zero sinks (softmax_with_sinks)
            sinks = torch.zeros((heads,), dtype=torch.float32, device=q.device)

        # pad the S dimensions to BLOCK_N so every KV copy in the kernel is a
        # static full-block copy; padded columns fall outside every window
        # and are masked out by the kernel.
        ori_pad = ((ori_seq + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        if ori_pad != ori_seq:
            padded = torch.zeros((batch, ori_pad, 1, dim), dtype=q.dtype,
                                 device=q.device)
            padded[:, :ori_seq, :, :] = ori_kv
            ori_kv = padded
        cmp_pad = max(((cmp_seq + BLOCK_N - 1) // BLOCK_N) * BLOCK_N, 1)
        if cmp_pad != cmp_seq:
            padded = torch.zeros((batch, cmp_pad, 1, dim), dtype=q.dtype,
                                 device=q.device)
            padded[:, :cmp_seq, :, :] = cmp_kv
            cmp_kv = padded

        cfg = SparseFlashMlaConfig(
            batch=batch,
            q_seq=q_seq,
            heads=heads,
            dim=dim,
            ori_seq=ori_seq,
            cmp_seq=cmp_seq,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=attr["ori_mask_mode"],
            cmp_mask_mode=attr["cmp_mask_mode"],
            ori_win_left=attr["ori_win_left"],
            ori_win_right=attr["ori_win_right"],
            softmax_scale=float(softmax_scale),
            dtype=dtype_str,
        )
        kernel = tl_sparse_flash_mla_fwd(cfg)
        # single KV stream [B, ori_pad | cmp_pad, D] (the design declares the
        # S dimensions concatenated as one 3D tensor; the kv-head dim of 1 is
        # squeezed), pre-allocated output + mask-index constant (the kernel
        # writes the output in place)
        kv_all = torch.zeros((batch, ori_pad + cmp_pad, dim), dtype=q.dtype,
                             device=q.device)
        kv_all[:, :ori_pad, :] = ori_kv[:, :, 0, :]
        kv_all[:, ori_pad:, :] = cmp_kv[:, :, 0, :]
        output = torch.empty((batch, q_seq, heads, dim),
                             dtype=q.dtype, device=q.device)
        idx = torch.arange(BLOCK_N, dtype=torch.float32, device=q.device)
        kernel(q, kv_all, sinks, idx, output)
        return output


def get_init_inputs():
    return []
