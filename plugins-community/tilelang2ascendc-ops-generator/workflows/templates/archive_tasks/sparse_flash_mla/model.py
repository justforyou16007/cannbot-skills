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

"""SparseFlashMla torch reference model.

Reference implementation of the SparseFlashMla operator for the BSND/BSND
layout pair. It covers the three supported attention scenarios:

* SWA (Sliding Window Attention): only ``ori_kv`` participates, masked by
  a band/causal window ``[q_pos - ori_win_left - 1, q_pos + ori_win_right]``.
* HCA (Heavily Compressed Attention): ``cmp_kv`` holds the compressed KV;
  the first ``floor(causal_pos / cmp_ratio)`` compressed tokens are attended.
* CSA (Compressed Sparse Attention): a subset of ``cmp_kv`` tokens selected
  by ``cmp_sparse_indices`` is attended. Per query, only the first
  ``min(K, threshold)`` entries of the [K] index row are considered; -1
  padding stops the scan, and entries ``>= threshold`` are masked (see
  ``gather_cmp_indices``).

Math::

    O = softmax_with_sinks(Q @ K^T * softmax_scale) @ V
    K = V = concat(ori_kv[win], cmp_kv_selected)

where the softmax denominator additionally contains ``exp(sinks)`` (attention
sinks only normalize, they contribute no value). All accumulation is done in
fp32; the output is cast back to the input dtype.

Case attributes (mask modes, window bounds, compression ratio, softmax scale)
are carried by the ``attrs`` tensor so that both the reference model and the
AscendC wrapper consume identical parameters:

    attrs = [softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode,
             ori_win_left, ori_win_right]

``-1`` marks a parameter as unused (e.g. ``cmp_ratio`` for SWA).
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn

# attribute tensor layout
ATTR_SOFTMAX_SCALE = 0
ATTR_CMP_RATIO = 1
ATTR_ORI_MASK_MODE = 2
ATTR_CMP_MASK_MODE = 3
ATTR_ORI_WIN_LEFT = 4
ATTR_ORI_WIN_RIGHT = 5

# ori_mask_mode / cmp_mask_mode values (see sparse_flash_mla README)
MASK_NO = 0
MASK_CAUSAL = 3  # RightDownCausal
MASK_BAND = 4    # Sliding window band, needs ori_win_left / ori_win_right

UNLIMITED = -1  # window bound -1 means the direction is unbounded


def parse_attrs(attrs):
    """Decode the attributes tensor into a plain dict of floats."""
    if attrs is None:
        return {
            "softmax_scale": None,
            "cmp_ratio": None,
            "ori_mask_mode": MASK_BAND,
            "cmp_mask_mode": MASK_CAUSAL,
            "ori_win_left": 127.0,
            "ori_win_right": 0.0,
        }
    values = attrs.detach().float().cpu().reshape(-1).tolist()
    return {
        "softmax_scale": float(values[ATTR_SOFTMAX_SCALE]),
        "cmp_ratio": float(values[ATTR_CMP_RATIO]) if values[ATTR_CMP_RATIO] >= 0 else None,
        "ori_mask_mode": int(values[ATTR_ORI_MASK_MODE]),
        "cmp_mask_mode": int(values[ATTR_CMP_MASK_MODE]),
        "ori_win_left": float(values[ATTR_ORI_WIN_LEFT]),
        "ori_win_right": float(values[ATTR_ORI_WIN_RIGHT]),
    }


def resolve_scenario(cmp_kv, cmp_sparse_indices):
    """Derive the attention scenario from which optional inputs are present."""
    if cmp_kv is None:
        return "SWA"
    if cmp_sparse_indices is None:
        return "HCA"
    return "CSA"


def ori_window(q_pos, q_len, ori_len, mask_mode, win_left, win_right):
    """Return the [start, end) token slice of ori_kv attended by query q_pos.

    Band mode (4): the causal position is ``ori_len - q_len + q_pos + 1``;
    the window covers ``[causal - win_left - 1, causal + win_right)``.
    Causal mode (3): ``[0, causal)``. No-mask mode (0): ``[0, ori_len)``.
    """
    if mask_mode == MASK_NO:
        return 0, ori_len
    causal = ori_len - q_len + q_pos + 1
    if mask_mode == MASK_CAUSAL:
        return 0, max(causal, 0)
    if mask_mode == MASK_BAND:
        end = ori_len if win_right == UNLIMITED else min(max(causal + win_right, 0), ori_len)
        start = 0 if win_left == UNLIMITED else max(causal - win_left - 1, 0)
        return int(start), int(end)
    raise ValueError(f"unsupported ori_mask_mode: {mask_mode}")


def cmp_threshold(q_pos, q_len, ori_len, cmp_len, cmp_ratio, mask_mode):
    """Number of leading compressed tokens valid for query q_pos.

    Causal mode (3): ``floor((ori_len - q_len + q_pos + 1) / cmp_ratio)``.
    No-mask mode (0): ``floor(ori_len / cmp_ratio)``.
    ``cmp_len`` must be the *effective* compressed length, i.e.
    ``floor(ori_len / cmp_ratio)`` (or an explicit ``seqused_cmp_kv``); the
    packaged operator recovers the same quantity via
    ``cmp_len * cmp_ratio + cmp_residual_kv == ori_len``.
    """
    if mask_mode == MASK_NO:
        numerator = ori_len
    elif mask_mode == MASK_CAUSAL:
        numerator = ori_len - q_len + q_pos + 1
    else:
        raise ValueError(f"unsupported cmp_mask_mode: {mask_mode}")
    return max(0, min(int(numerator // cmp_ratio), cmp_len))


def gather_cmp_indices(sparse_indices, threshold):
    """Select cmp_kv token indices from a per-query sparse index row.

    ``sparse_indices`` is the [K] int32 row for one query; -1 entries are
    padding. This mirrors the packaged operator's topk semantics (verified
    against the arch35 CSA kernel, ``ComputeS2LoopInfo``/``GetRealCmpS2Idx``):

    * only the first ``min(K, threshold)`` entries of the row are considered
      at all (the cmp-side loop length is capped by the causal threshold);
    * scanning stops at the first -1 padding entry;
    * within the scanned prefix, only indices ``< threshold`` are valid
      (causal rule); invalid entries are skipped, scanning continues.
    """
    scanned = sparse_indices[: min(sparse_indices.numel(), max(threshold, 0))]
    picked = []
    for idx in scanned.tolist():
        if idx == -1:
            break
        if 0 <= idx < threshold:
            picked.append(idx)
    if not picked:
        return torch.empty((0,), dtype=sparse_indices.dtype,
                           device=sparse_indices.device)
    return torch.tensor(picked, dtype=sparse_indices.dtype,
                        device=sparse_indices.device)


def softmax_with_sinks(scores, sinks):
    """Online-free softmax whose denominator also counts ``exp(sinks)``.

    scores: [G, S] fp32, sinks: [G] fp32 (one value per query head).
    The sinks act as virtual keys with no value contribution, so only the
    normalization is affected.
    """
    row_max = torch.maximum(scores.max(dim=-1, keepdim=True)[0], sinks.unsqueeze(-1))
    probs = torch.exp(scores - row_max)
    denom = probs.sum(dim=-1, keepdim=True) + torch.exp(sinks.unsqueeze(-1) - row_max)
    return probs / denom


def sparse_flash_mla_golden(
    q,
    ori_kv,
    cmp_kv=None,
    cmp_sparse_indices=None,
    sinks=None,
    attrs=None,
):
    """CPU/NPU reference for SparseFlashMla (BSND layout only).

    Args:
        q: [B, S1, N1, D] query.
        ori_kv: [B, S2, N2, D] uncompressed KV (K == V).
        cmp_kv: optional [B, S2c, N2, D] compressed KV.
        cmp_sparse_indices: optional [B, S1, N2, K] int32 topk indices into cmp_kv.
        sinks: optional [N1] fp32 attention sinks.
        attrs: optional [6] fp32 attribute tensor (see module docstring).

    Returns:
        [B, S1, N1, D] attention output in q's dtype.
    """
    if q.ndim != 4 or ori_kv.ndim != 4:
        raise ValueError("q and ori_kv must be 4D [B, S, N, D]")
    if q.dtype != ori_kv.dtype:
        raise ValueError("q and ori_kv must have the same dtype")

    batch, q_seq, q_heads, head_dim = q.shape
    _, ori_seq, kv_heads, _ = ori_kv.shape
    if q_heads % kv_heads != 0:
        raise ValueError(f"N1={q_heads} must be a multiple of N2={kv_heads}")

    group = q_heads // kv_heads
    scenario = resolve_scenario(cmp_kv, cmp_sparse_indices)
    attrs = parse_attrs(attrs)

    softmax_scale = attrs["softmax_scale"]
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    cmp_seq = cmp_kv.shape[1] if cmp_kv is not None else 0
    cmp_ratio = int(attrs["cmp_ratio"]) if attrs["cmp_ratio"] is not None else 1

    if sinks is None:
        sinks = torch.zeros((q_heads,), dtype=torch.float32, device=q.device)

    q_f32 = q.to(torch.float32)
    ori_f32 = ori_kv.to(torch.float32)
    cmp_f32 = cmp_kv.to(torch.float32) if cmp_kv is not None else None

    out = torch.zeros((batch, q_seq, q_heads, head_dim), dtype=torch.float32, device=q.device)
    for b in range(batch):
        for n2 in range(kv_heads):
            head_slice = slice(n2 * group, (n2 + 1) * group)
            sinks_row = sinks[head_slice]  # [G]
            for s1 in range(q_seq):
                q_row = q_f32[b, s1, head_slice, :]  # [G, D]

                start, end = ori_window(
                    s1, q_seq, ori_seq, attrs["ori_mask_mode"],
                    attrs["ori_win_left"], attrs["ori_win_right"],
                )
                k_tiles = []
                if start < end:
                    k_tiles.append(ori_f32[b, start:end, n2, :])  # [win, D]

                threshold = cmp_threshold(
                    s1, q_seq, ori_seq, cmp_seq, cmp_ratio, attrs["cmp_mask_mode"],
                ) if cmp_f32 is not None else 0
                if threshold > 0 and cmp_f32 is not None:
                    if scenario == "CSA":
                        indices = gather_cmp_indices(
                            cmp_sparse_indices[b, s1, n2, :], threshold)
                        if indices.numel() > 0:
                            k_tiles.append(cmp_f32[b, indices, n2, :])
                    else:  # HCA: leading compressed tokens
                        k_tiles.append(cmp_f32[b, :threshold, n2, :])

                if not k_tiles:
                    # no valid keys at all -> output stays zero
                    continue
                k_concat = torch.cat(k_tiles, dim=0)  # [S_sel, D]

                scores = q_row @ k_concat.transpose(0, 1) * softmax_scale
                probs = softmax_with_sinks(scores, sinks_row)
                out[b, s1, head_slice, :] = probs @ k_concat

    return out.to(q.dtype)


class Model(nn.Module):
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
        return sparse_flash_mla_golden(
            q, ori_kv, cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_sparse_indices, sinks=sinks, attrs=attrs,
        )


def _parse_row(row):
    """Parse one CSV row; blank topk/cmp_ratio columns mean 'not provided'.

    ``ori_win_left``/``ori_win_right`` keep -1 as a legal value (unbounded
    window direction), so they are parsed as plain ints.
    """
    return {
        "case_name": row["case_name"],
        "scenario": row["scenario"],
        "layout_q": row["layout_q"],
        "layout_kv": row["layout_kv"],
        "batch_size": int(row["batch_size"]),
        "seq_len_q": int(row["seq_len_q"]),
        "seq_len_kv": int(row["seq_len_kv"]),
        "n_heads_q": int(row["n_heads_q"]),
        "n_heads_kv": int(row["n_heads_kv"]),
        "head_dim": int(row["head_dim"]),
        "topk": int(row["topk"]) if row["topk"].strip() not in ("", "None") else None,
        "cmp_ratio": int(row["cmp_ratio"]) if row["cmp_ratio"].strip() not in ("", "None") else None,
        "ori_mask_mode": int(row["ori_mask_mode"]),
        "cmp_mask_mode": int(row["cmp_mask_mode"]),
        "ori_win_left": int(row["ori_win_left"]),
        "ori_win_right": int(row["ori_win_right"]),
        "dtype": row["dtype"],
    }


def _read_case_csvs():
    """Read basic_case.csv + general_case.csv into a list of parsed param dicts.

    Both files sit next to model.py and share the header documented in
    ``_parse_row``. basic_case.csv holds one canonical case (decode SWA); the
    general set covers prefill / HCA / CSA variations.
    """
    task_dir = Path(__file__).resolve().parent
    rows = []
    for name in ("basic_case.csv", "general_case.csv"):
        csv_path = task_dir / name
        if not csv_path.is_file():
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows.extend(_parse_row(row) for row in csv.DictReader(f))
    return rows


def get_inputs_from_case(cfg: dict) -> list:
    """根据CSV测试用例参数构造一组 [q, ori_kv, cmp_kv, cmp_sparse_indices,
    sinks, attrs] 输入;可选输入在场景不用时置 ``None``。"""
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[cfg["dtype"]]
    b, s1, s2 = cfg["batch_size"], cfg["seq_len_q"], cfg["seq_len_kv"]
    n1, n2, d = cfg["n_heads_q"], cfg["n_heads_kv"], cfg["head_dim"]

    torch.manual_seed(hash(cfg["case_name"]) & 0x7FFFFFFF)
    q = (torch.rand(b, s1, n1, d) * 4 - 2).to(dtype)
    ori_kv = (torch.rand(b, s2, n2, d) * 4 - 2).to(dtype)

    cmp_kv = None
    cmp_sparse_indices = None
    if cfg["scenario"] in ("HCA", "CSA"):
        # compressed length is the *effective* length floor(s2 / ratio)
        # (the op derives seqused_cmp_kv the same way); a ceil here
        # would make the cmp-side causal threshold (recovered as
        # cmp_len * ratio + residual) exceed the true ori_len.
        cmp_seq = max(s2 // cfg["cmp_ratio"], 1)
        cmp_kv = (torch.rand(b, cmp_seq, n2, d) * 4 - 2).to(dtype)
    if cfg["scenario"] == "CSA":
        # per (batch, query, kv-head) row: random permutation of the
        # valid compressed positions, padded with -1 up to `topk`. The
        # golden/op only scan the first min(K, threshold) entries and
        # stop at the first -1 (see gather_cmp_indices).
        indices = torch.full((b, s1, n2, cfg["topk"]), -1, dtype=torch.int32)
        for ib in range(b):
            for is1 in range(s1):
                for in2 in range(n2):
                    valid = cmp_seq
                    count = min(valid, cfg["topk"])
                    if count > 0:
                        indices[ib, is1, in2, :count] = torch.randperm(valid)[:count]
        cmp_sparse_indices = indices

    sinks = (torch.rand(n1) * 0.4 - 0.2).to(torch.float32)
    attrs = torch.tensor(
        [
            d ** -0.5,
            float(cfg["cmp_ratio"]) if cfg["cmp_ratio"] is not None else -1.0,
            float(cfg["ori_mask_mode"]),
            float(cfg["cmp_mask_mode"]),
            float(cfg["ori_win_left"]),
            float(cfg["ori_win_right"]),
        ],
        dtype=torch.float32,
    )
    return [q, ori_kv, cmp_kv, cmp_sparse_indices, sinks, attrs]


def get_input_groups():
    """Build input groups from basic_case.csv + general_case.csv (all
    scenarios incl. CSA) for AscendC verification.

    Each group is ``[q, ori_kv, cmp_kv, cmp_sparse_indices, sinks, attrs]``;
    optional inputs are ``None`` when the scenario does not use them.
    """
    return [get_inputs_from_case(cfg) for cfg in _read_case_csvs()]


def get_input_groups_tilelang():
    """TileLang-only input groups: SWA/HCA cases.

    The TileLang design covers SWA and HCA but not CSA (per-query topk
    gather); verification_tilelang.py prefers this selector when present so
    the TileLang path never sees CSA inputs, while the AscendC path keeps the
    full CSV set from ``get_input_groups()``.
    """
    return [
        get_inputs_from_case(cfg)
        for cfg in _read_case_csvs()
        if cfg["scenario"] != "CSA"
    ]


def get_init_inputs_from_case(params: dict) -> list:
    """根据CSV测试用例参数构造模型初始化参数"""
    return []


def get_init_inputs():
    return []
