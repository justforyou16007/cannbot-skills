#!/usr/bin/env python3
"""scenario-routing 族判定谓词引擎。

按 scenario-routing/README.md 判定表顺序把算子特征归到 (family, branch, traits)。
纯函数，无副作用。判序（README 兜底规则 4）：
    FA 指纹（QK^T+softmax）→ 递归外积（无 QK^T 的 SSM/Scan）→ 独立 softmax（通道注意力/归约）→ conv → matmul → … → elementwise。
"""

from __future__ import annotations
from typing import Any

# 正交 trait 词表（见 schema.md workflow_ref.traits）
TRAITS = {
    "phase_bwd": "phase=bwd",
    "causal": "causal",
    "sparse_blocktable": "sparse-blocktable",
    "sparse_window": "sparse-window",
    "paged_decode": "paged-decode",
    "quant": "quant",
    "transpose_layout": "transpose-layout",
    "batch": "batch",
    "grouped": "grouped",
    "swiglu": "swiglu",
}


def _channel_attention(f: dict) -> bool:
    """通道注意力：global pool(mean) + FC + sigmoid/softmax + 通道 scale，无 QK^T。"""
    return (not f.get("has_qkt")) and f.get("has_global_pool") and f.get("has_channel_scale") \
        and (f.get("has_softmax") or f.get("has_sigmoid"))


def route(features: dict[str, Any]) -> dict:
    """返回 {family, workflow_file, branch, traits}。"""
    f = features
    traits: list[str] = []
    head = f.get("head_structure", "")

    # ---- 1. Attention 系（QK^T 指纹，先于无 QK^T 判）----
    if f.get("has_qkt") and f.get("has_softmax_between_matmuls"):
        # 正交 trait
        if f.get("is_backward"):
            traits.append(TRAITS["phase_bwd"])
        if f.get("sparse_mechanism") == "block_table":
            traits.append(TRAITS["sparse_blocktable"])
        elif f.get("sparse_mechanism") == "window":
            traits.append(TRAITS["sparse_window"])
        elif f.get("has_causal_mask"):
            traits.append(TRAITS["causal"])
        if f.get("has_paged_kv"):
            traits.append(TRAITS["paged_decode"])
        if f.get("has_quant"):
            traits.append(TRAITS["quant"])
        # head 结构分支
        if head == "mla" or f.get("has_latent_compression"):
            branch = "mla"
        elif head == "gqa" or (f.get("num_q_heads") and f.get("num_kv_heads")
                               and f["num_q_heads"] > f["num_kv_heads"]):
            branch = "gqa"
        else:
            branch = "standard"
        return {"family": "attention", "workflow_file": "flashattention.md",
                "branch": branch, "traits": traits}

    # ---- 2. SSM/Scan 系（无 QK^T + chunk-scan / 递归）----
    is_ssm = (not f.get("has_qkt")) and (f.get("has_delta_rule") or f.get("has_gated_decay")
                                         or f.get("is_chunk_scan"))
    if is_ssm:
        if f.get("is_backward"):
            traits.append(TRAITS["phase_bwd"])
        if f.get("has_quant"):
            traits.append(TRAITS["quant"])
        if f.get("has_delta_rule"):
            branch = "delta-rule"
        else:
            branch = "gla"
        return {"family": "ssm", "workflow_file": "ssm-scan.md",
                "branch": branch, "traits": traits}

    # ---- 3. 通道注意力（有 softmax/sigmoid 但无 QK^T）→ reduction/elementwise ----
    if _channel_attention(f):
        return {"family": "reduction", "workflow_file": "reduction.md",
                "branch": "channel-attention", "traits": []}

    # ---- 4. Conv ----
    if f.get("has_conv_params"):
        if f.get("conv_groups") and f["conv_groups"] == f.get("conv_in_channels"):
            branch = "C-5"  # depthwise
        elif f.get("conv_kh") == 1 and f.get("conv_kw") == 1:
            branch = "C-1"  # 1x1
        else:
            branch = "C-3"
        return {"family": "conv", "workflow_file": "convolution.md",
                "branch": branch, "traits": []}

    # ---- 5. Matmul 系 ----
    if f.get("has_matmul_backbone"):
        if f.get("transA") and f.get("transB"):
            traits.append(TRAITS["transpose_layout"])
        elif f.get("transA") or f.get("transB"):
            traits.append(TRAITS["transpose_layout"])
        if f.get("has_batch"):
            traits.append(TRAITS["batch"])
        if f.get("has_grouped"):
            traits.append(TRAITS["grouped"])
        if f.get("has_swiglu"):
            traits.append(TRAITS["swiglu"])
        if f.get("has_quant"):
            traits.append(TRAITS["quant"])
        return {"family": "matmul", "workflow_file": "matmul.md",
                "branch": "M" + (str(f.get("m_branch", 1))), "traits": traits}

    # ---- 6. Reduction / Sort / Conversion / Elementwise 兜底 ----
    if f.get("has_reduction"):
        return {"family": "reduction", "workflow_file": "reduction.md",
                "branch": "R", "traits": []}
    if f.get("has_sort"):
        return {"family": "sort", "workflow_file": "misc-families.md",
                "branch": "sort", "traits": []}
    if f.get("has_conversion"):
        return {"family": "conversion", "workflow_file": "conversion.md",
                "branch": "R1", "traits": []}
    return {"family": "elementwise", "workflow_file": "elementwise.md",
            "branch": "EW", "traits": []}
