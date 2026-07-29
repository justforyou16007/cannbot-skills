#!/usr/bin/env python3
"""Block schedule for fused im2col Conv2D."""

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

from .conv2d_base import (
    BASE_K, BASE_M, BASE_N, C0, FUSED_L1_BUDGET,
    ceildiv, output_extent, round_up,
)

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def fused_im2col_block_contract(cfg):
    """Return fused grid, residency scope, layouts, and L1 footprint."""
    ho = output_extent(
        cfg.height, cfg.kernel_h, cfg.stride_h, cfg.pad_h, cfg.dilation_h
    )
    wo = output_extent(
        cfg.width, cfg.kernel_w, cfg.stride_w, cfg.pad_w, cfg.dilation_w
    )
    m_size = ho * wo
    m_storage = round_up(m_size, BASE_M)
    k_size = cfg.ci * cfg.kernel_h * cfg.kernel_w
    k_storage = round_up(k_size, BASE_K)
    ci_storage = round_up(cfg.ci, C0)
    n_storage = round_up(round_up(cfg.co, C0), BASE_N)
    m_blocks = m_storage // BASE_M
    n_blocks = n_storage // BASE_N
    resident_groups = cfg.batch * n_blocks
    cores = min(20, resident_groups)
    l1_bytes = 2 * (
        cfg.height * cfg.width * ci_storage + k_storage * BASE_N
    )
    if l1_bytes > FUSED_L1_BUDGET:
        raise ValueError("fused block plan exceeds its physical L1 budget")
    return {
        "ho": ho,
        "wo": wo,
        "m_size": m_size,
        "m_storage": m_storage,
        "k_size": k_size,
        "k_storage": k_storage,
        "ci_storage": ci_storage,
        "n_storage": n_storage,
        "m_blocks": m_blocks,
        "n_blocks": n_blocks,
        "resident_groups": resident_groups,
        "cores": cores,
        "groups_per_core": ceildiv(resident_groups, cores),
        "k_panels": ceildiv(k_size, BASE_K),
        "pipeline_stages": 2,
        "l1_bytes": l1_bytes,
        "l1_budget": FUSED_L1_BUDGET,
        "residency_scope": "(image,n_block) across all m_blocks",
        "x_l1_layout": "NC1HWC0 image",
        "weight_k_order": ("ci1", "kh", "kw", "ci0"),
        "dataflow": "x/weight GM->resident L1->im2col/L0B->MMA->GM",
        "consumer_contract": "common epilogue/matrix",
    }


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def conv2d_base_process_fused_block(cfg):
    contract = fused_im2col_block_contract(cfg)
    m_size = contract["m_size"]
    m_storage = contract["m_storage"]
    k_size = contract["k_size"]
    k_storage = contract["k_storage"]
    ci_storage = contract["ci_storage"]
    n_storage = contract["n_storage"]
    m_blocks = contract["m_blocks"]
    n_blocks = contract["n_blocks"]

    @T.prim_func
    def process_fused(
        x: T.Tensor(
            (cfg.batch, cfg.height * cfg.width, ci_storage), cfg.dtype
        ),
        weight: T.Tensor((k_storage, n_storage), cfg.dtype),
        acc: T.Tensor((cfg.batch, m_storage, n_storage), "float32"),
    ):
        with T.Kernel(contract["cores"], is_npu=True) as (cid, _):
            x_l1 = T.alloc_L1(
                (cfg.height * cfg.width, ci_storage), cfg.dtype
            )
            b_l1 = T.alloc_L1((k_storage, BASE_N), cfg.dtype)
            T.annotate_layout({
                b_l1: make_zn_layout(b_l1),
            })
            a_l0 = T.alloc_L0A((BASE_M, BASE_K), cfg.dtype)
            b_l0 = T.alloc_L0B((BASE_K, BASE_N), cfg.dtype)
            c_l0 = T.alloc_L0C((BASE_M, BASE_N), "float32")
            for task in T.serial(contract["groups_per_core"]):
                group = cid + task * contract["cores"]
                if group < contract["resident_groups"]:
                    image = group // n_blocks
                    n_start = (group % n_blocks) * BASE_N
                    with T.Scope("C"):
                        # Resident across every M tile in this group.
                        T.copy(x[image, 0, 0], x_l1)
                        T.copy(weight[0, n_start], b_l1)
                        for m_block in T.serial(m_blocks):
                            m_start = m_block * BASE_M
                            for panel in T.Pipelined(
                                contract["k_panels"],
                                num_stages=contract["pipeline_stages"],
                            ):
                                k_start = panel * BASE_K
                                k_actual = T.min(
                                    BASE_K, k_size - k_start
                                )
                                # LoadA3D.
                                T.tile.im2col(
                                    a_l0, x_l1,
                                    image_shape=(
                                        cfg.height, cfg.width, cfg.ci,
                                    ),
                                    kernel=(cfg.kernel_h, cfg.kernel_w),
                                    stride=(cfg.stride_h, cfg.stride_w),
                                    dilation=(
                                        cfg.dilation_h, cfg.dilation_w,
                                    ),
                                    padding=(
                                        cfg.pad_w, cfg.pad_w,
                                        cfg.pad_h, cfg.pad_h,
                                    ),
                                    pos_m=m_start, pos_k=k_start,
                                    valid_m=T.min(
                                        BASE_M, m_size - m_start
                                    ),
                                    valid_k=k_actual, pad_value=0.0,
                                )
                                # SplitB then Compute.
                                T.copy(b_l1[k_start, 0], b_l0)
                                T.mma(
                                    a_l0, b_l0, c_l0,
                                    init=(panel == 0),
                                )
                            # CopyOut.
                            T.copy(
                                c_l0, acc[image, m_start, n_start]
                            )

    return process_fused


conv2d_base_fused_block = conv2d_base_process_fused_block


__all__ = [
    "fused_im2col_block_contract",
    "conv2d_base_process_fused_block",
    "conv2d_base_fused_block",
]
