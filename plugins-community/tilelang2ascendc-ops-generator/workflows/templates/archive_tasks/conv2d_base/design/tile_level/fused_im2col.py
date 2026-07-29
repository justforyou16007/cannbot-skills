#!/usr/bin/env python3
"""Tile-level fused NC1HWC0 window expansion and GEMM."""

from dataclasses import dataclass

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

from ..block_level.conv2d_base import (
    BASE_K,
    BASE_M,
    BASE_N,
    C0,
    FUSED_L1_BUDGET,
    ceildiv,
    output_extent,
    round_up,
)

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@dataclass(frozen=True)
class FusedIm2colPlan:
    """Static fused schedule and the layouts shared with auxiliary kernels."""

    ho: int
    wo: int
    hw: int
    ci_storage: int
    m_size: int
    m_storage: int
    n_storage: int
    k_size: int
    k_storage: int
    m_blocks: int
    n_blocks: int
    k_panels: int
    total_blocks: int
    resident_groups: int
    used_cores: int
    groups_per_core: int
    x_l1_bytes: int
    weight_l1_bytes: int
    x_layout: str = "logical [HW,Ci], physical NC1HWC0 in L1"
    weight_k_order: tuple = ("ci1", "kh", "kw", "ci0")
    accumulator_layout: str = "[N,M_storage,Co_storage] FP32"


def make_fused_im2col_plan(cfg):
    """Create the plan and enforce the fused path's layout/L1 contract."""
    if cfg.ci % C0 != 0:
        raise ValueError("fused im2col requires a complete C0 channel group")
    ho = output_extent(
        cfg.height, cfg.kernel_h, cfg.stride_h, cfg.pad_h, cfg.dilation_h
    )
    wo = output_extent(
        cfg.width, cfg.kernel_w, cfg.stride_w, cfg.pad_w, cfg.dilation_w
    )
    hw = cfg.height * cfg.width
    ci_storage = round_up(cfg.ci, C0)
    m_size = ho * wo
    m_storage = round_up(m_size, BASE_M)
    n_storage = round_up(round_up(cfg.co, C0), BASE_N)
    k_size = cfg.ci * cfg.kernel_h * cfg.kernel_w
    k_storage = round_up(k_size, BASE_K)
    m_blocks = m_storage // BASE_M
    n_blocks = n_storage // BASE_N
    total_blocks = cfg.batch * m_blocks * n_blocks
    resident_groups = cfg.batch * n_blocks
    used_cores = min(20, resident_groups)
    dtype_bytes = 2
    x_l1_bytes = hw * ci_storage * dtype_bytes
    weight_l1_bytes = k_storage * BASE_N * dtype_bytes
    if x_l1_bytes + weight_l1_bytes > FUSED_L1_BUDGET:
        raise ValueError("fused resident tensors exceed the L1 design budget")
    return FusedIm2colPlan(
        ho=ho,
        wo=wo,
        hw=hw,
        ci_storage=ci_storage,
        m_size=m_size,
        m_storage=m_storage,
        n_storage=n_storage,
        k_size=k_size,
        k_storage=k_storage,
        m_blocks=m_blocks,
        n_blocks=n_blocks,
        k_panels=ceildiv(k_size, BASE_K),
        total_blocks=total_blocks,
        resident_groups=resident_groups,
        used_cores=used_cores,
        groups_per_core=ceildiv(resident_groups, used_cores),
        x_l1_bytes=x_l1_bytes,
        weight_l1_bytes=weight_l1_bytes,
    )


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def conv2d_base_process_fused(cfg):
    """Mirror ``Conv2dBaseKernel::ProcessFused`` and ``ComputeBlockFused``."""
    plan = make_fused_im2col_plan(cfg)

    @T.prim_func
    def process_fused(
        x_channel_inner: T.Tensor(
            (cfg.batch, plan.hw, plan.ci_storage), cfg.dtype
        ),
        # K = (ci1, kh, kw, ci0).
        weight_kmajor: T.Tensor(
            (plan.k_storage, plan.n_storage), cfg.dtype
        ),
        acc: T.Tensor(
            (cfg.batch, plan.m_storage, plan.n_storage), "float32"
        ),
    ):
        with T.Kernel(plan.used_cores, is_npu=True) as (cid, _):
            # x_l1 is an image tensor, not a GEMM matrix. T.tile.im2col owns
            # its NC1HWC0 interpretation; only the B matrix receives Zn
            # annotation for the subsequent L1 -> L0B copy.
            x_l1 = T.alloc_L1((plan.hw, plan.ci_storage), cfg.dtype)
            b_l1 = T.alloc_L1((plan.k_storage, BASE_N), cfg.dtype)
            T.annotate_layout({
                b_l1: make_zn_layout(b_l1),
            })
            a_l0 = T.alloc_L0A((BASE_M, BASE_K), cfg.dtype)
            b_l0 = T.alloc_L0B((BASE_K, BASE_N), cfg.dtype)
            c_l0 = T.alloc_L0C((BASE_M, BASE_N), "float32")
            # ProcessFused residency: x is resident for an image and B is
            # resident for an output-channel tile.  Grouping (image,n_block)
            # keeps both live while ComputeBlockFused walks all m blocks.
            for local_group in T.serial(plan.groups_per_core):
                group_id = cid + local_group * plan.used_cores
                if group_id < plan.resident_groups:
                    image = group_id // plan.n_blocks
                    n_block = group_id % plan.n_blocks
                    n_start = n_block * BASE_N
                    with T.Scope("C"):
                        T.copy(x_channel_inner[image, 0, 0], x_l1)
                        T.copy(weight_kmajor[0, n_start], b_l1)
                        for m_block in T.serial(plan.m_blocks):
                            m_start = m_block * BASE_M
                            m_actual = T.min(
                                BASE_M, plan.m_size - m_start
                            )
                            for k_panel in T.Pipelined(
                                plan.k_panels, num_stages=2
                            ):
                                k_start = k_panel * BASE_K
                                k_actual = T.min(
                                    BASE_K, plan.k_size - k_start
                                )
                                # LoadA3D: NC1HWC0 L1 -> im2col L0A.
                                T.tile.im2col(
                                    a_l0,
                                    x_l1,
                                    image_shape=(
                                        cfg.height, cfg.width, cfg.ci,
                                    ),
                                    kernel=(
                                        cfg.kernel_h, cfg.kernel_w,
                                    ),
                                    stride=(
                                        cfg.stride_h, cfg.stride_w,
                                    ),
                                    dilation=(
                                        cfg.dilation_h, cfg.dilation_w,
                                    ),
                                    padding=(
                                        cfg.pad_w, cfg.pad_w,
                                        cfg.pad_h, cfg.pad_h,
                                    ),
                                    pos_m=m_start,
                                    pos_k=k_start,
                                    valid_m=m_actual,
                                    valid_k=k_actual,
                                    pad_value=0.0,
                                )
                                # SplitB: resident K-major L1 -> L0B.
                                T.copy(b_l1[k_start, 0], b_l0)
                                # Compute: first K panel clears L0C; later
                                # panels accumulate, matching cmatrixInitVal.
                                T.mma(
                                    a_l0, b_l0, c_l0,
                                    init=(k_panel == 0),
                                )
                            # CopyOut: L0C -> [image,m,co] FP32 workspace.
                            T.copy(
                                c_l0, acc[image, m_start, n_start]
                            )

    return process_fused


# Compatibility alias used by model_new_tilelang.py.
conv2d_base_fused = conv2d_base_process_fused


__all__ = [
    "FusedIm2colPlan",
    "make_fused_im2col_plan",
    "conv2d_base_process_fused",
    "conv2d_base_fused",
]
