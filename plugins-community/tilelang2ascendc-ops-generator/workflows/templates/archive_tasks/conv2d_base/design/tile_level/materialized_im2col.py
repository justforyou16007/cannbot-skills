#!/usr/bin/env python3
"""Tile-level AIV im2col producer and AIC GEMM consumer."""

from dataclasses import dataclass

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

from ..block_level.conv2d_base import (
    BASE_K,
    BASE_M,
    BASE_N,
    C0,
    ceildiv,
    output_extent,
    round_up,
)

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@dataclass(frozen=True)
class MaterializedIm2colPlan:
    """Producer/consumer contract for the GM im2col implementation."""

    ho: int
    wo: int
    ci_pad: int
    m_size: int
    m_storage: int
    k_size: int
    k_storage: int
    n_storage: int
    producer_m_blocks: int
    producer_k_blocks: int
    consumer_n_blocks: int
    producer_tiles: int
    consumer_blocks: int
    workspace_elements: int
    strategy: str
    k_order: tuple = ("kh", "kw", "ci1", "ci0")
    workspace_layout: str = "[N,M_storage,K_storage] ND"
    accumulator_layout: str = "[N,M_storage,Co_storage] FP32"


def make_materialized_im2col_plan(cfg):
    """Describe both AIV production and AIC consumption of the workspace."""
    ho = output_extent(
        cfg.height, cfg.kernel_h, cfg.stride_h, cfg.pad_h, cfg.dilation_h
    )
    wo = output_extent(
        cfg.width, cfg.kernel_w, cfg.stride_w, cfg.pad_w, cfg.dilation_w
    )
    ci_pad = round_up(cfg.ci, C0)
    m_size = ho * wo
    m_storage = round_up(m_size, BASE_M)
    k_size = cfg.kernel_h * cfg.kernel_w * ci_pad
    k_storage = round_up(k_size, BASE_K)
    n_storage = round_up(round_up(cfg.co, C0), BASE_N)
    producer_m_blocks = m_storage // BASE_M
    producer_k_blocks = k_storage // BASE_K
    consumer_n_blocks = n_storage // BASE_N
    return MaterializedIm2colPlan(
        ho=ho,
        wo=wo,
        ci_pad=ci_pad,
        m_size=m_size,
        m_storage=m_storage,
        k_size=k_size,
        k_storage=k_storage,
        n_storage=n_storage,
        producer_m_blocks=producer_m_blocks,
        producer_k_blocks=producer_k_blocks,
        consumer_n_blocks=consumer_n_blocks,
        producer_tiles=(
            cfg.batch * producer_m_blocks * producer_k_blocks
        ),
        consumer_blocks=(
            cfg.batch * producer_m_blocks * consumer_n_blocks
        ),
        workspace_elements=cfg.batch * m_storage * k_storage,
        strategy=(
            "unrolled_k"
            if cfg.ci >= C0 and cfg.kernel_w <= 7
            else "serial_k"
        ),
    )


def _problem(cfg):
    plan = make_materialized_im2col_plan(cfg)
    return (
        plan.wo,
        plan.ci_pad,
        plan.m_size,
        plan.m_storage,
        plan.k_size,
        plan.k_storage,
        plan.n_storage,
    )


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def conv2d_base_im2col_process(cfg):
    """Mirror ``Conv2dIm2colKernel::Process`` fast/legacy dispatch."""
    plan = make_materialized_im2col_plan(cfg)
    wo, ci_pad, m_size, m_storage, k_size, k_storage, _ = _problem(cfg)
    x_shape = (
        (cfg.batch, cfg.ci, cfg.height, cfg.width)
        if cfg.data_format == "NCHW"
        else (cfg.batch, cfg.height, cfg.width, cfg.ci)
    )
    m_blocks = m_storage // BASE_M
    k_blocks = k_storage // BASE_K
    total_tiles = cfg.batch * m_blocks * k_blocks
    used_cores = min(40, total_tiles)
    tasks_per_core = ceildiv(total_tiles, used_cores)
    use_fast = plan.strategy == "unrolled_k"

    @T.prim_func
    def process_fast(
        x_raw: T.Tensor(x_shape, cfg.dtype),
        im2col_gm: T.Tensor(
            (cfg.batch, m_storage, k_storage), cfg.dtype
        ),
    ):
        with T.Kernel(used_cores, is_npu=True) as (cid, _):
            packed_ub = T.alloc_ub((BASE_M, BASE_K), cfg.dtype)
            for local_task in T.serial(tasks_per_core):
                tile_id = cid + local_task * used_cores
                if tile_id < total_tiles:
                    image = tile_id // (m_blocks * k_blocks)
                    remainder = tile_id % (m_blocks * k_blocks)
                    m_start = (remainder // k_blocks) * BASE_M
                    k_start = (remainder % k_blocks) * BASE_K
                    with T.Scope("V"):
                        T.tile.fill(packed_ub, 0.0)
                        # Fast schedule fixes BASE_K at compile time so the
                        # index decode and load sequence can be unrolled.
                        for local_m in T.serial(BASE_M):
                            m_index = m_start + local_m
                            oh = m_index // wo
                            ow = m_index % wo
                            for local_k in T.unroll(BASE_K):
                                k_index = k_start + local_k
                                channel = k_index % ci_pad
                                kernel_index = k_index // ci_pad
                                kh = kernel_index // cfg.kernel_w
                                kw = kernel_index % cfg.kernel_w
                                ih = (
                                    oh * cfg.stride_h - cfg.pad_h
                                    + kh * cfg.dilation_h
                                )
                                iw = (
                                    ow * cfg.stride_w - cfg.pad_w
                                    + kw * cfg.dilation_w
                                )
                                valid = T.And(
                                    T.And(
                                        T.And(
                                            m_index < m_size,
                                            k_index < k_size,
                                        ),
                                        channel < cfg.ci,
                                    ),
                                    T.And(
                                        T.And(ih >= 0, ih < cfg.height),
                                        T.And(iw >= 0, iw < cfg.width),
                                    ),
                                )
                                if valid:
                                    if cfg.data_format == "NCHW":
                                        packed_ub[
                                            local_m, local_k
                                        ] = x_raw[
                                            image, channel, ih, iw
                                        ]
                                    else:
                                        packed_ub[
                                            local_m, local_k
                                        ] = x_raw[
                                            image, ih, iw, channel
                                        ]
                        T.copy(
                            packed_ub,
                            im2col_gm[image, m_start, k_start],
                        )

    @T.prim_func
    def process_legacy(
        x_raw: T.Tensor(x_shape, cfg.dtype),
        im2col_gm: T.Tensor(
            (cfg.batch, m_storage, k_storage), cfg.dtype
        ),
    ):
        with T.Kernel(used_cores, is_npu=True) as (cid, _):
            tile_ub = T.alloc_ub((BASE_M, BASE_K), cfg.dtype)
            for local_task in T.serial(tasks_per_core):
                tile_id = cid + local_task * used_cores
                if tile_id < total_tiles:
                    image = tile_id // (m_blocks * k_blocks)
                    remainder = tile_id % (m_blocks * k_blocks)
                    m_start = (remainder // k_blocks) * BASE_M
                    k_start = (remainder % k_blocks) * BASE_K
                    with T.Scope("V"):
                        T.tile.fill(tile_ub, 0.0)
                        for local_m in T.serial(BASE_M):
                            m_index = m_start + local_m
                            oh = m_index // wo
                            ow = m_index % wo
                            for local_k in T.serial(BASE_K):
                                k_index = k_start + local_k
                                channel = k_index % ci_pad
                                kernel_index = k_index // ci_pad
                                kh = kernel_index // cfg.kernel_w
                                kw = kernel_index % cfg.kernel_w
                                ih = (
                                    oh * cfg.stride_h - cfg.pad_h
                                    + kh * cfg.dilation_h
                                )
                                iw = (
                                    ow * cfg.stride_w - cfg.pad_w
                                    + kw * cfg.dilation_w
                                )
                                if m_index < m_size:
                                    if k_index < k_size:
                                        if channel < cfg.ci:
                                            spatial_valid = T.And(
                                                T.And(ih >= 0, ih < cfg.height),
                                                T.And(iw >= 0, iw < cfg.width),
                                            )
                                            if spatial_valid:
                                                if cfg.data_format == "NCHW":
                                                    tile_ub[
                                                        local_m, local_k
                                                    ] = x_raw[
                                                        image, channel, ih, iw
                                                    ]
                                                else:
                                                    tile_ub[
                                                        local_m, local_k
                                                    ] = x_raw[
                                                        image, ih, iw, channel
                                                    ]
                        T.copy(
                            tile_ub,
                            im2col_gm[image, m_start, k_start],
                        )

    return process_fast if use_fast else process_legacy


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def conv2d_base_process(cfg):
    """Mirror normal ``Conv2dBaseKernel::Process`` over im2col GM."""
    plan = make_materialized_im2col_plan(cfg)
    _, _, m_size, m_storage, _, k_storage, n_storage = _problem(cfg)
    m_blocks = m_storage // BASE_M
    n_blocks = n_storage // BASE_N
    total_blocks = cfg.batch * m_blocks * n_blocks
    used_cores = min(20, total_blocks)
    tasks_per_core = ceildiv(total_blocks, used_cores)

    @T.prim_func
    def process(
        im2col_gm: T.Tensor(
            (cfg.batch, m_storage, k_storage), cfg.dtype
        ),
        # K = (kh, kw, ci1, ci0).
        weight_kmajor: T.Tensor((k_storage, n_storage), cfg.dtype),
        acc: T.Tensor((cfg.batch, m_storage, n_storage), "float32"),
    ):
        with T.Kernel(used_cores, is_npu=True) as (cid, _):
            a_l1 = T.alloc_L1((BASE_M, BASE_K), cfg.dtype)
            b_l1 = T.alloc_L1((BASE_K, BASE_N), cfg.dtype)
            T.annotate_layout({
                a_l1: make_zn_layout(a_l1),
                b_l1: make_zn_layout(b_l1),
            })
            a_l0 = T.alloc_L0A((BASE_M, BASE_K), cfg.dtype)
            b_l0 = T.alloc_L0B((BASE_K, BASE_N), cfg.dtype)
            c_l0 = T.alloc_L0C((BASE_M, BASE_N), "float32")
            for local_task in T.serial(tasks_per_core):
                block_id = cid + local_task * used_cores
                if block_id < total_blocks:
                    image = block_id // (m_blocks * n_blocks)
                    remainder = block_id % (m_blocks * n_blocks)
                    m_start = (remainder // n_blocks) * BASE_M
                    n_start = (remainder % n_blocks) * BASE_N
                    m_actual = T.min(BASE_M, m_size - m_start)
                    n_actual = T.min(BASE_N, cfg.co - n_start)
                    with T.Scope("C"):
                        for k_panel in T.Pipelined(
                            k_storage // BASE_K, num_stages=2
                        ):
                            k_start = k_panel * BASE_K
                            # LoadA and LoadB.
                            T.copy(
                                im2col_gm[image, m_start, k_start], a_l1
                            )
                            T.copy(
                                weight_kmajor[k_start, n_start], b_l1
                            )
                            # SplitA and SplitB.
                            T.copy(a_l1, a_l0)
                            T.copy(b_l1, b_l0)
                            # Compute.
                            T.mma(a_l0, b_l0, c_l0, init=(k_panel == 0))
                        # CopyOut.
                        T.copy(c_l0, acc[image, m_start, n_start])
                        _ = (m_actual, n_actual)

    return process


# Compatibility aliases used by model_new_tilelang.py.
conv2d_base_software_im2col = conv2d_base_im2col_process
conv2d_base_materialized_gemm = conv2d_base_process


def materialized_stage_contract(cfg):
    """Public launch dependency consumed by the wrapper and design docs."""
    plan = make_materialized_im2col_plan(cfg)
    return {
        "launch_order": (
            "conv2d_base_software_im2col",
            "conv2d_base_materialized_gemm",
            "conv2d_base_epilogue",
        ),
        "producer_core": "AIV",
        "consumer_core": "AIC",
        "dependency": plan.workspace_layout,
        "workspace_elements": plan.workspace_elements,
        "k_order": plan.k_order,
        "producer_strategy": plan.strategy,
    }


__all__ = [
    "MaterializedIm2colPlan",
    "make_materialized_im2col_plan",
    "materialized_stage_contract",
    "conv2d_base_im2col_process",
    "conv2d_base_process",
    "conv2d_base_software_im2col",
    "conv2d_base_materialized_gemm",
]
