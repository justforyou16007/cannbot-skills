#!/usr/bin/env python3
"""Block schedules for materialized im2col producer and GEMM consumer."""

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

from .conv2d_base import (
    BASE_K, BASE_M, BASE_N, C0, ceildiv, output_extent, round_up,
)

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _problem(cfg):
    ho = output_extent(
        cfg.height, cfg.kernel_h, cfg.stride_h, cfg.pad_h, cfg.dilation_h
    )
    wo = output_extent(
        cfg.width, cfg.kernel_w, cfg.stride_w, cfg.pad_w, cfg.dilation_w
    )
    m_size = ho * wo
    m_storage = round_up(m_size, BASE_M)
    k_size = cfg.kernel_h * cfg.kernel_w * round_up(cfg.ci, C0)
    k_storage = round_up(k_size, BASE_K)
    n_storage = round_up(round_up(cfg.co, C0), BASE_N)
    return m_size, m_storage, k_size, k_storage, n_storage


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def conv2d_base_im2col_process_block(cfg):
    m_size, m_storage, k_size, k_storage, _ = _problem(cfg)
    wo = output_extent(
        cfg.width, cfg.kernel_w, cfg.stride_w, cfg.pad_w, cfg.dilation_w
    )
    ci_pad = round_up(cfg.ci, C0)
    use_fast = cfg.ci % C0 == 0 and cfg.kernel_w <= 7
    shape = (
        (cfg.batch, cfg.ci, cfg.height, cfg.width)
        if cfg.data_format == "NCHW"
        else (cfg.batch, cfg.height, cfg.width, cfg.ci)
    )
    m_blocks = m_storage // BASE_M
    k_blocks = k_storage // BASE_K
    total = cfg.batch * m_blocks * k_blocks
    cores = min(40, total)
    tasks_per_core = ceildiv(total, cores)

    @T.prim_func
    def process_im2col(
        x: T.Tensor(shape, cfg.dtype),
        workspace: T.Tensor(
            (cfg.batch, m_storage, k_storage), cfg.dtype
        ),
    ):
        with T.Kernel(cores, is_npu=True) as (cid, _):
            tile_ub = T.alloc_ub((BASE_M, BASE_K), cfg.dtype)
            nchw_16x16 = T.alloc_ub((C0, C0), cfg.dtype)
            channel_inner_16x16 = T.alloc_ub((C0, C0), cfg.dtype)
            for task in T.serial(tasks_per_core):
                tile = cid + task * cores
                if tile < total:
                    image = tile // (m_blocks * k_blocks)
                    rem = tile % (m_blocks * k_blocks)
                    m_start = (rem // k_blocks) * BASE_M
                    k_start = (rem % k_blocks) * BASE_K
                    with T.Scope("V"):
                        T.tile.fill(tile_ub, 0.0)
                        if use_fast:
                            # Fast path batches both M and channel by C0.
                            # NCHW forms [C0,M16] then uses the documented
                            # 16x16 transpose; NHWC copies each C0 vector.
                            for m_group in T.serial(BASE_M // C0):
                                for k_group in T.serial(BASE_K // C0):
                                    k_base = k_start + k_group * C0
                                    channel_base = k_base % ci_pad
                                    kernel_index = k_base // ci_pad
                                    kh = kernel_index // cfg.kernel_w
                                    kw = kernel_index % cfg.kernel_w
                                    T.tile.fill(nchw_16x16, 0.0)
                                    T.tile.fill(channel_inner_16x16, 0.0)
                                    if cfg.data_format == "NCHW":
                                        for ci0 in T.unroll(C0):
                                            for m_lane in T.unroll(C0):
                                                m_index = (
                                                    m_start + m_group * C0
                                                    + m_lane
                                                )
                                                oh = m_index // wo
                                                ow = m_index % wo
                                                ih = (
                                                    oh * cfg.stride_h
                                                    - cfg.pad_h
                                                    + kh * cfg.dilation_h
                                                )
                                                iw = (
                                                    ow * cfg.stride_w
                                                    - cfg.pad_w
                                                    + kw * cfg.dilation_w
                                                )
                                                valid = T.And(
                                                    T.And(
                                                        m_index < m_size,
                                                        k_base < k_size,
                                                    ),
                                                    T.And(
                                                        T.And(
                                                            ih >= 0,
                                                            ih < cfg.height,
                                                        ),
                                                        T.And(
                                                            iw >= 0,
                                                            iw < cfg.width,
                                                        ),
                                                    ),
                                                )
                                                if valid:
                                                    nchw_16x16[
                                                        ci0, m_lane
                                                    ] = x[
                                                        image,
                                                        channel_base + ci0,
                                                        ih,
                                                        iw,
                                                    ]
                                        T.tile.transpose(
                                            channel_inner_16x16,
                                            nchw_16x16,
                                        )
                                    else:
                                        for m_lane in T.unroll(C0):
                                            m_index = (
                                                m_start + m_group * C0 + m_lane
                                            )
                                            oh = m_index // wo
                                            ow = m_index % wo
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
                                                    m_index < m_size,
                                                    k_base < k_size,
                                                ),
                                                T.And(
                                                    T.And(
                                                        ih >= 0,
                                                        ih < cfg.height,
                                                    ),
                                                    T.And(
                                                        iw >= 0,
                                                        iw < cfg.width,
                                                    ),
                                                ),
                                            )
                                            if valid:
                                                T.copy(
                                                    x[
                                                        image, ih, iw,
                                                        channel_base,
                                                    ],
                                                    channel_inner_16x16[
                                                        m_lane, 0
                                                    ],
                                                )
                                    for m_lane in T.unroll(C0):
                                        for ci0 in T.unroll(C0):
                                            tile_ub[
                                                m_group * C0 + m_lane,
                                                k_group * C0 + ci0,
                                            ] = channel_inner_16x16[
                                                m_lane, ci0
                                            ]
                        else:
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
                                            tile_ub[
                                                local_m, local_k
                                            ] = x[image, channel, ih, iw]
                                        else:
                                            tile_ub[
                                                local_m, local_k
                                            ] = x[image, ih, iw, channel]
                        T.copy(
                            tile_ub,
                            workspace[image, m_start, k_start],
                        )

    return process_im2col


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def conv2d_base_process_block(cfg):
    _, m_storage, _, k_storage, n_storage = _problem(cfg)
    m_blocks = m_storage // BASE_M
    n_blocks = n_storage // BASE_N
    total = cfg.batch * m_blocks * n_blocks
    cores = min(20, total)
    tasks_per_core = ceildiv(total, cores)

    @T.prim_func
    def process(
        workspace: T.Tensor(
            (cfg.batch, m_storage, k_storage), cfg.dtype
        ),
        weight: T.Tensor((k_storage, n_storage), cfg.dtype),
        acc: T.Tensor((cfg.batch, m_storage, n_storage), "float32"),
    ):
        with T.Kernel(cores, is_npu=True) as (cid, _):
            a_l1 = T.alloc_L1((BASE_M, BASE_K), cfg.dtype)
            b_l1 = T.alloc_L1((BASE_K, BASE_N), cfg.dtype)
            T.annotate_layout({
                a_l1: make_zn_layout(a_l1),
                b_l1: make_zn_layout(b_l1),
            })
            a_l0 = T.alloc_L0A((BASE_M, BASE_K), cfg.dtype)
            b_l0 = T.alloc_L0B((BASE_K, BASE_N), cfg.dtype)
            c_l0 = T.alloc_L0C((BASE_M, BASE_N), "float32")
            for task in T.serial(tasks_per_core):
                block = cid + task * cores
                if block < total:
                    image = block // (m_blocks * n_blocks)
                    rem = block % (m_blocks * n_blocks)
                    m_start = (rem // n_blocks) * BASE_M
                    n_start = (rem % n_blocks) * BASE_N
                    with T.Scope("C"):
                        for panel in T.Pipelined(
                            k_storage // BASE_K, num_stages=2
                        ):
                            k_start = panel * BASE_K
                            # LoadA and LoadB.
                            T.copy(
                                workspace[image, m_start, k_start], a_l1
                            )
                            T.copy(weight[k_start, n_start], b_l1)
                            # SplitA and SplitB.
                            T.copy(a_l1, a_l0)
                            T.copy(b_l1, b_l0)
                            # Compute.
                            T.mma(a_l0, b_l0, c_l0, init=(panel == 0))
                        # CopyOut.
                        T.copy(c_l0, acc[image, m_start, n_start])

    return process


conv2d_base_im2col_producer_block = conv2d_base_im2col_process_block
conv2d_base_materialized_gemm_block = conv2d_base_process_block


def materialized_launch_pipeline(cfg):
    m_size, m_storage, k_size, k_storage, n_storage = _problem(cfg)
    m_blocks = m_storage // BASE_M
    k_blocks = k_storage // BASE_K
    n_blocks = n_storage // BASE_N
    return {
        "launch_order": (
            "conv2d_base_im2col_producer_block",
            "conv2d_base_materialized_gemm_block",
            "conv2d_base_epilogue",
        ),
        "dependency": "im2col_workspace[N,M_storage,K_storage]",
        "workspace_shape": (cfg.batch, m_storage, k_storage),
        "workspace_elements": cfg.batch * m_storage * k_storage,
        "accumulator_shape": (cfg.batch, m_storage, n_storage),
        "logical_m": m_size,
        "logical_k": k_size,
        "producer_grid": (cfg.batch, m_blocks, k_blocks),
        "consumer_grid": (cfg.batch, m_blocks, n_blocks),
        "producer_core_type": "AIV",
        "consumer_core_type": "AIC",
        "producer_strategy": (
            "c0_grouped_transpose"
            if cfg.ci % C0 == 0 and cfg.kernel_w <= 7
            else "scalar_index_decode"
        ),
        "k_order": ("kh", "kw", "ci1", "ci0"),
        "consumer_dataflow": "GM->L1->L0->MMA->L0C->GM",
        "pipeline_stages": 2,
    }


__all__ = [
    "conv2d_base_im2col_producer_block",
    "conv2d_base_materialized_gemm_block",
    "conv2d_base_im2col_process_block",
    "conv2d_base_process_block",
    "materialized_launch_pipeline",
]
