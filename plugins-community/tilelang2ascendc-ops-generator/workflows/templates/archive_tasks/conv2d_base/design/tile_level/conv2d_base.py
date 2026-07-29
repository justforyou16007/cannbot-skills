#!/usr/bin/env python3
"""Unified exports and public auxiliary kernels for TileLang Conv2D."""

from dataclasses import dataclass

import tilelang
import tilelang.language as T

from ..block_level.conv2d_base import (
    BASE_K, BASE_M, BASE_N, C0, ceildiv, output_extent, round_up,
)

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@dataclass(frozen=True)
class TilePipelineContract:
    """End-to-end launch and intermediate-layout contract for one route."""

    mode: str
    stages: tuple
    x_layout: str
    weight_layout: str
    accumulator_layout: str
    output_layout: str
    workspace_layout: str


def resolve_tile_pipeline(cfg, mode):
    """Describe every TileLang kernel launch and its data dependency."""
    output_layout = (
        "[N,Co,Ho,Wo]" if cfg.data_format == "NCHW" else "[N,Ho,Wo,Co]"
    )
    if mode == "direct_1x1":
        return TilePipelineContract(
            mode=mode,
            stages=("process_direct", "epilogue"),
            x_layout="raw NCHW viewed as [N,Ci,H*W]",
            weight_layout="raw OIHW viewed as [Co,Ci] for 1x1",
            accumulator_layout="[N,Co_storage,HW_storage] FP32",
            output_layout=output_layout,
            workspace_layout="padded FP32 accumulator only",
        )
    if mode == "fused_im2col":
        fused_stages = (
            ("xtrans", "weight_repack", "process_fused", "epilogue")
            if cfg.data_format == "NCHW"
            else ("weight_repack", "process_fused", "epilogue")
        )
        return TilePipelineContract(
            mode=mode,
            stages=fused_stages,
            x_layout="[N,HW,Ci_storage], interpreted as NC1HWC0 in L1",
            weight_layout="[K_storage,Co_storage], K=(ci1,kh,kw,ci0)",
            accumulator_layout="[N,M_storage,Co_storage] FP32",
            output_layout=output_layout,
            workspace_layout="accumulator only",
        )
    if mode == "materialized_im2col":
        return TilePipelineContract(
            mode=mode,
            stages=(
                "weight_repack", "software_im2col",
                "materialized_gemm", "epilogue",
            ),
            x_layout="public NCHW/NHWC",
            weight_layout="[K_storage,Co_storage], K=(kh,kw,ci1,ci0)",
            accumulator_layout="[N,M_storage,Co_storage] FP32",
            output_layout=output_layout,
            workspace_layout="[N,M_storage,K_storage]",
        )
    raise ValueError(f"unsupported conv2d tile pipeline: {mode}")


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def conv2d_base_xtrans_process(cfg, mode):
    """Normalize x for direct ([N,C,HW]) or fused ([N,HW,C]) compute."""
    hw = cfg.height * cfg.width
    ci_storage = (
        round_up(cfg.ci, BASE_K)
        if mode == "direct_1x1"
        else round_up(cfg.ci, C0)
    )
    if mode == "direct_1x1":
        hw_storage = round_up(hw, BASE_N)
        out_shape = (cfg.batch, ci_storage, hw_storage)
    else:
        out_shape = (cfg.batch, hw, ci_storage)
    in_shape = (
        (cfg.batch, cfg.ci, cfg.height, cfg.width)
        if cfg.data_format == "NCHW"
        else (cfg.batch, cfg.height, cfg.width, cfg.ci)
    )
    tiles = cfg.batch * ceildiv(hw, 128)
    cores = min(40, tiles)

    @T.prim_func
    def process_xtrans(
        x_raw: T.Tensor(in_shape, cfg.dtype),
        x_normalized: T.Tensor(out_shape, cfg.dtype),
    ):
        with T.Kernel(cores, is_npu=True) as (cid, _):
            src_ub = T.alloc_ub((128, ci_storage), cfg.dtype)
            dst_ub = T.alloc_ub((ci_storage, 128), cfg.dtype)
            for tile in T.serial(ceildiv(tiles, cores)):
                task = cid + tile * cores
                if task < tiles:
                    image = task // ceildiv(hw, 128)
                    hw_start = (task % ceildiv(hw, 128)) * 128
                    valid_hw = T.min(128, hw - hw_start)
                    with T.Scope("V"):
                        T.tile.fill(src_ub, 0.0)
                        for local_hw in T.serial(valid_hw):
                            spatial = hw_start + local_hw
                            ih = spatial // cfg.width
                            iw = spatial % cfg.width
                            for channel in T.serial(cfg.ci):
                                if cfg.data_format == "NCHW":
                                    src_ub[local_hw, channel] = x_raw[
                                        image, channel, ih, iw
                                    ]
                                else:
                                    src_ub[local_hw, channel] = x_raw[
                                        image, ih, iw, channel
                                    ]
                        if mode == "direct_1x1":
                            for ci_block in T.serial(ci_storage // C0):
                                for hw_block in T.serial(128 // C0):
                                    T.tile.transpose(
                                        dst_ub[
                                            ci_block * C0, hw_block * C0
                                        ],
                                        src_ub[
                                            hw_block * C0, ci_block * C0
                                        ],
                                    )
                            T.copy(
                                dst_ub,
                                x_normalized[image, 0, hw_start],
                            )
                        else:
                            T.copy(
                                src_ub,
                                x_normalized[image, hw_start, 0],
                            )

    return process_xtrans


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def conv2d_base_weight_repack_process(cfg, mode):
    """Pack OIHW into the K order and padded matrix required by ``mode``."""
    ci_pad = round_up(cfg.ci, C0)
    if mode == "direct_1x1":
        out_shape = (round_up(cfg.co, BASE_M), round_up(cfg.ci, BASE_K))
    else:
        k_size = cfg.kernel_h * cfg.kernel_w * ci_pad
        out_shape = (
            round_up(k_size, BASE_K),
            round_up(round_up(cfg.co, C0), BASE_N),
        )
    row_blocks = ceildiv(out_shape[0], C0)
    col_blocks = ceildiv(out_shape[1], C0)
    total_blocks = row_blocks * col_blocks
    cores = min(40, total_blocks)

    @T.prim_func
    def process_weight_repack(
        weight_oihw: T.Tensor(
            (cfg.co, cfg.ci, cfg.kernel_h, cfg.kernel_w), cfg.dtype
        ),
        weight_packed: T.Tensor(out_shape, cfg.dtype),
    ):
        with T.Kernel(cores, is_npu=True) as (cid, _):
            tile_ub = T.alloc_ub((C0, C0), cfg.dtype)
            for task in T.serial(ceildiv(total_blocks, cores)):
                tile = cid + task * cores
                if tile < total_blocks:
                    row_start = (tile // col_blocks) * C0
                    col_start = (tile % col_blocks) * C0
                    with T.Scope("V"):
                        T.tile.fill(tile_ub, 0.0)
                        for local_row in T.serial(C0):
                            for local_col in T.serial(C0):
                                row = row_start + local_row
                                col = col_start + local_col
                                if mode == "direct_1x1":
                                    if row < cfg.co and col < cfg.ci:
                                        tile_ub[local_row, local_col] = (
                                            weight_oihw[row, col, 0, 0]
                                        )
                                else:
                                    ci1_count = ci_pad // C0
                                    ci0 = row % C0
                                    packed = row // C0
                                    if mode == "fused_im2col":
                                        kw = packed % cfg.kernel_w
                                        packed = packed // cfg.kernel_w
                                        kh = packed % cfg.kernel_h
                                        ci1 = packed // cfg.kernel_h
                                    else:
                                        ci1 = packed % ci1_count
                                        packed = packed // ci1_count
                                        kw = packed % cfg.kernel_w
                                        kh = packed // cfg.kernel_w
                                    channel = ci1 * C0 + ci0
                                    if (
                                        row < out_shape[0]
                                        and col < cfg.co
                                        and channel < cfg.ci
                                    ):
                                        tile_ub[local_row, local_col] = (
                                            weight_oihw[
                                                col, channel, kh, kw
                                            ]
                                        )
                        T.copy(
                            tile_ub,
                            weight_packed[row_start, col_start],
                        )

    return process_weight_repack


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def conv2d_base_epilogue_process(cfg, mode, has_bias):
    """Crop FP32 acc, add bias, cast, and restore NCHW/NHWC."""
    ho = output_extent(
        cfg.height, cfg.kernel_h, cfg.stride_h, cfg.pad_h, cfg.dilation_h
    )
    wo = output_extent(
        cfg.width, cfg.kernel_w, cfg.stride_w, cfg.pad_w, cfg.dilation_w
    )
    direct = mode == "direct_1x1"
    if direct:
        acc_shape = (
            cfg.batch, round_up(cfg.co, BASE_M),
            round_up(cfg.height * cfg.width, BASE_N),
        )
    else:
        acc_shape = (
            cfg.batch, round_up(ho * wo, BASE_M),
            round_up(round_up(cfg.co, C0), BASE_N),
        )
    out_shape = (
        (cfg.batch, cfg.co, ho, wo)
        if cfg.data_format == "NCHW"
        else (cfg.batch, ho, wo, cfg.co)
    )
    rows = cfg.batch * ho * wo
    tiles = ceildiv(rows, 64) * ceildiv(cfg.co, 64)
    cores = min(40, tiles)

    @T.prim_func
    def process_epilogue(
        acc: T.Tensor(acc_shape, "float32"),
        bias: T.Tensor((cfg.co,), cfg.dtype),
        output: T.Tensor(out_shape, cfg.dtype),
    ):
        with T.Kernel(cores, is_npu=True) as (cid, _):
            acc_ub = T.alloc_ub((64, 64), "float32")
            out_ub = T.alloc_ub((64, 64), cfg.dtype)
            bias_typed_ub = T.alloc_ub((64,), cfg.dtype)
            bias_ub = T.alloc_ub((64,), "float32")
            for task in T.serial(ceildiv(tiles, cores)):
                tile = cid + task * cores
                if tile < tiles:
                    row_tile = tile // ceildiv(cfg.co, 64)
                    co_tile = tile % ceildiv(cfg.co, 64)
                    row_start = row_tile * 64
                    co_start = co_tile * 64
                    valid_rows = T.min(64, rows - row_start)
                    valid_co = T.min(64, cfg.co - co_start)
                    with T.Scope("V"):
                        T.tile.fill(acc_ub, 0.0)
                        for local_row in T.serial(valid_rows):
                            row = row_start + local_row
                            image = row // (ho * wo)
                            spatial = row % (ho * wo)
                            for local_co in T.serial(valid_co):
                                channel = co_start + local_co
                                if direct:
                                    acc_ub[local_row, local_co] = acc[
                                        image, channel, spatial
                                    ]
                                else:
                                    acc_ub[local_row, local_co] = acc[
                                        image, spatial, channel
                                    ]
                        if has_bias:
                            T.tile.fill(bias_typed_ub, 0.0)
                            T.tile.fill(bias_ub, 0.0)
                            T.copy(
                                bias[co_start:co_start + valid_co],
                                bias_typed_ub[0:valid_co],
                            )
                            T.tile.cast(
                                bias_ub, bias_typed_ub,
                                mode="CAST_NONE", count=64,
                            )
                            for local_row in T.serial(64):
                                T.tile.add(
                                    acc_ub[local_row, 0],
                                    acc_ub[local_row, 0],
                                    bias_ub,
                                )
                        T.tile.cast(
                            out_ub, acc_ub, mode="CAST_RINT",
                            count=64 * 64,
                        )
                        for local_row in T.serial(valid_rows):
                            row = row_start + local_row
                            image = row // (ho * wo)
                            spatial = row % (ho * wo)
                            oh = spatial // wo
                            ow = spatial % wo
                            for local_co in T.serial(valid_co):
                                channel = co_start + local_co
                                if cfg.data_format == "NCHW":
                                    output[image, channel, oh, ow] = out_ub[
                                        local_row, local_co
                                    ]
                                else:
                                    output[image, oh, ow, channel] = out_ub[
                                        local_row, local_co
                                    ]

    return process_epilogue


# Public compatibility aliases retained for model_new_tilelang.py.
conv2d_base_xtrans = conv2d_base_xtrans_process
conv2d_base_weight_repack = conv2d_base_weight_repack_process
conv2d_base_epilogue = conv2d_base_epilogue_process


from .direct_1x1 import (  # noqa: E402
    Direct1x1Plan,
    conv2d_base_direct_1x1,
    conv2d_base_process_direct,
    make_direct_1x1_plan,
)
from .fused_im2col import (  # noqa: E402
    FusedIm2colPlan,
    conv2d_base_fused,
    conv2d_base_process_fused,
    make_fused_im2col_plan,
)
from .materialized_im2col import (  # noqa: E402
    MaterializedIm2colPlan,
    conv2d_base_materialized_gemm,
    conv2d_base_im2col_process,
    conv2d_base_process,
    conv2d_base_software_im2col,
    make_materialized_im2col_plan,
    materialized_stage_contract,
)

__all__ = [
    "TilePipelineContract",
    "resolve_tile_pipeline",
    "Direct1x1Plan",
    "make_direct_1x1_plan",
    "FusedIm2colPlan",
    "make_fused_im2col_plan",
    "MaterializedIm2colPlan",
    "make_materialized_im2col_plan",
    "materialized_stage_contract",
    "conv2d_base_process_direct",
    "conv2d_base_process_fused",
    "conv2d_base_im2col_process",
    "conv2d_base_process",
    "conv2d_base_direct_1x1",
    "conv2d_base_fused",
    "conv2d_base_materialized_gemm",
    "conv2d_base_software_im2col",
    "conv2d_base_xtrans",
    "conv2d_base_weight_repack",
    "conv2d_base_epilogue",
    "conv2d_base_xtrans_process",
    "conv2d_base_weight_repack_process",
    "conv2d_base_epilogue_process",
]
