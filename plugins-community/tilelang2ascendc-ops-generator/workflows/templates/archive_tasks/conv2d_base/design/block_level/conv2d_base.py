#!/usr/bin/env python3
"""Shared configuration, dispatch, and exports for block-level Conv2D."""

from dataclasses import dataclass

BASE_M = 128
BASE_N = 128
BASE_K = 64
C0 = 16
FUSED_L1_BUDGET = 500 * 1024


@dataclass(frozen=True)
class Conv2dConfig:
    batch: int
    ci: int
    height: int
    width: int
    co: int
    kernel_h: int
    kernel_w: int
    stride_h: int = 1
    stride_w: int = 1
    pad_h: int = 0
    pad_w: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    data_format: str = "NCHW"
    dtype: str = "float16"


@dataclass(frozen=True)
class DesignRoute:
    mode: str
    block_level: str
    tile_level: str
    stages: tuple


@dataclass(frozen=True)
class AuxiliarySchedule:
    stage: str
    core_type: str
    task_count: int
    tile_elements: int


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def round_up(value, alignment):
    return ceildiv(value, alignment) * alignment


def output_extent(input_extent, kernel, stride, padding, dilation):
    return (input_extent + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def validate_config(cfg):
    if cfg.data_format not in ("NCHW", "NHWC"):
        raise ValueError(f"unsupported data_format: {cfg.data_format}")
    if cfg.dtype not in ("float16", "bfloat16"):
        raise ValueError(f"unsupported dtype: {cfg.dtype}")
    if min(
        cfg.batch, cfg.ci, cfg.height, cfg.width, cfg.co,
        cfg.kernel_h, cfg.kernel_w, cfg.stride_h, cfg.stride_w,
        cfg.dilation_h, cfg.dilation_w,
    ) <= 0:
        raise ValueError("tensor extents, stride, and dilation must be positive")
    ho = output_extent(
        cfg.height, cfg.kernel_h, cfg.stride_h, cfg.pad_h, cfg.dilation_h
    )
    wo = output_extent(
        cfg.width, cfg.kernel_w, cfg.stride_w, cfg.pad_w, cfg.dilation_w
    )
    if ho <= 0 or wo <= 0:
        raise ValueError(f"invalid output extent: ({ho}, {wo})")


def select_mode(cfg):
    validate_config(cfg)
    direct = (
        cfg.data_format == "NCHW"
        and cfg.kernel_h == cfg.kernel_w == 1
        and cfg.stride_h == cfg.stride_w == 1
        and cfg.pad_h == cfg.pad_w == 0
        and cfg.dilation_h == cfg.dilation_w == 1
        and cfg.ci % C0 == 0
    )
    if direct:
        return "direct_1x1"
    ci_pad = round_up(cfg.ci, C0)
    k_size = ci_pad * cfg.kernel_h * cfg.kernel_w
    k_storage = round_up(k_size, BASE_K)
    # The fused Expert kernel allocates a complete BASE_N resident B panel.
    # Routing uses the physical allocation, not the logical Co extent.
    b_cols = BASE_N
    l1_bytes = 2 * (
        ci_pad * cfg.height * cfg.width + k_storage * b_cols
    )
    fused = (
        cfg.ci % C0 == 0
        and (cfg.data_format == "NHWC" or (cfg.height * cfg.width) % 2 == 0)
        and l1_bytes <= FUSED_L1_BUDGET
    )
    return "fused_im2col" if fused else "materialized_im2col"


def resolve_design(cfg):
    mode = select_mode(cfg)
    fused_stages = (
        (
            "xtrans_aiv", "weight_repack_aiv",
            "process_fused_aic", "epilogue_aiv",
        )
        if cfg.data_format == "NCHW"
        else ("weight_repack_aiv", "process_fused_aic", "epilogue_aiv")
    )
    stages = {
        "direct_1x1": (
            "process_direct_aic", "epilogue_aiv",
        ),
        "fused_im2col": fused_stages,
        "materialized_im2col": (
            "weight_repack_aiv", "software_im2col_aiv",
            "materialized_gemm_aic", "epilogue_aiv",
        ),
    }[mode]
    return DesignRoute(
        mode=mode,
        block_level=f"block_level/{mode}.py",
        tile_level=f"tile_level/{mode}.py",
        stages=stages,
    )


def resolve_auxiliary_schedules(cfg):
    """Block schedules for normalization, repack, and epilogue kernels."""
    mode = select_mode(cfg)
    ho = output_extent(
        cfg.height, cfg.kernel_h, cfg.stride_h, cfg.pad_h, cfg.dilation_h
    )
    wo = output_extent(
        cfg.width, cfg.kernel_w, cfg.stride_w, cfg.pad_w, cfg.dilation_w
    )
    weight_elements = cfg.co * cfg.ci * cfg.kernel_h * cfg.kernel_w
    schedules = []
    if mode == "fused_im2col" and cfg.data_format == "NCHW":
        schedules.append(AuxiliarySchedule(
            "xtrans", "AIV",
            cfg.batch * ceildiv(cfg.height * cfg.width, 128), 128 * C0,
        ))
    if mode != "direct_1x1":
        schedules.append(AuxiliarySchedule(
            "weight_repack", "AIV", ceildiv(weight_elements, 256), 256,
        ))
    schedules.append(
        AuxiliarySchedule(
            "epilogue", "AIV",
            ceildiv(cfg.batch * ho * wo, 64) * ceildiv(cfg.co, 64),
            64 * 64,
        )
    )
    return tuple(schedules)


from .direct_1x1 import (  # noqa: E402
    conv2d_base_direct_1x1_block,
    conv2d_base_process_direct_block,
    direct_1x1_block_contract,
)
from .fused_im2col import (  # noqa: E402
    conv2d_base_fused_block,
    conv2d_base_process_fused_block,
    fused_im2col_block_contract,
)
from .materialized_im2col import (  # noqa: E402
    conv2d_base_im2col_producer_block,
    conv2d_base_materialized_gemm_block,
    conv2d_base_im2col_process_block,
    conv2d_base_process_block,
    materialized_launch_pipeline,
)

__all__ = [
    "Conv2dConfig",
    "select_mode",
    "resolve_design",
    "resolve_auxiliary_schedules",
    "direct_1x1_block_contract",
    "fused_im2col_block_contract",
    "conv2d_base_direct_1x1_block",
    "conv2d_base_fused_block",
    "conv2d_base_im2col_producer_block",
    "conv2d_base_materialized_gemm_block",
    "conv2d_base_process_direct_block",
    "conv2d_base_process_fused_block",
    "conv2d_base_im2col_process_block",
    "conv2d_base_process_block",
    "materialized_launch_pipeline",
]
