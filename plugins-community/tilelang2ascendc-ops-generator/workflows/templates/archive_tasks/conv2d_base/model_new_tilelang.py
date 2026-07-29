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

"""Unified TileLang wrapper for the three ``conv2d_base`` designs.

The fused kernel uses ``T.tile.im2col``.  Importing this module is safe while
that intrinsic is under integration; building the fused kernel requires the
TileLang registration and Ascend lowering described in ``design/README.md``.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn


_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from design.block_level.conv2d_base import (  # noqa: E402
    Conv2dConfig,
    output_extent,
    select_mode,
)
from design.tile_level.conv2d_base import (  # noqa: E402
    conv2d_base_epilogue as tl_epilogue,
    conv2d_base_direct_1x1 as tl_direct_1x1,
    conv2d_base_fused as tl_fused_im2col,
    conv2d_base_materialized_gemm as tl_materialized_gemm,
    conv2d_base_software_im2col as tl_software_im2col,
    conv2d_base_weight_repack as tl_weight_repack,
    conv2d_base_xtrans as tl_xtrans,
)


class ModelNew(nn.Module):
    """Conv2D with direct, fused-im2col, and materialized-im2col routing."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _parse_pair(name, value, *, allow_zero):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must be a two-element list or tuple")
        result = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(f"{name} values must be integers, got {value}")
            if item < 0 or (item == 0 and not allow_zero):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} values must be {qualifier}, got {value}")
            result.append(int(item))
        return tuple(result)

    @staticmethod
    def _dtype_name(dtype):
        return str(dtype).split(".")[-1]

    def _make_config(
        self, x, weight, stride, padding, dilation, data_format
    ):
        if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
            raise TypeError("x and weight must be torch.Tensor instances")
        if x.ndim != 4 or weight.ndim != 4:
            raise ValueError(
                f"x and weight must be 4D, got x:{x.ndim}D weight:{weight.ndim}D"
            )
        if data_format not in ("NCHW", "NHWC"):
            raise ValueError(
                f"data_format must be 'NCHW' or 'NHWC', got {data_format!r}"
            )
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"unsupported dtype {x.dtype}; expected float16 or bfloat16"
            )
        if weight.dtype != x.dtype:
            raise ValueError(
                f"dtype mismatch: x is {x.dtype}, weight is {weight.dtype}"
            )
        if weight.device != x.device:
            raise ValueError(
                f"device mismatch: x is on {x.device}, weight is on {weight.device}"
            )

        stride_h, stride_w = self._parse_pair(
            "stride", stride, allow_zero=False
        )
        pad_h, pad_w = self._parse_pair(
            "padding", padding, allow_zero=True
        )
        dilation_h, dilation_w = self._parse_pair(
            "dilation", dilation, allow_zero=False
        )

        co, weight_ci, kernel_h, kernel_w = map(int, weight.shape)
        if min(co, weight_ci, kernel_h, kernel_w) <= 0:
            raise ValueError(f"weight dimensions must be positive: {weight.shape}")

        if data_format == "NCHW":
            batch, ci, height, width = map(int, x.shape)
        else:
            batch, height, width, ci = map(int, x.shape)
        if min(batch, ci, height, width) <= 0:
            raise ValueError(f"x dimensions must be positive: {x.shape}")
        if ci != weight_ci:
            raise ValueError(
                f"input channels mismatch: x has {ci}, weight has {weight_ci}"
            )

        cfg = Conv2dConfig(
            batch=batch,
            ci=ci,
            height=height,
            width=width,
            co=co,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
            data_format=data_format,
            dtype=self._dtype_name(x.dtype),
        )
        ho = output_extent(
            height, kernel_h, stride_h, pad_h, dilation_h
        )
        wo = output_extent(
            width, kernel_w, stride_w, pad_w, dilation_w
        )
        if ho <= 0 or wo <= 0:
            raise ValueError(f"calculated output size is invalid: ({ho}, {wo})")
        return cfg, ho, wo

    @staticmethod
    def _validate_bias(bias, x, co):
        if bias is None:
            return
        if not isinstance(bias, torch.Tensor):
            raise TypeError("bias must be a torch.Tensor or None")
        if bias.ndim != 1 or int(bias.shape[0]) != co:
            raise ValueError(f"bias must have shape [{co}], got {tuple(bias.shape)}")
        if bias.dtype != x.dtype:
            raise ValueError(
                f"dtype mismatch: x is {x.dtype}, bias is {bias.dtype}"
            )
        if bias.device != x.device:
            raise ValueError(
                f"device mismatch: x is on {x.device}, bias is on {bias.device}"
            )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias=None,
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        data_format="NCHW",
    ):
        cfg, ho, wo = self._make_config(
            x, weight, stride, padding, dilation, data_format
        )
        self._validate_bias(bias, x, cfg.co)
        mode = select_mode(cfg)

        if mode == "direct_1x1":
            kernel = tl_direct_1x1(cfg)
            acc = kernel(weight, x)
        elif mode == "fused_im2col":
            repack = tl_weight_repack(cfg, mode)
            weight_packed = repack(weight)
            if cfg.data_format == "NCHW":
                xtrans = tl_xtrans(cfg, mode)
                x_channel_inner = xtrans(x)
            else:
                x_channel_inner = x
            kernel = tl_fused_im2col(cfg)
            acc = kernel(x_channel_inner, weight_packed)
        else:
            repack = tl_weight_repack(cfg, mode)
            weight_packed = repack(weight)
            im2col_kernel = tl_software_im2col(cfg)
            gemm_kernel = tl_materialized_gemm(cfg)
            im2col_workspace = im2col_kernel(x)
            acc = gemm_kernel(im2col_workspace, weight_packed)

        # bias_arg is a metadata-compatible view when bias is absent; the
        # epilogue's static has_bias flag prevents any read from it.
        bias_arg = bias if bias is not None else weight[:, 0, 0, 0]
        epilogue = tl_epilogue(cfg, mode, bias is not None)
        return epilogue(acc, bias_arg)
