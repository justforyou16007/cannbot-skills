#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"

try:
    import conv2d_base_ext  # noqa: F401 — whl path
except ImportError:
    import glob as _glob
    _libs = _glob.glob(str(_KERNEL_BUILD / "conv2d_base_ext*.so"))
    if _libs:
        torch.ops.load_library(_libs[0])


class ModelNew(nn.Module):
    """
    Conv2d via custom AscendC kernel (910B).

    Interface mirrors model.Model:
      forward(x, weight, bias=None, stride=[1,1], padding=[0,0],
              dilation=[1,1], data_format="NCHW", groups=1)
    x logical shape is always [N, C, H, W]; for data_format="NHWC" the
    physical tensor is [N, H, W, C]. weight is always [Co, Ci/groups, kH, kW].
    Layout normalization (NHWC + spatial/channel zero-pad + weight K-major)
    happens inside the host op. groups > 1 routes to the pure-AIV depthwise
    kernel (gate: NCHW, groups==C, Co%C==0, stride==1, dilation==1,
    kH*kW<=64); other grouped forms are rejected host-side.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias=None,
                stride=[1, 1], padding=[0, 0], dilation=[1, 1],
                data_format="NCHW", groups=1):
        return torch.ops.npu.conv2d_base(
            x, weight, bias,
            [int(stride[0]), int(stride[1])],
            [int(padding[0]), int(padding[1])],
            [int(dilation[0]), int(dilation[1])],
            str(data_format),
            int(groups))
