#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED to NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# AscendC 泛化 kernel 入口（deliverer Phase 5 工程化产物：kernel/ 注册的
# matmul_gen_{fp16,bf16,fp32}，运行期 M/N/K，设计见 design/tile_level/matmul.py）。

import ctypes
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu  # noqa: F401

import sysconfig

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
_so_path = _KERNEL_BUILD / f"matmul_gen_ext{_ext_suffix}"
if _so_path.is_file():
    ctypes.CDLL(str(_so_path), mode=ctypes.RTLD_GLOBAL)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        if a.dtype == torch.float16:
            return torch.ops.npu.matmul_gen_fp16(a, b)
        elif a.dtype == torch.bfloat16:
            return torch.ops.npu.matmul_gen_bf16(a, b)
        elif a.dtype == torch.float32:
            return torch.ops.npu.matmul_gen_fp32(a, b)
        else:
            raise ValueError(f"Unsupported dtype: {a.dtype}")


def get_init_inputs():
    return []
