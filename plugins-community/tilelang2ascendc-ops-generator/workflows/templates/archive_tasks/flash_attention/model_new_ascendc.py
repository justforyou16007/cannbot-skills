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

import ctypes
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import sysconfig

# Phase 5 工程化产物（deliverer 路线）：kernel/ 注册的
# flash_attention_gen_{fp16,bf16,fp32}，设备代码与 ascendc/flash_attention.cpp
# 同源（tilelang 编译产物的手工泛化，全量 50/50 验证）。
# 早期参考实现已移至 kernel_reference/（保留可构建）。
_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
_so_path = _KERNEL_BUILD / f"flash_attention_gen_ext{_ext_suffix}"
if _so_path.is_file():
    ctypes.CDLL(str(_so_path), mode=ctypes.RTLD_GLOBAL)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                is_causal: bool = False):
        if q.dtype == torch.float16:
            return torch.ops.npu.flash_attention_gen_fp16(q, k, v, is_causal)
        elif q.dtype == torch.bfloat16:
            return torch.ops.npu.flash_attention_gen_bf16(q, k, v, is_causal)
        elif q.dtype == torch.float32:
            return torch.ops.npu.flash_attention_gen_fp32(q, k, v, is_causal)
        else:
            raise ValueError(f"Unsupported dtype: {q.dtype}")


def get_init_inputs():
    return []
