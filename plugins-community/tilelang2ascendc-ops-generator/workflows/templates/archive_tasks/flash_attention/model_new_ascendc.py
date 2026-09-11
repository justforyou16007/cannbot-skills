#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# MHA AscendC 入口（cann-bench level4/gqa，BSND 布局）。
# Phase 5 工程化产物：kernel/ 注册的 mha_gen_{fp16,bf16}，设备代码与
# ascendc/gqa.cpp 同源（tilelang 编译产物的手工泛化，全量 20/20 验证）。

import ctypes
import sysconfig
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu  # noqa: F401

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
_so_path = _KERNEL_BUILD / f"mha_gen_ext{_ext_suffix}"
if _so_path.is_file():
    ctypes.CDLL(str(_so_path), mode=ctypes.RTLD_GLOBAL)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                scaleValue: float = -1.0, is_causal: bool = False):
        if query.dtype == torch.float16:
            return torch.ops.npu.mha_gen_fp16(query, key, value,
                                              float(scaleValue), bool(is_causal))
        elif query.dtype == torch.bfloat16:
            return torch.ops.npu.mha_gen_bf16(query, key, value,
                                              float(scaleValue), bool(is_causal))
        else:
            raise ValueError(f"Unsupported dtype: {query.dtype}")


def get_init_inputs():
    return []
