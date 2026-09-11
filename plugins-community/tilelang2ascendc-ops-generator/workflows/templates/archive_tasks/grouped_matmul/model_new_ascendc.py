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

# GroupedMatmul AscendC 入口。加载 kernel/build/grouped_matmul_ext*.so，
# forward 调用注册算子 torch.ops.npu.grouped_matmul，按 split_item 包装输出。
# 禁止 torch 计算算子（split_item 的切片是张量视图，非计算）。

import ctypes
import glob
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

TASK_DIR = Path(__file__).resolve().parent


def _load_ext():
    pattern = str(TASK_DIR / "kernel" / "build" / "grouped_matmul_ext*.so")
    so_files = glob.glob(pattern)
    if not so_files:
        raise RuntimeError(
            f"grouped_matmul_ext .so not found: {pattern}（先跑 kernel/setup.py 构建）")
    ctypes.CDLL(so_files[0], mode=ctypes.RTLD_GLOBAL)


_load_ext()


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor = None, group_list=None,
                split_item: int = 0, transpose_weight: bool = False):
        if isinstance(group_list, torch.Tensor):
            # 先 .cpu()（一次同步 D2H）再转 list，避免在 NPU 上做 dtype cast
            # kernel（.to(torch.int64) 会在 device 端先起一个 cast kernel）。
            ends = [int(v) for v in group_list.cpu().tolist()]
            gl_arg = group_list
        else:
            ends = [int(v) for v in group_list]
            gl_arg = torch.tensor(ends, dtype=torch.int64, device=x.device)

        E = weight.shape[0]
        y = torch.ops.npu.grouped_matmul(
            x, weight, bias, gl_arg, int(split_item), bool(transpose_weight))

        if split_item in (0, 1):
            starts = [0] + ends[:-1]
            return [y[starts[g]:ends[g]] for g in range(E)]
        return [y]


def get_init_inputs():
    return []
