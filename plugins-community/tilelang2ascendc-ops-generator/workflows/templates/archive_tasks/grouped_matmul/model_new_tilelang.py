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

# TileLang GroupedMatmul 入口（ascendc 后端）。设计：design/tile_level/grouped_matmul.py
# （per-shape 编译；最终交付物为手写泛化 AscendC kernel，运行期 shape + group_list）。
#
# 本文件只做：输入布局整理 + block_metadata 预计算（张量创建/变换）+ kernel 调用
# + split_item 输出包装（torch view）。禁止 torch 计算算子。

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design.tile_level.grouped_matmul import (
    GroupedMatmulConfig,
    _tile_cfg,
    grouped_matmul_fwd as tl_gmm_fwd,
)

# (dtype, M, N, K, E, has_bias, tw, total_m_blocks, metadata_tuple) -> 已编译 kernel
_KERNEL_CACHE: dict = {}


def _build_metadata(ends, block_M, n_num, device):
    """host 预计算 block_metadata[total_m_blocks, 3] = [g, m_start, valid_m]。

    空组（m_g=0）不产 block。valid_m 记录该 m 砖的有效行数（尾块 < block_M）。
    返回 (metadata_tensor, total_m_blocks)。
    """
    starts = [0] + ends[:-1]
    rows = []
    for g in range(len(ends)):
        s, e = starts[g], ends[g]
        m_g = e - s
        if m_g <= 0:
            continue  # 空组跳过
        nblk = (m_g + block_M - 1) // block_M
        for b in range(nblk):
            m_start = s + b * block_M
            valid_m = min(block_M, e - m_start)
            rows.append((g, m_start, valid_m))
    total_m_blocks = len(rows)
    meta = torch.tensor(rows, dtype=torch.int32, device=device)
    return meta, total_m_blocks


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor = None, group_list=None,
                split_item: int = 0, transpose_weight: bool = False):
        M, K = x.shape
        E = weight.shape[0]
        N = weight.shape[1] if transpose_weight else weight.shape[2]
        dtype_str = str(x.dtype).split(".")[-1]
        if dtype_str not in ("float16", "bfloat16", "float32"):
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        if isinstance(group_list, torch.Tensor):
            ends = [int(v) for v in group_list.to(torch.int64).tolist()]
        else:
            ends = [int(v) for v in group_list]

        block_M, block_N, _, _ = _tile_cfg(dtype_str)
        has_bias = bias is not None

        # N 尾块在 kernel 内处理（P0：消除 host N-padding）。weight 按原始 N
        # 读 GM（B 分形装载 realTailN 钳位 + 尾列零填充，fixpipe 写出 N 钳位），
        # 不再 host pad 到 block_N。输出 Y 直接按原始 N 分配，无切片。依据
        # GENERALIZE_DEBUG_LOG ISSUE-8：编译器修复 copy_gm_to_l1_linear
        # （commit e50437fc）后，B/bias 的 N 尾块在 kernel 内正确处理。
        n_num = (N + block_N - 1) // block_N

        meta, total_m_blocks = _build_metadata(ends, block_M, n_num, x.device)

        # bias：fp32，按原始 N（行 stride=N，与 kernel 的 a.N 一致）。kernel 内
        # bias 的 GM→L1 非对齐尾块从对齐边界读 ALIGN 列，GM 终点越界 <ALIGN 列
        # （<=7 列 / 28B，NPU GM 越界读不触发硬件异常，已对照验证；越界读的垃圾
        # 写入 L1 尾列，fixpipe 写出时 N 钳位丢弃，不进有效输出）。
        # 缺省传零占位（has_bias=False 不走 bias 路径，内容不被读）。
        if has_bias:
            bias_arg = bias if bias.dtype == torch.float32 else bias.float()
        else:
            bias_arg = torch.zeros((E, N), dtype=torch.float32,
                                   device=x.device)

        key = (dtype_str, M, N, K, E, has_bias, transpose_weight, total_m_blocks)
        kernel = _KERNEL_CACHE.get(key)
        if kernel is None:
            cfg = GroupedMatmulConfig(
                m=M, n=N, k=K, e=E, dtype=dtype_str, has_bias=has_bias,
                transpose_weight=transpose_weight, total_m_blocks=total_m_blocks)
            kernel = tl_gmm_fwd(cfg)
            _KERNEL_CACHE[key] = kernel

        y = kernel(x, weight, bias_arg, meta)

        if split_item in (0, 1):
            starts = [0] + ends[:-1]
            return [y[starts[g]:ends[g]] for g in range(E)]
        return [y]


def get_init_inputs():
    return []
