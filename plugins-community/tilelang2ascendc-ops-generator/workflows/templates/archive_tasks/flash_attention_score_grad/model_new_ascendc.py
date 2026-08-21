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

import glob
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_LIB_PATTERN = str(_KERNEL_BUILD / "flash_attention_score_grad_ext*")

# 优先 whl 导入（触发 register.cpp 中的 TORCH_LIBRARY 注册）
try:
    import flash_attention_score_grad_ext  # noqa: F401
except ImportError:
    # 回退：直接加载 kernel/build/ 下的 .so
    _libs = glob.glob(_LIB_PATTERN)
    if _libs:
        torch.ops.load_library(_libs[0])


class ModelNew(nn.Module):
    """通过 torch.ops.npu.flash_attention_score_grad 调用 kernel/ 下独立构建的
    FlashAttentionScoreGrad AscendC kernel（arch22 S1s2Bn2gs1s2 通用路径 re-host，
    BNSD 布局，keep_prob=1.0，无 pse，sparse_mode=0）。

    需先执行 kernel/build.sh 完成构建（或 pip install kernel/dist/*.whl）；
    接口与 model.py 的 Model.forward 完全一致，输出为 (dq, dk, dv)。
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dy: torch.Tensor,
        softmax_max: torch.Tensor,
        softmax_sum: torch.Tensor,
        attention_in: torch.Tensor,
        atten_mask: torch.Tensor = None,
        scale: float = None,
    ):
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or dy.ndim != 4:
            raise ValueError(
                f"Expected 4D BNSD tensors, got q.ndim={q.ndim}, k.ndim={k.ndim}, "
                f"v.ndim={v.ndim}, dy.ndim={dy.ndim}"
            )
        if not (q.dtype == k.dtype == v.dtype == dy.dtype):
            raise ValueError(
                f"All input dtypes must match, got q={q.dtype}, k={k.dtype}, "
                f"v={v.dtype}, dy={dy.dtype}"
            )

        batch, n_heads_q, seq_q, d_qk = q.shape
        seq_kv = k.shape[2]
        if scale is None:
            scale = 1.0 / math.sqrt(d_qk)

        device = torch.device("npu")
        q_n = q.to(device)
        k_n = k.to(device)
        v_n = v.to(device)
        dy_n = dy.to(device)
        softmax_max_n = softmax_max.float().to(device)
        softmax_sum_n = softmax_sum.float().to(device)
        attention_in_n = attention_in.to(device)
        atten_mask_n = atten_mask.bool().to(device) if atten_mask is not None else None

        torch.npu.manual_seed(0)
        dq, dk, dv = torch.ops.npu.flash_attention_score_grad(
            q_n,
            k_n,
            v_n,
            dy_n,
            softmax_max_n,
            softmax_sum_n,
            attention_in_n,
            atten_mask_n,
            scale,
        )
        if q.device.type == "npu":
            # 输入已在 NPU：直接返回 NPU 张量（与 model.py 输出设备跟随输入的语义一致），
            # 避免无谓的 D2H+H2D 往返（profiling 场景实测每迭代可省 ~450us）
            torch.npu.synchronize()
            return dq, dk, dv
        # 输入在 CPU：pinned buffer + non_blocking 批量 D2H
        # （pageable .cpu() 每路约 3 倍耗时； pinned 实测 104us→31us/0.5MB）
        dq_c = torch.empty(dq.shape, dtype=dq.dtype, pin_memory=True)
        dk_c = torch.empty(dk.shape, dtype=dk.dtype, pin_memory=True)
        dv_c = torch.empty(dv.shape, dtype=dv.dtype, pin_memory=True)
        dq_c.copy_(dq, non_blocking=True)
        dk_c.copy_(dk, non_blocking=True)
        dv_c.copy_(dv, non_blocking=True)
        torch.npu.synchronize()
        return dq_c, dk_c, dv_c


def get_init_inputs():
    return []
