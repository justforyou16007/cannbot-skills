#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# TileLang MHA 入口（ascendc 后端，cann-bench level4/mha，BSND 布局）。
#
# 设计：design/tile_level/mha.py（per-shape 版；G=1，ring 流水，
#   causal diffS 裁剪 + UB 现算 mask）。per-shape 编译：dim/bs/bg/block_m 为
#   编译期常量，buffer 精确分配；每个 distinct (shape, causal, dtype) 编译一次
#   （进程级缓存）。最终交付物为手写泛化 AscendC kernel（ascendc/mha.cpp，
#   运行期 shape + 运行期 buffer）。

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design.tile_level.mha import (  # noqa: E402
    GQAConfig,          # MHA 设计体内的 GQA 兼容配置（heads_kv=heads_q）
    gqa_fwd as tl_mha_fwd,
)

# (dtype, B, Nq, Nkv, S, Skv, D, causal) -> 已编译 kernel 的进程级缓存
_KERNEL_CACHE: dict = {}


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                scaleValue: float = -1.0, is_causal: bool = False):
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("Expected 4D BSND tensors")
        if not (query.dtype == key.dtype == value.dtype):
            raise ValueError("All input dtypes must match")

        dtype_str = str(query.dtype).split(".")[-1]
        if dtype_str not in ("float16", "bfloat16"):
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        batch, q_seq_len, heads_q, dim = query.shape
        kv_seq_len, heads_kv = key.shape[1], key.shape[2]

        if heads_kv != heads_q:
            raise ValueError(f"MHA requires N_kv == N_q, got {heads_kv} vs {heads_q}")
        if is_causal and q_seq_len > kv_seq_len:
            raise ValueError("is_causal requires S <= S_kv")

        ck = (dtype_str, batch, heads_q, heads_kv, q_seq_len, kv_seq_len, dim,
              bool(is_causal))
        kernel = _KERNEL_CACHE.get(ck)
        if kernel is None:
            cfg = GQAConfig(
                batch=batch, heads_q=heads_q, heads_kv=heads_kv,
                q_seq_len=q_seq_len, kv_seq_len=kv_seq_len, dim=dim,
                causal=bool(is_causal), dtype=dtype_str,
            )
            kernel = tl_mha_fwd(cfg)
            _KERNEL_CACHE[ck] = kernel

            # 首调收敛校验（tilelang 首调暂态垃圾：多模块进程首次调用可能全错，
            # 后续调用逐位一致）。新编译的 kernel 先用当前输入 warmup 一次并丢弃，
            # 正式调用返回收敛后的结果。
            sm_scale_w = float(scaleValue) if scaleValue > 0 else (1.0 / dim) ** 0.5
            kernel(query, key, value, sm_scale_w)  # warmup, discard
            torch.npu.synchronize()

        # sm_scale 作为 prim_func 标量参数在运行时传入；<=0 自动 1/sqrt(D)
        sm_scale = float(scaleValue) if scaleValue > 0 else (1.0 / dim) ** 0.5
        return kernel(query, key, value, sm_scale)


def get_init_inputs():
    return []
