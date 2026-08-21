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

"""Shared config and helpers for flash_attention_score_grad (FAG) designs.

对齐归档 kernel 的 arch22 S1s2Bn2gs1s2 通用模板路径（BNSD、keep_prob=1.0、
无 pse、sparse_mode=0、无 rope、ND 输出），设计参数与
kernel/op_host/fag_tiling_standalone.cpp 的 host tiling 逐一对应。
"""

from collections import namedtuple
from dataclasses import dataclass

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# ---- 与生产 tiling 对齐的常量（fag_tiling_standalone.cpp） ----
INITIAL_S1_SPLIT_NUM = 128       # S1 核内基本块初始值
INITIAL_S2_SPLIT_NUM = 64        # S2 核内基本块初始值
S1CV_RATIO_DEFAULT = 1           # S1 方向 CV 配比（cube:vector 核内块数比）
S2CV_RATIO_DEFAULT = 8           # S2 方向 CV 配比：CV 数据量 1:16 的组成项之一
CUBE_BLOCK = 128                 # Cube 基本块 128x128（fp16 输入 fp32 累加）
VEC_BLOCK_S1 = 8                 # Vector 基本块 S1 方向 8 行（fp32 32KB 预算）
C0_SIZE = 16
CORE_LIST_NUM = 50               # 核间切分表上限（blockStarts/blockEnds 数组长度）
SYS_WORKSPACE_RESERVED = 16 * 1024 * 1024  # dav-c220 RESERVED_WORKSPACE

# 归档合法性 gate 之外的模板路由阈值（生产多模板体系，供路由参考；
# 归档统一走泛化模板，性能影响见 design/README.md §5）
SMALL_SEQ_LIMIT = 1024           # S1s2Bn2 模板：S1<1024 && S2<1024
NGS1S2_BN_GSS_LIMIT = 32 * 1024  # Ngs1s2Bn 模板：G*S1*S2 <= 32KB
BNGS1S2_B_LIMIT = 64 * 128       # Bngs1s2B 模板：N2*G*S1*S2 <= 64*128


@dataclass(frozen=True)
class FagConfig:
    """FAG 问题维度（BNSD 布局）。

    q:  (b, n1, s1, d)     n1 = n2 * g（GQA 时 g > 1）
    k:  (b, n2, s2, d)
    v:  (b, n2, s2, value_d)
    dy: (b, n1, s1, value_d)
    softmax_max/sum: (b, n1, s1, 8) fp32（前向统计量，只用 [...,:1]）
    attention_in:    (b, n1, s1, value_d)
    atten_mask:      (1|b, 1|n1, s1, s2) bool，True 表示屏蔽；causal 用上三角表达
    输出: dq (b,n1,s1,d) / dk (b,n2,s2,d) / dv (b,n2,s2,value_d)
    """
    b: int
    n1: int          # query heads
    n2: int          # kv heads
    s1: int          # query seqlen
    s2: int          # kv seqlen
    d: int           # qk head dim
    value_d: int     # v head dim（允许 d != value_d）
    dtype: str = "float16"   # float16 / bfloat16 / float32
    has_mask: bool = False
    scale: float = 0.0       # 0 表示取 1/sqrt(d)

    @property
    def g(self):
        return self.n1 // self.n2


# 三段式阶段的 workspace 容器（保持 T.prim_func 签名精简）
PreWorkspaces = namedtuple("PreWorkspaces", ["ws_dq", "ws_dk", "ws_dv"])
MainWorkspaces = namedtuple("MainWorkspaces", ["ws_s", "ws_p", "ws_dp", "ws_ds"])
PipelineConfig = namedtuple("PipelineConfig", [
    "s1_cv_ratio", "s2_cv_ratio", "ring_slots", "s1_outer", "s2_outer",
])


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def round_up(value, alignment):
    return ceildiv(value, alignment) * alignment


def validate_config(cfg: FagConfig):
    """归档合法性 gate（与 model_new_ascendc.py / fag_tiling_standalone 一致）。"""
    if cfg.dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"unsupported dtype: {cfg.dtype}")
    if min(cfg.b, cfg.n1, cfg.n2, cfg.s1, cfg.s2, cfg.d, cfg.value_d) <= 0:
        raise ValueError("tensor extents must be positive")
    if cfg.n1 % cfg.n2 != 0:
        raise ValueError(f"GQA requires n1 % n2 == 0, got n1={cfg.n1}, n2={cfg.n2}")
    if cfg.d > 768 or cfg.value_d > 768:
        raise ValueError("head dim must be <= 768 (D_MAX_LIMIT)")
    # 归档固定路径约束：BNSD / keep_prob=1.0 / 无 pse / sparse_mode=0 / 无 rope
    # NZ 输出（mm12IsNZOut/mm345IsNZOut）在归档用例（s2%128==0 且 d 不在
    # {72,80,88,96}）下不命中，设计中保持 ND 语义。


def block_split(total_blocks, core_num):
    """核间切分（对应生产 FuzzyForBestSplit + DoBlockSplit 的语义）：

    BN2GS1 合并轴切成 total_blocks 个任务块，按 core_num 均分，
    返回 (block_outer, tail)——前 block_outer 块每核满配，尾部由最后一个核承担。
    生产实现输出 blockStarts/blockEnds[CORE_LIST_NUM] 表。
    """
    outer = ceildiv(total_blocks, core_num)
    tail = total_blocks - (core_num - 1) * outer
    return outer, tail


def cv_tile_params(cfg: FagConfig):
    """核内 CV 切分参数（对应生产 FuzzyForBestSplit 的 BNSD 子集）。

    C 侧基本块 (s1_c, s2_c) = (128, s2Inner)，V 侧基本块 (s1_v, s2_v)：
    S1 方向配比 s1_cv_ratio = s1_c / s1_v，S2 方向配比 s2_cv_ratio 默认 8，
    组成 CV 数据量 1:16（Cube 128x1024 fp16 vs Vector 8x1024 fp32 预算）。
    """
    s1_inner = min(INITIAL_S1_SPLIT_NUM, round_up(cfg.s1, C0_SIZE))
    s2_inner = min(INITIAL_S2_SPLIT_NUM, round_up(cfg.s2, C0_SIZE))
    return {
        "s1_inner": s1_inner,
        "s2_inner": s2_inner,
        "s1_cv_inner": s1_inner // S1CV_RATIO_DEFAULT,
        "s2_cv_inner": s2_inner * S2CV_RATIO_DEFAULT,
        "s1_cv_ratio": S1CV_RATIO_DEFAULT,
        "s2_cv_ratio": S2CV_RATIO_DEFAULT,
        "s1_outer": ceildiv(cfg.s1, s1_inner),
        "s2_outer": ceildiv(cfg.s2, s2_inner),
        # SFMG（FlashSoftmaxGrad）D 方向切分：sfmgd = (outer, inner, tail)
        "sfmg_d_inner": round_up(cfg.value_d, C0_SIZE),
        "sfmg_d_outer": 1,
    }


def output_sizes(cfg: FagConfig):
    """dq/dk/dv 元素数（Post 全量写回范围，at::empty 正确性依据）。"""
    return {
        "q_size": cfg.b * cfg.n1 * cfg.s1 * cfg.d,
        "kv_size": cfg.b * cfg.n2 * cfg.s2 * cfg.d,
        "v_size": cfg.b * cfg.n2 * cfg.s2 * cfg.value_d,
    }


def production_template_route(cfg: FagConfig, core_num):
    """生产多模板路由（参考用；归档统一返回 's1s2_bn2gs1s2'）。

    实测（docs/perf/）：小 shape 泛化模板 scalar 占比 75%+、核间不均最高 31.8%，
    生产对 S1s2Bn2 / SameAB 的路由可规避，是 vs 生产算子 0.91x 差距的主因。
    """
    if cfg.dtype != "float32" and cfg.s1 < SMALL_SEQ_LIMIT and cfg.s2 < SMALL_SEQ_LIMIT \
            and (cfg.b * cfg.n2 * 2 > core_num or cfg.g == 1):
        return "s1s2_bn2"        # 核间 B/N2，核内 G/S1/S2
    if cfg.g * cfg.s1 * cfg.s2 <= NGS1S2_BN_GSS_LIMIT and cfg.g == 1 and cfg.s2 < 1536:
        return "ngs1s2_bn"       # 核间 B/N2.o，核内 N2.i/G/S1/S2
    if cfg.n2 * cfg.g * cfg.s1 * cfg.s2 <= BNGS1S2_B_LIMIT:
        return "bngs1s2_b"       # 核间 B，核内 B.i/N2/G/S1/S2
    if cfg.s1 >= SMALL_SEQ_LIMIT or cfg.s2 >= SMALL_SEQ_LIMIT:
        return "s1s2_bn2gs1s2_sab"  # Cube 主核（V0/V1 复用左矩阵，L0C 累加）
    return "s1s2_bn2gs1s2"       # 泛化兜底
