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

"""Block-level TileLang design for flash_attention_score_grad (S1s2Bn2gs1s2 泛化路径).

捕获任务分解、Pre/Main/Post 阶段边界、CV 双子图（SubGrapA/SubGrapB）的
ring-buffer 归属与跨 Scope 同步。细粒度数学留给 tile_level/fag_s1s2.py。

与生产 kernel 的对应关系：
  FlashAttentionScoreGradPre::Process        -> pre 阶段（INIT_OUTPUT）
  FlashAttentionScoreGradS1s2Bn2gs1s2::Process -> main 阶段（SubGrapA/B ping-pong）
  FlashAttentionScoreGradPost::Process       -> post 阶段（cast+scale 全量写回）
"""

import tilelang
import tilelang.language as T

from ..fag_common import (
    CORE_LIST_NUM, MainWorkspaces, PipelineConfig, PreWorkspaces,
    block_split, cv_tile_params, output_sizes, pass_configs, validate_config,
)


def fag_s1s2_block_contract(cfg, core_num):
    """返回完整的 block 域与 producer/consumer 契约。"""
    validate_config(cfg)
    cv = cv_tile_params(cfg)
    sizes = output_sizes(cfg)
    # 核间：BN2GS1 合并轴任务块（每块 = 一个 (b,n2,g,s1o) 的 S1 条带）
    total_blocks = cfg.b * cfg.n2 * cfg.g * cv["s1_outer"]
    cores = min(core_num, total_blocks, CORE_LIST_NUM)
    block_outer, block_tail = block_split(total_blocks, cores)
    return {
        "layout": "BNSD (mock_layout=0)，keep_prob=1.0，无 pse，sparse_mode=0",
        "total_blocks": total_blocks,
        "cores": cores,
        "block_outer": block_outer,
        "block_tail": block_tail,
        "cv": cv,
        "sizes": sizes,
        # CV 配比：C 侧基本块 (128, s2_inner)，V 侧 (s1_cv_inner, s2_cv_inner)，
        # 数据量比 1:16（设计文档 §3.1：cube 128x1024 fp16 / vector 8x1024 fp32）
        "cv_data_ratio": "1:16 (S1 nRatio=%d, S2 ratio=%d)" % (
            cv["s1_cv_ratio"], cv["s2_cv_ratio"]),
        "dataflow": {
            "pre": "AIV 清零 fp32 workspace（dq/dk/dv 累积区；fp32 时直接清 dvGm）",
            "main": "AIC mm1(S=Q@K^T) -> AIV(softmax P / sfmg / ds) -> AIC mm3/mm4(dq/dk/dv 累加)",
            "post": "AIV 读 fp32 workspace -> Muls(scale) -> Cast -> 按 numel 全量写回 dq/dk/dv",
        },
        "consumer_contract": "host 侧 dq/dk/dv 可 at::empty（Pre+Main+Post 保证全量覆写）",
    }


# ---------------------------------------------------------------------------
#  循环体 helper —— main 阶段每个 (s1o, s2o) tile 的 CV 双子图
#  对应生产 SubGrapA / SubGrapB：同一片 S1 条带的相邻两个 S2 tile 以
#  ping-pong 方式复用 eventIdList，实现 CV preload（Cube 双发，设计文档 §4.2）
# ---------------------------------------------------------------------------

def _sub_graph_a(t, slot, cid, vid, cfg: PipelineConfig):
    """SubGrapA：Cube1(t) -> Vector1(t) -> Cube2(t) 的 ping 通道。"""
    with T.Scope("C"):
        # TODO(tile-level, C1):
        # - load q/k tile，mm1: S = Q @ K^T（本 s1o 条带 × 第 t 个 s2 tile）
        # - S 写入 ws_s[cid, slot]（fp32），置 FIX flag 通知 V1
        T.set_cross_flag("FIX", 0)

    with T.Scope("v"):
        # TODO(tile-level, V1):
        # - 等 C1(t)；P = SimpleSoftmax(S, atten_mask)（复用前向 softmax_max/sum）
        # - sfmg = FlashSoftmaxGrad(dy, attention_in)（D 方向 sfmgd 切分）
        # - ds = P ∘ (dp - sfmg) * scale，写入 ws_ds[cid, slot]，通知 C2
        T.wait_cross_flag(0)
        T.set_cross_flag("MTE3", 1)

    with T.Scope("C"):
        # TODO(tile-level, C2):
        # - 等 V1(t)；mm3: dq += ds @ K；mm4: dk += ds^T @ Q；dv += P^T @ dy
        # - GQA：dk/dv 按 n2 头累加（g 组内归约经 workspace 多核/多块累加）
        T.wait_cross_flag(1)


def _sub_graph_b(t, slot, cid, vid, cfg: PipelineConfig):
    """SubGrapB：与 SubGrapA 结构相同的 pong 通道（eventIdList 独立）。"""
    with T.Scope("C"):
        # TODO(tile-level, C1'): 同 SubGrapA 的 C1，写 ws_s[cid, slot]
        T.set_cross_flag("FIX", 2)

    with T.Scope("v"):
        # TODO(tile-level, V1'): 同 SubGrapA 的 V1，写 ws_ds[cid, slot]
        T.wait_cross_flag(2)
        T.set_cross_flag("MTE3", 3)

    with T.Scope("C"):
        # TODO(tile-level, C2'): 同 SubGrapA 的 C2（累加同一组 dq/dk/dv workspace）
        T.wait_cross_flag(3)


@tilelang.jit(out_idx=[8, 9, 10], workspace_idx=[11], pass_configs=pass_configs)
def fag_s1s2_block(cfg, core_num):
    """三段式 block 骨架：pre（清零）-> main（CV ping-pong）-> post（写回）。"""
    contract = fag_s1s2_block_contract(cfg, core_num)
    cv = contract["cv"]
    pipeline_cfg = PipelineConfig(
        cv["s1_cv_ratio"], cv["s2_cv_ratio"], 2, cv["s1_outer"], cv["s2_outer"],
    )

    @T.prim_func
    def main(
        q: T.Tensor((cfg.b, cfg.n1, cfg.s1, cfg.d), cfg.dtype),
        k: T.Tensor((cfg.b, cfg.n2, cfg.s2, cfg.d), cfg.dtype),
        v: T.Tensor((cfg.b, cfg.n2, cfg.s2, cfg.value_d), cfg.dtype),
        dy: T.Tensor((cfg.b, cfg.n1, cfg.s1, cfg.value_d), cfg.dtype),
        softmax_max: T.Tensor((cfg.b, cfg.n1, cfg.s1, 8), "float32"),
        softmax_sum: T.Tensor((cfg.b, cfg.n1, cfg.s1, 8), "float32"),
        attention_in: T.Tensor((cfg.b, cfg.n1, cfg.s1, cfg.value_d), cfg.dtype),
        atten_mask: T.Tensor((1, 1, cfg.s1, cfg.s2), "bool"),
        dq: T.Tensor((cfg.b, cfg.n1, cfg.s1, cfg.d), cfg.dtype),
        dk: T.Tensor((cfg.b, cfg.n2, cfg.s2, cfg.d), cfg.dtype),
        dv: T.Tensor((cfg.b, cfg.n2, cfg.s2, cfg.value_d), cfg.dtype),
        workspaces: MainWorkspaces,
    ):
        ws_s = workspaces.ws_s      # mm1 输出 S（fp32，CV 交互）
        ws_p = workspaces.ws_p      # softmax 结果 P
        ws_dp = workspaces.ws_dp    # dp = dy @ V^T（fp32）
        ws_ds = workspaces.ws_ds    # ds = P ∘ (dp - sfmg) * scale

        with T.Kernel(contract["cores"], is_npu=True) as (cid, vid):
            # ---- pre 阶段（FlashAttentionScoreGradPre，INIT_OUTPUT=true）----
            with T.Scope("v"):
                # TODO(tile-level, PRE):
                # - 按 preTilingData 的 q/kv/v PreBlockFactor 把本核负责的
                #   dq/dk/dv fp32 workspace 区段清零（fp32 dtype 时 dv 直接清 dvGm）
                pass

            # ---- main 阶段（S1s2Bn2gs1s2::Process）----
            # 本核任务块：cid * block_outer .. +block_outer（尾核到 block_tail）
            # 每块 = (b, n2, g, s1o) 条带；块内 s2o 方向 ping-pong 展开 SubGrapA/B
            for task in T.serial(contract["block_outer"]):
                # TODO(tile-level): 由 task 解码 (bz, n2_idx, g_idx, s1o)
                for s2o in T.serial(cv["s2_outer"]):
                    if s2o % 2 == 0:
                        _sub_graph_a(s2o, s2o % 2, cid, vid, pipeline_cfg)
                    else:
                        _sub_graph_b(s2o, s2o % 2, cid, vid, pipeline_cfg)

            # ---- post 阶段（FlashAttentionScoreGradPost，ND 路径）----
            with T.Scope("v"):
                # TODO(tile-level, POST):
                # - 按 postTilingData 的 q/kv/v PostBlockFactor/BaseNum/TailNum
                #   读 fp32 workspace，Muls(scaleValue)（dq/dk）+ Cast 到输出 dtype，
                #   按 [0, qSize)/[0, kvSize)/[0, vSize) 全量写回 dq/dk/dv GM
                #   （fp32 dtype：dv 已由 pre+main 直写，post 只处理 dq/dk）
                pass

    return main


fag_s1s2_bn2gs1s2_block = fag_s1s2_block

__all__ = [
    "fag_s1s2_block_contract",
    "fag_s1s2_block",
    "fag_s1s2_bn2gs1s2_block",
]
