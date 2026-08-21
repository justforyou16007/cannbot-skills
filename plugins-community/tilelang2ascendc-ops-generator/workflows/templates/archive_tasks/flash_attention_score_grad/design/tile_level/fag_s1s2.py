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

"""Tile-level TileLang design for flash_attention_score_grad (S1s2Bn2gs1s2 泛化路径).

在 block_level/fag_s1s2.py 的阶段骨架上填充各级 buffer、搬运与计算。
对应生产 kernel/op_kernel/flash_attention_score_grad_{pre,s1s2_bn2gs1s2,post}.h。

主计算 6 步（设计文档 §1，keep_prob=1.0 / 无 pse 化简）：
  S  = Q @ K^T * scale           (mm1, Cube)
  P  = exp(S - softmax_max) / softmax_sum，mask 位置置零（Vector）
  dP = dy @ V^T                  (Cube)
  sfmg = rowsum(dy ∘ attention_in)（FlashSoftmaxGrad，Vector，D 方向 sfmgd 切分）
  dS = P ∘ (dP - sfmg) * scale   (Vector)
  dq += dS @ K ; dk += dS^T @ Q ; dv += P^T @ dy  (Cube，fp32 workspace 累加)
"""

from collections import namedtuple

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

from ..fag_common import (
    MainWorkspaces, PreWorkspaces, block_split, cv_tile_params, output_sizes,
    pass_configs, validate_config,
)

# ---- 各级 buffer 容器（G.FNM.03：helper 签名保持精简） ----
C1Bufs = namedtuple("C1Bufs", ["q_l1", "k_l1", "dy_l1", "v_l1",
                               "lhs_l0", "rhs_l0", "acc_s_l0c", "acc_dp_l0c"])
V1Bufs = namedtuple("V1Bufs", ["s_ub", "p_ub", "dp_ub", "ds_ub",
                               "sfmg_ub", "mask_ub", "stats_ub"])
C2Bufs = namedtuple("C2Bufs", ["ds_l1", "p_l1", "q_l1", "k_l1", "dy_l1",
                               "lhs_l0", "rhs_l0", "acc_dq_l0c", "acc_dkv_l0c"])
PostBufs = namedtuple("PostBufs", ["ws_ub", "out_ub"])

TaskContext = namedtuple("TaskContext", [
    "bz", "n2_idx", "g_idx", "s1o", "s2o", "cid", "vid",
])


# ---------------------------------------------------------------------------
#  PRE 阶段：fp32 workspace 清零（生产 FlashAttentionScoreGradPre，INIT_OUTPUT）
# ---------------------------------------------------------------------------

def _pre_zero_workspace(ws_dq, ws_dk, ws_dv, cid, contract):
    """按 preTilingData 的 PreBlockFactor/Total/Tail 清零本核区段。

    fp32 dtype 时 dv 不经过 workspace：Pre 直接清零 dvGm，Main 直写累加。
    """
    with T.Scope("v"):
        for buf in (ws_dq, ws_dk, ws_dv):
            T.tile.fill(buf, 0.0)
        # 生产实现按 qPreBlockTotal/kvPreBlockTotal/vPreBlockTotal 分核，
        # 每核 InitOutput(fp32)(gm[offset], size, 0)；此处以 fill 表达同语义。


# ---------------------------------------------------------------------------
#  MAIN 阶段：C1（mm1 + dp）/ V1（softmax+sfmg+ds）/ C2（dq/dk/dv 累加）
# ---------------------------------------------------------------------------

def _c1_compute_s_dp(ctx: TaskContext, cfg, cv, bufs: C1Bufs,
                     q, k, dy, v, workspaces: MainWorkspaces, scale):
    """Cube1：S = Q@K^T、dP = dy@V^T，fp32 结果写入 CV 交互 workspace。

    生产：mm1（NORM_DISABLE_INIT，L0C 输出）；dp 与 S 同一 Cube 窗口内双发
    （CV preload，设计文档 §4.2）。
    """
    s1_b = ctx.s1o * cv["s1_inner"]
    s2_b = ctx.s2o * cv["s2_inner"]
    slot = ctx.s2o % 2
    with T.Scope("C"):
        # load q[bz, n2*g+g_idx, s1 条带] 与 k/v/dy 的 s2 tile 到 L1（NZ 布局）
        T.copy(q[ctx.bz, ctx.n2_idx * cfg.g + ctx.g_idx,
                 s1_b:s1_b + cv["s1_inner"], :], bufs.q_l1)
        T.copy(k[ctx.bz, ctx.n2_idx, s2_b:s2_b + cv["s2_inner"], :], bufs.k_l1)
        T.copy(dy[ctx.bz, ctx.n2_idx * cfg.g + ctx.g_idx,
                  s1_b:s1_b + cv["s1_inner"], :], bufs.dy_l1)
        T.copy(v[ctx.bz, ctx.n2_idx, s2_b:s2_b + cv["s2_inner"], :], bufs.v_l1)
        # SplitA/SplitB：L1 -> L0A/L0B
        T.copy(bufs.q_l1, bufs.lhs_l0)
        T.copy(bufs.k_l1, bufs.rhs_l0)
        # mm1: S = Q @ K^T（rhs 转置由 mma 的 transpose 标志消化，不物化）
        T.mma(bufs.lhs_l0, bufs.rhs_l0, bufs.acc_s_l0c, init=True)
        # L0C(fp32) -> GM workspace（CV 交互通路）
        T.copy(bufs.acc_s_l0c, workspaces.ws_s[ctx.cid, slot])
        # dp = dy @ V^T（同窗口第二发，覆盖 V1 的 Vector 计算时长）
        T.copy(bufs.dy_l1, bufs.lhs_l0)
        T.copy(bufs.v_l1, bufs.rhs_l0)
        T.mma(bufs.lhs_l0, bufs.rhs_l0, bufs.acc_dp_l0c, init=True)
        T.copy(bufs.acc_dp_l0c, workspaces.ws_dp[ctx.cid, slot])
        T.set_cross_flag("FIX", slot * 2)  # 通知 V1：S/dP 就绪


def _v1_softmax_sfmg_ds(ctx: TaskContext, cfg, cv, bufs: V1Bufs,
                        softmax_max, softmax_sum, attention_in, dy,
                        atten_mask, workspaces: MainWorkspaces, scale):
    """Vector1：P 重建、FlashSoftmaxGrad、dS 合成。

    - P = exp(S - softmax_max) / softmax_sum（前向统计量，不再重算 max/sum）；
      atten_mask（True=屏蔽）位置显式置零；causal 由上三角 mask 表达。
    - sfmg = rowsum(dy ∘ attention_in)，D 方向按 sfmgd(outer/inner/tail) 切分，
      对应生产 CalcSoftMaxGrad / CalcSoftMaxGradPingPong。
    - ds = P ∘ (dp - sfmg) * scale。
    """
    slot = ctx.s2o % 2
    s1_b = ctx.s1o * cv["s1_inner"]
    s2_b = ctx.s2o * cv["s2_inner"]
    with T.Scope("v"):
        T.wait_cross_flag(slot * 2)
        # S/dP GM -> UB（V 侧基本块 s1_cv_inner x s2_cv_inner）
        T.copy(workspaces.ws_s[ctx.cid, slot], bufs.s_ub)
        T.copy(workspaces.ws_dp[ctx.cid, slot], bufs.dp_ub)
        # 前向统计量（每行取 [...,:1]）：softmax_max/sum
        T.copy(softmax_max[ctx.bz, ctx.n2_idx * cfg.g + ctx.g_idx,
                           s1_b:s1_b + cv["s1_inner"], 0], bufs.stats_ub)
        # P = exp(S - max) / sum
        T.tile.sub(bufs.p_ub, bufs.s_ub, bufs.stats_ub)
        T.tile.exp(bufs.p_ub, bufs.p_ub)
        T.copy(softmax_sum[ctx.bz, ctx.n2_idx * cfg.g + ctx.g_idx,
                           s1_b:s1_b + cv["s1_inner"], 0], bufs.stats_ub)
        T.tile.div(bufs.p_ub, bufs.p_ub, bufs.stats_ub)
        if cfg.has_mask:
            # atten_mask bool（True=屏蔽）-> 选择性置零（生产 CalcAttenMaskBool）
            T.copy(atten_mask[0, 0, s1_b:s1_b + cv["s1_inner"],
                              s2_b:s2_b + cv["s2_inner"]], bufs.mask_ub)
            T.tile.mul(bufs.p_ub, bufs.p_ub, bufs.mask_ub)  # 取反掩码在 tile 层展开
        # sfmg = rowsum(dy ∘ attention_in)（D 方向 sfmgd 切分，tail 对齐）
        T.copy(dy[ctx.bz, ctx.n2_idx * cfg.g + ctx.g_idx,
                  s1_b:s1_b + cv["s1_inner"], :], bufs.s_ub)
        T.tile.mul(bufs.sfmg_ub, bufs.s_ub, attention_in[
            ctx.bz, ctx.n2_idx * cfg.g + ctx.g_idx, s1_b:s1_b + cv["s1_inner"], :])
        # rowsum 归约（生产 SoftmaxGradFront 高阶 API）
        # ds = P ∘ (dp - sfmg) * scale
        T.tile.sub(bufs.ds_ub, bufs.dp_ub, bufs.sfmg_ub)
        T.tile.mul(bufs.ds_ub, bufs.ds_ub, bufs.p_ub)
        T.tile.mul(bufs.ds_ub, bufs.ds_ub, scale)
        # P/dS 写回 CV workspace，通知 C2
        T.copy(bufs.p_ub, workspaces.ws_p[ctx.cid, slot])
        T.copy(bufs.ds_ub, workspaces.ws_ds[ctx.cid, slot])
        T.set_cross_flag("MTE3", slot * 2 + 1)


def _c2_accumulate_dqkv(ctx: TaskContext, cfg, cv, bufs: C2Bufs,
                        q, k, dy, workspaces: MainWorkspaces,
                        ws_dq, ws_dk, ws_dv):
    """Cube2：dq += dS@K；dk += dS^T@Q；dv += P^T@dy（fp32 workspace 累加）。

    生产：mm3/mm4 两个 matmul 对象（init=False 持续累加）；
    转置（dS^T、P^T）由 mma transpose 标志消化。GQA 时多个 (g_idx, s1o)
    任务对同一 (bz, n2_idx) 的 dk/dv 区段累加（多核经 workspace 归约）。
    """
    slot = ctx.s2o % 2
    with T.Scope("C"):
        T.wait_cross_flag(slot * 2 + 1)
        # dS/P GM -> L1 -> L0
        T.copy(workspaces.ws_ds[ctx.cid, slot], bufs.ds_l1)
        T.copy(workspaces.ws_p[ctx.cid, slot], bufs.p_l1)
        # dq += dS @ K
        T.copy(bufs.ds_l1, bufs.lhs_l0)
        T.copy(bufs.k_l1, bufs.rhs_l0)
        T.mma(bufs.lhs_l0, bufs.rhs_l0, bufs.acc_dq_l0c, init=(ctx.s2o == 0))
        # dk += dS^T @ Q（lhs 转置）
        T.copy(bufs.q_l1, bufs.rhs_l0)
        T.mma(bufs.ds_l1, bufs.rhs_l0, bufs.acc_dkv_l0c, init=(ctx.s2o == 0))
        # dv += P^T @ dy
        T.copy(bufs.dy_l1, bufs.rhs_l0)
        T.mma(bufs.p_l1, bufs.rhs_l0, bufs.acc_dkv_l0c, init=(ctx.s2o == 0))
        # L0C -> fp32 workspace（s2o 方向全部完成后由 Post 统一写回）
        T.copy(bufs.acc_dq_l0c, ws_dq[ctx.cid])
        T.copy(bufs.acc_dkv_l0c, ws_dk[ctx.cid])


# ---------------------------------------------------------------------------
#  POST 阶段：fp32 workspace -> 输出 dtype，Muls(scale) + 全量写回
# ---------------------------------------------------------------------------

def _post_writeback(bufs: PostBufs, ws_gm, out_gm, begin, base_num, tail_num,
                    block_total, scale, cast_to):
    """生产 FlashAttentionScoreGradPost 的 ND 路径（单输出张量）。

    覆盖性契约：begin..block_total 恰好铺满 [0, numel)，是 host 侧
    at::empty 替代 at::zeros 的正确性依据（fp32 dv 例外：Pre 清零 dvGm +
    Main 直写，Post 不参与）。
    """
    with T.Scope("v"):
        i = begin
        while i < block_total:
            data_size = T.min(base_num, block_total - i)
            T.copy(ws_gm[i:i + data_size], bufs.ws_ub)
            T.tile.mul(bufs.ws_ub, bufs.ws_ub, scale)
            if cast_to != "float32":
                T.copy(bufs.ws_ub, bufs.out_ub)  # fp32 -> fp16/bf16（CAST_ROUND）
                T.copy(bufs.out_ub, out_gm[i:i + data_size])
            else:
                T.copy(bufs.ws_ub, out_gm[i:i + data_size])
            i += base_num


@tilelang.jit(out_idx=[8, 9, 10], workspace_idx=[11], pass_configs=pass_configs)
def fag_s1s2_tile(cfg, core_num, post_base_num=8192):
    """S1s2Bn2gs1s2 泛化路径完整 tile-level 设计（Pre/Main/Post 单入口）。"""
    validate_config(cfg)
    cv = cv_tile_params(cfg)
    sizes = output_sizes(cfg)
    total_blocks = cfg.b * cfg.n2 * cfg.g * cv["s1_outer"]
    cores = min(core_num, total_blocks)
    block_outer, _ = block_split(total_blocks, cores)
    scale = cfg.scale if cfg.scale else 1.0 / (cfg.d ** 0.5)

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
        with T.Kernel(cores, is_npu=True) as (cid, vid):
            # Expert buffers：一个 CV 基本块在各存储级的驻留
            c1 = C1Bufs(
                q_l1=T.alloc_L1((cv["s1_inner"], cfg.d), cfg.dtype),
                k_l1=T.alloc_L1((cv["s2_inner"], cfg.d), cfg.dtype),
                dy_l1=T.alloc_L1((cv["s1_inner"], cfg.value_d), cfg.dtype),
                v_l1=T.alloc_L1((cv["s2_inner"], cfg.value_d), cfg.dtype),
                lhs_l0=T.alloc_L0A((cv["s1_inner"], cfg.d), cfg.dtype),
                rhs_l0=T.alloc_L0B((cfg.d, cv["s2_inner"]), cfg.dtype),
                acc_s_l0c=T.alloc_L0C((cv["s1_inner"], cv["s2_inner"]), "float32"),
                acc_dp_l0c=T.alloc_L0C((cv["s1_inner"], cv["s2_inner"]), "float32"),
            )
            T.annotate_layout({c1.q_l1: make_zn_layout(c1.q_l1),
                               c1.k_l1: make_zn_layout(c1.k_l1)})
            v1 = V1Bufs(
                s_ub=T.alloc_ub((cv["s1_cv_inner"], cv["s2_cv_inner"]), "float32"),
                p_ub=T.alloc_ub((cv["s1_cv_inner"], cv["s2_cv_inner"]), "float32"),
                dp_ub=T.alloc_ub((cv["s1_cv_inner"], cv["s2_cv_inner"]), "float32"),
                ds_ub=T.alloc_ub((cv["s1_cv_inner"], cv["s2_cv_inner"]), "float32"),
                sfmg_ub=T.alloc_ub((cv["s1_cv_inner"], cv["s2_cv_inner"]), "float32"),
                mask_ub=T.alloc_ub((cv["s1_cv_inner"], cv["s2_cv_inner"]), "bool"),
                stats_ub=T.alloc_ub((cv["s1_inner"],), "float32"),
            )
            post = PostBufs(
                ws_ub=T.alloc_ub((post_base_num,), "float32"),
                out_ub=T.alloc_ub((post_base_num,), cfg.dtype),
            )
            pre_ws = PreWorkspaces(
                workspaces.ws_s, workspaces.ws_p, workspaces.ws_dp)  # 命名复用

            # ---- PRE ----
            _pre_zero_workspace(pre_ws.ws_dq, pre_ws.ws_dk, pre_ws.ws_dv,
                                cid, None)

            # ---- MAIN：本核任务块 × s2o ping-pong（SubGrapA/B 展开）----
            for task in T.serial(block_outer):
                # 任务解码：(bz, n2_idx, g_idx, s1o)
                blk = cid * block_outer + task
                s1o = blk % cv["s1_outer"]
                g_idx = blk // cv["s1_outer"] % cfg.g
                n2_idx = blk // cv["s1_outer"] // cfg.g % cfg.n2
                bz = blk // cv["s1_outer"] // cfg.g // cfg.n2
                for s2o in T.serial(cv["s2_outer"]):
                    ctx = TaskContext(bz, n2_idx, g_idx, s1o, s2o, cid, vid)
                    _c1_compute_s_dp(ctx, cfg, cv, c1, q, k, dy, v,
                                     workspaces, scale)
                    _v1_softmax_sfmg_ds(ctx, cfg, cv, v1, softmax_max,
                                        softmax_sum, attention_in, dy,
                                        atten_mask, workspaces, scale)
                    _c2_accumulate_dqkv(ctx, cfg, cv, None, q, k, dy,
                                        workspaces, pre_ws.ws_dq,
                                        pre_ws.ws_dk, pre_ws.ws_dv)

            # ---- POST：按 numel 全量写回（ND 路径，CAST_DV 语义）----
            q_total = sizes["q_size"]
            kv_total = sizes["kv_size"]
            v_total = sizes["v_size"]
            per_core = T.ceildiv(q_total, cores)
            _post_writeback(post, pre_ws.ws_dq, dq,
                            cid * per_core, post_base_num, 0,
                            T.min((cid + 1) * per_core, q_total), scale,
                            cfg.dtype)
            _post_writeback(post, pre_ws.ws_dk, dk,
                            cid * per_core, post_base_num, 0,
                            T.min((cid + 1) * per_core, kv_total), scale,
                            cfg.dtype)
            if cfg.dtype != "float32":
                # CAST_DV：fp16/bf16 由 Post cast 写回；fp32 由 Pre+Main 直写
                _post_writeback(post, pre_ws.ws_dv, dv,
                                cid * per_core, post_base_num, 0,
                                T.min((cid + 1) * per_core, v_total), 1.0,
                                cfg.dtype)

    return main


fag_s1s2_bn2gs1s2_tile = fag_s1s2_tile

__all__ = ["fag_s1s2_tile", "fag_s1s2_bn2gs1s2_tile"]
