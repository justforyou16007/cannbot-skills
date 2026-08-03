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

"""Block-level 设计：Stage3 — chunk 内贡献累加（独立 jit kernel，对应 tile-level `stage3_ker`）。

对应 AscendC `Stage3`（stage3.h）。消费 Stage1 的 qkt/gCumExp/vInner 与 Stage2 写好的
out（attn_inter 部分），把 chunk 内贡献累加进 out。**不依赖状态，无串行**。
满足 `BlockLevelDesign-quality-checklist.md`。

## C/V 子阶段（checklist #4）

    V4  attn_c = mask(qkt · decay) · scale    （AIV 生成 masked qkt，写 tmpGM）
    C4  out += attn_c @ v_new                 （cube，T.gemm_v0，累加到 out）

## workspace 读/写（checklist #6）

    读：ws_qkt / ws_gcumexp / ws_vinner（含 Stage2 的 v_prime 贡献，即 v_new）
    写：out（累加 attn_c@v_new）

## 块内遍历（checklist #2）

    with T.Kernel(used_core_num):
        for unit in 分给本 kernel 的 (b, head) 集合:
            if b == 当前 batch 且 属于本 chunk:
                # 每单元：V4 -> C4
"""

import tilelang
import tilelang.language as T

from design.chunk_gated_delta_rule_common import CHUNK_SIZE, pass_configs


@tilelang.jit(out_idx=[], workspace_idx=[], pass_configs=pass_configs)
def stage3_ker(batch, seq_len, nk, nv, dk, dv, chunk_size=CHUNK_SIZE, scale=1.0):
    C = chunk_size
    total_units = batch * nv
    used_core_num = min(20, total_units)

    @T.prim_func
    def main(
        ws_qkt: T.Tensor((C, C), "bfloat16"),
        ws_gcumexp: T.Tensor((C,), "float32"),
        ws_vinner: T.Tensor((C, dv), "bfloat16"),
        ci: T.Tensor((1,), "int32"),
        bb: T.Tensor((1,), "int32"),
        Lb: T.Tensor((1,), "int32"),
        out: T.Tensor((seq_len, nv, dv), "bfloat16"),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            # TODO(tile-level): unit 分配；`h=unit%nv, s0=base[b]+ci[0]*C`
            #                   `valid=T.if_then_else(...)`；`if b==bb[0]:`

            # V4：attn_c = mask(qkt·decay)·scale
            with T.Scope("V"):
                # TODO(tile-level):
                #   attn_c[i][j] = ws_qkt[i][j]*g_exp[i]*g_exp_neg[j]*scale (j<=i，因果含对角)
                #   T.copy(attn_c, tmpGM)  # 供 C4 cube 读
                pass

            # C4：out += attn_c @ v_new（累加到 Stage2 写的 out）
            with T.Scope("C"):
                # TODO(tile-level):
                #   T.gemm_v0(attn_c, v_new, out_acc)   # v_new = ws_vinner（含 v_prime 贡献）
                #   T.copy 降型(out_acc) -> out[s0:s0+valid, h, 0:dv] 累加
                pass

    return main
