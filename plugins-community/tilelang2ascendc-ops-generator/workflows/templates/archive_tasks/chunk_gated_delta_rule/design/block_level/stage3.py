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

"""Block-level 设计：Stage3 — chunk 内贡献累加（V4 / C4）。

对应 AscendC `Stage3`（stage3.h）。消费 Stage1 的 qkt/gCumExp/vInner 与 Stage2 写好的
out（attn_inter 部分），对当前 chunk：

    V4  attn_c = mask(qkt * decay) * scale     # decay[i][j]=g_exp[i]*g_exp_neg[j] (j<=i)
                                                # 因果 mask 保留 j<=i（含对角）
    C4  out += attn_c @ v_new                  # (C,C) @ (C,Dv)，累加到 out_slice

说明：
- v_new 由 Stage2 在 v_inner 缓冲区内完成（v_inner - v_prime），此处直接读。
- out 最终 = Stage2 的 attn_inter + Stage3 的 attn_c@v_new，两部分分工完成。
- 本阶段不依赖历史状态；C/V 交接经 workspace/UB。

## C/V 子阶段（块内）

    V4  attn_c = mask(qkt*decay)*scale（qkt 来自 ws_qkt，decay 由 ws_gcumexp 折叠）
    C4  out_slice += attn_c @ v_new（v_new 在 v_inner 缓冲区内）
"""

import tilelang.language as T


def run_stage3(
    ws_qkt,           # (C, C) bf16，Stage1 产物（原始 q@k^T）
    ws_gcumexp,       # (C,)   fp32，Stage1 产物（exp(g_cum)）
    ws_vinner,        # (C, Dv) bf16，含 Stage2 的 v_prime 贡献（即 v_new）
    scale, valid, C, dk, dv,
    out_slice,        # out[s0:s0+valid, h, :]，累加 attn_c @ v_new
):
    """把当前 chunk 的 chunk 内贡献 attn_c@v_new 累加到 out_slice（见顶部公式）。"""
    # ---- V4：attn_c = mask(qkt*decay)*scale ----
    with T.Scope("V"):
        # TODO(tile-level): 由 ws_qkt 与 ws_gcumexp 折叠出 decay，施加因果 mask（j<=i），乘 scale
        pass

    # ---- C4：out += attn_c @ v_new ----
    with T.Scope("C"):
        # TODO(tile-level): T.gemm(累加, attn_c, v_new) -> out_slice
        pass

    T.barrier_all()
