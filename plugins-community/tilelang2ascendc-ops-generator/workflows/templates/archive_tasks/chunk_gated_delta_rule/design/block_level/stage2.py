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

"""Block-level 设计：Stage2 — 跨 chunk 递推（状态依赖的部分）。

对应 AscendC：`kernel/op_kernel/chunk_gated_delta_rule_stage2.h` 的 `Stage2` 类。
消费 Stage1 产出的 workspace（kCumDecay / vInner / qPrime / gCum / kg）与当前状态
S（首组用 initial_state，后续用上一组 final_state），对组内每个 chunk：

  g_last    = exp(g_cum[last])                          # CalGCumExp：本 chunk 末尾衰减
  v_prime   = k_cumdecay @ S                            # cube matmul（CalVPrime）
  v_new     = v_inner - v_prime                          # 减法在 vInner 上完成
  attn_inter = qPrime @ S                               # cube matmul -> 写入 out（CalAttnInter）
  S         = g_last * S + kg^T @ v_new                 # CalStateNew：状态更新 -> final_state

说明：
- `v_new = v_inner - v_prime`：v_inner 是 chunk 内反演值，减掉「历史状态对本 chunk
  的贡献」v_prime 得到真正的增量。AscendC 里通过 CalVPrime 把 `k_cumdecay@S` 累加到
  vInner 缓冲区（符号经 k_cumdecay 约定吸收）。
- `attn_inter` 是 out 的第一部分（历史状态贡献），直接写 `out`；本 chunk 内部
  的贡献 `attn_c @ v_new` 由 Stage3 再累加。
- 状态 S 更新为 `g_last*S + kg^T@v_new`（kg = k*exp(g_last-g_cum)，Stage1 产出），
  写回 final_state，供下一 chunk-group 使用。
- 一个 chunk-group 内各 chunk 的 S 是串行递推的；跨 chunk-group 通过 GM final_state
  携带 + 全核同步。
"""


def stage2(
    k_cumdecay,      # (Nv, group_len, Dk)  bf16，Stage1 产出
    v_inner,         # (Nv, group_len, Dv)  bf16，Stage1 产出
    q_prime,         # (Nv, group_len, Dk)  bf16，Stage1 产出
    g_cum,           # (Nv, group_len)      fp32，Stage1 产出
    kg,              # (Nv, group_len, Dk)  bf16，Stage1 产出
    state_in,        # (Nv, Dv, Dk)         首组=initial_state，后续=final_state
    out,             # (T, Nv, Dv)          bf16，写 attn_inter
    final_state,     # (B, Nv, Dv, Dk)      bf16，写更新后状态
):
    """对组内每个 chunk 用当前状态 S 计算 out 的状态部分 + 更新 S（见顶部公式）。"""
    # AIC：v_prime = k_cumdecay @ S；attn_inter = qPrime @ S -> out；
    #      S = g_last*S + kg^T @ v_new -> final_state
    # AIV：g_last = exp(g_cum[-1])；v_new = v_inner - v_prime
    raise NotImplementedError("Stage2 计算细节见 tile_level 设计与 golden（model.py）")
