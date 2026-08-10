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

"""Tile-level Stage2 - 跨 chunk 递推（单 kernel 内的函数，非独立 jit）。

Mirrors block_level/stage2.py / AscendC `Stage2`（stage2.h），但为单 kernel 内联函数。
**C/V 分工与同步完全照抄 AscendC（AIC/AIV 并行，3 flag）**。
**任务分片封装在此函数内部**（对应 stage2.h `Stage2::Process`）：`nvPerCore = ceil(Nv/core)`
按 `coreId` 切核，核内 `for cId` 串行整条 chunk 链。状态 S **直接走 GM
`initial_state`/`final_state`**：`curState = (gi==0 && cId==0) ? initial_state[b,nvId]
: final_state[b,nvId]`；组间经 `final_state` 串行。workspace 行偏移 `row = cId * C`。

    C3   attn_inter = qPrime @ S^T → 直写 out                        （cube，↔ CalAttnInter）
         v_inner += k_cumdecay @ S^T（gemm→L0C → atomic_add 到 ws_v_inner）（cube，↔ CalVPrime）
    V3b  g_last = exp(g_cum[C-1])；final_state = g_last * S（普通写）  （向量，↔ CopyOutState）
    C3'' final_state += ws_v_inner^T @ kg（gemm→L0C → atomic_add 到 final_state）（cube，↔ CalStateNew）

final_state 由 **V3b（写 g_last·S）+ C3''（atomic_add 累加 dstate）协作**。v_new 不落 GM：
`ws_v_inner` 被 C3 原子加更新后即 v_new（stage3 直接读它）。无 V2 / ws_v_prime / ws_v_new /
ws_attn_inter / ws_dstate。

同步 3 flag（每轮 cId 无条件）：6↔0x2（C 读完→V 可写，防覆盖）、8↔0x5（V 写完→C 可累加）、
10↔0x4（C 累加完→V 下轮读）。V3b 内 set 8 后 wait 10（照抄 AscendC AIV 分支）。

**buffer 分配纪律**：本 stage 全部 on-chip buffer 在函数顶部一次分配（静态），任务分片循环
（`for nvId / for cId`）内只复用，不再 alloc。
"""


import tilelang.language as T


@T.macro(hygienic=False)
def run_stage2(cid, used_core_num, group_chunks,
               nk, nv, dk, dv, C,
               b, gi, gpos, cg_len, num_chunk,
               ws_k_cumdecay, ws_qprime, ws_v_inner, ws_kg,
               initial_state, final_state, out,
               ws_g_exp):
    # ---- 统一分配本 stage 全部 on-chip buffer（函数顶部，分片循环外静态复用）----
    # C3 attn_inter 直写 out + v_inner 原子加累加
    kcd_l1 = T.alloc_L1((C, dk), "bfloat16")
    sp_s_l1 = T.alloc_L1((dv, dk), "bfloat16")      # curState (dv, dk)
    qprime_l1 = T.alloc_L1((C, dk), "bfloat16")
    ai_l0 = T.alloc_L0C((C, dv), "float32")
    vi2_l0 = T.alloc_L0C((C, dv), "float32")
    # V3b final_state = g_last * S（普通写衰减部分）
    g_exp_s2 = T.alloc_ub((C,), "float32")
    state_bf_s2 = T.alloc_ub((dv, dk), "bfloat16")
    state_s2 = T.alloc_ub((dv, dk), "float32")
    # C3'' final_state += v_new^T @ kg（原子加 dstate）
    vn_l1_s2 = T.alloc_L1((C, dv), "bfloat16")      # v_new = ws_v_inner
    kg_l1 = T.alloc_L1((C, dk), "bfloat16")
    ds_l0 = T.alloc_L0C((dv, dk), "float32")

    # ---- AscendC Stage2::Process：nvPerCore = ceil(Nv/coreNum) 切核，核内串行 cId ----
    nv_per_core = (nv + used_core_num - 1) // used_core_num
    nv_start = cid * nv_per_core
    nv_end = T.if_then_else(nv_start + nv_per_core < nv, nv_start + nv_per_core, nv)

    for nvId in T.serial(nv):
        if nvId >= nv_start and nvId < nv_end:
            for cId in T.serial(group_chunks):
                if cId < num_chunk:
                    s0 = gpos + cId * C
                    valid = T.if_then_else(cg_len - cId * C < C, cg_len - cId * C, C)
                    row = cId * C

                    # ---- C3（C scope / AIC）：attn_inter 直写 out + v_inner 原子加累加（读 curState）----
                    with T.Scope("C"):
                        # curState = (gi==0 && cId==0) ? initial_state[b,nvId] : final_state[b,nvId]
                        if cId == 0:
                            if gi == 0:
                                T.copy(initial_state[b, nvId, 0, 0], sp_s_l1)
                            else:
                                T.copy(final_state[b, nvId, 0, 0], sp_s_l1)
                        else:
                            T.copy(final_state[b, nvId, 0, 0], sp_s_l1)
                        # attn_inter = qPrime @ S^T -> L0C -> 直写 out（↔ CalAttnInter）
                        T.copy(ws_qprime[nvId, row, 0], qprime_l1)
                        T.gemm_v0(qprime_l1, sp_s_l1, ai_l0, transpose_B=True, init=True)
                        T.copy(ai_l0, out[s0:s0 + valid, nvId, 0:dv])
                        # v_inner += k_cumdecay @ S^T（L0C→GM 原子加，↔ CalVPrime；tilelang 已放宽 L0C 原子加 dtype 检查）
                        T.copy(ws_k_cumdecay[nvId, row, 0], kcd_l1)
                        T.gemm_v0(kcd_l1, sp_s_l1, vi2_l0, transpose_B=True, init=True)
                        T.tile.atomic_add(ws_v_inner[nvId, row, 0], vi2_l0)
                        T.set_cross_flag("FIX", 6)          # C 读完 curState -> V 可写 final_state（↔ 0x2，防覆盖，C scope 内 = AIC 侧）

                    # ---- V3b（V scope / AIV）：wait 0x2；v_inner 累加；写 g_last*S；set 0x5 ----
                    with T.Scope("V"):
                        T.wait_cross_flag(6)            # ↔ 0x2：C 读完 -> 可写 final_state
                        T.copy(ws_g_exp[nvId, row], g_exp_s2)
                        gl = g_exp_s2[C - 1]            # g_last = exp(g_cum[C-1])
                        # curState 同 C3 来源
                        if cId == 0:
                            if gi == 0:
                                T.copy(initial_state[b, nvId, 0, 0], state_bf_s2)
                            else:
                                T.copy(final_state[b, nvId, 0, 0], state_bf_s2)
                        else:
                            T.copy(final_state[b, nvId, 0, 0], state_bf_s2)
                        T.copy(state_bf_s2, state_s2)
                        T.tile.mul(state_s2, state_s2, gl)      # S = g_last * S
                        T.copy(state_s2, state_bf_s2)
                        T.copy(state_bf_s2, final_state[b, nvId, 0, 0])   # 普通写，只写衰减部分（↔ CopyOutState）
                        # set 放 V scope 内（AIV 流），照抄 AscendC AIV 分支。
                        T.set_cross_flag("MTE3", 8)     # V 写完 g_last*S -> C 可 atomic_add（↔ 0x5）

                    # ---- C3''（C scope / AIC）：wait 0x5；原子加 dstate；set 0x4 ----
                    with T.Scope("C"):
                        T.wait_cross_flag(8)            # ↔ 0x5：V 写完 g_last*S -> 可 atomic_add
                        # v_new = ws_v_inner（C3 已把 k_cumdecay@S^T 原子加进它）
                        T.copy(ws_v_inner[nvId, row, 0], vn_l1_s2)
                        T.copy(ws_kg[nvId, row, 0], kg_l1)
                        T.gemm_v0(vn_l1_s2, kg_l1, ds_l0, transpose_A=True, init=True)  # (dv, dk)
                        T.tile.atomic_add(final_state[b, nvId, 0, 0], ds_l0)   # final_state += dstate（↔ CalStateNew，L0C→GM 原子加）
                        T.set_cross_flag("FIX", 10)     # C 累加完 -> V 下轮读 final_state（↔ 0x4，C scope 内 = AIC 侧）

                    # ---- V3c（V scope / AIV）：wait 0x4（C 累加完 -> 下轮 V 读安全）----
                    with T.Scope("V"):
                        T.wait_cross_flag(10)           # ↔ 0x4：等本轮 C3'' 累加完 -> 下轮 V 读 final_state 安全
                        pass
