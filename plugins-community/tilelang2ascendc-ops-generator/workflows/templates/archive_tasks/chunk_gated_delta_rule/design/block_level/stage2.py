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

"""Block-level 设计：Stage2 — 跨 chunk 递推（单 kernel 内的 T.macro 骨架）。

对应 AscendC `Stage2`（stage2.h），**C/V 分工与同步完全照抄**。细算在
`design/tile_level/stage2.py` 落地（TODO(tile-level)）。
**任务分片封装在 stage 内部**（照抄 stage2.h `Stage2::Process`）：`nvPerCore = ceil(Nv/core)` 按
`coreId` 切核，核内 `for cId` 串行整条 chunk 链。状态 S 直接走 GM `initial_state`/`final_state`
（`curState = (gi==0 && cId==0) ? initState : finalState`，即 stage2.h `Process` 的 `curState` +
入口 `RunStage2(cg, curState, finalState[b])`），组间经 `final_state` 串行。满足 checklist #4/#5/#6/#7。

## C/V 子阶段（照抄 AscendC 分工，checklist #4）

    C3   attn_inter = qPrime @ S^T → 直写 out                          （cube，↔ CalAttnInter）
         v_inner += k_cumdecay @ S^T（gemm→L0C → atomic_add 到 ws_v_inner） （cube，↔ CalVPrime）
    V3b  g_last = exp(g_cum[C-1])；final_state = g_last * S（普通写）     （向量，↔ CopyOutState）
    C3'' final_state += ws_v_inner^T @ kg（gemm→L0C → atomic_add 到 final_state）（cube，↔ CalStateNew）

final_state 由 **V3b（写 g_last·S）+ C3''（atomic_add 累加 dstate）协作完成**，照抄 AscendC 的
AIV `CopyOutState` + AIC `CalStateNew`（`IterateAll(finalState, enAtomic=ATOMIC_ADD)`）。
v_new 不单独落 GM：`ws_v_inner` 被 C3 原子加更新后即 v_new（↔ AscendC `CalVPrime` 累加 vInner）。
tilelang 的 `T.tile.atomic_add(L0C → GM)` ↔ AscendC `SetAtomicAdd + CopyCubeOut`
（见 stage2_math_and_block_level_alignment.md §5）。**无 V2 / ws_v_prime / ws_v_new / ws_attn_inter / ws_dstate**。

## 同步（checklist #5，AIC↔AIV 并行，与 AscendC 0x2/0x5/0x4 一一对应、每轮无条件）

每轮 cId 内，C scope（AIC）与 V scope（AIV）**并行执行**，靠 3 个 cross_flag 握手：

    C scope（AIC）: 读 curState；gemm attn_inter / v_inner累加 → set 6（↔ 0x2 读完 → V 可写）
    V scope（AIV）:  wait 6（↔ 0x2）；读 curState；final_state = g_last*S → set 8（↔ 0x5 写完 → C 可累加）
                    → wait 10（↔ 0x4 等本轮 C 累加完，下轮 V 读 final_state 安全）
    C scope（AIC）:  wait 8（↔ 0x5）；atomic_add dstate 进 final_state → set 10（↔ 0x4 累加完 → V 下轮读）

flag 编号：6 ↔ 0x2、8 ↔ 0x5、10 ↔ 0x4。阶段边界由入口 `T.sync_all()` 兜底。
编号避开 stage1（0/1/2）、stage3（9/7）。
"""

import tilelang.language as T


@T.macro(hygienic=False)
def run_stage2(cid, used_core_num, group_chunks,
               nk, nv, dk, dv, C,
               b, gi, gpos, cg_len, num_chunk,
               ws_k_cumdecay, ws_qprime, ws_v_inner, ws_kg,
               initial_state, final_state, out,
               ws_g_exp):
    """跨 chunk 递推（骨架；任务分片照抄 stage2.h Process，细算见 tile_level/stage2.py）。"""
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

                    # ---- C3（C scope / AIC，并行）：读 curState，gemm attn_inter + v_inner 累加 ----
                    with T.Scope("C"):
                        # TODO(tile-level): curState = (gi==0 && cId==0) ? initial_state[b,nvId]
                        #                                   : final_state[b,nvId]
                        #                   attn_inter = ws_qprime @ curState^T（gemm→L0C）
                        #                                → 直写 out[s0:s0+valid, nvId, 0:dv]（↔ CalAttnInter）
                        #                   vi  = ws_k_cumdecay @ curState^T（gemm→L0C，k_cumdecay 带负号）
                        #                   T.tile.atomic_add(ws_v_inner[nvId,row], vi)（↔ CalVPrime，v_inner +=）
                        pass
                    T.set_cross_flag("FIX", 6)          # C 读完 curState -> V 可写 final_state（↔ 0x2，防覆盖）

                    # ---- V3b（V scope / AIV，并行）：wait 0x2，写 g_last*S，set 0x5，wait 0x4 ----
                    with T.Scope("V"):
                        T.wait_cross_flag(6)            # ↔ 0x2：C 读完 curState -> 可写 final_state
                        # TODO(tile-level): g_last = g_exp[C-1]；final_state = g_last * curState
                        #                   （普通写，只写衰减部分；同 curState 来源，↔ CopyOutState）
                        pass
                        # ↔ AscendC AIV 分支：CopyOutState 后 set 0x5、wait 0x4（等本轮 C 累加完，
                        # 下轮 V 的 CopyIn 读 final_state 才安全）。set/wait 放 V scope 内 = AIV 流。
                        T.set_cross_flag("MTE3", 8)     # V 写完 g_last*S -> C 可 atomic_add（↔ 0x5）
                        T.wait_cross_flag(10)           # ↔ 0x4：等本轮 C 累加完 -> 下轮 V 读安全

                    # ---- C3''（C scope / AIC，并行）：wait 0x5，atomic_add dstate，set 0x4 ----
                    with T.Scope("C"):
                        T.wait_cross_flag(8)            # ↔ 0x5：V 写完 g_last*S -> 可 atomic_add
                        # TODO(tile-level): T.gemm_v0(ws_v_inner[nvId,row], ws_kg[nvId,row], ds_l0,
                        #                   transpose_A=True, init=True)   # v_new 已由 C3 累加进 ws_v_inner
                        #                   T.tile.atomic_add(final_state[b,nvId], ds_l0)  # L0C -> GM 原子加（↔ CalStateNew）
                        pass
                        T.set_cross_flag("FIX", 10)     # C 累加完 -> V 下轮读 final_state（↔ 0x4）
