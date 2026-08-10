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

"""Block-level 设计：Stage3 — chunk 内贡献累加（单 kernel 内的 T.macro 骨架）。

对应 AscendC `Stage3`（stage3.h），**C/V 分工与同步完全照抄**。细算在
`design/tile_level/stage3.py` 落地（TODO(tile-level)）。
**任务分片封装在 stage 内部**（照抄 stage3.h `Stage3::Process`）：融合任务
`totalChunks = Nv × chunkNum` 均分（`chunksPerCore = ceil`；`idx → nvId = idx/chunkNum,
cgId = idx%chunkNum`）。**不依赖状态，无串行**。workspace 行偏移 `row = cgId * C`。
满足 checklist #4/#5/#6/#7。

## C/V 子阶段（照抄 AscendC 分工，checklist #4）

    V4  attn_c = mask(qkt · decay) · scale   → ws_attn_c        （AIV 向量，↔ CalMaskedQKT 写 tmpGM）
    C4  out += attn_c @ v_new（gemm→L0C → atomic_add 到 out）    （cube，↔ AICProcess enAtomic=1）

v_new = `ws_v_inner`（Stage2 已把 v_inner 原子加更新为 v_new，不另设 ws_v_new）。
C4 把 `attn_c @ v_new` **原子加直写 out**（out 已有 Stage2 的 attn_inter），照抄 AscendC
`IterateAll(attnOut, enAtomic=1)`。**无 V5 / ws_out_cont**。

## 同步（checklist #5，AIC↔AIV 并行，与 AscendC stage3 的 0x4/0x3 对应）

    - flag 9 ↔ AscendC 0x4：V4 写完 attn_c → `set 9`（AIV）；C4 读前 `wait 9`（AIC）。
    - flag 7 ↔ AscendC 0x3：C4 累加完 → `set 7`（AIC）；V4 轮尾 `wait 7`（AIV，等**本轮** C4 set 7，
      下 idx V4 才可覆盖 ws_attn_c）。照抄 AscendC AIV 分支：`set 0x4` 后 `wait 0x3`。
      注：tilelang 的 ws_attn_c 按 (nvId, row) 位置独立，不同 chunk 不重叠，0x3 的防覆盖语义
      在 tilelang 里本可省；保留是为了对齐 AscendC 结构。
  阶段边界由入口 `T.sync_all()` 兜底。编号避开 stage1（0/1/2/3/4/5）、stage2（6/8/10）。
"""

import tilelang.language as T


@T.macro(hygienic=False)
def run_stage3(cid, used_core_num, group_chunks,
               nk, nv, dk, dv, C, scale,
               gpos, cg_len, num_chunk,
               out, ws_qkt, ws_v_inner, ws_g_exp, ws_g_exp_neg,
               ws_attn_c):
    """chunk 内贡献累加（骨架；任务分片照抄 stage3.h Process，细算见 tile_level/stage3.py）。"""
    # ---- AscendC Stage3::Process：totalChunks = nv*num_chunk 融合均分（chunksPerCore=ceil）----
    total_chunks = nv * num_chunk
    chunks_per_core = (total_chunks + used_core_num - 1) // used_core_num
    start = cid * chunks_per_core
    end = T.if_then_else(start + chunks_per_core < total_chunks,
                         start + chunks_per_core, total_chunks)

    for idx in T.serial(nv * group_chunks):
        if idx >= start and idx < end:
            nvId = idx // num_chunk
            cgId = idx % num_chunk
            s0 = gpos + cgId * C
            valid = T.if_then_else(cg_len - cgId * C < C, cg_len - cgId * C, C)
            row = cgId * C

            # ---- V4（V scope / AIV，并行）：写 attn_c → set 0x4 → wait 0x3（↔ CalMaskedQKT 写 tmpGM）----
            with T.Scope("V"):
                # TODO(tile-level): attn_c[i][j] = ws_qkt[i][j]*g_exp[i]*g_exp_neg[j]*scale (j<=i)
                #                   因果含对角 -> ws_attn_c[nvId, row, ...]
                # set/wait 放 V scope 内（AIV 流），照抄 AscendC AIV 分支：set 0x4 后 wait 0x3
                # （等本轮 C4 累加完，下 idx V4 才可覆盖 ws_attn_c，防覆盖）。
                T.set_cross_flag("MTE3", 9)         # ↔ 0x4：V4 写完 attn_c -> C4 可读
                T.wait_cross_flag(7)                # ↔ 0x3：等本轮 C4 累加完 -> 下 idx V4 可覆盖 ws_attn_c
                pass

            # ---- C4（C scope / AIC，并行）：wait 0x4 → 累加 out → set 0x3（↔ AICProcess enAtomic=1）----
            with T.Scope("C"):
                T.wait_cross_flag(9)                # ↔ 0x4：等 V4 写完 attn_c
                # TODO(tile-level): T.gemm_v0(ws_attn_c[nvId,row], ws_v_inner[nvId,row], oc_l0, init=True)
                #                   T.tile.atomic_add(out[s0:s0+valid, nvId, 0:dv], oc_l0)  # 累加 Stage2 的 attn_inter
                T.set_cross_flag("FIX", 7)          # ↔ 0x3：C4 累加完 -> V4 可覆盖 ws_attn_c
                pass
