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

"""Block-level 设计：Stage3 — chunk 内贡献累加。

对应 AscendC：`kernel/op_kernel/chunk_gated_delta_rule_stage3.h` 的 `Stage3` 类。
消费 Stage1 产出的 qkt / gCumExp / vInner 与 Stage2 写好的 out，对组内每个 chunk：

  attn_c = mask(qkt * decay)          # qkt 乘 decay 并施加因果 mask（j<=i 保留），
                                      # 再乘 scale（AIV：Muls + Cast，AIC 用 masked 矩阵）
  out   += attn_c @ v_new             # cube matmul，累加到 Stage2 已写的 attn_inter 上

说明：
- `attn_c` 是 chunk 内的因果注意力矩阵（只对 j<=i 生效，含对角线），对应 golden 的
  `(q*scale @ k^T * decay).tril()`。
- `out = attn_inter + attn_c @ v_new` 的两部分由 Stage2（attn_inter）与 Stage3
  （attn_c @ v_new）分工完成，最终写回 out 的 bf16 输出。
- 本阶段不再依赖历史状态，只做 chunk 内矩阵乘；与 Stage2 之间通过全核同步保证
  out 的 attn_inter 部分已就绪。
"""


def stage3(
    qkt,           # (Nv, group_len, C)   bf16，Stage1 产出（原始 q@k^T）
    g_cum_exp,     # (Nv, group_len)      fp32，Stage1 产出
    v_inner,       # (Nv, group_len, Dv)  bf16，Stage1 产出（含 v_prime 贡献，见 Stage2）
    scale,         # q 缩放因子
    out,           # (T, Nv, Dv)          bf16，累加 attn_c @ v_new
):
    """对组内每个 chunk 把 chunk 内贡献 attn_c @ v_new 累加到 out（见顶部公式）。"""
    # AIV：attn_c = mask(qkt * decay) * scale（decay 由 g_cum_exp 折叠）
    # AIC：out += attn_c @ v_new（v_new 在 v_inner 缓冲区内，含 Stage2 的 v_prime 贡献）
    raise NotImplementedError("Stage3 计算细节见 tile_level 设计与 golden（model.py）")
