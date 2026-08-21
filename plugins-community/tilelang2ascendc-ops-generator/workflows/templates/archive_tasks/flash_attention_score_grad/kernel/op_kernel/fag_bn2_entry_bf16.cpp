/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 单入口编译单元：每个入口独立的 ascendc_library target（独立 host stub 注册），
// 避免多入口合并二进制超过 RegisterAscendBinary 的大小限制（实测上限在 8.9MB~17MB 之间）。
#include "fag_bn2_entry_common.h"

extern "C" __global__ __aicore__ void flash_attention_score_grad_bn2_bf16(FAG_BN2_ENTRY_DECL)
{
    FAG_BN2_ENTRY_BODY(bfloat16_t, false, FAG_BN2_MM12_OUT_FORMAT);
}
