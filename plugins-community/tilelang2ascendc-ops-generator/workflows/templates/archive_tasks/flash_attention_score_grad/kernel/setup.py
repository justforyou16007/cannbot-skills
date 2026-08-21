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
# Setup for: flash_attention_score_grad_archive/kernel
# 与 archive_tasks 模板一致，但使用 kernel/ 目录下的自包含 _kernel_setup_common.py
# （归档独立于 workflows/templates/archive_tasks 公共目录）。

import sys
from pathlib import Path

_KERNEL_DIR = Path(__file__).resolve().parent
if str(_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(_KERNEL_DIR))
from _kernel_setup_common import kernel_setup

kernel_setup("flash_attention_score_grad", kernel_dir=__file__)
