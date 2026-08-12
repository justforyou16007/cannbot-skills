#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Build sparse_flash_mla torch 直调适配层并安装为 .so（kernel/build/）
# Usage: bash build.sh
#
# 说明：sparse_flash_mla 的 AscendC kernel 是 TTK 模板算子，只能由 CANN
# 算子框架（ops-transformer）编译并安装为 custom_transformer vendor（见
# README.md「本机构建与部署」）。本脚本只构建调用该 vendor aclnn 接口的
# torch 扩展（sparse_flash_mla_ext），前置条件是 vendor 已安装
# （$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so
# 存在）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Source CANN environment
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" 2>/dev/null || true
fi

# Activate conda if available
if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "${CONDA_DEFAULT_ENV}" 2>/dev/null || true
fi

VENDOR_LIB="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
if [[ ! -f "${VENDOR_LIB}" ]]; then
    echo "[build.sh] ERROR: ${VENDOR_LIB} 不存在。" >&2
    echo "[build.sh] 请先按 README.md「本机构建与部署」完成框架构建 + install.sh 部署。" >&2
    exit 1
fi

# 1) cmake 构建适配层扩展（产出 ${BUILD_DIR}/sparse_flash_mla_ext*.so）
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DSOC_VERSION="${SOC_VERSION:-Ascend910B2}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j "$(nproc)"

# 2) 可选：打包为可 pip 安装的扩展（_kernel_setup_common.kernel_setup 流程）
if [[ "${PACK_WHEEL:-0}" == "1" ]]; then
    python3 "${SCRIPT_DIR}/setup.py" bdist_wheel --dist-dir "${BUILD_DIR}/dist"
fi

echo "[build.sh] 完成：${BUILD_DIR}/sparse_flash_mla_ext*.so"
echo "[build.sh] 运行期环境："
echo "[build.sh]   export ASCEND_CUSTOM_OPP_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}/opp/vendors/custom_transformer"
echo "[build.sh]   export LD_LIBRARY_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}/opp/vendors/custom_transformer/op_api/lib:\${LD_LIBRARY_PATH}"
