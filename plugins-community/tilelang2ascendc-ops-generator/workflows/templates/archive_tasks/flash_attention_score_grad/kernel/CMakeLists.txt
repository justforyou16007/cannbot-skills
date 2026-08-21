# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# FlashAttentionScoreGrad archive standalone build:
#   - ascendc_library 编译 op_kernel/fag_s1s2_entry.cpp（arch22 S1s2Bn2gs1s2 通用路径，
#     3 dtype x 2 mask 共 6 个入口）
#   - torch 扩展（op_host 包装 + 独立 tiling + register.cpp）注册
#     torch.ops.npu.flash_attention_score_grad
# 原生产版 opp 构建入口备份于 CMakeLists.opp.txt。
cmake_minimum_required(VERSION 3.16.0)
project(flash_attention_score_grad_kernel)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION "$ENV{SOC_VERSION}" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# Find ascendc cmake (with CANN 9.0 aarch64-linux fallback)
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# AscendC kernel libraries：每个入口一个独立 target。
# 该 kernel 单入口编译产物约 8.9MB，多入口合并会超过 RegisterAscendBinary 的
# 大小上限（实测 17MB 即注册失败 107000 param invalid），因此按入口拆分独立注册。
set(FAG_ENTRIES fp16 fp16_mask bf16 bf16_mask fp32 fp32_mask)
foreach(ENTRY ${FAG_ENTRIES})
    ascendc_library(kernels_${ENTRY} STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/fag_s1s2_entry_${ENTRY}.cpp
    )
    ascendc_include_directories(kernels_${ENTRY} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
        # 生产版 kernel 依赖的 ops 公共头（util.h / pse.h / dropmask.h 等），随 CANN opp 发布
        ${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common
    )
    # 与生产版 opp 编译选项对齐（kernel 内显式同步，关闭自动同步）
    ascendc_compile_options(kernels_${ENTRY} PRIVATE
        --cce-auto-sync=off
        --cce-long-call=true
        -Wno-deprecated-declarations
    )
endforeach()

# S1s2Bn2（小 shape 路由）入口：同样每入口一个独立 target
set(FAG_BN2_ENTRIES fp16 fp16_mask bf16 bf16_mask fp32 fp32_mask)
foreach(ENTRY ${FAG_BN2_ENTRIES})
    ascendc_library(kernels_bn2_${ENTRY} STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/fag_bn2_entry_${ENTRY}.cpp
    )
    ascendc_include_directories(kernels_bn2_${ENTRY} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
        ${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common
    )
    ascendc_compile_options(kernels_bn2_${ENTRY} PRIVATE
        --cce-auto-sync=off
        --cce-long-call=true
        -Wno-deprecated-declarations
    )
endforeach()

# host tiling API（libtiling_api.a 为静态库）与平台库
set(ASCEND_LIB_DIR ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/lib64)
if(NOT EXISTS ${ASCEND_LIB_DIR}/libtiling_api.a)
    set(ASCEND_LIB_DIR ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64)
endif()

# torch extension library (op_host wrapper + standalone tiling + register)
add_library(flash_attention_score_grad_ext SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/flash_attention_score_grad_torch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/fag_tiling_standalone.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/fag_tiling_bn2_standalone.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/register.cpp
)
target_link_libraries(flash_attention_score_grad_ext PRIVATE
    kernels_fp16
    kernels_fp16_mask
    kernels_bf16
    kernels_bf16_mask
    kernels_fp32
    kernels_fp32_mask
    kernels_bn2_fp16
    kernels_bn2_fp16_mask
    kernels_bn2_bf16
    kernels_bn2_bf16_mask
    kernels_bn2_fp32
    kernels_bn2_fp32_mask
    torch_npu
    ${ASCEND_LIB_DIR}/libtiling_api.a
    ${ASCEND_LIB_DIR}/libplatform.so
    m
    dl
)

execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_PATH
)
message("TORCH_PATH is ${TORCH_PATH}")
set(ENV{ASCEND_HOME_PATH} ${ASCEND_CANN_PACKAGE_PATH})
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

target_link_directories(flash_attention_score_grad_ext PRIVATE
    ${TORCH_PATH}/lib
    ${TORCH_NPU_PATH}/lib
    ${ASCEND_LIB_DIR}
)
target_include_directories(flash_attention_score_grad_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include/kernels_fp16
    ${CMAKE_CURRENT_BINARY_DIR}/include/kernels_fp16_mask
    ${CMAKE_CURRENT_BINARY_DIR}/include/kernels_bf16
    ${CMAKE_CURRENT_BINARY_DIR}/include/kernels_bf16_mask
    ${CMAKE_CURRENT_BINARY_DIR}/include/kernels_fp32
    ${CMAKE_CURRENT_BINARY_DIR}/include/kernels_fp32_mask
    ${CMAKE_CURRENT_BINARY_DIR}/include/kernels_bn2_fp16
    ${CMAKE_CURRENT_BINARY_DIR}/include/kernels_bn2_fp16_mask
    ${CMAKE_CURRENT_BINARY_DIR}/include/kernels_bn2_bf16
    ${CMAKE_CURRENT_BINARY_DIR}/include/kernels_bn2_bf16_mask
    ${CMAKE_CURRENT_BINARY_DIR}/include/kernels_bn2_fp32
    ${CMAKE_CURRENT_BINARY_DIR}/include/kernels_bn2_fp32_mask
    ${TORCH_NPU_PATH}/include
    ${TORCH_PATH}/include
    ${TORCH_PATH}/include/torch/csrc/api/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/ascendc/include/highlevel_api
)

execute_process(COMMAND python3 -m pybind11 --includes
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${PYBIND11_INC})
target_compile_options(flash_attention_score_grad_ext PRIVATE
    ${PYBIND11_INC}
    -D_GLIBCXX_USE_CXX11_ABI=1
)

execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
)
set_target_properties(flash_attention_score_grad_ext PROPERTIES
    OUTPUT_NAME flash_attention_score_grad_ext
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
)
