cmake_minimum_required(VERSION 3.16.0)
project(sparse_flash_mla_kernel)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION "$ENV{SOC_VERSION}" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# sparse_flash_mla 是 TTK 模板算子，AscendC kernel 只能由 CANN 算子框架
# （ops-transformer 构建系统）编译，见 README.md「本机构建与部署」。
# 本 CMake 只构建 torch 直调适配层：op_host/sparse_flash_mla.cpp（aclnn
# C API 包装）+ register.cpp（TORCH_LIBRARY_FRAGMENT(npu) 注册
# torch.ops.npu.sparse_flash_mla*），链接 framework 构建安装出的
# libcust_opapi.so（custom_transformer vendor）。

# torch extension library (aclnn adapter + register)
add_library(sparse_flash_mla_ext SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/sparse_flash_mla.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/register.cpp
)

set(VENDOR_DIR ${ASCEND_CANN_PACKAGE_PATH}/opp/vendors/custom_transformer)

target_link_libraries(sparse_flash_mla_ext PRIVATE
    torch_npu
    m
    dl
)
# aclnn 接口与 acl 运行时
target_link_directories(sparse_flash_mla_ext PRIVATE
    ${VENDOR_DIR}/op_api/lib
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_PATH
)
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
target_link_directories(sparse_flash_mla_ext PRIVATE
    ${TORCH_PATH}/lib
    ${TORCH_NPU_PATH}/lib
)
target_include_directories(sparse_flash_mla_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${VENDOR_DIR}/op_api/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/asc/include
    ${TORCH_NPU_PATH}/include
    ${TORCH_NPU_PATH}/include/third_party/acl/inc
    ${TORCH_NPU_PATH}/include/third_party/hccl/inc
    ${TORCH_PATH}/include
    ${TORCH_PATH}/include/torch/csrc/api/include
)

execute_process(COMMAND python3 -m pybind11 --includes
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${PYBIND11_INC})
target_compile_options(sparse_flash_mla_ext PRIVATE
    ${PYBIND11_INC}
    -D_GLIBCXX_USE_CXX11_ABI=1
)

execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
)
set_target_properties(sparse_flash_mla_ext PROPERTIES
    OUTPUT_NAME sparse_flash_mla_ext
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
)
