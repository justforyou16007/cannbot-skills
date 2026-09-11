cmake_minimum_required(VERSION 3.16.0)
project(mha_gen_kernel)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION "$ENV{SOC_VERSION}" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Release)")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

if(CMAKE_SYSTEM_PROCESSOR)
  set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)
else()
  set(SYSTEM_PREFIX aarch64-linux)
endif()

# kernel 源由 bisheng -xasc 一次编译（设备码 + <<<>>> host 启动 stub 同 TU），
# 与 gcc 编译的 host/register 一起链进 <op>_ext.so
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/bin/bisheng)
  set(BISHENG ${ASCEND_CANN_PACKAGE_PATH}/bin/bisheng)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng)
  set(BISHENG ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng)
else()
  message(FATAL_ERROR "bisheng compiler does not exist, please check ASCEND_CANN_PACKAGE_PATH.")
endif()

# SOC -> npu-arch（910B 系列 = dav-2201）
set(NPU_ARCH "dav-2201" CACHE STRING "bisheng --npu-arch")

# vendored tilelang-ascend 头文件（scripts/vendor_deps.py 生成）；交付自包含，
# 禁止回退 TL_ROOT
set(VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
if(EXISTS ${VENDOR_DIR}/src)
    set(TL_INCLUDE_DIRS
        ${VENDOR_DIR}/src
        ${VENDOR_DIR}/3rdparty/catlass/include
        ${VENDOR_DIR}/3rdparty/shmem/include
        ${VENDOR_DIR}/3rdparty/shmem/src/device)
else()
    message(FATAL_ERROR "kernel/vendor/ 不存在——先用 scripts/vendor_deps.py 生成 vendor")
endif()

# 与 legacy 构建逐项对齐的 include 集（asc/tikcfw 头 + CANN 公共头）
set(KERNEL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/impl/adv_api
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/impl/basic_api
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/impl/c_api
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/impl/basic_api/reg_compute
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/impl/simt_api
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/impl/utils
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/include
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/include/adv_api
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/include/basic_api
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/include/aicpu_api
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/include/c_api
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/include/basic_api/reg_compute
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/include/simt_api
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/include/utils
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/tikcpp/tikcfw
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/tikcpp/tikcfw/interface
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/tikcpp/tikcfw/impl
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
    ${TL_INCLUDE_DIRS})
set(KERNEL_INCLUDE_OPTIONS)
foreach(INC_DIR IN LISTS KERNEL_INCLUDE_DIRS)
  list(APPEND KERNEL_INCLUDE_OPTIONS "-I${INC_DIR}")
endforeach()

# 设备侧 flags 与 legacy ascendc.cmake 逐项对齐：--cce-auto-sync（gen 产物
# 依赖编译器自动插同步，缺了会竞态）、-O3、mllvm 栈/溢出检查参数
set(KERNEL_BISHENG_FLAGS
    --npu-arch=${NPU_ARCH} -xasc -std=c++17 -fPIC -O3 -DNDEBUG
    --cce-auto-sync
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -DCATLASS_ARCH=2201 -DBACKEND_HYBM
)
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/include/version/asc_devkit_version.h)
  list(APPEND KERNEL_BISHENG_FLAGS
    -include ${ASCEND_CANN_PACKAGE_PATH}/include/version/asc_devkit_version.h)
endif()

set(KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mha_kernels.cpp)
set(KERNEL_OBJ ${CMAKE_CURRENT_BINARY_DIR}/mha_kernels.o)
add_custom_command(
  OUTPUT ${KERNEL_OBJ}
  COMMAND ${BISHENG} ${KERNEL_BISHENG_FLAGS} ${KERNEL_INCLUDE_OPTIONS}
          -c ${KERNEL_SRC} -o ${KERNEL_OBJ}
  DEPENDS ${KERNEL_SRC}
          ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mha_kernel.h
          ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mha_launch.h
          ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/kernel_common.h
          ${CMAKE_CURRENT_SOURCE_DIR}/ops.h
  VERBATIM
)
add_custom_target(kernels_obj DEPENDS ${KERNEL_OBJ})

# torch extension library (op_host + register + kernel objects)
add_library(mha_gen_ext SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/mha_gen.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/register.cpp
    ${KERNEL_OBJ}
)
add_dependencies(mha_gen_ext kernels_obj)

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

target_link_directories(mha_gen_ext PRIVATE
    ${TORCH_PATH}/lib
    ${TORCH_NPU_PATH}/lib
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)
target_include_directories(mha_gen_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${TORCH_NPU_PATH}/include
    ${TORCH_PATH}/include
    ${TORCH_PATH}/include/torch/csrc/api/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/include
)

target_link_libraries(mha_gen_ext PRIVATE
    torch
    torch_cpu
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    ascendc_runtime
    runtime
    profapi
    m
    dl
)

execute_process(COMMAND python3 -m pybind11 --includes
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${PYBIND11_INC})
target_compile_options(mha_gen_ext PRIVATE
    ${PYBIND11_INC}
    -D_GLIBCXX_USE_CXX11_ABI=1
)

execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
)
set_target_properties(mha_gen_ext PROPERTIES
    OUTPUT_NAME mha_gen_ext
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
    BUILD_RPATH "${TORCH_PATH}/lib;${TORCH_NPU_PATH}/lib;${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/lib64;${ASCEND_CANN_PACKAGE_PATH}/lib64"
)
