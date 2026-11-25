# ----------------------------------------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# DsaGatherRmt 自定义算子最小工程（对齐 asc-devkit custom_op 样例形态，仅 ascend950）
# 构建：mkdir build && cd build && cmake .. && make -j binary package → custom_opp_*.run
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(dsa_gather_rmt_prj)
find_package(ASC REQUIRED)

# hcomm 设备头（adv_api/hcomm/hcomm.h）所在 devkit 根；默认 ~/asc-devkit，可用 -DASC_DEVKIT_ROOT 覆盖
if(NOT ASC_DEVKIT_ROOT)
    set(ASC_DEVKIT_ROOT "$ENV{HOME}/asc-devkit" CACHE PATH "asc-devkit root for hcomm.h")
endif()
if(NOT EXISTS ${ASC_DEVKIT_ROOT}/include/adv_api/hcomm/hcomm.h)
    message(FATAL_ERROR "${ASC_DEVKIT_ROOT}/include/adv_api/hcomm/hcomm.h not found, set -DASC_DEVKIT_ROOT=")
endif()

set(ASCEND_COMPUTE_UNIT ascend950)
set(package_name kvoffload)

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
