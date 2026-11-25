# ----------------------------------------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# Kernel 侧：dsa_gather_rmt AscendC 内核（hcomm 直驱远端回源）
# ----------------------------------------------------------------------------------------------------------

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE DsaGatherRmt
    KERNEL_DIR dsa_gather_rmt
    KERNEL_FILE dsa_gather_rmt.cpp
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# hcomm 设备头 include（kv_offload 同源做法：asc-devkit/include）
if(COMMAND ascendc_include_directories)
    ascendc_include_directories(ascendc_kernels PRIVATE
        ${ASC_DEVKIT_ROOT}/include
    )
else()
    message(WARNING "ascendc_include_directories not available; hcomm.h path may fail")
endif()

npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)
