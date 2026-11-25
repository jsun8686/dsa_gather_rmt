/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * DsaGatherRmt tiling 数据结构（host 侧填充 / kernel 侧读取，两侧共用本头；
 * 对齐 asc-devkit custom_op 样例的 POD 形态，由 kernel_tiling 链反序列化）。
 * attr/shape 派生量全在此（原直调版 24 标量内核参数的 GE 化）。
 */

#ifndef DSA_GATHER_RMT_TILING_H
#define DSA_GATHER_RMT_TILING_H

#include <cstdint>

namespace optiling {

struct DsaGatherRmtTilingData {
    int32_t numRequests;        // attr num_requests（= pool_indices.shape[0] 校验）
    int32_t queryCount;         // topk_token_positions.shape[1]
    int32_t blockSize;          // attr block_size
    int32_t kvDim;              // resident_kv_cache.shape[3]
    int32_t ropeDim;            // resident_rope_cache.shape[3]
    int32_t kvElemSize;         // dtype 派生（fp16=2）
    int32_t ropeElemSize;       // dtype 派生（fp16=2）
    int32_t hostPoolCapacity;   // host_block_table.shape[0]
    int32_t hostTableWidth;     // host_block_table.shape[1]
    int32_t residentTableWidth; // resident_block_table.shape[1]
    int32_t hostBlkNum;         // host_rmt_channel.shape[0]（P_src）
    int32_t residentBlkNum;     // resident_kv_cache.shape[0]（P_dst）
};

}  // namespace optiling

#endif  // DSA_GATHER_RMT_TILING_H
