/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * dsa_gather_rmt GE 版内核：dsa_gather 远端池适配（纯远端，无本地分支/无 0 哨兵）。
 *
 * 与 kv_offload_proto 直调版差异（逻辑等价平移）：
 *   - 入参形态：N × __gm__ uint8_t*（每 tensor 一个）+ workspace + tiling
 *     （GE 算子标准入口；attr/shape 派生量全部经 TilingData 由 host 侧 tiling 写入）
 *   - blockDim 由 tiling 的 SetBlockDim(attr block_dim) 决定，核内 GetBlockNum/GetBlockIdx 不变
 *   - status 槽为第 12 个输入张量（int64 [blockDim, 8]，每核 64B 独立槽，协议不变）
 *
 * 语义基准：vllm-ascend-csrc dsa_gather golden.py——miss_mask 唯一闸门（命中/padding 零搬运）
 * + pool_entry 两跳源寻址 + req_id 一跳目标寻址 + 三层校验（索引非负/逻辑块不越表宽/
 * 物理块不越池，全在 AIV 上）。搬运：Hcomm<UBC_CTP> ReadNbi 直写 resident 池
 * （record 粒度 kv/rope 各一发）；收尾：去重通道集逐个 Commit<PIPE_MTE3> + Drain。
 * 并行：pair（req_id×query_id）按核均分（尾核+1），核间零依赖。
 * 错误码：0x1xx=Init；0x2xx=kv ReadNbi（xx=pair 序号，[16:24]=hcomm ret）；
 * 0x3xx=rope ReadNbi；0x4xx/0x5xx=Commit/Drain（[24]=通道序号）；0x182=通道超上限。
 */

#include "kvc_common.h"
#include "dsa_gather_rmt_tiling.h"extern "C" __global__ __aicore__ void dsa_gather_rmt(
    __gm__ uint8_t *residentKv, __gm__ uint8_t *residentRope, __gm__ uint8_t *residentBlockTable,
    __gm__ uint8_t *hostBlockTable, __gm__ uint8_t *requestPoolIndices, __gm__ uint8_t *topkTokenPositions,
    __gm__ uint8_t *residentSlots, __gm__ uint8_t *missMask, __gm__ uint8_t *rmtChannel,
    __gm__ uint8_t *rmtKvAddr, __gm__ uint8_t *rmtRopeAddr, __gm__ uint8_t *status,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    (void)workspace;
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(optiling::DsaGatherRmtTilingData);
    GET_TILING_DATA(tilingData, tiling);

    const uint32_t coreId = GetBlockIdx();
    KvcStatusWriter w;
    // status 张量 int64 [blockDim, 8]：每核 64B 独立槽
    KvcStatusInit(w, reinterpret_cast<uint64_t>(status) + static_cast<uint64_t>(coreId) * 64);

    const uint64_t pairCount = static_cast<uint64_t>(tilingData.numRequests) * tilingData.queryCount;
    const uint32_t coreCount = GetBlockNum();
    const uint64_t pairsPerCore = pairCount / coreCount;
    const uint64_t tailCores = pairCount % coreCount;
    const uint64_t corePairCount = pairsPerCore + (coreId < tailCores ? 1 : 0);
    const uint64_t pairBegin = static_cast<uint64_t>(coreId) * pairsPerCore + (coreId < tailCores ? coreId : tailCores);
    const uint64_t pairEnd = pairBegin + corePairCount;

    __gm__ int32_t *resTbl = reinterpret_cast<__gm__ int32_t *>(residentBlockTable);
    __gm__ int32_t *hostTbl = reinterpret_cast<__gm__ int32_t *>(hostBlockTable);
    __gm__ int32_t *poolIdx = reinterpret_cast<__gm__ int32_t *>(requestPoolIndices);
    __gm__ int32_t *topkPos = reinterpret_cast<__gm__ int32_t *>(topkTokenPositions);
    __gm__ int32_t *resSlots = reinterpret_cast<__gm__ int32_t *>(residentSlots);
    __gm__ int32_t *mask = reinterpret_cast<__gm__ int32_t *>(missMask);
    __gm__ int64_t *chTbl = reinterpret_cast<__gm__ int64_t *>(rmtChannel);
    __gm__ int64_t *kvAddrTbl = reinterpret_cast<__gm__ int64_t *>(rmtKvAddr);
    __gm__ int64_t *ropeAddrTbl = reinterpret_cast<__gm__ int64_t *>(rmtRopeAddr);
    __gm__ uint8_t *resKvGm = residentKv;
    __gm__ uint8_t *resRopeGm = residentRope;

    const uint64_t kvRecBytes = static_cast<uint64_t>(tilingData.kvDim) * tilingData.kvElemSize;
    const uint64_t ropeRecBytes = static_cast<uint64_t>(tilingData.ropeDim) * tilingData.ropeElemSize;
    const uint32_t bs = tilingData.blockSize;

    TBuf<TPosition::VECOUT> wsBuf;
    w.pipe.InitBuffer(wsBuf, KVC_HCOMM_WS_SIZE);
    LocalTensor<uint8_t> ws = wsBuf.Get<uint8_t>();

    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm;
    bool hcommInited = false;
    bool rmtQueued = false;
    ChannelHandle rmtChs[8];
    uint32_t rmtChCnt = 0;
    uint64_t copied = 0;

    if (pairBegin < pairEnd) {  // 本核有任务才取行首元数据
        uint32_t reqId = static_cast<uint32_t>(pairBegin / tilingData.queryCount);
        uint32_t queryId = static_cast<uint32_t>(pairBegin % tilingData.queryCount);
        int32_t poolEntry = poolIdx[reqId];
        bool validPool =
            poolEntry >= 0 && static_cast<uint32_t>(poolEntry) < tilingData.hostPoolCapacity;

        for (uint64_t pair = pairBegin; pair < pairEnd; ++pair) {
            const uint64_t idxOff = pair;  // dense decode 布局
            const uint64_t maskOff = pair;
            if (validPool && mask[maskOff] != 0) {
                const int32_t token = topkPos[idxOff];
                const int32_t slot = resSlots[idxOff];
                if (token >= 0 && slot >= 0) {
                    const uint32_t srcLogical = static_cast<uint32_t>(token) / bs;
                    const uint32_t srcOff = static_cast<uint32_t>(token) % bs;
                    const uint32_t dstLogical = static_cast<uint32_t>(slot) / bs;
                    const uint32_t dstOff = static_cast<uint32_t>(slot) % bs;
                    if (srcLogical < tilingData.hostTableWidth && dstLogical < tilingData.residentTableWidth) {
                        const int32_t srcBlk = hostTbl[static_cast<uint64_t>(poolEntry) *
                                                          tilingData.hostTableWidth + srcLogical];
                        const int32_t dstBlk =
                            resTbl[static_cast<uint64_t>(reqId) * tilingData.residentTableWidth + dstLogical];
                        if (srcBlk >= 0 && dstBlk >= 0 &&
                            static_cast<uint32_t>(srcBlk) < tilingData.hostBlkNum &&
                            static_cast<uint32_t>(dstBlk) < tilingData.residentBlkNum) {
                            const uint64_t dstKvOff =
                                (static_cast<uint64_t>(dstBlk) * bs + dstOff) * kvRecBytes;
                            const uint64_t dstRopeOff =
                                (static_cast<uint64_t>(dstBlk) * bs + dstOff) * ropeRecBytes;
                            // D2a 每核独立通道：chTbl 平铺 [hostBlkNum × coreCount]
                            const int64_t ch = chTbl[static_cast<uint64_t>(srcBlk) * coreCount + coreId];
                            if (!hcommInited) {
                                int32_t ret = hcomm.Init(ws, KVC_HCOMM_WS_SIZE);
                                if (ret != 0) {
                                    KvcStatusFinish(w, 0x100 + static_cast<uint64_t>(ret));
                                    KvcKernelEpilogue();
                                    return;
                                }
                                hcommInited = true;
                            }
                            int32_t ret = hcomm.template ReadNbi<false>(
                                static_cast<ChannelHandle>(ch),
                                reinterpret_cast<GM_ADDR>(resKvGm + dstKvOff),
                                reinterpret_cast<GM_ADDR>(kvAddrTbl[srcBlk] +
                                                          static_cast<int64_t>(srcOff) * kvRecBytes),
                                kvRecBytes);
                            if (ret != 0) {
                                KvcStatusFinish(w, 0x200 + (pair & 0xFF) + (static_cast<uint64_t>(ret) << 16));
                                KvcKernelEpilogue();
                                return;
                            }
                            if (ropeRecBytes > 0) {
                                ret = hcomm.template ReadNbi<false>(
                                    static_cast<ChannelHandle>(ch),
                                    reinterpret_cast<GM_ADDR>(resRopeGm + dstRopeOff),
                                    reinterpret_cast<GM_ADDR>(ropeAddrTbl[srcBlk] +
                                                              static_cast<int64_t>(srcOff) * ropeRecBytes),
                                    ropeRecBytes);
                                if (ret != 0) {
                                    KvcStatusFinish(w, 0x300 + (pair & 0xFF) +
                                                    (static_cast<uint64_t>(ret) << 16));
                                    KvcKernelEpilogue();
                                    return;
                                }
                            }
                            bool known = false;
                            for (uint32_t ci = 0; ci < rmtChCnt; ++ci) {
                                if (rmtChs[ci] == static_cast<ChannelHandle>(ch)) {
                                    known = true;
                                    break;
                                }
                            }
                            if (!known) {
                                if (rmtChCnt >= sizeof(rmtChs) / sizeof(rmtChs[0])) {
                                    KvcStatusFinish(w, 0x182);
                                    KvcKernelEpilogue();
                                    return;
                                }
                                rmtChs[rmtChCnt++] = static_cast<ChannelHandle>(ch);
                            }
                            rmtQueued = true;
                            copied++;
                        }
                    }
                }
            }
            ++queryId;
            if (queryId == tilingData.queryCount && pair + 1 < pairEnd) {  // 行切换：重取池行
                queryId = 0;
                ++reqId;
                poolEntry = poolIdx[reqId];
                validPool =
                    poolEntry >= 0 && static_cast<uint32_t>(poolEntry) < tilingData.hostPoolCapacity;
            }
        }
    }

    if (rmtQueued) {  // 逐通道 Commit+Drain（WQE 与 MTE3 同管道保序；多通道 V2 成果）
        for (uint32_t ci = 0; ci < rmtChCnt; ++ci) {
            int32_t ret = hcomm.template Commit<PIPE_MTE3>(rmtChs[ci]);
            if (ret != 0) {
                KvcStatusFinish(w, 0x400 + (static_cast<uint64_t>(ci) << 24) +
                                (static_cast<uint64_t>(ret) & 0xFFFFFF));
                KvcKernelEpilogue();
                return;
            }
            ret = hcomm.Drain(rmtChs[ci]);
            if (ret != 0) {
                KvcStatusFinish(w, 0x500 + (static_cast<uint64_t>(ci) << 24) +
                                (static_cast<uint64_t>(ret) & 0xFFFFFF));
                KvcKernelEpilogue();
                return;
            }
        }
    }
    KvcStatusSet(w, 1, copied);  // 诊断区：本核搬运 record 数
    KvcStatusFinish(w, 0);
    KvcKernelEpilogue();
}
