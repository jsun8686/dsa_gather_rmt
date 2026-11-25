/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * kvc 内核公共部分（GE 版平移）：status 写回（UB+MTE3）/ c310 epilogue。
 */

#ifndef KVC_KERNEL_COMMON_H
#define KVC_KERNEL_COMMON_H

#include "kernel_operator.h"
#include "adv_api/hcomm/hcomm.h"

using namespace AscendC;

constexpr uint32_t KVC_HCOMM_WS_SIZE = 512;  // HCOMM_URMA_TMP_BUF_SIZE

// status 槽协议（GM，int64 [blockDim, 8]，每 block 独占 64B）：
//   终态：低 32 位 = errCode，高 32 位 = doneSeq(1)——host 判 (val>>32)==1 后取低 32 位。
//   [1] 起为诊断区（如 copied 计数），poll 不读；整页 64B 经 UB+DataCopy(MTE3) 写回。
constexpr uint64_t KVC_STATUS_PENDING = 0;

struct KvcStatusWriter {
    TPipe pipe;
    TBuf<TPosition::VECOUT> buf;
    LocalTensor<uint64_t> ub;
    uint64_t gmAddr = 0;
};

static inline __aicore__ void KvcStatusInit(KvcStatusWriter &w, uint64_t statusAddr)
{
    w.gmAddr = statusAddr;
    if (statusAddr == 0) {
        return;
    }
    w.pipe.InitBuffer(w.buf, 64);
    w.ub = w.buf.Get<uint64_t>();
    for (uint32_t i = 0; i < 8; i++) {
        w.ub(i) = 0;
    }
}

static inline __aicore__ void KvcStatusFlush(KvcStatusWriter &w)
{
    if (w.gmAddr == 0) {
        return;
    }
    constexpr TEventID evtSMte3 = 0;
    constexpr TEventID evtMte3S = 1;
    SetFlag<HardEvent::S_MTE3>(evtSMte3);
    WaitFlag<HardEvent::S_MTE3>(evtSMte3);
    GlobalTensor<uint64_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(w.gmAddr), 8);
    DataCopy(gm, w.ub, 8);
    SetFlag<HardEvent::MTE3_S>(evtMte3S);
    WaitFlag<HardEvent::MTE3_S>(evtMte3S);
}

static inline __aicore__ void KvcStatusSet(KvcStatusWriter &w, uint32_t idx, uint64_t value)
{
    if (w.gmAddr == 0) {
        return;
    }
    w.ub(idx) = value;
    KvcStatusFlush(w);
}

// 终态写回（ub(0) 单槽）：(doneSeq=1 << 32) | (errCode & 0xFFFFFFFF)
static inline __aicore__ void KvcStatusFinish(KvcStatusWriter &w, uint64_t errCode)
{
    KvcStatusSet(w, 0, (1ULL << 32) | (errCode & 0xFFFFFFFFULL));
}

// c310 收尾 flush（950 平台约定，纯设备编译需手动补齐；依赖 __DAV_C310__）。
static inline __aicore__ void KvcKernelEpilogue()
{
#if defined(__DAV_C310__)
    pipe_barrier(PIPE_ALL);
    dsb(mem_dsb_t::DSB_ALL);
    dci();
#endif
}

#endif  // KVC_KERNEL_COMMON_H
