/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * DsaGatherRmt host 侧（OpDef + Tiling + InferShape 一体，对齐 asc-devkit custom_op 样例形态）。
 *
 * 签名对齐 vllm-ascend-csrc dsa_gather：8 tensor + 2 attr 原名原位（删 host_kv_cache/
 * host_rope_cache——纯远端设计，源=EMS 池，寻址角色由 rmt 表接管）；新增 3 个 rmt 表 +
 * status 槽张量（图外错误码通道）+ block_dim attr（D2a 每核独立通道核数）。
 * 输出 = resident_kv_cache/resident_rope_cache 引用写回（GE in-place 语义，老项目同名模式）。
 * V1 dtype 档：float16（bf16/int8 档后续扩，kernel 字节搬运与 dtype 无关）。
 */

#include "../op_kernel/dsa_gather_rmt/dsa_gather_rmt_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static constexpr uint64_t ASCENDC_TOOLS_WORKSPACE = static_cast<uint64_t>(16) * 1024 * 1024;

static int64_t GetDim(gert::TilingContext *context, uint32_t inputIndex, uint32_t dim)
{
    auto shape = context->GetInputShape(inputIndex);
    if (shape == nullptr) {
        return -1;
    }
    return shape->GetStorageShape().GetDim(dim);
}

static ge::graphStatus Tiling4DsaGatherRmt(gert::TilingContext *context)
{
    DsaGatherRmtTilingData *tiling = context->GetTilingData<DsaGatherRmtTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(context, "tiling data is nullptr"), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context, "attrs is nullptr"), return ge::GRAPH_FAILED);
    int64_t blockSize = *attrs->GetAttrPointer<int64_t>(0);
    int64_t numRequests = *attrs->GetAttrPointer<int64_t>(1);
    int64_t blockDim = *attrs->GetAttrPointer<int64_t>(2);

    // 输入序号与 OpDef 声明一致：0=kv 1=rope 2=res_tbl 3=host_tbl 4=pool_idx 5=topk
    // 6=slots 7=mask 8=rmt_ch 9=rmt_kv 10=rmt_rope 11=status
    int64_t topkDim0 = GetDim(context, 5, 0);
    int64_t queryCount = GetDim(context, 5, 1);
    int64_t kvDim = GetDim(context, 0, 3);
    int64_t ropeDim = GetDim(context, 1, 3);
    int64_t hostPoolCapacity = GetDim(context, 3, 0);
    int64_t hostTableWidth = GetDim(context, 3, 1);
    int64_t residentTableWidth = GetDim(context, 2, 1);
    int64_t hostBlkNum = GetDim(context, 8, 0);
    int64_t residentBlkNum = GetDim(context, 0, 0);

    OP_CHECK_IF((blockSize <= 0), OP_LOGE(context, "blockSize %ld is invalid", blockSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((numRequests <= 0 || numRequests != topkDim0),
                OP_LOGE(context, "numRequests %ld != topk dim0 %ld", numRequests, topkDim0),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((queryCount <= 0), OP_LOGE(context, "queryCount %ld is invalid", queryCount),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((kvDim <= 0 || ropeDim <= 0),
                OP_LOGE(context, "kvDim %ld or ropeDim %ld is invalid", kvDim, ropeDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((blockDim <= 0 || blockDim > 48),
                OP_LOGE(context, "blockDim %ld exceeds channel limit 48", blockDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((hostPoolCapacity <= 0 || hostTableWidth <= 0),
                OP_LOGE(context, "host block table shape is invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((residentTableWidth <= 0), OP_LOGE(context, "resident table width is invalid"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((hostBlkNum <= 0 || residentBlkNum <= 0),
                OP_LOGE(context, "block num is invalid"), return ge::GRAPH_FAILED);

    tiling->numRequests = static_cast<int32_t>(numRequests);
    tiling->queryCount = static_cast<int32_t>(queryCount);
    tiling->blockSize = static_cast<int32_t>(blockSize);
    tiling->kvDim = static_cast<int32_t>(kvDim);
    tiling->ropeDim = static_cast<int32_t>(ropeDim);
    tiling->kvElemSize = 2;  // fp16 档
    tiling->ropeElemSize = 2;
    tiling->hostPoolCapacity = static_cast<int32_t>(hostPoolCapacity);
    tiling->hostTableWidth = static_cast<int32_t>(hostTableWidth);
    tiling->residentTableWidth = static_cast<int32_t>(residentTableWidth);
    tiling->hostBlkNum = static_cast<int32_t>(hostBlkNum);
    tiling->residentBlkNum = static_cast<int32_t>(residentBlkNum);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = ASCENDC_TOOLS_WORKSPACE;  // AscendC 系统区（user 区不使用）

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {
static graphStatus DsaGatherRmtInferShape(gert::InferShapeContext *context)
{
    // 引用写回语义：输出 shape = 输入 shape
    const gert::Shape *kvShape = context->GetInputShape(0);
    gert::Shape *kvOutShape = context->GetOutputShape(0);
    OP_CHECK_IF(kvShape == nullptr || kvOutShape == nullptr,
                OP_LOGE("DsaGatherRmt", "kv shape ptr is nullptr"), return GRAPH_FAILED);
    *kvOutShape = *kvShape;
    const gert::Shape *ropeShape = context->GetInputShape(1);
    gert::Shape *ropeOutShape = context->GetOutputShape(1);
    OP_CHECK_IF(ropeShape == nullptr || ropeOutShape == nullptr,
                OP_LOGE("DsaGatherRmt", "rope shape ptr is nullptr"), return GRAPH_FAILED);
    *ropeOutShape = *ropeShape;
    return GRAPH_SUCCESS;
}

static graphStatus DsaGatherRmtInferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class DsaGatherRmt : public OpDef {
public:
    explicit DsaGatherRmt(const char *name) : OpDef(name)
    {
        this->Input("resident_kv_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("resident_rope_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("resident_block_table")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("host_block_table")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("request_pool_indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("topk_token_positions")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("resident_slots")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("miss_mask")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        // 新增三表（源物理块坐标系 [P_src] int64，src_blk 查表）
        this->Input("host_rmt_channel")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("host_rmt_kv_addr")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("host_rmt_rope_addr")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        // status 槽张量：int64 [blockDim, 8]，每核 64B 独立槽（终态 doneSeq|errCode）
        this->Input("status")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        // 引用写回输出（GE in-place 语义）
        this->Output("resident_kv_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("resident_rope_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Attr("block_size").AttrType(REQUIRED).Int(0);
        this->Attr("num_requests").AttrType(REQUIRED).Int(0);
        this->Attr("block_dim").AttrType(REQUIRED).Int(1);
        this->SetInferShape(ge::DsaGatherRmtInferShape).SetInferDataType(ge::DsaGatherRmtInferDataType);
        this->AICore()
            .SetTiling(optiling::Tiling4DsaGatherRmt)
            .AddConfig("ascend950");
    }
};
OP_ADD(DsaGatherRmt);
}  // namespace ops
