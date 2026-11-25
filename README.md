# dsa_gather_rmt GE 入图工程（自定义算子最小工程）

dsa_gather_rmt 算子的 GE 图模式交付件工程——对齐 **asc-devkit custom_op 官方样例形态**
（`examples/01_simd_cpp_api/02_features/99_acl_based/00_acl_compilation/custom_op`），
零三方依赖（不使用 ops-transformer 大仓），构建产物为标准 `custom_opp_*.run`
自定义算子包（vendors 目录：op_proto / op_tiling / op_impl / op_api）。

## 目标链路

```
torch.compile(backend=get_npu_backend())                     # torch_npu.dynamo.torchair 内置
  → FX 图 → register_fx_node_ge_converter(torch.ops._C_kvops.dsa_gather_rmt)
  → torchair.ge.custom_op("DsaGatherRmt", ...)               # converter（阶段 3，在 kv_offload_proto 仓）
  → GE 图 → 算子名 DsaGatherRmt 从 ASCEND_CUSTOM_OPP_PATH 的 vendors 包解析
      op_proto（GE 原型注册）/ op_tiling（SetBlockDim=attr block_dim）
      op_impl（AscendC kernel .o + ops-info，autogen 自动）/ op_api（aclnn eager，autogen 自动）
```

## 目录结构

```
├── dsa_gather_rmt/
│   ├── CMakeLists.txt                  # 顶层（npu_op_package RUN 包）
│   ├── op_host/
│   │   ├── CMakeLists.txt              # npu_op_code_gen + tiling/opapi/proto 三库
│   │   └── dsa_gather_rmt_host.cpp     # OpDef+Tiling+InferShape 一体（12 输入 + 2 引用输出 + 3 attr）
│   └── op_kernel/
│       ├── CMakeLists.txt              # npu_op_kernel_sources/library（ascendc 编译）
│       └── dsa_gather_rmt/
│           ├── dsa_gather_rmt.cpp      # 内核（hcomm ReadNbi 直写，逻辑平移自 kv_offload_proto 直调版）
│           ├── dsa_gather_rmt_tiling.h # TilingData 结构（host/kernel 共用）
│           └── kvc_common.h            # status 写回（UB+MTE3）/ c310 epilogue
└── scripts/
    └── build_ge_ops.sh                 # 一键：cmake + make binary package + 部署
```

## 构建（NPU 机，950PR）

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh   # CANN
# 依赖：~/asc-devkit（含 adv_api/hcomm/hcomm.h 设备头）
bash scripts/build_ge_ops.sh                     # → ./output_custom/vendors/kvoffload/
```

## 运行时挂载（GE 图模式必需）

```bash
export ASCEND_CUSTOM_OPP_PATH=<工程>/output_custom/vendors/kvoffload:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=<工程>/output_custom/vendors/kvoffload/op_api/lib/:${LD_LIBRARY_PATH}
```

## 算子签名（对齐 vllm-ascend-csrc dsa_gather，仅新增 3 个 rmt 参数）

输入（12 tensor）：
`resident_kv_cache / resident_rope_cache [P,BS,1,dim] fp16`
`resident_block_table / host_block_table / request_pool_indices / topk_token_positions /
resident_slots / miss_mask int32`
`host_rmt_channel [P_src, block_dim] / host_rmt_kv_addr / host_rmt_rope_addr [P_src] int64`
`status [block_dim, 8] int64`（每核 64B 槽，终态 `(doneSeq=1<<32)|errCode`，图外校验）

输出（2，引用写回）：`resident_kv_cache / resident_rope_cache`

attr（3）：`block_size / num_requests / block_dim`（block_dim = D2a 每核独立通道核数，≤48）

TilingData（shape/attr 派生 12 量）见 `dsa_gather_rmt_tiling.h`；内核逻辑、错误码协议、
hcomm 建链纪律（GE 首跑前 eager warmup）与 kv_offload_proto 直调版完全一致。

## 约束与已知差异

- V1 dtype 档：float16（bf16/int8 档后续扩——kernel 字节搬运与 dtype 无关，扩档只改 OpDef DataType 与 tiling elemSize）
- 每进程单 block_dim 值（hcomm 多池重建问题见 kv_offload_proto doc §9）
- 内核源双轨：本工程 class 形态（GE 链）与 kv_offload_proto extern "C" 直调形态（eager/capture
  已验收链）并存，阶段 2 统一后删旧版
