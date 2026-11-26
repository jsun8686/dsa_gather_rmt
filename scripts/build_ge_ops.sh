#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# KV offload prototype is licensed under Mulan PSL v2.
# DsaGatherRmt 自定义算子一键构建（对齐 asc-devkit custom_op 官方链）：
#   cmake + make binary package → custom_opp_*.run → 部署到 CANN vendors
# 用法（NPU 机，已 source CANN set_env.sh）：
#   bash scripts/build_ge_ops.sh [安装目录，默认 ./output_custom]
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OP_DIR="$ROOT_DIR/dsa_gather_rmt"
BUILD_DIR="$OP_DIR/build"
INSTALL_DIR=${1:-"$ROOT_DIR/output_custom"}
JOBS=${JOBS:-$(nproc)}

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "[ge_ops] ASCEND_HOME_PATH not set; source CANN set_env.sh first" >&2
    exit 1
fi

if [ ! -f "$HOME/asc-devkit/include/adv_api/hcomm/hcomm.h" ] && [ -z "${ASC_DEVKIT_ROOT:-}" ]; then
    echo "[ge_ops] asc-devkit not found at ~/asc-devkit; set ASC_DEVKIT_ROOT" >&2
    exit 1
fi

echo "[ge_ops] build DsaGatherRmt custom op (ascend950) ..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake_args=(..)
if [ -n "${ASC_DEVKIT_ROOT:-}" ]; then
    cmake_args+=(-DASC_DEVKIT_ROOT="$ASC_DEVKIT_ROOT")
fi
cmake "${cmake_args[@]}"
make -j"$JOBS" binary package

RUN_PKG=$(find "$BUILD_DIR" -maxdepth 1 -name "custom_opp_*.run" -print -quit)
if [ -z "$RUN_PKG" ]; then
    echo "[ge_ops] ERROR: custom_opp_*.run not found in $BUILD_DIR" >&2
    exit 1
fi

echo "[ge_ops] install run package to $INSTALL_DIR ..."
mkdir -p "$INSTALL_DIR"
"$RUN_PKG" --install-path="$INSTALL_DIR"

VENDOR_DIR=$(find "$INSTALL_DIR" -maxdepth 4 -type d -path "*vendors*" -print -quit)
if [ -z "$VENDOR_DIR" ]; then
    echo "[ge_ops] ERROR: vendors dir not found under $INSTALL_DIR" >&2
    exit 1
fi
echo "[ge_ops] done. vendors dir: $VENDOR_DIR"
echo "[ge_ops] runtime env (GE 图模式必需):"
cat <<EOF
export ASCEND_CUSTOM_OPP_PATH=${VENDOR_DIR}:\${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=${VENDOR_DIR}/op_api/lib/:\${LD_LIBRARY_PATH:-}
EOF
