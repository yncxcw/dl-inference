#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${DLI_BUILD_DIR:-${ROOT_DIR}/build}"
BUILD_TYPE="${DLI_BUILD_TYPE:-Release}"
BUILD_JOBS="${DLI_BUILD_JOBS:-$(nproc 2>/dev/null || echo 2)}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DDLI_BUILD_TESTS="${DLI_BUILD_TESTS:-ON}" \
  -DDLI_BUILD_EXAMPLES="${DLI_BUILD_EXAMPLES:-ON}" \
  -DDLI_ENABLE_TRITON_AOT="${DLI_ENABLE_TRITON_AOT:-ON}" \
  -DDLI_ENABLE_GRPC="${DLI_ENABLE_GRPC:-OFF}" \
  -DDLI_BUILD_PYTHON_BINDINGS="${DLI_BUILD_PYTHON_BINDINGS:-ON}"

cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"
