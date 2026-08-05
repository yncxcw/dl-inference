#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${DLI_BUILD_DIR:-${ROOT_DIR}/build}"

if [[ "${DLI_SKIP_BUILD:-0}" != "1" ]]; then
  "${ROOT_DIR}/build.sh"
fi

ctest --test-dir "${BUILD_DIR}" --output-on-failure "$@"
