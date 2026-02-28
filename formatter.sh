#!/usr/bin/env bash
set -euo pipefail

REQUIRED_BLACK_VERSION="22.3.0"
if ! command -v black >/dev/null 2>&1; then
  echo "[formatter] ERROR: black not found. Install via: pip install black==${REQUIRED_BLACK_VERSION}" >&2
  exit 1
fi

black_version_output="$(black --version 2>/dev/null || true)"
if [[ "${black_version_output}" =~ ([0-9]+\.[0-9]+\.[0-9]+) ]]; then
  black_version="${BASH_REMATCH[1]}"
else
  echo "[formatter] ERROR: failed to parse black version from: ${black_version_output}" >&2
  exit 1
fi

if [[ "${black_version}" != "${REQUIRED_BLACK_VERSION}" ]]; then
  echo "[formatter] ERROR: black==${black_version} found, require black==${REQUIRED_BLACK_VERSION}" >&2
  echo "[formatter] Fix: pip install black==${REQUIRED_BLACK_VERSION}" >&2
  exit 1
fi

# install via: sudo apt-get install clang-format
find gsplat/cuda/include \
  -path "third_party" -prune -o \
  -type f \( -iname "*.cpp" -o -iname "*.cuh" -o -iname "*.cu" -o -iname "*.h" \) \
  -exec clang-format -i {} \;

# install via: pip install black==22.3.0
BLACK_TARGETS=(. gsplat tests examples profiling)
existing=()
for p in "${BLACK_TARGETS[@]}"; do
  if [[ -e "$p" ]]; then
    existing+=("$p")
  fi
done

black "${existing[@]}"
