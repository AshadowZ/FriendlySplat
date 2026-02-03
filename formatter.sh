#!/usr/bin/env bash
set -euo pipefail

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
