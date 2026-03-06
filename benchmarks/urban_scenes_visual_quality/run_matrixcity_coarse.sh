#!/usr/bin/env bash
set -euo pipefail

SCENE="aerial"
if [[ "${1:-}" != "" && "${1:0:1}" != "-" ]]; then
  SCENE="$1"
  shift || true
fi

DATA_ROOT="${DATA_ROOT:-/media/joker/p3500/3DGS_Dataset}"
DEVICE="${DEVICE:-cuda:0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

case "${SCENE}" in
  aerial)
    ;;
  *)
    echo "Usage: $0 [aerial] [--data-root PATH] [--device cuda:0]" >&2
    exit 1
    ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      echo "Usage: $0 [aerial] [--data-root PATH] [--device cuda:0]" >&2
      exit 0
      ;;
    --data-root)
      DATA_ROOT="${2:?}"
      shift 2
      ;;
    --device)
      DEVICE="${2:?}"
      shift 2
      ;;
    --force)
      FORCE_TRAIN="1"
      shift
      ;;
    *)
      echo "[error] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

DATA_DIR="${DATA_ROOT}/MatrixCity/${SCENE}_train"
RESULT_DIR="${DATA_ROOT}/benchmark/urban_benchmark/matrix_benchmark/${SCENE}/coarse"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[error] data dir not found: ${DATA_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}/sparse" ]]; then
  echo "[error] missing COLMAP sparse dir: ${DATA_DIR}/sparse" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}/images" ]]; then
  echo "[error] missing images dir: ${DATA_DIR}/images" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}/moge_normal" ]]; then
  echo "[error] missing normal prior dir: ${DATA_DIR}/moge_normal" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}/moge_depth" ]]; then
  echo "[error] missing depth prior dir: ${DATA_DIR}/moge_depth" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"

echo "[scene] ${SCENE}"
echo "[data_dir] ${DATA_DIR}"
echo "[result_dir] ${RESULT_DIR}"
echo "[device] ${DEVICE}"

FINAL_CKPT="${RESULT_DIR}/ckpts/ckpt_step030000.pt"
if [[ "${FORCE_TRAIN}" != "1" && -f "${FINAL_CKPT}" ]]; then
  echo "[skip] train: found final checkpoint ${FINAL_CKPT}"
  exit 0
fi

python friendly_splat/trainer.py \
  --io.data-dir "${DATA_DIR}" \
  --io.result-dir "${RESULT_DIR}" \
  --io.device "${DEVICE}" \
  --io.export-ply \
  --io.save-ckpt \
  --tb.enable \
  --data.data-factor 1 \
  --data.preload none \
  --data.normal-dir-name moge_normal \
  --data.depth-dir-name moge_depth \
  --postprocess.use-bilateral-grid \
  --optim.max-steps 30000 \
  --optim.sh-degree 2 \
  --strategy.impl improved \
  --strategy.prune-opa 0.05 \
  --strategy.densification-budget 4000000 \
  --optim.visible-adam \
  --strategy.refine-stop-iter 20000 \
  --hard-prune.enable \
  --hard-prune.policy fixed_percent \
  --hard-prune.start-step 3000 \
  --hard-prune.stop-step 18000 \
  --hard-prune.percent-per-event 0.3
