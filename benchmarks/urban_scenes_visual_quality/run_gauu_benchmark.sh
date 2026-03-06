#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/urban_scenes_visual_quality/run_gauu_benchmark.sh [all|Modern_Building|Residence|Russian_Building] [--data-root PATH] [--result-root PATH] [--device cuda:0]

Optional:
  --data-root PATH        Default: /media/joker/p3500/3DGS_Dataset
  --result-root PATH      Default: <data-root>/benchmark/urban_benchmark/gauu_benchmark
  --device DEVICE         Default: cuda:0
  --force                 Train even if final checkpoint exists
  --viewer                Enable online viewer (default: off)
  --test-every N          Holdout stride for eval split. Default: 10
EOF
}

die() {
  echo "[error] $*" >&2
  exit 1
}

target="${1:-all}"
if [[ "${1:-}" != "" && "${1:0:1}" != "-" ]]; then
  shift || true
fi

data_root="/media/joker/p3500/3DGS_Dataset"
result_root=""
device="cuda:0"
force="0"
disable_viewer="1"
test_every="10"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --data-root) data_root="${2:?}"; shift 2 ;;
    --result-root) result_root="${2:?}"; shift 2 ;;
    --device) device="${2:?}"; shift 2 ;;
    --force) force="1"; shift ;;
    --viewer) disable_viewer="0"; shift ;;
    --test-every) test_every="${2:?}"; shift 2 ;;
    *) die "unknown arg: $1" ;;
  esac
done

if [[ -z "${result_root}" ]]; then
  result_root="${data_root}/benchmark/urban_benchmark/gauu_benchmark"
fi

case "${target}" in
  all|Modern_Building|Residence|Russian_Building) ;;
  *) die "target must be one of: all, Modern_Building, Residence, Russian_Building" ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

scene_list=()
if [[ "${target}" == "all" ]]; then
  scene_list=(Modern_Building Residence Russian_Building)
else
  scene_list=("${target}")
fi

run_scene() {
  local scene="$1"
  local data_dir="${data_root}/GauU-Scene/${scene}"
  local result_dir="${result_root}/${scene}"
  local final_ckpt="${result_dir}/ckpts/ckpt_step090000.pt"

  [[ -d "${data_dir}" ]] || die "missing scene data dir: ${data_dir}"
  [[ -d "${data_dir}/sparse" ]] || die "missing COLMAP sparse dir: ${data_dir}/sparse"
  [[ -d "${data_dir}/images_3p4175" ]] || die "missing downsampled images dir: ${data_dir}/images_3p4175"
  [[ -d "${data_dir}/moge_normal" ]] || die "missing normal prior dir: ${data_dir}/moge_normal"
  mkdir -p "${result_dir}"

  echo "[scene] ${scene}"
  echo "[data_dir] ${data_dir}"
  echo "[result_dir] ${result_dir}"

  if [[ "${force}" != "1" && -f "${final_ckpt}" ]]; then
    echo "[skip] train: found final checkpoint ${final_ckpt}"
  else
    python friendly_splat/trainer.py \
      --io.data-dir "${data_dir}" \
      --io.result-dir "${result_dir}" \
      --io.device "${device}" \
      --io.export-ply \
      --io.save-ckpt \
      --tb.enable \
      --data.data-factor 3.4175 \
      --data.preload none \
      --data.normal-dir-name moge_normal \
      --data.benchmark-train-split \
      --data.test-every "${test_every}" \
      --postprocess.use-bilateral-grid \
      --optim.max-steps 30000 \
      --optim.steps-scaler 3.0 \
      --optim.sh-degree 2 \
      --optim.visible-adam \
      --strategy.impl improved \
      --strategy.densification-budget 12000000 \
      --strategy.refine-stop-iter 20000 \
      --hard-prune.enable \
      --hard-prune.policy fixed_percent \
      --hard-prune.start-step 3000 \
      --hard-prune.stop-step 18000 \
      --hard-prune.percent-per-event 0.3 \
      --gns.gns-enable \
      --gns.reg-start 20001 \
      --gns.reg-end 27000 \
      --gns.final_budget 6000000 \
      $([[ "${disable_viewer}" == "1" ]] && echo --viewer.disable-viewer)
  fi

  python benchmarks/urban_scenes_visual_quality/eval_single_scene.py \
    --result-dir "${result_dir}" \
    --data-dir "${data_dir}" \
    --device "${device}" \
    --split test \
    --metrics-backend gsplat \
    --lpips-net alex \
    --compute-cc-metrics
}

for scene in "${scene_list[@]}"; do
  run_scene "${scene}"
done
