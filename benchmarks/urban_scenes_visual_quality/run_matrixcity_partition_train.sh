#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/urban_scenes_visual_quality/run_matrixcity_partition_train.sh [aerial] [--data-root PATH] [--device cuda:0]

Arguments:
  scene: aerial

Optional:
  --data-root PATH      Dataset root. Default: /media/joker/p3500/3DGS_Dataset
  --device DEVICE       Default: cuda:0
  --coarse-ckpt PATH    Default: auto-pick latest ckpt_step*.pt under coarse/ckpts
  --force               Train even if final ckpt exists
  --viewer              Enable online viewer (default: off)

Hyperparams (optional overrides):
  --max-steps N                 Default: 30000
  --steps-scaler S              Default: 3.0
  --sh-degree D                 Default: 2
  --densification-budget N      Default: 12000000
  --refine-stop-iter N          Default: 20000
  --hard-prune-start-step N     Default: 1000
  --hard-prune-stop-step N      Default: 18000
  --hard-prune-percent P        Default: 0.3
  --gns-reg-start N             Default: 20001
  --gns-reg-end N               Default: 27000
  --gns-final-budget N          Default: 6000000

Selection:
  --only block_00_00 [--only block_01_00 ...]
EOF
}

die() {
  echo "[error] $*" 1>&2
  exit 1
}

scene="aerial"
if [[ "${1:-}" != "" && "${1:0:1}" != "-" ]]; then
  scene="$1"
  shift || true
fi
if [[ "${scene}" != "aerial" ]]; then
  die "scene must be 'aerial', got '${scene}'"
fi

data_root="/media/joker/p3500/3DGS_Dataset"
device="cuda:0"
coarse_ckpt=""
force="0"
disable_viewer="1"

max_steps="30000"
steps_scaler="3.0"
sh_degree="2"
densification_budget="11000000"
refine_stop_iter="20000"

hard_prune_start_step="1000"
hard_prune_stop_step="18000"
hard_prune_percent="0.3"

gns_reg_start="20001"
gns_reg_end="27000"
gns_final_budget="5500000"

only_blocks=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --data-root) data_root="${2:?}"; shift 2 ;;
    --device) device="${2:?}"; shift 2 ;;
    --coarse-ckpt) coarse_ckpt="${2:?}"; shift 2 ;;
    --force) force="1"; shift ;;
    --viewer) disable_viewer="0"; shift ;;
    --max-steps) max_steps="${2:?}"; shift 2 ;;
    --steps-scaler) steps_scaler="${2:?}"; shift 2 ;;
    --sh-degree) sh_degree="${2:?}"; shift 2 ;;
    --densification-budget) densification_budget="${2:?}"; shift 2 ;;
    --refine-stop-iter) refine_stop_iter="${2:?}"; shift 2 ;;
    --hard-prune-start-step) hard_prune_start_step="${2:?}"; shift 2 ;;
    --hard-prune-stop-step) hard_prune_stop_step="${2:?}"; shift 2 ;;
    --hard-prune-percent) hard_prune_percent="${2:?}"; shift 2 ;;
    --gns-reg-start) gns_reg_start="${2:?}"; shift 2 ;;
    --gns-reg-end) gns_reg_end="${2:?}"; shift 2 ;;
    --gns-final-budget) gns_final_budget="${2:?}"; shift 2 ;;
    --only) only_blocks+=("${2:?}"); shift 2 ;;
    *) die "unknown arg: $1" ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

data_dir="${data_root}/MatrixCity/${scene}_train"
partition_dir="${data_root}/benchmark/urban_benchmark/matrix_benchmark/${scene}/partition"
blocks_dir="${partition_dir}/blocks"
result_root="${partition_dir}/trained_blocks"

[[ -d "${data_dir}" ]] || die "missing data_dir: ${data_dir}"
[[ -d "${blocks_dir}" ]] || die "missing blocks_dir: ${blocks_dir}"
mkdir -p "${result_root}"

if [[ -z "${coarse_ckpt}" ]]; then
  coarse_ckpt_dir="${data_root}/benchmark/urban_benchmark/matrix_benchmark/${scene}/coarse/ckpts"
  [[ -d "${coarse_ckpt_dir}" ]] || die "missing coarse ckpt dir: ${coarse_ckpt_dir}"
  coarse_ckpt="$(ls -1 "${coarse_ckpt_dir}"/ckpt_step*.pt 2>/dev/null | sort | tail -n 1 || true)"
  [[ -n "${coarse_ckpt}" ]] || die "no coarse checkpoint found under: ${coarse_ckpt_dir}"
fi
[[ -f "${coarse_ckpt}" ]] || die "missing coarse_ckpt: ${coarse_ckpt}"

final_step="$(python -c 'import sys; print(int(round(float(sys.argv[1]) * float(sys.argv[2]))))' "${max_steps}" "${steps_scaler}")"
final_ckpt_name="$(printf "ckpt_step%06d.pt" "${final_step}")"

echo "[scene] ${scene}"
echo "[data_dir] ${data_dir}"
echo "[partition_dir] ${partition_dir}"
echo "[coarse_ckpt] ${coarse_ckpt}"
echo "[result_root] ${result_root}"
echo "[expected_final_ckpt] ${final_ckpt_name}"

train_one_block() {
  local block_id="$1"
  local list_path="$2"
  local block_result_dir="${result_root}/${block_id}"

  mkdir -p "${block_result_dir}"
  echo "[train] ${block_id}"
  echo "[result] ${block_result_dir}"

  fs-train \
    --io.data-dir "${data_dir}" \
    --io.result-dir "${block_result_dir}" \
    --io.device "${device}" \
    --io.export-ply \
    --io.save-ckpt \
    --tb.enable \
    --data.data-factor 1 \
    --data.preload none \
    --data.depth-dir-name moge_depth \
    --data.normal-dir-name moge_normal \
    --data.train-image-list-file "${list_path}" \
    --postprocess.use-bilateral-grid \
    --optim.max-steps "${max_steps}" \
    --optim.steps-scaler "${steps_scaler}" \
    --optim.sh-degree "${sh_degree}" \
    --optim.visible-adam \
    --strategy.impl improved \
    --strategy.densification-budget "${densification_budget}" \
    --strategy.refine-stop-iter "${refine_stop_iter}" \
    --hard-prune.enable \
    --hard-prune.policy fixed_percent \
    --hard-prune.start-step "${hard_prune_start_step}" \
    --hard-prune.stop-step "${hard_prune_stop_step}" \
    --hard-prune.percent-per-event "${hard_prune_percent}" \
    --gns.gns-enable \
    --gns.reg-start "${gns_reg_start}" \
    --gns.reg-end "${gns_reg_end}" \
    --gns.final_budget "${gns_final_budget}" \
    --init.init-type from_ckpt \
    --init.init-ckpt-path "${coarse_ckpt}" \
    $([[ "${disable_viewer}" == "1" ]] && echo --viewer.disable-viewer)
}

want_only() {
  local block_id="$1"
  if [[ "${#only_blocks[@]}" -eq 0 ]]; then
    return 0
  fi
  local b
  for b in "${only_blocks[@]}"; do
    if [[ "${block_id}" == "${b}" ]]; then
      return 0
    fi
  done
  return 1
}

shopt -s nullglob
lists=( "${blocks_dir}"/block_*_train_images.txt )
shopt -u nullglob
[[ "${#lists[@]}" -gt 0 ]] || die "no *_train_images.txt found under: ${blocks_dir}"

if [[ "${force}" != "1" ]]; then
  all_done="1"
  for list_path in "${lists[@]}"; do
    fname="$(basename "${list_path}")"
    block_id="${fname%_train_images.txt}"
    if ! want_only "${block_id}"; then
      continue
    fi
    if [[ ! -f "${result_root}/${block_id}/ckpts/${final_ckpt_name}" ]]; then
      all_done="0"
      break
    fi
  done
  if [[ "${all_done}" == "1" ]]; then
    echo "[skip] ${scene}: all blocks already trained under ${result_root}"
    exit 0
  fi
fi

trained=0
skipped=0
for list_path in "${lists[@]}"; do
  fname="$(basename "${list_path}")"
  block_id="${fname%_train_images.txt}"
  if ! want_only "${block_id}"; then
    continue
  fi
  if [[ -f "${result_root}/${block_id}/ckpts/${final_ckpt_name}" && "${force}" != "1" ]]; then
    echo "[skip] ${block_id}: ${result_root}/${block_id}/ckpts/${final_ckpt_name}"
    skipped=$((skipped + 1))
    continue
  fi
  train_one_block "${block_id}" "${list_path}"
  trained=$((trained + 1))
done

echo "[done] trained=${trained} skipped=${skipped}"
