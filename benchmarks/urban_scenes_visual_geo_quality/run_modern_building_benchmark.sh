#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/home/zjr/aaa_glj_repo/3DGS_Dataset/GauU-Scene/Modern_Building"
RESULT_ROOT="/home/zjr/aaa_glj_repo/3DGS_Dataset/benchmark/urban_benchmark/gauu_benchmark"
DEVICE="cuda:0"
MODE="${1:-baseline}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

run_one() {
  local mode="$1"
  local result_dir
  local hard_prune_args=()

  case "${mode}" in
    baseline)
      result_dir="${RESULT_ROOT}/Modern_Building"
      ;;
    hardprune_fixed_percent)
      result_dir="${RESULT_ROOT}/Modern_Building_hardprune_fixed_percent_s5000_e18000"
      hard_prune_args=(
        --hard-prune.enable
        --hard-prune.policy fixed_percent
        --hard-prune.start-step 5000
        --hard-prune.stop-step 18000
      )
      ;;
    *)
      echo "Usage: $0 [baseline|hardprune_fixed_percent|all]" >&2
      exit 1
      ;;
  esac

  echo "[mode] ${mode}"
  echo "[result_dir] ${result_dir}"

  # 1) Train with runtime eval enabled (without cc metrics for speed).
  # Default max_steps is 30000 and steps_scaler is set to 3.0 in this script.
  # So the expected final train step is 90000.
  local final_ckpt="${result_dir}/ckpts/ckpt_step090000.pt"
  if [[ "${FORCE_TRAIN}" != "1" && -f "${final_ckpt}" ]]; then
    echo "[skip] train: found final checkpoint ${final_ckpt}"
  else
    python friendly_splat/trainer.py \
      --io.data-dir "${DATA_DIR}" \
      --io.result-dir "${result_dir}" \
      --io.device "${DEVICE}" \
      --io.export-ply \
      --io.save-ckpt \
      --data.data-factor 3.4175 \
      --data.normal-dir-name moge_normal \
      --data.benchmark-train-split \
      --data.test-every 10 \
      --strategy.densification-budget 8000000 \
      --strategy.refine-stop-iter 20000 \
      --strategy.prune-scale3d 0.05 \
      --optim.steps-scaler 3.0 \
      --optim.sh-degree 2 \
      --optim.visible-adam \
      --postprocess.use-bilateral-grid \
      --viewer.disable-viewer \
      --tb.enable \
      --eval.enable \
      --eval.eval-every-n 4000 \
      --eval.split test \
      --eval.no-compute-cc-metrics \
      "${hard_prune_args[@]}"
  fi

  # 2) Final offline evaluation from the latest checkpoint: compute both metrics + cc_metrics.
  python benchmarks/urban_scenes_visual_geo_quality/eval_single_scene.py \
    --result-dir "${result_dir}" \
    --device "${DEVICE}" \
    --split test \
    --metrics-backend inria \
    --lpips-net vgg \
    --compute-cc-metrics
}

if [[ "${MODE}" == "all" ]]; then
  run_one baseline
  run_one hardprune_fixed_percent
else
  run_one "${MODE}"
fi
