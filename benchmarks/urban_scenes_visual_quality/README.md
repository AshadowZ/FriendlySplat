# Urban Scenes Visual Quality Benchmark

This folder is reserved for an end-to-end visual-quality benchmarking pipeline on large-scale urban
scenes (e.g. GauU-Scene / MatrixCity style datasets).

Partitioning/merging algorithm references are adapted from CityGaussian.

## Notes

- Evaluation in this folder is aligned with the CityGaussian v2 visual-metric setting: `gsplat`
  metric backend with LPIPS-`alex`.
- On a single `RTX 4090`, the GauU-Scene pipeline reaches roughly paper-level visual quality in about
  `1.5h` per full benchmark run.
- On a single `RTX 4090`, the MatrixCity aerial pipeline with `2x2` partitioning reaches a level above
  the original paper in about `6h`.
- Typical GauU-Scene result from `gauu_benchmark/summary.md`: mean train time `3920.6s` (`1.089h`),
  `6.0M` gaussians, `cc_psnr=24.382`, `cc_ssim=0.7943`, `cc_lpips=0.1843`.
- Typical MatrixCity aerial result from `matrix_benchmark/summary.md`: merged step `90000`,
  `13.59M` gaussians, `cc_psnr=28.761`, `cc_ssim=0.8642`, `cc_lpips=0.1570`.
- GauU-Scene does not ship a usable `points3D.bin` model point cloud in the current local setup, and
  COLMAP was not rerun here, so depth prior is not enabled for GauU training in this repo.
- Because of that GauU results can still contain some floaters; if COLMAP is rerun and depth prior
  is restored, the metrics should improve further.

Planned stages (scripts will live here):
- Preprocess: image downsample + prior generation (e.g. MoGe normal/depth).
- Train: coarse/global model.
- Partition: split into spatial blocks and assign images per block (optional visibility/content).
- Train: per-partition fine-tuning and exports.
- Merge: merge partition checkpoints/PLY back to a single model.
- Eval: evaluate merged model on a separate test set (e.g. `*_test`).

## End-to-end: MatrixCity aerial (train -> partition -> finetune -> merge -> eval)

Assume:

- `DATA_ROOT=/media/joker/p3500/3DGS_Dataset`
- Training set: `${DATA_ROOT}/MatrixCity/aerial_train`
- Test set: `${DATA_ROOT}/MatrixCity/aerial_test`

Commands:

```bash
DATA_ROOT=/media/joker/p3500/3DGS_Dataset
DEVICE=cuda:0

# (0) [Optional] Generate MoGe priors if missing.
# Produces `${DATA_ROOT}/MatrixCity/aerial_train/moge_depth` and `moge_normal`.
# python tools/geometry_prior/moge_infer.py \
#   --data-dir "${DATA_ROOT}/MatrixCity/aerial_train" \
#   --factor 1 \
#   --save-depth \
#   --out-depth-dir moge_depth \
#   --out-normal-dir moge_normal

# (1) Train coarse model.
bash benchmarks/urban_scenes_visual_quality/run_matrixcity_coarse.sh aerial \
  --data-root "${DATA_ROOT}" \
  --device "${DEVICE}"

# (2) Partition from coarse.
python benchmarks/urban_scenes_visual_quality/partition_from_coarse.py \
  --data-dir "${DATA_ROOT}/MatrixCity/aerial_train" \
  --coarse-dir "${DATA_ROOT}/benchmark/urban_benchmark/matrix_benchmark/aerial/coarse" \
  --out-dir "${DATA_ROOT}/benchmark/urban_benchmark/matrix_benchmark/aerial/partition" \
  --block-dim 2 2 \
  --content-threshold 0.05

# (3) Train all partitions (skips blocks that already have the final ckpt).
bash benchmarks/urban_scenes_visual_quality/run_matrixcity_partition_train.sh aerial \
  --data-root "${DATA_ROOT}" \
  --device "${DEVICE}"

# (4) Merge blocks into a single model (ckpt + PLY).
python benchmarks/urban_scenes_visual_quality/merge_partitions_ckpt.py \
  --partition-dir "${DATA_ROOT}/benchmark/urban_benchmark/matrix_benchmark/aerial/partition" \
  --trained-blocks-dir "${DATA_ROOT}/benchmark/urban_benchmark/matrix_benchmark/aerial/partition/trained_blocks"

# (5) Evaluate on aerial_test (uses all test images; writes metrics to result_dir/eval/).
python benchmarks/urban_scenes_visual_quality/eval_single_scene.py \
  --result-dir "${DATA_ROOT}/benchmark/urban_benchmark/matrix_benchmark/aerial/merged" \
  --eval-data-dir "${DATA_ROOT}/MatrixCity/aerial_test" \
  --use-ply \
  --metrics-backend gsplat \
  --lpips-net alex

# (6) Write a one-page summary.md under matrix_benchmark/.
python benchmarks/urban_scenes_visual_quality/summarize_matrixcity_benchmark.py \
  --benchmark-root "${DATA_ROOT}/benchmark/urban_benchmark/matrix_benchmark"
```

## End-to-end: GauU-Scene (train -> eval -> summarize)

Assume:

- `DATA_ROOT=/media/joker/p3500/3DGS_Dataset`
- Scene root: `${DATA_ROOT}/GauU-Scene`
- Per-scene required folders: `images_3p4175`, `sparse`, `moge_normal`

Commands:

```bash
DATA_ROOT=/media/joker/p3500/3DGS_Dataset
DEVICE=cuda:0

# (0) [Optional] Ensure GauU priors/downsampled images are ready.
# python benchmarks/urban_scenes_visual_quality/preprocess_gauu_batch.py \
#   --gauu-root "${DATA_ROOT}/GauU-Scene" \
#   --factor 3.4175 \
#   --save-normal \
#   --skip-existing

# (1) Train + evaluate all 3 GauU scenes (Modern_Building / Residence / Russian_Building).
bash benchmarks/urban_scenes_visual_quality/run_gauu_benchmark.sh all \
  --data-root "${DATA_ROOT}" \
  --device "${DEVICE}"

# (2) [Optional] Run only one scene.
# bash benchmarks/urban_scenes_visual_quality/run_gauu_benchmark.sh Modern_Building \
#   --data-root "${DATA_ROOT}" \
#   --device "${DEVICE}"

# (3) Summarize GauU results into a single table.
python benchmarks/urban_scenes_visual_quality/summarize_gauu_benchmark.py \
  --benchmark-root "${DATA_ROOT}/benchmark/urban_benchmark/gauu_benchmark"
```

## File Reference (Function + CLI Interface)

### `run_matrixcity_coarse.sh`

Function:
- Launch MatrixCity `aerial_train` coarse training with a fixed training profile.
- Default output: `${DATA_ROOT}/benchmark/urban_benchmark/matrix_benchmark/aerial/coarse`.
- Auto-skip when final checkpoint exists (unless forced).

Interface:
```bash
bash benchmarks/urban_scenes_visual_quality/run_matrixcity_coarse.sh [aerial] \
  [--data-root PATH] \
  [--device cuda:0] \
  [--force]
```

### `run_gauu_benchmark.sh`

Function:
- Run GauU-Scene training + offline eval in one script.
- Default target is all three scenes: `Modern_Building`, `Residence`, `Russian_Building`.
- Default output root: `${DATA_ROOT}/benchmark/urban_benchmark/gauu_benchmark`.
- Uses `benchmark_train_split` + `test_every` holdout to evaluate `test` split per scene.

Interface:
```bash
bash benchmarks/urban_scenes_visual_quality/run_gauu_benchmark.sh \
  [all|Modern_Building|Residence|Russian_Building] \
  [--data-root PATH] \
  [--result-root PATH] \
  [--device cuda:0] \
  [--force] \
  [--viewer] \
  [--test-every N]
```

### `partition_from_coarse.py`

Function:
- Read coarse checkpoint points and COLMAP train cameras.
- Perform quantile XY partition + `location OR visibility` image assignment.
- Export block train lists and `manifest.json` under `partition/`.

Interface:
```bash
python benchmarks/urban_scenes_visual_quality/partition_from_coarse.py \
  --data-dir /path/to/scene_train \
  --coarse-dir /path/to/coarse_result \
  [--out-dir /path/to/partition_result] \
  [--block-dim X Y] \
  [--content-threshold 0.05] \
  [--location-enlarge-frac 0.0] \
  [--visibility-enlarge-frac 0.0] \
  [--no-visibility-assignment] \
  [--visibility-max-points 200000] \
  [--seed 42]
```

### `run_matrixcity_partition_train.sh`

Function:
- Iterate all `partition/blocks/block_*_train_images.txt`.
- Train each block from coarse checkpoint init (`init_type=from_ckpt`).
- Save per-block checkpoints/PLY to `partition/trained_blocks/`.

Interface:
```bash
bash benchmarks/urban_scenes_visual_quality/run_matrixcity_partition_train.sh [aerial] \
  [--data-root PATH] \
  [--device cuda:0] \
  [--coarse-ckpt /path/to/ckpt_stepXXXXXX.pt] \
  [--force] \
  [--viewer] \
  [--only block_00_00 --only block_01_00] \
  [--max-steps N] \
  [--steps-scaler S] \
  [--sh-degree D] \
  [--densification-budget N] \
  [--refine-stop-iter N] \
  [--hard-prune-start-step N] \
  [--hard-prune-stop-step N] \
  [--hard-prune-percent P] \
  [--gns-reg-start N] \
  [--gns-reg-end N] \
  [--gns-final-budget N]
```

### `merge_partitions_ckpt.py`

Function:
- Merge all partition checkpoints at a common step into one merged model.
- Output merged checkpoint, PLY, cfg and merge report.

Interface:
```bash
python benchmarks/urban_scenes_visual_quality/merge_partitions_ckpt.py \
  --partition-dir /path/to/partition \
  --trained-blocks-dir /path/to/partition/trained_blocks \
  [--out-dir /path/to/merged] \
  [--step STEP] \
  [--ply-format ply|ply_compressed]
```

### `eval_single_scene.py`

Function:
- Evaluate a trained/merged scene with checkpoint or exported PLY.
- Save metrics JSON under `${result_dir}/eval/`.
- Default metric backend is `gsplat`; default LPIPS net is `alex`.

Interface:
```bash
python benchmarks/urban_scenes_visual_quality/eval_single_scene.py \
  --result-dir /path/to/result_dir \
  [--eval-data-dir /path/to/eval_dataset] \
  [--use-ply] \
  [--ply-path /path/to/splats_stepXXXXXX.ply] \
  [--step STEP] \
  [--split train|test] \
  [--device cuda:0] \
  [--preload none|cuda] \
  [--max-images N] \
  [--metrics-backend gsplat|inria] \
  [--lpips-net alex|vgg] \
  [--compute-cc-metrics | --no-compute-cc-metrics]
```

### `summarize_matrixcity_benchmark.py`

Function:
- Scan benchmark root and summarize per-scene coarse/partition/merge/eval results.
- Write markdown summary (default: `<benchmark-root>/summary.md`).

Interface:
```bash
python benchmarks/urban_scenes_visual_quality/summarize_matrixcity_benchmark.py \
  --benchmark-root /path/to/matrix_benchmark \
  [--out /path/to/summary.md] \
  [--scene aerial --scene another_scene]
```

### `summarize_gauu_benchmark.py`

Function:
- Scan GauU benchmark root and summarize per-scene visual metrics to one markdown table.
- Includes `metrics` (`psnr/ssim/lpips`), `cc-metrics` (`cc_psnr/cc_ssim/cc_lpips`),
  `train_time_s/train_time_h`, `num_gaussians`, `num_eval_images`, `sec_per_img`.
- Adds a final `MEAN` row across available scenes.

Interface:
```bash
python benchmarks/urban_scenes_visual_quality/summarize_gauu_benchmark.py \
  [--benchmark-root /path/to/gauu_benchmark] \
  [--out /path/to/summary.md] \
  [--scene Modern_Building --scene Residence]
```
