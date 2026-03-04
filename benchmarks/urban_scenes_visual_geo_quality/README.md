# Urban Scenes Visual + Geometry Quality Benchmark

This folder is reserved for an end-to-end benchmarking pipeline on large-scale urban
scenes (e.g. GauU-Scene / MatrixCity style datasets).

Partitioning/merging algorithm references are adapted from CityGaussian.

Planned stages (scripts will live here):
- Preprocess: image downsample + prior generation (e.g. MoGe normal/depth).
- Train: coarse/global model.
- Partition: split into spatial blocks and assign images per block (optional visibility/content).
- Train: per-partition fine-tuning and exports.

## MatrixCity coarse training

Use:

```bash
bash benchmarks/urban_scenes_visual_geo_quality/run_matrixcity_coarse.sh aerial
# or
bash benchmarks/urban_scenes_visual_geo_quality/run_matrixcity_coarse.sh street
```

Environment overrides:

- `DATA_ROOT` (default: `/media/joker/p3500/3DGS_Dataset`)
- `DEVICE` (default: `cuda:0`)
- `FORCE_TRAIN=1` to ignore existing final checkpoint and retrain.

## MatrixCity partition from coarse

Use:

```bash
python benchmarks/urban_scenes_visual_geo_quality/partition_from_coarse.py \
  --data-dir /path/to/scene_train \
  --coarse-dir /path/to/coarse_result \
  --out-dir /path/to/partition_result \
  --block-dim 3 3
```

Behavior:
- Loads latest coarse checkpoint from `.../coarse/ckpts/ckpt_step*.pt` and uses coarse Gaussian `means` for partition support.
- Does `location OR visibility` assignment for each block by default.
- Minimal interface: `data/coarse/out/block_dim/content_threshold` + visibility controls.
- Saves debug artifacts by default: `partitions.png` and `blocks/*.png`.

## Train partitions batch

Use:

```bash
bash benchmarks/urban_scenes_visual_geo_quality/run_matrixcity_partition_train.sh aerial
# or
bash benchmarks/urban_scenes_visual_geo_quality/run_matrixcity_partition_train.sh street
```

Environment overrides (optional):

- `--data-root PATH` (default: `/media/joker/p3500/3DGS_Dataset`)
- `--device DEVICE` (default: `cuda:0`)
- `--coarse-ckpt PATH` (default: auto-pick latest under `.../coarse/ckpts/ckpt_step*.pt`)
- `--force` to ignore existing final checkpoint and retrain.

Default training profile:
- `steps_scaler=3.0`, `sh_degree=2`
- `densification_budget=12000000`, `refine_stop_iter=20000`
- hard prune fixed-percent: `start=1000`, `stop=18000`, `percent=0.3`
- GNS: `gns_enable=True`, `reg_start=20001`, `reg_end=27000`, `final_budget=6000000`
