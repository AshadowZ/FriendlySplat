# Instance clustering for FriendlySplat scenes

This tool performs offline 3D instance clustering for a trained FriendlySplat scene.

At a high level it:

1. loads Gaussian splats from a checkpoint or PLY export;
2. loads COLMAP cameras from the scene;
3. loads per-image SAM-style masks from `<data_dir>/sam/...`;
4. tracks mask-to-Gaussian correspondences with `gsplat` rasterization;
5. clusters masks across views into 3D instances;
6. exports instance-colored point clouds and per-Gaussian labels.

Script entry point:

- [instascene_gauscluster.py](/home/joker/learning/FriendlySplat/tools/segment/instascene_gauscluster.py)

## Install

Install FriendlySplat with the segment extra:

```bash
pip install -e ".[segment]" --no-build-isolation
```

## Optional GPU DBSCAN

Post-processing uses DBSCAN. By default the tool can fall back to CPU DBSCAN,
but on large scenes this stage may become slow.

Optional GPU DBSCAN packages:

- `cupy`
- `cuml`

Example installation for CUDA 12.x environments:

```bash
pip install cupy-cuda12x
pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
```

Notes:

- `cupy` / `cuml` are optional. If they are missing, the tool falls back to CPU DBSCAN.
- On large scenes, CPU DBSCAN can be noticeably slower during post-processing.
- Install only the CuPy package that matches your CUDA major version.
- `cuml` wheels are primarily intended for Linux / WSL2 environments and should match your CUDA version.

## Inputs

The tool needs two kinds of inputs:

- a FriendlySplat source:
  - `--result-dir`
  - or `--ckpt-paths`
  - or `--ply-paths`
- a COLMAP scene root, either:
  - explicitly via `--data-dir`
  - or inferred from `result_dir/cfg.yml` when `--result-dir` is used

Supported source modes:

- `--result-dir`: load from a FriendlySplat result directory
- `--ckpt-paths`: load one or more explicit checkpoints
- `--ply-paths`: load one or more explicit PLY exports

When `--result-dir` is used:

- the tool prefers `ckpts/` by default
- falls back to `ply/` if no checkpoint exists
- `--step` can be used to target a specific step

## Expected scene layout

The COLMAP scene should follow the standard FriendlySplat layout:

```text
<data_dir>/
  images/
  sparse/0/
```

The tool uses `FriendlySplat`'s `ColmapDataParser`, so scene normalization and
split-related flags should match training as closely as possible.

## Expected mask layout

The tool looks for masks under one of:

- `<data_dir>/sam/mask_sorted`
- `<data_dir>/sam/mask`
- `<data_dir>/sam/mask_filtered`

You can override this with `--mask-dir-name`.

Each mask file should match the training image stem. For example:

- `images/frame_0001.png` -> `sam/mask_sorted/frame_0001.png`
- `images/frame_0001.jpg` -> `sam/mask_sorted/frame_0001.npy`

Accepted mask file formats:

- `.png`
- `.jpg`
- `.jpeg`
- `.npy`

Mask data structure:

- each mask file is a single 2D label map with shape `H x W`
- the mask resolution must exactly match the corresponding camera/image resolution
- if an image file contains multiple channels, only the first channel is used
- `.npy` masks should store an integer label array directly

Pixel semantics:

- label `0` is treated as background
- positive integer labels are treated as per-frame instance masks
- pixels with the same positive integer inside one frame are treated as belonging to the same 2D instance
- labels are interpreted per frame, not globally across the whole scene

In other words, the tool expects an "instance id image" for each frame rather than
a binary foreground mask.

## Mask generation

This repo does not try to integrate mask inference directly.

If you need a practical segmentation model for generating these per-image instance
label maps, you can use the modified EntitySeg pipeline from:

- https://github.com/zju3dv/instascene

In practice, the EntitySeg setup used by InstaScene is still reasonably effective
for panorama-style full-image segmentation. The expected workflow here is:

1. run mask inference outside FriendlySplat;
2. write the predicted per-frame instance label maps under `<data_dir>/sam/...`;
3. run `fs-segment` on the prepared masks.

This repository intentionally does not bundle or maintain that mask-inference codepath.

## Outputs

Outputs are always written under `cluster_result/` in the inferred FriendlySplat
results directory.

Typical output path:

```text
<result_dir>/cluster_result/
```

Generated files:

- `color_cluster.ply`
- `instance_labels.npy`
- `color_cluster_knn.ply`
- `instance_labels_knn.npy`

There is no separate `--output-dir` flag in this tool.

## Basic usage

Recommended package entrypoint:

```bash
fs-segment \
  --result-dir /path/to/results \
  --data-dir /path/to/scene \
  --split train
```

Legacy script path also works:

```bash
python3 tools/segment/instascene_gauscluster.py \
  --result-dir /path/to/results \
  --data-dir /path/to/scene \
  --split train
```

## Common workflows

### 1. Cluster from a FriendlySplat result directory

```bash
fs-segment \
  --result-dir /path/to/results \
  --data-dir /path/to/scene \
  --split train
```

This is the most common mode. By default the tool:

- prefers checkpoints under `ckpts/`
- falls back to PLY exports under `ply/`
- writes outputs to `<result_dir>/cluster_result/`

### 2. Cluster a specific training step

```bash
fs-segment \
  --result-dir /path/to/results \
  --step 60000
```

This resolves:

- `ckpts/ckpt_step060000.pt` if available
- otherwise the latest compatible source under the result directory

### 3. Cluster from explicit checkpoints

```bash
fs-segment \
  --ckpt-paths /path/to/ckpt_step030000.pt \
  --data-dir /path/to/scene
```

### 4. Cluster from explicit PLY exports

```bash
fs-segment \
  --ply-paths /path/to/splats_step030000.ply \
  --data-dir /path/to/scene
```

### 5. Use GPU DBSCAN during post-processing

```bash
fs-segment \
  --result-dir /path/to/results \
  --data-dir /path/to/scene \
  --use-gpu-dbscan
```

This requires optional `cupy` and `cuml` to be installed.

### 6. Cluster over all views

```bash
fs-segment \
  --result-dir /path/to/results \
  --data-dir /path/to/scene \
  --split all
```

## Important arguments

Source selection:

- `--result-dir`: FriendlySplat result directory containing `ckpts/` and/or `ply/`
- `--ckpt-paths`: explicit checkpoint path(s)
- `--ply-paths`: explicit PLY path(s)
- `--step`: optional specific step number

Dataset / split alignment:

- `--data-dir`: COLMAP scene root
- `--data-factor`: image downsample factor; should match the mask resolution
- `--normalize-world-space` / `--align-world-axes`: should match training-time values
- `--split {train,test,all}`: choose which view set is used
- `--mask-dir-name`: choose a custom mask subfolder under `<data_dir>/sam/`

Runtime:

- `--device`: rendering device, e.g. `cuda` or `cpu`

Post-processing:

- `--point-filter-threshold`: consistency threshold used during point filtering
- `--dbscan-eps`: DBSCAN epsilon
- `--dbscan-min-points`: DBSCAN minimum point count
- `--use-gpu-dbscan`: use cuML DBSCAN if available

## Practical tips

- Start with `--result-dir ... --split train` before trying more aggressive settings.
- Make sure `--data-factor` matches the actual mask resolution; mismatched mask/image sizes will hurt correspondence quality.
- If the scene was trained with world normalization or axis alignment, pass matching values here.
- If clustering quality looks unstable, verify that the SAM masks align with the COLMAP images exactly.
- If post-processing is slow on large scenes, consider installing optional GPU DBSCAN and using `--use-gpu-dbscan`.
- `--split all` can improve coverage, but also increases runtime and may include views you do not want in evaluation-style runs.

## Notes on source behavior

- When `--result-dir` resolves to a checkpoint, omitted `--normalize-world-space` and `--align-world-axes` values are inherited from `result_dir/cfg.yml`.
- When the selected source is a PLY, `--normalize-world-space` and `--align-world-axes` default to `False` unless explicitly provided.
- This matches the assumption that exported PLY splats are already aligned with COLMAP world coordinates.

## Troubleshooting

- `No usable ckpt/ply source found`:
  check that `--result-dir` contains `ckpts/` or `ply/`, or pass explicit `--ckpt-paths` / `--ply-paths`.
- `data-dir is required`:
  either provide `--data-dir` explicitly or make sure `result_dir/cfg.yml` contains the training dataset path.
- `mask directory not found`:
  verify the `<data_dir>/sam/...` layout or use `--mask-dir-name`.
- `Pixel-to-Gaussian correspondences are unavailable`:
  your installed `gsplat` build likely does not support `track_pixel_gaussians=True`.
- CPU DBSCAN is too slow:
  install optional `cupy` / `cuml`, then rerun with `--use-gpu-dbscan`.
